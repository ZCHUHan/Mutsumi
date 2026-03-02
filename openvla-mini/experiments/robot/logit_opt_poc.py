import argparse
import json
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import requests
from experiments.robot.token_action_converter import TokenActionConverter

from pathlib import Path
img_path = (Path(__file__).resolve().parent / "robot.jpg")  # => /WORKSPACE/RoboMonkey/openvla-mini/experiments/robot/robot.jpg

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("["):
        return list(map(int, json.loads(raw)))
    return [int(x) for x in raw.split(",")]


def parse_float_list(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("["):
        return list(map(float, json.loads(raw)))
    return [float(x) for x in raw.split(",")]


def build_initial_logits(
    batch_size: int,
    action_dim: int,
    num_bins: int,
    init_scale: float,
    device: torch.device,
    mode: str,
    init_bins: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mode == "zeros":
        logits = torch.zeros(batch_size, action_dim, num_bins, device=device)
        return logits
    if mode == "random":
        logits = torch.randn(batch_size, action_dim, num_bins, device=device) * init_scale
        return logits
    if mode != "peaked":
        raise ValueError(f"Unsupported init mode: {mode}")

    logits = torch.full(
        (batch_size, action_dim, num_bins),
        fill_value=-init_scale,
        device=device,
    )
    if init_bins is None:
        init_bins = torch.randint(0, num_bins, (batch_size, action_dim), device=device)
    logits.scatter_(2, init_bins.unsqueeze(-1), init_scale)
    return logits


def token_ids_to_bin_indices(
    token_ids: torch.Tensor, num_bins: int, vocab_size: int = 32000
) -> torch.Tensor:
    indices = vocab_size - token_ids - 1
    return torch.clamp(indices, min=0, max=num_bins - 1)


def _extract_logprob_entry(entry: List[Any]) -> Tuple[float, int]:
    if not isinstance(entry, list) or len(entry) < 2:
        raise ValueError("Invalid logprob entry format.")
    logprob = float(entry[0])
    token_id = int(entry[1])
    return logprob, token_id


def build_logits_from_vla(
    instruction: str,
    image_path: str,
    action_server_url: str,
    batch_size: int,
    action_dim: int,
    num_bins: int,
    top_logprobs_num: int,
    temperature: float,
    logit_floor: float,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray, List[List[int]]]:
    payload = {
        "instructions": [instruction for _ in range(batch_size)],
        "image_path": image_path,
        "temperature": temperature,
        "top_logprobs_num": top_logprobs_num,
    }
    payload["image_path"] = str(img_path)
    response = requests.post(
        f"{action_server_url}/batch", json=payload, timeout=120
    )
    if response.status_code != 200:
        raise RuntimeError(f"Action server error: {response.text}")
    data = response.json()

    output_ids = data.get("output_ids")
    if output_ids is None:
        raise ValueError("Missing output_ids from action server response.")
    actions = data.get("actions")
    if actions is None:
        raise ValueError("Missing actions from action server response.")
    output_logprobs = data.get("output_logprobs", [])
    output_top_logprobs = data.get("output_top_logprobs", [])

    if len(output_ids) != batch_size:
        raise ValueError("Unexpected batch size in action server response.")

    logits = torch.full(
        (batch_size, action_dim, num_bins),
        fill_value=logit_floor,
        device=device,
        dtype=torch.float32,
    )

    for b in range(batch_size):
        if len(output_ids[b]) != action_dim:
            raise ValueError("Action dim mismatch from action server response.")
        for t in range(action_dim):
            if output_top_logprobs and output_top_logprobs[b]:
                top_list = output_top_logprobs[b][t]
                for entry in top_list:
                    logprob, token_id = _extract_logprob_entry(entry)
                    bin_idx = token_ids_to_bin_indices(
                        torch.tensor(token_id, device=device), num_bins=num_bins
                    ).item()
                    logits[b, t, bin_idx] = logprob
            else:
                token_id = output_ids[b][t]
                if output_logprobs:
                    logprob, _ = _extract_logprob_entry(output_logprobs[b][t])
                else:
                    logprob = 0.0
                bin_idx = token_ids_to_bin_indices(
                    torch.tensor(token_id, device=device), num_bins=num_bins
                ).item()
                logits[b, t, :] = logit_floor
                logits[b, t, bin_idx] = logprob
    return logits, np.array(actions), output_ids


def action_mean_var_from_logits(
    logits: torch.Tensor,
    converter: TokenActionConverter,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=-1)
    centers = converter.bin_centers_tensor(
        device=logits.device, dtype=probs.dtype
    )
    mean = (probs * centers).sum(dim=-1)
    second = (probs * (centers**2)).sum(dim=-1)
    var = torch.clamp(second - mean**2, min=0.0)

    mask_t, low_t, high_t = converter._action_norm_stats_tensors(
        device=logits.device, dtype=probs.dtype
    )
    scale = 0.5 * (high_t - low_t)
    mean_unnorm = torch.where(mask_t, mean * scale + 0.5 * (high_t + low_t), mean)
    var_unnorm = torch.where(mask_t, var * (scale**2), var)
    return mean_unnorm, var_unnorm


def target_prob_mass(probs: torch.Tensor, low: int, high: int) -> torch.Tensor:
    return probs[..., low : high + 1].sum(dim=-1)


def kl_divergence(p0: torch.Tensor, p1: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    log_p0 = torch.log(torch.clamp(p0, min=eps))
    log_p1 = torch.log(torch.clamp(p1, min=eps))
    return (p0 * (log_p0 - log_p1)).sum(dim=-1)


def run(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_seed(args.seed)

    if args.target_bin is not None:
        args.target_low = args.target_bin
        args.target_high = args.target_bin

    if args.target_low < 0 or args.target_high >= args.num_bins:
        raise ValueError("Target bin range out of bounds.")
    if args.target_low > args.target_high:
        raise ValueError("target_low must be <= target_high.")

    init_bins_list = parse_int_list(args.init_bins)
    init_bins = None
    if init_bins_list is not None:
        if len(init_bins_list) != args.action_dim:
            raise ValueError("init_bins must have length == action_dim.")
        init_bins = torch.tensor(init_bins_list, device=device).unsqueeze(0)

    vla_actions = None
    vla_output_ids = None
    if args.init_mode == "vla":
        if not args.vla_image_path or not args.vla_instruction:
            raise ValueError("init_mode=vla requires --vla-image-path and --vla-instruction.")
        logits, vla_actions, vla_output_ids = build_logits_from_vla(
            instruction=args.vla_instruction,
            image_path=args.vla_image_path,
            action_server_url=args.action_server_url,
            batch_size=args.batch_size,
            action_dim=args.action_dim,
            num_bins=args.num_bins,
            top_logprobs_num=args.top_logprobs_num,
            temperature=args.vla_temperature,
            logit_floor=args.vla_logit_floor,
            device=device,
        )
    else:
        logits = build_initial_logits(
            batch_size=args.batch_size,
            action_dim=args.action_dim,
            num_bins=args.num_bins,
            init_scale=args.init_scale,
            device=device,
            mode=args.init_mode,
            init_bins=init_bins,
        )

    if args.optimize_space == "logit":
        raise ValueError(
            "Logit-space optimization (gumbel-softmax) has been removed. "
            "Use optimize_space=action instead."
        )

    if args.optimize_space != "action":
        raise ValueError(f"Unsupported optimize_space: {args.optimize_space}")

    if args.action_target_value is None and (
        args.action_target_low is None or args.action_target_high is None
    ):
        raise ValueError(
            "Action optimization requires --action-target-value or --action-target-low/--action-target-high."
        )

    init_action_list = parse_float_list(args.init_action)
    if init_action_list is not None:
        if len(init_action_list) != args.action_dim:
            raise ValueError("init_action must have length == action_dim.")
        a_init = torch.tensor(init_action_list, device=device).unsqueeze(0).repeat(
            args.batch_size, 1
        )
    else:
        if vla_actions is None:
            raise ValueError("Action optimization requires init_mode=vla or --init-action.")
        a_init = torch.tensor(vla_actions, device=device, dtype=torch.float32)

    if a_init.shape[1] != args.action_dim:
        raise ValueError("Action dim mismatch between init actions and action_dim.")

    if args.prox_mode == "mahalanobis":
        if args.init_mode != "vla":
            raise ValueError("prox_mode=mahalanobis requires init_mode=vla for variance estimation.")
        converter = TokenActionConverter(unnorm_key=args.unnorm_key)
        _, var = action_mean_var_from_logits(logits, converter)
        precision = 1.0 / (var + args.precision_eps)
    else:
        precision = None

    a = a_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([a], lr=args.lr)

    success_step = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)

        x = a[:, args.action_target_dim]
        if args.action_target_value is not None:
            verifier = -((x - args.action_target_value) ** 2)
            success = (x - args.action_target_value).abs() <= args.action_success_eps
        else:
            low = args.action_target_low
            high = args.action_target_high
            penalty = torch.relu(low - x) ** 2 + torch.relu(x - high) ** 2
            verifier = -penalty
            success = (x >= low) & (x <= high)

        if args.prox_mode == "l2":
            prox = (a - a_init).pow(2).mean()
        else:
            prox = ((a - a_init) ** 2 * precision).mean()

        loss = -verifier.mean() + args.prox_lambda * prox

        loss.backward()
        optimizer.step()

        success_rate = success.float().mean().item()
        if success_step is None and success_rate == 1.0:
            success_step = step
        if args.early_stop and success_rate == 1.0 and step >= args.early_stop_min_steps:
            if step % args.log_every != 0:
                print(
                    f"Step {step:03d} | loss={loss.item():.4f} "
                    f"| success_rate={success_rate:.3f}"
                )
            print("Early stop: success_rate reached 1.000")
            break

        if step % args.log_every == 0 or step == args.steps - 1:
            msg = (
                f"Step {step:03d} | loss={loss.item():.4f} "
                f"| success_rate={success_rate:.3f}"
            )
            msg += f" | prox={prox.item():.4f}"
            print(msg)

    print("\n=== Summary ===")
    print(f"Final actions: {a.detach().cpu().numpy()[0].tolist()}")
    print(f"Target dim: {args.action_target_dim}")
    if args.action_target_value is not None:
        print(f"Target value: {args.action_target_value}")
    else:
        print(f"Target range: [{args.action_target_low}, {args.action_target_high}]")
    print(f"Final success rate: {success_rate:.3f}")
    if success_step is None:
        print("Steps to success: not reached")
    else:
        print(f"Steps to success: {success_step}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PoC: test-time logit optimization with Gumbel-Softmax."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--num-bins", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-min-steps", type=int, default=0)
    parser.add_argument(
        "--optimize-space", type=str, default="logit", choices=["logit", "action"]
    )

    parser.add_argument("--target-low", type=int, default=100)
    parser.add_argument("--target-high", type=int, default=150)
    parser.add_argument("--target-bin", type=int, default=None)

    parser.add_argument(
        "--init-mode",
        type=str,
        default="peaked",
        choices=["peaked", "random", "zeros", "vla"],
    )
    parser.add_argument("--init-scale", type=float, default=8.0)
    parser.add_argument(
        "--init-bins",
        type=str,
        default=None,
        help='Comma list or JSON list, length == action_dim. Example: "10,20,30,40,50,60,70"',
    )
    parser.add_argument("--vla-image-path", type=str, default="openvla-mini/experiments/robot/robot.jpg")
    parser.add_argument("--vla-instruction", type=str, default=None)
    parser.add_argument("--action-server-url", type=str, default="http://127.0.0.1:3200")
    parser.add_argument("--top-logprobs-num", type=int, default=0)
    parser.add_argument("--vla-temperature", type=float, default=1.0)
    parser.add_argument("--vla-logit-floor", type=float, default=-10.0)
    parser.add_argument("--unnorm-key", type=str, default="bridge_orig")
    parser.add_argument("--init-action", type=str, default=None)
    parser.add_argument("--action-target-dim", type=int, default=0)
    parser.add_argument("--action-target-value", type=float, default=None)
    parser.add_argument("--action-target-low", type=float, default=None)
    parser.add_argument("--action-target-high", type=float, default=None)
    parser.add_argument("--action-success-eps", type=float, default=1e-3)
    parser.add_argument("--prox-mode", type=str, default="l2", choices=["l2", "mahalanobis"])
    parser.add_argument("--prox-lambda", type=float, default=0.1)
    parser.add_argument("--precision-eps", type=float, default=1e-6)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(args)
