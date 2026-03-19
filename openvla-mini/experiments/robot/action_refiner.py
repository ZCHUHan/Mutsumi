from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import traceback

class DifferentiableVerifier:
    """Interface for a differentiable verifier.

    Implementations must return a torch.Tensor with gradients enabled
    w.r.t. the input actions.
    """

    def score(self, instruction: str, image_path: str, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def load_differentiable_verifier(cfg, device: torch.device, dtype: torch.dtype) -> DifferentiableVerifier:
    module_path = getattr(cfg, "diff_verifier_module", None)
    class_name = getattr(cfg, "diff_verifier_class", None)

    if module_path is None:
        module_path = "differentiable_verifier"
        class_name = "DifferentiableRobotRewardModel"
        repo_root = Path(__file__).resolve().parents[3]
        monkey_src = repo_root / "monkey-verifier" / "src"
        if monkey_src.exists():
            sys.path.insert(0, str(monkey_src))

    if module_path and class_name:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            try:
                return cls(cfg=cfg, device=device, dtype=dtype)
            except TypeError:
                return cls()
        except Exception as exc:
            raise RuntimeError(
                "Failed to load differentiable verifier. "
                "This project is configured to require monkey-verifier for refine mode "
                "(fallback is disabled). "
                f"module={module_path} class={class_name} error={exc}\n"
                + traceback.format_exc()
            ) from exc

    raise ValueError(
        "Invalid differentiable verifier config: expected both "
        "diff_verifier_module and diff_verifier_class (or neither to use defaults)."
    )


@dataclass
class ActionRefineResult:
    actions: np.ndarray
    rewards: np.ndarray
    reward_trace: Optional[np.ndarray] = None
    best_actions: Optional[np.ndarray] = None
    best_rewards: Optional[np.ndarray] = None
    stats: Optional[dict] = None


def _build_prior_stats(
    actions_init: torch.Tensor,
    prior_mode: str,
    eps: float,
    prior_mean: Optional[torch.Tensor] = None,
    prior_var: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = actions_init if prior_mean is None else prior_mean
    if prior_mode == "l2":
        inv_var = torch.ones_like(mean)
        return mean, inv_var
    if prior_mode != "diag":
        raise ValueError(f"Unsupported prior_mode: {prior_mode}")
    if prior_var is None:
        inv_var = torch.ones_like(mean)
    else:
        inv_var = 1.0 / (prior_var + eps)
    return mean, inv_var


def _coerce_prior_tensor(
    prior: Optional[object],
    reference: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Optional[torch.Tensor]:
    if prior is None:
        return None
    prior_t = torch.as_tensor(prior, device=device, dtype=dtype)
    if prior_t.ndim == reference.ndim - 1:
        prior_t = prior_t.unsqueeze(0).expand(reference.shape[0], *prior_t.shape)
    elif prior_t.ndim == reference.ndim and prior_t.shape[0] == 1 and reference.shape[0] > 1:
        prior_t = prior_t.expand(reference.shape[0], *prior_t.shape[1:])
    if prior_t.shape != reference.shape:
        raise ValueError(
            f"{name} shape {tuple(prior_t.shape)} must match actions_init shape {tuple(reference.shape)}."
        )
    return prior_t


def _proximal_penalty(
    actions: torch.Tensor,
    mean: torch.Tensor,
    inv_var: torch.Tensor,
) -> torch.Tensor:
    return ((actions - mean) ** 2 * inv_var).sum(dim=-1)


def refine_actions_with_grad(
    actions_init: np.ndarray,
    instruction: str,
    image_path: str,
    verifier: DifferentiableVerifier,
    steps: int,
    lr: float,
    prox_weight: float,
    prior_mode: str,
    eps: float,
    select_mode: str = "final_forward",
    log_every: int = 0,
    clamp_low: Optional[torch.Tensor] = None,
    clamp_high: Optional[torch.Tensor] = None,
    prior_mean: Optional[object] = None,
    prior_var: Optional[object] = None,
    freeze_gripper: bool = False,
    gripper_index: int = -1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    return_trace: bool = False,
    track_best: bool = False,
) -> ActionRefineResult:
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32

    actions = torch.as_tensor(actions_init, device=device, dtype=torch.float32)
    actions_init_t = actions.detach().clone()
    prior_mean_t = _coerce_prior_tensor(
        prior_mean, reference=actions_init_t, device=device, dtype=dtype, name="prior_mean"
    )
    prior_var_t = _coerce_prior_tensor(
        prior_var, reference=actions_init_t, device=device, dtype=dtype, name="prior_var"
    )
    mean, inv_var = _build_prior_stats(
        actions_init_t,
        prior_mode=prior_mode,
        eps=eps,
        prior_mean=prior_mean_t,
        prior_var=prior_var_t,
    )

    actions = actions.detach().requires_grad_(True)
    # PDB-R1: refine循环入口，检查初始 actions (requires_grad=True), prior mean/inv_var
    #import pdb; pdb.set_trace()  # actions.shape, actions.requires_grad, mean.shape, inv_var
    reward_trace = [] if return_trace else None
    track_best = bool(track_best or select_mode == "best_rewards")
    best_actions = None
    best_rewards = None
    for step in range(int(steps)):
        # PDB-R2: refine每一步，建议只在 step==0 时打开，检查 verifier.score 的输出
        #if step == 0: import pdb; pdb.set_trace()  # step, actions, verifier type
        rewards = verifier.score(instruction, image_path, actions)
        if rewards.ndim == 0:
            rewards = rewards.expand(actions.shape[0])
        if reward_trace is not None:
            reward_trace.append(rewards.detach().float().cpu().numpy())
        prox = _proximal_penalty(actions, mean, inv_var)
        loss = (-(rewards) + prox_weight * prox).mean()
        grad = torch.autograd.grad(loss, actions, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            if log_every and (step % log_every == 0 or step == steps - 1):
                update = -lr * grad
                reward_mean = rewards.float().mean()
                reward_std = rewards.float().std(unbiased=False)
                reward_min = rewards.float().min()
                reward_max = rewards.float().max()
                grad_norm = grad.float().norm(dim=-1).mean()
                update_norm = update.float().norm(dim=-1).mean()
                action_delta = (actions - actions_init_t).float().norm(dim=-1).mean()
                prox_mean = prox.float().mean()
                clamp_frac = 0.0
                if clamp_low is not None and clamp_high is not None:
                    finite_mask = torch.isfinite(clamp_low) & torch.isfinite(clamp_high)
                    if finite_mask.any():
                        at_low = finite_mask & (actions <= clamp_low + 1e-6)
                        at_high = finite_mask & (actions >= clamp_high - 1e-6)
                        clamp_frac = (at_low | at_high).float().mean().item()
                print(
                    "[action_refine] "
                    f"step={step+1}/{steps} "
                    f"reward(mean/std/min/max)={reward_mean:.4f}/{reward_std:.4f}/{reward_min:.4f}/{reward_max:.4f} "
                    f"grad_norm={grad_norm:.3e} "
                    f"update_norm={update_norm:.3e} "
                    f"action_delta={action_delta:.3e} "
                    f"prox_mean={prox_mean:.3e} "
                    f"clamp_frac={clamp_frac:.3e}"
                )
            if track_best:
                if best_rewards is None:
                    best_rewards = rewards.detach().clone()
                    best_actions = actions.detach().clone()
                else:
                    improved = rewards > best_rewards
                    if improved.any():
                        best_rewards = torch.where(improved, rewards, best_rewards)
                        best_actions = torch.where(
                            improved.unsqueeze(-1), actions, best_actions
                        )
            actions -= lr * grad
            if clamp_low is not None and clamp_high is not None:
                actions = torch.max(torch.min(actions, clamp_high), clamp_low)
            if freeze_gripper:
                actions[..., gripper_index] = actions_init_t[..., gripper_index]
            # PDB-R3: 梯度更新后，检查 grad norm, action变化量, clamp是否触边
            #if step == int(steps) - 1: import pdb; pdb.set_trace()  # actions, grad.norm(), rewards
        actions = actions.detach().requires_grad_(True)

    if select_mode == "final_forward":
        with torch.no_grad():
            final_rewards = verifier.score(instruction, image_path, actions)
            final_actions = actions
    elif select_mode == "last_rewards":
        final_rewards = rewards.detach()
        final_actions = actions.detach()
    elif select_mode == "best_rewards":
        if best_rewards is None or best_actions is None:
            final_rewards = rewards.detach()
            final_actions = actions.detach()
        else:
            final_rewards = best_rewards.detach()
            final_actions = best_actions.detach()
    else:
        raise ValueError(f"Unsupported select_mode: {select_mode}")

    return ActionRefineResult(
        actions=final_actions.detach().cpu().numpy(),
        rewards=final_rewards.detach().float().cpu().numpy(),
        reward_trace=None if reward_trace is None else np.stack(reward_trace, axis=0),
        best_actions=None if best_actions is None else best_actions.detach().cpu().numpy(),
        best_rewards=None if best_rewards is None else best_rewards.detach().float().cpu().numpy(),
    )

