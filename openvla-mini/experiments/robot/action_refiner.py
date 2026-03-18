from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
import math
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


def refine_actions_with_budget(
    actions_init: np.ndarray,
    instruction: str,
    image_path: str,
    verifier: DifferentiableVerifier,
    budget_forward_eq: float,
    backward_eq: float = 2.0,
    allocation: str = "uniform",
    cap_strategy: str = "first",
    warmup_steps: int = 2,
    min_steps: int = 1,
    max_steps: Optional[int] = None,
    lr: float = 1e-2,
    prox_weight: float = 0.1,
    prior_mode: str = "diag",
    eps: float = 1e-6,
    select_mode: str = "best_rewards",
    log_every: int = 0,
    clamp_low: Optional[torch.Tensor] = None,
    clamp_high: Optional[torch.Tensor] = None,
    prior_mean: Optional[object] = None,
    prior_var: Optional[object] = None,
    freeze_gripper: bool = False,
    gripper_index: int = -1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> ActionRefineResult:
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    actions_init = np.asarray(actions_init)
    num_candidates = int(actions_init.shape[0])
    stats = {
        "forward_eq_budget": float(budget_forward_eq),
        "backward_eq": float(backward_eq),
        "allocation": str(allocation),
        "cap_strategy": str(cap_strategy),
        "select_mode": str(select_mode),
        "warmup_steps": int(warmup_steps),
        "min_steps": int(min_steps),
        "max_steps": None if max_steps is None else int(max_steps),
        "num_candidates": num_candidates,
        "orig_candidates": num_candidates,
        "truncated_candidates": False,
    }
    orig_candidates = num_candidates

    if num_candidates == 0:
        stats["forward_eq_used"] = 0.0
        return ActionRefineResult(actions=actions_init, rewards=np.array([]), stats=stats)

    step_cost = 1.0 + float(backward_eq)
    extra_forward = num_candidates if select_mode == "final_forward" else 0
    budget_steps = int(math.floor((float(budget_forward_eq) - extra_forward) / step_cost))
    budget_steps = max(0, budget_steps)
    stats["step_cost"] = float(step_cost)
    stats["extra_forward"] = float(extra_forward)
    stats["budget_steps"] = int(budget_steps)

    if min_steps < 1:
        min_steps = 1
    max_candidates = budget_steps // min_steps if budget_steps > 0 else 0
    if 0 < max_candidates < num_candidates:
        strategy = str(cap_strategy).lower()
        if strategy == "random":
            selected_idx = np.random.choice(num_candidates, size=max_candidates, replace=False)
            actions_init = actions_init[selected_idx]
        else:
            selected_idx = slice(0, max_candidates)
            actions_init = actions_init[:max_candidates]
        num_candidates = int(actions_init.shape[0])
        stats["num_candidates"] = num_candidates
        stats["truncated_candidates"] = True
        stats["max_candidates"] = int(max_candidates)
        if prior_mean is not None:
            if isinstance(prior_mean, torch.Tensor):
                if prior_mean.ndim >= 2:
                    prior_mean = prior_mean[selected_idx]
            else:
                prior_mean_arr = np.asarray(prior_mean)
                if prior_mean_arr.ndim >= 2:
                    prior_mean = prior_mean_arr[selected_idx]
        if prior_var is not None:
            if isinstance(prior_var, torch.Tensor):
                if prior_var.ndim >= 2:
                    prior_var = prior_var[selected_idx]
            else:
                prior_var_arr = np.asarray(prior_var)
                if prior_var_arr.ndim >= 2:
                    prior_var = prior_var_arr[selected_idx]
        extra_forward = num_candidates if select_mode == "final_forward" else 0
        budget_steps = int(math.floor((float(budget_forward_eq) - extra_forward) / step_cost))
        budget_steps = max(0, budget_steps)
        stats["extra_forward"] = float(extra_forward)
        stats["budget_steps"] = int(budget_steps)

    if budget_steps <= 0:
        max_candidates = max(1, int(math.floor(float(budget_forward_eq))))
        if num_candidates > max_candidates:
            strategy = str(cap_strategy).lower()
            if strategy == "random":
                idx = np.random.choice(num_candidates, size=max_candidates, replace=False)
                actions_init = actions_init[idx]
            else:
                actions_init = actions_init[:max_candidates]
            num_candidates = int(actions_init.shape[0])
            stats["num_candidates"] = num_candidates
            stats["truncated_candidates"] = True
        with torch.no_grad():
            rewards = verifier.score(
                instruction, image_path, torch.as_tensor(actions_init, device=device, dtype=torch.float32)
            )
            if rewards.ndim == 0:
                rewards = rewards.expand(num_candidates)
        stats["forward_eq_used"] = float(num_candidates)
        stats["budget_exceeded"] = float(budget_forward_eq) < float(orig_candidates)
        return ActionRefineResult(
            actions=actions_init,
            rewards=rewards.detach().cpu().numpy(),
            stats=stats,
        )

    allocation = str(allocation).lower()
    if allocation not in {"uniform", "adaptive"}:
        raise ValueError(f"Unsupported allocation: {allocation}")

    steps_per_candidate = None
    forward_eq_used = 0.0

    if allocation == "uniform":
        steps = max(min_steps, budget_steps // num_candidates)
        if max_steps is not None:
            steps = min(steps, int(max_steps))
        steps_per_candidate = [int(steps)] * num_candidates
        result = refine_actions_with_grad(
            actions_init=actions_init,
            instruction=instruction,
            image_path=image_path,
            verifier=verifier,
            steps=steps,
            lr=lr,
            prox_weight=prox_weight,
            prior_mode=prior_mode,
            eps=eps,
            select_mode=select_mode,
            log_every=log_every,
            clamp_low=clamp_low,
            clamp_high=clamp_high,
            prior_mean=prior_mean,
            prior_var=prior_var,
            freeze_gripper=freeze_gripper,
            gripper_index=gripper_index,
            device=device,
            dtype=dtype,
            return_trace=False,
            track_best=(select_mode == "best_rewards"),
        )
        forward_eq_used = num_candidates * steps * step_cost + extra_forward
        stats["steps_per_candidate"] = steps_per_candidate
        stats["forward_eq_used"] = float(forward_eq_used)
        return ActionRefineResult(
            actions=result.actions,
            rewards=result.rewards,
            best_actions=result.best_actions,
            best_rewards=result.best_rewards,
            stats=stats,
        )

    warmup_steps = min(int(warmup_steps), budget_steps // num_candidates)
    warmup_steps = max(1, warmup_steps)
    if max_steps is not None:
        warmup_steps = min(warmup_steps, int(max_steps))

    warmup = refine_actions_with_grad(
        actions_init=actions_init,
        instruction=instruction,
        image_path=image_path,
        verifier=verifier,
        steps=warmup_steps,
        lr=lr,
        prox_weight=prox_weight,
        prior_mode=prior_mode,
        eps=eps,
        select_mode="last_rewards",
        log_every=log_every,
        clamp_low=clamp_low,
        clamp_high=clamp_high,
        prior_mean=prior_mean,
        prior_var=prior_var,
        freeze_gripper=freeze_gripper,
        gripper_index=gripper_index,
        device=device,
        dtype=dtype,
        return_trace=True,
        track_best=True,
    )

    remaining_steps = max(0, budget_steps - warmup_steps * num_candidates)
    if warmup.reward_trace is not None and warmup.reward_trace.shape[0] > 1:
        slopes = warmup.reward_trace[-1] - warmup.reward_trace[0]
    elif warmup.reward_trace is not None:
        slopes = warmup.reward_trace[-1]
    else:
        slopes = warmup.rewards

    slopes = np.asarray(slopes, dtype=np.float32)
    pos = np.maximum(slopes, 0.0)
    if remaining_steps <= 0:
        steps_per_candidate = [int(warmup_steps)] * num_candidates
        final_actions = warmup.actions
        final_rewards = warmup.rewards
        best_actions = warmup.best_actions
        best_rewards = warmup.best_rewards
    else:
        if pos.sum() <= 0:
            weights = np.ones_like(pos) / float(num_candidates)
        else:
            weights = pos / float(pos.sum())
        alloc = np.floor(weights * remaining_steps).astype(int)
        leftover = int(remaining_steps - alloc.sum())
        if leftover > 0:
            residual = weights * remaining_steps - alloc
            order = np.argsort(-residual)
            alloc[order[:leftover]] += 1

        steps_per_candidate = (alloc + warmup_steps).astype(int).tolist()
        if max_steps is not None:
            steps_per_candidate = [min(int(max_steps), s) for s in steps_per_candidate]

        final_actions = np.array(warmup.actions, copy=True)
        final_rewards = np.array(warmup.rewards, copy=True)
        best_actions = None if warmup.best_actions is None else np.array(warmup.best_actions, copy=True)
        best_rewards = None if warmup.best_rewards is None else np.array(warmup.best_rewards, copy=True)

        for idx, total_steps in enumerate(steps_per_candidate):
            extra_steps = int(total_steps - warmup_steps)
            if extra_steps <= 0:
                continue
            extra = refine_actions_with_grad(
                actions_init=final_actions[idx : idx + 1],
                instruction=instruction,
                image_path=image_path,
                verifier=verifier,
                steps=extra_steps,
                lr=lr,
                prox_weight=prox_weight,
                prior_mode=prior_mode,
                eps=eps,
                select_mode="last_rewards",
                log_every=log_every,
                clamp_low=clamp_low,
                clamp_high=clamp_high,
                prior_mean=None if prior_mean is None else np.asarray(prior_mean)[idx : idx + 1],
                prior_var=None if prior_var is None else np.asarray(prior_var)[idx : idx + 1],
                freeze_gripper=freeze_gripper,
                gripper_index=gripper_index,
                device=device,
                dtype=dtype,
                return_trace=False,
                track_best=True,
            )
            final_actions[idx] = extra.actions[0]
            final_rewards[idx] = extra.rewards[0]
            if best_rewards is not None and extra.best_rewards is not None:
                if extra.best_rewards[0] > best_rewards[idx]:
                    best_rewards[idx] = extra.best_rewards[0]
                    best_actions[idx] = extra.best_actions[0]

    forward_eq_used = float(sum(steps_per_candidate) * step_cost)
    if select_mode == "final_forward":
        with torch.no_grad():
            final_rewards_tensor = verifier.score(
                instruction,
                image_path,
                torch.as_tensor(final_actions, device=device, dtype=torch.float32),
            )
            if final_rewards_tensor.ndim == 0:
                final_rewards_tensor = final_rewards_tensor.expand(num_candidates)
        final_rewards = final_rewards_tensor.detach().cpu().numpy()
        forward_eq_used += float(num_candidates)
    elif select_mode == "best_rewards":
        if best_rewards is not None and best_actions is not None:
            final_rewards = best_rewards
            final_actions = best_actions

    stats["steps_per_candidate"] = steps_per_candidate
    stats["forward_eq_used"] = float(forward_eq_used)

    return ActionRefineResult(
        actions=final_actions,
        rewards=final_rewards,
        best_actions=best_actions,
        best_rewards=best_rewards,
        stats=stats,
    )
