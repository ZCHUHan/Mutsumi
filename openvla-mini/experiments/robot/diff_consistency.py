"""Shared loading/validation for differentiable verifier consistency config."""

import json
import os
from typing import Any


def apply_diff_consistency_config(cfg: Any) -> None:
    """Load and apply training-time consistency config for differentiable verifier."""
    config_path = getattr(cfg, "diff_consistency_config", None)
    if not config_path:
        return

    config_path = os.path.abspath(str(config_path))
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"diff_consistency_config not found: {config_path}")

    with open(config_path, "r") as fin:
        consistency = json.load(fin)

    if not isinstance(consistency, dict):
        raise ValueError(
            f"diff_consistency_config must be a JSON object, got {type(consistency)}"
        )

    strict = bool(getattr(cfg, "diff_consistency_strict", True))
    required_keys = [
        "action_placeholder_token",
        "action_placeholder_id",
        "action_dim",
        "diff_action_bins",
        "diff_action_min",
        "diff_action_max",
        "diff_action_token_ids",
        "diff_action_strict_token_check",
    ]
    missing = [k for k in required_keys if k not in consistency]
    if missing and strict:
        raise ValueError(
            "diff_consistency_config is missing required keys: " + ", ".join(missing)
        )

    if strict:
        has_score_mode = ("diff_score_mode" in consistency) or (
            "rm_loss_type" in consistency
        )
        has_activation = ("diff_reward_activation" in consistency) or (
            "reward_output_activation" in consistency
        )
        if not has_score_mode:
            raise ValueError(
                "diff_consistency_config is missing score semantics. "
                "Expected one of: diff_score_mode or rm_loss_type."
            )
        if not has_activation:
            raise ValueError(
                "diff_consistency_config is missing activation semantics. "
                "Expected one of: diff_reward_activation or reward_output_activation."
            )

    key_map = {
        "action_placeholder_token": "action_placeholder_token",
        "action_placeholder_id": "action_placeholder_id",
        "action_dim": "action_dim",
        "diff_action_bins": "diff_action_bins",
        "diff_action_min": "diff_action_min",
        "diff_action_max": "diff_action_max",
        "diff_action_token_ids": "diff_action_token_ids",
        "diff_action_sigma": "diff_action_sigma",
        "diff_action_strict_token_check": "diff_action_strict_token_check",
        "diff_action_log_diagnostics": "diff_action_log_diagnostics",
        "diff_score_mode": "diff_score_mode",
        "diff_reward_activation": "diff_reward_activation",
    }
    for src_key, dst_key in key_map.items():
        if src_key in consistency and consistency[src_key] is not None:
            setattr(cfg, dst_key, consistency[src_key])

    # Backward-compatible fallback.
    if (
        "diff_reward_activation" not in consistency
        and "reward_output_activation" in consistency
        and consistency["reward_output_activation"] is not None
    ):
        setattr(cfg, "diff_reward_activation", consistency["reward_output_activation"])

    # Backward-compatible fallback.
    if (
        "diff_score_mode" not in consistency
        and "rm_loss_type" in consistency
        and consistency["rm_loss_type"] is not None
    ):
        rm_loss_type = str(consistency["rm_loss_type"]).lower()
        if rm_loss_type in {"energy", "rmse"}:
            setattr(cfg, "diff_score_mode", "energy")
        else:
            setattr(cfg, "diff_score_mode", "reward")

    token_ids = getattr(cfg, "diff_action_token_ids", None)
    if token_ids is not None:
        bins = int(getattr(cfg, "diff_action_bins"))
        if len(token_ids) != bins:
            raise ValueError(
                "Loaded diff_action_token_ids length does not match diff_action_bins: "
                f"{len(token_ids)} != {bins}"
            )
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("Loaded diff_action_token_ids must be unique.")

    print(
        "[diff_consistency] loaded "
        f"path={config_path} "
        f"placeholder={getattr(cfg, 'action_placeholder_token', None)} "
        f"action_dim={getattr(cfg, 'action_dim', None)} "
        f"bins={getattr(cfg, 'diff_action_bins', None)} "
        f"range=[{getattr(cfg, 'diff_action_min', None)},{getattr(cfg, 'diff_action_max', None)}] "
        f"score_mode={getattr(cfg, 'diff_score_mode', None)} "
        f"activation={getattr(cfg, 'diff_reward_activation', None)}"
    )


def warn_missing_diff_consistency(cfg: Any) -> None:
    """Warn when refine is enabled without explicit consistency config."""
    if bool(getattr(cfg, "use_action_refine", False)) and not getattr(
        cfg, "diff_consistency_config", None
    ):
        print(
            "[diff_consistency][warning] use_action_refine=True but "
            "diff_consistency_config is not set. "
            "Inference may drift from training-time energy verifier semantics."
        )
