"""
run_bridgev2_eval.py

Runs a model in a real-world Bridge V2 environment.

Usage:
    # OpenVLA:
    python experiments/robot/bridge/run_bridgev2_eval.py \\
            --model_family openvla \\
            --pretrained_checkpoint openvla/openvla-7b
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import draccus

# Append current directory so that interpreter can find experiments.robot
sys.path.append(".")
from experiments.robot.diff_consistency import (
    apply_diff_consistency_config,
    warn_missing_diff_consistency,
)
from experiments.robot.bridge.bridgev2_utils import (
    get_next_task_label,
    get_preprocessed_image,
    get_widowx_env,
    refresh_obs,
    save_rollout_data,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                               # Model family
    pretrained_checkpoint: Union[str, Path] = ""                # Pretrained checkpoint path
    load_in_8bit: bool = False                                  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                                  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = False                                   # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # WidowX environment-specific parameters
    #################################################################################################################
    host_ip: str = "localhost"
    port: int = 5556

    # Note: Setting initial orientation with a 30 degree offset, which makes the robot appear more natural
    init_ee_pos: List[float] = field(default_factory=lambda: [0.3, -0.09, 0.26])
    init_ee_quat: List[float] = field(default_factory=lambda: [0, -0.259, 0, -0.966])
    bounds: List[List[float]] = field(default_factory=lambda: [
            [0.1, -0.20, -0.01, -1.57, 0],
            [0.45, 0.25, 0.30, 1.57, 0],
        ]
    )

    camera_topics: List[Dict[str, str]] = field(default_factory=lambda: [{"name": "/blue/image_raw"}])

    blocking: bool = False                                      # Whether to use blocking control
    max_episodes: int = 50                                      # Max number of episodes to run
    max_steps: int = 60                                         # Max number of timesteps per episode
    control_frequency: float = 5                                # WidowX control frequency

    #################################################################################################################
    # Utils
    #################################################################################################################
    save_data: bool = False                                     # Whether to save rollout data (images, actions, etc.)

    # fmt: on

    # Robomonkey Config
    initial_samples: int = 5
    augmented_samples: int = 32
    action_server_port: int = 3200
    reward_server_port: int = 3100
    reward_batch_size: int = 2

    # Differentiable action refinement
    use_action_refine: bool = False
    action_refine_steps: int = 10
    action_refine_lr: float = 1e-2
    action_refine_prox_weight: float = 0.1
    action_refine_prior: str = "diag"  # "diag" or "l2"
    action_refine_eps: float = 1e-6
    action_refine_clamp: bool = True
    action_refine_select: str = "final_forward"  # "final_forward", "last_rewards", "best_rewards"
    action_refine_normalize: bool = True
    action_refine_log_every: int = 0
    action_refine_allocation: str = "uniform"  # "uniform" or "adaptive"
    action_refine_warmup_steps: int = 2
    action_refine_min_steps: int = 1
    action_refine_max_steps: Optional[int] = None
    action_refine_freeze_gripper: bool = True  # keep gripper fixed during refine
    action_refine_gripper_index: int = -1  # last action dim by default
    action_refine_every_n_steps: int = 1  # refine every N control steps
    action_refine_start_step: int = 0  # first step index eligible for refine
    action_refine_end_step: Optional[int] = None  # optional last step index eligible for refine
    action_refine_skip_strategy: str = "first"  # "first" or "rerank"
    diff_action_bins: int = 512
    diff_action_min: float = -1.0
    diff_action_max: float = 1.0
    diff_action_sigma: Optional[float] = 0.008
    diff_action_strict_token_check: bool = True
    diff_action_log_diagnostics: bool = False
    diff_action_token_ids: Optional[list[int]] = None
    action_placeholder_token: str = "<ACT>"
    diff_score_mode: str = "energy"  # "reward" or "energy"
    diff_reward_activation: str = "softplus"  # "identity", "softplus", or "sigmoid"
    diff_consistency_config: Optional[str] = None
    diff_consistency_strict: bool = True
    diff_verifier_module: Optional[str] = None
    diff_verifier_class: Optional[str] = None
    diff_verifier_hidden_dim: int = 256
    diff_verifier_ckpt: Optional[str] = None
    action_dim: int = 7
    verifier_forward_eq_budget: Optional[float] = None
    verifier_forward_eq_backward_ratio: float = 2.0
    verifier_budget_apply_to: str = "both"  # "rerank", "refine", or "both"
    verifier_budget_rerank_strategy: str = "first"  # "first" or "random"
    verifier_forward_eq_log: bool = True
    verifier_forward_eq_track: bool = False


@draccus.wrap()
def eval_model_in_bridge_env(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    assert not cfg.center_crop, "`center_crop` should be disabled for Bridge evaluations!"
    assert cfg.initial_samples > 0, "Invalid initial_samples: should be > 0"
    assert cfg.augmented_samples > 0, "Invalid augmented_samples: should be > 0"

    apply_diff_consistency_config(cfg)
    warn_missing_diff_consistency(cfg)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = "bridge_orig"

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize the WidowX environment
    env = get_widowx_env(cfg, model)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    task_label = ""
    episode_idx = 0
    while episode_idx < cfg.max_episodes:
        # Get task description from user
        task_label = get_next_task_label(task_label)

        # Reset environment
        obs, _ = env.reset()

        # Setup
        t = 0
        step_duration = 1.0 / cfg.control_frequency
        replay_images = []
        if cfg.save_data:
            rollout_images = []
            rollout_states = []
            rollout_actions = []

        # Start episode
        input(f"Press Enter to start episode {episode_idx+1}...")
        print("Starting episode... Press Ctrl-C to terminate episode early!")
        last_tstamp = time.time()
        while t < cfg.max_steps:
            try:
                curr_tstamp = time.time()
                if curr_tstamp > last_tstamp + step_duration:
                    print(f"t: {t}")
                    print(f"Previous step elapsed time (sec): {curr_tstamp - last_tstamp:.2f}")
                    last_tstamp = time.time()

                    # Refresh the camera image and proprioceptive state
                    obs = refresh_obs(obs, env)

                    # Save full (not preprocessed) image for replay video
                    replay_images.append(obs["full_image"])

                    # Get preprocessed image
                    obs["full_image"] = get_preprocessed_image(obs, resize_size)

                    # Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        obs,
                        task_label,
                        processor=processor,
                        step_idx=t,
                    )

                    # [If saving rollout data] Save preprocessed image, robot state, and action
                    if cfg.save_data:
                        rollout_images.append(obs["full_image"])
                        rollout_states.append(obs["proprio"])
                        rollout_actions.append(action)

                    # Execute action
                    print("action:", action)
                    obs, _, _, _, _ = env.step(action)
                    t += 1

            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\nCaught KeyboardInterrupt: Terminating episode early.")
                else:
                    print(f"\nCaught exception: {e}")
                break

        # Save a replay video of the episode
        save_rollout_video(replay_images, episode_idx)

        # [If saving rollout data] Save rollout data
        if cfg.save_data:
            save_rollout_data(replay_images, rollout_images, rollout_states, rollout_actions, idx=episode_idx)

        # Redo episode or continue
        if input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1


if __name__ == "__main__":
    eval_model_in_bridge_env()
