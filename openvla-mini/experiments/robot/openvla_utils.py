"""Utils for evaluating the OpenVLA policy."""

import math
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoProcessor
import imageio

from prismatic.models.load import load_vla

import requests
import json_numpy as json

import sys
sys.path.append("../")

from experiments.robot.token_action_converter import TokenActionConverter
from experiments.robot.action_refiner import (
    load_differentiable_verifier,
    refine_actions_with_grad,
    refine_actions_with_budget,
)

_DIFF_VERIFIER = None
_DIFF_VERIFIER_LOGGED = False
_VERIFIER_FORWARD_EQ_TOTAL = 0.0
_VERIFIER_FORWARD_EQ_LAST = 0.0
_VERIFIER_FORWARD_EQ_LAST_META = None
_REFINE_SKIP_AUG_LOGGED = False
_REFINE_POLICY_LOGGED = False


def reset_verifier_forward_eq() -> None:
    global _VERIFIER_FORWARD_EQ_TOTAL, _VERIFIER_FORWARD_EQ_LAST, _VERIFIER_FORWARD_EQ_LAST_META
    _VERIFIER_FORWARD_EQ_TOTAL = 0.0
    _VERIFIER_FORWARD_EQ_LAST = 0.0
    _VERIFIER_FORWARD_EQ_LAST_META = None


def get_verifier_forward_eq() -> dict:
    return {
        "total": float(_VERIFIER_FORWARD_EQ_TOTAL),
        "last": float(_VERIFIER_FORWARD_EQ_LAST),
        "last_meta": _VERIFIER_FORWARD_EQ_LAST_META,
    }


def _track_forward_eq(cfg, forward_eq_used, mode=None, meta=None) -> None:
    if not bool(getattr(cfg, "verifier_forward_eq_track", False)):
        return
    global _VERIFIER_FORWARD_EQ_TOTAL, _VERIFIER_FORWARD_EQ_LAST, _VERIFIER_FORWARD_EQ_LAST_META
    value = float(forward_eq_used)
    _VERIFIER_FORWARD_EQ_TOTAL += value
    _VERIFIER_FORWARD_EQ_LAST = value
    if mode is None and meta is None:
        _VERIFIER_FORWARD_EQ_LAST_META = None
    else:
        detail = {}
        if mode is not None:
            detail["mode"] = str(mode)
        if meta:
            detail.update(meta)
        _VERIFIER_FORWARD_EQ_LAST_META = detail


def _get_differentiable_verifier(cfg):
    global _DIFF_VERIFIER, _DIFF_VERIFIER_LOGGED
    if _DIFF_VERIFIER is None:
        _DIFF_VERIFIER = load_differentiable_verifier(
            cfg,
            device=DEVICE,
            dtype=torch.float32,
        )
    if not _DIFF_VERIFIER_LOGGED:
        name = f"{_DIFF_VERIFIER.__class__.__module__}.{_DIFF_VERIFIER.__class__.__name__}"
        print(f"[verifier] differentiable verifier={name}")
        if _DIFF_VERIFIER.__class__.__name__ == "ActionOnlyVerifier":
            print("[verifier][warning] fallback ActionOnlyVerifier in use; reward alignment may be weak.")
        _DIFF_VERIFIER_LOGGED = True
    return _DIFF_VERIFIER


def _budget_applies(cfg, mode: str) -> bool:
    apply_to = str(getattr(cfg, "verifier_budget_apply_to", "both")).lower()
    return apply_to in {"both", mode}


def _log_forward_eq(cfg, message: str) -> None:
    if bool(getattr(cfg, "verifier_forward_eq_log", True)):
        print(message)


def preprocess_actions(output_ids, action):
    # Convert arrays to numpy arrays if they aren't already
    output_ids = np.array(output_ids)
    output_ids = np.where(output_ids == 31745, 31744, output_ids)
    action = np.array(action)
    
    # Apply the range filter
    range_mask = np.all((output_ids >= 31744) & (output_ids <= 32000), axis=1)
    output_ids = output_ids[range_mask]
    action = action[range_mask]
    
    return output_ids, action

def get_unique_actions(output_ids, action):
    output_ids = np.array(output_ids)
    action = np.array(action)
    
    # Get unique rows and their indices
    unique_rows, indices = np.unique(output_ids, axis=0, return_index=True)
    
    # Sort indices to maintain original order
    indices = sorted(indices)
    
    return output_ids[indices], action[indices]

def get_rewards(instruction, image_path, actions, cfg):
    # Initialize rewards list
    all_rewards = []
    
    # Batch size is configurable to trade off speed vs memory.
    # The default is 2 to fit a RTX4090 with 24GB memory.
    batch_size = max(1, int(getattr(cfg, "reward_batch_size", 2)))
    num_batches = math.ceil(len(actions) / batch_size)
    
    for i in range(num_batches):
        # Get the current batch of actions
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(actions))
        action_batch = actions[start_idx:end_idx]
        
        payload = {
            "instruction": instruction,
            "image_path": image_path,
            "action": action_batch
        }
        
        response = requests.post(f"http://127.0.0.1:{cfg.reward_server_port}/process", data=json.dumps(payload))
        response_data = json.loads(response.text)
        
        all_rewards.extend(response_data["rewards"])
    
    return all_rewards

def get_batch_actions(instruction: str, image_path: str, batch_size: int = 4, temperature: float = 1.0, cfg = None):
    """
    Get multiple predictions by making individual requests to the processing server.
    """
    # Verify image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    payload = {
        "instructions": [instruction] * int(batch_size),
        "image_path": image_path,
        "temperature": temperature,
    }
    
    response = requests.post(
        f"http://127.0.0.1:{cfg.action_server_port}/batch",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    if response.status_code != 200:
        raise Exception(f"Error from server: {response.text}")
    
    response_data = json.loads(response.text)
    output_ids = np.array(response_data["output_ids"])
    actions = np.array(response_data["actions"])
    if output_ids.shape[0] != int(batch_size):
        print(
            "[action_server][warning] Unexpected batch size in response: "
            f"requested={batch_size}, received={output_ids.shape[0]}"
        )
    
    return output_ids, actions

def generate_augmented_samples_from_batch(batch_actions, num_samples=32):
    """
    Generate augmented samples based on the mean and variance of a batch of actions.
    """
    # Calculate mean and variance for each dimension
    mean_values = np.mean(batch_actions, axis=0)
    var_values = np.var(batch_actions, axis=0)
    
    # Define valid ranges for the action dimensions
    min_values = np.array([-0.02872725307941437,
                         -0.04170349963009357,
                         -0.026093858778476715,
                         -0.08092105075716972,
                         -0.09288699507713317,
                         -0.20718276381492615,
                         0.0])
    max_values = np.array([0.028309678435325586,
                         0.040855254605412394,
                         0.040161586627364146,
                         0.08192047759890528,
                         0.07792850524187081,
                         0.20382574498653397,
                         1.0])
    converter = TokenActionConverter()
        
    # Generate all samples at once
    augmented_array = np.random.normal(
        mean_values, np.sqrt(var_values), 
        size=(num_samples, 7)
    )
    
    # For the 7th dimension (binary), use probability based on mean
    augmented_array[:, -1] = (mean_values[-1] >= 0.5).astype(float)
    
    # Clip values to valid range
    augmented_array[:, :-1] = np.clip(
        augmented_array[:, :-1], 
        min_values[:-1], 
        max_values[:-1]
    )
    
    augmented_ids = np.zeros((num_samples, 7), dtype=np.int64)
    for i in range(num_samples):
        augmented_ids[i] = converter.action_to_token(augmented_array[i])
    
    return augmented_ids, augmented_array

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path
    
def get_prismatic_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Prepare for model loading.
    print(f"[*] Initializing Generation Playground with `{cfg.model_family}`")
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    # set_seed(cfg.seed)
    # Load VLA checkpoint.
    print(f"Loading VLM from checkpoint: {cfg.pretrained_checkpoint}")
    vla = load_vla(
        cfg.pretrained_checkpoint,
        hf_token=hf_token,
        load_for_training=False,
    )
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"
    # Cast to half precision.
    vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
    vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(DEVICE)
    return vla


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    return None


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def apply_center_crop(im, t_h, t_w):
    """
    Source: https://github.com/ARISE-Initiative/robomimic/blob/5dee58f9cc1235010d0877142b54d0e82dd23986/robomimic/utils/obs_utils.py#L268

    Takes a center crop of an image.

    Args:
        im (np.array or torch.Tensor): image of shape (..., height, width, channel)
        t_h (int): height of crop
        t_w (int): width of crop

    Returns:
        im (np.array or torch.Tensor): center cropped image
    """
    assert im.shape[-3] >= t_h and im.shape[-2] >= t_w
    assert im.shape[-1] in [1, 3, 6]
    crop_h = int((im.shape[-3] - t_h) / 2)
    crop_w = int((im.shape[-2] - t_w) / 2)
    return im[..., crop_h : crop_h + t_h, crop_w : crop_w + t_w, :]

#
def get_vla_action(
    vla,
    processor,
    base_vla_name,
    obs,
    task_label,
    unnorm_key,
    center_crop=False,
    cfg=None,
    step_idx=None,
):
    """Generates an action with the VLA policy."""

    # only supports 1 image
    if isinstance(obs["full_image"], list):
        obs["full_image"] = obs["full_image"][0]

    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

        # Save processed image and path for inference
        transfer_dir = f"./transfer_images/"
        os.makedirs(transfer_dir, exist_ok=True)
        image_path = f"{transfer_dir}/vla_processed_img.jpg"
        image.save(image_path)
    
    # Get initial action samples from VLA Serving Engine
    instruction = task_label.lower()
    image_path = str(Path("./transfer_images/vla_processed_img.jpg").absolute())
    output_ids, actions = get_batch_actions(
        instruction=instruction,
        image_path=image_path,
        batch_size=cfg.initial_samples,
        temperature=1,
        cfg=cfg
    )
    #import pdb; pdb.set_trace()
    # Preprocess initial actions
    use_action_refine = bool(getattr(cfg, "use_action_refine", False))
    if cfg.initial_samples == 1 and cfg.augmented_samples == 1 and not use_action_refine:
        return actions[0] 
    output_ids, actions = preprocess_actions(output_ids, actions)
    #import pdb; pdb.set_trace()
    _, unique = get_unique_actions(output_ids, actions)

    if len(unique) == 1 and not use_action_refine:
        return unique[0]

    if use_action_refine:
        global _REFINE_SKIP_AUG_LOGGED, _REFINE_POLICY_LOGGED
        if cfg.augmented_samples != 1 and not _REFINE_SKIP_AUG_LOGGED:
            print("Action refine enabled; skipping Gaussian augmented_samples.")
            _REFINE_SKIP_AUG_LOGGED = True

        output_ids, actions = get_unique_actions(output_ids, actions)
        reward_img = str(Path("./transfer_images/reward_img.jpg").absolute())

        refine_every = max(1, int(getattr(cfg, "action_refine_every_n_steps", 1)))
        refine_start = int(getattr(cfg, "action_refine_start_step", 0))
        refine_end = getattr(cfg, "action_refine_end_step", None)
        if refine_end is not None:
            refine_end = int(refine_end)
        skip_strategy = str(getattr(cfg, "action_refine_skip_strategy", "first")).lower()
        if skip_strategy not in {"first", "rerank"}:
            raise ValueError(
                f"Unsupported action_refine_skip_strategy={skip_strategy}; expected 'first' or 'rerank'."
            )
        if not _REFINE_POLICY_LOGGED:
            print(
                "[action_refine] schedule "
                f"every_n_steps={refine_every} "
                f"start_step={refine_start} "
                f"end_step={refine_end} "
                f"skip_strategy={skip_strategy}"
            )
            _REFINE_POLICY_LOGGED = True

        should_refine = True
        if step_idx is not None:
            cur_step = int(step_idx)
            if cur_step < refine_start:
                should_refine = False
            if refine_end is not None and cur_step > refine_end:
                should_refine = False
            if should_refine and (cur_step - refine_start) % refine_every != 0:
                should_refine = False

        if not should_refine:
            if skip_strategy == "rerank":
                rewards = get_rewards(instruction, reward_img, output_ids, cfg=cfg)
                _track_forward_eq(cfg, float(len(output_ids)), mode="refine_skip_rerank")
                return actions[int(np.argmax(rewards))]
            return actions[0]

        verifier = _get_differentiable_verifier(cfg)
        converter = TokenActionConverter(unnorm_key=cfg.unnorm_key)
        use_normalize = bool(getattr(cfg, "action_refine_normalize", True))
        budget_forward_eq = getattr(cfg, "verifier_forward_eq_budget", None)
        backward_eq = float(getattr(cfg, "verifier_forward_eq_backward_ratio", 2.0))
        allocation = str(getattr(cfg, "action_refine_allocation", "uniform"))
        warmup_steps = int(getattr(cfg, "action_refine_warmup_steps", 2))
        min_steps = int(getattr(cfg, "action_refine_min_steps", 1))
        max_steps = getattr(cfg, "action_refine_max_steps", None)
        freeze_gripper = bool(getattr(cfg, "action_refine_freeze_gripper", True))
        gripper_index = int(getattr(cfg, "action_refine_gripper_index", -1))
        if use_normalize:
            refine_actions = converter.normalize_actions(actions)
            clamp_low = clamp_high = None
            if bool(getattr(cfg, "action_refine_clamp", True)):
                mask_t, _, _ = converter._action_norm_stats_tensors(
                    device=DEVICE,
                    dtype=torch.float32,
                )
                clamp_low = torch.where(
                    mask_t,
                    torch.full_like(mask_t, -1.0, dtype=torch.float32),
                    torch.full_like(mask_t, -float("inf"), dtype=torch.float32),
                )
                clamp_high = torch.where(
                    mask_t,
                    torch.full_like(mask_t, 1.0, dtype=torch.float32),
                    torch.full_like(mask_t, float("inf"), dtype=torch.float32),
                )
        else:
            refine_actions = actions
            clamp_low = clamp_high = None
            if bool(getattr(cfg, "action_refine_clamp", True)):
                mask_t, low_t, high_t = converter._action_norm_stats_tensors(
                    device=DEVICE,
                    dtype=torch.float32,
                )
                clamp_low = torch.where(mask_t, low_t, torch.full_like(low_t, -float("inf")))
                clamp_high = torch.where(mask_t, high_t, torch.full_like(high_t, float("inf")))

        select_mode = str(getattr(cfg, "action_refine_select", "best_rewards"))
        if budget_forward_eq is not None and _budget_applies(cfg, "refine"):
            result = refine_actions_with_budget(
                actions_init=refine_actions,
                instruction=instruction,
                image_path=reward_img,
                verifier=verifier,
                budget_forward_eq=float(budget_forward_eq),
                backward_eq=backward_eq,
                allocation=allocation,
                cap_strategy=str(getattr(cfg, "verifier_budget_rerank_strategy", "first")),
                warmup_steps=warmup_steps,
                min_steps=min_steps,
                max_steps=None if max_steps is None else int(max_steps),
                lr=float(getattr(cfg, "action_refine_lr", 1e-2)),
                prox_weight=float(getattr(cfg, "action_refine_prox_weight", 0.1)),
                prior_mode=str(getattr(cfg, "action_refine_prior", "diag")),
                eps=float(getattr(cfg, "action_refine_eps", 1e-6)),
                select_mode=select_mode,
                log_every=int(getattr(cfg, "action_refine_log_every", 0)),
                clamp_low=clamp_low,
                clamp_high=clamp_high,
                prior_mean=refine_actions,
                freeze_gripper=freeze_gripper,
                gripper_index=gripper_index,
                device=DEVICE,
                dtype=torch.float32,
            )
            if result.stats is not None:
                stats = result.stats
                forward_eq_used = float(stats.get("forward_eq_used", 0.0))
                _track_forward_eq(cfg, forward_eq_used, mode="refine")
        else:
            steps = int(getattr(cfg, "action_refine_steps", 10))
            # import pdb; pdb.set_trace()
            result = refine_actions_with_grad(
                actions_init=refine_actions,
                instruction=instruction,
                image_path=reward_img,
                verifier=verifier,
                steps=steps,
                lr=float(getattr(cfg, "action_refine_lr", 1e-2)),
                prox_weight=float(getattr(cfg, "action_refine_prox_weight", 0.1)),
                prior_mode=str(getattr(cfg, "action_refine_prior", "diag")),
                eps=float(getattr(cfg, "action_refine_eps", 1e-6)),
                select_mode=select_mode,
                log_every=int(getattr(cfg, "action_refine_log_every", 0)),
                clamp_low=clamp_low,
                clamp_high=clamp_high,
                freeze_gripper=freeze_gripper,
                gripper_index=gripper_index,
                device=DEVICE,
                dtype=torch.float32,
            )
            num_candidates = int(refine_actions.shape[0])
            extra_forward = num_candidates if select_mode == "final_forward" else 0
            forward_eq_used = num_candidates * steps * (1.0 + backward_eq) + extra_forward
            if _budget_applies(cfg, "refine"):
                _log_forward_eq(
                    cfg,
                    "[verifier_budget][refine] "
                    f"forward_eq_used={forward_eq_used:.1f} "
                    f"num_candidates={num_candidates} "
                    f"steps={steps} "
                    f"select_mode={select_mode}",
                )
            _track_forward_eq(cfg, forward_eq_used, mode="refine")
        if result.stats is not None:
            stats = result.stats
            _log_forward_eq(
                cfg,
                "[verifier_budget][refine] "
                f"forward_eq_used={stats.get('forward_eq_used', 0):.1f} "
                f"budget={stats.get('forward_eq_budget', None)} "
                f"num_candidates={stats.get('num_candidates', 0)} "
                f"allocation={stats.get('allocation', '')} "
                f"steps_per_candidate={stats.get('steps_per_candidate', None)}",
            )
            if bool(stats.get("truncated_candidates", False)):
                _log_forward_eq(
                    cfg,
                    "[verifier_budget][refine] "
                    f"truncated_candidates orig={stats.get('orig_candidates', None)} "
                    f"kept={stats.get('num_candidates', None)} "
                    f"budget_steps={stats.get('budget_steps', None)} "
                    f"min_steps={stats.get('min_steps', None)}",
                )
        refined_actions = result.actions
        if use_normalize:
            refined_actions = converter.unnormalize_actions(refined_actions)
        selected_index = int(np.argmax(result.rewards))
        # import pdb; pdb.set_trace()
        return refined_actions[selected_index]

    # Generate augmented samples based on the mean and variance of a batch of actions.
    budget_forward_eq = getattr(cfg, "verifier_forward_eq_budget", None)
    num_samples = cfg.augmented_samples
    if budget_forward_eq is not None and _budget_applies(cfg, "rerank"):
        num_samples = max(num_samples, int(budget_forward_eq))
    output_ids, actions = generate_augmented_samples_from_batch(
        batch_actions=actions,
        num_samples=num_samples
    )

    # Score each action with robomonkey verifier
    output_ids, actions = get_unique_actions(output_ids, actions)
    reward_img = str(Path("./transfer_images/reward_img.jpg").absolute())
    if budget_forward_eq is not None and _budget_applies(cfg, "rerank"):
        max_actions = max(1, int(budget_forward_eq))
        if len(actions) > max_actions:
            strategy = str(getattr(cfg, "verifier_budget_rerank_strategy", "first")).lower()
            if strategy == "random":
                idx = np.random.choice(len(actions), size=max_actions, replace=False)
                output_ids = output_ids[idx]
                actions = actions[idx]
            else:
                output_ids = output_ids[:max_actions]
                actions = actions[:max_actions]
            _log_forward_eq(
                cfg,
                "[verifier_budget][rerank] "
                f"capped_candidates={len(actions)} "
                f"budget={budget_forward_eq} "
                f"strategy={strategy}",
            )
    rewards = get_rewards(instruction, reward_img, output_ids, cfg=cfg)
    forward_eq_used = float(len(output_ids))
    if _budget_applies(cfg, "rerank"):
        _log_forward_eq(
            cfg,
            "[verifier_budget][rerank] "
            f"forward_eq_used={len(actions)} "
            f"num_candidates={len(actions)}",
        )
    _track_forward_eq(cfg, forward_eq_used, mode="rerank")

    selected_index = np.argmax(rewards)

    return actions[selected_index]


def get_prismatic_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False, **kwargs):
    """Generates an action with the VLA policy."""

    if not isinstance(obs["full_image"], list):
        obs["full_image"] = [obs["full_image"]]

    processed_images = []

    for img in obs["full_image"]:
        image = Image.fromarray(img)
        image = image.convert("RGB")

        # (If trained with image augmentations) Center crop image and then resize back up to original size.
        # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), we must multiply
        #            the original height and width by sqrt(0.9) -- not 0.9!
        if center_crop:
            temp_image = np.array(image)  # (H, W, C)
            crop_scale = 0.9
            sqrt_crop_scale = math.sqrt(crop_scale)
            temp_image_cropped = apply_center_crop(
                temp_image,
                t_h=int(sqrt_crop_scale * temp_image.shape[0]),
                t_w=int(sqrt_crop_scale * temp_image.shape[1]),
            )
            temp_image = Image.fromarray(temp_image_cropped)
            temp_image = temp_image.resize(
                image.size, Image.Resampling.BILINEAR
            )  # IMPORTANT: dlimp uses BILINEAR resize
            image = temp_image

        processed_images.append(image)

    # extract for single image
    if len(processed_images) == 1:
        processed_images = processed_images[0]

    action = vla.predict_action(processed_images, task_label, unnorm_key=unnorm_key, **kwargs)
    return action
