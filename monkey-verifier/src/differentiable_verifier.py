from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from action_processing import ActionTokenizer
from infer_server import RobotRewardModel
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

@dataclass
class _KVCacheEntry:
    pixel_values: torch.Tensor
    prefix_input_ids: torch.Tensor
    suffix_input_ids: torch.Tensor
    prefix_past: tuple
    prefix_len: int
    prefix_attn_mask: torch.Tensor
    suffix_embeds: torch.Tensor
    suffix_attn_mask: torch.Tensor
    dtype: torch.dtype
    device: torch.device


class DifferentiableRobotRewardModel(RobotRewardModel):
    """Differentiable wrapper for the Monkey Verifier reward model.

    Note: This assumes your action-to-token path is differentiable in your local
    monkey-verifier variant. If actions are discretized into token IDs, gradients
    w.r.t. actions will not flow.
    """

    def __init__(
        self,
        cfg: Optional[object] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self._device_override = device
        self._dtype_override = dtype
        cfg_get = (cfg.get if isinstance(cfg, dict) else lambda k, d=None: getattr(cfg, k, d))

        # Keep score semantics explicit: the optimizer always maximizes "reward".
        self.diff_score_mode = str(cfg_get("diff_score_mode", "reward")).lower()
        if self.diff_score_mode not in {"reward", "energy"}:
            raise ValueError(
                f"cfg.diff_score_mode must be 'reward' or 'energy', got {self.diff_score_mode}"
            )

        # Support both new and legacy config keys for output activation.
        activation_cfg = cfg_get(
            "diff_reward_activation",
            cfg_get("reward_output_activation", "identity"),
        )
        self.reward_output_activation = str(activation_cfg).lower()
        if self.reward_output_activation not in {"identity", "softplus", "sigmoid"}:
            raise ValueError(
                "reward output activation must be one of "
                "{'identity', 'softplus', 'sigmoid'}, "
                f"got {self.reward_output_activation}"
        )

        # Placeholder consistency: align refine path with legacy rerank path.
        self.placeholder_token = str(cfg_get("action_placeholder_token", "placeholder"))
        expected_placeholder_id = int(cfg_get("action_placeholder_id", 12983))
        tokenizer_placeholder_id = int(
            self.tokenizer.convert_tokens_to_ids(self.placeholder_token)
        )
        if tokenizer_placeholder_id == int(self.tokenizer.unk_token_id):
            raise ValueError(
                f"Unknown action_placeholder_token={self.placeholder_token}. "
                "Make sure tokenizer and training config are aligned."
            )
        if tokenizer_placeholder_id != expected_placeholder_id:
            raise ValueError(
                "Placeholder id mismatch for refine verifier: "
                f"token='{self.placeholder_token}' maps to id={tokenizer_placeholder_id}, "
                f"but expected action_placeholder_id={expected_placeholder_id}. "
                "Set diff_consistency_config or CLI args to keep refine and rerank consistent."
            )
        self.placeholder_id = expected_placeholder_id
        print(
            "[diff_verifier] "
            f"placeholder_token={self.placeholder_token} "
            f"placeholder_id={self.placeholder_id} "
            f"score_mode={self.diff_score_mode} "
            f"activation={self.reward_output_activation}"
        )

        self._action_dim = int(cfg_get("action_dim", 7))
        cfg_token_ids = cfg_get("diff_action_token_ids", None)
        cfg_bins = cfg_get("diff_action_bins", None)
        if cfg_bins is None and cfg_token_ids is not None:
            bins = int(len(cfg_token_ids))
        else:
            bins = int(cfg_get("diff_action_bins", 512))
        min_action = float(cfg_get("diff_action_min", -1.0))
        max_action = float(cfg_get("diff_action_max", 1.0))
        self.action_tokenizer = ActionTokenizer(
            self.tokenizer,
            bins=bins,
            min_action=min_action,
            max_action=max_action,
        )
        self.conv_templates = conv_templates
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.Image = Image
        self._num_bins = int(getattr(self.action_tokenizer, "n_bins", bins))
        self._min_action = float(getattr(self.action_tokenizer, "min_action", min_action))
        self._max_action = float(getattr(self.action_tokenizer, "max_action", max_action))
        # NOTE: Keep this aligned with ActionTokenizer's discretization support points.
        tokenizer_bins = getattr(self.action_tokenizer, "bins", None)
        if tokenizer_bins is not None and len(tokenizer_bins) == self._num_bins:
            self._bin_centers = torch.as_tensor(tokenizer_bins, dtype=torch.float32)
        else:
            self._bin_centers = torch.linspace(self._min_action, self._max_action, self._num_bins)
        bin_width = float(self._bin_centers[1] - self._bin_centers[0]) if self._num_bins > 1 else 1.0
        cfg_sigma = cfg_get("diff_action_sigma", None)
        self._soft_sigma = float(cfg_sigma) if cfg_sigma is not None else bin_width
        self._strict_token_check = bool(cfg_get("diff_action_strict_token_check", True))
        self._log_action_diag = bool(cfg_get("diff_action_log_diagnostics", False))
        self._action_diag_logged = False
        self._hard_action_token_ids = self._build_hard_action_token_ids()
        if cfg_token_ids is not None:
            token_ids = torch.as_tensor(cfg_token_ids, dtype=torch.long).flatten()
            if token_ids.numel() != self._num_bins:
                raise ValueError(
                    "diff_action_token_ids must match the number of action bins."
                )
            self._action_token_ids = token_ids
        else:
            self._action_token_ids = self._hard_action_token_ids.clone()
        self._validate_action_token_ids()
        self._kv_cache: Dict[Tuple[str, str], _KVCacheEntry] = {}
        self._img_cache: Dict[Tuple[str, torch.device], torch.Tensor] = {}

    def _build_hard_action_token_ids(self) -> torch.Tensor:
        """Build canonical token ids from integer bin indices to avoid float-digitize ambiguity."""
        begin = getattr(self.action_tokenizer, "action_token_begin_idx", None)
        if begin is None:
            begin = int(self.tokenizer.vocab_size - 1000 - (self._num_bins + 1))
        begin = int(begin)
        # ActionTokenizer maps discretized bin i in [1..n] to token_id=(begin + (n + 1 - i)).
        token_ids = torch.arange(begin + self._num_bins, begin, -1, dtype=torch.long)
        return token_ids

    def _validate_action_token_ids(self) -> None:
        token_ids = self._action_token_ids
        vocab_size = int(self.tokenizer.vocab_size)
        if int(token_ids.min()) < 0 or int(token_ids.max()) >= vocab_size:
            raise ValueError(
                "Action token ids are out of tokenizer range: "
                f"min={int(token_ids.min())}, max={int(token_ids.max())}, vocab_size={vocab_size}."
            )
        if torch.unique(token_ids).numel() != token_ids.numel():
            raise ValueError("Action token ids must be unique.")
        if not torch.equal(token_ids.cpu(), self._hard_action_token_ids.cpu()):
            max_abs = int((token_ids.cpu() - self._hard_action_token_ids.cpu()).abs().max().item())
            message = (
                "Soft action token ids do not match ActionTokenizer hard discretization. "
                f"max_abs_diff={max_abs} "
                f"soft_range=[{int(token_ids.min())}, {int(token_ids.max())}] "
                f"hard_range=[{int(self._hard_action_token_ids.min())}, {int(self._hard_action_token_ids.max())}]"
            )
            if self._strict_token_check:
                raise ValueError(message)
            print(f"[diff_verifier][warning] {message}")

    def _action_token_ids_on(self, device: torch.device) -> torch.Tensor:
        return self._action_token_ids.to(device=device, dtype=torch.long)

    @torch.no_grad()
    def action_token_diagnostics(
        self,
        sample_count: int = 16,
        sigma: Optional[float] = None,
    ) -> Dict[str, Union[int, float, bool]]:
        """Diagnostic helper for hard/soft token alignment and embedding consistency."""
        device = self._device_override or next(self.model.parameters()).device
        dtype = self._dtype_override or next(self.model.parameters()).dtype
        backbone = self.model.backbone_model
        backbone.set_adapter(self.model.adapter_name)
        centers = self._bin_centers.to(device=device, dtype=torch.float32)
        token_ids = self._action_token_ids_on(device)
        token_embeds = backbone.get_model().embed_tokens(token_ids).to(device=device, dtype=dtype)

        sample_count = max(1, min(int(sample_count), self._num_bins))
        probe_idx = (
            torch.linspace(0, self._num_bins - 1, steps=sample_count, device=device)
            .round()
            .long()
            .unique(sorted=True)
        )
        probe_actions = centers[probe_idx]
        if sigma is None:
            sigma_val = min(self._soft_sigma, float((centers[1] - centers[0]).abs())) if self._num_bins > 1 else self._soft_sigma
            sigma_val = max(1e-6, 0.1 * sigma_val)
        else:
            sigma_val = max(1e-6, float(sigma))
        sigma_t = torch.tensor(sigma_val, device=device, dtype=torch.float32)

        diff = probe_actions.unsqueeze(-1) - centers
        weights = torch.softmax(-0.5 * (diff / sigma_t).pow(2), dim=-1)
        soft_embeds = torch.matmul(weights.to(dtype), token_embeds)
        hard_embeds = token_embeds[probe_idx]
        cosine = F.cosine_similarity(soft_embeds.float(), hard_embeds.float(), dim=-1)

        hard_ids = self._hard_action_token_ids.to(device=device)
        return {
            "vocab_size": int(self.tokenizer.vocab_size),
            "num_bins": int(self._num_bins),
            "soft_token_min": int(token_ids.min().item()),
            "soft_token_max": int(token_ids.max().item()),
            "hard_token_min": int(hard_ids.min().item()),
            "hard_token_max": int(hard_ids.max().item()),
            "hard_soft_exact_match": bool(torch.equal(token_ids.cpu(), hard_ids.cpu())),
            "hard_soft_max_abs_diff": int((token_ids - hard_ids).abs().max().item()),
            "embedding_cosine_min": float(cosine.min().item()),
            "embedding_cosine_mean": float(cosine.mean().item()),
            "embedding_cosine_max": float(cosine.max().item()),
            "diag_sigma": float(sigma_val),
        }

    def score(
        self,
        instruction: str,
        image_path: str,
        actions: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        if torch.is_tensor(actions):
            action_tensor = actions
        else:
            action_tensor = torch.as_tensor(actions, dtype=torch.float32)

        return self._forward_rewards(instruction, image_path, action_tensor)

    def _build_prompt_parts(self, instruction: str) -> Tuple[str, str, str]:
        instruction = instruction.lower().rstrip(".")
        prefix = (
            "shows the current observation from the robot's wrist-mounted camera. "
            f"The robot manipulation arm is attempting to {instruction}. "
            "Please evaluate the quality of the robot action.\n\n"
            f"The robot action ({self._action_dim}D) is:\n"
        )
        # Keep leading/trailing spaces so each placeholder maps to one token.
        placeholders = " " + " ".join([self.placeholder_token] * self._action_dim) + " "
        suffix = "Quality score:"
        return prefix, placeholders, suffix

    def _tokenize_prompt_parts(
        self, prefix: str, placeholders: str, suffix: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conv_mode = "vicuna_v1"
        conv_template = self.conv_templates[conv_mode].copy()
        full = self.DEFAULT_IMAGE_TOKEN + "\n" + prefix + placeholders + suffix
        conv = conv_template.copy()
        conv.append_message(conv.roles[0], full)
        prompt = conv.get_prompt().replace("<image>", f" {self.placeholder_token} ")
        in_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            max_length=self.tokenizer.model_max_length + 2,
            truncation=True,
        ).input_ids

        first_image_idx = (in_ids == self.placeholder_id).nonzero()
        if first_image_idx.numel() == 0:
            raise ValueError(
                "Image placeholder token not found in prompt tokenization. "
                f"placeholder_token={self.placeholder_token}, placeholder_id={self.placeholder_id}"
            )
        start_idx = first_image_idx[0][1].item()
        in_ids[0, start_idx : start_idx + 1] = -200
        in_ids = in_ids[:, :-1]

        action_positions = (in_ids[0] == self.placeholder_id).nonzero().flatten()
        if action_positions.numel() != self._action_dim:
            raise ValueError(
                f"Expected {self._action_dim} action placeholders, got {action_positions.numel()}."
            )
        action_positions = torch.sort(action_positions)[0]
        if not torch.all(
            action_positions
            == torch.arange(
                action_positions[0],
                action_positions[0] + self._action_dim,
                device=action_positions.device,
            )
        ):
            raise ValueError("Action placeholders are not contiguous.")

        p0 = int(action_positions[0].item())
        p_last = int(action_positions[-1].item())
        prefix_ids = in_ids[:, :p0]
        suffix_ids = in_ids[:, p_last + 1 :]
        return in_ids, prefix_ids, suffix_ids

    def _get_pixel_values_cached(
        self,
        image_path: str,
        device: torch.device,
        dtype_img: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        key = (image_path, device)
        cached = self._img_cache.get(key)
        if cached is not None:
            return cached

        processor = self.data_args.image_processor
        image = self.Image.open(image_path).convert("RGB")

        if self.data_args.image_aspect_ratio == "pad":
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                if width > height:
                    result = self.Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                result = self.Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

            image = expand2square(
                image, tuple(int(x * 255) for x in processor.image_mean)
            )
        pixel_values = processor.preprocess(image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device=device, dtype=dtype_img)
        self._img_cache[key] = pixel_values
        return pixel_values

    def _repeat_past(self, past, batch_size: int):
        if batch_size == 1:
            return past
        out = []
        for layer in past:
            if layer is None:
                out.append(None)
                continue
            if len(layer) == 2:
                k, v = layer
                extra = None
            else:
                k, v, extra = layer
            k2 = k.expand(batch_size, *k.shape[1:]).contiguous()
            v2 = v.expand(batch_size, *v.shape[1:]).contiguous()
            out.append((k2, v2) if extra is None else (k2, v2, extra))
        return tuple(out)

    def _get_cache_entry(
        self, instruction: str, image_path: str, device: torch.device, dtype: torch.dtype
    ) -> _KVCacheEntry:
        key = (instruction, image_path)
        cached = self._kv_cache.get(key)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached

        backbone = self.model.backbone_model
        backbone.set_adapter(self.model.adapter_name)
        backbone.config.use_cache = True

        with torch.no_grad():
            pixel_values = self._get_pixel_values_cached(
                image_path, device=device, dtype_img=torch.bfloat16
            )
            prefix, placeholders, suffix = self._build_prompt_parts(instruction)
            full_ids, prefix_ids, suffix_ids = self._tokenize_prompt_parts(
                prefix, placeholders, suffix
            )

            attention_mask = full_ids.ne(self.tokenizer.pad_token_id).long()
            full_ids = full_ids.to(device=device, dtype=torch.int64)
            attention_mask = attention_mask.to(device=device, dtype=torch.int64)
            _, attention_mask, _, inputs_embeds, _ = backbone.prepare_inputs_labels_for_multimodal(
                full_ids, attention_mask, None, None, pixel_values
            )

            if inputs_embeds is None:
                raise ValueError("Expected inputs_embeds from multimodal preparation.")

            inputs_embeds = inputs_embeds.to(device=device, dtype=dtype)
            attention_mask = attention_mask.to(device=device, dtype=torch.int64)

            input_len = full_ids.shape[1]
            embed_len = inputs_embeds.shape[1]
            shift = embed_len - input_len

            image_idx = (full_ids[0] == -200).nonzero().flatten()
            if image_idx.numel() == 0:
                raise ValueError("IMAGE_TOKEN_INDEX not found in input_ids.")
            image_idx = int(image_idx[0].item())

            action_pos = (full_ids[0] == self.placeholder_id).nonzero().flatten()
            if action_pos.numel() != self._action_dim:
                raise ValueError(
                    f"Expected {self._action_dim} action placeholders, got {action_pos.numel()}."
                )
            action_pos = torch.sort(action_pos)[0]
            shifted = action_pos + (action_pos > image_idx) * shift
            p0s = int(shifted[0].item())
            p_last = int(shifted[-1].item())

            prefix_embeds = inputs_embeds[:, :p0s, :]
            prefix_mask = attention_mask[:, :p0s]
            suffix_embeds = inputs_embeds[:, p_last + 1 :, :]
            suffix_mask = attention_mask[:, p_last + 1 :]

            core_model = getattr(backbone, "model", backbone)
            if hasattr(core_model, "model"):
                core_model = core_model.model
            cache_len = int(prefix_embeds.shape[1])
            prev_cache_shape = getattr(core_model.config, "cache_shape", None)
            core_model.config.cache_shape = (cache_len,)
            pos_prefix = torch.arange(prefix_embeds.shape[1], device=device).unsqueeze(0)
            out = core_model(
                input_ids=None,
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_mask,
                position_ids=pos_prefix,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            core_model.config.cache_shape = prev_cache_shape
            prefix_past = out.past_key_values
            if prefix_past is None or any(layer is None for layer in prefix_past):
                raise ValueError("Failed to build KV cache; cache_shape may be unset.")
            prefix_len = prefix_embeds.shape[1]

        entry = _KVCacheEntry(
            pixel_values=pixel_values,
            prefix_input_ids=prefix_ids.to(device),
            suffix_input_ids=suffix_ids.to(device),
            prefix_past=prefix_past,
            prefix_len=prefix_len,
            prefix_attn_mask=prefix_mask,
            suffix_embeds=suffix_embeds,
            suffix_attn_mask=suffix_mask,
            dtype=dtype,
            device=device,
        )
        self._kv_cache[key] = entry
        return entry

    def _forward_rewards(
        self,
        instruction: str,
        image_path: str,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        device = self._device_override or next(self.model.parameters()).device
        dtype = self._dtype_override or next(self.model.parameters()).dtype
        actions = actions.to(device=device, dtype=torch.float32)
        if actions.ndim != 2:
            raise ValueError("Actions must be a 2D tensor [batch, action_dim].")
        batch_size, action_dim = actions.shape
        if action_dim != self._action_dim:
            raise ValueError(
                f"Differentiable verifier expects {self._action_dim}D actions."
            )
        if self._log_action_diag and not self._action_diag_logged:
            diag = self.action_token_diagnostics()
            print(
                "[diff_verifier][diag] "
                f"soft_range=[{diag['soft_token_min']},{diag['soft_token_max']}] "
                f"hard_range=[{diag['hard_token_min']},{diag['hard_token_max']}] "
                f"exact_match={diag['hard_soft_exact_match']} "
                f"cosine(min/mean/max)="
                f"{diag['embedding_cosine_min']:.4f}/"
                f"{diag['embedding_cosine_mean']:.4f}/"
                f"{diag['embedding_cosine_max']:.4f}"
            )
            self._action_diag_logged = True

        instruction = instruction.lower().rstrip(".")
        backbone = self.model.backbone_model
        backbone.set_adapter(self.model.adapter_name)
        backbone.config.use_cache = True

        cache_entry = self._get_cache_entry(instruction, image_path, device=device, dtype=dtype)

        centers = self._bin_centers.to(device=device, dtype=torch.float32)
        actions = torch.clamp(actions, self._min_action, self._max_action)
        sigma = torch.tensor(self._soft_sigma, device=device, dtype=torch.float32)
        diff = actions.unsqueeze(-1) - centers
        weights = torch.softmax(-0.5 * (diff / sigma).pow(2), dim=-1)

        # PDB-V1: soft kernel核心，检查 weights 的 sparsity（每行最大值是否接近1.0 = 接近hard）
        #import pdb; pdb.set_trace()  # weights.max(-1).values, sigma, actions[0], centers[:5]

        token_ids = self._action_token_ids_on(device)
        token_embeds = backbone.get_model().embed_tokens(token_ids)
        token_embeds = token_embeds.to(device=device, dtype=dtype)
        action_embeds = torch.matmul(weights.to(dtype), token_embeds)

        # PDB-V2: soft embedding结果，可对比 hard embedding: token_embeds[weights.argmax(-1)]
        #import pdb; pdb.set_trace()  # action_embeds.shape, action_embeds[0,0,:5] vs hard

        suffix_embeds = cache_entry.suffix_embeds.expand(batch_size, -1, -1)
        cur_embeds = torch.cat([action_embeds, suffix_embeds], dim=1)

        # No padding in the cached prompt; let the model build a full-ones mask.
        attention_mask = None

        cur_len = cur_embeds.shape[1]
        position_ids = torch.arange(
            cache_entry.prefix_len,
            cache_entry.prefix_len + cur_len,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1)

        core_model = getattr(backbone, "model", backbone)
        if hasattr(core_model, "model"):
            core_model = core_model.model
        past = self._repeat_past(cache_entry.prefix_past, batch_size)
        prev_cache_shape = getattr(core_model.config, "cache_shape", None)
        core_model.config.cache_shape = None
        outputs = core_model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past,
            inputs_embeds=cur_embeds,
            position_ids=position_ids,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        core_model.config.cache_shape = prev_cache_shape
        last_hidden_state = outputs.last_hidden_state[:, -1, :].type_as(
            self.model.reward_head.weight
        )
        raw_rewards = self.model.reward_head(last_hidden_state).squeeze(-1)
        if self.reward_output_activation == "sigmoid":
            rewards = torch.sigmoid(raw_rewards)
        elif self.reward_output_activation == "softplus":
            rewards = F.softplus(raw_rewards)
        else:
            rewards = raw_rewards

        # If verifier output is "energy" (lower is better), convert to reward.
        if self.diff_score_mode == "energy":
            rewards = -rewards

        # PDB-V3: verifier最终输出，检查 raw_rewards vs rewards, grad_fn 是否存在
        #import pdb; pdb.set_trace()  # raw_rewards, rewards, rewards.grad_fn, self.diff_score_mode
        return rewards

    def _inject_action_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        if actions.ndim != 2:
            raise ValueError("Actions must be a 2D tensor [batch, action_dim].")

        batch_size, action_dim = actions.shape
        input_len = input_ids.shape[1]
        embed_len = inputs_embeds.shape[1]
        shift = embed_len - input_len

        centers = self._bin_centers.to(device=device, dtype=torch.float32)
        actions = torch.clamp(actions, self._min_action, self._max_action)
        sigma = torch.tensor(self._soft_sigma, device=device, dtype=torch.float32)
        diff = actions.unsqueeze(-1) - centers
        weights = torch.softmax(-0.5 * (diff / sigma).pow(2), dim=-1)

        token_ids = self._action_token_ids_on(device)
        token_embeds = self.model.backbone_model.get_model().embed_tokens(token_ids)
        token_embeds = token_embeds.to(device=device, dtype=dtype)
        action_embeds = torch.matmul(weights.to(dtype), token_embeds)

        new_inputs = inputs_embeds.clone()
        ref_ids = input_ids[0]
        if not torch.equal(input_ids, ref_ids.expand_as(input_ids)):
            for b in range(batch_size):
                image_idx = (input_ids[b] == -200).nonzero()
                if image_idx.numel() == 0:
                    raise ValueError("IMAGE_TOKEN_INDEX not found in input_ids.")
                image_idx = int(image_idx[0].item())
                action_positions = (input_ids[b] == self.placeholder_id).nonzero().flatten()
                if action_positions.numel() != action_dim:
                    raise ValueError(
                        f"Expected {action_dim} action placeholders, got {action_positions.numel()}."
                    )
                action_positions = torch.sort(action_positions)[0]
                shifted_positions = action_positions + (action_positions > image_idx) * shift
                for i in range(action_dim):
                    pos = int(shifted_positions[i].item())
                    new_inputs[b, pos, :] = action_embeds[b, i, :]
            return new_inputs

        image_idx = (ref_ids == -200).nonzero()
        if image_idx.numel() == 0:
            raise ValueError("IMAGE_TOKEN_INDEX not found in input_ids.")
        image_idx = int(image_idx[0].item())
        action_positions = (ref_ids == self.placeholder_id).nonzero().flatten()
        if action_positions.numel() != action_dim:
            raise ValueError(
                f"Expected {action_dim} action placeholders, got {action_positions.numel()}."
            )
        action_positions = torch.sort(action_positions)[0]
        shifted_positions = action_positions + (action_positions > image_idx) * shift
        batch_idx = torch.arange(batch_size, device=device)[:, None]
        new_inputs[batch_idx, shifted_positions[None, :], :] = action_embeds
        return new_inputs
