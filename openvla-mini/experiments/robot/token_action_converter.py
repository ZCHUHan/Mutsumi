from transformers import AutoConfig
import numpy as np
import torch

class TokenActionConverter:
    def __init__(self, n_action_bins: int = 256, unnorm_key: str = "bridge_orig"):
        self.bins = np.linspace(-1, 1, n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size = 32000
        self.unnorm_key = unnorm_key
        self.config = AutoConfig.from_pretrained(
            "openvla/openvla-7b", trust_remote_code=True
        ).to_dict()
        self.norm_stats = self.config["norm_stats"]
        assert unnorm_key is not None
        if unnorm_key not in self.norm_stats:
            raise ValueError(
                f"The `unnorm_key` you chose ({unnorm_key = }) is not in the available statistics. "
                f"Please choose from: {self.norm_stats.keys()}"
            )

    def token_to_action(self, output_ids):
        """Convert token IDs to actions."""
        predicted_action_token_ids = np.array(output_ids)
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.norm_stats[self.unnorm_key]["action"]
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(
            action_norm_stats["q01"]
        )
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) *
            (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions

    @property
    def num_bins(self) -> int:
        return int(self.bin_centers.shape[0])

    def action_to_token(self, actions):
        """Convert actions back to token IDs."""
        # First, normalize the actions back to [-1, 1] range
        action_norm_stats = self.norm_stats[self.unnorm_key]["action"]
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(
            action_norm_stats["q01"]
        )
        normalized_actions = np.where(
            mask,
            2 * (actions - action_low) / (action_high - action_low) - 1,
            actions
        )
        discretized_actions = np.array([
            np.abs(self.bin_centers - val).argmin()
            for val in normalized_actions
        ])
        output_ids = self.vocab_size - discretized_actions - 1
        output_ids = np.array(output_ids)
        output_ids = np.where(output_ids == 31745, 31744, output_ids)

        return output_ids

    def normalize_actions(self, actions):
        """Normalize actions to [-1, 1] using dataset statistics."""
        action_norm_stats = self.norm_stats[self.unnorm_key]["action"]
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high = np.array(action_norm_stats["q99"])
        action_low = np.array(action_norm_stats["q01"])
        actions = np.array(actions)
        return np.where(
            mask,
            2 * (actions - action_low) / (action_high - action_low) - 1,
            actions,
        )

    def unnormalize_actions(self, normalized_actions):
        """Unnormalize actions from [-1, 1] using dataset statistics."""
        action_norm_stats = self.norm_stats[self.unnorm_key]["action"]
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high = np.array(action_norm_stats["q99"])
        action_low = np.array(action_norm_stats["q01"])
        normalized_actions = np.array(normalized_actions)
        return np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

    def bin_centers_tensor(self, device=None, dtype=None) -> torch.Tensor:
        return torch.as_tensor(
            self.bin_centers,
            device=device,
            dtype=dtype if dtype is not None else torch.float32,
        )

    def _action_norm_stats_tensors(self, device, dtype):
        action_norm_stats = self.norm_stats[self.unnorm_key]["action"]
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high = np.array(action_norm_stats["q99"])
        action_low = np.array(action_norm_stats["q01"])
        mask_t = torch.as_tensor(mask, device=device, dtype=torch.bool)
        high_t = torch.as_tensor(action_high, device=device, dtype=dtype)
        low_t = torch.as_tensor(action_low, device=device, dtype=dtype)
        return mask_t, low_t, high_t

    def token_ids_to_bin_indices(self, token_ids):
        if torch.is_tensor(token_ids):
            indices = self.vocab_size - token_ids - 1
            return torch.clamp(indices, min=0, max=self.num_bins - 1)
        indices = self.vocab_size - np.array(token_ids) - 1
        return np.clip(indices, a_min=0, a_max=self.num_bins - 1)

    def bin_indices_to_token_ids(self, bin_indices):
        if torch.is_tensor(bin_indices):
            token_ids = self.vocab_size - bin_indices - 1
            return torch.where(
                token_ids == 31745, token_ids.new_full(token_ids.shape, 31744), token_ids
            )
        token_ids = self.vocab_size - np.array(bin_indices) - 1
        return np.where(token_ids == 31745, 31744, token_ids)

    def soft_token_probs_to_action(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Differentiable decoding: expected action value per dimension.
        Expects probs with shape (..., action_dim, num_bins).
        """
        centers = self.bin_centers_tensor(device=probs.device, dtype=probs.dtype)
        normalized_actions = torch.matmul(probs, centers)
        mask_t, low_t, high_t = self._action_norm_stats_tensors(
            device=probs.device, dtype=probs.dtype
        )
        return torch.where(
            mask_t,
            0.5 * (normalized_actions + 1) * (high_t - low_t) + low_t,
            normalized_actions,
        )