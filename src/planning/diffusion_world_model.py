"""Diffusion world model wrapper for planning."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from diffusers import AutoencoderKL

from diffusion.df_sample import dfot_sample
from diffusion.gaussian_diffusion import GaussianDiffusion
from utils.vae_ops import encode_first_stage


class DiffusionWorldModel(nn.Module):
    """Wrapper to make diffusion model compatible with planning interface."""

    def __init__(
        self,
        model: nn.Module,
        vae: AutoencoderKL,
        diffusion: GaussianDiffusion,
        args,
    ):
        """
        Initialize diffusion world model.

        Args:
            model: NanoWM transformer model
            vae: VAE encoder/decoder
            diffusion: Gaussian diffusion process
            args: Training/sampling config
        """
        super().__init__()
        self.model = model
        self.vae = vae
        self.diffusion = diffusion
        self.args = args
        self.device = next(model.parameters()).device
        self.vae_scale_factor = vae.config.scaling_factor
        self.vae_precision = getattr(args.experiment.infra, "vae_precision", "fp32")

    @torch.no_grad()
    def encode_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode observations to latent embeddings.

        Args:
            obs: Observation dict
                - 'visual': [B, T, C, H, W] visual observations
                - 'proprio': [B, T, D] proprioceptive states (optional)

        Returns:
            Embeddings dict
                - 'visual': [B, T, D_visual] visual feature embeddings
                - 'proprio': [B, T, D_proprio] proprioceptive embeddings (or None)
        """
        frames = obs["visual"]  # [B, T, C, H, W]
        B, T, C, H, W = frames.shape

        frames_flat = frames.reshape(B * T, C, H, W)
        latents = encode_first_stage(self.vae, frames_flat, precision=self.vae_precision)

        # Get feature embeddings from NanoWM encoder
        # Use the model's patch embedding and positional encoding
        latents = latents.reshape(B, T, *latents.shape[1:])  # [B, T, C_lat, H_lat, W_lat]

        # For planning, we use the raw latents as visual features
        # Flatten spatial dimensions to get feature vectors
        C_lat, H_lat, W_lat = latents.shape[2:]
        z_visual = latents.reshape(B, T, C_lat * H_lat * W_lat)  # [B, T, D_visual]

        # Proprioceptive embeddings (pass through if available)
        z_proprio = obs.get("proprio", None)

        return {"visual": z_visual, "proprio": z_proprio}

    @torch.no_grad()
    def rollout(
        self,
        obs_0: Dict[str, torch.Tensor],
        act: torch.Tensor,
        num_sampling_steps: Optional[int] = None,
        eta: float = 0.0,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Autoregressive rollout: generates chunks of (num_frames - n_context)
        new frames at a time, feeding the last generated frame as context
        for the next chunk, until the full horizon is covered.
        """
        context_frames = obs_0["visual"]  # [B, 1, C, H, W]
        B, _, C, H, W = context_frames.shape
        horizon = act.shape[1]

        if num_sampling_steps is None:
            num_sampling_steps = self.args.model.num_sampling_steps

        num_frames = self.args.model.num_frames
        n_context = self.args.model.n_context_frames
        gen_per_chunk = num_frames - n_context
        scheduling_mode = self.args.model.scheduling_mode

        # Encode initial context
        ctx_flat = context_frames.reshape(B, C, H, W)
        ctx_latents = encode_first_stage(self.vae, ctx_flat, precision=self.vae_precision)
        # ctx_latents: [B, C_lat, H_lat, W_lat]

        all_latents = [ctx_latents.unsqueeze(1)]  # list of [B, 1, C_lat, H_lat, W_lat]
        act_offset = 0

        while act_offset < horizon:
            chunk_len = min(gen_per_chunk, horizon - act_offset)

            # Always generate a full chunk (num_frames) to match temp_embed size,
            # but only keep chunk_len new frames.
            act_chunk = act[:, act_offset : act_offset + num_frames]
            if act_chunk.shape[1] < num_frames:
                pad = torch.zeros(B, num_frames - act_chunk.shape[1], act.shape[2], device=act.device)
                act_chunk = torch.cat([act_chunk, pad], dim=1)

            # Context latent for this chunk
            cur_ctx = all_latents[-1][:, -1:, :, :, :]  # [B, 1, C_lat, H_lat, W_lat]

            shape = [B, num_frames, *cur_ctx.shape[2:]]

            chunk_latents = dfot_sample(
                diffusion=self.diffusion,
                model=self.model,
                shape=shape,
                context=cur_ctx,
                n_context_frames=n_context,
                model_kwargs={"action": act_chunk},
                scheduling_mode=scheduling_mode,
                num_sampling_steps=num_sampling_steps,
                eta=eta,
                history_stabilization_level=self.args.experiment.diffusion.history_stabilization_level,
            )  # [B, total_frames, C_lat, H_lat, W_lat]

            # Keep only the newly generated frames (skip context)
            new_latents = chunk_latents[:, n_context : n_context + chunk_len]
            all_latents.append(new_latents)
            act_offset += chunk_len

        # Concatenate: [context_frame] + [all generated chunks]
        generated_latents = torch.cat(all_latents, dim=1)  # [B, 1+horizon, ...]

        B_out, T_total, C_lat, H_lat, W_lat = generated_latents.shape
        z_visual = generated_latents.reshape(B_out, T_total, C_lat * H_lat * W_lat)
        z_obses = {"visual": z_visual, "proprio": None}

        return z_obses, None

    def forward(self, *args, **kwargs):
        """Forward pass delegates to rollout."""
        return self.rollout(*args, **kwargs)
