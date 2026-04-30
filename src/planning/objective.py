"""Objective functions for planning."""

import numpy as np
import torch
import torch.nn as nn


def create_objective_fn(alpha: float = 1.0, base: float = 2.0, mode: str = "last"):
    """
    Create objective function for planning.

    Args:
        alpha: Weight for proprioceptive loss
        base: Base for exponential weighting (only used for mode="all")
        mode: "last" (loss on final frame) or "all" (loss on all frames)

    Returns:
        Objective function that takes (z_obs_pred, z_obs_tgt) and returns loss [B]
    """
    metric = nn.MSELoss(reduction="none")

    def objective_fn_last(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on the last predicted frame.

        Args:
            z_obs_pred: dict
                - 'visual': [B, T, D_visual] predicted visual embeddings
                - 'proprio': [B, T, D_proprio] or None
            z_obs_tgt: dict
                - 'visual': [B, T, D_visual] target visual embeddings
                - 'proprio': [B, T, D_proprio] or None

        Returns:
            loss: [B] loss per batch element
        """
        # Visual loss
        loss_visual = metric(
            z_obs_pred["visual"][:, -1:],
            z_obs_tgt["visual"][:, -1:]
        ).mean(dim=tuple(range(1, z_obs_pred["visual"].ndim)))

        # Proprioceptive loss (if available)
        if z_obs_pred.get("proprio") is not None and z_obs_tgt.get("proprio") is not None:
            loss_proprio = metric(
                z_obs_pred["proprio"][:, -1:],
                z_obs_tgt["proprio"][:, -1:]
            ).mean(dim=tuple(range(1, z_obs_pred["proprio"].ndim)))
            loss = loss_visual + alpha * loss_proprio
        else:
            loss = loss_visual

        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on all predicted frames with exponential weighting.

        Args:
            z_obs_pred: dict
                - 'visual': [B, T, D_visual] predicted visual embeddings
                - 'proprio': [B, T, D_proprio] or None
            z_obs_tgt: dict
                - 'visual': [B, T, D_visual] target visual embeddings
                - 'proprio': [B, T, D_proprio] or None

        Returns:
            loss: [B] loss per batch element
        """
        T = z_obs_pred["visual"].shape[1]

        # Exponential weighting coefficients
        coeffs = np.array([base**i for i in range(T)], dtype=np.float32)
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)

        # Visual loss
        loss_visual = metric(
            z_obs_pred["visual"],
            z_obs_tgt["visual"]
        ).mean(dim=tuple(range(2, z_obs_pred["visual"].ndim)))
        loss_visual = (loss_visual * coeffs).mean(dim=1)

        # Proprioceptive loss (if available)
        if z_obs_pred.get("proprio") is not None and z_obs_tgt.get("proprio") is not None:
            loss_proprio = metric(
                z_obs_pred["proprio"],
                z_obs_tgt["proprio"]
            ).mean(dim=tuple(range(2, z_obs_pred["proprio"].ndim)))
            loss_proprio = (loss_proprio * coeffs).mean(dim=1)
            loss = loss_visual + alpha * loss_proprio
        else:
            loss = loss_visual

        return loss

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    else:
        raise NotImplementedError(f"Unknown mode: {mode}")
