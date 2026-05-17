from __future__ import annotations

from collections.abc import Callable

import torch

from src.explainability.gradcam import normalize_map


def integrated_gradients(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    baseline: torch.Tensor | None = None,
    steps: int = 50,
    output_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute Integrated Gradients for a scalar regression prediction.

    Args:
        model: Regression model.
        image_tensor: Single-image batch with shape ``(1, C, H, W)``.
        baseline: Optional baseline tensor with the same shape as image_tensor.
        steps: Number of interpolation steps between baseline and image.
        output_fn: Optional function that converts model output to shape ``(batch,)``.

    Returns:
        Tensor with shape ``(C, H, W)`` containing signed attributions.
    """

    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError("Integrated Gradients expects a single-image batch with shape (1, C, H, W).")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    output_fn = output_fn or (lambda output: output.squeeze(-1))
    baseline = torch.zeros_like(image_tensor) if baseline is None else baseline.to(image_tensor.device)

    gradients = []
    for alpha in torch.linspace(0.0, 1.0, steps, device=image_tensor.device):
        interpolated = baseline + alpha * (image_tensor - baseline)
        interpolated.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        score = output_fn(model(interpolated)).reshape(-1)[0]
        gradient = torch.autograd.grad(score, interpolated, retain_graph=False)[0]
        gradients.append(gradient.detach())

    avg_gradients = torch.stack(gradients, dim=0).mean(dim=0)
    attributions = (image_tensor - baseline) * avg_gradients
    return attributions.squeeze(0).detach()


def channel_reduced_ig_map(attributions: torch.Tensor) -> torch.Tensor:
    """Convert signed channel attributions into a normalized 2D importance map."""

    if attributions.ndim != 3:
        raise ValueError("Expected attributions with shape (C, H, W).")
    return normalize_map(attributions.abs().sum(dim=0))
