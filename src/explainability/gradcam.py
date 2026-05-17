from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


def get_resnet18_last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the final convolutional layer used by torchvision ResNet18."""

    if not hasattr(model, "layer4"):
        raise ValueError("Expected a ResNet-like model with a 'layer4' attribute.")
    return model.layer4[-1].conv2


class GradCAM:
    """Grad-CAM for scalar regression outputs.

    The attribution is computed with respect to the model's scalar prediction,
    not a class logit. The returned heatmap highlights image regions that
    influenced the predicted halo-related property.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        output_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.target_layer = target_layer or get_resnet18_last_conv_layer(model)
        self.output_fn = output_fn or (lambda output: output.squeeze(-1))
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Generate a normalized Grad-CAM heatmap for one image tensor.

        Args:
            image_tensor: Tensor with shape ``(1, C, H, W)``.

        Returns:
            Tensor with shape ``(H, W)`` in the range ``[0, 1]``.
        """

        if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
            raise ValueError("GradCAM expects a single-image batch with shape (1, C, H, W).")

        self.model.zero_grad(set_to_none=True)
        output = self.model(image_tensor)
        score = self.output_fn(output).reshape(-1)[0]
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)
        return normalize_map(cam)


def normalize_map(attribution: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    attribution = attribution.detach()
    attribution = attribution - attribution.min()
    denom = attribution.max() + eps
    return attribution / denom
