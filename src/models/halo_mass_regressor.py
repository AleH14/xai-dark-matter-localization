from __future__ import annotations

import torch
from torch import nn


def build_resnet18_regressor(
    pretrained: bool = False,
    dropout: float = 0.0,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a ResNet18 model with a scalar regression head."""

    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:  # pragma: no cover - depends on training env
        raise ImportError("torchvision is required to build the ResNet18 regressor.") from exc

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
    else:
        model.fc = nn.Linear(in_features, 1)

    return model


def build_halo_mass_regressor(config: dict | None = None) -> nn.Module:
    """Factory for halo-mass regression models."""

    config = config or {}
    architecture = config.get("architecture", "resnet18").lower()

    if architecture != "resnet18":
        raise ValueError(f"Unsupported architecture '{architecture}'. Currently supported: resnet18.")

    return build_resnet18_regressor(
        pretrained=bool(config.get("pretrained", False)),
        dropout=float(config.get("dropout", 0.0)),
        freeze_backbone=bool(config.get("freeze_backbone", False)),
    )


def predict_scalar(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Run model and return shape ``(batch,)`` predictions."""

    return model(images).squeeze(-1)
