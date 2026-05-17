from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.image_dataset import HaloMassImageDataset
from src.models.halo_mass_regressor import build_halo_mass_regressor, predict_scalar
from src.training.evaluate_regression import (
    evaluate_from_config,
    get_device,
    regression_metrics,
    set_random_seed,
)


def load_yaml_config(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on training env
        raise ImportError("PyYAML is required to read regression configs.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_transform(config: dict, train: bool):
    from torchvision import transforms

    data_config = config.get("data", {})
    augmentation_config = config.get("augmentation", {})
    image_size = int(data_config.get("image_size", 224))

    steps = [transforms.Resize((image_size, image_size))]
    if train and augmentation_config.get("horizontal_flip", True):
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    if train and augmentation_config.get("vertical_flip", False):
        steps.append(transforms.RandomVerticalFlip(p=0.5))
    if train and float(augmentation_config.get("rotation_degrees", 0)) > 0:
        steps.append(transforms.RandomRotation(float(augmentation_config["rotation_degrees"])))

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)


def make_dataset(config: dict, split: str, train: bool) -> HaloMassImageDataset:
    data_config = config.get("data", {})
    return HaloMassImageDataset(
        csv_path=data_config["metadata_csv"],
        split=split,
        root_dir=data_config.get("root_dir", "."),
        transform=build_transform(config, train=train),
        target_column=data_config.get("target_column", "halo_mass_log10"),
        image_mode=data_config.get("image_mode", "RGB"),
        return_metadata=False,
    )


def make_dataloader(config: dict, split: str, train: bool) -> DataLoader:
    training_config = config.get("training", {})
    dataset = make_dataset(config, split=split, train=train)
    return DataLoader(
        dataset,
        batch_size=int(training_config.get("batch_size", 32)),
        shuffle=train,
        num_workers=int(training_config.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def make_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name in {"smooth_l1", "smoothl1", "huber"}:
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss '{name}'. Use 'mse' or 'smooth_l1'.")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        predictions = predict_scalar(model, images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses = []
    y_true = []
    y_pred = []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device).float()
        predictions = predict_scalar(model, images)
        loss = criterion(predictions, targets)

        losses.append(float(loss.detach().cpu().item()))
        y_true.extend(targets.detach().cpu().numpy().tolist())
        y_pred.extend(predictions.detach().cpu().numpy().tolist())

    metrics = regression_metrics(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    return float(np.mean(losses)) if losses else float("nan"), metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def save_loss_curve(history: pd.DataFrame, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Regression Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def train_from_config(config: dict) -> dict[str, float]:
    set_random_seed(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))

    output_dir = Path(config.get("output", {}).get("output_dir", "outputs/regression_resnet18"))
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_dataloader(config, split="train", train=True)
    val_loader = make_dataloader(config, split="val", train=False)

    model = build_halo_mass_regressor(config.get("model", {})).to(device)
    criterion = make_loss(config.get("training", {}).get("loss", "smooth_l1"))
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=float(config.get("training", {}).get("learning_rate", 1e-4)),
        weight_decay=float(config.get("training", {}).get("weight_decay", 1e-4)),
    )

    epochs = int(config.get("training", {}).get("epochs", 20))
    best_val_loss = float("inf")
    history_rows = []

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
        }
        history_rows.append(history_row)

        print(
            f"Epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_mae={val_metrics['mae']:.6f} val_rmse={val_metrics['rmse']:.6f} "
            f"val_r2={val_metrics['r2']:.6f}"
        )

        save_checkpoint(checkpoints_dir / "last_model.pt", model, optimizer, epoch, best_val_loss, config)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(checkpoints_dir / "best_model.pt", model, optimizer, epoch, best_val_loss, config)

    history = pd.DataFrame(history_rows)
    history_path = output_dir / "training_history.csv"
    history.to_csv(history_path, index=False)
    save_loss_curve(history, output_dir / "loss_curve.png")
    (output_dir / "config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Saved best checkpoint: {checkpoints_dir / 'best_model.pt'}")
    print(f"Saved history: {history_path}")
    print(f"Saved loss curve: {output_dir / 'loss_curve.png'}")

    test_metrics = evaluate_from_config(config, checkpoint_path=checkpoints_dir / "best_model.pt", split="test")
    return test_metrics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 halo-mass regressor.")
    parser.add_argument("--config", default="configs/regression_resnet18.yaml")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    train_from_config(config)


if __name__ == "__main__":
    main()
