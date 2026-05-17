from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.image_dataset import HaloMassImageDataset
from src.models.halo_mass_regressor import build_halo_mass_regressor, predict_scalar


def load_yaml_config(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on training env
        raise ImportError("PyYAML is required to read regression configs.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(math.sqrt(np.mean(residual ** 2)))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def build_eval_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def make_dataloader(config: dict, split: str) -> DataLoader:
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    image_size = int(data_config.get("image_size", 224))

    dataset = HaloMassImageDataset(
        csv_path=data_config["metadata_csv"],
        split=split,
        root_dir=data_config.get("root_dir", "."),
        transform=build_eval_transform(image_size),
        target_column=data_config.get("target_column", "halo_mass_log10"),
        image_mode=data_config.get("image_mode", "RGB"),
        return_metadata=True,
    )

    return DataLoader(
        dataset,
        batch_size=int(training_config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )


def load_checkpoint_model(config: dict, checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    model = build_halo_mass_regressor(config.get("model", {}))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    y_true = []
    y_pred = []

    for images, targets, metadata in dataloader:
        images = images.to(device)
        targets = targets.to(device).float()
        predictions = predict_scalar(model, images)

        targets_np = targets.detach().cpu().numpy()
        predictions_np = predictions.detach().cpu().numpy()

        y_true.extend(targets_np.tolist())
        y_pred.extend(predictions_np.tolist())

        for i in range(len(targets_np)):
            rows.append(
                {
                    "image_path": metadata["image_path"][i],
                    "subhalo_id": metadata["subhalo_id"][i].item()
                    if hasattr(metadata["subhalo_id"][i], "item")
                    else metadata["subhalo_id"][i],
                    "split": metadata["split"][i],
                    "y_true": float(targets_np[i]),
                    "y_pred": float(predictions_np[i]),
                    "residual": float(predictions_np[i] - targets_np[i]),
                }
            )

    predictions_df = pd.DataFrame(rows)
    metrics = regression_metrics(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    return predictions_df, metrics


def save_scatter_plot(predictions_df: pd.DataFrame, metrics: dict[str, float], output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = predictions_df["y_true"]
    y_pred = predictions_df["y_pred"]
    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=18, alpha=0.75)
    plt.plot([low, high], [low, high], color="black", linewidth=1.5)
    plt.xlabel("True log10 halo mass")
    plt.ylabel("Predicted log10 halo mass")
    plt.title(f"Predicted vs True (MAE={metrics['mae']:.3f}, R2={metrics['r2']:.3f})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def evaluate_from_config(config: dict, checkpoint_path: str | Path | None = None, split: str = "test") -> dict[str, float]:
    seed = int(config.get("seed", 42))
    set_random_seed(seed)

    device = get_device(config.get("device", "auto"))
    output_dir = Path(config.get("output", {}).get("output_dir", "outputs/regression_resnet18"))
    checkpoint_path = Path(checkpoint_path or output_dir / "checkpoints" / "best_model.pt")

    dataloader = make_dataloader(config, split=split)
    model = load_checkpoint_model(config, checkpoint_path, device)
    predictions_df, metrics = evaluate_model(model, dataloader, device)

    predictions_path = output_dir / f"predictions_{split}.csv"
    metrics_path = output_dir / f"metrics_{split}.json"
    scatter_path = output_dir / f"predicted_vs_true_{split}.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_scatter_plot(predictions_df, metrics, scatter_path)

    print(f"Evaluation split: {split}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"R2:   {metrics['r2']:.6f}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved scatter plot: {scatter_path}")
    return metrics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate halo-mass regression checkpoint.")
    parser.add_argument("--config", default="configs/regression_resnet18.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    evaluate_from_config(config, checkpoint_path=args.checkpoint, split=args.split)


if __name__ == "__main__":
    main()
