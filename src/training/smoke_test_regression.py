from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.image_dataset import HaloMassImageDataset
from src.models.halo_mass_regressor import build_halo_mass_regressor, predict_scalar
from src.training.evaluate_regression import regression_metrics, set_random_seed


def create_synthetic_regression_data(output_dir: str | Path, seed: int = 42, samples: int = 12) -> Path:
    """Create a tiny image regression dataset for smoke testing."""

    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    splits = ["train"] * max(6, samples - 4) + ["val"] * 2 + ["test"] * 2
    splits = splits[:samples]
    while len(splits) < samples:
        splits.append("train")
    random.Random(seed).shuffle(splits)

    for index in range(samples):
        image = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
        image_path = image_dir / f"sample_{index:03d}.png"
        Image.fromarray(image).save(image_path)

        halo_mass_log10 = 11.0 + 0.05 * index
        rows.append(
            {
                "image_path": str(image_path),
                "subhalo_id": index,
                "halo_mass_raw": 10 ** halo_mass_log10,
                "halo_mass_log10": halo_mass_log10,
                "split": splits[index],
            }
        )

    csv_path = output_dir / "regression_metadata.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def run_smoke_test(output_dir: str | Path = "outputs/smoke_test", seed: int = 42) -> dict[str, float]:
    """Run one tiny forward/backward/evaluation pass through the regression stack."""

    set_random_seed(seed)
    output_dir = Path(output_dir)
    csv_path = create_synthetic_regression_data(output_dir, seed=seed)

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = HaloMassImageDataset(csv_path=csv_path, split="train", root_dir=".", transform=transform)
    test_dataset = HaloMassImageDataset(csv_path=csv_path, split="test", root_dir=".", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_halo_mass_regressor({"architecture": "resnet18", "pretrained": False, "dropout": 0.0}).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    images, targets = next(iter(train_loader))
    images = images.to(device)
    targets = targets.to(device).float()
    optimizer.zero_grad(set_to_none=True)
    predictions = predict_scalar(model, images)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, targets in test_loader:
            predictions = predict_scalar(model, images.to(device))
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(predictions.detach().cpu().numpy().tolist())

    metrics = regression_metrics(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    checkpoint_path = output_dir / "smoke_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "seed": seed, "metrics": metrics}, checkpoint_path)

    print("Smoke test completed")
    print(f"Device: {device}")
    print(f"Metadata CSV: {csv_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"MAE: {metrics['mae']:.6f} | RMSE: {metrics['rmse']:.6f} | R2: {metrics['r2']:.6f}")
    return metrics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny synthetic smoke test for the regression stack.")
    parser.add_argument("--output-dir", default="outputs/smoke_test")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_smoke_test(output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
