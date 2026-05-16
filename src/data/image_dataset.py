from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - useful in non-training environments
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


class HaloMassImageDataset(Dataset):
    """Lazy PyTorch Dataset for image-to-halo-mass regression.

    The dataset reads only the CSV at initialization. Images are opened in
    ``__getitem__`` so large image collections do not need to fit in memory.
    """

    def __init__(
        self,
        csv_path: str | Path,
        split: str | None = None,
        root_dir: str | Path = ".",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        target_column: str = "halo_mass_log10",
        image_mode: str = "RGB",
        return_metadata: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.target_column = target_column
        self.image_mode = image_mode
        self.return_metadata = return_metadata

        df = pd.read_csv(self.csv_path)
        if split is not None:
            df = df[df["split"].astype(str) == split].copy()

        required = {"image_path", "subhalo_id", target_column}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, image_path: str | Path) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        return self.root_dir / path

    def _default_to_tensor(self, image: Image.Image):
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = array[None, :, :]
        else:
            array = np.transpose(array, (2, 0, 1))

        if torch is None:
            return array
        return torch.from_numpy(array)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = self._resolve_image_path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert(self.image_mode)
        image = self.transform(image) if self.transform else self._default_to_tensor(image)

        target_value = float(row[self.target_column])
        if self.target_transform:
            target = self.target_transform(target_value)
        elif torch is not None:
            target = torch.tensor(target_value, dtype=torch.float32)
        else:
            target = np.float32(target_value)

        if not self.return_metadata:
            return image, target

        metadata = {
            "image_path": str(row["image_path"]),
            "subhalo_id": row["subhalo_id"],
            "halo_mass_raw": float(row["halo_mass_raw"]) if "halo_mass_raw" in row else None,
            "split": row["split"] if "split" in row else None,
        }
        return image, target, metadata
