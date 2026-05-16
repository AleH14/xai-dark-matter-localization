from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path("data/processed/TNG-DM-XAI-v1/metadata_with_images.csv")
DEFAULT_OUTPUT_CSV = Path("data/processed/TNG-DM-XAI-v1/regression_metadata.csv")
DEFAULT_LABEL_CANDIDATES = (
    "group_m_crit200",
    "Group_M_Crit200",
    "halo_mass",
    "halo_mass_raw",
    "mass",
)


def _resolve_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def _choose_label_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"Requested label column '{requested}' was not found. "
                f"Available columns: {', '.join(df.columns)}"
            )
        return requested

    for column in DEFAULT_LABEL_CANDIDATES:
        if column in df.columns:
            return column

    raise ValueError(
        "No halo-mass label column found. Expected one of: "
        + ", ".join(DEFAULT_LABEL_CANDIDATES)
    )


def _ensure_image_column(df: pd.DataFrame) -> pd.DataFrame:
    if "image_path" in df.columns:
        return df

    for candidate in ("image_path_224", "image_path_512", "raw_image_path"):
        if candidate in df.columns:
            out = df.copy()
            out["image_path"] = out[candidate]
            return out

    raise ValueError(
        "No image path column found. Expected 'image_path', 'image_path_224', "
        "'image_path_512', or 'raw_image_path'."
    )


def _filter_resolution(df: pd.DataFrame, resolution: int | None) -> pd.DataFrame:
    if resolution is None or "resolution" not in df.columns:
        return df
    return df[df["resolution"].astype(str) == str(resolution)].copy()


def _assign_splits(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> pd.Series:
    if "split" in df.columns and df["split"].notna().all():
        return df["split"].astype(str)

    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total:.4f} "
            f"({train_size}, {val_size}, {test_size})."
        )

    unique_ids = pd.Series(df["subhalo_id"].unique())
    if len(unique_ids) < 3:
        raise ValueError("Need at least three unique subhalos to create train/val/test splits.")

    rng = np.random.default_rng(random_state)
    shuffled_ids = unique_ids.to_numpy(copy=True)
    rng.shuffle(shuffled_ids)

    n_total = len(shuffled_ids)
    n_train = max(1, int(round(n_total * train_size)))
    n_val = max(1, int(round(n_total * val_size)))
    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = 1

    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val:]

    train_ids = set(train_ids)
    val_ids = set(val_ids)
    test_ids = set(test_ids)

    def split_for(subhalo_id: object) -> str:
        if subhalo_id in train_ids:
            return "train"
        if subhalo_id in val_ids:
            return "val"
        if subhalo_id in test_ids:
            return "test"
        raise RuntimeError(f"Subhalo id {subhalo_id!r} was not assigned to a split.")

    return df["subhalo_id"].map(split_for)


def _print_stats(df: pd.DataFrame, dropped_missing_images: int, dropped_missing_labels: int) -> None:
    target = df["halo_mass_log10"]
    print("\nRegression dataset built")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print("\nSplit counts:")
    print(df["split"].value_counts().reindex(["train", "val", "test"], fill_value=0))
    print("\nTarget halo_mass_log10:")
    print(f"  min:  {target.min():.6f}")
    print(f"  max:  {target.max():.6f}")
    print(f"  mean: {target.mean():.6f}")
    print(f"  std:  {target.std(ddof=0):.6f}")
    print("\nDropped rows:")
    print(f"  missing images: {dropped_missing_images}")
    print(f"  missing/invalid labels: {dropped_missing_labels}")


def build_regression_metadata(
    input_csv: str | Path = DEFAULT_INPUT_CSV,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    label_column: str | None = None,
    resolution: int | None = 224,
    project_root: str | Path = ".",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a clean image-to-halo-mass regression metadata CSV.

    The output CSV contains only image paths with valid physical labels. Images
    are not loaded into memory; paths are checked on disk and labels are
    converted to log10 values for regression.
    """

    project_root = Path(project_root).resolve()
    input_csv = _resolve_path(input_csv, project_root)
    output_csv = _resolve_path(output_csv, project_root)

    df = pd.read_csv(input_csv)
    df = _ensure_image_column(df)
    df = _filter_resolution(df, resolution)

    required_columns = {"image_path", "subhalo_id"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    label_column = _choose_label_column(df, label_column)

    work = df.copy()
    work["halo_mass_raw"] = pd.to_numeric(work[label_column], errors="coerce")
    valid_label = work["halo_mass_raw"].notna() & np.isfinite(work["halo_mass_raw"]) & (work["halo_mass_raw"] > 0)
    dropped_missing_labels = int((~valid_label).sum())
    work = work[valid_label].copy()

    image_exists = work["image_path"].apply(lambda value: _resolve_path(value, project_root).exists())
    dropped_missing_images = int((~image_exists).sum())
    work = work[image_exists].copy()

    if work.empty:
        raise ValueError("No valid samples remain after filtering missing images and labels.")

    work["halo_mass_log10"] = np.log10(work["halo_mass_raw"].astype(float))
    work["split"] = _assign_splits(work, train_size, val_size, test_size, random_state)

    out = work[["image_path", "subhalo_id", "halo_mass_raw", "halo_mass_log10", "split"]].copy()
    out = out.sort_values(["split", "subhalo_id", "image_path"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    _print_stats(out, dropped_missing_images, dropped_missing_labels)
    print(f"\nSaved: {output_csv}")
    return out


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build image-to-halo-mass regression metadata.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    build_regression_metadata(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        label_column=args.label_column,
        resolution=args.resolution,
        project_root=args.project_root,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
