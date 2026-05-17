from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REGIONS = ("center", "middle", "outer")


def create_radial_region_masks(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Create center, middle, and outer radial analysis masks.

    Regions are defined geometrically:
    - center: r < 0.30
    - middle: 0.30 <= r < 0.70
    - outer: r >= 0.70

    These masks are analysis regions, not physical dark matter labels.
    """

    height, width = shape
    y, x = np.ogrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    radius_norm = radius / radius.max()

    return {
        "center": radius_norm < 0.30,
        "middle": (radius_norm >= 0.30) & (radius_norm < 0.70),
        "outer": radius_norm >= 0.70,
    }


def attribution_to_2d(attribution: np.ndarray) -> np.ndarray:
    """Convert an attribution array into a non-negative 2D importance map."""

    array = np.asarray(attribution)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    if array.ndim == 2:
        importance = np.abs(array)
    elif array.ndim == 3:
        # Support both CHW and HWC by reducing over the likely channel axis.
        if array.shape[0] in (1, 3, 4):
            importance = np.abs(array).sum(axis=0)
        elif array.shape[-1] in (1, 3, 4):
            importance = np.abs(array).sum(axis=-1)
        else:
            importance = np.abs(array).sum(axis=0)
    elif array.ndim == 4 and array.shape[0] == 1:
        return attribution_to_2d(array[0])
    else:
        raise ValueError(f"Unsupported attribution shape: {array.shape}")

    return importance.astype(np.float32)


def normalize_importance_map(importance: np.ndarray) -> np.ndarray:
    """Normalize an importance map to sum to 1, safely handling zero maps."""

    importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
    importance = np.clip(importance, a_min=0.0, a_max=None)
    total = float(importance.sum())
    if total <= 0:
        return np.zeros_like(importance, dtype=np.float32)
    return (importance / total).astype(np.float32)


def compute_radial_importance(attribution: np.ndarray) -> dict[str, float]:
    """Compute attribution percentage in each radial region."""

    importance = attribution_to_2d(attribution)
    normalized = normalize_importance_map(importance)
    masks = create_radial_region_masks(normalized.shape)

    metrics = {}
    for region in REGIONS:
        metrics[f"{region}_importance_percent"] = float(normalized[masks[region]].sum() * 100.0)
    return metrics


def _resolve_path(path_value: str | Path, base_dir: str | Path = ".") -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def _load_summary(summary_csv: str | Path | None) -> pd.DataFrame:
    if summary_csv is None:
        return pd.DataFrame()
    summary_path = Path(summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"XAI summary CSV not found: {summary_path}")
    return pd.read_csv(summary_path)


def _discover_attribution_rows(
    summary_df: pd.DataFrame,
    xai_dir: str | Path,
    method: str,
) -> list[dict[str, object]]:
    xai_dir = Path(xai_dir)
    rows: list[dict[str, object]] = []

    if method == "gradcam" and "gradcam_npy" in summary_df.columns:
        for _, row in summary_df.iterrows():
            rows.append(
                {
                    "method": "gradcam",
                    "attribution_path": row["gradcam_npy"],
                    "dataset_index": row.get("dataset_index"),
                    "subhalo_id": row.get("subhalo_id"),
                    "image_path": row.get("image_path"),
                    "true_halo_mass_log10": row.get("true_halo_mass_log10"),
                    "pred_halo_mass_log10": row.get("pred_halo_mass_log10"),
                }
            )
        return rows

    if method == "integrated_gradients" and "integrated_gradients_raw_npy" in summary_df.columns:
        for _, row in summary_df.iterrows():
            rows.append(
                {
                    "method": "integrated_gradients",
                    "attribution_path": row["integrated_gradients_raw_npy"],
                    "dataset_index": row.get("dataset_index"),
                    "subhalo_id": row.get("subhalo_id"),
                    "image_path": row.get("image_path"),
                    "true_halo_mass_log10": row.get("true_halo_mass_log10"),
                    "pred_halo_mass_log10": row.get("pred_halo_mass_log10"),
                }
            )
        return rows

    patterns = {
        "gradcam": "gradcam_*.npy",
        "integrated_gradients": "integrated_gradients_*_raw.npy",
    }
    for path in sorted(xai_dir.glob(patterns[method])):
        rows.append({"method": method, "attribution_path": str(path)})
    return rows


def build_radial_importance_table(
    xai_dir: str | Path = "results/xai",
    summary_csv: str | Path | None = "results/xai/xai_summary.csv",
    output_csv: str | Path = "results/analysis/radial_importance.csv",
    methods: Iterable[str] = ("gradcam", "integrated_gradients"),
    base_dir: str | Path = ".",
) -> pd.DataFrame:
    """Compute radial importance metrics for saved attribution arrays."""

    xai_dir = Path(xai_dir)
    summary_df = _load_summary(summary_csv)
    rows = []

    for method in methods:
        if method not in {"gradcam", "integrated_gradients"}:
            raise ValueError(f"Unsupported method: {method}")

        for item in _discover_attribution_rows(summary_df, xai_dir, method):
            attribution_path = _resolve_path(item["attribution_path"], base_dir=base_dir)
            if not attribution_path.exists():
                print(f"Skipping missing attribution file: {attribution_path}")
                continue

            attribution = np.load(attribution_path)
            metrics = compute_radial_importance(attribution)
            total_percent = sum(metrics[f"{region}_importance_percent"] for region in REGIONS)

            rows.append(
                {
                    **item,
                    **metrics,
                    "total_importance_percent": float(total_percent),
                    "attribution_shape": "x".join(str(v) for v in attribution.shape),
                }
            )

    result = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    print(f"Saved radial importance CSV: {output_csv}")
    if not result.empty:
        print(f"Rows: {len(result)}")
        print(result[[f"{region}_importance_percent" for region in REGIONS]].mean())
    else:
        print("No attribution maps were found.")

    return result


def _long_format(result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in result.iterrows():
        for region in REGIONS:
            rows.append(
                {
                    "method": row["method"],
                    "region": region,
                    "importance_percent": row[f"{region}_importance_percent"],
                    "true_halo_mass_log10": row.get("true_halo_mass_log10"),
                    "pred_halo_mass_log10": row.get("pred_halo_mass_log10"),
                    "subhalo_id": row.get("subhalo_id"),
                }
            )
    return pd.DataFrame(rows)


def save_radial_importance_plots(
    result: pd.DataFrame,
    output_dir: str | Path = "results/analysis",
) -> None:
    """Save mean, boxplot, and halo-mass relation plots."""

    if result.empty:
        print("No plots saved because radial importance table is empty.")
        return

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_df = _long_format(result)
    region_order = list(REGIONS)

    mean_df = (
        long_df.groupby(["method", "region"], as_index=False)["importance_percent"]
        .mean()
        .pivot(index="region", columns="method", values="importance_percent")
        .reindex(region_order)
    )
    ax = mean_df.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Mean attribution importance (%)")
    ax.set_xlabel("Radial region")
    ax.set_title("Mean attribution importance by radial region")
    ax.legend(title="Method")
    plt.tight_layout()
    plt.savefig(output_dir / "mean_importance_by_region.png", dpi=160)
    plt.close()

    methods = list(long_df["method"].dropna().unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5), squeeze=False)
    for ax, method in zip(axes[0], methods):
        data = [
            long_df[(long_df["method"] == method) & (long_df["region"] == region)]["importance_percent"].dropna()
            for region in region_order
        ]
        ax.boxplot(data, labels=region_order)
        ax.set_title(method)
        ax.set_ylabel("Attribution importance (%)")
        ax.set_xlabel("Radial region")
    plt.tight_layout()
    plt.savefig(output_dir / "boxplot_importance_by_region.png", dpi=160)
    plt.close()

    mass_df = long_df.dropna(subset=["true_halo_mass_log10"])
    if not mass_df.empty:
        fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5), squeeze=False)
        for ax, method in zip(axes[0], methods):
            method_df = mass_df[mass_df["method"] == method]
            for region in region_order:
                region_df = method_df[method_df["region"] == region]
                ax.scatter(
                    region_df["true_halo_mass_log10"],
                    region_df["importance_percent"],
                    s=24,
                    alpha=0.75,
                    label=region,
                )
            ax.set_title(method)
            ax.set_xlabel("True log10 halo mass")
            ax.set_ylabel("Attribution importance (%)")
            ax.legend(title="Region")
        plt.tight_layout()
        plt.savefig(output_dir / "importance_vs_true_halo_mass.png", dpi=160)
        plt.close()

    print(f"Saved plots under: {output_dir}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute radial importance metrics from XAI attribution maps.")
    parser.add_argument("--xai-dir", default="results/xai")
    parser.add_argument("--summary-csv", default="results/xai/xai_summary.csv")
    parser.add_argument("--output-csv", default="results/analysis/radial_importance.csv")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["gradcam", "integrated_gradients"],
        choices=["gradcam", "integrated_gradients"],
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_radial_importance_table(
        xai_dir=args.xai_dir,
        summary_csv=args.summary_csv,
        output_csv=args.output_csv,
        methods=args.methods,
    )
    save_radial_importance_plots(result, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
