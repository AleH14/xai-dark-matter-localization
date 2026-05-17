from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


def create_project_directories(root: str | Path = ".") -> list[Path]:
    """Create the directory skeleton used by the project."""

    root = Path(root)
    data_root = Path(os.getenv("DATA_ROOT", "data"))
    dataset_name = os.getenv("DATASET_NAME", "TNG-DM-XAI-v1")
    output_root = Path(os.getenv("OUTPUT_ROOT", "outputs"))
    results_root = Path(os.getenv("RESULTS_ROOT", "results"))

    directories = (
        data_root / "raw" / "tng" / "metadata_raw",
        data_root / "raw" / "tng" / "cutouts",
        data_root / "processed" / dataset_name,
        output_root,
        results_root / "xai",
        results_root / "analysis",
    )

    created = []
    for directory in directories:
        path = directory if directory.is_absolute() else root / directory
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create local project directories and load .env defaults.")
    parser.add_argument("--root", default=".", help="Project root directory.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    load_dotenv()
    directories = create_project_directories(args.root)
    print("Project directories are ready:")
    for path in directories:
        print(f"  {path}")


if __name__ == "__main__":
    main()
