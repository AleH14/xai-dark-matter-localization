import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

DATASET_DIR = Path("data/processed/TNG-DM-XAI-v1")


def create_radial_masks(size=512):
    y, x = np.ogrid[:size, :size]

    cx = size / 2
    cy = size / 2

    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_norm = r / r.max()

    center = (r_norm < 0.30).astype(np.uint8) * 255
    middle = ((r_norm >= 0.30) & (r_norm < 0.70)).astype(np.uint8) * 255
    outer = (r_norm >= 0.70).astype(np.uint8) * 255

    return center, middle, outer


def save_mask(mask, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def build_masks(metadata_path):
    df = pd.read_csv(metadata_path)

    center_mask, middle_mask, outer_mask = create_radial_masks(size=512)

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = row["sample_id"]
        split = row["split"]

        center_path = DATASET_DIR / "masks" / split / f"{sample_id}_center.png"
        middle_path = DATASET_DIR / "masks" / split / f"{sample_id}_middle.png"
        outer_path = DATASET_DIR / "masks" / split / f"{sample_id}_outer.png"

        save_mask(center_mask, center_path)
        save_mask(middle_mask, middle_path)
        save_mask(outer_mask, outer_path)

        row = row.to_dict()
        row["mask_center_path"] = str(center_path)
        row["mask_middle_path"] = str(middle_path)
        row["mask_outer_path"] = str(outer_path)

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(DATASET_DIR / "metadata.csv", index=False)


if __name__ == "__main__":
    build_masks(DATASET_DIR / "metadata_processed.csv")