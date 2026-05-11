import pandas as pd
from pathlib import Path
from tqdm import tqdm
from tng_api import download_file

DATASET_DIR = Path("data/processed/TNG-DM-XAI-v1")
RAW_IMAGE_DIR = Path("data/raw/tng/images_mock")


def download_mock_images(metadata_path, mock_type="image.png"):
    df = pd.read_csv(metadata_path)
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        subhalo_id = int(row["subhalo_id"])
        subhalo_url = row["subhalo_url"]

        sample_id = f"TNG100_99_{subhalo_id}_p0"

        image_url = subhalo_url + f"stellar_mocks/{mock_type}"
        output_path = RAW_IMAGE_DIR / f"{sample_id}.png"

        try:
            download_file(image_url, output_path)
            row = row.to_dict()
            row["sample_id"] = sample_id
            row["projection_id"] = 0
            row["raw_image_path"] = str(output_path)
            rows.append(row)

        except Exception as e:
            print(f"Fallo subhalo {subhalo_id}: {e}")

    out_df = pd.DataFrame(rows)
    out_path = DATASET_DIR / "metadata_with_raw_images.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Guardado: {out_path}")
    print(f"Imágenes descargadas: {len(out_df)}")


if __name__ == "__main__":
    download_mock_images(
        metadata_path="data/processed/TNG-DM-XAI-v1/metadata_candidates.csv",
        mock_type="image.png"
    )