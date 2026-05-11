import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

DATASET_DIR = Path("data/processed/TNG-DM-XAI-v1")


def preprocess_image(input_path, output_224, output_512):
    img = Image.open(input_path).convert("RGB")
    img = np.array(img)

    # Eliminar valores raros
    img = np.nan_to_num(img)

    # Resize para entrenamiento
    img_224 = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    # Resize para XAI/figuras
    img_512 = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)

    output_224.parent.mkdir(parents=True, exist_ok=True)
    output_512.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(img_224.astype(np.uint8)).save(output_224)
    Image.fromarray(img_512.astype(np.uint8)).save(output_512)


def preprocess_dataset(metadata_path):
    df = pd.read_csv(metadata_path)
    processed_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = row["sample_id"]
        split = row.get("split", "unsplit")

        raw_path = Path(row["raw_image_path"])

        output_224 = DATASET_DIR / "images_224" / split / f"{sample_id}.png"
        output_512 = DATASET_DIR / "images_512" / split / f"{sample_id}.png"

        try:
            preprocess_image(raw_path, output_224, output_512)

            row = row.to_dict()
            row["image_path_224"] = str(output_224)
            row["image_path_512"] = str(output_512)
            processed_rows.append(row)

        except Exception as e:
            print(f"Fallo procesando {sample_id}: {e}")

    out_df = pd.DataFrame(processed_rows)
    out_df.to_csv(DATASET_DIR / "metadata_processed.csv", index=False)


if __name__ == "__main__":
    preprocess_dataset(DATASET_DIR / "metadata_with_splits.csv")