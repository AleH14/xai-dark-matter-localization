import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATASET_DIR = Path("data/processed/TNG-DM-XAI-v1")


def create_splits(metadata_path):
    df = pd.read_csv(metadata_path)

    unique_subhalos = df["subhalo_id"].unique()

    train_ids, temp_ids = train_test_split(
        unique_subhalos,
        test_size=0.30,
        random_state=42
    )

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=42
    )

    def assign_split(subhalo_id):
        if subhalo_id in train_ids:
            return "train"
        elif subhalo_id in val_ids:
            return "val"
        else:
            return "test"

    df["split"] = df["subhalo_id"].apply(assign_split)

    df.to_csv(DATASET_DIR / "metadata_with_splits.csv", index=False)

    df[df["split"] == "train"].to_csv(DATASET_DIR / "train.csv", index=False)
    df[df["split"] == "val"].to_csv(DATASET_DIR / "val.csv", index=False)
    df[df["split"] == "test"].to_csv(DATASET_DIR / "test.csv", index=False)

    print(df["split"].value_counts())


if __name__ == "__main__":
    create_splits(DATASET_DIR / "metadata_with_raw_images.csv")