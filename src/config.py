import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TNG_API_KEY = os.getenv("TNG_API_KEY")

DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))

RAW_TNG_DIR = DATA_ROOT / "raw" / "tng"
PROCESSED_DIR = DATA_ROOT / "processed"
DATASET_NAME = os.getenv("DATASET_NAME", "TNG-DM-XAI-v1")
DATASET_DIR = PROCESSED_DIR / DATASET_NAME

SNAPSHOT = int(os.getenv("TNG_SNAPSHOT", 99))
SIMULATION = os.getenv("TNG_SIMULATION", "TNG100-1")

BASE_URL = f"https://www.tng-project.org/api/{SIMULATION}"