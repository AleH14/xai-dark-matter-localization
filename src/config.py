import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TNG_API_KEY = os.getenv("TNG_API_KEY")

# Data storage configuration
USE_GOOGLE_DRIVE = os.getenv("USE_GOOGLE_DRIVE", "False").lower() == "true"
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")

# Create directories if they don't exist
DATA_ROOT.mkdir(parents=True, exist_ok=True)

RAW_TNG_DIR = DATA_ROOT / "raw" / "tng"
PROCESSED_DIR = DATA_ROOT / "processed"
DATASET_NAME = os.getenv("DATASET_NAME", "TNG-DM-XAI-v1")
DATASET_DIR = PROCESSED_DIR / DATASET_NAME

# Create subdirectories
RAW_TNG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT = int(os.getenv("TNG_SNAPSHOT", 99))
SIMULATION = os.getenv("TNG_SIMULATION", "TNG100-1")

BASE_URL = f"https://www.tng-project.org/api/{SIMULATION}"