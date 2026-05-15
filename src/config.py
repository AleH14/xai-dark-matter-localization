import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Auto-setup Colab environment if running in Colab
try:
    from google.colab import drive
    _is_colab = True
except ImportError:
    _is_colab = False

if _is_colab:
    # Mount Google Drive
    drive.mount('/content/drive', force_remount=False)
    # Change to project directory
    _project_path = "/content/drive/MyDrive/xai-dark-matter-localization"
    os.chdir(_project_path)
    # Add to path
    if _project_path not in sys.path:
        sys.path.insert(0, _project_path)

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