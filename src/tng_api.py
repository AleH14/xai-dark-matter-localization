import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TNG_API_KEY")

if API_KEY is None:
    raise RuntimeError("No se encontró TNG_API_KEY en el archivo .env")

HEADERS = {
    "api-key": API_KEY
}


def get_json(url, params=None, retries=3, sleep=1):
    """
    Hace una petición GET a la API y devuelve JSON.
    """
    for attempt in range(retries):
        response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code == 200:
            return response.json()

        print(f"Error {response.status_code}: {response.text}")
        time.sleep(sleep)

    raise RuntimeError(f"No se pudo obtener JSON desde {url}")


def download_file(url, output_path, params=None, retries=3, sleep=1):
    """
    Descarga archivo desde la API.
    Sirve para PNG, HDF5, FITS, etc.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return output_path

    for attempt in range(retries):
        response = requests.get(url, headers=HEADERS, params=params, stream=True)

        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return output_path

        print(f"Error {response.status_code}: {response.text}")
        time.sleep(sleep)

    raise RuntimeError(f"No se pudo descargar archivo desde {url}")