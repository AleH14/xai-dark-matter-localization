# Ejecución en Google Colab

## Preparación Inicial (Una sola vez)

### 1. **Clona el repositorio**

En la primera celda de tu notebook de Colab:

```bash
cd /content/drive/MyDrive
git clone https://github.com/tu-usuario/xai-dark-matter-localization.git
cd xai-dark-matter-localization
```

### 2. **Instala dependencias**

```bash
pip install -r requirements.txt
```

Esto instalará:
- `python-dotenv` (para variables de entorno)
- `requests` (para API)
- `pandas`, `numpy` (procesamiento de datos)
- `pillow` (procesamiento de imágenes)
- `scikit-image` (visión por computadora)
- etc.

---

## Setup Automático en Cada Notebook

### Opción 1: Setup Automático (Recomendado) ✅

El proyecto está configurado para hacer setup automático en Colab. **Solo importa config.py:**

```python
from src.config import DATA_ROOT, TNG_API_KEY, DATASET_DIR

print(f"✓ Colab setup completado automáticamente")
print(f"Working directory: {DATA_ROOT}")
print(f"Dataset directory: {DATASET_DIR}")
```

Esto automáticamente:
- ✓ Monta Google Drive
- ✓ Cambia al directorio del proyecto
- ✓ Configura las rutas correctas

### Opción 2: Setup Explícito

Si prefieres control explícito, usa:

```python
from src.colab_setup import setup_colab

setup_colab(verbose=True)
```

Esto te mostrará toda la información del setup.

---

## Pipeline de Descargar Dataset

### **Paso 1: Test de Conectividad (01_api_test.ipynb)**

Verifica que el API de TNG funcione correctamente:

```python
import requests
from src.config import BASE_URL, TNG_API_KEY

# Test básico
response = requests.get(BASE_URL, headers={"api-key": TNG_API_KEY})
print(response.status_code)  # Debe ser 200
```

### **Paso 2: Seleccionar Subhalos (02_select_subhalos.ipynb)**

Selecciona los subhalos de materia oscura a procesar:

```python
# Carga y filtra subhalos del catálogo TNG
# Define criterios de selección (masa, redshift, etc.)
# Guarda lista en CSV
```

**Tiempo estimado:** 5-10 minutos  
**Salida:** `data/raw/tng/metadata_raw/subhalos.csv`

### **Paso 3: Descargar Imágenes (03_download_images.ipynb)**

Descarga las imágenes de cutouts del API de TNG:

```python
# Para cada subhalo en la lista
# Descarga imágenes en 2 resoluciones: 224x224 y 512x512
# Guarda en data/raw/tng/cutouts/
```

**Tiempo estimado:** 30-60 minutos (depende del número de subhalos)  
**Nota:** Las descargas de Colab pueden pausarse. Usa `tqdm` para tracking.

### **Paso 4: Preprocesar Imágenes (04_preprocess_images.ipynb)**

Normaliza y redimensiona las imágenes:

```python
# Normalización de píxeles
# Conversión a escala de grises/RGB según sea necesario
# Redimensionamiento a 224x224 y 512x512
# Split en train/val/test
```

**Tiempo estimado:** 10-20 minutos  
**Salida:** `data/processed/TNG-DM-XAI-v1/images_224/` y `images_512/`

### **Paso 5: Crear Masks (05_build_masks.ipynb)**

Genera máscaras de segmentación para materia oscura:

```python
# Identifica píxeles de materia oscura
# Crea máscaras binarias
# Asigna a splits correspondientes
```

**Tiempo estimado:** 15-30 minutos  
**Salida:** `data/processed/TNG-DM-XAI-v1/masks/`

### **Paso 6: Estadísticas (06_dataset_statistics.ipynb)**

Analiza el dataset completado:

```python
# Visualiza distribuciones
# Calcula estadísticas por split
# Verifica integridad de datos
```

---

## Recomendaciones para Colab

### ⚠️ **Timeout y Desconexiones**

Los notebooks pueden tardar mucho. Para evitar desconexiones:

```python
# Instala extensión para mantener Colab activo
!pip install -q PyDrive

# O usa este script en una celda
import time
from IPython.display import clear_output

def keep_alive():
    while True:
        clear_output(wait=True)
        print("Colab activo...")
        time.sleep(30)

# Ejecuta en otra pestaña del navegador
# keep_alive()
```

### 📦 **Guardar Progreso en Drive**

Después de cada paso importante:

```python
import shutil
shutil.copytree('/content/drive/MyDrive/xai-dark-matter-data/processed', 
                 '/content/drive/MyDrive/backup_processed')
```

### 🚀 **Acelerar Descargas**

Divide el trabajo en múltiples ejecuciones:

```python
# Descarga 100 subhalos hoy
# Descarga otros 100 mañana
# Combina al final
```

---

## Estructura Final en Drive

Después de completar todo:

```
MyDrive/
└── xai-dark-matter-data/
    ├── raw/
    │   └── tng/
    │       ├── metadata_raw/
    │       ├── cutouts/          (imágenes descargadas)
    │       └── images_mock/
    └── processed/
        └── TNG-DM-XAI-v1/
            ├── metadata.csv
            ├── train.csv
            ├── val.csv
            ├── test.csv
            ├── images_224/
            │   ├── train/
            │   ├── val/
            │   └── test/
            ├── images_512/
            │   ├── train/
            │   ├── val/
            │   └── test/
            └── masks/
                ├── train/
                ├── val/
                └── test/
```

---

## Troubleshooting

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: No module named 'src'` | Ejecuta: `sys.path.append('/content/drive/MyDrive/xai-dark-matter-localization')` |
| `FileNotFoundError: data/` | Verifica que `DATA_ROOT` en `.env` apunte a la ruta correcta en Drive |
| `API connection timeout` | Reinicia el kernel y reintenta |
| `Drive desconectado` | Remonta: `drive.mount('/content/drive', force_remount=True)` |
| Descarga lenta | Usa GPU de Colab (aunque no la usa tu código, Colab prioriza conexión con GPU) |

---

## Tiempo Total Estimado

- **Configuración inicial:** 5 minutos
- **Pipeline completo:** 1.5 - 3 horas (depende de cantidad de subhalos)

**Recomendación:** Ejecuta en horarios donde no uses la máquina, deja correr un notebook a la vez.
