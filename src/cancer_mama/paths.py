"""Rutas compartidas del proyecto, independientes del directorio de ejecución."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "breastdcedl"
RESULTS_DIR = PROJECT_ROOT / "resultados"
DOCS_DIR = PROJECT_ROOT / "docs"
RUNTIME_DIR = PROJECT_ROOT / ".codex_tmp"
MULTIMODAL_MANIFEST = PROJECT_ROOT / "modelos/versiones/v002_multimodal_clinico_20261007/manifiesto.json"
MULTIMODAL_RUNS = RESULTS_DIR / "04_entrenamiento/multimodal_actual"
