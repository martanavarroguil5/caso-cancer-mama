#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA no disponible"; print(torch.cuda.get_device_name(0))'
"$PYTHON" -m cancer_mama.seleccion_cortes preparar
"$PYTHON" -m cancer_mama.seleccion_cortes entrenar --prueba --dispositivo cuda --workers 0
"$PYTHON" -m cancer_mama.seleccion_cortes entrenar --dispositivo cuda --workers "${WORKERS:-4}"
"$PYTHON" -m cancer_mama.seleccion_cortes comparar
