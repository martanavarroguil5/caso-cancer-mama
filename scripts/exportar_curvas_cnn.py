"""Exporta pérdidas por fold y época desde los históricos de la CNN, sin entrenar."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcr_party.recursos_app import exportar_curvas_cnn

if __name__ == "__main__":
    try:
        print(exportar_curvas_cnn())
    except FileNotFoundError as error:
        raise SystemExit(str(error))
