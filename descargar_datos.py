#!/usr/bin/env python3
"""Descarga el dataset BreastDCEDL desde el bucket publico del curso.

    python descargar_datos.py
    python descargar_datos.py --pacientes 5
    python descargar_datos.py --solo-test
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Falta 'requests'. Instala las dependencias con: python -m pip install -r requirements.txt")


BASE = "https://storage.googleapis.com/usecasesf-breastdcedl-alumnos-8264/breastdcedl"
AUXILIARES = [
    "GUIA.md",
    "LICENSE",
    "utils_caso.py",
    "ver_muestras.py",
    "metadata/samples.csv",
    "metadata/patients.csv",
    "metadata/excluded_patients.csv",
    "metadata/statistics.json",
    "documentation/caso_breastdcedl.pdf",
    "documentation/figures/bucket_estructura.png",
    "documentation/figures/example_dce.png",
    "documentation/figures/estructura_dataset.png",
    "documentation/figures/pcr_two_patient_paths.png",
]


def bajar(sesion: requests.Session, ruta: str, destino: Path) -> tuple[str, bool, str]:
    """Descarga un fichero si no existe ya."""
    salida = destino / ruta
    if salida.exists() and salida.stat().st_size > 0:
        return ruta, True, "ya estaba"

    temporal = salida.with_name(salida.name + ".part")
    try:
        respuesta = sesion.get(f"{BASE}/{ruta}", timeout=60)
        respuesta.raise_for_status()
        salida.parent.mkdir(parents=True, exist_ok=True)
        temporal.write_bytes(respuesta.content)
        temporal.replace(salida)
        return ruta, True, "ok"
    except Exception as error:
        temporal.unlink(missing_ok=True)
        return ruta, False, str(error)[:80]


def descargar_lote(rutas: list[str], destino: Path, hilos: int, etiqueta: str) -> list[str]:
    """Descarga en paralelo y devuelve los ficheros que fallaron."""
    fallos: list[str] = []
    total = len(rutas)
    with requests.Session() as sesion:
        adaptador = requests.adapters.HTTPAdapter(pool_maxsize=hilos, max_retries=3)
        sesion.mount("https://", adaptador)
        with ThreadPoolExecutor(max_workers=hilos) as pool:
            futuros = [pool.submit(bajar, sesion, ruta, destino) for ruta in rutas]
            for hechos, futuro in enumerate(as_completed(futuros), 1):
                ruta, ok, _ = futuro.result()
                if not ok:
                    fallos.append(ruta)
                if hechos % 50 == 0 or hechos == total:
                    print(
                        f"\r  {etiqueta}: {hechos}/{total}"
                        f"{f'  ({len(fallos)} fallos)' if fallos else ''}",
                        end="",
                        flush=True,
                    )
    print()
    return fallos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", type=Path, default=Path("breastdcedl"))
    parser.add_argument("--hilos", type=int, default=16)
    parser.add_argument("--solo-test", action="store_true")
    parser.add_argument("--pacientes", type=int, metavar="N")
    args = parser.parse_args()

    if args.hilos < 1 or (args.pacientes is not None and args.pacientes < 1):
        parser.error("--hilos y --pacientes deben ser numeros positivos")

    destino: Path = args.destino
    destino.mkdir(parents=True, exist_ok=True)
    print(f"Destino: {destino.resolve()}\n")

    fallos = descargar_lote(AUXILIARES, destino, args.hilos, "documentacion")
    if fallos:
        print(f"No se pudieron descargar {len(fallos)} archivos auxiliares.")
        for ruta in fallos[:5]:
            print(f"  {ruta}")
        sys.exit(1)

    with (destino / "metadata/samples.csv").open(encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))

    if args.solo_test:
        filas = [fila for fila in filas if fila["split"] == "test"]
    if args.pacientes is not None:
        elegidas = set(sorted({fila["patient_id"] for fila in filas})[: args.pacientes])
        filas = [fila for fila in filas if fila["patient_id"] in elegidas]

    imagenes = [fila[columna] for fila in filas for columna in ("path_pre", "path_early", "path_late")]
    pacientes = len({fila["patient_id"] for fila in filas})
    print(f"\n{len(imagenes)} imagenes de {len(filas)} cortes, {pacientes} pacientes\n")

    fallos = descargar_lote(imagenes, destino, args.hilos, "imagenes")
    if fallos:
        print(f"\nReintentando {len(fallos)} archivos...")
        descargar_lote(fallos, destino, max(4, args.hilos // 2), "reintento")

    faltan = [ruta for ruta in imagenes if not (destino / ruta).is_file() or (destino / ruta).stat().st_size == 0]
    if faltan:
        print(f"FALTAN {len(faltan)} imagenes. Vuelve a ejecutar para reanudar.")
        for ruta in faltan[:5]:
            print(f"  {ruta}")
        sys.exit(1)

    print(f"COMPLETO: {len(imagenes)} imagenes en {destino.resolve()}")
    print("Siguiente paso: lee GUIA.md")


if __name__ == "__main__":
    main()
