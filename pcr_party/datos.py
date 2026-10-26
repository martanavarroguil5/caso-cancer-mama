"""Índice, caché de píxeles y particiones por paciente.

Solo se leen las carpetas del split pedido. El entrenamiento usa exclusivamente
``train``; ``test`` se reserva para la evaluación final de un modelo congelado.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "breastdcedl"
CACHE = RAIZ / "cache"
FASES = ("PRE", "EARLY", "LATE")
COLUMNAS_FASE = ("path_pre", "path_early", "path_late")
LADO = 256


def leer_png(ruta: Path) -> np.ndarray:
    """Lee un PNG gris de 8 bits y 256×256; cualquier otra cosa es un error."""
    with Image.open(ruta) as imagen:
        if imagen.format != "PNG" or imagen.mode != "L" or imagen.size != (LADO, LADO):
            raise ValueError(
                f"{ruta}: se espera PNG gris L {LADO}×{LADO}, recibido "
                f"{imagen.format} {imagen.mode} {imagen.size}")
        return np.asarray(imagen, dtype=np.uint8).copy()


def cargar_indice(split: str = "train", raiz: Path = DATOS) -> pd.DataFrame:
    """Un corte por fila, con etiqueta, fold, cohorte y objetivos auxiliares."""
    if split not in ("train", "test"):
        raise ValueError("Solo existen los splits públicos train y test")
    samples = pd.read_csv(raiz / "metadata/samples.csv")
    patients = pd.read_csv(raiz / "metadata/patients.csv")
    cortes = samples[samples["split"] == split].copy()
    for columna in COLUMNAS_FASE:
        if not cortes[columna].str.startswith(f"dataset/{split}/").all():
            raise ValueError(f"Hay rutas fuera de dataset/{split}/ en {columna}")
    clinica = patients.rename(columns={"pid": "patient_id", "dataset": "cohorte"})
    clinica = clinica[["patient_id", "split", "pCR", "cohorte", "HR", "HER2", "tum_vol"]]
    cortes = cortes.merge(clinica, on="patient_id", how="left", validate="many_to_one",
                          suffixes=("", "_paciente"))
    if (cortes["split_paciente"] != split).any():
        raise ValueError("samples.csv y patients.csv discrepan en el split")
    if (cortes["pCR"] != cortes["pCR_paciente"]).any():
        raise ValueError("samples.csv y patients.csv discrepan en la etiqueta pCR")
    cortes = cortes.drop(columns=["split_paciente", "pCR_paciente"])
    cortes["pCR"] = cortes["pCR"].astype(int)
    return cortes.sort_values(["patient_id", "slice_index"]).reset_index(drop=True)


def cargar_imagenes(indice: pd.DataFrame, raiz: Path = DATOS,
                    cache: Path | None = CACHE) -> np.ndarray:
    """Devuelve uint8 [N, 3, 256, 256] en orden PRE, EARLY, LATE.

    La caché se identifica por el hash de los ``sample_id``; dividir entre 255
    ocurre después, en GPU, exactamente igual en entrenamiento e inferencia.
    """
    clave = hashlib.sha256("\n".join(indice["sample_id"]).encode()).hexdigest()[:16]
    ruta = None if cache is None else cache / f"pixeles_{clave}.npy"
    if ruta is not None and ruta.exists():
        return np.load(ruta)
    pixeles = np.empty((len(indice), 3, LADO, LADO), dtype=np.uint8)
    rutas = indice[list(COLUMNAS_FASE)].to_numpy()

    def leer(i: int) -> None:
        for canal, relativa in enumerate(rutas[i]):
            pixeles[i, canal] = leer_png(raiz / relativa)

    with ThreadPoolExecutor(max_workers=16) as hilos:
        list(hilos.map(leer, range(len(indice))))
    if ruta is not None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal = ruta.with_name(ruta.stem + ".parcial.npy")
        np.save(temporal, pixeles)
        temporal.rename(ruta)
    return pixeles


def tabla_pacientes(indice: pd.DataFrame) -> pd.DataFrame:
    """Una fila por paciente; falla si un atributo de paciente varía entre cortes."""
    grupos = indice.groupby("patient_id", sort=True)
    for columna in ("pCR", "fold", "cohorte"):
        if (grupos[columna].nunique(dropna=False) > 1).any():
            raise ValueError(f"{columna} no es constante dentro de una paciente")
    return grupos.agg(pCR=("pCR", "first"), fold=("fold", "first"),
                      cohorte=("cohorte", "first"), HR=("HR", "first"),
                      HER2=("HER2", "first"), tum_vol=("tum_vol", "first"),
                      n_cortes=("sample_id", "size")).reset_index()


def separar_seleccion(pacientes: pd.DataFrame, fraccion: float, semilla: int) -> set[str]:
    """Muestra estratificada por cohorte × pCR, determinista e independiente del modelo."""
    rng = np.random.default_rng(semilla)
    elegidas: list[str] = []
    for _, grupo in pacientes.groupby(["cohorte", "pCR"], sort=True):
        ids = np.sort(grupo["patient_id"].to_numpy())
        rng.shuffle(ids)
        elegidas.extend(ids[:int(round(fraccion * len(ids)))])
    return set(elegidas)


def roles_fold(pacientes: pd.DataFrame, fold: int, fraccion_seleccion: float) -> pd.DataFrame:
    """fit / seleccion / exterior para un fold exterior de samples.csv.

    El exterior es el fold original completo. Selección solo decide la parada;
    nunca ve el exterior. Las semillas de modelo no cambian estos roles.
    """
    if fold not in set(pacientes["fold"]):
        raise ValueError(f"Fold {fold} inexistente")
    roles = pacientes[["patient_id", "pCR", "cohorte", "fold"]].copy()
    desarrollo = roles[roles["fold"] != fold]
    seleccion = separar_seleccion(desarrollo, fraccion_seleccion, semilla=1000 + fold)
    roles["rol"] = np.where(roles["fold"] == fold, "exterior",
                            np.where(roles["patient_id"].isin(seleccion), "seleccion", "fit"))
    return roles
