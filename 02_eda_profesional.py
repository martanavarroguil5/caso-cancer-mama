#!/usr/bin/env python3
"""EDA profesional, reproducible y a nivel de paciente para BreastDCEDL.

El análisis separa cinco preguntas:

1. ¿Los datos son completos, coherentes y trazables?
2. ¿Cómo se distribuyen la clase objetivo y las variables clínicas?
3. ¿Existen diferencias de adquisición entre cohortes o entre train y test?
4. ¿Qué estructura contienen las imágenes antes de entrenar una CNN?
5. ¿Qué variables deben excluirse, vigilarse o conservarse para modelado?

La asociación con pCR se estudia solo en train. Test se usa únicamente para
medir cambio de distribución, evitando tomar decisiones a partir de sus etiquetas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from textwrap import dedent

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "breastdcedl"
SALIDA_PREDETERMINADA = RAIZ / "resultados" / "02_eda"
os.environ.setdefault("MPLCONFIGDIR", str(RAIZ / ".codex_tmp" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


FASES = ("PRE", "EARLY", "LATE")
RUTAS = ("path_pre", "path_early", "path_late")

VARIABLES_ADQUISICION = [
    "n_xy", "n_z", "n_times", "pre", "post_early", "post_late",
    "slice_thick", "xy_spacing",
]
VARIABLES_LOCALIZACION = [
    "mask_start", "mask_end", "sraw", "eraw", "scol", "ecol", "tum_vol",
]
VARIABLES_CLINICAS_NUM = ["age", "tum_vol"]
VARIABLES_CLINICAS_CAT = [
    "menopause", "race_white", "race_black", "HR", "HER2", "HR_HER2_STATUS",
    "TripleNeg", "HER2pos", "HRposHER2neg",
]
VARIABLES_CATEGORICAS_EDA = [
    "dataset", "menopause", "race_white", "race_black", "HR", "HER2",
    "HR_HER2_STATUS", "TripleNeg", "HER2pos", "HRposHER2neg",
]

COLORES = {
    "pcr0": "#4C78A8",
    "pcr1": "#F58518",
    "train": "#4C78A8",
    "test": "#E45756",
    "spy1": "#4C78A8",
    "spy2": "#54A24B",
    "duke": "#B279A2",
    "neutral": "#79706E",
}


def configurar_estilo() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
    })


def sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def nivel_categoria(serie: pd.Series) -> pd.Series:
    """Convierte categorías en texto sin transformar ausentes en ceros."""
    salida = serie.astype("string")
    return salida.fillna("No disponible")


def iqr(serie: pd.Series) -> float:
    valores = pd.to_numeric(serie, errors="coerce").dropna()
    if valores.empty:
        return float("nan")
    return float(valores.quantile(0.75) - valores.quantile(0.25))


def smd(grupo_a: pd.Series, grupo_b: pd.Series) -> float:
    """Diferencia estandarizada: media de B menos media de A."""
    a = pd.to_numeric(grupo_a, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(grupo_b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    escala = math.sqrt((float(np.var(a, ddof=1)) + float(np.var(b, ddof=1))) / 2)
    if not np.isfinite(escala) or escala == 0:
        return 0.0 if np.isclose(np.mean(a), np.mean(b)) else float("nan")
    return float((np.mean(b) - np.mean(a)) / escala)


def total_variation(a: pd.Series, b: pd.Series) -> float:
    pa = nivel_categoria(a).value_counts(normalize=True)
    pb = nivel_categoria(b).value_counts(normalize=True)
    categorias = pa.index.union(pb.index)
    return float(0.5 * (pa.reindex(categorias, fill_value=0) - pb.reindex(categorias, fill_value=0)).abs().sum())


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    tabla = pd.crosstab(nivel_categoria(x), nivel_categoria(y)).to_numpy(dtype=float)
    n = tabla.sum()
    if n == 0 or min(tabla.shape) < 2:
        return 0.0
    esperado = np.outer(tabla.sum(axis=1), tabla.sum(axis=0)) / n
    valido = esperado > 0
    chi2 = float((((tabla - esperado) ** 2)[valido] / esperado[valido]).sum())
    denominador = n * min(tabla.shape[0] - 1, tabla.shape[1] - 1)
    return float(math.sqrt(chi2 / denominador)) if denominador > 0 else 0.0


def hist_quantile(hist: np.ndarray, q: float, offset: int = 0) -> float:
    if hist.sum() == 0:
        return float("nan")
    posicion = q * (int(hist.sum()) - 1)
    return float(np.searchsorted(np.cumsum(hist), posicion, side="right") - offset)


def resumen_calidad(patients: pd.DataFrame, samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filas = []
    for columna in patients.columns:
        serie = patients[columna]
        filas.append({
            "tabla": "patients",
            "variable": columna,
            "dtype": str(serie.dtype),
            "filas": len(serie),
            "no_nulos": int(serie.notna().sum()),
            "ausentes": int(serie.isna().sum()),
            "ausentes_pct": float(serie.isna().mean()),
            "unicos": int(serie.nunique(dropna=True)),
            "constante": bool(serie.nunique(dropna=True) <= 1),
        })
    for columna in samples.columns:
        serie = samples[columna]
        filas.append({
            "tabla": "samples",
            "variable": columna,
            "dtype": str(serie.dtype),
            "filas": len(serie),
            "no_nulos": int(serie.notna().sum()),
            "ausentes": int(serie.isna().sum()),
            "ausentes_pct": float(serie.isna().mean()),
            "unicos": int(serie.nunique(dropna=True)),
            "constante": bool(serie.nunique(dropna=True) <= 1),
        })
    calidad = pd.DataFrame(filas)

    faltantes_cohorte = []
    for columna in patients.columns:
        if patients[columna].isna().any():
            for cohorte, grupo in patients.groupby("dataset"):
                faltantes_cohorte.append({
                    "variable": columna,
                    "cohorte": cohorte,
                    "n": len(grupo),
                    "ausentes": int(grupo[columna].isna().sum()),
                    "ausentes_pct": float(grupo[columna].isna().mean()),
                })
    return calidad, pd.DataFrame(faltantes_cohorte)


def enriquecer_pacientes(patients: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    cortes = samples.groupby("patient_id").agg(
        n_cortes_seleccionados=("sample_id", "size"),
        z_min_seleccionado=("slice_index", "min"),
        z_max_seleccionado=("slice_index", "max"),
        fold=("fold", "first"),
    )
    df = patients.merge(cortes, left_on="pid", right_index=True, how="left", validate="one_to_one")
    df["tumor_z_span"] = df["mask_end"] - df["mask_start"] + 1
    df["crop_height"] = df["eraw"] - df["sraw"] + 1
    df["crop_width"] = df["ecol"] - df["scol"] + 1
    df["fov_xy"] = df["n_xy"] * df["xy_spacing"]
    df["log1p_tum_vol"] = np.log1p(df["tum_vol"].clip(lower=0))
    return df


def extraer_features_paciente(patient_id: str, grupo: pd.DataFrame) -> dict:
    grupo = grupo.sort_values("slice_index")
    volumen = np.empty((len(grupo), 3, 256, 256), dtype=np.uint8)
    for i, fila in enumerate(grupo.itertuples(index=False)):
        for j, columna in enumerate(RUTAS):
            with Image.open(DATOS / getattr(fila, columna)) as png:
                volumen[i, j] = np.asarray(png.convert("L"), dtype=np.uint8)

    tejido = volumen[:, 0] > 25
    n_tejido = int(tejido.sum())
    total = int(tejido.size)
    resultado: dict[str, float | int | str] = {
        "pid": patient_id,
        "n_cortes_imagen": int(len(grupo)),
        "background_fraction": float(1 - n_tejido / total),
    }

    for j, fase in enumerate(FASES):
        todos = volumen[:, j].ravel()
        valores = volumen[:, j][tejido]
        hist = np.bincount(valores, minlength=256)
        prefijo = fase.lower()
        resultado[f"{prefijo}_mean_all"] = float(todos.mean() / 255)
        resultado[f"{prefijo}_mean_tissue"] = float(valores.mean() / 255) if n_tejido else float("nan")
        resultado[f"{prefijo}_std_tissue"] = float(valores.std() / 255) if n_tejido else float("nan")
        resultado[f"{prefijo}_p10_tissue"] = hist_quantile(hist, 0.10) / 255
        resultado[f"{prefijo}_p50_tissue"] = hist_quantile(hist, 0.50) / 255
        resultado[f"{prefijo}_p90_tissue"] = hist_quantile(hist, 0.90) / 255
        resultado[f"{prefijo}_saturation_fraction"] = float((todos == 255).mean())

    pre = volumen[:, 0].astype(np.int16)
    early = volumen[:, 1].astype(np.int16)
    late = volumen[:, 2].astype(np.int16)
    realce = (early - pre)[tejido]
    lavado = (late - early)[tejido]
    hist_realce = np.bincount(realce + 255, minlength=511)
    resultado["enhancement_mean"] = float(realce.mean() / 255) if n_tejido else float("nan")
    resultado["enhancement_std"] = float(realce.std() / 255) if n_tejido else float("nan")
    resultado["enhancement_p90"] = hist_quantile(hist_realce, 0.90, offset=255) / 255
    resultado["enhancement_positive_fraction"] = float((realce > 0).mean()) if n_tejido else float("nan")
    resultado["washout_mean"] = float(lavado.mean() / 255) if n_tejido else float("nan")
    resultado["washout_fraction"] = float((lavado < 0).mean()) if n_tejido else float("nan")
    return resultado


def extraer_features_imagen(samples: pd.DataFrame, salida_csv: Path, workers: int, recalcular: bool) -> pd.DataFrame:
    if salida_csv.exists() and not recalcular:
        features = pd.read_csv(salida_csv)
        if set(features.pid) == set(samples.patient_id) and len(features) == samples.patient_id.nunique():
            print(f"Reutilizando {salida_csv}")
            return features

    grupos = [(patient_id, grupo.copy()) for patient_id, grupo in samples.groupby("patient_id", sort=True)]
    resultados: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futuros = {
            pool.submit(extraer_features_paciente, patient_id, grupo): patient_id
            for patient_id, grupo in grupos
        }
        for i, futuro in enumerate(as_completed(futuros), start=1):
            resultados.append(futuro.result())
            if i % 100 == 0 or i == len(futuros):
                print(f"\rCaracterísticas de imagen: {i:,}/{len(futuros):,} pacientes", end="", flush=True)
    print()
    features = pd.DataFrame(resultados).sort_values("pid").reset_index(drop=True)
    features.to_csv(salida_csv, index=False)
    return features


def tabla_resumen_numerico(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    filas = []
    for variable in variables:
        if variable not in df:
            continue
        for split, grupo in df.groupby("split"):
            valores = pd.to_numeric(grupo[variable], errors="coerce")
            filas.append({
                "variable": variable,
                "split": split,
                "n": int(valores.notna().sum()),
                "ausentes": int(valores.isna().sum()),
                "media": float(valores.mean()),
                "desviacion": float(valores.std()),
                "min": float(valores.min()),
                "p25": float(valores.quantile(0.25)),
                "mediana": float(valores.median()),
                "p75": float(valores.quantile(0.75)),
                "max": float(valores.max()),
            })
    return pd.DataFrame(filas)


def asociaciones_numericas_train(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    train = df[df.split == "train"]
    filas = []
    for variable in variables:
        if variable not in train:
            continue
        valores = pd.to_numeric(train[variable], errors="coerce")
        g0 = valores[train.pCR == 0]
        g1 = valores[train.pCR == 1]
        if valores.notna().sum() < 10 or valores.nunique(dropna=True) < 2:
            continue
        filas.append({
            "variable": variable,
            "n": int(valores.notna().sum()),
            "ausentes_pct": float(valores.isna().mean()),
            "n_pcr0": int(g0.notna().sum()),
            "n_pcr1": int(g1.notna().sum()),
            "mediana_pcr0": float(g0.median()),
            "iqr_pcr0": iqr(g0),
            "mediana_pcr1": float(g1.median()),
            "iqr_pcr1": iqr(g1),
            "smd_pcr1_menos_pcr0": smd(g0, g1),
            "correlacion_pcr": float(pd.concat([valores, train.pCR], axis=1).corr().iloc[0, 1]),
        })
    salida = pd.DataFrame(filas)
    if not salida.empty:
        salida["abs_smd"] = salida.smd_pcr1_menos_pcr0.abs()
        salida = salida.sort_values(["abs_smd", "variable"], ascending=[False, True])
    return salida


def asociaciones_categoricas_train(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    train = df[df.split == "train"]
    filas = []
    for variable in variables:
        if variable not in train:
            continue
        categorias = nivel_categoria(train[variable])
        if categorias.nunique() < 2:
            continue
        tabla = pd.DataFrame({"categoria": categorias, "pCR": train.pCR}).groupby("categoria").pCR.agg(["size", "sum", "mean"])
        for categoria, fila in tabla.iterrows():
            filas.append({
                "variable": variable,
                "categoria": categoria,
                "n": int(fila["size"]),
                "pcr_positivos": int(fila["sum"]),
                "tasa_pcr": float(fila["mean"]),
                "cramers_v_variable": cramers_v(categorias, train.pCR.astype(str)),
            })
    salida = pd.DataFrame(filas)
    return salida.sort_values(["cramers_v_variable", "variable", "n"], ascending=[False, True, False])


def analizar_shift(df: pd.DataFrame, numericas: list[str], categoricas: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df.split == "train"]
    test = df[df.split == "test"]
    filas_num = []
    for variable in numericas:
        if variable not in df or pd.to_numeric(df[variable], errors="coerce").nunique(dropna=True) < 2:
            continue
        filas_num.append({
            "variable": variable,
            "media_train": float(pd.to_numeric(train[variable], errors="coerce").mean()),
            "media_test": float(pd.to_numeric(test[variable], errors="coerce").mean()),
            "mediana_train": float(pd.to_numeric(train[variable], errors="coerce").median()),
            "mediana_test": float(pd.to_numeric(test[variable], errors="coerce").median()),
            "smd_test_menos_train": smd(train[variable], test[variable]),
            "ausentes_train_pct": float(train[variable].isna().mean()),
            "ausentes_test_pct": float(test[variable].isna().mean()),
        })
    shift_num = pd.DataFrame(filas_num)
    shift_num["abs_smd"] = shift_num.smd_test_menos_train.abs()
    shift_num = shift_num.sort_values(["abs_smd", "variable"], ascending=[False, True])

    filas_cat = []
    for variable in categoricas:
        if variable in df and nivel_categoria(df[variable]).nunique() > 1:
            filas_cat.append({
                "variable": variable,
                "total_variation_train_test": total_variation(train[variable], test[variable]),
                "niveles": int(nivel_categoria(df[variable]).nunique()),
            })
    shift_cat = pd.DataFrame(filas_cat).sort_values("total_variation_train_test", ascending=False)
    return shift_num, shift_cat


def pca_numpy(df: pd.DataFrame, variables: list[str], prefijo: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    disponibles = [v for v in variables if v in df and pd.to_numeric(df[v], errors="coerce").nunique(dropna=True) > 1]
    matriz = df[disponibles].apply(pd.to_numeric, errors="coerce")
    medianas = matriz.median()
    matriz = matriz.fillna(medianas)
    medias = matriz.mean()
    desv = matriz.std(ddof=0).replace(0, np.nan)
    z = ((matriz - medias) / desv).dropna(axis=1)
    disponibles = z.columns.tolist()
    u, singular, vt = np.linalg.svd(z.to_numpy(dtype=float), full_matrices=False)
    varianza = singular ** 2
    proporcion = varianza / varianza.sum()
    scores = z.to_numpy(dtype=float) @ vt[:2].T
    tabla_scores = df[["pid", "pCR", "dataset", "split"]].copy()
    tabla_scores[f"{prefijo}_pc1"] = scores[:, 0]
    tabla_scores[f"{prefijo}_pc2"] = scores[:, 1]
    tabla_loadings = pd.DataFrame({
        "variable": disponibles,
        "pc1": vt[0],
        "pc2": vt[1],
        "abs_pc1": np.abs(vt[0]),
        "abs_pc2": np.abs(vt[1]),
    }).sort_values("abs_pc1", ascending=False)
    metadatos = {
        "variables": disponibles,
        "explained_variance_pc1": float(proporcion[0]),
        "explained_variance_pc2": float(proporcion[1]),
        "imputacion": "mediana global solo para visualización PCA",
        "escalado": "z-score",
    }
    return tabla_scores, tabla_loadings, metadatos


def pares_correlacionados(df: pd.DataFrame, variables: list[str], umbral: float = 0.90) -> pd.DataFrame:
    disponibles = [v for v in variables if v in df and pd.to_numeric(df[v], errors="coerce").nunique(dropna=True) > 1]
    corr = df[disponibles].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    filas = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            valor = corr.loc[a, b]
            if pd.notna(valor) and abs(valor) >= umbral:
                filas.append({"variable_a": a, "variable_b": b, "spearman": float(valor), "abs_spearman": abs(float(valor))})
    return pd.DataFrame(filas).sort_values("abs_spearman", ascending=False) if filas else pd.DataFrame(columns=["variable_a", "variable_b", "spearman", "abs_spearman"])


def guardar_figura(fig: plt.Figure, ruta: Path) -> None:
    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figura_calidad(df: pd.DataFrame, faltantes: pd.DataFrame, salida: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    clases = df.groupby(["split", "pCR"]).size().unstack(fill_value=0).reindex(["train", "test"])
    clases.plot(kind="bar", stacked=True, color=[COLORES["pcr0"], COLORES["pcr1"]], ax=axes[0, 0])
    axes[0, 0].set_title("Pacientes por clase y split")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Pacientes")
    axes[0, 0].tick_params(axis="x", rotation=0)
    axes[0, 0].legend(["pCR=0", "pCR=1"])
    for contenedor in axes[0, 0].containers:
        axes[0, 0].bar_label(contenedor, label_type="center", color="white")

    cohortes = df.groupby(["dataset", "split"]).size().unstack(fill_value=0).reindex(["spy1", "spy2", "duke"])
    cohortes.plot(kind="bar", color=[COLORES["test"], COLORES["train"]], ax=axes[0, 1])
    axes[0, 1].set_title("Composición de cohortes")
    axes[0, 1].set_xlabel("Cohorte")
    axes[0, 1].set_ylabel("Pacientes")
    axes[0, 1].tick_params(axis="x", rotation=0)

    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    axes[1, 0].barh(missing.index[::-1], 100 * missing.values[::-1], color=COLORES["neutral"])
    axes[1, 0].set_title("Valores ausentes")
    axes[1, 0].set_xlabel("Porcentaje de pacientes")
    axes[1, 0].set_xlim(0, max(15, 100 * missing.max() * 1.15))
    for y, valor in enumerate((100 * missing.values[::-1])):
        axes[1, 0].text(valor + 0.2, y, f"{valor:.1f}%", va="center", fontsize=9)

    cortes = df["n_cortes_seleccionados"].value_counts().sort_index()
    axes[1, 1].bar(cortes.index.astype(str), cortes.values, color=COLORES["train"])
    axes[1, 1].set_title("Cortes seleccionados por paciente")
    axes[1, 1].set_xlabel("Número de cortes")
    axes[1, 1].set_ylabel("Pacientes")
    for x, valor in enumerate(cortes.values):
        axes[1, 1].text(x, valor + max(cortes.values) * 0.015, str(valor), ha="center", fontsize=9)

    fig.suptitle("Calidad y composición del dataset", fontsize=16, fontweight="bold")
    guardar_figura(fig, salida)


def figura_clinica(df: pd.DataFrame, salida: Path) -> None:
    train = df[df.split == "train"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    for i, (variable, etiqueta) in enumerate([("age", "Edad"), ("log1p_tum_vol", "log(1 + volumen tumoral)")]):
        ax = axes[0, i]
        datos = [train.loc[train.pCR == clase, variable].dropna() for clase in (0, 1)]
        bp = ax.boxplot(datos, tick_labels=["pCR=0", "pCR=1"], patch_artist=True, showfliers=False)
        for parche, color in zip(bp["boxes"], [COLORES["pcr0"], COLORES["pcr1"]]):
            parche.set_facecolor(color)
            parche.set_alpha(0.75)
        ax.set_title(f"{etiqueta} por respuesta (train)")
        ax.set_ylabel(etiqueta)

    subtipo = (
        pd.DataFrame({"subtipo": nivel_categoria(train.HR_HER2_STATUS), "pCR": train.pCR})
        .groupby("subtipo").pCR.agg(["size", "mean"]).sort_values("mean")
    )
    axes[1, 0].barh(subtipo.index.tolist(), 100 * subtipo["mean"], color=COLORES["pcr1"])
    axes[1, 0].set_title("Tasa pCR por subtipo (train)")
    axes[1, 0].set_xlabel("pCR=1 (%)")
    for y, (_, fila) in enumerate(subtipo.iterrows()):
        axes[1, 0].text(100 * fila["mean"] + 0.5, y, f"{100*fila['mean']:.1f}% (n={int(fila['size'])})", va="center", fontsize=9)

    cohorte = train.groupby("dataset").pCR.agg(["size", "mean"]).reindex(["spy1", "spy2", "duke"])
    axes[1, 1].bar(cohorte.index, 100 * cohorte["mean"], color=[COLORES[c] for c in cohorte.index])
    axes[1, 1].set_title("Tasa pCR por cohorte (train)")
    axes[1, 1].set_ylabel("Pacientes pCR=1 (%)")
    axes[1, 1].set_ylim(0, max(40, float((100 * cohorte["mean"]).max()) * 1.25))
    for x, (_, fila) in enumerate(cohorte.iterrows()):
        axes[1, 1].text(x, 100 * fila["mean"] + 0.7, f"{100*fila['mean']:.1f}%\nn={int(fila['size'])}", ha="center", fontsize=9)

    fig.suptitle("Variables clínicas y respuesta", fontsize=16, fontweight="bold")
    guardar_figura(fig, salida)


def figura_cohortes(df: pd.DataFrame, salida: Path) -> None:
    variables = ["n_z", "n_times", "slice_thick", "xy_spacing", "fov_xy", "tumor_z_span"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, variable in zip(axes.ravel(), variables):
        datos = [df.loc[df.dataset == cohorte, variable].dropna() for cohorte in ("spy1", "spy2", "duke")]
        bp = ax.boxplot(datos, tick_labels=["spy1", "spy2", "duke"], patch_artist=True, showfliers=False)
        for parche, cohorte in zip(bp["boxes"], ("spy1", "spy2", "duke")):
            parche.set_facecolor(COLORES[cohorte])
            parche.set_alpha(0.75)
        ax.set_title(variable)
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Adquisición y preprocesado por cohorte", fontsize=16, fontweight="bold")
    guardar_figura(fig, salida)


def panel_pca(ax: plt.Axes, scores: pd.DataFrame, x: str, y: str, colorear: str, titulo: str) -> None:
    if colorear == "dataset":
        for categoria in ("spy1", "spy2", "duke"):
            grupo = scores[scores.dataset == categoria]
            ax.scatter(grupo[x], grupo[y], s=16, alpha=0.55, label=categoria, color=COLORES[categoria], edgecolors="none")
    else:
        for categoria in (0, 1):
            grupo = scores[(scores.split == "train") & (scores.pCR == categoria)]
            ax.scatter(grupo[x], grupo[y], s=16, alpha=0.55, label=f"pCR={categoria}", color=COLORES[f"pcr{categoria}"], edgecolors="none")
    ax.axhline(0, color="#BBBBBB", linewidth=0.7)
    ax.axvline(0, color="#BBBBBB", linewidth=0.7)
    ax.set_title(titulo)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=1.4)


def figura_pca(scores_clin: pd.DataFrame, meta_clin: dict, scores_img: pd.DataFrame, meta_img: dict, salida: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    panel_pca(axes[0, 0], scores_clin, "clin_pc1", "clin_pc2", "dataset", "PCA clínica/adquisición por cohorte")
    panel_pca(axes[0, 1], scores_clin, "clin_pc1", "clin_pc2", "pCR", "PCA clínica/adquisición por pCR (train)")
    panel_pca(axes[1, 0], scores_img, "img_pc1", "img_pc2", "dataset", "PCA de imagen por cohorte")
    panel_pca(axes[1, 1], scores_img, "img_pc1", "img_pc2", "pCR", "PCA de imagen por pCR (train)")
    axes[0, 0].set_xlabel(f"PC1 ({meta_clin['explained_variance_pc1']:.1%})")
    axes[0, 0].set_ylabel(f"PC2 ({meta_clin['explained_variance_pc2']:.1%})")
    axes[0, 1].set_xlabel(f"PC1 ({meta_clin['explained_variance_pc1']:.1%})")
    axes[0, 1].set_ylabel(f"PC2 ({meta_clin['explained_variance_pc2']:.1%})")
    axes[1, 0].set_xlabel(f"PC1 ({meta_img['explained_variance_pc1']:.1%})")
    axes[1, 0].set_ylabel(f"PC2 ({meta_img['explained_variance_pc2']:.1%})")
    axes[1, 1].set_xlabel(f"PC1 ({meta_img['explained_variance_pc1']:.1%})")
    axes[1, 1].set_ylabel(f"PC2 ({meta_img['explained_variance_pc2']:.1%})")
    fig.suptitle("Estructura multivariante a nivel de paciente", fontsize=16, fontweight="bold")
    guardar_figura(fig, salida)


def figura_shift_asociaciones(shift: pd.DataFrame, asociaciones: pd.DataFrame, salida: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    top_shift = shift.dropna(subset=["abs_smd"]).head(15).sort_values("smd_test_menos_train")
    colores_shift = [COLORES["test"] if x > 0 else COLORES["train"] for x in top_shift.smd_test_menos_train]
    axes[0].barh(top_shift.variable, top_shift.smd_test_menos_train, color=colores_shift)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].axvline(-0.2, color="#BBBBBB", linewidth=0.8, linestyle="--")
    axes[0].axvline(0.2, color="#BBBBBB", linewidth=0.8, linestyle="--")
    axes[0].set_title("Cambio de distribución: test − train")
    axes[0].set_xlabel("Diferencia estandarizada (SMD)")

    top_assoc = asociaciones.dropna(subset=["abs_smd"]).head(15).sort_values("smd_pcr1_menos_pcr0")
    colores_assoc = [COLORES["pcr1"] if x > 0 else COLORES["pcr0"] for x in top_assoc.smd_pcr1_menos_pcr0]
    axes[1].barh(top_assoc.variable, top_assoc.smd_pcr1_menos_pcr0, color=colores_assoc)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].axvline(-0.2, color="#BBBBBB", linewidth=0.8, linestyle="--")
    axes[1].axvline(0.2, color="#BBBBBB", linewidth=0.8, linestyle="--")
    axes[1].set_title("Asociación descriptiva con pCR en train")
    axes[1].set_xlabel("Diferencia estandarizada: pCR=1 − pCR=0")
    fig.suptitle("Señales para vigilar antes de modelar", fontsize=16, fontweight="bold")
    guardar_figura(fig, salida)


def cargar_corte_central(samples: pd.DataFrame, patient_id: str) -> tuple[np.ndarray, pd.Series]:
    grupo = samples[samples.patient_id == patient_id].sort_values("slice_index")
    fila = grupo.iloc[len(grupo) // 2]
    canales = []
    for columna in RUTAS:
        with Image.open(DATOS / fila[columna]) as png:
            canales.append(np.asarray(png.convert("L"), dtype=np.float32) / 255)
    return np.stack(canales), fila


def figura_atipicos(df: pd.DataFrame, samples: pd.DataFrame, salida: Path) -> list[dict]:
    criterios = [
        ("Mayor realce medio", "enhancement_mean", False),
        ("Menor realce medio", "enhancement_mean", True),
        ("Mayor saturación EARLY", "early_saturation_fraction", False),
        ("Mayor fondo", "background_fraction", False),
    ]
    seleccion = []
    usados = set()
    for etiqueta, variable, ascendente in criterios:
        candidatos = df.sort_values(variable, ascending=ascendente)
        fila = next(f for _, f in candidatos.iterrows() if f.pid not in usados)
        usados.add(fila.pid)
        seleccion.append((etiqueta, variable, fila))

    fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
    registros = []
    for columna, (etiqueta, variable, paciente) in enumerate(seleccion):
        imagen, fila = cargar_corte_central(samples, paciente.pid)
        realce = imagen[1] - imagen[0]
        axes[0, columna].imshow(imagen[1], cmap="gray", vmin=0, vmax=1)
        axes[1, columna].imshow(realce, cmap="magma", vmin=0, vmax=max(float(realce.max()), 1e-6))
        axes[0, columna].set_title(f"{etiqueta}\n{paciente.pid}", fontsize=10)
        axes[1, columna].set_title(f"{variable}={paciente[variable]:.3f}", fontsize=9)
        axes[0, columna].axis("off")
        axes[1, columna].axis("off")
        registros.append({
            "criterio": etiqueta,
            "variable": variable,
            "pid": paciente.pid,
            "valor": float(paciente[variable]),
            "sample_id": fila.sample_id,
        })
    axes[0, 0].text(-0.08, 0.5, "EARLY", rotation=90, va="center", ha="right", transform=axes[0, 0].transAxes, fontweight="bold")
    axes[1, 0].text(-0.08, 0.5, "EARLY − PRE", rotation=90, va="center", ha="right", transform=axes[1, 0].transAxes, fontweight="bold")
    fig.suptitle("Casos extremos para revisión visual", fontsize=16, fontweight="bold")
    guardar_figura(fig, salida)
    return registros


def politica_variables(df: pd.DataFrame) -> pd.DataFrame:
    filas = [
        ("pid", "agrupación", "Excluir como predictor", "Identificador; usar para agrupar cortes y evitar fuga."),
        ("pCR", "objetivo", "Solo etiqueta", "Variable que se predice."),
        ("split", "partición", "Excluir como predictor", "Codifica train/test de forma directa."),
        ("test", "partición", "Excluir como predictor", "Redundante con split; fuga directa."),
        ("fold", "partición", "Excluir como predictor", "Solo controla validación cruzada."),
        ("dataset", "cohorte", "Monitorizar/estratificar", "Puede capturar hospital y protocolo; útil para auditoría, peligroso como atajo."),
        ("pre", "adquisición", "Excluir", "Constante en todo el dataset."),
        ("post_early/post_late", "adquisición", "Monitorizar", "Índices de tiempo potencialmente ligados a cohorte."),
        ("n_xy/n_z/n_times", "adquisición", "Monitorizar", "Describen el protocolo y pueden codificar cohorte."),
        ("slice_thick/xy_spacing", "adquisición", "Monitorizar", "Resolución física; potencial cambio de dominio."),
        ("mask/crop coordinates", "preprocesado", "No usar inicialmente", "Derivadas de anotación y preprocesado; posible atajo y no siempre disponibles en producción."),
        ("age", "clínica", "Candidata", "Usar solo con imputación dentro de cada fold."),
        ("tum_vol", "clínica", "Candidata", "Muy asimétrica; considerar log1p e imputación por fold."),
        ("HR/HER2/subtipo", "clínica", "Candidata", "Biológicamente relevante; evitar duplicar variables derivadas redundantes."),
        ("menopause", "clínica", "Candidata con cautela", "13,5% ausente y disponibilidad dependiente de la fuente."),
        ("race indicators", "sensible", "Auditar; no usar por defecto", "Variable sensible, incompleta y no causal; evaluar equidad por separado."),
        ("image features EDA", "imagen", "Diagnóstico, no entrada obligatoria", "Resúmenes diseñados para entender sesgo; la CNN recibirá las fases completas."),
    ]
    return pd.DataFrame(filas, columns=["variable_o_grupo", "rol", "recomendacion", "motivo"])


def redactar_informe(
    df: pd.DataFrame,
    calidad: pd.DataFrame,
    faltantes_cohorte: pd.DataFrame,
    assoc_num: pd.DataFrame,
    assoc_cat: pd.DataFrame,
    shift_num: pd.DataFrame,
    shift_cat: pd.DataFrame,
    corr: pd.DataFrame,
    load_clin: pd.DataFrame,
    meta_clin: dict,
    load_img: pd.DataFrame,
    meta_img: dict,
    atipicos: list[dict],
    salida: Path,
) -> None:
    train = df[df.split == "train"]
    test = df[df.split == "test"]
    tasa_train = float(train.pCR.mean())
    tasa_test = float(test.pCR.mean())
    faltantes = calidad[(calidad.tabla == "patients") & (calidad.ausentes > 0)].sort_values("ausentes_pct", ascending=False)
    top_num = assoc_num.head(8)
    top_shift = shift_num.head(8)
    top_cat_vars = assoc_cat.drop_duplicates("variable").head(6)
    cohort_rates = train.groupby("dataset").pCR.agg(["size", "mean"])
    menopause_missing = (
        faltantes_cohorte[faltantes_cohorte.variable == "menopause"]
        .set_index("cohorte").ausentes_pct.to_dict()
    )

    def lista_tabla(tabla: pd.DataFrame, columnas: list[str], formatos: dict[str, str] | None = None) -> str:
        formatos = formatos or {}
        lineas = ["| " + " | ".join(columnas) + " |", "|" + "|".join(["---"] * len(columnas)) + "|"]
        for _, fila in tabla.iterrows():
            valores = []
            for columna in columnas:
                valor = fila[columna]
                if columna in formatos and pd.notna(valor):
                    valor = formatos[columna].format(valor)
                valores.append(str(valor))
            lineas.append("| " + " | ".join(valores) + " |")
        return "\n".join(lineas)

    tabla_faltantes = faltantes[["variable", "ausentes", "ausentes_pct"]].copy()
    tabla_faltantes["ausentes_pct"] *= 100
    tabla_assoc = top_num[["variable", "mediana_pcr0", "mediana_pcr1", "smd_pcr1_menos_pcr0"]]
    tabla_shift = top_shift[["variable", "media_train", "media_test", "smd_test_menos_train"]]
    tabla_cat = top_cat_vars[["variable", "cramers_v_variable"]]
    tabla_corr = corr.head(12)[["variable_a", "variable_b", "spearman"]]

    principales_clin = ", ".join(load_clin.head(5).variable.tolist())
    principales_img = ", ".join(load_img.head(5).variable.tolist())
    cohorte_texto = ", ".join(
        f"{idx}: {fila['mean']:.1%} (n={int(fila['size'])})" for idx, fila in cohort_rates.iterrows()
    )

    texto = dedent(f"""
    # EDA profesional de BreastDCEDL

    ## Resumen ejecutivo

    - La unidad de análisis correcta es la paciente: **{len(df):,} pacientes**, {len(train):,} en train y {len(test):,} en test.
    - La clase positiva representa **{tasa_train:.1%} de train**. Predecir siempre pCR=0 daría {1-tasa_train:.1%} de accuracy, por lo que accuracy aislada no será una métrica suficiente.
    - Los datos clínicos están casi completos salvo `menopause` ({df.menopause.isna().mean():.1%} ausente). Los vacíos son "no disponible", no ceros.
    - Cohorte, adquisición y preprocesado están relacionados. La tasa pCR de train también cambia por cohorte ({cohorte_texto}). Esto crea riesgo de que el modelo aprenda protocolo/hospital en vez de respuesta tumoral.
    - Las mayores diferencias numéricas con pCR incluyen coordenadas de recorte y tiempos de adquisición, no solo variables clínicas. Es una señal de posible confusión por cohorte o preprocesado.
    - La PCA se usa como diagnóstico de estructura, no como prueba de separabilidad. No aparece una separación limpia de pCR en los dos primeros componentes; sí existe estructura por cohorte.
    - Las asociaciones que se muestran son descriptivas y se calcularon solo en train. No implican causalidad ni rendimiento predictivo fuera de muestra.

    ## 1. Fuentes y granularidad

    - `patients.csv`: una fila por paciente, objetivo, variables clínicas, cohorte y adquisición.
    - `samples.csv`: una fila por corte; unas diez observaciones correlacionadas por paciente.
    - PNG: PRE, EARLY y LATE son fases temporales alineadas, no canales RGB.
    - `excluded_patients.csv`: dos pacientes excluidas. La causa se conserva como código de error, pero su significado clínico/técnico no está documentado con suficiente claridad.

    Las tablas originales no se modifican. Las características derivadas se guardan aparte y siempre incluyen `pid` para trazabilidad.

    ## 2. Calidad y valores ausentes

    {lista_tabla(tabla_faltantes, ['variable', 'ausentes', 'ausentes_pct'], {'ausentes_pct': '{:.1f}%'})}

    Hallazgos de calidad:

    - `pre` es constante (0) y no aporta información como predictor.
    - `test` y `split` describen la misma partición y deben excluirse de cualquier modelo.
    - `HR_HER2_STATUS`, `TripleNeg`, `HER2pos` y `HRposHER2neg` son representaciones derivadas de HR/HER2. Incluirlas todas duplica información.
    - Las coordenadas de máscara y recorte proceden del preprocesado. Pueden actuar como atajos y no deberían entrar en un primer modelo.
    - La ausencia de `menopause` depende claramente de la fuente: spy1 {menopause_missing.get('spy1', float('nan')):.1%}, spy2 {menopause_missing.get('spy2', float('nan')):.1%} y duke {menopause_missing.get('duke', float('nan')):.1%}. Debe conservarse como información de disponibilidad y tratarse dentro de cada fold.

    ## 3. Composición y desbalance

    Train contiene {int((train.pCR == 0).sum()):,} pacientes pCR=0 y {int((train.pCR == 1).sum()):,} pCR=1. Test contiene {int((test.pCR == 0).sum()):,} y {int((test.pCR == 1).sum()):,}, respectivamente. La tasa positiva de test ({tasa_test:.1%}) se informa solo para describir el conjunto entregado; no se utiliza para seleccionar variables ni decisiones del modelo.

    Los folds mantienen pacientes completas, pero la tasa positiva varía entre {train.groupby('fold').pCR.mean().min():.1%} y {train.groupby('fold').pCR.mean().max():.1%}. El resultado final debe promediar los cinco folds y reportar dispersión.

    ## 4. Variables clínicas y asociación descriptiva con pCR

    Variables numéricas con mayor diferencia estandarizada en train:

    {lista_tabla(tabla_assoc, ['variable', 'mediana_pcr0', 'mediana_pcr1', 'smd_pcr1_menos_pcr0'], {'mediana_pcr0': '{:.3f}', 'mediana_pcr1': '{:.3f}', 'smd_pcr1_menos_pcr0': '{:.3f}'})}

    Variables categóricas con mayor asociación descriptiva:

    {lista_tabla(tabla_cat, ['variable', 'cramers_v_variable'], {'cramers_v_variable': '{:.3f}'})}

    Estas magnitudes sirven para priorizar inspección. No se realizan afirmaciones clínicas ni selección definitiva basadas en este EDA.

    ## 5. Cohortes, adquisición y cambio de distribución

    Variables con mayor diferencia entre test y train:

    {lista_tabla(tabla_shift, ['variable', 'media_train', 'media_test', 'smd_test_menos_train'], {'media_train': '{:.3f}', 'media_test': '{:.3f}', 'smd_test_menos_train': '{:.3f}'})}

    En variables categóricas, los mayores cambios de distribución son:

    {lista_tabla(shift_cat.head(8), ['variable', 'total_variation_train_test'], {'total_variation_train_test': '{:.3f}'})}

    Una |SMD| cercana o superior a 0,2 ya merece revisión; valores mayores no prueban fuga, pero sí cambio de dominio. Cohorte y parámetros de adquisición deben formar parte del análisis de errores y de las métricas estratificadas.

    ## 6. Redundancia y correlación

    Pares con correlación de Spearman absoluta ≥ 0,90:

    {lista_tabla(tabla_corr, ['variable_a', 'variable_b', 'spearman'], {'spearman': '{:.3f}'}) if not tabla_corr.empty else 'No se detectaron pares por encima del umbral.'}

    Para modelos tabulares no conviene introducir simultáneamente variables equivalentes o derivadas. Para la CNN, estas correlaciones sirven principalmente para detectar qué metadatos podrían funcionar como atajos.

    ## 7. PCA clínica y de imagen

    PCA clínica/adquisición: PC1 explica {meta_clin['explained_variance_pc1']:.1%} y PC2 {meta_clin['explained_variance_pc2']:.1%}. Las mayores contribuciones absolutas a PC1 son: {principales_clin}.

    PCA de características de imagen: PC1 explica {meta_img['explained_variance_pc1']:.1%} y PC2 {meta_img['explained_variance_pc2']:.1%}. Las mayores contribuciones absolutas a PC1 son: {principales_img}.

    La PCA trabaja con una fila por paciente, imputación mediana y escalado z-score. La imputación es únicamente para la visualización; cualquier pipeline predictivo deberá ajustar imputadores usando solo el fold de entrenamiento.

    ## 8. Revisión visual de extremos

    Se revisaron automáticamente pacientes extremos por realce, saturación y fracción de fondo:

    ```json
    {json.dumps(atipicos, ensure_ascii=False, indent=2)}
    ```

    Son candidatos para inspección, no errores confirmados. No se elimina ningún paciente automáticamente.

    ## 9. Decisiones para el modelado

    1. Dividir siempre por paciente y usar los folds ya definidos.
    2. Mantener test cerrado durante selección de arquitectura, hiperparámetros y umbral.
    3. Ajustar normalización e imputación exclusivamente en el train de cada fold.
    4. Empezar con un modelo de imagen sin metadatos de adquisición para reducir atajos.
    5. Comparar después con un modelo multimodal limitado a variables clínicas justificadas.
    6. Reportar métricas por paciente, cohorte y subtipo, además del promedio global.
    7. Evaluar calibración, sensibilidad, especificidad, AUC y matriz de confusión; no solo accuracy.
    8. Tratar raza como variable sensible para auditoría de equidad, no como explicación causal.

    ## 10. Limitaciones

    - Es un EDA observacional. Las asociaciones no son causales.
    - Los cortes de una paciente están correlacionados; no se usan como observaciones independientes.
    - Las cohortes se adquirieron y procesaron de manera distinta.
    - Varias unidades clínicas no están explicitadas en el diccionario original; se conservan los nombres sin inventar unidades.
    - La PCA es lineal y resume varianza global; no demuestra que una CNN vaya a clasificar pCR.
    - No se corrigió por comparaciones múltiples porque no se presentan pruebas confirmatorias ni p-valores.

    ## Archivos generados

    - `tablas/patient_image_features.csv`: características visuales por paciente.
    - `tablas/patient_eda_features.csv`: tabla paciente integrada, sin alterar las fuentes.
    - `tablas/feature_policy.csv`: política de uso de variables.
    - `tablas/associations_*` y `split_shift_*`: resultados trazables del análisis.
    - `figuras/`: calidad, clínica, cohortes, PCA, shift y casos extremos.
    """).strip()
    # Las tablas interpoladas rompen el patrón de sangría del literal. Se retira
    # la sangría estructural sin alterar las fuentes ni los valores analizados.
    texto = "\n".join(
        linea[4:] if linea.startswith("    ") else linea
        for linea in texto.splitlines()
    ) + "\n"
    salida.write_text(texto, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=SALIDA_PREDETERMINADA)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--recalcular-imagenes", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers debe ser positivo")

    configurar_estilo()
    tablas = args.salida / "tablas"
    figuras = args.salida / "figuras"
    tablas.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)

    patients = pd.read_csv(DATOS / "metadata" / "patients.csv")
    samples = pd.read_csv(DATOS / "metadata" / "samples.csv")
    calidad, faltantes_cohorte = resumen_calidad(patients, samples)
    pacientes = enriquecer_pacientes(patients, samples)
    imagenes = extraer_features_imagen(
        samples, tablas / "patient_image_features.csv", args.workers, args.recalcular_imagenes
    )
    df = pacientes.merge(imagenes, on="pid", how="left", validate="one_to_one")

    variables_imagen = [c for c in imagenes.columns if c not in {"pid", "n_cortes_imagen"}]
    variables_numericas = list(dict.fromkeys(
        VARIABLES_ADQUISICION
        + VARIABLES_LOCALIZACION
        + ["age", "tumor_z_span", "crop_height", "crop_width", "fov_xy", "log1p_tum_vol", "n_cortes_seleccionados"]
        + variables_imagen
    ))
    variables_shift_cat = VARIABLES_CATEGORICAS_EDA + ["n_xy", "n_times", "post_early", "post_late"]

    resumen_num = tabla_resumen_numerico(df, variables_numericas)
    assoc_num = asociaciones_numericas_train(df, variables_numericas)
    assoc_cat = asociaciones_categoricas_train(df, VARIABLES_CATEGORICAS_EDA)
    shift_num, shift_cat = analizar_shift(df, variables_numericas, variables_shift_cat)
    corr = pares_correlacionados(df[df.split == "train"], variables_numericas + ["HR", "HER2", "TripleNeg", "HER2pos", "HRposHER2neg"])

    pca_clin_vars = [
        "age", "log1p_tum_vol", "n_xy", "n_z", "n_times", "post_early", "post_late",
        "slice_thick", "xy_spacing", "tumor_z_span", "crop_height", "crop_width", "fov_xy",
        "menopause", "race_white", "race_black", "HR", "HER2",
    ]
    pca_img_vars = variables_imagen
    scores_clin, load_clin, meta_clin = pca_numpy(df, pca_clin_vars, "clin")
    scores_img, load_img, meta_img = pca_numpy(df, pca_img_vars, "img")

    calidad.to_csv(tablas / "data_quality.csv", index=False)
    faltantes_cohorte.to_csv(tablas / "missingness_by_cohort.csv", index=False)
    resumen_num.to_csv(tablas / "numeric_summary.csv", index=False)
    assoc_num.to_csv(tablas / "associations_numeric_train.csv", index=False)
    assoc_cat.to_csv(tablas / "associations_categorical_train.csv", index=False)
    shift_num.to_csv(tablas / "split_shift_numeric.csv", index=False)
    shift_cat.to_csv(tablas / "split_shift_categorical.csv", index=False)
    corr.to_csv(tablas / "high_correlations_train.csv", index=False)
    scores_clin.to_csv(tablas / "pca_clinical_scores.csv", index=False)
    load_clin.to_csv(tablas / "pca_clinical_loadings.csv", index=False)
    scores_img.to_csv(tablas / "pca_image_scores.csv", index=False)
    load_img.to_csv(tablas / "pca_image_loadings.csv", index=False)
    df.to_csv(tablas / "patient_eda_features.csv", index=False)
    politica_variables(df).to_csv(tablas / "feature_policy.csv", index=False)

    figura_calidad(df, faltantes_cohorte, figuras / "01_calidad_y_composicion.png")
    figura_clinica(df, figuras / "02_clinica_y_pcr.png")
    figura_cohortes(df, figuras / "03_cohortes_y_adquisicion.png")
    figura_pca(scores_clin, meta_clin, scores_img, meta_img, figuras / "04_pca.png")
    figura_shift_asociaciones(shift_num, assoc_num, figuras / "05_shift_y_asociaciones.png")
    atipicos = figura_atipicos(df, samples, figuras / "06_casos_atipicos.png")

    metadatos = {
        "filas_patients": int(len(patients)),
        "filas_samples": int(len(samples)),
        "imagenes": int(len(samples) * 3),
        "fuentes_sha256": {
            "patients.csv": sha256(DATOS / "metadata" / "patients.csv"),
            "samples.csv": sha256(DATOS / "metadata" / "samples.csv"),
            "statistics.json": sha256(DATOS / "metadata" / "statistics.json"),
        },
        "entorno": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "asociaciones_pcr": "solo train",
        "uso_test": "solo shift de covariables; no selección por pCR",
        "pca_clinica": meta_clin,
        "pca_imagen": meta_img,
        "casos_atipicos": atipicos,
    }
    (args.salida / "metadata_eda.json").write_text(json.dumps(metadatos, ensure_ascii=False, indent=2), encoding="utf-8")
    redactar_informe(
        df, calidad, faltantes_cohorte, assoc_num, assoc_cat, shift_num, shift_cat,
        corr, load_clin, meta_clin, load_img, meta_img, atipicos, args.salida / "INFORME_EDA.md",
    )

    print(f"EDA completado: {args.salida.resolve()}")
    print(f"Informe: {(args.salida / 'INFORME_EDA.md').resolve()}")


if __name__ == "__main__":
    main()
