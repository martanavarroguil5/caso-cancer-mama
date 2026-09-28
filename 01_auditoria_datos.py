#!/usr/bin/env python3
"""Auditoría reproducible del dataset BreastDCEDL.

Comprueba la unidad estadística (paciente), etiquetas, particiones, ficheros e
intensidades de las tres fases DCE. También genera figuras de resumen y ejemplos.

Uso:

    python 01_auditoria_datos.py
    python 01_auditoria_datos.py --rapido       # inspecciona hasta 500 cortes
    python 01_auditoria_datos.py --max-cortes 100
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "breastdcedl"
SALIDA_PREDETERMINADA = RAIZ / "resultados" / "01_auditoria"
os.environ.setdefault("MPLCONFIGDIR", str(RAIZ / ".codex_tmp" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

FASES = ("PRE", "EARLY", "LATE")
COLUMNAS_RUTA = ("path_pre", "path_early", "path_late")


def hist_quantile(hist: np.ndarray, q: float) -> int:
    """Cuantil entero a partir de un histograma de niveles 0..255."""
    objetivo = q * max(int(hist.sum()) - 1, 0)
    return int(np.searchsorted(np.cumsum(hist), objetivo, side="right"))


def registro(nombre: str, condicion: bool, detalle: str) -> dict:
    return {"comprobacion": nombre, "ok": bool(condicion), "detalle": detalle}


def auditar_metadata(samples: pd.DataFrame, patients: pd.DataFrame) -> list[dict]:
    controles: list[dict] = []
    claves_samples = ["sample_id", "patient_id", "split", "pCR", "fold", *COLUMNAS_RUTA]
    claves_patients = ["pid", "pCR", "split"]

    controles.append(registro(
        "Sin nulos en campos esenciales de samples.csv",
        not samples[claves_samples].isna().any().any(),
        f"{int(samples[claves_samples].isna().sum().sum())} valores nulos",
    ))
    controles.append(registro(
        "Sin identificadores de corte duplicados",
        not samples.sample_id.duplicated().any(),
        f"{int(samples.sample_id.duplicated().sum())} duplicados",
    ))
    controles.append(registro(
        "Una fila por paciente en patients.csv",
        not patients.pid.duplicated().any(),
        f"{int(patients.pid.duplicated().sum())} duplicados",
    ))
    controles.append(registro(
        "Etiquetas binarias",
        set(samples.pCR.unique()) <= {0, 1} and set(patients.pCR.unique()) <= {0, 1},
        f"samples={sorted(samples.pCR.unique().tolist())}; patients={sorted(patients.pCR.unique().tolist())}",
    ))
    controles.append(registro(
        "Etiqueta constante dentro de cada paciente",
        bool((samples.groupby("patient_id").pCR.nunique() == 1).all()),
        f"{int((samples.groupby('patient_id').pCR.nunique() > 1).sum())} pacientes inconsistentes",
    ))
    controles.append(registro(
        "Cada paciente pertenece a un único split",
        bool((samples.groupby("patient_id").split.nunique() == 1).all()),
        f"{int((samples.groupby('patient_id').split.nunique() > 1).sum())} pacientes con fuga",
    ))
    ids_samples = set(samples.patient_id)
    ids_patients = set(patients.pid)
    controles.append(registro(
        "Los pacientes coinciden entre ambos CSV",
        ids_samples == ids_patients,
        f"solo samples={len(ids_samples - ids_patients)}; solo patients={len(ids_patients - ids_samples)}",
    ))

    unidos = samples[["sample_id", "patient_id", "split", "pCR"]].merge(
        patients[["pid", "split", "pCR"]], left_on="patient_id", right_on="pid",
        suffixes=("_sample", "_patient"), validate="many_to_one",
    )
    consistentes = (
        (unidos.split_sample == unidos.split_patient)
        & (unidos.pCR_sample == unidos.pCR_patient)
    )
    controles.append(registro(
        "Split y etiqueta coinciden entre samples y patients",
        bool(consistentes.all()),
        f"{int((~consistentes).sum())} cortes inconsistentes",
    ))

    folds_train = {int(valor) for valor in samples.loc[samples.split == "train", "fold"].unique()}
    folds_test = {int(valor) for valor in samples.loc[samples.split == "test", "fold"].unique()}
    controles.append(registro(
        "Folds válidos y separados por paciente",
        folds_train == {0, 1, 2, 3, 4}
        and folds_test == {-1}
        and bool((samples.groupby("patient_id").fold.nunique() == 1).all()),
        f"train={sorted(folds_train)}; test={sorted(folds_test)}",
    ))
    return controles


def auditar_imagenes(samples: pd.DataFrame, max_cortes: int | None) -> tuple[dict, list[dict]]:
    filas = samples if max_cortes is None else samples.head(max_cortes)
    histogramas = {fase: np.zeros(256, dtype=np.int64) for fase in FASES}
    hist_realce = np.zeros(511, dtype=np.int64)  # diferencias -255..255
    modos: Counter[str] = Counter()
    formatos: Counter[str] = Counter()
    tamanos: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    faltantes: list[str] = []
    errores: list[str] = []
    sufijos_incorrectos: list[str] = []
    por_paciente = defaultdict(lambda: np.zeros(4, dtype=np.float64))

    for numero, fila in enumerate(filas.itertuples(index=False), start=1):
        canales: list[np.ndarray] = []
        for fase, columna in zip(FASES, COLUMNAS_RUTA):
            ruta_relativa = getattr(fila, columna)
            ruta = DATOS / ruta_relativa
            if not ruta.is_file():
                faltantes.append(ruta_relativa)
                continue
            if not Path(ruta_relativa).stem.endswith(f"_{fase}"):
                sufijos_incorrectos.append(ruta_relativa)
            try:
                with Image.open(ruta) as png:
                    modos[png.mode] += 1
                    formatos[str(png.format)] += 1
                    tamanos[f"{png.width}x{png.height}"] += 1
                    canal = np.asarray(png)
                dtypes[str(canal.dtype)] += 1
                if canal.ndim != 2:
                    raise ValueError(f"se esperaban 2 dimensiones, recibidas {canal.shape}")
                if canal.dtype != np.uint8:
                    raise ValueError(f"se esperaba uint8, recibido {canal.dtype}")
                histogramas[fase] += np.bincount(canal.ravel(), minlength=256)
                canales.append(canal)
            except Exception as exc:  # continúa para poder listar todos los problemas
                errores.append(f"{ruta_relativa}: {exc}")

        if len(canales) == 3:
            pre, early, late = canales
            diferencia = early.astype(np.int16) - pre.astype(np.int16)
            hist_realce += np.bincount((diferencia + 255).ravel(), minlength=511)
            tejido = pre > 25  # aproximadamente PRE > 0,1 tras normalizar
            if tejido.any():
                acumulado = por_paciente[fila.patient_id]
                acumulado[:3] += [pre[tejido].sum(), early[tejido].sum(), late[tejido].sum()]
                acumulado[3] += int(tejido.sum())

        if numero % 1000 == 0 or numero == len(filas):
            print(f"\rImágenes auditadas: {numero:,}/{len(filas):,} cortes", end="", flush=True)
    print()

    rangos = []
    for fase in FASES:
        hist = histogramas[fase]
        valores = np.arange(256, dtype=np.float64)
        n = int(hist.sum())
        media = float((hist * valores).sum() / n) if n else float("nan")
        varianza = float((hist * (valores - media) ** 2).sum() / n) if n else float("nan")
        rangos.append({
            "fase": fase,
            "min": hist_quantile(hist, 0.0),
            "p01": hist_quantile(hist, 0.01),
            "p05": hist_quantile(hist, 0.05),
            "mediana": hist_quantile(hist, 0.5),
            "p95": hist_quantile(hist, 0.95),
            "p99": hist_quantile(hist, 0.99),
            "max": hist_quantile(hist, 1.0),
            "media": media,
            "desviacion": float(np.sqrt(varianza)),
            "fraccion_ceros": float(hist[0] / n) if n else float("nan"),
        })

    medias_paciente = []
    for patient_id, acumulado in por_paciente.items():
        if acumulado[3] > 0:
            medias_paciente.append({
                "patient_id": patient_id,
                "PRE": acumulado[0] / acumulado[3],
                "EARLY": acumulado[1] / acumulado[3],
                "LATE": acumulado[2] / acumulado[3],
            })
    medias_paciente_df = pd.DataFrame(medias_paciente)

    resumen = {
        "cortes_inspeccionados": int(len(filas)),
        "imagenes_inspeccionadas": int(sum(formatos.values())),
        "faltantes": len(faltantes),
        "errores_lectura": len(errores),
        "sufijos_incorrectos": len(sufijos_incorrectos),
        "formatos": dict(formatos),
        "modos": dict(modos),
        "tamanos": dict(tamanos),
        "dtypes": dict(dtypes),
        "rangos_uint8": rangos,
        "rangos_normalizados": [
            {clave: (valor / 255 if clave not in {"fase", "fraccion_ceros"} else valor)
             for clave, valor in fila.items()}
            for fila in rangos
        ],
        "realce_early_menos_pre_uint8": {
            "min": int(np.flatnonzero(hist_realce)[0] - 255) if hist_realce.any() else None,
            "mediana": int(hist_quantile(hist_realce, 0.5) - 255) if hist_realce.any() else None,
            "max": int(np.flatnonzero(hist_realce)[-1] - 255) if hist_realce.any() else None,
        },
        "pacientes_con_media_early_mayor_pre": (
            int((medias_paciente_df.EARLY > medias_paciente_df.PRE).sum())
            if not medias_paciente_df.empty else 0
        ),
        "pacientes_con_washout_medio": (
            int((medias_paciente_df.LATE < medias_paciente_df.EARLY).sum())
            if not medias_paciente_df.empty else 0
        ),
        "pacientes_evaluados_en_intensidad": int(len(medias_paciente_df)),
        "primeros_faltantes": faltantes[:10],
        "primeros_errores": errores[:10],
        "primeros_sufijos_incorrectos": sufijos_incorrectos[:10],
    }
    return resumen, rangos


def figura_resumen(samples: pd.DataFrame, patients: pd.DataFrame, rangos: list[dict], salida: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    tabla_clases = patients.groupby(["split", "pCR"]).size().unstack(fill_value=0).reindex(["train", "test"])
    tabla_clases.plot(kind="bar", stacked=True, color=["#4c78a8", "#f58518"], ax=axes[0, 0])
    axes[0, 0].set_title("Pacientes por clase y split")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Pacientes")
    axes[0, 0].tick_params(axis="x", rotation=0)
    axes[0, 0].legend(["pCR=0", "pCR=1"], frameon=False)
    for contenedor in axes[0, 0].containers:
        axes[0, 0].bar_label(contenedor, label_type="center", color="white", fontsize=9)

    train = samples[samples.split == "train"]
    folds = (
        train.drop_duplicates("patient_id")
        .groupby("fold")
        .agg(pacientes=("patient_id", "size"), pcr=("pCR", "mean"))
        .join(train.groupby("fold").size().rename("cortes"))
    )
    axes[0, 1].bar(folds.index.astype(str), folds.pacientes, color="#54a24b")
    axes[0, 1].set_title("Pacientes por fold de train")
    axes[0, 1].set_xlabel("Fold")
    axes[0, 1].set_ylabel("Pacientes")
    axes[0, 1].set_ylim(0, folds.pacientes.max() * 1.2)
    for x, fila in folds.iterrows():
        axes[0, 1].text(str(x), fila.pacientes + 3, f"{int(fila.pacientes)}\npCR {fila.pcr:.1%}", ha="center", fontsize=9)

    cortes_paciente = samples.groupby(["patient_id", "split"]).size().rename("cortes").reset_index()
    bins = np.arange(cortes_paciente.cortes.min() - 0.5, cortes_paciente.cortes.max() + 1.5)
    for split, color in [("train", "#4c78a8"), ("test", "#e45756")]:
        axes[1, 0].hist(
            cortes_paciente.loc[cortes_paciente.split == split, "cortes"], bins=bins,
            alpha=0.65, label=split, color=color,
        )
    axes[1, 0].set_title("Cortes por paciente")
    axes[1, 0].set_xlabel("Número de cortes")
    axes[1, 0].set_ylabel("Pacientes")
    axes[1, 0].legend(frameon=False)

    for i, fila in enumerate(rangos):
        axes[1, 1].plot([fila["p01"], fila["p99"]], [i, i], color="#bab0ac", linewidth=8, solid_capstyle="butt")
        axes[1, 1].plot([fila["p05"], fila["p95"]], [i, i], color="#72b7b2", linewidth=14, solid_capstyle="butt")
        axes[1, 1].plot(fila["mediana"], i, marker="|", markersize=18, color="black", markeredgewidth=2)
    axes[1, 1].set_yticks(range(len(rangos)), [fila["fase"] for fila in rangos])
    axes[1, 1].set_xlim(0, 255)
    axes[1, 1].set_title("Intensidad global por fase")
    axes[1, 1].set_xlabel("Nivel de gris uint8 (barra: p1–p99; centro: p5–p95)")
    axes[1, 1].grid(axis="x", alpha=0.25)

    fig.suptitle("Auditoría de BreastDCEDL", fontsize=16, fontweight="bold")
    fig.savefig(salida, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cargar_tres_fases(fila) -> np.ndarray:
    canales = []
    for columna in COLUMNAS_RUTA:
        with Image.open(DATOS / getattr(fila, columna)) as png:
            canales.append(np.asarray(png.convert("L"), dtype=np.float32) / 255.0)
    return np.stack(canales)


def figura_ejemplos(samples: pd.DataFrame, salida: Path) -> list[dict]:
    seleccion = []
    for etiqueta in (0, 1):
        grupo = samples[(samples.split == "train") & (samples.pCR == etiqueta)]
        paciente = sorted(grupo.patient_id.unique())[0]
        cortes = grupo[grupo.patient_id == paciente].sort_values("slice_index")
        seleccion.append(cortes.iloc[len(cortes) // 2])

    fig, axes = plt.subplots(2, 5, figsize=(16, 7), constrained_layout=True)
    usados = []
    for fila_indice, fila in enumerate(seleccion):
        imagen = cargar_tres_fases(fila)
        realce = imagen[1] - imagen[0]
        compuesto = imagen.transpose(1, 2, 0)
        for columna, (plano, titulo, cmap, limites) in enumerate([
            (imagen[0], "PRE", "gray", (0, 1)),
            (imagen[1], "EARLY", "gray", (0, 1)),
            (imagen[2], "LATE", "gray", (0, 1)),
            (realce, "EARLY − PRE", "magma", (0, max(float(realce.max()), 1e-6))),
            (compuesto, "Compuesto visual", None, (None, None)),
        ]):
            if cmap is None:
                axes[fila_indice, columna].imshow(plano)
            else:
                axes[fila_indice, columna].imshow(plano, cmap=cmap, vmin=limites[0], vmax=limites[1])
            axes[fila_indice, columna].set_title(titulo)
            axes[fila_indice, columna].axis("off")
        axes[fila_indice, 0].text(
            -0.04, 0.5,
            f"pCR={int(fila.pCR)}\n{fila.patient_id}\nz={int(fila.slice_index)}",
            rotation=90, va="center", ha="right", transform=axes[fila_indice, 0].transAxes,
            fontsize=10, fontweight="bold",
        )
        usados.append({
            "sample_id": fila.sample_id,
            "patient_id": fila.patient_id,
            "pCR": int(fila.pCR),
            "slice_index": int(fila.slice_index),
        })
    fig.suptitle("Ejemplos: las tres fases temporales y el realce", fontsize=15, fontweight="bold")
    fig.savefig(salida, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return usados


def escribir_informe(
    samples: pd.DataFrame,
    patients: pd.DataFrame,
    controles: list[dict],
    imagenes: dict,
    ejemplos: list[dict],
    salida: Path,
) -> None:
    clases_paciente = patients.groupby(["split", "pCR"]).size().unstack(fill_value=0)
    clases_corte = samples.groupby(["split", "pCR"]).size().unstack(fill_value=0)
    cortes = samples.groupby("patient_id").size()
    train_pacientes = patients[patients.split == "train"]
    tasa_train = float(train_pacientes.pCR.mean())
    tasa_test = float(patients.loc[patients.split == "test", "pCR"].mean())
    ratio_train = int(clases_paciente.loc["train", 0]) / int(clases_paciente.loc["train", 1])
    pacientes_fold = samples[samples.split == "train"].drop_duplicates("patient_id")
    tasas_fold = pacientes_fold.groupby("fold").pCR.mean()
    comprobaciones = "\n".join(
        f"- [{'x' if item['ok'] else ' '}] {item['comprobacion']}: {item['detalle']}"
        for item in controles
    )
    rangos = "\n".join(
        f"| {r['fase']} | {r['min']} | {r['p01']} | {r['mediana']} | {r['p99']} | {r['max']} |"
        for r in imagenes["rangos_uint8"]
    )
    texto = f"""# Auditoría inicial de BreastDCEDL

## Unidad estadística y clases

- {len(patients):,} pacientes y {len(samples):,} cortes ({len(samples) * 3:,} PNG).
- Train: {int(clases_paciente.loc['train', 0]):,} pacientes pCR=0 y {int(clases_paciente.loc['train', 1]):,} pCR=1.
- Test: {int(clases_paciente.loc['test', 0]):,} pacientes pCR=0 y {int(clases_paciente.loc['test', 1]):,} pCR=1.
- Por corte, train contiene {int(clases_corte.loc['train', 0]):,} pCR=0 y {int(clases_corte.loc['train', 1]):,} pCR=1.
- La clase positiva representa {tasa_train:.1%} de train y {tasa_test:.1%} de test; en train hay {ratio_train:.2f} negativos por positivo.
- Un clasificador trivial que siempre prediga pCR=0 obtendría {1 - tasa_train:.1%} de accuracy en train.
- Cada paciente aporta entre {int(cortes.min())} y {int(cortes.max())} cortes (mediana {float(cortes.median()):.0f}).
- Los folds no tienen fuga, pero su tasa pCR varía entre {tasas_fold.min():.1%} y {tasas_fold.max():.1%}; conviene promediar los cinco folds.

## Comprobaciones de integridad

{comprobaciones}

- Ficheros ausentes: {imagenes['faltantes']}.
- Errores de lectura: {imagenes['errores_lectura']}.
- Formato/modo/tamaño/dtype: {imagenes['formatos']} / {imagenes['modos']} / {imagenes['tamanos']} / {imagenes['dtypes']}.

## Canales y rangos

Orden de entrada: **PRE, EARLY, LATE**. Son tiempos de adquisición, no RGB.

| Fase | mín | p1 | mediana | p99 | máx |
|---|---:|---:|---:|---:|---:|
{rangos}

Al dividir por 255, el tensor tiene forma `(3, 256, 256)`, tipo `float32` y rango `[0, 1]`.
En tejido (`PRE > 0,1`), EARLY tiene mayor media que PRE en
{imagenes['pacientes_con_media_early_mayor_pre']}/{imagenes['pacientes_evaluados_en_intensidad']} pacientes.
La media de LATE queda por debajo de EARLY (patrón de *washout*) en
{imagenes['pacientes_con_washout_medio']}/{imagenes['pacientes_evaluados_en_intensidad']} pacientes
({imagenes['pacientes_con_washout_medio'] / imagenes['pacientes_evaluados_en_intensidad']:.1%}).
El realce `EARLY - PRE` es la vista que mejor hace visible la entrada de contraste.

## Ejemplos mostrados

{json.dumps(ejemplos, ensure_ascii=False, indent=2)}

## Figuras

- `resumen.png`: clases, folds, cortes por paciente y rangos de intensidad.
- `ejemplos_fases.png`: casos pCR=0 y pCR=1 con PRE, EARLY, LATE y realce.
"""
    salida.write_text(texto, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=SALIDA_PREDETERMINADA)
    parser.add_argument("--rapido", action="store_true", help="equivale a --max-cortes 500")
    parser.add_argument("--max-cortes", type=int, help="limita la auditoría de PNG; omitir para revisar todos")
    args = parser.parse_args()
    if args.max_cortes is not None and args.max_cortes < 1:
        parser.error("--max-cortes debe ser positivo")
    max_cortes = 500 if args.rapido and args.max_cortes is None else args.max_cortes

    args.salida.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(DATOS / "metadata" / "samples.csv")
    patients = pd.read_csv(DATOS / "metadata" / "patients.csv")

    controles = auditar_metadata(samples, patients)
    imagenes, rangos = auditar_imagenes(samples, max_cortes)
    figura_resumen(samples, patients, rangos, args.salida / "resumen.png")
    ejemplos = figura_ejemplos(samples, args.salida / "ejemplos_fases.png")

    resumen = {
        "totales": {
            "pacientes": int(len(patients)),
            "cortes": int(len(samples)),
            "imagenes_esperadas": int(len(samples) * 3),
        },
        "controles_metadata": controles,
        "imagenes": imagenes,
        "ejemplos": ejemplos,
    }
    (args.salida / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    escribir_informe(samples, patients, controles, imagenes, ejemplos, args.salida / "INFORME.md")

    fallos = [control for control in controles if not control["ok"]]
    fallos_totales = len(fallos) + imagenes["faltantes"] + imagenes["errores_lectura"] + imagenes["sufijos_incorrectos"]
    print(f"Resultado: {'OK' if fallos_totales == 0 else 'REVISAR'}")
    print(f"Informe: {(args.salida / 'INFORME.md').resolve()}")
    print(f"Figuras: {(args.salida / 'resumen.png').resolve()} y {(args.salida / 'ejemplos_fases.png').resolve()}")


if __name__ == "__main__":
    main()
