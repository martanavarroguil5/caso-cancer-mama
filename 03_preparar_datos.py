#!/usr/bin/env python3
"""Prepara y verifica DataLoaders sin fuga para el baseline de imágenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
SALIDA_BASE = RAIZ / "resultados" / "03_preparacion"
os.environ.setdefault("MPLCONFIGDIR", str(RAIZ / ".codex_tmp" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import pipeline_datos as pdatos


def sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def cargar_o_calcular_estadisticas(
    ruta: Path,
    particiones: pdatos.Particiones,
    workers: int,
    recalcular: bool,
) -> pdatos.EstadisticasImagen:
    hash_samples = sha256(pdatos.DATOS / "metadata" / "samples.csv")
    if ruta.exists() and not recalcular:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        if (
            datos.get("fold_validation") == particiones.fold_validation
            and datos.get("samples_sha256") == hash_samples
        ):
            valores = datos["estadisticas"]
            valores["mean_channels"] = tuple(valores["mean_channels"])
            valores["std_channels"] = tuple(valores["std_channels"])
            print(f"Reutilizando estadísticas de {ruta}")
            return pdatos.EstadisticasImagen(**valores)

    estadisticas = pdatos.calcular_estadisticas(
        particiones.train,
        batch_size=32,
        workers=workers,
    )
    ruta.write_text(
        json.dumps(
            {
                "fold_validation": particiones.fold_validation,
                "samples_sha256": hash_samples,
                "estadisticas": asdict(estadisticas),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return estadisticas


def resumen_particion(filas: pd.DataFrame, rol: str) -> dict:
    pacientes = filas.drop_duplicates("patient_id")
    return {
        "rol": rol,
        "cortes": int(len(filas)),
        "pacientes": int(filas.patient_id.nunique()),
        "cortes_pcr0": int((filas.pCR == 0).sum()),
        "cortes_pcr1": int((filas.pCR == 1).sum()),
        "pacientes_pcr0": int((pacientes.pCR == 0).sum()),
        "pacientes_pcr1": int((pacientes.pCR == 1).sum()),
        "tasa_pcr_pacientes": float(pacientes.pCR.mean()),
    }


def crear_manifiesto(
    particiones: pdatos.Particiones,
    patients: pd.DataFrame,
    salida: Path,
) -> pd.DataFrame:
    registros = []
    for rol, filas in (
        ("train", particiones.train),
        ("validation", particiones.validation),
        ("test_closed", particiones.test),
    ):
        agrupado = filas.groupby("patient_id").agg(
            n_cortes=("sample_id", "size"),
            fold=("fold", "first"),
            pCR=("pCR", "first"),
        ).reset_index()
        agrupado["role"] = rol
        registros.append(agrupado)
    manifiesto = pd.concat(registros, ignore_index=True)
    manifiesto = manifiesto.merge(
        patients[["pid", "dataset"]],
        left_on="patient_id",
        right_on="pid",
        how="left",
        validate="one_to_one",
    ).drop(columns="pid")
    # La etiqueta de test no debe circular por el pipeline de desarrollo.
    manifiesto.loc[manifiesto.role == "test_closed", "pCR"] = pd.NA
    manifiesto = manifiesto[["patient_id", "role", "fold", "dataset", "n_cortes", "pCR"]]
    manifiesto.sort_values(["role", "patient_id"]).to_csv(salida, index=False)
    return manifiesto


def verificar_lote(
    batch: dict,
    batch_size: int,
    normalizacion: str,
) -> dict:
    imagenes = batch["image"]
    etiquetas = batch["label"]
    if imagenes.ndim != 4 or imagenes.shape[1:] != (3, 256, 256):
        raise RuntimeError(f"Forma de lote incorrecta: {tuple(imagenes.shape)}")
    if imagenes.shape[0] > batch_size:
        raise RuntimeError("El lote supera batch_size")
    if imagenes.dtype != torch.float32:
        raise RuntimeError(f"dtype incorrecto: {imagenes.dtype}")
    if not torch.isfinite(imagenes).all():
        raise RuntimeError("El lote contiene NaN o infinitos")
    if not set(etiquetas.unique().tolist()) <= {0.0, 1.0}:
        raise RuntimeError("El lote contiene etiquetas no binarias")
    if len(batch["patient_id"]) != len(batch["sample_id"]):
        raise RuntimeError("Metadatos desalineados")
    minimo = float(imagenes.min())
    maximo = float(imagenes.max())
    if normalizacion == "none" and (minimo < 0 or maximo > 1):
        raise RuntimeError(f"Valores fuera de [0,1]: {minimo}, {maximo}")
    return {
        "shape": list(imagenes.shape),
        "dtype": str(imagenes.dtype),
        "min": minimo,
        "max": maximo,
        "n_positivos": int((etiquetas == 1).sum()),
        "n_negativos": int((etiquetas == 0).sum()),
        "patient_ids_unicos_en_lote": int(len(set(batch["patient_id"]))),
    }


def desnormalizar(imagen: torch.Tensor, normalizacion: str, stats: pdatos.EstadisticasImagen) -> torch.Tensor:
    if normalizacion == "shared":
        return imagen * stats.shared_std + stats.shared_mean
    return imagen


def figura_lote(
    batch: dict,
    normalizacion: str,
    stats: pdatos.EstadisticasImagen,
    salida: Path,
) -> None:
    cantidad = min(4, len(batch["image"]))
    fig, axes = plt.subplots(cantidad, 4, figsize=(12, 3 * cantidad), constrained_layout=True)
    axes = np.atleast_2d(axes)
    titulos = ("PRE", "EARLY", "LATE", "EARLY − PRE")
    for fila in range(cantidad):
        imagen = desnormalizar(batch["image"][fila].detach().cpu(), normalizacion, stats).numpy()
        realce = imagen[1] - imagen[0]
        planos = (imagen[0], imagen[1], imagen[2], realce)
        for columna, (plano, titulo) in enumerate(zip(planos, titulos)):
            if columna < 3:
                axes[fila, columna].imshow(plano, cmap="gray", vmin=0, vmax=1)
            else:
                limite = max(float(np.abs(plano).max()), 1e-6)
                axes[fila, columna].imshow(plano, cmap="coolwarm", vmin=-limite, vmax=limite)
            axes[fila, columna].set_title(titulo if fila == 0 else "")
            axes[fila, columna].axis("off")
        axes[fila, 0].text(
            -0.06,
            0.5,
            f"{batch['patient_id'][fila]}\n{batch['sample_id'][fila]}\npCR={int(batch['label'][fila])}",
            rotation=90,
            va="center",
            ha="right",
            transform=axes[fila, 0].transAxes,
            fontsize=8,
        )
    fig.suptitle("Lote de entrenamiento: fases alineadas y realce", fontsize=15, fontweight="bold")
    fig.savefig(salida, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def escribir_informe(
    salida: Path,
    fold: int,
    normalizacion: str,
    res_particiones: list[dict],
    stats: pdatos.EstadisticasImagen,
    pos_weight_cortes: float,
    pos_weight_pacientes: float,
    smoke_train: dict,
    smoke_validation: dict,
) -> None:
    particiones_md = "\n".join(
        f"| {r['rol']} | {r['pacientes']:,} | {r['cortes']:,} | {r['pacientes_pcr0']:,} | {r['pacientes_pcr1']:,} | {r['tasa_pcr_pacientes']:.1%} |"
        for r in res_particiones
    )
    canales = "\n".join(
        f"| {fase} | {media:.6f} | {desv:.6f} |"
        for fase, media, desv in zip(pdatos.FASES, stats.mean_channels, stats.std_channels)
    )
    texto = f"""# Preparación de datos — fold {fold}

## Decisiones

- La partición se realiza por paciente, nunca por corte.
- Fold {fold} se reserva para validación; los otros cuatro folds forman train.
- Test permanece cerrado y no se crea un DataLoader de test.
- El baseline usa solo imágenes. Los vacíos clínicos no entran todavía en el modelo.
- Los PNG se convierten a `float32` en `[0,1]`; no se aplica normalización de ImageNet.
- Normalización activa: `{normalizacion}`.
- El único aumento inicial es volteo horizontal aleatorio, aplicado simultáneamente a PRE, EARLY y LATE.
- Validación no tiene aumentos y mantiene el orden (`shuffle=False`).

## Particiones

| Rol | Pacientes | Cortes | Pacientes pCR=0 | Pacientes pCR=1 | Tasa pCR |
|---|---:|---:|---:|---:|---:|
{particiones_md}

No existe ningún paciente compartido entre train, validación y test.

## Estadísticas calculadas solo con train

| Canal | Media | Desviación |
|---|---:|---:|
{canales}

- Media compartida: `{stats.shared_mean:.6f}`.
- Desviación compartida: `{stats.shared_std:.6f}`.
- Rango observado: `[{stats.min_value:.1f}, {stats.max_value:.1f}]`.
- Cortes utilizados: {stats.slices:,}.

La configuración por defecto conserva `[0,1]`. La opción `shared` aplica una única media y desviación a los tres canales para no romper su relación temporal.

## Desbalance

- `pos_weight` por corte: `{pos_weight_cortes:.6f}`.
- Razón negativa/positiva por paciente: `{pos_weight_pacientes:.6f}`.

Para `BCEWithLogitsLoss` se utilizará el peso por corte calculado únicamente con train. No se combinará inicialmente con sobremuestreo.

## Prueba de humo

- Train: `{json.dumps(smoke_train, ensure_ascii=False)}`
- Validación: `{json.dumps(smoke_validation, ensure_ascii=False)}`

Las imágenes tienen forma `(batch, 3, 256, 256)`, tipo `float32`, etiquetas binarias y metadatos de paciente/corte alineados.

## Uso posterior

```python
import pipeline_datos as pdatos

samples = pdatos.cargar_samples()
particiones = pdatos.crear_particiones(samples, fold_validation={fold})
# Cargar las estadísticas guardadas o calcularlas solo con particiones.train.
# construir_dataloaders(..., incluir_test=False)
```

El test solo debe abrirse una vez que arquitectura, pérdida, umbral y método de agregación por paciente estén cerrados.
"""
    salida.write_text(texto, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--normalizacion", choices=("none", "shared"), default="none")
    parser.add_argument("--recalcular-estadisticas", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size debe ser positivo y workers no negativo")

    pdatos.fijar_semilla(args.semilla)
    salida = SALIDA_BASE / f"fold_{args.fold}"
    salida.mkdir(parents=True, exist_ok=True)
    samples = pdatos.cargar_samples()
    patients = pdatos.cargar_patients()
    particiones = pdatos.crear_particiones(samples, fold_validation=args.fold)
    workers_efectivos = args.workers
    try:
        stats = cargar_o_calcular_estadisticas(
            salida / "stats_train.json",
            particiones,
            workers=workers_efectivos,
            recalcular=args.recalcular_estadisticas,
        )
    except PermissionError:
        if workers_efectivos == 0:
            raise
        workers_efectivos = 0
        print("El entorno no permite procesos auxiliares; se continúa con workers=0")
        stats = cargar_o_calcular_estadisticas(
            salida / "stats_train.json",
            particiones,
            workers=workers_efectivos,
            recalcular=args.recalcular_estadisticas,
        )

    loaders = pdatos.construir_dataloaders(
        particiones,
        stats,
        batch_size=args.batch_size,
        workers=workers_efectivos,
        semilla=args.semilla,
        normalizacion=args.normalizacion,
        incluir_test=False,
    )
    if "test" in loaders:
        raise RuntimeError("Test no debe cargarse durante desarrollo")

    batch_train = next(iter(loaders["train"]))
    batch_validation = next(iter(loaders["validation"]))
    smoke_train = verificar_lote(batch_train, args.batch_size, args.normalizacion)
    smoke_validation = verificar_lote(batch_validation, args.batch_size, args.normalizacion)
    figura_lote(batch_train, args.normalizacion, stats, salida / "lote_train.png")
    crear_manifiesto(particiones, patients, salida / "manifiesto_particiones.csv")

    res_particiones = [
        resumen_particion(particiones.train, "train"),
        resumen_particion(particiones.validation, "validation"),
        resumen_particion(particiones.test, "test_closed"),
    ]
    pos_weight_cortes = pdatos.pos_weight_cortes(particiones.train)
    pos_weight_pacientes = pdatos.pos_weight_pacientes(particiones.train)
    configuracion = {
        "fold_validation": args.fold,
        "batch_size": args.batch_size,
        "workers_requested": args.workers,
        "workers_effective": workers_efectivos,
        "semilla": args.semilla,
        "normalizacion": args.normalizacion,
        "augment_train": "RandomHorizontalFlipJoint(p=0.5)",
        "augment_validation": None,
        "test_loader_created": False,
        "particiones": res_particiones,
        "estadisticas_train": asdict(stats),
        "pos_weight_cortes": pos_weight_cortes,
        "pos_weight_pacientes": pos_weight_pacientes,
        "smoke_train": smoke_train,
        "smoke_validation": smoke_validation,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    (salida / "configuracion.json").write_text(
        json.dumps(configuracion, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    escribir_informe(
        salida / "INFORME_PREPARACION.md",
        args.fold,
        args.normalizacion,
        res_particiones,
        stats,
        pos_weight_cortes,
        pos_weight_pacientes,
        smoke_train,
        smoke_validation,
    )

    print("Preparación validada correctamente")
    print(f"Train: {len(particiones.train):,} cortes / {particiones.train.patient_id.nunique():,} pacientes")
    print(f"Validación: {len(particiones.validation):,} cortes / {particiones.validation.patient_id.nunique():,} pacientes")
    print("Test: cerrado; no se creó DataLoader")
    print(f"Salida: {salida.resolve()}")


if __name__ == "__main__":
    main()
