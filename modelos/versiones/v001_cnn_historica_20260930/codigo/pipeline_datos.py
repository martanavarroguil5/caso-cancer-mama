"""Pipeline reutilizable de datos para BreastDCEDL.

Principios:

- la partición se hace por paciente;
- test no se carga durante desarrollo salvo petición explícita;
- PRE, EARLY y LATE se transforman siempre juntas;
- la escala base es float32 en [0, 1], sin normalización de ImageNet;
- cualquier estadística se calcula solo con el train del fold activo.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "breastdcedl"
FASES = ("PRE", "EARLY", "LATE")
COLUMNAS_RUTA = ("path_pre", "path_early", "path_late")


@dataclass(frozen=True)
class Particiones:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    fold_validation: int


@dataclass(frozen=True)
class EstadisticasImagen:
    mean_channels: tuple[float, float, float]
    std_channels: tuple[float, float, float]
    shared_mean: float
    shared_std: float
    min_value: float
    max_value: float
    slices: int
    pixels_per_channel: int


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def inicializar_worker(worker_id: int) -> None:
    del worker_id
    semilla = torch.initial_seed() % (2**32)
    random.seed(semilla)
    np.random.seed(semilla)


def cargar_samples(raiz: Path | str = DATOS) -> pd.DataFrame:
    return pd.read_csv(Path(raiz) / "metadata" / "samples.csv")


def cargar_patients(raiz: Path | str = DATOS) -> pd.DataFrame:
    return pd.read_csv(Path(raiz) / "metadata" / "patients.csv")


def crear_particiones(samples: pd.DataFrame, fold_validation: int = 0) -> Particiones:
    if fold_validation not in range(5):
        raise ValueError(f"fold_validation debe estar entre 0 y 4; recibido {fold_validation}")

    requeridas = {"sample_id", "patient_id", "split", "fold", "pCR", *COLUMNAS_RUTA}
    faltan = requeridas - set(samples.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en samples.csv: {sorted(faltan)}")
    if samples.sample_id.duplicated().any():
        raise ValueError("Hay sample_id duplicados")
    if (samples.groupby("patient_id").pCR.nunique() != 1).any():
        raise ValueError("La etiqueta no es constante dentro de alguna paciente")
    if (samples.groupby("patient_id").split.nunique() != 1).any():
        raise ValueError("Alguna paciente aparece en más de un split")

    desarrollo = samples[samples.split == "train"]
    train = desarrollo[desarrollo.fold != fold_validation].copy().reset_index(drop=True)
    validation = desarrollo[desarrollo.fold == fold_validation].copy().reset_index(drop=True)
    test = samples[samples.split == "test"].copy().reset_index(drop=True)

    conjuntos = {
        "train": set(train.patient_id),
        "validation": set(validation.patient_id),
        "test": set(test.patient_id),
    }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        solape = conjuntos[a] & conjuntos[b]
        if solape:
            raise RuntimeError(f"Fuga entre {a} y {b}: {len(solape)} pacientes")
    if not train.patient_id.nunique() or not validation.patient_id.nunique() or not test.patient_id.nunique():
        raise RuntimeError("Alguna partición está vacía")

    return Particiones(train, validation, test, fold_validation)


def cargar_imagen(fila, raiz: Path | str = DATOS) -> torch.Tensor:
    planos = []
    raiz = Path(raiz)
    for columna in COLUMNAS_RUTA:
        ruta = raiz / getattr(fila, columna)
        with Image.open(ruta) as png:
            plano = np.asarray(png.convert("L"), dtype=np.float32) / 255.0
        if plano.shape != (256, 256):
            raise ValueError(f"Dimensión inesperada en {ruta}: {plano.shape}")
        planos.append(plano)
    return torch.from_numpy(np.stack(planos))


class RandomHorizontalFlipJoint:
    """Voltea los tres canales a la vez; nunca altera su alineamiento."""

    def __init__(self, probability: float = 0.5):
        if not 0 <= probability <= 1:
            raise ValueError("probability debe estar entre 0 y 1")
        self.probability = probability

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < self.probability:
            return torch.flip(image, dims=(-1,))
        return image


class SharedStandardize:
    """Aplica la misma transformación afín a PRE, EARLY y LATE."""

    def __init__(self, mean: float, std: float):
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError("mean y std deben ser finitos y std positiva")
        self.mean = float(mean)
        self.std = float(std)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std


class ComposeJoint:
    def __init__(self, transforms: list[Callable[[torch.Tensor], torch.Tensor]]):
        self.transforms = transforms

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            image = transform(image)
        return image


class BreastDCESliceDataset(Dataset):
    """Una observación por corte, conservando IDs para evaluación por paciente."""

    def __init__(
        self,
        filas: pd.DataFrame,
        raiz: Path | str = DATOS,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.filas = filas.reset_index(drop=True).copy()
        self.raiz = Path(raiz)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filas)

    def __getitem__(self, indice: int) -> dict:
        fila = self.filas.iloc[indice]
        image = cargar_imagen(fila, self.raiz)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(float(fila.pCR), dtype=torch.float32),
            "patient_id": str(fila.patient_id),
            "sample_id": str(fila.sample_id),
            "slice_index": torch.tensor(int(fila.slice_index), dtype=torch.int64),
        }


def crear_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    semilla: int,
    pin_memory: bool | None = None,
) -> DataLoader:
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size debe ser positivo y workers no negativo")
    generador = torch.Generator().manual_seed(semilla)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        persistent_workers=workers > 0,
        worker_init_fn=inicializar_worker if workers > 0 else None,
        generator=generador,
    )


def calcular_estadisticas(
    filas_train: pd.DataFrame,
    raiz: Path | str = DATOS,
    batch_size: int = 32,
    workers: int = 0,
) -> EstadisticasImagen:
    """Media y desviación exactas usando únicamente los cortes de train."""
    dataset = BreastDCESliceDataset(filas_train, raiz=raiz, transform=None)
    loader = crear_loader(dataset, batch_size, shuffle=False, workers=workers, semilla=0)
    suma = torch.zeros(3, dtype=torch.float64)
    suma2 = torch.zeros(3, dtype=torch.float64)
    minimo = float("inf")
    maximo = float("-inf")
    pixeles = 0

    for numero, batch in enumerate(loader, start=1):
        imagenes = batch["image"].to(torch.float64)
        suma += imagenes.sum(dim=(0, 2, 3))
        suma2 += (imagenes * imagenes).sum(dim=(0, 2, 3))
        pixeles += int(imagenes.shape[0] * imagenes.shape[2] * imagenes.shape[3])
        minimo = min(minimo, float(imagenes.min()))
        maximo = max(maximo, float(imagenes.max()))
        if numero % 50 == 0 or numero == len(loader):
            print(f"\rEstadísticas train: {numero:,}/{len(loader):,} lotes", end="", flush=True)
    print()

    medias = suma / pixeles
    varianzas = torch.clamp(suma2 / pixeles - medias * medias, min=0)
    desviaciones = torch.sqrt(varianzas)
    total_pixeles = pixeles * 3
    shared_mean = float(suma.sum() / total_pixeles)
    shared_var = float(suma2.sum() / total_pixeles - shared_mean**2)
    return EstadisticasImagen(
        mean_channels=tuple(float(x) for x in medias),
        std_channels=tuple(float(x) for x in desviaciones),
        shared_mean=shared_mean,
        shared_std=math_sqrt_nonnegative(shared_var),
        min_value=minimo,
        max_value=maximo,
        slices=len(dataset),
        pixels_per_channel=pixeles,
    )


def math_sqrt_nonnegative(value: float) -> float:
    return float(np.sqrt(max(value, 0.0)))


def construir_transforms(
    entrenamiento: bool,
    normalizacion: str,
    estadisticas: EstadisticasImagen,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    if normalizacion not in {"none", "shared"}:
        raise ValueError("normalizacion debe ser 'none' o 'shared'")
    transforms: list[Callable[[torch.Tensor], torch.Tensor]] = []
    if entrenamiento:
        transforms.append(RandomHorizontalFlipJoint(0.5))
    if normalizacion == "shared":
        transforms.append(SharedStandardize(estadisticas.shared_mean, estadisticas.shared_std))
    return ComposeJoint(transforms) if transforms else None


def construir_dataloaders(
    particiones: Particiones,
    estadisticas: EstadisticasImagen,
    batch_size: int = 32,
    workers: int = 0,
    semilla: int = 42,
    normalizacion: str = "none",
    incluir_test: bool = False,
) -> dict[str, DataLoader]:
    loaders = {
        "train": crear_loader(
            BreastDCESliceDataset(
                particiones.train,
                transform=construir_transforms(True, normalizacion, estadisticas),
            ),
            batch_size=batch_size,
            shuffle=True,
            workers=workers,
            semilla=semilla,
        ),
        "validation": crear_loader(
            BreastDCESliceDataset(
                particiones.validation,
                transform=construir_transforms(False, normalizacion, estadisticas),
            ),
            batch_size=batch_size,
            shuffle=False,
            workers=workers,
            semilla=semilla,
        ),
    }
    if incluir_test:
        loaders["test"] = crear_loader(
            BreastDCESliceDataset(
                particiones.test,
                transform=construir_transforms(False, normalizacion, estadisticas),
            ),
            batch_size=batch_size,
            shuffle=False,
            workers=workers,
            semilla=semilla,
        )
    return loaders


def pos_weight_cortes(filas: pd.DataFrame) -> float:
    positivos = int((filas.pCR == 1).sum())
    negativos = int((filas.pCR == 0).sum())
    if positivos == 0:
        raise ValueError("No hay cortes positivos")
    return negativos / positivos


def pos_weight_pacientes(filas: pd.DataFrame) -> float:
    pacientes = filas.drop_duplicates("patient_id")
    positivos = int((pacientes.pCR == 1).sum())
    negativos = int((pacientes.pCR == 0).sum())
    if positivos == 0:
        raise ValueError("No hay pacientes positivas")
    return negativos / positivos
