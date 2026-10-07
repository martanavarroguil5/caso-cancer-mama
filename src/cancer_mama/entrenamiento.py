#!/usr/bin/env python3
"""Paso 04: CNN, entrenamiento por paciente, comparación OOF e inferencia.

Reutiliza pipeline_datos.py del commit de Marta 69a44eb. El test reservado
ya fue evaluado: aquí se conservan sus resultados y no se vuelve a abrir.
Las variantes descartadas se documentan en EXPERIMENTOS_DESCARTADOS.md.
Las ejecuciones nuevas se guardan aparte del modelo final histórico.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             confusion_matrix, brier_score_loss, log_loss, roc_curve, f1_score)
import sklearn
import PIL
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import datos as pdatos
from .paths import PACKAGE_DIR, PROJECT_ROOT, RESULTS_DIR

RAIZ = PROJECT_ROOT
SALIDA = RESULTS_DIR / "04_entrenamiento"
SCHEMA_VERSION = 1
EPS = 1e-7
METHODS = ("mean", "max", "median")
CLINICAL_COLUMNS = ("age", "tum_vol", "HR", "HER2")
CLINICAL_MODEL_FEATURES = ("age", "log1p_tum_vol", "HR", "HER2")


def configuracion(nombre="raw_rot90"):
    if nombre not in {"base_raw", "raw_rot90", "pool_dropout_wd", "pool_dropout_wd_clinical",
                      "patient_level_clinical"}:
        raise ValueError("Configuración desconocida")
    config = {
        "name": nombre,
        "model": {"representation": "raw", "channels": [24, 48, 96, 160],
                  "hidden": 64, "dropout": 0.20},
        "training": {"epochs": 46, "min_epochs": 12, "patience": 10,
                     "min_delta": 0.001, "batch_size": 64, "lr": 0.0008,
                     "min_lr": 0.00001, "weight_decay": 0.0001,
                     "clip_grad_norm": 5.0, "amp": True, "num_workers": 4,
                     "cpu_threads": 8, "deterministic": True, "review_interval": 10},
        "augmentation": {"hflip": 0.5, "rot90": nombre != "base_raw"},
        "evaluation": {"threshold": 0.5, "checkpoint_metric": "patient_auc"},
    }
    if nombre in {"pool_dropout_wd", "pool_dropout_wd_clinical", "patient_level_clinical"}:
        config["model"].update(pooling="intermedio", dropout=.35)
        config["training"]["weight_decay"] = .001
    if nombre in {"pool_dropout_wd_clinical", "patient_level_clinical"}:
        config["model"]["clinical"] = {
            "raw_columns": list(CLINICAL_COLUMNS),
            "features": list(CLINICAL_MODEL_FEATURES),
            "missing_indicators": True,
        }
    if nombre == "patient_level_clinical":
        # Cada paciente aparece una vez por época y todos sus cortes contribuyen
        # conjuntamente a una única BCE, igual que la unidad de evaluación.
        config["training"].update(loss_unit="patient", patients_per_batch=6)
        config["evaluation"].update(
            threshold_strategy="max_specificity_at_min_sensitivity",
            min_sensitivity=.90,
        )
    return config


class Bloque(nn.Sequential):
    def __init__(self, entrada, salida, primero=False, pooling="original"):
        stride = 2 if pooling == "original" or primero else 1
        capas = [nn.Conv2d(entrada, salida, 3, stride=stride, padding=1, bias=False),
                 nn.BatchNorm2d(salida), nn.ReLU(inplace=True),
                 nn.Conv2d(salida, salida, 3, padding=1, bias=False),
                 nn.BatchNorm2d(salida), nn.ReLU(inplace=True)]
        if primero or pooling == "intermedio":
            capas.append(nn.MaxPool2d(2))
        super().__init__(*capas)


class GlobalMaxPool(nn.Module):
    def forward(self, x):
        # Mantiene el camino determinista de gradiente CUDA del informe.
        return torch.amax(x, dim=(-2, -1), keepdim=True)


class CNN(nn.Module):
    """Ocho convoluciones y 551.913 parámetros; original o pooling intermedio.

    La variante conserva las salidas 64/32/16/8. En los bloques 2-4 sustituye
    la reducción por stride 2 por Conv stride 1 y MaxPool 2 al terminar el bloque.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = deepcopy(config or {})
        self.config = self.config.get("model", self.config)
        if self.config.get("representation", "raw") != "raw":
            raise ValueError("Este paso utiliza las fases originales PRE/EARLY/LATE")
        pooling = self.config.get("pooling", "original")
        if pooling not in {"original", "intermedio"}:
            raise ValueError("Pooling debe ser original o intermedio")
        canales = self.config.get("channels", [24, 48, 96, 160])
        if len(canales) != 4 or any(not isinstance(c, int) or c < 1 for c in canales):
            raise ValueError("Se requieren cuatro anchos de bloque positivos")
        if self.config.get("normalization", "batch") != "batch":
            raise ValueError("La red activa utiliza BatchNorm")
        self.features = nn.Sequential(*[Bloque(a, b, i == 0, pooling)
            for i, (a, b) in enumerate(zip([3, *canales[:-1]], canales))])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = GlobalMaxPool()
        hidden = int(self.config.get("hidden", 64))
        dropout = float(self.config.get("dropout", 0.20))
        clinical = self.config.get("clinical")
        self.clinical_enabled = clinical is not None
        if self.clinical_enabled:
            if clinical.get("features") != list(CLINICAL_MODEL_FEATURES):
                raise ValueError("Variables clínicas incompatibles")
            preprocessing = clinical.get("preprocessing")
            if not isinstance(preprocessing, dict):
                raise ValueError("Falta el preprocesado clínico ajustado en train")
            expected = len(CLINICAL_MODEL_FEATURES)
            for key in ("medians", "means", "stds"):
                values = preprocessing.get(key)
                if not isinstance(values, list) or len(values) != expected or not np.isfinite(values).all():
                    raise ValueError(f"Preprocesado clínico inválido: {key}")
                self.register_buffer("clinical_" + key, torch.tensor(values, dtype=torch.float32))
            if (self.clinical_stds <= 0).any():
                raise ValueError("Las desviaciones clínicas deben ser positivas")
            clinical_width = expected * (2 if clinical.get("missing_indicators", True) else 1)
            self.image_projection = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(canales[-1] * 2, hidden), nn.ReLU(inplace=True),
                nn.Dropout(dropout))
            # Las variables estandarizadas entran directamente en el único logit.
            # Así la rama clínica puede reproducir un baseline lineal sin obligar
            # a la CNN a reaprender HR/HER2 a partir de la imagen.
            self.classifier = nn.Linear(hidden + clinical_width, 1)
        else:
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(canales[-1]*2, hidden),
                nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.apply(self._initialize)
        if self.clinical_enabled and clinical.get("initialization") is not None:
            initialization = clinical["initialization"]
            coefficients = initialization.get("coefficients")
            if (not isinstance(coefficients, list) or len(coefficients) != clinical_width or
                    not np.isfinite(coefficients).all() or
                    not np.isfinite(initialization.get("intercept", np.nan))):
                raise ValueError("Inicialización clínica inválida")
            with torch.no_grad():
                self.classifier.weight.zero_()
                self.classifier.weight[0, hidden:] = torch.tensor(coefficients, dtype=torch.float32)
                self.classifier.bias.fill_(float(initialization["intercept"]))

    @staticmethod
    def _initialize(capa):
        if isinstance(capa, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(capa.weight, mode="fan_in", nonlinearity="relu")
            if capa.bias is not None:
                nn.init.zeros_(capa.bias)
        elif isinstance(capa, nn.BatchNorm2d):
            nn.init.ones_(capa.weight)
            nn.init.zeros_(capa.bias)

    def preprocess(self, x):
        return 2*x - 1

    def preprocess_clinical(self, clinical):
        if clinical is None or clinical.ndim != 2 or clinical.shape[1] != len(CLINICAL_MODEL_FEATURES):
            shape = None if clinical is None else tuple(clinical.shape)
            raise ValueError(f"Esperadas variables clínicas [N,4], recibido {shape}")
        missing = torch.isnan(clinical)
        values = torch.where(missing, self.clinical_medians, clinical)
        values = (values - self.clinical_means) / self.clinical_stds
        if self.config["clinical"].get("missing_indicators", True):
            values = torch.cat((values, missing.to(values.dtype)), dim=1)
        return values

    def forward(self, x, clinical=None):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError(f"Esperado [N,3,256,256], recibido {tuple(x.shape)}")
        z = self.features(self.preprocess(x))
        z = torch.cat((self.avgpool(z).flatten(1), self.maxpool(z).flatten(1)), dim=1)
        if self.clinical_enabled:
            z = torch.cat((self.image_projection(z), self.preprocess_clinical(clinical)), dim=1)
        elif clinical is not None:
            raise ValueError("Este modelo histórico no admite variables clínicas")
        return self.classifier(z)


def numero_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def cargar_train(root, include_clinical=False):
    """Comprueba identidades globales y convierte etiquetas solamente de train."""
    rows, identidades, ids = [], {}, set()
    with (Path(root)/"metadata/samples.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            patient, sample, split = row["patient_id"], row["sample_id"], row["split"]
            if sample in ids:
                raise ValueError("sample_id duplicado")
            ids.add(sample)
            if split not in {"train", "test"}:
                raise ValueError("Split desconocido")
            if patient in identidades and identidades[patient] != split:
                raise ValueError("Fuga por paciente entre splits")
            identidades[patient] = split
            if split == "train":
                rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Train vacío")
    for name in ("pCR", "fold", "slice_index"):
        values = pd.to_numeric(frame[name], errors="raise")
        if values.isna().any() or (values != values.astype(int)).any():
            raise ValueError(f"{name} debe contener enteros")
        frame[name] = values.astype(int)
    if not frame.pCR.isin([0, 1]).all() or not frame.fold.isin(range(5)).all():
        raise ValueError("Train requiere pCR binaria y folds 0-4")
    for column in ("pCR", "fold"):
        if (frame.groupby("patient_id")[column].nunique() != 1).any():
            raise ValueError(f"Paciente con {column} inconsistente")
    required_patient_columns = ["dataset", *(CLINICAL_COLUMNS if include_clinical else ())]
    missing_patient_columns = [column for column in required_patient_columns if column not in frame]
    if missing_patient_columns:
        with (Path(root)/"metadata/patients.csv").open(newline="", encoding="utf-8") as f:
            patients = pd.DataFrame([r for r in csv.DictReader(f) if r["split"] == "train"])
        if patients.pid.duplicated().any():
            raise ValueError("Paciente duplicada en patients.csv")
        unavailable = set(missing_patient_columns) - set(patients.columns)
        if unavailable:
            raise ValueError(f"Faltan variables en patients.csv: {sorted(unavailable)}")
        frame = frame.merge(patients[["pid", *missing_patient_columns]], left_on="patient_id", right_on="pid",
                            how="left", validate="many_to_one").drop(columns="pid")
    if frame.dataset.isna().any():
        raise ValueError("Paciente sin cohorte")
    if include_clinical:
        for column in CLINICAL_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            observed = frame[column].dropna().to_numpy(dtype=float)
            if not np.isfinite(observed).all():
                raise ValueError(f"{column} contiene valores no finitos")
        if (frame[["age", "tum_vol"]].dropna() < 0).any().any():
            raise ValueError("Edad y volumen tumoral no pueden ser negativos")
        for column in ("HR", "HER2"):
            observed = frame[column].dropna()
            if not observed.isin([0, 1]).all():
                raise ValueError(f"{column} debe ser binaria o ausente")
    return frame.reset_index(drop=True)


def clinical_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = set(CLINICAL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan variables clínicas: {sorted(missing)}")
    raw = frame[list(CLINICAL_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    observed = raw.to_numpy(dtype=np.float64)
    if not np.isfinite(observed[~np.isnan(observed)]).all():
        raise ValueError("Las variables clínicas contienen infinitos")
    tumor_volume = raw["tum_vol"].to_numpy(dtype=np.float64)
    if np.any(tumor_volume[np.isfinite(tumor_volume)] < 0):
        raise ValueError("El volumen tumoral no puede ser negativo")
    return np.column_stack([
        raw["age"].to_numpy(dtype=np.float64),
        np.log1p(tumor_volume),
        raw["HR"].to_numpy(dtype=np.float64),
        raw["HER2"].to_numpy(dtype=np.float64),
    ])


def fit_clinical_preprocessing(train: pd.DataFrame) -> dict:
    """Ajusta imputación y escala usando una fila por paciente del fold de train."""
    patients = train.drop_duplicates("patient_id").sort_values("patient_id")
    values = clinical_matrix(patients)
    if np.isnan(values).all(axis=0).any():
        raise ValueError("Una variable clínica no tiene valores observados en train")
    medians = np.nanmedian(values, axis=0)
    imputed = np.where(np.isnan(values), medians, values)
    means = imputed.mean(axis=0)
    stds = imputed.std(axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)
    return {
        "fit_scope": "unique_patients_in_training_partition",
        "medians": medians.tolist(),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "n_patients": int(len(patients)),
    }


def fit_clinical_initializer(train: pd.DataFrame, preprocessing: dict) -> dict:
    """Ajusta el baseline lineal solo con pacientes del train del fold."""
    patients = train.drop_duplicates("patient_id").sort_values("patient_id")
    values = clinical_matrix(patients)
    medians = np.asarray(preprocessing["medians"], dtype=float)
    means = np.asarray(preprocessing["means"], dtype=float)
    stds = np.asarray(preprocessing["stds"], dtype=float)
    missing = np.isnan(values)
    standardized = (np.where(missing, medians, values) - means) / stds
    design = np.column_stack((standardized, missing.astype(float)))
    labels = patients.pCR.to_numpy(dtype=int)
    if set(labels) != {0, 1}:
        raise ValueError("La inicialización clínica requiere ambas clases en train")
    baseline = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                  class_weight="balanced", random_state=0)
    baseline.fit(design, labels)
    return {"method": "fold_train_logistic_balanced", "coefficients": baseline.coef_[0].tolist(),
            "intercept": float(baseline.intercept_[0]), "n_patients": int(len(patients))}


def particion(samples, fold_val):
    if fold_val not in range(5) or not samples.split.eq("train").all():
        raise ValueError("La partición solo acepta train y folds 0-4")
    train = samples.loc[samples.fold.ne(fold_val)].copy().reset_index(drop=True)
    val = samples.loc[samples.fold.eq(fold_val)].copy().reset_index(drop=True)
    if train.empty or val.empty or set(train.patient_id) & set(val.patient_id):
        raise ValueError("Partición vacía o fuga por paciente")
    if train.pCR.nunique() != 2 or val.pCR.nunique() != 2:
        raise ValueError("Train y validación deben contener ambas clases")
    return train, val


class DatasetEntrenamiento(pdatos.BreastDCESliceDataset):
    """Reutiliza el cargador de Marta, sin estandarización ni aumentos de validación."""
    def __init__(self, filas, root, cache=None, clinical=False):
        self.cache = cache
        self.clinical = bool(clinical)
        if self.clinical:
            clinical_matrix(filas)
        if not filas.split.eq("train").all():
            raise ValueError("El entrenamiento solo admite train")
        root = Path(root).resolve()
        for row in filas.itertuples(index=False):
            for column, phase in zip(pdatos.COLUMNAS_RUTA, pdatos.FASES):
                path = Path(getattr(row, column))
                expected = Path("dataset/train")/str(row.patient_id)/f"{row.sample_id}_{phase}.png"
                if path != expected or not (root/path).resolve().is_relative_to(root):
                    raise ValueError("Ruta/fase no coincide con paciente, muestra y train")
        super().__init__(filas, raiz=root, transform=None)

    def __getitem__(self, indice):
        row = self.filas.iloc[indice]
        if self.cache is not None:
            result = {"image": self.cache.imagen(row.sample_id),
                "label": torch.tensor(float(row.pCR),dtype=torch.float32),
                "patient_id": str(row.patient_id), "sample_id": str(row.sample_id),
                "slice_index": torch.tensor(int(row.slice_index),dtype=torch.int64)}
            if self.clinical:
                result["clinical"] = torch.from_numpy(clinical_matrix(row.to_frame().T)[0].astype(np.float32))
            return result
        for column in pdatos.COLUMNAS_RUTA:
            with Image.open(self.raiz/getattr(row, column)) as png:
                if png.format != "PNG" or png.mode != "L" or png.size != (256, 256):
                    raise ValueError("Se requiere PNG monocromo de 8 bits y 256x256")
        result = super().__getitem__(indice)
        if self.clinical:
            result["clinical"] = torch.from_numpy(clinical_matrix(row.to_frame().T)[0].astype(np.float32))
        return result


class PatientBatchSampler:
    """Agrupa pacientes completos sin ponderarlos por su número de cortes."""
    def __init__(self, samples, patients_per_batch, generator, shuffle=True):
        if not isinstance(patients_per_batch, int) or patients_per_batch < 1:
            raise ValueError("patients_per_batch debe ser un entero positivo")
        self.generator = generator
        self.shuffle = bool(shuffle)
        self.patients_per_batch = patients_per_batch
        self.patient_ids = list(dict.fromkeys(samples.patient_id.astype(str)))
        self.indices = {patient: [] for patient in self.patient_ids}
        for index, patient in enumerate(samples.patient_id.astype(str)):
            self.indices[patient].append(index)
        if not self.patient_ids or any(not indices for indices in self.indices.values()):
            raise ValueError("No se pudieron construir lotes completos por paciente")

    def __iter__(self):
        order = list(range(len(self.patient_ids)))
        if self.shuffle:
            order = torch.randperm(len(order), generator=self.generator).tolist()
        for start in range(0, len(order), self.patients_per_batch):
            batch = []
            for position in order[start:start + self.patients_per_batch]:
                batch.extend(self.indices[self.patient_ids[position]])
            yield batch

    def __len__(self):
        return (len(self.patient_ids) + self.patients_per_batch - 1) // self.patients_per_batch


def make_loader(samples, root, batch_size, workers, generator, shuffle=False, cache=None, clinical=False,
                patients_per_batch=None):
    # Los lectores trabajan en CPU; Linux usa fork y Windows/macOS spawn.
    context = ("fork" if sys.platform.startswith("linux") else "spawn") if workers else None
    dataset = DatasetEntrenamiento(samples, root, cache, clinical)
    common = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=False,
                  worker_init_fn=pdatos.inicializar_worker, generator=generator,
                  multiprocessing_context=context)
    if patients_per_batch is not None:
        sampler = PatientBatchSampler(samples, patients_per_batch, generator, shuffle)
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **common)


def aumentar_geometria(x, generator, hflip=0.5, rot90=True):
    # Cada decisión se aplica al corte completo, conservando las tres fases.
    if hflip:
        mask = (torch.rand(len(x), generator=generator) < hflip).to(x.device)
        x[mask] = x[mask].flip(-1)
    if rot90:
        ks = torch.randint(0, 4, (len(x),), generator=generator)
        for k in (1, 2, 3):
            mask = (ks == k).to(x.device)
            x[mask] = torch.rot90(x[mask], k, dims=(-2, -1))
    return x


def batch_device(batch, device):
    # pipeline_datos ya convierte a float32 /255; no dividir una segunda vez.
    x = batch["image"].to(device,non_blocking=True)
    if x.dtype == torch.uint8:
        x = x.float().div_(255.0)
    return x, batch["label"].to(device,non_blocking=True)


def batch_clinical(batch, device):
    clinical = batch.get("clinical")
    return clinical.to(device, non_blocking=True) if clinical is not None else None


def aggregate_patient_logits(logits, labels, patient_ids):
    """Convierte logits por corte en una decisión diferenciable por paciente."""
    if logits.ndim != 1 or labels.ndim != 1 or len(logits) != len(labels) or len(logits) != len(patient_ids):
        raise ValueError("Logits, etiquetas y pacientes deben estar alineados")
    groups = {}
    for index, patient in enumerate(map(str, patient_ids)):
        groups.setdefault(patient, []).append(index)
    patient_logits, patient_labels = [], []
    for indices in groups.values():
        index = torch.tensor(indices, device=logits.device)
        current_labels = labels.index_select(0, index)
        if not torch.equal(current_labels, current_labels[:1].expand_as(current_labels)):
            raise ValueError("Etiquetas inconsistentes dentro de una paciente")
        probability = torch.sigmoid(logits.index_select(0, index).float()).mean()
        patient_logits.append(torch.logit(probability.clamp(EPS, 1 - EPS)))
        patient_labels.append(current_labels[0].float())
    return torch.stack(patient_logits), torch.stack(patient_labels)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    # Preserve float32 accumulation policy as part of the experiment.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def capture_rng(generator: torch.Generator, augmentation_generator: torch.Generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state(), "augmentation": augmentation_generator.get_state()}


def restore_rng(state, generator: torch.Generator, augmentation_generator: torch.Generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    generator.set_state(state["loader"].cpu())
    augmentation_generator.set_state(state["augmentation"].cpu())


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int,
                    best_auc: float, bad_epochs: int, config: dict,
                    generator, augmentation_generator, history: list, best_epoch: int):
    payload = {"schema_version": SCHEMA_VERSION, "state_dict": model.state_dict(),
               "model_config": deepcopy(config["model"]), "config": deepcopy(config),
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "scaler": scaler.state_dict(), "epoch": epoch, "best_auc": best_auc,
               "best_epoch": best_epoch, "bad_epochs": bad_epochs,
               "rng": capture_rng(generator, augmentation_generator), "history": history}
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None,
                    generator=None, augmentation_generator=None):
    # Only locally generated checkpoints: numpy/Python RNG require weights_only=False.
    # Public inference exports remove RNG/optimizer and support safe weight loading.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if generator is not None and augmentation_generator is not None:
        restore_rng(checkpoint["rng"], generator, augmentation_generator)
    return checkpoint


def export_inference_checkpoint(run_dir: Path):
    """Export locally produced best.pt to a weights-only inference artifact.

    Full checkpoints contain NumPy RNG and are trusted local resume artifacts;
    they must never be loaded as uploaded/public inference files. This export
    deliberately excludes optimizer, RNG, histories and patient/label metadata.
    """
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    payload = {"schema_version": SCHEMA_VERSION,
               "state_dict": {k: v.detach().cpu() for k, v in checkpoint["state_dict"].items()},
               "model_config": deepcopy(checkpoint["model_config"])}
    temporary = run_dir / "inference.tmp"
    torch.save(payload, temporary)
    # Ensure the actual file is compatible with PyTorch's restricted loader.
    torch.load(temporary, map_location="cpu", weights_only=True)
    temporary.replace(run_dir / "inference.pt")
    return run_dir / "inference.pt"


def patient_metrics(samples: pd.DataFrame, probabilities: np.ndarray, threshold: float = 0.5):
    if len(samples) != len(probabilities):
        raise ValueError("Predicciones/filas no alineadas")
    frame = samples[["patient_id", "pCR"]].copy()
    frame["prob"] = probabilities
    patients = frame.groupby("patient_id", sort=True).agg(pCR=("pCR", "first"), prob=("prob", "mean"))
    truth, prob = patients.pCR.to_numpy(), patients.prob.to_numpy()
    if len(np.unique(truth)) != 2:
        raise ValueError("Validación debe contener pacientes de ambas clases")
    tn, fp, fn, tp = confusion_matrix(truth, prob >= threshold, labels=[0, 1]).ravel()
    return {"patient_auc": float(roc_auc_score(truth, prob)),
            "patient_ap": float(average_precision_score(truth, prob)),
            "patient_accuracy": float(accuracy_score(truth, prob >= threshold)),
            "patient_f1": float(f1_score(truth,prob >= threshold,zero_division=0)),
            "patient_precision": float(tp/(tp+fp)) if tp+fp else 0.0,
            "patient_recall": float(tp/(tp+fn)),
            "patient_balanced_accuracy": float((tp/(tp+fn)+tn/(tn+fp))/2),
            "patient_sensitivity": float(tp / (tp + fn)),
            "patient_specificity": float(tn / (tn + fp)),
            "patient_brier": float(brier_score_loss(truth, prob)),
            "patient_logloss": float(log_loss(truth, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
            "patient_tn": int(tn), "patient_fp": int(fp), "patient_fn": int(fn), "patient_tp": int(tp),
            "n_patients": len(patients)}


def amp_context(device, enabled):
    return torch.amp.autocast("cuda", dtype=torch.float16) if enabled and device.type == "cuda" else nullcontext()


def train_epoch(model, loader, optimizer, scaler, criterion, device, amp, augmentation,
                augmentation_generator, clip_norm=5.0, max_batches=None, loss_unit="slice"):
    model.train()
    losses = torch.zeros((), device=device)
    count = 0
    bar = loader
    for batch_idx, batch in enumerate(bar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = batch_device(batch, device)
        clinical = batch_clinical(batch, device)
        x = aumentar_geometria(x, augmentation_generator, **augmentation)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, amp):
            logits = model(x, clinical).squeeze(1)
            targets = y
            if loss_unit == "patient":
                logits, targets = aggregate_patient_logits(logits, y, batch["patient_id"])
            elif loss_unit != "slice":
                raise ValueError("Unidad de pérdida desconocida")
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Pérdida no finita; no se seleccionará este run")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # AMP may legitimately overflow at the current scale; GradScaler skips
        # that optimizer step and lowers scale. Do not turn this into a crash.
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=not amp)
        scaler.step(optimizer)
        scaler.update()
        units = len(targets)
        losses += loss.detach() * units
        count += units
    return float(losses / count)


@torch.inference_mode()
def predict(model, loader, device, amp=False, criterion=None):
    model.eval()
    probabilities = []
    loss_sum, n = torch.zeros((), device=device), 0
    for batch in loader:
        x, y = batch_device(batch, device)
        clinical = batch_clinical(batch, device)
        with amp_context(device, amp):
            logits = model(x, clinical).squeeze(1)
            if criterion is not None:
                units = len(y)
                loss_sum += criterion(logits, y) * units
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
        n += len(y)
    return np.concatenate(probabilities), float(loss_sum / n) if criterion is not None else None


def history_write(path: Path, history: list):
    temp = path.with_suffix(".tmp")
    pd.DataFrame(history).to_csv(temp, index=False)
    temp.replace(path)


def config_hash(config):
    # Runtime device/root/output are storage metadata; training schedule is immutable.
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def run_training(config: dict, loss_name: str, seed: int, fold: int, root: Path,
                 out: Path, device: torch.device, smoke=False,
                 resume="last", until_epoch=None, cache=None):
    config = deepcopy(config)
    config["loss"], config["seed"], config["fold"] = loss_name, seed, fold
    if loss_name not in {"normal", "weighted"}:
        raise ValueError("Pérdida desconocida")
    config["smoke"] = bool(smoke)
    config["source_sha256"] = {
        "04_entrenamiento.py": sha256(Path(__file__)),
        "pipeline_datos.py": sha256(PACKAGE_DIR / "datos.py"),
    }
    config["device_type"] = device.type
    tconf = config["training"]
    loss_unit = tconf.get("loss_unit", "slice")
    if loss_unit not in {"slice", "patient"}:
        raise ValueError("La unidad de pérdida debe ser slice o patient")
    patients_per_batch = tconf.get("patients_per_batch")
    if loss_unit == "patient" and (not isinstance(patients_per_batch, int) or patients_per_batch < 1):
        raise ValueError("La BCE por paciente requiere patients_per_batch positivo")
    if loss_unit == "slice" and patients_per_batch is not None:
        raise ValueError("patients_per_batch solo se admite con BCE por paciente")
    unsupported_batch = any(key in tconf for key in
        ("batch_policy", "patient_batch_size", "paired_augmentation"))
    if unsupported_batch or config["evaluation"].get("protocol") == "nested_holdout":
        raise ValueError("Configuración experimental retirada del entrenamiento activo")
    if until_epoch is not None and not 1 <= until_epoch <= int(tconf["epochs"]):
        raise ValueError("La época objetivo debe estar dentro del presupuesto fijado")
    torch.set_num_threads(int(tconf.get("cpu_threads", 8)))
    seed_everything(seed, tconf.get("deterministic", True))
    clinical_enabled = config["model"].get("clinical") is not None
    all_train = cargar_train(root, include_clinical=clinical_enabled)
    train, val = particion(all_train, fold)
    signature_frame = all_train.drop(columns=list(CLINICAL_COLUMNS), errors="ignore")
    config["data_signature"] = hashlib.sha256(signature_frame.to_csv(index=False).encode()).hexdigest()
    if clinical_enabled:
        clinical_roster = all_train.drop_duplicates("patient_id").sort_values("patient_id")
        clinical_values = clinical_roster[["patient_id", *CLINICAL_COLUMNS]]
        config["clinical_signature"] = hashlib.sha256(
            clinical_values.to_csv(index=False).encode()).hexdigest()
        preprocessing = fit_clinical_preprocessing(train)
        config["model"]["clinical"]["preprocessing"] = preprocessing
        config["model"]["clinical"]["initialization"] = fit_clinical_initializer(train, preprocessing)
    if loss_name == "weighted" and loss_unit == "patient":
        patient_labels = train.drop_duplicates("patient_id").pCR
        config["pos_weight"] = float((patient_labels == 0).sum() / (patient_labels == 1).sum())
    else:
        config["pos_weight"] = pdatos.pos_weight_cortes(train) if loss_name == "weighted" else 1.0
    config["n_train_slices"], config["n_val_slices"] = len(train), len(val)
    config["n_train_patients"], config["n_val_patients"] = train.patient_id.nunique(), val.patient_id.nunique()
    run_dir = out / config["name"] / loss_name / f"seed_{seed}" / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "summary.json").exists():
        summary = json.loads((run_dir / "summary.json").read_text())
        if summary.get("config_hash") != config_hash(config):
            raise ValueError(f"Configuración distinta para run existente: {run_dir}")
        if until_epoch is not None and summary.get("epochs_completed",0) >= until_epoch:
            block_path=run_dir/f"bloque_{until_epoch:03d}.json"
            block=json.loads(block_path.read_text())
            for name,key in ((f"inference_{until_epoch:03d}.pt","inference_sha256"),
                             (f"oof_slices_{until_epoch:03d}.csv","oof_sha256")):
                if sha256(run_dir/name)!=block[key]:
                    raise ValueError("Artefacto de revisión modificado")
            return block
        if summary.get("status") == "complete":
            for name, key in (("inference.pt", "inference_sha256"), ("oof_slices.csv", "oof_sha256")):
                if sha256(run_dir/name) != summary[key]:
                    raise ValueError(f"Artefacto modificado: {run_dir/name}")
            print(f"Ya completado: {run_dir}", flush=True)
            return summary
    if (run_dir / "config.json").exists():
        previous = json.loads((run_dir / "config.json").read_text())
        if config_hash(previous) != config_hash(config):
            raise ValueError(f"Cambió la configuración de {run_dir}; usa un out nuevo")
    json_write(run_dir / "config.json", config)
    if not (run_dir / "environment.json").exists():
        json_write(run_dir / "environment.json", environment(device))
    if smoke:
        train = train.groupby("pCR", group_keys=False).head(4).reset_index(drop=True)
        val = val.groupby("pCR", group_keys=False).head(4).reset_index(drop=True)
    generator = torch.Generator().manual_seed(seed + fold * 1000)
    aug_generator = torch.Generator().manual_seed(seed + fold * 1000 + 1)
    # Separate generator: validation iteration must not alter training shuffle RNG.
    val_generator = torch.Generator().manual_seed(seed + 60000 + fold)
    workers = 0 if smoke else int(tconf["num_workers"])
    batch_size = min(8, tconf["batch_size"]) if smoke else tconf["batch_size"]
    train_loader = make_loader(train, root, batch_size, workers, generator, shuffle=True,
                               cache=cache, clinical=clinical_enabled,
                               patients_per_batch=patients_per_batch if loss_unit == "patient" else None)
    val_loader = make_loader(val, root, batch_size, workers, val_generator, cache=cache,
                             clinical=clinical_enabled)
    train_eval_loader = make_loader(train, root, batch_size, workers,
        torch.Generator().manual_seed(seed+70000+fold), cache=cache, clinical=clinical_enabled)
    model = CNN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tconf["lr"], weight_decay=tconf["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tconf["epochs"], eta_min=tconf["min_lr"])
    amp = bool(tconf["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config["pos_weight"], device=device))
    start_epoch, best_auc, bad_epochs, history, best_epoch = 0, -1.0, 0, [], -1
    checkpoint_path = run_dir / f"{resume}.pt"
    if checkpoint_path.exists():
        loaded = load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, generator, aug_generator)
        if config_hash(loaded["config"]) != config_hash(config):
            raise ValueError("Checkpoint con configuración incompatible")
        start_epoch, best_auc = loaded["epoch"] + 1, loaded["best_auc"]
        bad_epochs, history = loaded["bad_epochs"], loaded["history"]
        best_epoch = loaded["best_epoch"]
        print(f"Reanudando {run_dir} época {start_epoch + 1}", flush=True)
    elif clinical_enabled:
        baseline_probabilities, _ = predict(model, val_loader, device, False, criterion)
        baseline_metrics = patient_metrics(val, baseline_probabilities, config["evaluation"]["threshold"])
        best_auc = baseline_metrics["patient_auc"]
        save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, scaler, -1, best_auc,
                        bad_epochs, config, generator, aug_generator, history, best_epoch)
        print(f"Baseline clínico fold-train: valAUC={best_auc:.4f}", flush=True)
    print(f"Run {run_dir}: {numero_parametros(model):,} parámetros; {len(train)} cortes train, "
          f"{len(val)} validación, pos_weight={config['pos_weight']:.4f}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    epochs_limit = 1 if smoke else (until_epoch or int(tconf["epochs"]))
    stop_reason = "max_epochs"
    # An epoch-complete checkpoint may precede a crash while writing OOF/summary.
    already_stopped = (start_epoch >= tconf["min_epochs"] and bad_epochs >= tconf["patience"])
    if already_stopped:
        stop_reason = "early_stopping"
    for epoch in range(start_epoch, epochs_limit if not already_stopped else start_epoch):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tic = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch(model, train_loader, optimizer, scaler, criterion, device, amp,
                                 config["augmentation"], aug_generator, tconf["clip_grad_norm"],
                                 max_batches=1 if smoke else None, loss_unit=loss_unit)
        probabilities, val_loss = predict(model, val_loader, device, False, criterion)
        metrics = patient_metrics(val, probabilities, config["evaluation"]["threshold"])
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - tic
        # Best checkpoint tracks the maximum AUC; min_delta only controls stopping.
        improved = metrics["patient_auc"] > best_auc
        substantial = metrics["patient_auc"] > best_auc + tconf["min_delta"]
        bad_epochs = 0 if substantial else bad_epochs + 1
        if improved:
            best_auc, best_epoch = metrics["patient_auc"], epoch
        row = {"epoch": epoch + 1, "lr": lr, "train_loss": train_loss,
               "val_loss": val_loss, **metrics, "epoch_seconds": elapsed,
               "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0.0,
               "bad_epochs": bad_epochs, "smoke": smoke}
        will_stop = not smoke and epoch+1 >= tconf["min_epochs"] and bad_epochs >= tconf["patience"]
        review = (epoch+1) % int(tconf.get("review_interval",10)) == 0 or epoch+1 == epochs_limit or will_stop
        if review:
            train_prob, train_eval_loss = predict(model,train_eval_loader,device,False,criterion)
            train_metrics = patient_metrics(train,train_prob,config["evaluation"]["threshold"])
            row.update({"train_"+k:v for k,v in train_metrics.items()})
            row["train_eval_loss"] = train_eval_loss
        history.append(row)
        if improved:
            save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_auc,
                            bad_epochs, config, generator, aug_generator, history, best_epoch)
        save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best_auc,
                        bad_epochs, config, generator, aug_generator, history, best_epoch)
        history_write(run_dir / "history.csv", history)
        if review:
            json_write(run_dir/f"revision_{epoch+1:03d}.json",row)
            graficar_historia(run_dir)
            checkpoint = {"schema_version":SCHEMA_VERSION,"model_config":deepcopy(config["model"]),
                "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}}
            torch.save(checkpoint,run_dir/f"epoch_{epoch+1:03d}.pt")
        print(f"Epoch {epoch + 1:02d}/{epochs_limit} loss={train_loss:.4f} "
              f"valAUC={metrics['patient_auc']:.4f} best={best_auc:.4f} "
              f"F1={metrics['patient_f1']:.4f} AP={metrics['patient_ap']:.4f} sens={metrics['patient_sensitivity']:.3f} "
              f"spec={metrics['patient_specificity']:.3f} t={elapsed:.1f}s VRAM={row['peak_vram_gib']:.2f}GiB", flush=True)
        if not smoke and epoch + 1 >= tconf["min_epochs"] and bad_epochs >= tconf["patience"]:
            stop_reason = "early_stopping"
            break
    if not (run_dir / "best.pt").exists():
        raise RuntimeError("No existe checkpoint válido")
    load_checkpoint(run_dir / "best.pt", model)
    # Validation/OOF use float32 to match CPU deployment inference.
    probabilities, val_loss = predict(model, val_loader, device, False, criterion)
    metrics = patient_metrics(val, probabilities, config["evaluation"]["threshold"])
    oof = val[["sample_id", "patient_id", "fold", "pCR", "dataset"]].copy()
    oof["prob"], oof["split"], oof["seed"], oof["loss"] = probabilities, "train", seed, loss_name
    oof["smoke"] = bool(smoke)
    oof.to_csv(run_dir / "oof_slices.csv", index=False)
    completed = smoke or stop_reason == "early_stopping" or len(history) >= int(tconf["epochs"])
    summary = {"schema_version": SCHEMA_VERSION, "status": "complete" if completed else "paused", "smoke": bool(smoke),
               "eligible_for_selection": completed and not smoke, "config_hash": config_hash(config),
               "config_name": config["name"], "loss": loss_name, "seed": seed, "fold": fold,
               "epochs_completed": len(history), "best_epoch": best_epoch + 1,
               "stop_reason": "smoke_only" if smoke else (stop_reason if completed else "bloque_completado"), "best_auc": best_auc,
               "val_metrics": metrics, "val_loss": val_loss, "parameters": numero_parametros(model),
               "total_train_seconds": sum(r["epoch_seconds"] for r in history),
               "current_session_seconds": time.perf_counter() - wall_start,
               "peak_vram_gib": max(r["peak_vram_gib"] for r in history),
               "checkpoint": "best.pt", "test_images_loaded": 0,
               "inference_amp": False, "patient_aggregation": "mean", "loss_unit": loss_unit}
    export_inference_checkpoint(run_dir)
    summary["inference_sha256"] = sha256(run_dir/"inference.pt")
    summary["oof_sha256"] = sha256(run_dir/"oof_slices.csv")
    json_write(run_dir / "summary.json", summary)
    if until_epoch is not None:
        shutil.copy2(run_dir/"inference.pt",run_dir/f"inference_{epochs_limit:03d}.pt")
        shutil.copy2(run_dir/"oof_slices.csv",run_dir/f"oof_slices_{epochs_limit:03d}.csv")
        json_write(run_dir/f"bloque_{epochs_limit:03d}.json",summary)
    return summary


def environment(device):
    result = {"python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "pillow": PIL.__version__,
        "cuda_runtime": torch.version.cuda, "device": str(device),
        "code_sha256": {
            "04_entrenamiento.py": sha256(Path(__file__)),
            "pipeline_datos.py": sha256(PACKAGE_DIR / "datos.py"),
        },
        "determinism_scope": "Mismo código, datos, entorno y hardware; reanudación por época."}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(gpu=props.name, gpu_total_gib=props.total_memory/2**30)
    try:
        result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=RAIZ, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        result["git_commit"] = None
    return result


def validate_oof(frame: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns={"pCR": "label", "prob": "probability"}).copy()
    required = {"sample_id", "patient_id", "label", "fold", "probability", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Faltan columnas OOF: {sorted(required - set(frame.columns))}")
    if not (frame["split"] == "train").all():
        raise ValueError("OOF debe contener exclusivamente train; test/validation prohibidos")
    if frame.sample_id.duplicated().any():
        raise ValueError("Una muestra tiene varias predicciones OOF en la misma semilla")
    if set(frame.sample_id) != set(roster.sample_id):
        raise ValueError("OOF incompleto o con muestras ajenas al roster de train")
    reference = roster.set_index("sample_id").loc[frame.sample_id]
    for column in ("patient_id", "label", "fold"):
        if not np.array_equal(frame[column].to_numpy(), reference[column].to_numpy()):
            raise ValueError(f"OOF {column} no coincide con metadatos de train")
    probability = frame.probability.to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilidades OOF fuera de [0,1] o no finitas")
    frame["cohort"] = reference.cohort.to_numpy()
    return frame.sort_values("sample_id").reset_index(drop=True)


def aggregate_patients(frame: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    if method not in METHODS:
        raise ValueError(f"Agregación desconocida: {method}")
    for column in ("label", "fold", "split", "cohort"):
        if column in frame and frame.groupby("patient_id")[column].nunique().max() > 1:
            raise ValueError(f"Paciente con {column} inconsistentes")
    fields = {column: "first" for column in ("label", "fold", "split", "cohort") if column in frame}
    fields.update(probability=method, sample_id="count")
    return frame.groupby("patient_id", sort=True).agg(fields).rename(columns={"sample_id": "n_slices"}).reset_index()


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if len(y) == 0 or len(y) != len(p) or not set(y).issubset({0, 1}):
        raise ValueError("Se necesitan etiquetas binarias y probabilidades de igual longitud")
    y = y.astype(int)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilidades inválidas")
    if not 0 <= threshold <= 1:
        raise ValueError("Umbral fuera de [0,1]")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    both = len(np.unique(y)) == 2
    return {
        "n_patients": len(y), "n_positive": int(y.sum()), "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y,pred,zero_division=0)),
        "precision": tp/(tp+fp) if tp+fp else 0.0,
        "recall": tp/(tp+fn) if tp+fn else None,
        "balanced_accuracy": float((tp/(tp+fn)+tn/(tn+fp))/2) if both else None,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "threshold": float(threshold), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def fit_platt(y: np.ndarray, p: np.ndarray) -> dict:
    if len(np.unique(y)) != 2:
        raise ValueError("La calibración requiere ambas clases")
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0)
    model.fit(logit(p).reshape(-1, 1), y)
    slope = float(model.coef_[0, 0])
    constrained = slope < 0
    intercept = float(model.intercept_[0])
    if constrained:
        # Optimum of logistic loss at the boundary slope=0, free intercept.
        # Preserve the predeclared score direction instead of flipping a weak CNN.
        slope, intercept = 0.0, float(logit(np.array([np.mean(y)]))[0])
    return {"method": "regularized_platt", "input": "logit_patient_aggregated_probability",
            "slope": slope, "intercept": intercept, "C": 1.0,
            "clip_epsilon": EPS, "scope": "patient", "fit_data": "train_oof",
            "nonnegative_slope_constraint": True, "boundary_solution": constrained}


def apply_calibration(p: np.ndarray, calibration: dict) -> np.ndarray:
    z = calibration["slope"] * logit(p) + calibration["intercept"]
    return np.exp(-np.logaddexp(0, -z))


def crossfit_calibration(patients: pd.DataFrame) -> np.ndarray:
    result = np.empty(len(patients), dtype=float)
    folds = sorted(patients.fold.unique())
    if len(folds) < 2:
        raise ValueError("Cross-fitting requiere al menos dos folds completos")
    for fold in folds:
        held_out = patients.fold.to_numpy() == fold
        calibration = fit_platt(patients.label.to_numpy()[~held_out], patients.probability.to_numpy()[~held_out])
        result[held_out] = apply_calibration(patients.probability.to_numpy()[held_out], calibration)
    return result


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """ROC thresholds + 0.5 + feasible all-negative boundary; deterministic ties."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    binary_metrics(y, p)  # Validate labels/range/finiteness before threshold search.
    if len(np.unique(y)) != 2:
        raise ValueError("Youden requiere ambas clases")
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    thresholds, scores = thresholds[valid], (tpr - fpr)[valid]
    # sklearn's leading infinity encodes all-negative decisions. Represent it
    # with the closest feasible finite threshold in [0,1], preserving >= semantics.
    extra = [0.5]
    all_negative = float(np.nextafter(p.max(), np.inf))
    if all_negative <= 1:
        extra.append(all_negative)
    extra_scores = []
    for threshold in extra:
        metrics = binary_metrics(y, p, threshold)
        extra_scores.append(metrics["sensitivity"] + metrics["specificity"] - 1)
    thresholds = np.concatenate([thresholds, np.asarray(extra)])
    scores = np.concatenate([scores, np.asarray(extra_scores)])
    best = thresholds[np.isclose(scores, scores.max(), atol=1e-12, rtol=0)]
    return float(sorted(best, key=lambda t: (abs(t - 0.5), -t))[0])


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, min_sensitivity: float) -> float:
    """Máxima especificidad sujeta a una sensibilidad mínima, con empates deterministas."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    binary_metrics(y, p)  # Validación común de etiquetas y probabilidades.
    if len(np.unique(y)) != 2:
        raise ValueError("La selección de umbral requiere ambas clases")
    if not 0 < min_sensitivity <= 1:
        raise ValueError("min_sensitivity debe pertenecer a (0,1]")
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], p)))
    feasible = []
    for threshold in candidates:
        metrics = binary_metrics(y, p, float(threshold))
        if metrics["sensitivity"] + 1e-12 >= min_sensitivity:
            feasible.append(metrics)
    if not feasible:
        raise RuntimeError("No existe un umbral que alcance la sensibilidad solicitada")
    best = max(feasible, key=lambda m: (m["specificity"], m["precision"], m["threshold"]))
    return float(best["threshold"])


def comparar_ejecuciones(root, output):
    """Compara ejecuciones nuevas completas; no altera el modelo final histórico."""
    train = cargar_train(root)
    data_signature = hashlib.sha256(train.to_csv(index=False).encode()).hexdigest()
    roster = train.rename(columns={"pCR": "label", "dataset": "cohort"})
    groups, fingerprints = {}, {}
    for path in sorted(Path(output).rglob("summary.json")):
        summary = json.loads(path.read_text())
        if summary.get("status") != "complete" or not summary.get("eligible_for_selection"):
            continue
        folder = path.parent
        config = json.loads((folder/"config.json").read_text())
        if summary["config_hash"] != config_hash(config):
            raise ValueError(f"Configuración modificada: {folder}")
        if config.get("smoke"):
            continue
        if config.get("data_signature") != data_signature:
            raise ValueError("Los datos actuales no coinciden con las ejecuciones")
        for name, key in (("inference.pt", "inference_sha256"), ("oof_slices.csv", "oof_sha256")):
            if sha256(folder/name) != summary[key]:
                raise ValueError(f"Artefacto modificado: {folder/name}")
        fold = int(config["fold"])
        oof = validate_oof(pd.read_csv(folder/"oof_slices.csv"), roster[roster.fold.eq(fold)])
        if config["name"] not in {"base_raw", "raw_rot90", "base_raw_pool", "raw_rot90_pool",
                                  "pool_dropout_wd", "pool_dropout_wd_clinical",
                                  "patient_level_clinical"}:
            raise ValueError("Nombre de configuración desconocido")
        key = (config["name"], config["loss"], int(config["seed"]))
        groups.setdefault(key, {})
        if fold in groups[key]:
            raise ValueError("Fold duplicado")
        if not (folder/"inference.pt").exists():
            raise ValueError("Faltan pesos de inferencia")
        groups[key][fold] = (oof, folder/"inference.pt", config)
        comparable = {k: v for k, v in config.items() if k not in {
            "loss", "seed", "fold", "pos_weight", "n_train_slices", "n_val_slices",
            "n_train_patients", "n_val_patients", "clinical_signature"}}
        comparable = deepcopy(comparable)
        if comparable.get("model", {}).get("clinical"):
            comparable["model"]["clinical"].pop("preprocessing", None)
            comparable["model"]["clinical"].pop("initialization", None)
        fingerprint = config_hash(comparable)
        if config["name"] in fingerprints and fingerprints[config["name"]] != fingerprint:
            raise ValueError("Comparación con distinto código, datos o presupuesto")
        fingerprints[config["name"]] = fingerprint
    candidates, predictions = [], {}
    for name in sorted({key[0] for key in groups}):
        losses = ("weighted",) if name in {"pool_dropout_wd", "patient_level_clinical"} else ("normal", "weighted")
        expected = {(name, loss, seed) for loss in losses for seed in (42, 2026)}
        if not expected.issubset(groups) or any(set(groups[key]) != set(range(5)) for key in expected):
            raise ValueError(f"{name}: se requieren pérdidas {losses}, dos semillas y cinco folds completos")
        for loss in losses:
            by_seed = []
            for seed in (42, 2026):
                by_seed.append(validate_oof(pd.concat([groups[name,loss,seed][fold][0]
                    for fold in range(5)], ignore_index=True), roster))
            slices = by_seed[0].copy()
            if not slices.sample_id.equals(by_seed[1].sample_id):
                raise ValueError("Semillas con distinto índice OOF")
            slices["probability"] = (slices.probability.to_numpy()+by_seed[1].probability.to_numpy())/2
            for method in METHODS:
                patients = aggregate_patients(slices, method)
                metrics = binary_metrics(patients.label, patients.probability)
                fold_auc = []
                for fold in range(5):
                    held_out = patients[patients.fold.eq(fold)]
                    auc = binary_metrics(held_out.label, held_out.probability)["roc_auc"]
                    if auc is None:
                        raise ValueError(f"Fold {fold} sin ambas clases; AUC no definida")
                    fold_auc.append(auc)
                metrics["fold_roc_auc"] = fold_auc
                metrics["mean_fold_roc_auc"] = float(np.mean(fold_auc))
                metrics["min_fold_roc_auc"] = float(np.min(fold_auc))
                candidate = f"{name}_{loss}_{method}"
                candidates.append({"candidate": candidate, "configuration": name, "loss": loss,
                    "aggregation": method, "metrics": metrics})
                predictions[candidate] = patients
    if not candidates:
        raise ValueError("No hay comparaciones completas; las pruebas no son seleccionables")
    selected = min(candidates, key=lambda c: (-c["metrics"]["mean_fold_roc_auc"],
        -c["metrics"]["roc_auc"], c["metrics"]["brier"],
        c["aggregation"] != "mean", c["candidate"]))
    patients = predictions[selected["candidate"]]
    calibration = fit_platt(patients.label.to_numpy(), patients.probability.to_numpy())
    patients["calibrated_crossfit"] = crossfit_calibration(patients)
    patients["calibrated_fit"] = apply_calibration(patients.probability.to_numpy(), calibration)
    selected_config = groups[selected["configuration"], selected["loss"], 42][0][2]
    evaluation = selected_config.get("evaluation", {})
    if evaluation.get("threshold_strategy") == "max_specificity_at_min_sensitivity":
        target_sensitivity = float(evaluation.get("min_sensitivity", .90))
        threshold = threshold_for_sensitivity(
            patients.label.to_numpy(), patients.calibrated_fit.to_numpy(), target_sensitivity)
        decision_policy = {"strategy": "max_specificity_at_min_sensitivity",
                           "min_sensitivity": target_sensitivity,
                           "fit_data": "train_oof_apparent"}
    else:
        threshold = youden_threshold(patients.label.to_numpy(), patients.calibrated_fit.to_numpy())
        decision_policy = {"strategy": "youden", "fit_data": "train_oof_apparent"}
    dest = Path(output)/"comparacion"
    dest.mkdir(parents=True, exist_ok=True)
    report = {"status": "desarrollo_sin_test", "warning":
        "La selección de candidatos y checkpoints con OOF introduce optimismo; no es validación anidada.",
        "candidates": candidates, "selected": selected,
        "decision_policy": decision_policy,
        "calibrated_crossfit_at_0_5": binary_metrics(patients.label, patients.calibrated_crossfit),
        "calibrated_apparent": binary_metrics(patients.label, patients.calibrated_fit, threshold)}
    json_write(dest/"seleccion.json", report)
    patients.to_csv(dest/"oof_pacientes.csv", index=False)
    records = [{"candidate": c["candidate"], **c["metrics"]} for c in candidates]
    pd.DataFrame(records).drop(columns="confusion_matrix").to_csv(dest/"comparacion.csv", index=False)
    models = []
    for seed in (42, 2026):
        for fold in range(5):
            _, path, config = groups[selected["configuration"], selected["loss"], seed][fold]
            models.append({"path": os.path.relpath(path.resolve(), dest.resolve()), "sha256": sha256(path),
                "seed": seed, "fold": fold, "model_config": config["model"]})
    selected_clinical = any(item["model_config"].get("clinical") is not None for item in models)
    manifest = {"schema_version": 1, "status": "desarrollo_sin_test", "models": models,
        "aggregation": selected["aggregation"], "calibration": calibration, "threshold": threshold,
        "decision_policy": decision_policy,
        "test_evaluated_once": False, "selection": selected,
        "preprocessing": {"phases": list(pdatos.FASES), "shape": [3,256,256],
                          "scaling": "uint8 / 255; CNN: 2*x-1",
                          "clinical_required": selected_clinical,
                          "clinical_raw_columns": list(CLINICAL_COLUMNS) if selected_clinical else []}}
    json_write(dest/"modelo_desarrollo.json", manifest)
    (dest/"modelo_desarrollo.json.sha256").write_text(sha256(dest/"modelo_desarrollo.json")+"\n")
    return report


def cargar_manifest(path):
    path = Path(path).resolve()
    expected_checksum = path.with_suffix(path.suffix+".sha256").read_text().strip()
    raw = path.read_bytes()
    canonical_checksum = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    if expected_checksum not in {sha256(path), canonical_checksum}:
        raise ValueError("El manifiesto ha cambiado")
    manifest = json.loads(path.read_text())
    if len(manifest["models"]) != 10 or {(m["seed"],m["fold"]) for m in manifest["models"]} != {
            (seed,fold) for seed in (42,2026) for fold in range(5)}:
        raise ValueError("Se requieren diez modelos: dos semillas y cinco folds")
    if manifest["aggregation"] not in METHODS or not 0 <= manifest["threshold"] <= 1:
        raise ValueError("Agregación o umbral inválidos")
    calibration = manifest["calibration"]
    if calibration.get("clip_epsilon") != EPS or calibration["slope"] < 0 or not np.isfinite(
            [calibration["slope"], calibration["intercept"]]).all():
        raise ValueError("Calibración incompatible")
    clinical_flags = {item["model_config"].get("clinical") is not None for item in manifest["models"]}
    if len(clinical_flags) != 1:
        raise ValueError("El ensemble mezcla modelos con y sin variables clínicas")
    clinical_required = clinical_flags.pop()
    if bool(manifest.get("preprocessing", {}).get("clinical_required", False)) != clinical_required:
        raise ValueError("El manifiesto no describe correctamente las variables clínicas")
    for item in manifest["models"]:
        model_path = path.parent/item["path"]
        if sha256(model_path) != item["sha256"]:
            raise ValueError(f"Pesos modificados: {model_path}")
    for source in manifest.get("provenance_files", []):
        if sha256(path.parent/source["path"]) != source["sha256"]:
            raise ValueError(f"Evidencia histórica modificada: {source['path']}")
    return manifest


def verificar_modelo_final(path):
    manifest = cargar_manifest(path)
    parameter_counts = set()
    for item in manifest["models"]:
        artifact = torch.load(Path(path).parent/item["path"], map_location="cpu", weights_only=True)
        if artifact["model_config"] != item["model_config"]:
            raise ValueError("Configuración de pesos y manifiesto diferentes")
        model = CNN(artifact["model_config"])
        parameters = numero_parametros(model)
        if not model.clinical_enabled and parameters != 551913:
            raise ValueError("La arquitectura no coincide con el PDF")
        parameter_counts.add(parameters)
        model.load_state_dict(artifact["state_dict"], strict=True)
    if len(parameter_counts) != 1:
        raise ValueError("Los modelos del ensemble tienen distinto número de parámetros")
    return {"models": 10, "parameters_per_model": parameter_counts.pop(), "checksums_verified": True,
        "test_evaluated_once": manifest["test_evaluated_once"], "test_repeated": False,
        "aggregation": manifest["aggregation"], "threshold": manifest["threshold"],
        "clinical_required": manifest.get("preprocessing", {}).get("clinical_required", False)}


@torch.inference_mode()
def predecir_cortes(path, triples, clinical=None):
    manifest = cargar_manifest(path)
    clinical_required = bool(manifest.get("preprocessing", {}).get("clinical_required", False))
    if clinical_required and clinical is None:
        raise ValueError("El modelo multimodal requiere edad, volumen tumoral, HR y HER2")
    if not clinical_required and clinical is not None:
        raise ValueError("El modelo histórico no admite variables clínicas")
    clinical_values = None
    if clinical_required:
        if set(clinical) != set(CLINICAL_COLUMNS):
            raise ValueError(f"Se requieren exactamente estas variables clínicas: {list(CLINICAL_COLUMNS)}")
        clinical_values = clinical_matrix(pd.DataFrame([clinical])).astype(np.float32)
    images, sample_ids, patient_ids = [], set(), set()
    for triple in triples:
        if len(triple) != 3:
            raise ValueError("Cada corte requiere tres fases")
        arrays, sample = [], None
        for phase, png_path in zip(pdatos.FASES, triple):
            png_path = Path(png_path)
            if not png_path.stem.endswith("_"+phase):
                raise ValueError("Los nombres deben terminar en _PRE, _EARLY y _LATE")
            current = png_path.stem[:-(len(phase)+1)]
            if sample is not None and current != sample:
                raise ValueError("Las fases no corresponden al mismo corte")
            sample = current
            with Image.open(png_path) as png:
                if png.format != "PNG" or png.mode != "L" or png.size != (256,256):
                    raise ValueError("Se requiere PNG monocromo de 8 bits y 256x256")
                arrays.append(np.asarray(png,dtype=np.float32)/255.0)
        match = re.fullmatch(r"(.+)_z[0-9]+", sample)
        if not match or sample in sample_ids:
            raise ValueError("Identificador de corte inválido o duplicado")
        sample_ids.add(sample)
        patient_ids.add(match.group(1))
        images.append(torch.from_numpy(np.stack(arrays)))
    if not images or len(patient_ids) != 1:
        raise ValueError("Se necesitan cortes de una sola paciente")
    x = torch.stack(images)
    clinical_tensor = None
    if clinical_values is not None:
        clinical_tensor = torch.from_numpy(clinical_values).repeat(len(images), 1)
    predictions = []
    for item in manifest["models"]:
        artifact = torch.load(Path(path).parent/item["path"],map_location="cpu",weights_only=True)
        model = CNN(artifact["model_config"]).eval()
        model.load_state_dict(artifact["state_dict"],strict=True)
        predictions.append(torch.sigmoid(model(x, clinical_tensor).squeeze(1)).numpy())
    slices = np.mean(predictions,axis=0)
    raw = float(getattr(np, manifest["aggregation"])(slices))
    calibrated = float(apply_calibration(np.array([raw]),manifest["calibration"])[0])
    return {"patient_id": next(iter(patient_ids)), "n_slices": len(images), "probability_raw": raw,
        "probability_calibrated": calibrated, "threshold": manifest["threshold"],
        "pCR": int(calibrated >= manifest["threshold"]), "models": len(predictions),
        "clinical_used": clinical_required,
        "notice": "Uso educativo. La calibración se ajustó con varios cortes por paciente."}


class CacheTrain:
    """Memmap uint8 de train: evita decodificar los mismos PNG en cada época."""
    def __init__(self,samples,root,folder,workers=4):
        if not samples.split.eq("train").all():
            raise ValueError("La caché solo admite train")
        folder = Path(folder)
        folder.mkdir(parents=True,exist_ok=True)
        columns = ["sample_id","pCR","fold",*pdatos.COLUMNAS_RUTA]
        signature = hashlib.sha256(samples[columns].to_csv(index=False).encode()).hexdigest()
        self.path = folder/f"train_{signature[:16]}_uint8.npy"
        shape = (len(samples),3,256,256)
        metadata = self.path.with_suffix(".json")
        self.index = {str(sample):i for i,sample in enumerate(samples.sample_id)}
        # La validación de rutas también se aplica al crear la caché.
        DatasetEntrenamiento(samples,root)
        if not self.path.exists():
            temporary = self.path.with_suffix(".tmp.npy")
            array = np.lib.format.open_memmap(temporary,mode="w+",dtype=np.uint8,shape=shape)
            def read(row):
                channels=[]
                for column in pdatos.COLUMNAS_RUTA:
                    with Image.open(Path(root)/getattr(row,column)) as png:
                        if png.format!="PNG" or png.mode!="L" or png.size!=(256,256):
                            raise ValueError("PNG incompatible con el contrato")
                        channels.append(np.array(png,dtype=np.uint8))
                return np.stack(channels)
            try:
                with ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
                    for i,image in enumerate(pool.map(read,samples.itertuples(index=False))):
                        array[i]=image
                array.flush()
                del array
                temporary.replace(self.path)
                json_write(metadata,{"signature":signature,"shape":list(shape),"dtype":"uint8","split":"train"})
            except BaseException:
                if "array" in locals():
                    del array
                temporary.unlink(missing_ok=True)
                raise
        record=json.loads(metadata.read_text())
        if record["signature"]!=signature or record["split"]!="train":
            raise ValueError("Caché de otro índice")
        self.memory=np.load(self.path,mmap_mode="r")
        if self.memory.shape!=shape or self.memory.dtype!=np.uint8:
            raise ValueError("Caché incompleta o incompatible")

    def imagen(self,sample_id):
        return torch.from_numpy(np.array(self.memory[self.index[str(sample_id)]],copy=True))

    def __getstate__(self):
        # NumPy serializa el contenido del memmap. Reabrirlo evita copiar gigabytes
        # cuando Windows/macOS o un lector configurado con spawn crea un worker.
        state = self.__dict__.copy()
        state.pop("memory", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.memory = np.load(self.path, mmap_mode="r")


def pyplot():
    # Los ficheros de matplotlib también quedan dentro de los resultados.
    os.environ.setdefault("MPLCONFIGDIR",str(SALIDA/".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def graficar_historia(folder):
    plt=pyplot()
    history=pd.read_csv(Path(folder)/"history.csv")
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    axes[0,0].plot(history.epoch,history.train_loss,label="Train durante aprendizaje")
    axes[0,0].plot(history.epoch,history.val_loss,label="Validación")
    if "train_eval_loss" in history:
        clean=history.dropna(subset=["train_eval_loss"])
        axes[0,0].plot(clean.epoch,clean.train_eval_loss,"o--",label="Train en evaluación")
    axes[0,0].set(title="Pérdida BCE",ylabel="Pérdida")
    for ax,metric,title in [(axes[0,1],"patient_auc","ROC-AUC por paciente"),
                            (axes[1,0],"patient_f1","F1 de pCR=1 · umbral 0,5"),
                            (axes[1,1],"patient_accuracy","Accuracy por paciente")]:
        ax.plot(history.epoch,history[metric],label="Validación")
        column="train_"+metric
        if column in history:
            clean=history.dropna(subset=[column])
            ax.plot(clean.epoch,clean[column],"o--",label="Train en evaluación")
        ax.set(title=title,ylim=(0,1))
    for ax in axes.flat:
        ax.set_xlabel("Época")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Aprendizaje y pérdida · "+Path(folder).parent.parent.parent.name,fontsize=14)
    fig.savefig(Path(folder)/"curvas.png",dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", nargs="?", choices=("verificar", "entrenar", "comparar", "predecir"), default="verificar")
    parser.add_argument("--datos",type=Path,default=pdatos.DATOS)
    parser.add_argument("--salida",type=Path,default=None)
    parser.add_argument("--manifest",type=Path,default=SALIDA/"modelo_final.json")
    parser.add_argument("--configuraciones",nargs="+",choices=("base_raw","raw_rot90","pool_dropout_wd",
        "pool_dropout_wd_clinical","patient_level_clinical"),default=["raw_rot90"])
    parser.add_argument("--perdidas",nargs="+",choices=("normal","ponderada"),default=["normal","ponderada"])
    parser.add_argument("--semillas",nargs="+",type=int,default=[42,2026])
    parser.add_argument("--folds",nargs="+",type=int,choices=range(5),default=list(range(5)))
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--lote",type=int,default=64)
    parser.add_argument("--epocas",type=int,default=46)
    parser.add_argument("--revision-cada",type=int,default=10)
    parser.add_argument("--hasta-epoca",type=int)
    parser.add_argument("--pooling",choices=("original","intermedio"),default="original")
    parser.add_argument("--dispositivo",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--prueba",action="store_true",help="Un lote y una época; excluido de la selección")
    for phase in ("pre","early","late"):
        parser.add_argument("--"+phase,type=Path,nargs="+")
    parser.add_argument("--edad", type=float)
    parser.add_argument("--volumen-tumoral", type=float)
    parser.add_argument("--hr", type=float, choices=(0.0, 1.0))
    parser.add_argument("--her2", type=float, choices=(0.0, 1.0))
    args = parser.parse_args()
    args.salida = args.salida or SALIDA/"ejecuciones"
    if args.workers < 0 or args.lote < 1 or args.epocas < 1 or args.revision_cada < 1:
        parser.error("workers >= 0, lote >= 1 y epocas >= 1")
    if args.accion == "entrenar":
        device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.dispositivo == "auto" else args.dispositivo
        if device_name == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA no está disponible")
        output = args.salida/"pruebas" if args.prueba else args.salida
        for name in dict.fromkeys(args.configuraciones):
            config = configuracion(name)
            if args.pooling == "intermedio" and config["model"].get("pooling", "original") != "intermedio":
                config["model"]["pooling"]="intermedio"
                config["name"]+="_pool"
            config["training"]["review_interval"]=args.revision_cada
            config["training"].update(num_workers=args.workers,batch_size=args.lote,epochs=args.epocas,
                                      min_epochs=min(12,args.epocas))
            for loss in dict.fromkeys(args.perdidas):
                for seed in dict.fromkeys(args.semillas):
                    for fold in dict.fromkeys(args.folds):
                        run_training(config,"weighted" if loss == "ponderada" else loss,seed,fold,
                                     args.datos,output,torch.device(device_name),smoke=args.prueba,until_epoch=args.hasta_epoca)
    elif args.accion == "comparar":
        result = comparar_ejecuciones(args.datos,args.salida)
        print(json.dumps(result["selected"],ensure_ascii=False,indent=2))
    elif args.accion == "predecir":
        if not args.pre or not args.early or not args.late or not len(args.pre)==len(args.early)==len(args.late):
            parser.error("predecir requiere el mismo número de --pre, --early y --late")
        torch.set_num_threads(8)
        clinical_args = (args.edad, args.volumen_tumoral, args.hr, args.her2)
        clinical = None if all(value is None for value in clinical_args) else {
            "age": args.edad, "tum_vol": args.volumen_tumoral, "HR": args.hr, "HER2": args.her2}
        print(json.dumps(predecir_cortes(args.manifest,list(zip(args.pre,args.early,args.late)),clinical),ensure_ascii=False,indent=2))
    else:
        print(json.dumps(verificar_modelo_final(args.manifest),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
