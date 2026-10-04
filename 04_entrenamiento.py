#!/usr/bin/env python3
"""Paso 04: CNN, entrenamiento por paciente, comparación OOF e inferencia.

Reutiliza pipeline_datos.py del commit de Marta 69a44eb. El test reservado
ya fue evaluado: aquí se conservan sus resultados y no se vuelve a abrir.
La acción ajustar compara pooling e hiperparámetros en bloques de diez épocas.
Las ejecuciones nuevas se guardan aparte del modelo final histórico.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import nullcontext, redirect_stdout, redirect_stderr
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
from multiprocessing import get_context

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             confusion_matrix, brier_score_loss, log_loss, roc_curve, f1_score,
                             precision_recall_curve)
import sklearn
import PIL
import torch
from torch import nn
from torch.utils.data import DataLoader

import pipeline_datos as pdatos

RAIZ = Path(__file__).resolve().parent
SALIDA = RAIZ / "resultados" / "04_entrenamiento"
SCHEMA_VERSION = 1
EPS = 1e-7
METHODS = ("mean", "max", "median")


def configuracion(nombre="raw_rot90"):
    if nombre not in {"base_raw", "raw_rot90"}:
        raise ValueError("Configuración desconocida")
    return {
        "name": nombre,
        "model": {"representation": "raw", "channels": [24, 48, 96, 160],
                  "hidden": 64, "dropout": 0.20},
        "training": {"epochs": 46, "min_epochs": 12, "patience": 10,
                     "min_delta": 0.001, "batch_size": 64, "lr": 0.0008,
                     "min_lr": 0.00001, "weight_decay": 0.0001,
                     "clip_grad_norm": 5.0, "amp": True, "num_workers": 4,
                     "cpu_threads": 8, "deterministic": True, "review_interval": 10},
        "augmentation": {"hflip": 0.5, "rot90": nombre == "raw_rot90"},
        "evaluation": {"threshold": 0.5, "checkpoint_metric": "patient_auc"},
    }


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
        self.features = nn.Sequential(*[Bloque(a, b, i == 0, pooling)
            for i, (a, b) in enumerate(zip([3, *canales[:-1]], canales))])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = GlobalMaxPool()
        hidden = int(self.config.get("hidden", 64))
        dropout = float(self.config.get("dropout", 0.20))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(canales[-1]*2, hidden),
            nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.apply(self._initialize)

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

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError(f"Esperado [N,3,256,256], recibido {tuple(x.shape)}")
        z = self.features(self.preprocess(x))
        z = torch.cat((self.avgpool(z).flatten(1), self.maxpool(z).flatten(1)), dim=1)
        return self.classifier(z)


def numero_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def cargar_train(root):
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
    if "dataset" not in frame:
        with (Path(root)/"metadata/patients.csv").open(newline="", encoding="utf-8") as f:
            patients = pd.DataFrame([r for r in csv.DictReader(f) if r["split"] == "train"])
        frame = frame.merge(patients[["pid", "dataset"]], left_on="patient_id", right_on="pid",
                            how="left", validate="many_to_one").drop(columns="pid")
    if frame.dataset.isna().any():
        raise ValueError("Paciente sin cohorte")
    return frame.reset_index(drop=True)


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
    def __init__(self, filas, root, cache=None):
        self.cache = cache
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
            return {"image": self.cache.imagen(row.sample_id),
                "label": torch.tensor(float(row.pCR),dtype=torch.float32),
                "patient_id": str(row.patient_id), "sample_id": str(row.sample_id),
                "slice_index": torch.tensor(int(row.slice_index),dtype=torch.int64)}
        for column in pdatos.COLUMNAS_RUTA:
            with Image.open(self.raiz/getattr(row, column)) as png:
                if png.format != "PNG" or png.mode != "L" or png.size != (256, 256):
                    raise ValueError("Se requiere PNG monocromo de 8 bits y 256x256")
        return super().__getitem__(indice)


def make_loader(samples, root, batch_size, workers, generator, shuffle=False, cache=None, patient_batch_size=None):
    # Los lectores trabajan solo en CPU. En Linux, fork conserva el memmap compartido
    # aunque este entrenamiento se ejecute dentro de un proceso creado con spawn.
    context = ("fork" if sys.platform.startswith("linux") else "spawn") if workers else None
    if patient_batch_size is not None:
        return DataLoader(DatasetPacientes(samples, root, cache), batch_size=patient_batch_size,
            shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available(),
            worker_init_fn=pdatos.inicializar_worker, generator=generator,
            collate_fn=collate_pacientes, multiprocessing_context=context)
    return DataLoader(DatasetEntrenamiento(samples, root, cache), batch_size=batch_size,
        shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=False, worker_init_fn=pdatos.inicializar_worker, generator=generator,
        multiprocessing_context=context)


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
                augmentation_generator, clip_norm=5.0, max_batches=None):
    model.train()
    losses = torch.zeros((), device=device)
    count = 0
    bar = loader
    for batch_idx, batch in enumerate(bar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = batch_device(batch, device)
        x = aumentar_geometria(x, augmentation_generator, **augmentation)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, amp):
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Pérdida no finita; no se seleccionará este run")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # AMP may legitimately overflow at the current scale; GradScaler skips
        # that optimizer step and lowers scale. Do not turn this into a crash.
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=not amp)
        scaler.step(optimizer)
        scaler.update()
        units = len(y)
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
        with amp_context(device, amp):
            logits = model(x).squeeze(1)
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
    config["source_sha256"] = {p.name: sha256(p) for p in (Path(__file__), RAIZ/"pipeline_datos.py")}
    config["device_type"] = device.type
    tconf = config["training"]
    if tconf.get("loss_unit", "slice") != "slice":
        raise ValueError("La pérdida por paciente fue retirada tras empeorar la referencia; solo se admite BCE por corte")
    if until_epoch is not None and not 1 <= until_epoch <= int(tconf["epochs"]):
        raise ValueError("La época objetivo debe estar dentro del presupuesto fijado")
    torch.set_num_threads(int(tconf.get("cpu_threads", 8)))
    seed_everything(seed, tconf.get("deterministic", True))
    all_train = cargar_train(root)
    nested = config["evaluation"].get("protocol") == "nested_holdout"
    if nested:
        roles, plan = particion_generalizacion(all_train, fold)
        train, val = roles["fit"], roles["selection"]
        config["partition_signature"] = hashlib.sha256(plan.to_csv(index=False).encode()).hexdigest()
    else:
        train, val = particion(all_train, fold)
    config["data_signature"] = hashlib.sha256(all_train.to_csv(index=False).encode()).hexdigest()
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
        # Include both labels and complete small groups; metrics are diagnostic only.
        if nested:
            def small(frame):
                ids = frame.drop_duplicates("patient_id").groupby("pCR").head(2).patient_id
                return frame[frame.patient_id.isin(ids)].reset_index(drop=True)
            train, val = small(train), small(val)
        else:
            train = train.groupby("pCR", group_keys=False).head(4).reset_index(drop=True)
            val = val.groupby("pCR", group_keys=False).head(4).reset_index(drop=True)
    generator = torch.Generator().manual_seed(seed + fold * 1000)
    aug_generator = torch.Generator().manual_seed(seed + fold * 1000 + 1)
    # Separate generator: validation iteration must not alter training shuffle RNG.
    val_generator = torch.Generator().manual_seed(seed + 60000 + fold)
    workers = 0 if smoke else int(tconf["num_workers"])
    batch_size = min(8, tconf["batch_size"]) if smoke else tconf["batch_size"]
    patient_batch_size = tconf.get("patient_batch_size")
    train_loader = make_loader(train, root, batch_size, workers, generator, shuffle=True, cache=cache, patient_batch_size=patient_batch_size)
    val_loader = make_loader(val, root, batch_size, workers, val_generator,cache=cache, patient_batch_size=patient_batch_size)
    train_eval_loader = make_loader(train,root,batch_size,workers,
        torch.Generator().manual_seed(seed+70000+fold),cache=cache, patient_batch_size=patient_batch_size)
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
                                 max_batches=1 if smoke else None)
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
    if nested:
        oof.to_csv(run_dir / "selection_slices.csv", index=False)
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
               "inference_amp": False, "patient_aggregation": "mean"}
    if nested:
        summary.update(prediction_role="inner_selection", outer_evaluation_used=False,
                       loss_unit="slice", patient_batch_size=patient_batch_size)
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
        "code_sha256": {p.name: sha256(p) for p in (Path(__file__), RAIZ/"pipeline_datos.py")},
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
        if config["name"] not in {"base_raw", "raw_rot90", "base_raw_pool", "raw_rot90_pool"}:
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
            "n_train_patients", "n_val_patients"}}
        fingerprint = config_hash(comparable)
        if config["name"] in fingerprints and fingerprints[config["name"]] != fingerprint:
            raise ValueError("Comparación con distinto código, datos o presupuesto")
        fingerprints[config["name"]] = fingerprint
    candidates, predictions = [], {}
    for name in sorted({key[0] for key in groups}):
        expected = {(name, loss, seed) for loss in ("normal", "weighted") for seed in (42, 2026)}
        if not expected.issubset(groups) or any(set(groups[key]) != set(range(5)) for key in expected):
            raise ValueError(f"{name}: se requieren ambas pérdidas, dos semillas y cinco folds completos")
        for loss in ("normal", "weighted"):
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
                candidate = f"{name}_{loss}_{method}"
                candidates.append({"candidate": candidate, "configuration": name, "loss": loss,
                    "aggregation": method, "metrics": metrics})
                predictions[candidate] = patients
    if not candidates:
        raise ValueError("No hay comparaciones completas; las pruebas no son seleccionables")
    selected = min(candidates, key=lambda c: (-c["metrics"]["roc_auc"], c["metrics"]["brier"],
        c["aggregation"] != "mean", c["candidate"]))
    patients = predictions[selected["candidate"]]
    calibration = fit_platt(patients.label.to_numpy(), patients.probability.to_numpy())
    patients["calibrated_crossfit"] = crossfit_calibration(patients)
    patients["calibrated_fit"] = apply_calibration(patients.probability.to_numpy(), calibration)
    threshold = youden_threshold(patients.label.to_numpy(), patients.calibrated_fit.to_numpy())
    dest = Path(output)/"comparacion"
    dest.mkdir(parents=True, exist_ok=True)
    report = {"status": "desarrollo_sin_test", "warning":
        "La selección de candidatos y checkpoints con OOF introduce optimismo; no es validación anidada.",
        "candidates": candidates, "selected": selected,
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
    manifest = {"schema_version": 1, "status": "desarrollo_sin_test", "models": models,
        "aggregation": selected["aggregation"], "calibration": calibration, "threshold": threshold,
        "test_evaluated_once": False, "selection": selected,
        "preprocessing": {"phases": list(pdatos.FASES), "shape": [3,256,256], "scaling": "uint8 / 255; CNN: 2*x-1"}}
    json_write(dest/"modelo_desarrollo.json", manifest)
    (dest/"modelo_desarrollo.json.sha256").write_text(sha256(dest/"modelo_desarrollo.json")+"\n")
    return report


def cargar_manifest(path):
    path = Path(path).resolve()
    if sha256(path) != path.with_suffix(path.suffix+".sha256").read_text().strip():
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
    for item in manifest["models"]:
        artifact = torch.load(Path(path).parent/item["path"], map_location="cpu", weights_only=True)
        if artifact["model_config"] != item["model_config"]:
            raise ValueError("Configuración de pesos y manifiesto diferentes")
        model = CNN(artifact["model_config"])
        if numero_parametros(model) != 551913:
            raise ValueError("La arquitectura no coincide con el PDF")
        model.load_state_dict(artifact["state_dict"], strict=True)
    return {"models": 10, "parameters_per_model": 551913, "checksums_verified": True,
        "test_evaluated_once": manifest["test_evaluated_once"], "test_repeated": False,
        "aggregation": manifest["aggregation"], "threshold": manifest["threshold"]}


@torch.inference_mode()
def predecir_cortes(path, triples):
    manifest = cargar_manifest(path)
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
    predictions = []
    for item in manifest["models"]:
        artifact = torch.load(Path(path).parent/item["path"],map_location="cpu",weights_only=True)
        model = CNN(artifact["model_config"]).eval()
        model.load_state_dict(artifact["state_dict"],strict=True)
        predictions.append(torch.sigmoid(model(x).squeeze(1)).numpy())
    slices = np.mean(predictions,axis=0)
    raw = float(getattr(np, manifest["aggregation"])(slices))
    calibrated = float(apply_calibration(np.array([raw]),manifest["calibration"])[0])
    return {"patient_id": next(iter(patient_ids)), "n_slices": len(images), "probability_raw": raw,
        "probability_calibrated": calibrated, "threshold": manifest["threshold"],
        "pCR": int(calibrated >= manifest["threshold"]), "models": len(predictions),
        "notice": "Uso educativo. La calibración se ajustó con varios cortes por paciente."}


class CacheTrain:
    """Memmap uint8 de train: evita decodificar los mismos PNG en cada ensayo."""
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
    os.environ.setdefault("MPLCONFIGDIR",str(SALIDA/"ajustes"/".matplotlib"))
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


def candidatos_ajuste(epocas=30):
    """Referencia, ablation de pooling y cuatro combinaciones de hiperparámetros."""
    specifications=[
        ("referencia","original",.0008,.20,.0001),
        ("pool_intermedio","intermedio",.0008,.20,.0001),
        ("pool_lr_baja","intermedio",.0003,.20,.0001),
        ("pool_lr_dropout","intermedio",.0003,.35,.0001),
        ("pool_lr_dropout_wd","intermedio",.0003,.35,.001),
        ("pool_dropout_wd","intermedio",.0008,.35,.001),
    ]
    configs=[]
    for name,pooling,lr,dropout,weight_decay in specifications:
        config=configuracion()
        config["name"]=name
        config["model"].update(pooling=pooling,dropout=dropout)
        # Presupuesto común: la revisión no reinicia pesos ni calendario coseno.
        config["training"].update(epochs=epocas,min_epochs=epocas,patience=epocas,
            lr=lr,weight_decay=weight_decay,review_interval=10)
        configs.append(config)
    return configs


def graficar_comparacion(folder,configs,fold,seed,target=None):
    plt=pyplot()
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    for config in configs:
        run=Path(folder)/"ejecuciones"/config["name"]/"weighted"/f"seed_{seed}"/f"fold_{fold}"
        if not (run/"history.csv").exists():
            continue
        h=pd.read_csv(run/"history.csv")
        if target is not None:
            h=h[h.epoch<=target]
        for ax,column,title in zip(axes,["val_loss","patient_auc","patient_f1"],
                                   ["Pérdida de validación","AUC de validación","F1 de validación · umbral 0,5"]):
            ax.plot(h.epoch,h[column],label=config["name"])
            ax.set(title=title,xlabel="Época")
            ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7,loc="best")
    fig.suptitle(f"Comparación de hiperparámetros · fold {fold}, semilla {seed}")
    fig.savefig(Path(folder)/"comparacion_curvas.png",dpi=150)
    plt.close(fig)


def ajustar_hiperparametros(root,output,device,epocas=30,intervalo=10,fold=0,seed=42,workers=4,lote=64):
    if epocas<intervalo or intervalo<1:
        raise ValueError("El presupuesto debe incluir al menos un bloque")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    configs=candidatos_ajuste(epocas)
    for config in configs:
        config["training"].update(review_interval=intervalo,num_workers=workers,batch_size=lote)
    protocol={"candidates":configs,"fold":fold,"seed":seed,"loss":"weighted",
        "epochs":epocas,"review_every":intervalo,"selection":"Mayor AUC por paciente; desempate F1 y pérdida.",
        "test_reserved":"Ya observado; no se abre ni se usa para ajustar.",
        "warning":"Selección exploratoria con un fold. Confirmar en otros folds antes de declarar mejora."}
    path=output/"protocolo.json"
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError("Protocolo distinto: utiliza otra --salida")
    json_write(path,protocol)
    cache=CacheTrain(cargar_train(root),root,output/".cache",workers)
    targets=list(range(intervalo,epocas+1,intervalo))
    if targets[-1]!=epocas:
        targets.append(epocas)
    records=[]
    for target in targets:
        current=[]
        for config in configs:
            summary=run_training(config,"weighted",seed,fold,Path(root),output/"ejecuciones",device,
                                 until_epoch=target,cache=cache)
            metrics=summary["val_metrics"]
            row={"revision_epoch":target,"candidate":config["name"],"epochs_completed":summary["epochs_completed"],
                "best_epoch":summary["best_epoch"],"pooling":config["model"]["pooling"],
                "lr":config["training"]["lr"],"dropout":config["model"]["dropout"],
                "weight_decay":config["training"]["weight_decay"],"val_loss":summary["val_loss"],**metrics}
            records.append(row)
            current.append(row)
            pd.DataFrame(records).to_csv(output/"comparacion.csv",index=False)
        current.sort(key=lambda r:(-r["patient_auc"],-r["patient_f1"],r["val_loss"],r["candidate"]))
        json_write(output/f"revision_{target:03d}.json",{"epoch":target,"ranking":current})
        graficar_comparacion(output,configs,fold,seed,target)
        print(f"REVISIÓN {target}: mejor candidato {current[0]['candidate']}, "
              f"AUC={current[0]['patient_auc']:.4f}, F1={current[0]['patient_f1']:.4f}",flush=True)
    baseline=next(row for row in current if row["candidate"]=="referencia")
    best=current[0]
    result={"status":"complete","best":best,"baseline":baseline,
        "delta_auc":best["patient_auc"]-baseline["patient_auc"],
        "delta_f1":best["patient_f1"]-baseline["patient_f1"],
        "selected_config":next(c for c in configs if c["name"]==best["candidate"]),
        "selection_warning":protocol["warning"],"test_repeated":False}
    json_write(output/"resultado_ajuste.json",result)
    return result

def configuraciones_tamano(epocas=30, intervalo=10, lote=64, workers=2):
    """Comparación controlada: solo difieren nombre y canales de los bloques."""
    actual = candidatos_ajuste(epocas)[-1]
    actual["name"] = "cnn_actual"
    actual["training"].update(review_interval=intervalo, batch_size=lote,
        num_workers=workers, cpu_threads=4)
    pequena = deepcopy(actual)
    pequena["name"] = "cnn_pequena"
    pequena["model"]["channels"] = [16, 32, 64, 128]
    return [actual, pequena]


def bootstrap_pareado(y, actual, pequena, repeticiones=2000):
    """IC de diferencias condicionado a las predicciones OOF; unidad paciente."""
    y, actual, pequena = np.asarray(y), np.asarray(actual), np.asarray(pequena)
    binary_metrics(y, actual)
    binary_metrics(y, pequena)
    if len(np.unique(y)) != 2 or repeticiones < 1:
        raise ValueError("Bootstrap requiere ambas clases y repeticiones positivas")
    rng = np.random.default_rng(20261004)
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    deltas = {key: [] for key in ("roc_auc", "average_precision", "f1")}
    metrics = {"roc_auc": roc_auc_score, "average_precision": average_precision_score,
               "f1": lambda labels, p: f1_score(labels, p >= .5, zero_division=0)}
    for _ in range(repeticiones):
        idx = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        for key, metric in metrics.items():
            deltas[key].append(float(metric(y[idx], pequena[idx])-metric(y[idx], actual[idx])))
    return {key: {"delta": float(metrics[key](y, pequena)-metrics[key](y, actual)),
                  "ci95": np.quantile(values, [.025, .975]).tolist()}
            for key, values in deltas.items()}


def tarea_tamano(config, seed, fold, root, output, device, smoke=False):
    """Un proceso tiene su propio RNG, contexto CUDA y registro de entrenamiento."""
    root, output = Path(root), Path(output)
    folder = output/"ejecuciones"/config["name"]/"weighted"/f"seed_{seed}"/f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/"entrenamiento.log").open("a", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            cache = CacheTrain(cargar_train(root), root, output/".cache", config["training"]["num_workers"])
            summary = run_training(config, "weighted", seed, fold, root,
                output/"ejecuciones", torch.device(device), smoke=smoke, cache=cache)
    return {"candidate": config["name"], "seed": seed, "fold": fold,
        "best_epoch": summary["best_epoch"], "epochs_completed": summary["epochs_completed"],
        "parameters": summary["parameters"], "val_loss": summary["val_loss"],
        **summary["val_metrics"]}


def curvas_tamano(output, configs, patients):
    plt = pyplot()
    colors = {"cnn_actual": "#2463a0", "cnn_pequena": "#bf5b18"}
    names = {"cnn_actual": "Actual", "cnn_pequena": "Pequeña"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for config in configs:
        histories = [pd.read_csv(path) for path in (Path(output)/"ejecuciones"/config["name"]).glob(
            "weighted/seed_*/fold_*/history.csv")]
        h = pd.concat(histories, ignore_index=True).groupby("epoch").mean(numeric_only=True)
        for ax, val, train, title in zip(axes, ["val_loss", "patient_auc", "patient_f1"],
            ["train_eval_loss", "train_patient_auc", "train_patient_f1"],
            ["Pérdida BCE", "AUC por paciente", "F1 por paciente · umbral 0,5"]):
            color, name = colors[config["name"]], names[config["name"]]
            ax.plot(h.index, h[val], color=color, label=name+" · validación")
            clean = h[h[train].notna()]
            ax.plot(clean.index, clean[train], "o--", color=color, label=name+" · train sin aumentos")
            ax.set(title=title, xlabel="Época")
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[2].set_ylim(0, 1)
    fig.suptitle("Media de diez entrenamientos por variante · cinco folds × dos semillas")
    fig.savefig(Path(output)/"curvas_comparacion.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for config in configs:
        name = config["name"]
        y, p = patients.label.to_numpy(), patients[name].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, color=colors[name], label=names[name])
        axes[1].plot(recall, precision, color=colors[name], label=names[name])
    axes[0].plot([0, 1], [0, 1], "k--", alpha=.4)
    axes[1].axhline(patients.label.mean(), color="k", linestyle="--", alpha=.4, label="Prevalencia")
    axes[0].set(title="ROC · OOF por paciente", xlabel="Falsos positivos", ylabel="Sensibilidad")
    axes[1].set(title="Precisión-recall · OOF por paciente", xlabel="Recall", ylabel="Precisión")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(Path(output)/"roc_precision_recall.png", dpi=160)
    plt.close(fig)


def evaluar_tamano(root, output, configs, repeticiones=2000, graficas=True):
    """Exige veinte runs íntegros y OOF completos, sin consultar test."""
    output = Path(output)
    train = cargar_train(root)
    roster = train.rename(columns={"pCR": "label", "dataset": "cohort"})
    signature = hashlib.sha256(train.to_csv(index=False).encode()).hexdigest()
    records, predictions, artifacts = [], {}, []
    sources = {p.name:sha256(p) for p in (Path(__file__), RAIZ/"pipeline_datos.py")}
    for config in configs:
        expected_parameters = numero_parametros(CNN(config))
        seeds = []
        for seed in (42, 2026):
            folds = []
            for fold in range(5):
                folder = output/"ejecuciones"/config["name"]/"weighted"/f"seed_{seed}"/f"fold_{fold}"
                if not (folder/"summary.json").is_file():
                    raise ValueError("Comparación incompleta: requiere veinte entrenamientos")
                summary = json.loads((folder/"summary.json").read_text())
                used = json.loads((folder/"config.json").read_text())
                if (summary.get("status") != "complete" or not summary.get("eligible_for_selection")
                    or used.get("smoke") or summary.get("test_images_loaded") != 0
                    or summary.get("epochs_completed") != config["training"]["epochs"]):
                    raise ValueError("Run incompleto, de prueba o con test: excluido")
                if summary["config_hash"] != config_hash(used) or used.get("data_signature") != signature:
                    raise ValueError("Configuración o datos de la comparación modificados")
                if used.get("source_sha256") != sources or summary.get("parameters") != expected_parameters:
                    raise ValueError("Código o número de parámetros incompatible con la comparación")
                if any(used.get(key) != value for key, value in config.items()):
                    raise ValueError("Los hiperparámetros no coinciden con el protocolo")
                if (used.get("seed"), used.get("fold"), used.get("loss")) != (seed, fold, "weighted"):
                    raise ValueError("Identidad del run incompatible")
                for filename, key in (("inference.pt", "inference_sha256"), ("oof_slices.csv", "oof_sha256")):
                    if sha256(folder/filename) != summary[key]:
                        raise ValueError("Pesos o predicciones modificados")
                oof = validate_oof(pd.read_csv(folder/"oof_slices.csv"), roster[roster.fold.eq(fold)])
                grouped = aggregate_patients(oof)
                metrics = binary_metrics(grouped.label, grouped.probability)
                for key in ("roc_auc", "f1", "average_precision"):
                    expected = summary["val_metrics"][{"roc_auc":"patient_auc", "f1":"patient_f1", "average_precision":"patient_ap"}[key]]
                    if not np.isclose(metrics[key], expected, atol=1e-7, rtol=0):
                        raise ValueError("Métricas del resumen no coinciden con OOF")
                records.append({"candidate":config["name"], "seed":seed, "fold":fold,
                    "best_epoch":summary["best_epoch"], "parameters":summary["parameters"],
                    "val_loss":summary["val_loss"], **{k:v for k,v in metrics.items() if k != "confusion_matrix"}})
                artifacts.append({"candidate":config["name"], "seed":seed, "fold":fold,
                    "path":str(folder.relative_to(output)), "config_hash":summary["config_hash"],
                    "source_sha256":used["source_sha256"], "environment":json.loads((folder/"environment.json").read_text()),
                    "inference_sha256":summary["inference_sha256"], "oof_sha256":summary["oof_sha256"]})
                folds.append(oof)
            seeds.append(validate_oof(pd.concat(folds, ignore_index=True), roster))
        slices = seeds[0].copy()
        if not slices.sample_id.equals(seeds[1].sample_id):
            raise ValueError("Índices OOF distintos entre semillas")
        slices["probability"] = (seeds[0].probability.to_numpy()+seeds[1].probability.to_numpy())/2
        predictions[config["name"]] = aggregate_patients(slices)
    frame = pd.DataFrame(records)
    frame.to_csv(output/"comparacion_runs.csv", index=False)
    patients = predictions["cnn_actual"].rename(columns={"probability":"cnn_actual"})
    other = predictions["cnn_pequena"]
    if not patients.patient_id.equals(other.patient_id):
        raise ValueError("Pacientes no alineadas")
    patients["cnn_pequena"] = other.probability.to_numpy()
    patients.to_csv(output/"oof_pacientes.csv", index=False)
    metric_names = ("roc_auc", "average_precision", "f1", "precision", "sensitivity", "specificity", "balanced_accuracy", "brier")
    aggregates, cohort_records, pairs = [], [], []
    for config in configs:
        name = config["name"]
        part = frame[frame.candidate.eq(name)]
        aggregates.append({"candidate":name, "parameters":int(part.parameters.iloc[0]),
            "mean_runs":{k:float(part[k].mean()) for k in metric_names},
            "std_runs":{k:float(part[k].std()) for k in metric_names},
            "oof":binary_metrics(patients.label, patients[name])})
        for cohort, group in patients.groupby("cohort"):
            cohort_records.append({"candidate":name, "cohort":cohort,
                **{k:v for k,v in binary_metrics(group.label, group[name]).items() if k != "confusion_matrix"}})
    for seed in (42, 2026):
        for fold in range(5):
            pair = frame[frame.seed.eq(seed) & frame.fold.eq(fold)].set_index("candidate")
            pairs.append({"seed":seed, "fold":fold,
                **{"delta_"+k:float(pair.loc["cnn_pequena",k]-pair.loc["cnn_actual",k]) for k in metric_names}})
    pd.DataFrame(pairs).to_csv(output/"diferencias_pareadas.csv", index=False)
    pd.DataFrame(cohort_records).to_csv(output/"metricas_cohortes.csv", index=False)
    result = {"status":"complete", "completed_runs":20, "patients":len(patients),
        "variants":aggregates, "paired_runs":pairs,
        "paired_bootstrap":bootstrap_pareado(patients.label.to_numpy(), patients.cnn_actual.to_numpy(),
            patients.cnn_pequena.to_numpy(), repeticiones),
        "bootstrap_repetitions":repeticiones, "bootstrap_unit":"patient",
        "test_repeated":False, "historical_model_replaced":False,
        "warning":"Desarrollo: checkpoints elegidos con validación. No es validación anidada ni evaluación externa. Los IC bootstrap están condicionados a las predicciones OOF y no incluyen toda la incertidumbre del entrenamiento.",
        "artifacts":artifacts}
    if graficas:
        curvas_tamano(output, configs, patients)
    json_write(output/"resultado.json", result)
    return result


def comparar_tamano(root, output, device, epocas=30, intervalo=10, lote=64, workers=2, paralelos=2, smoke=False):
    if epocas < intervalo or intervalo < 1 or paralelos not in (1, 2, 3, 4):
        raise ValueError("Presupuesto y revisión positivos; entre uno y cuatro procesos")
    root, output = Path(root), Path(output)
    output = output/"pruebas" if smoke else output
    output.mkdir(parents=True, exist_ok=True)
    configs = configuraciones_tamano(epocas, intervalo, lote, workers)
    cache = CacheTrain(cargar_train(root), root, output/".cache", workers)
    protocol = {"variants":configs, "seeds":[42,2026], "folds":list(range(5)),
        "loss":"weighted", "threshold":.5, "parallel_processes":paralelos,
        "smoke":smoke, "cache_sha256":sha256(cache.path),
        "source_sha256":{p.name:sha256(p) for p in (Path(__file__), RAIZ/"pipeline_datos.py")},
        "only_model_difference":"channels", "test_used":False}
    path = output/"protocolo.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Protocolo distinto: usa otra salida")
    json_write(path, protocol)
    seeds, folds = ([42], [0]) if smoke else ([42,2026], list(range(5)))
    tasks = [(config,seed,fold,str(root),str(output),str(device),smoke)
        for seed in seeds for fold in folds for config in configs]
    rows = []
    def progress(row):
        rows.append(row)
        pd.DataFrame(rows).to_csv(output/"progreso_runs.csv", index=False)
        json_write(output/"estado.json", {"status":"running", "completed":len(rows), "total":len(tasks)})
        print(f"COMPLETO {len(rows)}/{len(tasks)}: {row['candidate']} semilla {row['seed']} fold {row['fold']} "
            f"AUC={row['patient_auc']:.4f} AP={row['patient_ap']:.4f} F1={row['patient_f1']:.4f}", flush=True)
    if paralelos == 1:
        for task in tasks:
            progress(tarea_tamano(*task))
    else:
        with ProcessPoolExecutor(max_workers=paralelos, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(tarea_tamano, *task) for task in tasks]
            for future in as_completed(futures):
                progress(future.result())
    if smoke:
        result = {"status":"smoke_only", "eligible_for_selection":False, "runs":rows}
        json_write(output/"resultado_prueba.json", result)
    else:
        result = evaluar_tamano(root, output, configs)
    json_write(output/"estado.json", {"status":result["status"], "completed":len(rows), "total":len(tasks)})
    return result


# Auditoría y evaluación interna separada por paciente; entrenamiento BCE por corte.
class DatasetPacientes(torch.utils.data.Dataset):
    """Bolsas completas; no se rellenan imágenes que contaminarían BatchNorm."""
    def __init__(self, samples, root, cache=None):
        if not samples.patient_id.is_monotonic_increasing:
            raise ValueError("El loader de pacientes requiere filas ordenadas por patient_id")
        self.slices = DatasetEntrenamiento(samples, root, cache)
        self.groups = list(samples.groupby("patient_id", sort=False).indices.values())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        return [self.slices[int(i)] for i in self.groups[index]]


def collate_pacientes(bags):
    for bag in bags:
        if not bag or len({r["patient_id"] for r in bag}) != 1 or len({float(r["label"]) for r in bag}) != 1:
            raise ValueError("Bolsa con paciente/etiqueta inconsistente")
    if len({bag[0]["patient_id"] for bag in bags}) != len(bags):
        raise ValueError("Paciente repetida en lote")
    result = torch.utils.data.default_collate([row for bag in bags for row in bag])
    result["bag_sizes"] = [len(bag) for bag in bags]
    return result


def particion_generalizacion(samples, outer_fold):
    """Un holdout interno fijo para checkpoint y otro para calibración/umbral."""
    particion(samples, outer_fold)  # Contratos generales: solo train, sin solape.
    patients = samples.groupby("patient_id", sort=True).agg(
        pCR=("pCR", "first"), fold=("fold", "first"), dataset=("dataset", "first")).reset_index()
    development = patients[patients.fold.ne(outer_fold)].reset_index(drop=True)
    strata = development.dataset + "_" + development.pCR.astype(str)
    if strata.value_counts().min() < 4:
        raise ValueError("Insuficientes pacientes por cohorte/clase para separación interna")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=.30, random_state=20261004+outer_fold)
    fit_idx, remainder_idx = next(splitter.split(development, strata))
    remainder = development.iloc[remainder_idx].reset_index(drop=True)
    remainder_strata = remainder.dataset + "_" + remainder.pCR.astype(str)
    split_cal = StratifiedShuffleSplit(n_splits=1, test_size=.5, random_state=20261104+outer_fold)
    selection_idx, calibration_idx = next(split_cal.split(remainder, remainder_strata))
    role_ids = {"fit": set(development.iloc[fit_idx].patient_id),
                "selection": set(remainder.iloc[selection_idx].patient_id),
                "calibration": set(remainder.iloc[calibration_idx].patient_id),
                "evaluation": set(patients.loc[patients.fold.eq(outer_fold), "patient_id"])}
    validate_roles(role_ids, set(patients.patient_id))
    lookup = {pid: role for role, ids in role_ids.items() for pid in ids}
    patients["role"] = patients.patient_id.map(lookup)
    roles = {role: samples[samples.patient_id.isin(ids)].sort_values(
        ["patient_id", "slice_index"]).reset_index(drop=True) for role, ids in role_ids.items()}
    for frame in roles.values():
        if frame.pCR.nunique() != 2:
            raise ValueError("Cada rol necesita ambas clases")
    return roles, patients


def validate_roles(role_ids, roster):
    seen = set()
    for ids in role_ids.values():
        if seen & ids:
            raise ValueError("Fuga por paciente entre roles")
        seen.update(ids)
    if seen != roster:
        raise ValueError("Roles incompletos o pacientes ajenas")


def preservacion_manifest():
    files = [RAIZ/name for name in ("01_auditoria_datos.py", "02_eda_profesional.py",
                                  "03_preparar_datos.py", "pipeline_datos.py")]
    files += [SALIDA/"modelo_final.json"]
    for folder in (SALIDA/"historico", SALIDA/"modelos", RAIZ/"resultados/03_preparacion"):
        files += sorted(p for p in folder.rglob("*") if p.is_file())
    return {str(p.relative_to(RAIZ)): sha256(p) for p in files}


def auditar_generalizacion(samples, root, cache, output, workers):
    """Solo PNG train: equivalencia con caché, hashes y duplicados entre pacientes."""
    with (Path(root)/"metadata/patients.csv").open(newline="", encoding="utf-8") as f:
        metadata = pd.DataFrame([r for r in csv.DictReader(f) if r["split"] == "train"])
    if metadata.pid.duplicated().any() or set(metadata.pid) != set(samples.patient_id):
        raise ValueError("Roster de pacientes inconsistente")
    check = samples.merge(metadata[["pid", "pCR", "dataset"]].rename(columns={"pCR":"patient_label", "dataset":"patient_cohort"}),
                          left_on="patient_id", right_on="pid", validate="many_to_one")
    if not check.pCR.eq(pd.to_numeric(check.patient_label)).all() or not check.dataset.eq(check.patient_cohort).all():
        raise ValueError("Etiqueta/cohorte no coincide con patients.csv")
    if samples.duplicated(["patient_id", "slice_index"]).any():
        raise ValueError("Corte duplicado por paciente")
    def inspect(row):
        pixels, files = [], []
        for column in pdatos.COLUMNAS_RUTA:
            path = Path(root)/getattr(row, column)
            files.append(sha256(path))
            with Image.open(path) as png:
                if png.format != "PNG" or png.mode != "L" or png.size != (256,256):
                    raise ValueError("PNG fuente incompatible")
                pixels.append(np.array(png, dtype=np.uint8))
        image = np.stack(pixels)
        if not np.array_equal(image, cache.memory[cache.index[row.sample_id]]):
            raise ValueError("Caché no coincide con PNG fuente")
        return {"sample_id":row.sample_id, "patient_id":row.patient_id, "fold":int(row.fold),
                "pixel_sha256":hashlib.sha256(image.tobytes()).hexdigest(),
                **dict(zip(("pre_sha256", "early_sha256", "late_sha256"), files)),
                **{f"mean_{phase.lower()}":float(image[i].mean()/255) for i,phase in enumerate(pdatos.FASES)}}
    with ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
        inventory = pd.DataFrame(pool.map(inspect, samples.itertuples(index=False)))
    inventory.to_csv(output/"inventario_train.csv", index=False)
    duplicates = inventory.groupby("pixel_sha256").filter(lambda g: g.patient_id.nunique()>1)
    duplicates.to_csv(output/"duplicados_train.csv", index=False)
    if len(duplicates):
        raise ValueError("Imágenes completas idénticas en pacientes distintas; revisar antes de entrenar")
    patients = samples.groupby("patient_id").agg(pCR=("pCR","first"),cohort=("dataset","first"),
                                                 fold=("fold","first"),n_slices=("sample_id","size"))
    cohort = patients.groupby("cohort").agg(n_patients=("pCR","size"),prevalence=("pCR","mean"),
                                            mean_slices=("n_slices","mean"))
    cohort.to_csv(output/"auditoria_cohortes.csv")
    patient_intensity = inventory.groupby("patient_id")[["mean_pre","mean_early","mean_late"]].mean()
    patient_intensity.join(patients).groupby("cohort").mean(numeric_only=True).to_csv(output/"intensidad_cohortes.csv")
    report = {"n_patients":len(patients), "n_slices":len(samples), "n_train_png_read":len(samples)*3,
        "patient_prevalence":float(patients.pCR.mean()), "slice_prevalence":float(samples.pCR.mean()),
        "slice_counts":patients.n_slices.describe().to_dict(), "test_images_loaded":0,
        "metadata_test_labels_converted":0, "duplicate_triplets_between_patients":0,
        "inventory_sha256":sha256(output/"inventario_train.csv"), "cache_sha256":sha256(cache.path),
        "normalization":"uint8 /255 then 2*x-1; fixed; no estimated global statistics",
        "augmentation":"per slice, joint PRE/EARLY/LATE; train only; hflip .5 and rot90",
        "class_weight":"negative/positive slices of fit only; identical in both arms",
        "aggregation":"arithmetic mean of sigmoid probabilities; then mean of seeds",
        "limitations":["No se comprueban duplicados visuales con test, que permanece cerrado.",
            "Identidades anonimizadas distintas o datos de origen no auditables no se pueden descartar.",
            "Duke y I-SPY seleccionan cortes de forma diferente según la guía fuente.",
            "Los datos de desarrollo ya se observaron históricamente; evaluación interna, no independiente."]}
    json_write(output/"auditoria.json", report)
    return report


def predicciones_rol(config, seed, fold, role, samples, root, output, device, cache):
    folder = output/"ejecuciones"/config["name"]/"weighted"/f"seed_{seed}"/f"fold_{fold}"
    summary = json.loads((folder/"summary.json").read_text())
    if summary["status"] != "complete" or summary["smoke"] or sha256(folder/"inference.pt") != summary["inference_sha256"]:
        raise ValueError("Run incompleto o pesos modificados")
    artifact = torch.load(folder/"inference.pt",map_location="cpu",weights_only=True)
    model = CNN(artifact["model_config"]).to(device)
    model.load_state_dict(artifact["state_dict"])
    loader = make_loader(samples,root,config["training"]["batch_size"],config["training"]["num_workers"],
                         torch.Generator().manual_seed(8080+fold),cache=cache)
    probabilities,_ = predict(model,loader,device,False)
    result = samples[["sample_id","patient_id","pCR","fold","dataset"]].copy()
    result["prob"],result["seed"],result["role"] = probabilities,seed,role
    result.to_csv(folder/f"{role}_slices.csv",index=False)
    del model
    return result


def pacientes_ensemble(frames):
    reference = frames[0].set_index("sample_id").sort_index()
    for frame in frames[1:]:
        other = frame.set_index("sample_id").sort_index()
        pd.testing.assert_frame_equal(reference[["patient_id","pCR","fold","dataset"]],
                                      other[["patient_id","pCR","fold","dataset"]])
    reference["prob"] = np.mean([f.set_index("sample_id").sort_index().prob.to_numpy() for f in frames],axis=0)
    return reference.groupby("patient_id",sort=True).agg(label=("pCR","first"),fold=("fold","first"),
        cohort=("dataset","first"),probability=("prob","mean"),n_slices=("prob","size")).reset_index()


def reliability(y, p, bins=10):
    y,p = np.asarray(y),np.asarray(p)
    groups = np.minimum((p*bins).astype(int), bins-1)
    rows = []
    for i in range(bins):
        mask = groups == i
        if mask.any():
            rows.append({"bin":i,"n":int(mask.sum()),"mean_probability":float(p[mask].mean()),
                         "observed_fraction":float(y[mask].mean())})
    ece = sum(r["n"]*abs(r["mean_probability"]-r["observed_fraction"]) for r in rows)/len(y)
    return {"ece_10_fixed_bins":float(ece),"mean_probability":float(p.mean()),
            "observed_prevalence":float(y.mean()),"bins":rows}


def metricas_con_umbral(y, p, thresholds):
    # Permite decisiones de umbrales internos distintos en cada fold externo.
    result = binary_metrics(y,p)
    decisions = np.asarray(p) >= np.asarray(thresholds)
    classification = binary_metrics(y,decisions.astype(float),.5)
    for key in ("accuracy","f1","precision","recall","balanced_accuracy","sensitivity","specificity",
                "tn","fp","fn","tp","confusion_matrix"):
        result[key] = classification[key]
    result["threshold"] = "Youden en calibración interna de cada fold"
    result["calibration"] = reliability(y,p)
    return result


def bootstrap_generalizacion(patients, names, repetitions=2000):
    """Diferencias pareadas; no vuelve a seleccionar ni calibrar en el bootstrap."""
    rng = np.random.default_rng(20261004)
    groups = [g.index.to_numpy() for _,g in patients.groupby(["fold","cohort","label"])]
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("La comparación pareada requiere dos variantes distintas")
    a,b = (patients[f"{name}_raw"].to_numpy() for name in names)
    y,folds = patients.label.to_numpy(),patients.fold.to_numpy()
    def scores(idx):
        auc = np.mean([roc_auc_score(y[idx][folds[idx]==f],b[idx][folds[idx]==f])-
                       roc_auc_score(y[idx][folds[idx]==f],a[idx][folds[idx]==f]) for f in sorted(set(folds))])
        return [float(auc),float(roc_auc_score(y[idx],b[idx])-roc_auc_score(y[idx],a[idx])),
                float(average_precision_score(y[idx],b[idx])-average_precision_score(y[idx],a[idx]))]
    values = np.asarray([scores(np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]))
                         for _ in range(repetitions)])
    point = scores(np.arange(len(patients)))
    return {key:{"delta":point[i],"ci95":np.quantile(values[:,i],[.025,.975]).tolist()}
            for i,key in enumerate(("mean_outer_auc","pooled_raw_auc","pooled_raw_ap"))}


def graficar_generalizacion(output, configs, patients):
    plt = pyplot()
    fig,axes = plt.subplots(1,3,figsize=(15,4.5))
    for config in configs:
        name = config["name"]
        histories = [pd.read_csv(p) for p in (output/"ejecuciones"/name).rglob("history.csv")]
        history = pd.concat(histories).groupby("epoch").mean(numeric_only=True)
        axes[0].plot(history.index,history.patient_auc,label=name+" selección")
        review = history.dropna(subset=["train_patient_auc"])
        axes[0].plot(review.index,review.train_patient_auc,linestyle="--",label=name+" train")
        q = patients[name+"_raw"].to_numpy(); y=patients.label.to_numpy()
        fpr,tpr,_ = roc_curve(y,q); axes[1].plot(fpr,tpr,label=name)
        precision,recall,_ = precision_recall_curve(y,q); axes[2].plot(recall,precision,label=name)
    axes[0].set(xlabel="Época",ylabel="AUC por paciente",title="Train y selección interna")
    axes[1].plot([0,1],[0,1],color="gray",linestyle="--")
    axes[1].set(xlabel="1 - especificidad",ylabel="Sensibilidad",title="OOF externo interno: ROC cruda")
    axes[2].axhline(patients.label.mean(),color="gray",linestyle="--")
    axes[2].set(xlabel="Sensibilidad",ylabel="Precisión",title="OOF: precisión - sensibilidad")
    for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/"curvas_generalizacion.png",dpi=170); plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(10,4.5))
    for ax,scale in zip(axes,("raw","calibrated")):
        for config in configs:
            name=config["name"]; bins=reliability(patients.label,patients[name+"_"+scale])["bins"]
            ax.plot([r["mean_probability"] for r in bins],[r["observed_fraction"] for r in bins],"o-",label=name)
        ax.plot([0,1],[0,1],"--",color="gray"); ax.set(xlim=(0,1),ylim=(0,1),xlabel="Probabilidad media",
            ylabel="Fracción pCR",title="Calibración: "+scale); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/"calibracion.png",dpi=170); plt.close(fig)


def evaluar_generalizacion(samples, root, output, device, configs, cache, seeds=(42,2026)):
    """Congela TODAS las decisiones internas antes de inferir/evaluar folds externos."""
    if len(configs) != 2 or len({c["name"] for c in configs}) != 2:
        raise ValueError("La comparación requiere dos variantes distintas")
    decisions_path = output/"decisiones_congeladas.json"
    if decisions_path.exists():
        decisions = json.loads(decisions_path.read_text())
        result_path = output/"resultado.json"
        if result_path.exists() and json.loads(result_path.read_text())["decisions_sha256"] != sha256(decisions_path):
            raise ValueError("Decisiones modificadas después de evaluación")
        for name, folds in decisions["arms"].items():
            for fold, decision in folds.items():
                if sha256(output/"calibracion"/name/f"fold_{fold}.csv") != decision["prediction_sha256"]:
                    raise ValueError("Predicciones de calibración modificadas")
    else:
        decisions = {"fit_data":"inner_calibration_only", "evaluation_accessed":False,"arms":{}}
        for config in configs:
            name=config["name"]; decisions["arms"][name]={}
            for fold in range(5):
                roles,_ = particion_generalizacion(samples,fold)
                frames = [predicciones_rol(config,seed,fold,"calibration",roles["calibration"],root,output,device,cache) for seed in seeds]
                patients = pacientes_ensemble(frames)
                calibration = fit_platt(patients.label.to_numpy(),patients.probability.to_numpy())
                calibration["fit_data"] = "inner_calibration_only"
                calibrated = apply_calibration(patients.probability.to_numpy(),calibration)
                threshold = youden_threshold(patients.label.to_numpy(),calibrated)
                patients["calibrated"] = calibrated
                folder=output/"calibracion"/name; folder.mkdir(parents=True,exist_ok=True)
                patients.to_csv(folder/f"fold_{fold}.csv",index=False)
                decisions["arms"][name][str(fold)] = {"calibration":calibration,"threshold":threshold,
                    "n_calibration_patients":len(patients), "prediction_sha256":sha256(folder/f"fold_{fold}.csv"),
                    "models":{str(seed):sha256(output/"ejecuciones"/name/"weighted"/f"seed_{seed}"/f"fold_{fold}"/"inference.pt") for seed in seeds}}
        json_write(decisions_path, decisions)
    fold_rows, cohort_rows, assembled, run_rows = [],[],{},[]
    for config in configs:
        name=config["name"]; parts=[]
        for fold in range(5):
            decision=decisions["arms"][name][str(fold)]
            for seed in seeds:
                path=output/"ejecuciones"/name/"weighted"/f"seed_{seed}"/f"fold_{fold}"/"inference.pt"
                if sha256(path) != decision["models"][str(seed)]: raise ValueError("Pesos cambiaron después de congelar")
            roles,_ = particion_generalizacion(samples,fold)
            frames=[predicciones_rol(config,seed,fold,"evaluation",roles["evaluation"],root,output,device,cache) for seed in seeds]
            for seed,frame in zip(seeds,frames):
                indiv=pacientes_ensemble([frame]); metrics=binary_metrics(indiv.label,indiv.probability)
                run_rows.append({"candidate":name,"fold":fold,"seed":seed,**{k:v for k,v in metrics.items() if not isinstance(v,list)}})
            patients=pacientes_ensemble(frames)
            patients[name+"_raw"] = patients.pop("probability")
            patients[name+"_calibrated"] = apply_calibration(patients[name+"_raw"],decision["calibration"])
            patients[name+"_threshold"] = decision["threshold"]
            patients[name+"_decision"] = (patients[name+"_calibrated"] >= decision["threshold"]).astype(int)
            for scale in ("raw","calibrated"):
                m = (binary_metrics(patients.label,patients[name+"_raw"],.5) if scale=="raw" else
                     metricas_con_umbral(patients.label,patients[name+"_calibrated"],patients[name+"_threshold"]))
                fold_rows.append({"candidate":name,"fold":fold,"scale":scale,**{k:v for k,v in m.items() if not isinstance(v,(list,dict))}})
            parts.append(patients)
        assembled[name]=pd.concat(parts,ignore_index=True).sort_values("patient_id").reset_index(drop=True)
    names=[c["name"] for c in configs]
    patients=assembled[names[0]]
    for name in names[1:]:
        other=assembled[name]
        pd.testing.assert_frame_equal(patients[["patient_id","label","fold","cohort","n_slices"]],other[["patient_id","label","fold","cohort","n_slices"]])
        for col in other.columns:
            if col.startswith(name+"_"): patients[col]=other[col]
    if len(patients)!=samples.patient_id.nunique() or patients.patient_id.duplicated().any():
        raise ValueError("OOF por paciente incompleto")
    fold_frame=pd.DataFrame(fold_rows); metrics={}
    for name in names:
        metrics[name]={}
        for scale in ("raw","calibrated"):
            pooled=(binary_metrics(patients.label,patients[name+"_raw"],.5) if scale=="raw" else
                    metricas_con_umbral(patients.label,patients[name+"_calibrated"],patients[name+"_threshold"]))
            pooled["calibration"]=reliability(patients.label,patients[name+"_"+scale])
            metrics[name][scale]=pooled
            for cohort,g in patients.groupby("cohort"):
                m=(binary_metrics(g.label,g[name+"_raw"],.5) if scale=="raw" else
                   metricas_con_umbral(g.label,g[name+"_calibrated"],g[name+"_threshold"]))
                cohort_rows.append({"candidate":name,"cohort":cohort,"scale":scale,
                                   **{k:v for k,v in m.items() if not isinstance(v,(list,dict))}})
        selected=fold_frame[fold_frame.candidate.eq(name)&fold_frame.scale.eq("raw")]
        metrics[name]["mean_outer_auc"]=float(selected.roc_auc.mean())
        metrics[name]["std_outer_auc"]=float(selected.roc_auc.std(ddof=1))
    patients.to_csv(output/"oof_pacientes.csv",index=False)
    fold_frame.to_csv(output/"metricas_folds.csv",index=False)
    pd.DataFrame(cohort_rows).to_csv(output/"metricas_cohortes.csv",index=False)
    pd.DataFrame(run_rows).to_csv(output/"metricas_runs.csv",index=False)
    differences=bootstrap_generalizacion(patients,names)
    result={"status":"complete","primary_endpoint":"mean outer-fold AUC of two-seed ensemble, raw scores",
        "metrics":metrics,"paired_differences":differences,"test_images_loaded":0,
        "decisions_sha256":sha256(decisions_path),"oof_sha256":sha256(output/"oof_pacientes.csv"),
        "promotion":"none; retain reference and historical model",
        "limitations":["Evaluación interna en datos de desarrollo previamente observados.",
            "Un único holdout interno por fold; se entrena con aproximadamente 56% del desarrollo.",
            "IC bootstrap condicionados a modelos/predicciones; no incluyen incertidumbre total de selección/entrenamiento.",
            "Cinco folds comparten pacientes de aprendizaje; no son cinco experimentos independientes.",
            "AUC OOF agrupada puede cambiar por escalas distintas entre folds; criterio principal es media de AUC por fold.",
            "Calibración/umbral estimados en unas 132 pacientes por fold, con incertidumbre considerable."]}
    graficar_generalizacion(output,configs,patients)
    json_write(output/"resultado.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", nargs="?", choices=("verificar", "entrenar", "comparar", "predecir", "ajustar", "comparar-tamano"), default="verificar")
    parser.add_argument("--datos",type=Path,default=pdatos.DATOS)
    parser.add_argument("--salida",type=Path,default=None)
    parser.add_argument("--manifest",type=Path,default=SALIDA/"modelo_final.json")
    parser.add_argument("--configuraciones",nargs="+",choices=("base_raw","raw_rot90"),default=["raw_rot90"])
    parser.add_argument("--perdidas",nargs="+",choices=("normal","ponderada"),default=["normal","ponderada"])
    parser.add_argument("--semillas",nargs="+",type=int,default=[42,2026])
    parser.add_argument("--folds",nargs="+",type=int,choices=range(5),default=list(range(5)))
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--lote",type=int,default=64)
    parser.add_argument("--epocas",type=int,default=None)
    parser.add_argument("--revision-cada",type=int,default=10)
    parser.add_argument("--hasta-epoca",type=int)
    parser.add_argument("--pooling",choices=("original","intermedio"),default="original")
    parser.add_argument("--dispositivo",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--prueba",action="store_true",help="Un lote y una época; excluido de la selección")
    parser.add_argument("--paralelos",type=int,choices=(1,2,3,4),default=2)
    for phase in ("pre","early","late"):
        parser.add_argument("--"+phase,type=Path,nargs="+")
    args = parser.parse_args()
    args.epocas = (30 if args.accion in {"ajustar","comparar-tamano"} else 46) if args.epocas is None else args.epocas
    args.salida = args.salida or (SALIDA/"comparacion_tamano" if args.accion == "comparar-tamano" else
        SALIDA/"ajustes" if args.accion == "ajustar" else SALIDA/"ejecuciones")
    if args.workers < 0 or args.lote < 1 or args.epocas < 1 or args.revision_cada < 1:
        parser.error("workers >= 0, lote >= 1 y epocas >= 1")
    if args.accion in {"entrenar","ajustar","comparar-tamano"}:
        device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.dispositivo == "auto" else args.dispositivo
        if device_name == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA no está disponible")
        if args.accion == "comparar-tamano":
            if set(args.folds)!=set(range(5)) or set(args.semillas)!={42,2026}:
                parser.error("comparar-tamano utiliza los cinco folds y las semillas 42 y 2026")
            result = comparar_tamano(args.datos,args.salida,torch.device(device_name),args.epocas,
                args.revision_cada,args.lote,args.workers,args.paralelos,args.prueba)
            print(json.dumps({k:v for k,v in result.items() if k != "artifacts"},ensure_ascii=False,indent=2))
            return
        if args.accion == "ajustar":
            if args.prueba:
                parser.error("Comprueba el entrenamiento con entrenar --prueba antes de ajustar")
            if len(args.folds)!=1 or len(args.semillas)!=1:
                parser.error("ajustar requiere un --folds y una --semillas para comparación pareada")
            result=ajustar_hiperparametros(args.datos,args.salida,torch.device(device_name),args.epocas,
                args.revision_cada,args.folds[0],args.semillas[0],args.workers,args.lote)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return
        output = args.salida/"pruebas" if args.prueba else args.salida
        for name in dict.fromkeys(args.configuraciones):
            config = configuracion(name)
            if args.pooling == "intermedio":
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
        print(json.dumps(predecir_cortes(args.manifest,list(zip(args.pre,args.early,args.late))),ensure_ascii=False,indent=2))
    else:
        print(json.dumps(verificar_modelo_final(args.manifest),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
