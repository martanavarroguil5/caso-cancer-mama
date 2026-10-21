#!/usr/bin/env python3
"""CNN solo imagen, consciente de fases y con evaluación exterior protegida.

Este módulo no sustituye ni modifica el multimodal v002. Prepara un candidato
principal acorde con el caso docente: CNN 2D propia, desde cero, entrada
PRE/EARLY/LATE, media de cortes por paciente y ninguna variable clínica.

Cada fold exterior se usa una sola vez, después de seleccionar checkpoint,
calibración y umbral con una partición interna del resto de pacientes. El test
reservado nunca se carga desde este módulo.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch import nn

from . import datos as pdatos
from . import entrenamiento as base
from .paths import DATA_DIR, RESULTS_DIR


EXPERIMENT_NAME = "cnn_imagen_fases"
LOSSES = ("normal", "weighted")
SEEDS = (42, 2026)
FOLDS = tuple(range(5))
PATIENT_AGGREGATION = "mean"
EPS = 1e-7


def configuracion() -> dict:
    """Protocolo único, fijado antes del entrenamiento en GPU."""
    return {
        "name": EXPERIMENT_NAME,
        "model": {
            "architecture": "shared_phase_encoder_ordered_temporal_fusion",
            "input_phases": list(pdatos.FASES),
            "phase_encoder_shared": True,
            "stem_channels": [16, 32],
            "fusion_views": ["PRE", "EARLY", "LATE", "EARLY-PRE",
                             "LATE-PRE", "LATE-EARLY"],
            "fusion_channels": 64,
            "trunk_channels": [96, 128],
            "hidden": 64,
            "dropout": 0.40,
            "normalization": "batch",
            "clinical": None,
        },
        "training": {
            "epochs": 40,
            "min_epochs": 8,
            "patience": 8,
            "min_delta": 0.001,
            "batch_size": 32,
            "lr": 0.0008,
            "min_lr": 0.00001,
            "weight_decay": 0.001,
            "clip_grad_norm": 5.0,
            "amp": True,
            "num_workers": 4,
            "cpu_threads": 8,
            "deterministic": True,
        },
        "augmentation": {"hflip": 0.5, "rot90": True},
        "evaluation": {
            "unit": "patient",
            "slice_aggregation": PATIENT_AGGREGATION,
            "outer_protocol": "five_fold_patient_cv_with_inner_selection",
            "inner_selection_fraction": 0.15,
            "inner_split_seed": 20261014,
            "checkpoint_metric": "inner_patient_auc",
            "primary_threshold_strategy": "max_sensitivity_at_min_specificity",
            "min_specificity": 0.90,
            "secondary_threshold_strategy": "max_macro_f1",
            "test_policy": "never_load_during_development",
        },
    }


class ConvBlock(nn.Sequential):
    def __init__(self, entrada: int, salida: int):
        super().__init__(
            nn.Conv2d(entrada, salida, 3, padding=1, bias=False),
            nn.BatchNorm2d(salida),
            nn.ReLU(inplace=True),
            nn.Conv2d(salida, salida, 3, padding=1, bias=False),
            nn.BatchNorm2d(salida),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )


class CNNImagenFases(nn.Module):
    """CNN desde cero: codifica cada fase por separado con pesos compartidos.

    Las fases no se reinterpretan como RGB. El mismo encoder monocanal procesa
    PRE, EARLY y LATE; después se fusionan en orden sus características y tres
    diferencias temporales. Compartir el encoder limita los parámetros.
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = deepcopy(config or configuracion()["model"])
        self.config = self.config.get("model", self.config)
        if self.config.get("architecture") != "shared_phase_encoder_ordered_temporal_fusion":
            raise ValueError("Arquitectura de imagen desconocida")
        if self.config.get("input_phases") != list(pdatos.FASES):
            raise ValueError("Las fases deben estar ordenadas como PRE, EARLY y LATE")
        if self.config.get("phase_encoder_shared") is not True:
            raise ValueError("El protocolo requiere un encoder compartido")
        if self.config.get("clinical") is not None:
            raise ValueError("La CNN principal no admite variables clínicas")
        if self.config.get("normalization") != "batch":
            raise ValueError("El protocolo fijado utiliza BatchNorm")

        stem = self.config.get("stem_channels")
        trunk = self.config.get("trunk_channels")
        if (not isinstance(stem, list) or len(stem) != 2 or
                not isinstance(trunk, list) or len(trunk) != 2 or
                any(not isinstance(value, int) or value < 1 for value in [*stem, *trunk])):
            raise ValueError("Se requieren dos anchos positivos para encoder y tronco")
        fusion_channels = int(self.config.get("fusion_channels", 64))
        hidden = int(self.config.get("hidden", 64))
        dropout = float(self.config.get("dropout", 0.40))
        if fusion_channels < 1 or hidden < 1 or not 0 <= dropout < 1:
            raise ValueError("Anchuras o dropout inválidos")

        self.phase_encoder = nn.Sequential(
            ConvBlock(1, stem[0]),
            ConvBlock(stem[0], stem[1]),
        )
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(stem[-1] * 6, fusion_channels, 1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            ConvBlock(fusion_channels, trunk[0]),
            ConvBlock(trunk[0], trunk[1]),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = base.GlobalMaxPool()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(trunk[-1] * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def fusion_views(encoded: torch.Tensor) -> torch.Tensor:
        """Devuelve vistas ordenadas para [N, fase, C, H, W]."""
        if encoded.ndim != 5 or encoded.shape[1] != 3:
            raise ValueError("Se esperan características [N,3,C,H,W]")
        pre, early, late = encoded[:, 0], encoded[:, 1], encoded[:, 2]
        return torch.cat((pre, early, late, early-pre, late-pre, late-early), dim=1)

    def encode_phases(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError(f"Esperado [N,3,256,256], recibido {tuple(x.shape)}")
        n = len(x)
        phases = (2*x-1).reshape(n*3, 1, 256, 256)
        encoded = self.phase_encoder(phases)
        return encoded.reshape(n, 3, *encoded.shape[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_phases(x)
        z = self.temporal_fusion(self.fusion_views(encoded))
        z = self.trunk(z)
        z = torch.cat((self.avgpool(z).flatten(1), self.maxpool(z).flatten(1)), dim=1)
        return self.classifier(z)


def numero_parametros(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def particion_anidada(samples: pd.DataFrame, outer_fold: int, config: dict | None = None):
    """Separa fit/selección/outer sin compartir pacientes.

    La semilla de la selección depende del fold, no de la semilla del modelo;
    así normal/ponderada y ambas inicializaciones usan las mismas pacientes.
    """
    config = config or configuracion()
    if outer_fold not in FOLDS or not samples.split.eq("train").all():
        raise ValueError("La partición anidada solo admite train y folds 0-4")
    outer = samples.loc[samples.fold.eq(outer_fold)].copy().reset_index(drop=True)
    development = samples.loc[samples.fold.ne(outer_fold)].copy().reset_index(drop=True)
    patients = development.drop_duplicates("patient_id")[
        ["patient_id", "pCR", "dataset"]].sort_values("patient_id").reset_index(drop=True)
    strata = patients.dataset.astype(str) + "|" + patients.pCR.astype(str)
    counts = strata.value_counts()
    if len(counts) < 2 or (counts < 2).any():
        raise ValueError("No hay suficientes pacientes por cohorte y clase para selección interna")
    fraction = float(config["evaluation"]["inner_selection_fraction"])
    n_selection = max(len(counts), int(round(len(patients) * fraction)))
    if not 0 < fraction < 0.5 or n_selection > len(patients)-len(counts):
        raise ValueError("Fracción de selección interna inválida")
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=n_selection,
        random_state=int(config["evaluation"]["inner_split_seed"]) + outer_fold,
    )
    fit_index, selection_index = next(splitter.split(patients.patient_id, strata))
    fit_ids = set(patients.iloc[fit_index].patient_id)
    selection_ids = set(patients.iloc[selection_index].patient_id)
    fit = development[development.patient_id.isin(fit_ids)].copy().reset_index(drop=True)
    selection = development[development.patient_id.isin(selection_ids)].copy().reset_index(drop=True)
    patient_sets = [set(frame.patient_id) for frame in (fit, selection, outer)]
    if any(patient_sets[i] & patient_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Fuga de pacientes en la partición anidada")
    if any(frame.empty or frame.pCR.nunique() != 2 for frame in (fit, selection, outer)):
        raise ValueError("Fit, selección y outer deben contener ambas clases")
    if not set(fit.fold).issubset(set(FOLDS)-{outer_fold}) or not set(selection.fold).issubset(
            set(FOLDS)-{outer_fold}) or set(outer.fold) != {outer_fold}:
        raise RuntimeError("El fold exterior entró en fit o selección")
    return fit, selection, outer


def _small_balanced(frame: pd.DataFrame, patients_per_class: int = 2) -> pd.DataFrame:
    patients = (frame.drop_duplicates("patient_id").sort_values("patient_id")
                .groupby("pCR", group_keys=False).head(patients_per_class))
    return frame[frame.patient_id.isin(patients.patient_id)].reset_index(drop=True)


def patient_predictions(samples: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    if len(samples) != len(probabilities):
        raise ValueError("Predicciones y muestras no están alineadas")
    frame = samples[["sample_id", "patient_id", "pCR", "fold", "dataset"]].copy()
    frame["probability_raw"] = np.asarray(probabilities, dtype=float)
    if not np.isfinite(frame.probability_raw).all() or not frame.probability_raw.between(0, 1).all():
        raise ValueError("Probabilidades no finitas o fuera de [0,1]")
    return (frame.groupby("patient_id", sort=True)
            .agg(label=("pCR", "first"), fold=("fold", "first"),
                 cohort=("dataset", "first"), probability_raw=("probability_raw", "mean"),
                 n_slices=("sample_id", "count")).reset_index())


def metricas_binarias(y, probability, threshold: float = 0.5) -> dict:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if (len(y) == 0 or len(y) != len(probability) or set(y) != {0, 1} or
            not np.isfinite(probability).all() or
            ((probability < 0) | (probability > 1)).any() or not 0 <= threshold <= 1):
        raise ValueError("Etiquetas, probabilidades o umbral inválidos")
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel().tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=[0, 1], zero_division=0)
    return {
        "n_patients": int(len(y)),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, np.clip(probability, EPS, 1-EPS), labels=[0, 1])),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision_no_pcr": float(precision[0]),
        "recall_no_pcr": float(recall[0]),
        "f1_no_pcr": float(f1[0]),
        "precision_pcr": float(precision[1]),
        "recall_pcr": float(recall[1]),
        "f1_pcr": float(f1[1]),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, prediction, average="weighted", zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "support_no_pcr": int(support[0]), "support_pcr": int(support[1]),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _threshold_candidates(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    candidates = [0.0, 0.5, 1.0, *np.unique(probability).tolist()]
    all_negative = float(np.nextafter(probability.max(), np.inf))
    if all_negative <= 1:
        candidates.append(all_negative)
    return np.unique(np.asarray(candidates, dtype=float))


def threshold_min_specificity(y, probability, min_specificity: float = 0.90) -> float:
    """Máxima sensibilidad entre umbrales con especificidad mínima."""
    if not 0 < min_specificity <= 1:
        raise ValueError("min_specificity debe pertenecer a (0,1]")
    feasible = []
    for threshold in _threshold_candidates(probability):
        metrics = metricas_binarias(y, probability, float(threshold))
        if metrics["recall_no_pcr"] + 1e-12 >= min_specificity:
            feasible.append(metrics)
    if not feasible:
        raise RuntimeError("No existe un umbral con la especificidad solicitada")
    best = max(feasible, key=lambda item: (
        item["recall_pcr"], item["precision_pcr"], item["recall_no_pcr"], item["threshold"]))
    return float(best["threshold"])


def threshold_macro_f1(y, probability) -> float:
    metrics = [metricas_binarias(y, probability, float(value))
               for value in _threshold_candidates(probability)]
    best = max(metrics, key=lambda item: (
        item["macro_f1"], item["balanced_accuracy"], item["recall_no_pcr"],
        -abs(item["threshold"]-0.5)))
    return float(best["threshold"])


@torch.inference_mode()
def predict_slices(model, loader, device, criterion=None):
    model.eval()
    probabilities = []
    loss_sum = torch.zeros((), device=device)
    count = 0
    for batch in loader:
        x, y = base.batch_device(batch, device)
        logits = model(x).squeeze(1)
        if criterion is not None:
            loss_sum += criterion(logits, y) * len(y)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        count += len(y)
    return np.concatenate(probabilities), (float(loss_sum/count) if criterion is not None else None)


def train_epoch(model, loader, optimizer, scaler, criterion, device, amp,
                augmentation, augmentation_generator, clip_norm, max_batches=None):
    model.train()
    total = torch.zeros((), device=device)
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x, y = base.batch_device(batch, device)
        x = base.aumentar_geometria(x.clone(), augmentation_generator, **augmentation)
        optimizer.zero_grad(set_to_none=True)
        with base.amp_context(device, amp):
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Pérdida no finita")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=not amp)
        scaler.step(optimizer)
        scaler.update()
        total += loss.detach()*len(y)
        count += len(y)
    if count == 0:
        raise RuntimeError("El loader no produjo ningún lote")
    return float(total/count)


def _patient_auc(samples, probabilities) -> float:
    patients = patient_predictions(samples, probabilities)
    return float(roc_auc_score(patients.label, patients.probability_raw))


def graficar_historia(run_dir: Path,
                      title: str = "CNN solo imagen: fit frente a selección; outer no interviene"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = pd.read_csv(run_dir/"history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(history.epoch, history.fit_eval_loss, label="Fit, evaluación")
    axes[0].plot(history.epoch, history.selection_loss, label="Selección interna")
    axes[0].set(title="Pérdida por corte", xlabel="Época", ylabel="BCE")
    axes[1].plot(history.epoch, history.fit_patient_auc, label="Fit")
    axes[1].plot(history.epoch, history.selection_patient_auc, label="Selección interna")
    axes[1].set(title="AUC por paciente", xlabel="Época", ylabel="ROC-AUC", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(run_dir/"curvas_train_seleccion.png", dpi=160)
    plt.close(fig)


def _loader(samples, root, batch_size, workers, seed, shuffle=False):
    return base.make_loader(samples, root, batch_size, workers,
                            torch.Generator().manual_seed(seed), shuffle=shuffle,
                            clinical=False)


def run_training(config: dict, loss_name: str, seed: int, outer_fold: int,
                 root: Path, output: Path, device: torch.device, smoke=False,
                 model_type=CNNImagenFases, experiment_name: str | None = None,
                 extra_source_files: dict[str, Path] | None = None,
                 evaluate_outer: bool = True):
    """Entrena sin consultar el fold exterior hasta fijar el checkpoint."""
    config = deepcopy(config)
    if loss_name not in LOSSES or seed not in SEEDS or outer_fold not in FOLDS:
        raise ValueError("Pérdida, semilla o fold fuera del protocolo")
    experiment_name = experiment_name or EXPERIMENT_NAME
    if config.get("name") != experiment_name:
        raise ValueError("Configuración de experimento inesperada")
    config.update(loss=loss_name, seed=seed, outer_fold=outer_fold, smoke=bool(smoke),
                  device_type=device.type)
    if not evaluate_outer:
        config["selection_only"] = True
    config["source_sha256"] = {
        "entrenamiento_imagen.py": base.sha256(Path(__file__)),
        "datos.py": base.sha256(Path(pdatos.__file__)),
    }
    for name, path in (extra_source_files or {}).items():
        config["source_sha256"][name] = base.sha256(Path(path))
    training = config["training"]
    torch.set_num_threads(int(training["cpu_threads"]))
    base.seed_everything(seed, bool(training["deterministic"]))
    all_train = base.cargar_train(root, include_clinical=False)
    config["data_signature"] = hashlib.sha256(all_train.to_csv(index=False).encode()).hexdigest()
    fit, selection, outer = particion_anidada(all_train, outer_fold, config)
    if smoke:
        fit, selection, outer = (_small_balanced(frame) for frame in (fit, selection, outer))
    config["counts"] = {
        "fit_slices": int(len(fit)), "fit_patients": int(fit.patient_id.nunique()),
        "selection_slices": int(len(selection)),
        "selection_patients": int(selection.patient_id.nunique()),
        "outer_slices": int(len(outer)), "outer_patients": int(outer.patient_id.nunique()),
    }
    config["pos_weight"] = (float(pdatos.pos_weight_cortes(fit))
                            if loss_name == "weighted" else 1.0)

    run_dir = (Path(output)/experiment_name/loss_name/f"seed_{seed}"/f"fold_{outer_fold}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir/"config.json"
    if config_path.exists() and base.config_hash(json.loads(config_path.read_text())) != base.config_hash(config):
        raise ValueError(f"Configuración distinta en {run_dir}; usa otra salida")
    base.json_write(config_path, config)
    if (run_dir/"summary.json").exists():
        summary = json.loads((run_dir/"summary.json").read_text())
        if summary.get("config_hash") != base.config_hash(config):
            raise ValueError("El resumen no coincide con la configuración")
        if summary.get("status") == "complete":
            artifacts = [("inference.pt", "inference_sha256"),
                         ("selection_patients.csv", "selection_sha256")]
            if summary.get("outer_evaluated", True):
                artifacts.append(("outer_patients.csv", "outer_sha256"))
            for filename, key in artifacts:
                if base.sha256(run_dir/filename) != summary[key]:
                    raise ValueError(f"Artefacto modificado: {run_dir/filename}")
            print(f"Ya completado: {run_dir}", flush=True)
            return summary
    if not (run_dir/"environment.json").exists():
        base.json_write(run_dir/"environment.json", base.environment(device))

    workers = 0 if smoke else int(training["num_workers"])
    batch_size = min(8, int(training["batch_size"])) if smoke else int(training["batch_size"])
    train_generator = torch.Generator().manual_seed(seed + outer_fold*1000)
    augmentation_generator = torch.Generator().manual_seed(seed + outer_fold*1000 + 1)
    train_loader = base.make_loader(fit, root, batch_size, workers, train_generator,
                                    shuffle=True, clinical=False)
    fit_loader = _loader(fit, root, batch_size, workers, seed+50000+outer_fold)
    selection_loader = _loader(selection, root, batch_size, workers, seed+60000+outer_fold)
    outer_loader = (_loader(outer, root, batch_size, workers, seed+70000+outer_fold)
                    if evaluate_outer else None)

    model = model_type(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["lr"],
                                  weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training["epochs"], eta_min=training["min_lr"])
    amp = bool(training["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config["pos_weight"], device=device))
    start_epoch, best_auc, best_epoch, bad_epochs, history = 0, -1.0, -1, 0, []
    if (run_dir/"last.pt").exists():
        loaded = base.load_checkpoint(run_dir/"last.pt", model, optimizer, scheduler, scaler,
                                      train_generator, augmentation_generator)
        if base.config_hash(loaded["config"]) != base.config_hash(config):
            raise ValueError("Checkpoint incompatible")
        start_epoch = int(loaded["epoch"])+1
        best_auc, best_epoch, bad_epochs = (float(loaded["best_auc"]),
                                            int(loaded["best_epoch"]),
                                            int(loaded["bad_epochs"]))
        history = loaded["history"]
        print(f"Reanudando {run_dir} en época {start_epoch+1}", flush=True)

    epochs_limit = 1 if smoke else int(training["epochs"])
    print(f"Run {run_dir}: {numero_parametros(model):,} parámetros; "
          f"fit={config['counts']['fit_patients']} pacientes, "
          f"selección={config['counts']['selection_patients']}, "
          f"outer={config['counts']['outer_patients']}; test=0", flush=True)
    wall_start = time.perf_counter()
    for epoch in range(start_epoch, epochs_limit):
        # Una interrupción después de guardar last.pt no debe añadir una época
        # si ese checkpoint ya había alcanzado la parada temprana.
        if (not smoke and epoch >= int(training["min_epochs"]) and
                bad_epochs >= int(training["patience"])):
            break
        tic = time.perf_counter()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, criterion, device, amp,
            config["augmentation"], augmentation_generator, training["clip_grad_norm"],
            max_batches=1 if smoke else None)
        fit_probability, fit_loss = predict_slices(model, fit_loader, device, criterion)
        selection_probability, selection_loss = predict_slices(
            model, selection_loader, device, criterion)
        fit_auc = _patient_auc(fit, fit_probability)
        selection_auc = _patient_auc(selection, selection_probability)
        scheduler.step()
        previous_best = best_auc
        improved = selection_auc > best_auc + 1e-12
        substantial = selection_auc > best_auc + float(training["min_delta"])
        bad_epochs = 0 if substantial else bad_epochs+1
        if improved:
            best_auc, best_epoch = selection_auc, epoch
        row = {
            "epoch": epoch+1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_augmented_loss": train_loss,
            "fit_eval_loss": fit_loss,
            "selection_loss": selection_loss,
            "fit_patient_auc": fit_auc,
            "selection_patient_auc": selection_auc,
            "auc_gap": fit_auc-selection_auc,
            "best_selection_auc_before_epoch": previous_best,
            "bad_epochs": bad_epochs,
            "epoch_seconds": time.perf_counter()-tic,
        }
        history.append(row)
        if improved:
            base.save_checkpoint(run_dir/"best.pt", model, optimizer, scheduler, scaler,
                epoch, best_auc, bad_epochs, config, train_generator,
                augmentation_generator, history, best_epoch)
        base.save_checkpoint(run_dir/"last.pt", model, optimizer, scheduler, scaler,
            epoch, best_auc, bad_epochs, config, train_generator,
            augmentation_generator, history, best_epoch)
        base.history_write(run_dir/"history.csv", history)
        graficar_historia(run_dir, config.get(
            "history_plot_title",
            "CNN solo imagen: fit frente a selección; outer no interviene",
        ))
        print(f"Epoch {epoch+1:02d}/{epochs_limit}: fitAUC={fit_auc:.4f} "
              f"selecciónAUC={selection_auc:.4f} brecha={fit_auc-selection_auc:+.4f} "
              f"selecciónLoss={selection_loss:.4f}", flush=True)
        if (not smoke and epoch+1 >= int(training["min_epochs"]) and
                bad_epochs >= int(training["patience"])):
            break
    if not (run_dir/"best.pt").exists():
        raise RuntimeError("No se generó un checkpoint seleccionable")

    base.load_checkpoint(run_dir/"best.pt", model)
    selection_probability, selection_loss = predict_slices(model, selection_loader, device, criterion)
    selection_patients = patient_predictions(selection, selection_probability)
    selection_patients.to_csv(run_dir/"selection_patients.csv", index=False)
    outer_loss, outer_metrics, outer_sha256 = None, None, None
    if evaluate_outer:
        # En el ajuste anidado esta consulta se difiere hasta fijar parámetros.
        outer_probability, outer_loss = predict_slices(model, outer_loader, device, criterion)
        outer_patients = patient_predictions(outer, outer_probability)
        outer_patients.to_csv(run_dir/"outer_patients.csv", index=False)
        outer_slices = outer[["sample_id", "patient_id", "pCR", "fold", "dataset"]].copy()
        outer_slices["probability_raw"] = outer_probability
        outer_slices.to_csv(run_dir/"outer_slices.csv", index=False)
        outer_metrics = metricas_binarias(
            outer_patients.label, outer_patients.probability_raw, 0.5)
        outer_sha256 = base.sha256(run_dir/"outer_patients.csv")
    base.export_inference_checkpoint(run_dir)

    summary = {
        "schema_version": 1,
        "status": "complete",
        "eligible_for_selection": bool(not smoke and evaluate_outer),
        "outer_evaluated": bool(evaluate_outer),
        "smoke": bool(smoke),
        "config_hash": base.config_hash(config),
        "loss": loss_name,
        "seed": seed,
        "outer_fold": outer_fold,
        "best_epoch": best_epoch+1,
        "epochs_completed": len(history),
        "best_inner_patient_auc": best_auc,
        "inner_selection_loss": selection_loss,
        "outer_loss": outer_loss,
        "outer_raw_metrics_at_0_5": outer_metrics,
        "parameters": numero_parametros(model),
        "total_train_seconds": float(sum(row["epoch_seconds"] for row in history)),
        "current_session_seconds": time.perf_counter()-wall_start,
        "test_images_loaded": 0,
        "inference_sha256": base.sha256(run_dir/"inference.pt"),
        "outer_sha256": outer_sha256,
        "selection_sha256": base.sha256(run_dir/"selection_patients.csv"),
    }
    base.json_write(run_dir/"summary.json", summary)
    return summary


def _aligned_average(frames: list[pd.DataFrame]) -> pd.DataFrame:
    ordered = [frame.sort_values("patient_id").reset_index(drop=True) for frame in frames]
    reference = ordered[0]
    for other in ordered[1:]:
        for column in ("patient_id", "label", "fold", "cohort", "n_slices"):
            if not np.array_equal(reference[column].to_numpy(), other[column].to_numpy()):
                raise ValueError(f"Semillas desalineadas en {column}")
    result = reference.drop(columns="probability_raw").copy()
    result["probability_raw"] = np.mean(
        [frame.probability_raw.to_numpy(dtype=float) for frame in ordered], axis=0)
    return result


def _fold_candidate(loss_name: str, fold: int, runs: dict,
                    protocol: dict | None = None) -> tuple[dict, pd.DataFrame]:
    selection = _aligned_average([
        pd.read_csv(runs[loss_name, seed, fold]/"selection_patients.csv") for seed in SEEDS])
    outer = _aligned_average([
        pd.read_csv(runs[loss_name, seed, fold]/"outer_patients.csv") for seed in SEEDS])
    calibration = base.fit_platt(selection.label.to_numpy(), selection.probability_raw.to_numpy())
    calibration["fit_data"] = f"inner_selection_outer_fold_{fold}"
    selection["probability_calibrated"] = base.apply_calibration(
        selection.probability_raw.to_numpy(), calibration)
    outer["probability_calibrated"] = base.apply_calibration(
        outer.probability_raw.to_numpy(), calibration)
    min_specificity = (protocol or configuracion())["evaluation"]["min_specificity"]
    threshold = threshold_min_specificity(
        selection.label, selection.probability_calibrated, min_specificity)
    balanced_threshold = threshold_macro_f1(selection.label, selection.probability_calibrated)
    outer["threshold"] = threshold
    outer["prediction"] = (outer.probability_calibrated >= threshold).astype(int)
    result = {
        "fold": fold,
        "calibration": calibration,
        "primary_threshold": threshold,
        "secondary_macro_f1_threshold": balanced_threshold,
        "selection_primary_metrics": metricas_binarias(
            selection.label, selection.probability_calibrated, threshold),
        "outer_primary_metrics": metricas_binarias(
            outer.label, outer.probability_calibrated, threshold),
        "outer_secondary_metrics": metricas_binarias(
            outer.label, outer.probability_calibrated, balanced_threshold),
        "outer_raw_auc": float(roc_auc_score(outer.label, outer.probability_raw)),
    }
    return result, outer


def graficar_comparacion(output: Path, runs: dict,
                         title: str = "CNN solo imagen: mediana e IQR de semillas y folds"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {"normal": "#2764c4", "weighted": "#e68a00"}
    for loss_name in LOSSES:
        histories = []
        for seed in SEEDS:
            for fold in FOLDS:
                frame = pd.read_csv(runs[loss_name, seed, fold]/"history.csv")
                frame["run"] = f"{seed}-{fold}"
                histories.append(frame)
        history = pd.concat(histories, ignore_index=True)
        grouped = history.groupby("epoch")
        for metric, axis in (("selection_loss", axes[0]), ("selection_patient_auc", axes[1])):
            median = grouped[metric].median()
            q1 = grouped[metric].quantile(.25)
            q3 = grouped[metric].quantile(.75)
            axis.plot(median.index, median, color=colors[loss_name], label=loss_name)
            axis.fill_between(median.index, q1, q3, color=colors[loss_name], alpha=.18)
    axes[0].set(title="Pérdida en selección interna", xlabel="Época", ylabel="BCE")
    axes[1].set(title="AUC en selección interna", xlabel="Época", ylabel="ROC-AUC", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(title="Pérdida")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(Path(output)/"comparacion/curvas_train_validacion.png", dpi=170)
    plt.close(fig)


def comparar_ejecuciones(root: Path, output: Path, *, protocol: dict | None = None,
                         experiment_name: str | None = None,
                         model_class_name: str = "CNNImagenFases",
                         primary_model: str = "image_only_cnn_from_scratch",
                         preprocessing: dict | None = None,
                         figure_title: str | None = None) -> dict:
    """Compara BCE sin usar test y con umbrales ajustados solo internamente."""
    protocol = deepcopy(protocol or configuracion())
    experiment_name = experiment_name or EXPERIMENT_NAME
    if protocol.get("name") != experiment_name:
        raise ValueError("El protocolo no coincide con el experimento")
    all_train = base.cargar_train(root, include_clinical=False)
    data_signature = hashlib.sha256(all_train.to_csv(index=False).encode()).hexdigest()
    runs = {}
    for summary_path in sorted((Path(output)/experiment_name).rglob("summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or not summary.get("eligible_for_selection"):
            continue
        run_dir = summary_path.parent
        config = json.loads((run_dir/"config.json").read_text())
        if (config.get("name") != experiment_name or config.get("data_signature") != data_signature or
                summary.get("config_hash") != base.config_hash(config)):
            raise ValueError(f"Run incompatible o modificado: {run_dir}")
        key = (config["loss"], int(config["seed"]), int(config["outer_fold"]))
        if key in runs:
            raise ValueError(f"Run duplicado: {key}")
        for filename, hash_key in (("inference.pt", "inference_sha256"),
                                   ("outer_patients.csv", "outer_sha256"),
                                   ("selection_patients.csv", "selection_sha256")):
            if base.sha256(run_dir/filename) != summary[hash_key]:
                raise ValueError(f"Artefacto modificado: {run_dir/filename}")
        runs[key] = run_dir
    expected = {(loss_name, seed, fold) for loss_name in LOSSES
                for seed in SEEDS for fold in FOLDS}
    if set(runs) != expected:
        missing = sorted(expected-set(runs))
        raise ValueError(f"Se requieren 20 runs completos; faltan {missing}")

    candidates = []
    oof_by_loss = {}
    folds_by_loss = {}
    for loss_name in LOSSES:
        fold_results, outer_frames = [], []
        for fold in FOLDS:
            fold_result, outer = _fold_candidate(loss_name, fold, runs, protocol)
            fold_results.append(fold_result)
            outer_frames.append(outer)
        oof = pd.concat(outer_frames, ignore_index=True).sort_values("patient_id").reset_index(drop=True)
        if set(oof.patient_id) != set(all_train.patient_id):
            raise ValueError("OOF exterior incompleto")
        fold_auc = [item["outer_raw_auc"] for item in fold_results]
        primary = metricas_binarias(oof.label, oof.probability_calibrated,
                                    0.5)  # decisiones reales se agregan debajo
        predictions = oof.prediction.to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(oof.label, predictions, labels=[0, 1]).ravel().tolist()
        precision, recall, f1, support = precision_recall_fscore_support(
            oof.label, predictions, labels=[0, 1], zero_division=0)
        crossfit_decision = {
            "accuracy": float(accuracy_score(oof.label, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(oof.label, predictions)),
            "precision_no_pcr": float(precision[0]), "recall_no_pcr": float(recall[0]),
            "f1_no_pcr": float(f1[0]), "precision_pcr": float(precision[1]),
            "recall_pcr": float(recall[1]), "f1_pcr": float(f1[1]),
            "macro_f1": float(f1_score(oof.label, predictions, average="macro", zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "support_no_pcr": int(support[0]), "support_pcr": int(support[1]),
            "threshold": "fold_specific_inner_validation",
        }
        candidate = {
            "loss": loss_name,
            "fold_raw_auc": fold_auc,
            "mean_fold_raw_auc": float(np.mean(fold_auc)),
            "std_fold_raw_auc": float(np.std(fold_auc, ddof=1)),
            "pooled_raw_auc_descriptive": float(roc_auc_score(oof.label, oof.probability_raw)),
            "crossfit_calibrated_auc": primary["roc_auc"],
            "crossfit_calibrated_average_precision": primary["average_precision"],
            "crossfit_calibrated_brier": primary["brier"],
            "crossfit_calibrated_log_loss": primary["log_loss"],
            "crossfit_primary_decision": crossfit_decision,
        }
        candidates.append(candidate)
        oof_by_loss[loss_name] = oof
        folds_by_loss[loss_name] = fold_results

    selected = max(candidates, key=lambda item: (
        item["mean_fold_raw_auc"], item["crossfit_primary_decision"]["macro_f1"],
        -item["crossfit_calibrated_brier"], item["loss"] == "normal"))
    selected_loss = selected["loss"]
    selected_oof = oof_by_loss[selected_loss]
    final_calibration = base.fit_platt(
        selected_oof.label.to_numpy(), selected_oof.probability_raw.to_numpy())
    final_calibration["fit_data"] = "all_development_outer_oof_for_future_independent_test_only"
    final_probability = base.apply_calibration(
        selected_oof.probability_raw.to_numpy(), final_calibration)
    final_threshold = threshold_min_specificity(
        selected_oof.label, final_probability,
        protocol["evaluation"]["min_specificity"])
    always_no_pcr = metricas_binarias(
        selected_oof.label, np.zeros(len(selected_oof)), 0.5)

    destination = Path(output)/"comparacion"
    destination.mkdir(parents=True, exist_ok=True)
    for loss_name, frame in oof_by_loss.items():
        frame.to_csv(destination/f"oof_pacientes_{loss_name}.csv", index=False)
    records = []
    for candidate in candidates:
        records.append({key: value for key, value in candidate.items()
                        if key not in {"fold_raw_auc", "crossfit_primary_decision"}} |
                       {"fold_raw_auc": json.dumps(candidate["fold_raw_auc"]),
                        **{f"decision_{key}": value for key, value in
                           candidate["crossfit_primary_decision"].items()}})
    pd.DataFrame(records).to_csv(destination/"comparacion_perdidas.csv", index=False)
    report = {
        "status": "desarrollo_sin_test",
        "primary_model": primary_model,
        "clinical_variables_used": False,
        "patient_aggregation": PATIENT_AGGREGATION,
        "warning": ("La pérdida se elige con validación cruzada de desarrollo; sus métricas "
                    "no sustituyen un test independiente."),
        "candidates": candidates,
        "selected_loss_for_future_test": selected_loss,
        "selection_rule": "mean fold raw AUC; macro-F1; Brier; prefer normal on exact tie",
        "fold_details": folds_by_loss[selected_loss],
        "always_no_pcr_baseline": always_no_pcr,
        "future_independent_test_policy": {
            "calibration": final_calibration,
            "threshold": final_threshold,
            "strategy": "max_sensitivity_at_min_specificity",
            "min_specificity": protocol["evaluation"]["min_specificity"],
            "fit_data": "development OOF; do not report apparent decision metrics as test",
        },
        "test_images_loaded": 0,
    }
    base.json_write(destination/"seleccion.json", report)
    model_entries = []
    for seed in SEEDS:
        for fold in FOLDS:
            run_dir = runs[selected_loss, seed, fold]
            artifact = torch.load(run_dir/"inference.pt", map_location="cpu", weights_only=True)
            model_entries.append({
                "path": os.path.relpath((run_dir/"inference.pt").resolve(), destination.resolve()),
                "sha256": base.sha256(run_dir/"inference.pt"),
                "seed": seed, "outer_fold": fold,
                "model_config": artifact["model_config"],
            })
    manifest = {
        "schema_version": 1,
        "status": "desarrollo_sin_test",
        "model_class": model_class_name,
        "models": model_entries,
        "loss": selected_loss,
        "slice_aggregation": PATIENT_AGGREGATION,
        "ensemble": "mean probabilities across ten models",
        "calibration": final_calibration,
        "threshold": final_threshold,
        "threshold_policy": report["future_independent_test_policy"],
        "preprocessing": preprocessing or {
            "phases": list(pdatos.FASES), "shape": [3, 256, 256],
            "phase_semantics": "ordered DCE times, never RGB",
            "scaling": "uint8/255 followed by 2*x-1 inside each shared phase encoder",
            "clinical_required": False,
        },
        "test_evaluated_once": False,
    }
    base.json_write(destination/"modelo_desarrollo_imagen.json", manifest)
    (destination/"modelo_desarrollo_imagen.json.sha256").write_text(
        base.sha256(destination/"modelo_desarrollo_imagen.json")+"\n")
    graficar_comparacion(output, runs, figure_title or
                         "CNN solo imagen: mediana e IQR de semillas y folds")
    return report


def verificar() -> dict:
    config = configuracion()
    model = CNNImagenFases(config)
    return {
        "status": "preparado_sin_entrenar",
        "configuration": config["name"],
        "parameters": numero_parametros(model),
        "clinical_variables": False,
        "phase_encoder_shared": True,
        "patient_aggregation": PATIENT_AGGREGATION,
        "losses": list(LOSSES),
        "seeds": list(SEEDS),
        "outer_folds": list(FOLDS),
        "planned_runs": len(LOSSES)*len(SEEDS)*len(FOLDS),
        "test_images_loaded": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", nargs="?", choices=("verificar", "entrenar", "comparar"),
                        default="verificar")
    parser.add_argument("--datos", type=Path, default=DATA_DIR)
    parser.add_argument("--salida", type=Path,
                        default=RESULTS_DIR/"04_entrenamiento/cnn_imagen_fases_20261014")
    parser.add_argument("--perdidas", nargs="+", choices=LOSSES, default=list(LOSSES))
    parser.add_argument("--semillas", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--folds", nargs="+", type=int, choices=FOLDS, default=list(FOLDS))
    parser.add_argument("--epocas", type=int, default=40)
    parser.add_argument("--lote", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dispositivo", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--prueba", action="store_true",
                        help="Un lote/una época; nunca elegible para selección")
    args = parser.parse_args()
    if args.epocas < 1 or args.lote < 1 or args.workers < 0:
        parser.error("epocas y lote deben ser positivos; workers no negativo")
    if args.accion == "verificar":
        print(json.dumps(verificar(), ensure_ascii=False, indent=2))
        return
    if args.accion == "comparar":
        report = comparar_ejecuciones(args.datos, args.salida)
        print(json.dumps({"selected_loss": report["selected_loss_for_future_test"],
                          "test_images_loaded": 0}, ensure_ascii=False, indent=2))
        return
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.dispositivo == "auto" else args.dispositivo
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA no está disponible")
    config = configuracion()
    config["training"].update(epochs=args.epocas, batch_size=args.lote,
                              num_workers=args.workers,
                              min_epochs=min(config["training"]["min_epochs"], args.epocas))
    output = args.salida/"pruebas" if args.prueba else args.salida
    for loss_name in dict.fromkeys(args.perdidas):
        for seed in dict.fromkeys(args.semillas):
            for fold in dict.fromkeys(args.folds):
                run_training(config, loss_name, seed, fold, args.datos, output,
                             torch.device(device_name), smoke=args.prueba)


if __name__ == "__main__":
    main()
