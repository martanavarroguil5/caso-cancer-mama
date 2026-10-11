"""Ensayo FiLM frente a la CNN v002, desde cero y con selección interna.

V002 permanece activo. No se usa test ni se cambia el modelo automáticamente.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from . import entrenamiento as base
from .paths import PROJECT_ROOT, RESULTS_DIR


ARMS = {"control_weighted": {"film": False, "loss": "weighted"},
        "film_weighted": {"film": True, "loss": "weighted"},
        "control_normal": {"film": False, "loss": "normal"},
        "film_normal": {"film": True, "loss": "normal"}}
CRITERIA = ("auc",)
DEFAULT_OUTPUT = RESULTS_DIR / "04_entrenamiento/film_multimodal_20261011"


def now():
    return datetime.now(timezone.utc).isoformat()


def settings(smoke=False, workers=4):
    config = base.configuracion()
    return {"schema_version": 1, "model": config["model"],
            "training": dict(config["training"], num_workers=workers,
                             epochs=1 if smoke else 46, min_epochs=1 if smoke else 12),
            "augmentation": config["augmentation"], "smoke": smoke,
            "inner_fraction": 0.15, "split_seed": 20261010,
            "aggregation": "median", "checkpoint_metric": "patient_auc"}


def signature(frame):
    return hashlib.sha256(frame.sort_values("sample_id").to_csv(index=False).encode()).hexdigest()


def split_patients(samples, fold, fraction=0.15, split_seed=20261010):
    """Fija la selección interna sin acceder a etiquetas del fold exterior."""
    available, outer = base.particion(samples, fold)
    patients = available.drop_duplicates("patient_id").sort_values("patient_id")
    strata = patients.dataset.astype(str) + ":" + patients.pCR.astype(str)
    fit_ids, select_ids = train_test_split(
        patients.patient_id.to_numpy(), test_size=fraction, random_state=split_seed + fold,
        stratify=strata.to_numpy())
    fit = available[available.patient_id.isin(fit_ids)].copy().reset_index(drop=True)
    select = available[available.patient_id.isin(select_ids)].copy().reset_index(drop=True)
    sets = [set(frame.patient_id) for frame in (fit, select, outer)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Solape de pacientes entre fit, selección y evaluación")
    for frame in (fit, select, outer):
        if set(frame.pCR) != {0, 1}:
            raise ValueError("Cada partición requiere ambas clases")
    return fit, select, outer


def patient_frame(samples, probabilities):
    probabilities = np.asarray(probabilities)
    if len(samples) != len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError("Predicciones ausentes, no finitas o desalineadas")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilidades fuera de [0,1]")
    frame = samples[["patient_id", "pCR", "fold", "dataset"]].copy()
    frame["probability"] = probabilities
    return frame.groupby("patient_id", sort=True).agg(
        label=("pCR", "first"), fold=("fold", "first"), cohort=("dataset", "first"),
        probability=("probability", "median")).reset_index()


def evaluate(model, loader, samples, criterion, device):
    """Misma BCE del brazo, FP32, eval y sin aumentos en fit y selección."""
    probabilities, bce = base.predict(model, loader, device, False, criterion)
    patients = patient_frame(samples, probabilities)
    metrics = base.binary_metrics(patients.label.to_numpy(), patients.probability.to_numpy())
    return {"slice_bce": bce, "patient_auc": metrics["roc_auc"],
            "patient_logloss": metrics["log_loss"], "patient_brier": metrics["brier"],
            "patient_ap": metrics["average_precision"]}, probabilities


def new_tracker(criterion, value):
    if not np.isfinite(value):
        raise ValueError("Métrica de selección no finita")
    return {"criterion": criterion, "best": float(value), "reference": float(value),
            "best_epoch": 0, "bad_epochs": 0, "stopped": False, "stop_epoch": None}


def update_tracker(tracker, value, epoch, min_epochs, patience, min_delta):
    """La mejora mínima se acumula desde el último progreso significativo."""
    if tracker["stopped"]:
        return False
    if not np.isfinite(value):
        raise ValueError("Métrica de selección no finita")
    direction = 1 if tracker["criterion"] == "auc" else -1
    improved = direction * (value - tracker["best"]) > 0
    if improved:
        tracker.update(best=float(value), best_epoch=int(epoch))
    if direction * (value - tracker["reference"]) > min_delta:
        tracker.update(reference=float(value), bad_epochs=0)
    else:
        tracker["bad_epochs"] += 1
    if epoch >= min_epochs and tracker["bad_epochs"] >= patience:
        tracker.update(stopped=True, stop_epoch=int(epoch))
    return improved


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_torch(path, payload):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def plot_history(folder, history, trackers):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(folder).parents[2] / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, metric, title in zip(axes, ("slice_bce", "patient_logloss", "patient_auc"),
                                   ("BCE del brazo por corte", "Log-loss por paciente", "AUC por paciente")):
        for role, color, label in (("fit", "#267b89", "Fit en evaluación"),
                                   ("selection", "#c65a3e", "Selección interna")):
            axis.plot(frame.epoch, frame[role + "_" + metric], color=color, label=label)
        for criterion, style in (("auc", "--"),):
            axis.axvline(trackers[criterion]["best_epoch"], color="#777777", linestyle=style,
                         label="Elegido por " + criterion)
        axis.set(title=title, xlabel="Época")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[2].set_ylim(0, 1)
    fig.suptitle("Sin aumentos ni dropout · FP32 · época 0 incluida · exterior aún separado")
    fig.savefig(Path(folder) / "curvas.png", dpi=140)
    plt.close(fig)


def run_one(samples, root, output, arm, seed, fold, config, device, cache=None,
            progress=None, until_epoch=None):
    """Entrenamiento reanudable; solo selección interna AUC, incluida época cero."""
    if arm not in ARMS:
        raise ValueError("Brazo desconocido")
    config = deepcopy(config)
    config["model"]["film"] = ARMS[arm]["film"]
    tc = config["training"]
    torch.set_num_threads(int(tc["cpu_threads"]))
    base.seed_everything(seed, tc["deterministic"])
    fit, selection, outer = split_patients(samples, fold, config["inner_fraction"], config["split_seed"])
    if config["smoke"]:
        fit, selection, outer = [f.groupby("pCR", group_keys=False).head(4).reset_index(drop=True)
                                 for f in (fit, selection, outer)]
    clinical = config["model"]["clinical"]
    clinical["preprocessing"] = base.fit_clinical_preprocessing(fit)
    clinical["initialization"] = base.fit_clinical_initializer(fit, clinical["preprocessing"])
    config.update(arm=arm, seed=seed, fold=fold, loss=ARMS[arm]["loss"], lr=tc["lr"], data_signature=signature(samples),
                  partition_signatures={name: signature(frame) for name, frame in
                                        (("fit", fit), ("selection", selection), ("outer", outer))},
                  sources={Path(p).name: base.sha256(Path(p)) for p in
                           (__file__, base.__file__, base.pdatos.__file__)},
                  device=str(device))
    folder = Path(output) / arm / f"seed_{seed}" / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    config_hash = base.config_hash(config)
    if (folder / "config.json").exists():
        if base.config_hash(json.loads((folder / "config.json").read_text())) != config_hash:
            raise ValueError(f"Cambió código, datos o configuración: {folder}")
    base.json_write(folder / "config.json", config)
    if (folder / "summary.json").exists():
        summary = json.loads((folder / "summary.json").read_text())
        if summary["config_hash"] != config_hash:
            raise ValueError("Resumen de otra ejecución")
        if summary["status"] == "complete":
            for name, digest in summary["files"].items():
                if base.sha256(folder / name) != digest:
                    raise ValueError(f"Artefacto modificado: {folder / name}")
            return summary
    partitions = pd.concat([f[["patient_id", "pCR", "dataset"]].drop_duplicates().assign(role=role)
                            for role, f in (("fit", fit), ("selection", selection), ("outer", outer))])
    partitions.to_csv(folder / "particiones.csv", index=False)
    base.json_write(folder / "environment.json", base.environment(device))
    generator = torch.Generator().manual_seed(seed + fold * 1000)
    augmentation_generator = torch.Generator().manual_seed(seed + fold * 1000 + 1)
    workers = 0 if config["smoke"] else tc["num_workers"]
    batch_size = min(8, tc["batch_size"]) if config["smoke"] else tc["batch_size"]
    train_loader = base.make_loader(fit, root, batch_size, workers, generator, True, cache, True)
    fit_loader = base.make_loader(fit, root, batch_size, workers,
                                 torch.Generator().manual_seed(seed + 60000 + fold), cache=cache, clinical=True)
    selection_loader = base.make_loader(selection, root, batch_size, workers,
                                       torch.Generator().manual_seed(seed + 70000 + fold), cache=cache, clinical=True)
    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"], eta_min=tc["min_lr"])
    amp = bool(tc["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.)
    pos_weight = base.pdatos.pos_weight_cortes(fit) if config["loss"] == "weighted" else 1.0
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint = folder / "last.pt"
    trackers, best_states, history, start_epoch = {}, {}, [], 0

    def measure(epoch, train_loss=None, lr=None):
        fit_metrics, _ = evaluate(model, fit_loader, fit, criterion, device)
        select_metrics, _ = evaluate(model, selection_loader, selection, criterion, device)
        row = {"epoch": epoch, "lr": lr if lr is not None else config["lr"],
               "train_augmented_loss": train_loss}
        row.update({"fit_" + k: v for k, v in fit_metrics.items()})
        row.update({"selection_" + k: v for k, v in select_metrics.items()})
        return row

    def persist(epoch):
        # Un solo archivo atómico conserva trayectoria y mejor modelo.
        save_torch(checkpoint, {"config_hash": config_hash, "state_dict": cpu_state(model),
                               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                               "scaler": scaler.state_dict(), "epoch": epoch, "history": history,
                               "trackers": deepcopy(trackers), "best_states": best_states,
                               "rng": base.capture_rng(generator, augmentation_generator)})
        base.history_write(folder / "history.csv", history)
        plot_history(folder, history, trackers)
        if progress:
            progress({"arm": arm, "seed": seed, "fold": fold, "epoch": epoch,
                      "trackers": deepcopy(trackers), "metrics": history[-1]})

    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["config_hash"] != config_hash:
            raise ValueError("Checkpoint de otro código, dato o configuración")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        trackers, best_states, history = saved["trackers"], saved["best_states"], saved["history"]
        start_epoch = saved["epoch"]
        base.restore_rng(saved["rng"], generator, augmentation_generator)
    else:
        history.append(measure(0))
        for name, metric in (("auc", "patient_auc"),):
            trackers[name] = new_tracker(name, history[0]["selection_" + metric])
            best_states[name] = cpu_state(model)
        persist(0)
    limit = tc["epochs"] if until_epoch is None else min(tc["epochs"], until_epoch)
    for epoch in range(start_epoch + 1, limit + 1):
        if all(t["stopped"] for t in trackers.values()):
            break
        started = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        loss = base.train_epoch(model, train_loader, optimizer, scaler, criterion, device, amp,
                                config["augmentation"], augmentation_generator, tc["clip_grad_norm"],
                                max_batches=1 if config["smoke"] else None)
        scheduler.step()
        row = measure(epoch, loss, lr)
        row["seconds"] = time.perf_counter() - started
        row["peak_vram_gib"] = torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else 0.0
        history.append(row)
        for name, metric in (("auc", "patient_auc"),):
            delta = tc["min_delta"]
            if update_tracker(trackers[name], row["selection_" + metric], epoch,
                              tc["min_epochs"], tc["patience"], delta):
                best_states[name] = cpu_state(model)
        persist(epoch)
        print(f"{arm} seed={seed} fold={fold} epoch={epoch}/{tc['epochs']} "
              f"fitBCE={row['fit_slice_bce']:.4f} selectBCE={row['selection_slice_bce']:.4f} "
              f"selectNLL={row['selection_patient_logloss']:.4f} selectAUC={row['selection_patient_auc']:.4f} "
              f"t={row['seconds']:.1f}s", flush=True)
    complete = history[-1]["epoch"] >= tc["epochs"] or all(t["stopped"] for t in trackers.values())
    if not complete:
        return {"status": "paused", "epoch": history[-1]["epoch"]}
    # Solo ahora se construye el loader exterior: nunca decide épocas o parada.
    outer_loader = base.make_loader(outer, root, batch_size, workers,
                                   torch.Generator().manual_seed(seed + 80000 + fold), cache=cache, clinical=True)
    results = {}
    for name in CRITERIA:
        model.load_state_dict(best_states[name], strict=True)
        metrics, probabilities = evaluate(model, outer_loader, outer, criterion, device)
        export = {"schema_version": 1, "model_config": deepcopy(config["model"]),
                  "state_dict": best_states[name]}
        save_torch(folder / f"best_{name}.pt", export)
        frame = outer[["sample_id", "patient_id", "pCR", "fold", "dataset"]].copy()
        frame["probability"] = probabilities
        frame.to_csv(folder / f"oof_{name}.csv", index=False)
        _, selected_probabilities = evaluate(model, selection_loader, selection, criterion, device)
        selected_frame = selection[["sample_id", "patient_id", "pCR", "fold", "dataset"]].copy()
        selected_frame["probability"] = selected_probabilities
        selected_frame.to_csv(folder / f"selection_{name}.csv", index=False)
        results[name] = {"selection": trackers[name], "outer_metrics": metrics}
    files = ["config.json", "history.csv", "particiones.csv"] + [f"{prefix}_{name}.{ext}" for name in CRITERIA
                                                    for prefix, ext in (("best", "pt"), ("oof", "csv"), ("selection", "csv"))]
    summary = {"status": "complete", "eligible_for_comparison": not config["smoke"],
               "config_hash": config_hash, "arm": arm, "seed": seed, "fold": fold,
               "epochs": history[-1]["epoch"], "results": results, "parameters": base.numero_parametros(model),
               "training_seconds": sum(row.get("seconds", 0) for row in history),
               "peak_vram_gib": max(row.get("peak_vram_gib", 0) for row in history),
               "test_images_loaded": 0, "files": {name: base.sha256(folder / name) for name in files}}
    base.json_write(folder / "summary.json", summary)
    return summary


class CNNFiLM(base.CNN):
    """FiLM sobre las 64 características visuales, conservando la vía clínica."""

    def __init__(self, config):
        super().__init__(config)
        hidden = int(self.config["hidden"])
        # La capa nueva no consume el RNG del control: mismos pesos comunes y dropout.
        with torch.random.fork_rng(devices=[]):
            self.film = torch.nn.Linear(2 * len(base.CLINICAL_MODEL_FEATURES), 2 * hidden)
        torch.nn.init.zeros_(self.film.weight)
        torch.nn.init.zeros_(self.film.bias)

    def modulate(self, visual, clinical_design):
        gamma, beta = self.film(clinical_design).chunk(2, dim=1)
        return (1 + gamma) * visual + beta

    def forward(self, x, clinical=None):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError(f"Esperado [N,3,256,256], recibido {tuple(x.shape)}")
        features = self.features(self.preprocess(x))
        pooled = torch.cat((self.avgpool(features).flatten(1), self.maxpool(features).flatten(1)), dim=1)
        visual = self.image_projection(pooled)
        design = self.preprocess_clinical(clinical)
        return self.classifier(torch.cat((self.modulate(visual, design), design), dim=1))


def build_model(config):
    return CNNFiLM(config) if config.get("film", False) else base.CNN(config)


def ensemble_slices(frames):
    left, right = [f.sort_values("sample_id").reset_index(drop=True) for f in frames]
    keys = ["sample_id", "patient_id", "pCR", "fold", "dataset"]
    if not left[keys].equals(right[keys]) or left.sample_id.duplicated().any():
        raise ValueError("Semillas desalineadas o muestras duplicadas")
    left = left.copy()
    left["probability"] = (left.probability.to_numpy() + right.probability.to_numpy()) / 2
    return left


def metrics_for(frame, column="probability"):
    result = base.binary_metrics(frame.label, frame[column])
    result["mean_fold_auc"] = float(np.mean([
        base.roc_auc_score(part.label, part[column]) for _, part in frame.groupby("fold")]))
    return result


def paired_intervals(reference, candidate, repetitions=2000):
    """IC95% pareados para la única hipótesis principal: FiLM con BCE ponderada."""
    keys = ["patient_id", "label", "fold", "cohort"]
    if not reference[keys].equals(candidate[keys]):
        raise ValueError("Pacientes no alineadas para comparación pareada")
    rng = np.random.default_rng(20261011)
    y = reference.label.to_numpy()
    strata = list(reference.groupby(["fold", "cohort", "label"]).indices.values())
    folds = reference.fold.to_numpy()
    values = {key: [] for key in ("log_loss_calibrated", "brier_calibrated", "roc_auc", "mean_fold_auc")}
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
        yy, ff = y[indices], folds[indices]
        for key, fn, column in (
            ("log_loss_calibrated", lambda p: base.log_loss(yy, np.clip(p, base.EPS, 1-base.EPS), labels=[0, 1]), "calibrated"),
            ("brier_calibrated", lambda p: np.mean((yy-p)**2), "calibrated"),
            ("roc_auc", lambda p: base.roc_auc_score(yy, p), "probability"),
            ("mean_fold_auc", lambda p: np.mean([base.roc_auc_score(yy[ff == f], p[ff == f]) for f in sorted(set(ff))]), "probability"),
        ):
            values[key].append(float(fn(candidate[column].to_numpy()[indices]) - fn(reference[column].to_numpy()[indices])))
    return {key: {"lower": float(np.quantile(val, .025)), "upper": float(np.quantile(val, .975))} for key, val in values.items()}


def compare(samples, output, repetitions=2000):
    output = Path(output)
    candidates, summaries, common = {}, [], set()
    expected_signature = signature(samples)
    partitions_by_fold = {}
    seed_metrics = {}
    calibrations = {}
    for arm, arm_spec in ARMS.items():
        frames = []
        seed_outer = {seed: [] for seed in (42, 2026)}
        for fold in range(5):
            fit, select, outer = split_patients(samples, fold)
            outer_frames, select_frames = [], []
            for seed in (42, 2026):
                folder = output / arm / f"seed_{seed}" / f"fold_{fold}"
                summary = json.loads((folder / "summary.json").read_text())
                config = json.loads((folder / "config.json").read_text())
                if summary["status"] != "complete" or not summary["eligible_for_comparison"]:
                    raise ValueError("Se requieren los 40 runs completos, excluidas pruebas")
                if (config["arm"], config["seed"], config["fold"], config["loss"], config["model"]["film"]) != (arm, seed, fold, arm_spec["loss"], arm_spec["film"]):
                    raise ValueError("Run en carpeta incorrecta")
                if summary["config_hash"] != base.config_hash(config) or config["data_signature"] != expected_signature:
                    raise ValueError("Configuración o datos modificados")
                required = {"config.json", "history.csv", "particiones.csv", "best_auc.pt", "oof_auc.csv", "selection_auc.csv"}
                if not required.issubset(summary["files"]):
                    raise ValueError("Faltan artefactos verificables")
                for name, checksum in summary["files"].items():
                    if base.sha256(folder / name) != checksum:
                        raise ValueError(f"Artefacto modificado: {folder / name}")
                for role, part in (("fit", fit), ("selection", select), ("outer", outer)):
                    if config["partition_signatures"][role] != signature(part):
                        raise ValueError("Partición no coincide con el protocolo")
                partition_key = base.config_hash(config["partition_signatures"])
                if partitions_by_fold.setdefault(fold, partition_key) != partition_key:
                    raise ValueError("Los brazos no comparten pacientes")
                comparable = deepcopy(config)
                for key in ("arm", "seed", "fold", "loss", "partition_signatures"):
                    comparable.pop(key)
                comparable["model"].pop("film")
                for key in ("preprocessing", "initialization"):
                    comparable["model"]["clinical"].pop(key)
                common.add(base.config_hash(comparable))
                loaded = []
                for name, part in (("oof_auc.csv", outer), ("selection_auc.csv", select)):
                    f = pd.read_csv(folder / name).assign(split="train")
                    roster = part.rename(columns={"pCR": "label", "dataset": "cohort"})
                    base.validate_oof(f, roster)
                    loaded.append(f)
                outer_frames.append(loaded[0])
                select_frames.append(loaded[1])
                seed_outer[seed].append(patient_frame(outer, loaded[0].set_index("sample_id").loc[outer.sample_id].probability.to_numpy()))
                summaries.append(summary)
            select_ensemble = ensemble_slices(select_frames)
            outer_ensemble = ensemble_slices(outer_frames)
            selected_patients = patient_frame(select_ensemble, select_ensemble.probability)
            outer_patients = patient_frame(outer_ensemble, outer_ensemble.probability)
            cal = base.fit_platt(selected_patients.label.to_numpy(), selected_patients.probability.to_numpy())
            cal["fit_data"] = "inner_selection_patients_only"
            sp = base.apply_calibration(selected_patients.probability.to_numpy(), cal)
            threshold = base.youden_threshold(selected_patients.label.to_numpy(), sp)
            outer_patients["calibrated"] = base.apply_calibration(outer_patients.probability.to_numpy(), cal)
            outer_patients["threshold"] = threshold
            outer_patients["prediction"] = (outer_patients.calibrated >= threshold).astype(int)
            calibrations[f"{arm}_fold_{fold}"] = {"calibration": cal, "threshold": threshold, "n_selection": len(selected_patients)}
            frames.append(outer_patients)
        candidates[arm] = pd.concat(frames).sort_values("patient_id").reset_index(drop=True)
        seed_metrics[arm] = {str(seed): metrics_for(pd.concat(parts)) for seed, parts in seed_outer.items()}
    if len(common) != 1:
        raise ValueError("Código, datos o hiperparámetros distintos entre brazos")
    dest = output / "comparacion"
    dest.mkdir(exist_ok=True)
    rows = {}
    for name, frame in candidates.items():
        frame.to_csv(dest / f"oof_{name}.csv", index=False)
        threshold_metrics = base.binary_metrics(frame.label, frame.prediction.to_numpy(dtype=float))
        rows[name] = {"raw": metrics_for(frame), "calibrated": metrics_for(frame, "calibrated"),
                      "inner_youden": {k: threshold_metrics[k] for k in ("tn", "fp", "fn", "tp", "accuracy", "f1", "precision", "sensitivity", "specificity")},
                      "cohorts": {c: metrics_for(p) for c, p in frame.groupby("cohort")},
                      "folds": {str(f): metrics_for(p) for f, p in frame.groupby("fold")}}
    primary, reference = candidates["film_weighted"], candidates["control_weighted"]
    intervals = paired_intervals(reference, primary, repetitions)
    auc_delta = rows["film_weighted"]["raw"]["roc_auc"] - rows["control_weighted"]["raw"]["roc_auc"]
    # Referencia histórica explícita; no se utiliza para seleccionar épocas ni calibración.
    historical_path = PROJECT_ROOT / "resultados/06_informe_multimodal_20261007/comparacion_pareada.csv"
    hist = pd.read_csv(historical_path).sort_values("patient_id").reset_index(drop=True)
    if not hist[["patient_id", "label", "fold", "cohort"]].equals(primary[["patient_id", "label", "fold", "cohort"]]):
        raise ValueError("Histórico v002 desalineado")
    manifest = json.loads((PROJECT_ROOT / "modelos/versiones/v002_multimodal_clinico_20261007/manifiesto.json").read_text())
    hp = base.apply_calibration(hist.selected.to_numpy(), manifest["calibration"])
    historical = {"raw": base.binary_metrics(hist.label, hist.selected),
                  "saved_calibration_threshold": base.binary_metrics(hist.label, hp, manifest["threshold"]),
                  "warning": "OOF con selección histórica distinta; calibración y umbral aparentes. No es control equiparado."}
    promising = (auc_delta >= .005 and intervals["roc_auc"]["lower"] > 0
        and rows["film_weighted"]["calibrated"]["log_loss"] <= rows["control_weighted"]["calibrated"]["log_loss"]
        and rows["film_weighted"]["calibrated"]["brier"] <= rows["control_weighted"]["calibrated"]["brier"]
        and all(seed_metrics["film_weighted"][str(s)]["roc_auc"] >= seed_metrics["control_weighted"][str(s)]["roc_auc"] for s in (42, 2026)))
    report = {"metrics": rows, "seed_metrics": seed_metrics, "historical_v002": historical,
              "primary_comparison": {"candidate": "film_weighted", "reference": "control_weighted", "auc_delta": auc_delta,
                                     "intervals_95": intervals, "promising_internal_result": promising},
              "normal_bce_comparison": "Secondary; no automatic selection between losses",
              "calibrations": calibrations, "bootstrap_repetitions": repetitions, "n_runs": len(summaries),
              "automatic_promotion": False, "test_images_loaded": 0,
              "warning": "Evaluación interna adaptativa. El control reentrenado y los pesos originales v002 son referencias distintas. Los IC no corrigen toda la búsqueda histórica."}
    base.json_write(dest / "resumen.json", report)
    lines = ["# Ensayo FiLM", "", report["warning"], "", "| Modelo | AUC cruda | AP | Log-loss calibrada | Brier calibrado | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in rows.items():
        lines.append(f"| {name} | {m['raw']['roc_auc']:.6f} | {m['raw']['average_precision']:.6f} | {m['calibrated']['log_loss']:.6f} | {m['calibrated']['brier']:.6f} | {m['inner_youden']['fp']} | {m['inner_youden']['fn']} |")
    lines += ["", f"V002 histórico, AUC {historical['raw']['roc_auc']:.6f}; log-loss con su calibración guardada {historical['saved_calibration_threshold']['log_loss']:.6f}.",
              "Es una referencia descriptiva con otro protocolo, separada del control equiparado.",
              "", f"Señal interna favorable según criterio prefijado: {promising}.",
              "V002 sigue activo. Ninguna promoción automática ni evaluación de test."]
    (dest / "INFORME_RESULTADOS.md").write_text("\n".join(lines) + "\n")
    return report


def protected_snapshot():
    files = sorted((PROJECT_ROOT / "modelos/versiones").rglob("*"))
    files += [PROJECT_ROOT / "app.py", Path(base.__file__), Path(base.pdatos.__file__),
              PROJECT_ROOT / "src/cancer_mama/paths.py", PROJECT_ROOT / "src/cancer_mama/app_logic.py"]
    return {str(path.relative_to(PROJECT_ROOT)): base.sha256(path) for path in files
            if path.is_file() and "__pycache__" not in path.parts}


def input_image_signature(samples, root):
    # Solo imágenes de train; incluye bytes para rechazar una reanudación con PNG cambiados.
    base.DatasetEntrenamiento(samples, root)
    digest = hashlib.sha256()
    for row in samples.sort_values("sample_id").itertuples(index=False):
        for column in base.pdatos.COLUMNAS_RUTA:
            relative = getattr(row, column)
            digest.update(relative.encode())
            digest.update(base.sha256(Path(root) / relative).encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", choices=("ejecutar", "comparar"), nargs="?", default="ejecutar")
    parser.add_argument("--datos", type=Path, default=base.pdatos.DATOS)
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dispositivo", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prueba", action="store_true", help="Cuatro brazos, un fold y una época; excluidos de comparación")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("workers debe ser >= 0")
    if args.dispositivo == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA no disponible")
    output = args.salida / "pruebas" if args.prueba else args.salida
    output.mkdir(parents=True, exist_ok=True)
    # Este lanzador se ejecuta en el servidor Linux. El lock se libera también tras un crash.
    import fcntl
    with (output / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Ya existe una ejecución activa en esa salida")
        state = {"status": "running", "stage": "preparing", "pid": os.getpid(),
                 "started_at": now(), "completed_runs": 0, "total_runs": 4 if args.prueba else 40}

        def progress(details):
            state.update(details, updated_at=now())
            base.json_write(output / "estado.json", state)

        progress({})
        try:
            samples = base.cargar_train(args.datos, include_clinical=True)
            config = settings(args.prueba, args.workers)
            device = torch.device(args.dispositivo)
            protected = protected_snapshot()
            environment = base.environment(device)
            environment.pop("git_commit", None)  # El código se fija por hash, el commit se registra aparte.
            parameters = {"settings": config, "arms": ARMS, "selectors": list(CRITERIA),
                          "data_signature": signature(samples),
                          "train_png_sha256": input_image_signature(samples, args.datos),
                          "protected": protected, "sources": {
                              "experiment": base.sha256(Path(__file__)),
                              "protocol": base.sha256(PROJECT_ROOT / "docs/mejoras_cnn/PROTOCOLO_FILM.md"),
                              "training": base.sha256(Path(base.__file__)),
                              "data": base.sha256(Path(base.pdatos.__file__))},
                          "environment": environment}
            fingerprint = base.config_hash(parameters)
            protocol_path = output / "protocolo.json"
            if protocol_path.exists():
                if json.loads(protocol_path.read_text())["fingerprint"] != fingerprint:
                    raise ValueError("Salida de otro protocolo, entorno, datos o código; usa una salida nueva")
            else:
                if args.accion == "comparar":
                    raise ValueError("Falta el protocolo de entrenamiento")
                base.json_write(protocol_path, {"fingerprint": fingerprint, "parameters": parameters,
                    "created_at": now(), "git_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()})
            if args.accion == "ejecutar":
                progress({"stage": "cache"})
                cache = None if args.prueba else base.CacheTrain(samples, args.datos, output / ".cache", workers=8)
                for fold in ([0] if args.prueba else range(5)):
                    for seed in ([42] if args.prueba else (42, 2026)):
                        for arm in ARMS:
                            progress({"stage": "training", "arm": arm, "seed": seed, "fold": fold, "epoch": 0})
                            print(f"Inicio {arm} seed={seed} fold={fold}", flush=True)
                            run_one(samples, args.datos, output, arm, seed, fold, config, device, cache, progress)
                            if protected_snapshot() != protected:
                                raise RuntimeError("Cambió un archivo protegido durante el ensayo")
                            progress({"completed_runs": state["completed_runs"] + 1})
            if not args.prueba:
                progress({"stage": "comparison"})
                report = compare(samples, output)
                print(json.dumps(report["metrics"], indent=2), flush=True)
                progress({"completed_runs": 40})
            if protected_snapshot() != protected:
                raise RuntimeError("Los archivos protegidos han cambiado")
            progress({"status": "complete", "stage": "complete", "finished_at": now(), "protected_unchanged": True})
        except BaseException as exc:
            progress({"status": "failed", "error": str(exc), "finished_at": now()})
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
