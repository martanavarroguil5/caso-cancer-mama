"""Ajuste anidado de optimización; arquitectura cinética estrictamente fija.

Tres pares LR/WD y dos pérdidas, dos semillas, cinco folds: 60 entrenamientos.
La selección usa solamente pacientes internos. Los cinco ganadores se congelan
antes de leer imágenes exteriores. No se utiliza el test reservado.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
import torch

from . import entrenamiento as base
from . import entrenamiento_cinetico as cnn
from . import entrenamiento_imagen as imagen
from .paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR


OUTPUT = RESULTS_DIR / "04_entrenamiento/ajuste_cnn_cinetica_20261021"
TRIALS = {
    "referencia": {"lr": 8e-4, "weight_decay": 1e-3},
    "lr_menor": {"lr": 3e-4, "weight_decay": 1e-3},
    "lr_menor_wd_mayor": {"lr": 3e-4, "weight_decay": 1e-2},
}


def now():
    return datetime.now().astimezone().isoformat()


def protocol():
    architecture = cnn.configuracion()["model"]
    return {
        "architecture": architecture,
        "architecture_source_sha256": hashlib.sha256(
            inspect.getsource(cnn.CNNMultiescalaCinetica).encode()).hexdigest(),
        "parameters": imagen.numero_parametros(cnn.CNNMultiescalaCinetica()),
        "trials": TRIALS, "losses": list(imagen.LOSSES),
        "seeds": list(imagen.SEEDS), "folds": list(imagen.FOLDS),
        "training": cnn.configuracion()["training"],
        "planned_runs": 60,
        "selection": "inner AUC of mean probabilities of two seeds per outer fold",
        "tie_break": "reference first, then lower WD, then normal loss",
        "minimum_auc": .60, "objective_auc": .70,
        "outer_policy": "evaluate only after all five choices frozen",
        "test_images_loaded": 0,
    }


def training_config(trial):
    config = cnn.configuracion()
    config["name"] += "_" + trial
    config["training"].update(TRIALS[trial])
    if config["model"] != protocol()["architecture"]:
        raise ValueError("La arquitectura no puede cambiar durante el ajuste")
    return config


def run_path(output, trial, loss, seed, fold):
    name = cnn.EXPERIMENT_NAME + "_" + trial
    return Path(output) / name / loss / f"seed_{seed}" / f"fold_{fold}"


def check_run(path):
    summary = json.loads((path / "summary.json").read_text())
    config = json.loads((path / "config.json").read_text())
    if (summary["status"] != "complete" or summary.get("smoke") or
            summary.get("outer_evaluated") is not False or
            summary["config_hash"] != base.config_hash(config) or
            config["model"] != cnn.configuracion()["model"]):
        raise ValueError(f"Run incompatible: {path}")
    for name, key in (("inference.pt", "inference_sha256"),
                      ("selection_patients.csv", "selection_sha256")):
        if base.sha256(path/name) != summary[key]:
            raise ValueError(f"Artefacto modificado: {path/name}")
    artifact = torch.load(path/"inference.pt", map_location="cpu", weights_only=True)
    if any(not torch.isfinite(t).all() for t in artifact["state_dict"].values()):
        raise FloatingPointError(f"Pesos no finitos: {path}")
    checkpoint = torch.load(path/"last.pt", map_location="cpu", weights_only=False)
    steps = {int(state["step"].item()) for state in checkpoint["optimizer"]["state"].values()}
    expected = math.ceil(config["counts"]["fit_slices"] / config["training"]["batch_size"])
    expected *= summary["epochs_completed"]
    if len(steps) != 1 or max(steps) < .99*expected:
        raise RuntimeError(f"Demasiados pasos de AdamW omitidos: {path}, {steps}/{expected}")
    return {"finite_weights": True, "optimizer_steps": max(steps),
            "expected_steps": expected, "skipped_steps": expected-max(steps)}


def choose_inner(output, fold):
    records = []
    for trial in TRIALS:
        for loss in imagen.LOSSES:
            paths = [run_path(output, trial, loss, seed, fold) for seed in imagen.SEEDS]
            checks = [check_run(path) for path in paths]
            frames = [pd.read_csv(path/"selection_patients.csv") for path in paths]
            frame = imagen._aligned_average(frames)
            if fold in set(frame.fold):
                raise ValueError("El fold exterior se ha filtrado en la selección interna")
            records.append({
                "trial": trial, "loss": loss,
                "inner_auc": float(roc_auc_score(frame.label, frame.probability_raw)),
                "models": [{"path": str(path.relative_to(output)),
                            "sha256": base.sha256(path/"inference.pt"),
                            "seed": seed, "check": check}
                           for path, seed, check in zip(paths, imagen.SEEDS, checks)],
            })
    # max conserva el primer elemento en empates: referencia y BCE normal.
    winner = max(records, key=lambda record: record["inner_auc"])
    reference = max((r for r in records if r["trial"] == "referencia"),
                    key=lambda record: record["inner_auc"])
    return {"fold": fold, "frozen_at": now(), "selection_data": "inner_only",
            "candidates": records, "selected": winner, "reference": reference}


def evaluate_choice(root, output, choice, fold, device):
    """Primera lectura de imágenes exteriores, con elección ya inmutable."""
    if not (output/"seleccion_congelada.json").is_file():
        raise RuntimeError("Faltan las cinco elecciones congeladas")
    config = training_config(choice["trial"])
    all_train = base.cargar_train(root, include_clinical=False)
    _, _, outer = imagen.particion_anidada(all_train, fold, config)
    probabilities, inner_frames = [], []
    for entry in choice["models"]:
        path = output/entry["path"]
        if base.sha256(path/"inference.pt") != entry["sha256"]:
            raise ValueError("Pesos cambiados después de congelar la elección")
        artifact = torch.load(path/"inference.pt", map_location="cpu", weights_only=True)
        model = cnn.CNNMultiescalaCinetica(artifact["model_config"]).to(device)
        model.load_state_dict(artifact["state_dict"], strict=True)
        loader = imagen._loader(outer, root, 32, 4, entry["seed"]+70000+fold)
        probability, _ = imagen.predict_slices(model, loader, device)
        probabilities.append(probability)
        inner_frames.append(pd.read_csv(path/"selection_patients.csv"))
        del model
    result = imagen.patient_predictions(outer, np.mean(probabilities, axis=0))
    inner = imagen._aligned_average(inner_frames)
    calibration = base.fit_platt(inner.label.to_numpy(), inner.probability_raw.to_numpy())
    calibrated_inner = base.apply_calibration(inner.probability_raw.to_numpy(), calibration)
    threshold = imagen.threshold_min_specificity(inner.label, calibrated_inner, .90)
    result["probability_calibrated"] = base.apply_calibration(
        result.probability_raw.to_numpy(), calibration)
    result["prediction"] = (result.probability_calibrated >= threshold).astype(int)
    result["threshold"] = threshold
    result["trial"] = choice["trial"]
    result["loss"] = choice["loss"]
    return result, {"fold": fold, "trial": choice["trial"], "loss": choice["loss"],
                    "calibration": calibration, "threshold": threshold,
                    "outer_auc": float(roc_auc_score(result.label, result.probability_raw))}


def metrics(frame):
    raw = imagen.metricas_binarias(frame.label, frame.probability_raw, .5)
    calibrated = imagen.metricas_binarias(frame.label, frame.probability_calibrated, .5)
    tn, fp, fn, tp = confusion_matrix(frame.label, frame.prediction, labels=[0, 1]).ravel()
    folds = [float(roc_auc_score(group.label, group.probability_raw))
             for _, group in frame.groupby("fold")]
    return {"pooled_raw_auc": raw["roc_auc"], "mean_fold_auc": float(np.mean(folds)),
            "std_fold_auc": float(np.std(folds, ddof=1)), "fold_auc": folds,
            "average_precision": raw["average_precision"],
            "crossfit_calibrated_log_loss": calibrated["log_loss"],
            "crossfit_calibrated_brier": calibrated["brier"],
            "decision": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                         "sensitivity": float(tp/(tp+fn)), "specificity": float(tn/(tn+fp)),
                         "balanced_accuracy": float(balanced_accuracy_score(frame.label, frame.prediction)),
                         "macro_f1": float(f1_score(frame.label, frame.prediction, average="macro"))},
            "minimum_reached": raw["roc_auc"] >= .60,
            "objective_reached": raw["roc_auc"] >= .70}


def versions_inventory():
    directory = PROJECT_ROOT/"modelos/versiones"
    return {str(path.relative_to(directory)): base.sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def execute(root, output, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = protocol()
    plan["source_sha256"] = {name: base.sha256(Path(module.__file__)) for name, module in
                              (("ajuste_cinetico", __import__(__name__, fromlist=["_"])),
                               ("entrenamiento_cinetico", cnn), ("entrenamiento_imagen", imagen),
                               ("entrenamiento", base))}
    plan_path = output/"protocolo.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("El protocolo o código ha cambiado: usa otra salida")
    base.json_write(plan_path, plan)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible")
    if smoke:
        for loss in imagen.LOSSES:
            config = training_config("referencia")
            config["training"].update(epochs=1, min_epochs=1, batch_size=4, num_workers=0)
            imagen.run_training(config, loss, 42, 0, root, output/"pruebas", device,
                smoke=True, model_type=cnn.CNNMultiescalaCinetica,
                experiment_name=config["name"], evaluate_outer=False,
                extra_source_files={"entrenamiento_cinetico.py": Path(cnn.__file__)})
        return
    inventory_path = output/"versiones_antes.json"
    if not inventory_path.exists():
        base.json_write(inventory_path, versions_inventory())
    completed = 0
    state = {"status": "running", "started_at": now(), "pid": os.getpid(),
             "planned_runs": 60, "architecture_parameters": plan["parameters"],
             "minimum_auc": .60, "objective_auc": .70, "test_images_loaded": 0}
    for fold in imagen.FOLDS:
        for trial in TRIALS:
            config = training_config(trial)
            for loss in imagen.LOSSES:
                for seed in imagen.SEEDS:
                    state.update(updated_at=now(), completed_runs=completed,
                                 current={"fold": fold, "trial": trial, "loss": loss, "seed": seed})
                    base.json_write(output/"estado.json", state)
                    imagen.run_training(config, loss, seed, fold, root, output, device,
                        model_type=cnn.CNNMultiescalaCinetica, experiment_name=config["name"],
                        extra_source_files={"entrenamiento_cinetico.py": Path(cnn.__file__),
                                            "ajuste_cinetico.py": Path(__file__)},
                        evaluate_outer=False)
                    check = check_run(run_path(output, trial, loss, seed, fold))
                    print("VERIFICADO", fold, trial, loss, seed, check, flush=True)
                    completed += 1
        choice = choose_inner(output, fold)
        base.json_write(output/f"seleccion_fold_{fold}.json", choice)
        print("ELECCIÓN INTERNA", fold, choice["selected"]["trial"],
              choice["selected"]["loss"], choice["selected"]["inner_auc"], flush=True)
    frozen = {"status": "frozen_before_outer_evaluation", "frozen_at": now(),
              "protocol_sha256": base.sha256(plan_path),
              "folds": [json.loads((output/f"seleccion_fold_{fold}.json").read_text())
                        for fold in imagen.FOLDS]}
    frozen_path = output/"seleccion_congelada.json"
    if not frozen_path.exists():
        base.json_write(frozen_path, frozen)
    else:
        frozen = json.loads(frozen_path.read_text())
        if frozen["protocol_sha256"] != base.sha256(plan_path):
            raise ValueError("Elección congelada incompatible")
    state.update(status="evaluating_frozen_choices", completed_runs=completed, updated_at=now())
    base.json_write(output/"estado.json", state)
    results, details = {}, {}
    for arm in ("selected", "reference"):
        frames, fold_details = [], []
        for choice in frozen["folds"]:
            frame, detail = evaluate_choice(root, output, choice[arm], choice["fold"], device)
            frames.append(frame)
            fold_details.append(detail)
        frame = pd.concat(frames).sort_values("patient_id").reset_index(drop=True)
        expected = base.cargar_train(root, include_clinical=False).patient_id.unique()
        if frame.patient_id.duplicated().any() or set(frame.patient_id) != set(expected):
            raise ValueError("OOF incompleto o duplicado")
        frame.to_csv(output/f"oof_pacientes_{arm}.csv", index=False)
        results[arm], details[arm] = metrics(frame), fold_details
    unchanged = versions_inventory() == json.loads(inventory_path.read_text())
    if not unchanged:
        raise RuntimeError("Ha cambiado una versión protegida")
    report = {"status": "complete_development_only", "finished_at": now(),
              "warning": "OOF de desarrollo; el historial adaptativo requiere confirmación independiente",
              "architecture_parameters": plan["parameters"], "architecture_unchanged": True,
              "protocol_sha256": base.sha256(plan_path), "selection_sha256": base.sha256(frozen_path),
              "results": results, "fold_details": details, "previous_versions_unchanged": unchanged,
              "test_images_loaded": 0}
    base.json_write(output/"informe.json", report)
    state.update(status="complete", completed_runs=completed, updated_at=now(),
                 finished_at=now(), results=results, previous_versions_unchanged=unchanged)
    base.json_write(output/"estado.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", choices=("verificar", "entrenar", "prueba"))
    parser.add_argument("--datos", type=Path, default=DATA_DIR)
    parser.add_argument("--salida", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.accion == "verificar":
        print(json.dumps(protocol(), indent=2, ensure_ascii=False))
        return
    # flock se libera al morir el proceso, también en interrupciones del sistema.
    import fcntl
    args.salida.mkdir(parents=True, exist_ok=True)
    with (args.salida/"runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args.datos, args.salida, smoke=args.accion == "prueba")
        except Exception:
            base.json_write(args.salida/"error.json", {"at": now(), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
