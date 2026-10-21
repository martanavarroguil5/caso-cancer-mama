#!/usr/bin/env python3
"""Piloto controlado de entrenamiento anti-sobreajuste, sin abrir el test.

Compara una CNN temporal simple con la misma red entrenada mediante lotes por
paciente, peso igual por paciente, Dropout2d y aumentos conjuntos suaves. El
piloto usa folds exteriores 0 y 3, dos semillas y BCE normal/ponderada.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn

from . import datos as pdatos
from . import entrenamiento as base
from . import entrenamiento_imagen as imagen
from .paths import DATA_DIR, RESULTS_DIR


EXPERIMENT_NAME = "cnn_imagen_antioverfit_piloto"
DEFAULT_OUTPUT = RESULTS_DIR/"04_entrenamiento/piloto_antioverfit_20261021"
ARMS = ("control", "antioverfit")
PILOT_FOLDS = (0, 3)
TARGET_AUC = .60


def configuracion(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Brazo desconocido: {arm}")
    config = deepcopy(imagen.configuracion())
    config["name"] = f"{EXPERIMENT_NAME}_{arm}"
    config["model"] = {
        "architecture": "single_scale_early_temporal_fusion",
        "input_phases": list(pdatos.FASES),
        "backbone_channels": [24, 48, 96, 160],
        "hidden": 64,
        "dropout": .35,
        "spatial_dropout": .10 if arm == "antioverfit" else 0.0,
        "normalization": "batch",
        "clinical": None,
    }
    config["training"].update(
        epochs=15, min_epochs=4, patience=4, batch_size=64,
        lr=8e-4, weight_decay=1e-3,
        patients_per_batch=6 if arm == "antioverfit" else None,
        equal_patient_weight=arm == "antioverfit",
    )
    if arm == "antioverfit":
        config["augmentation"] = {
            "hflip": .5, "rot90": True, "translate": .03,
            "scale_min": .95, "scale_max": 1.05,
            "gain_min": .90, "gain_max": 1.10,
            "gamma_min": .90, "gamma_max": 1.10,
            "noise_std": .01,
        }
    config["evaluation"].update(
        outer_protocol="pilot_folds_0_and_3_with_inner_checkpoint_selection",
        checkpoint_metric="inner_patient_auc",
        test_policy="never_load_during_pilot",
    )
    config["history_plot_title"] = f"Piloto anti-sobreajuste: {arm}"
    return config


class CNNTemporalSimple(nn.Module):
    """CNN histórica simplificada: tres tiempos, una escala y fusión temprana."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = deepcopy(config or configuracion("control")["model"])
        self.config = self.config.get("model", self.config)
        if self.config.get("architecture") != "single_scale_early_temporal_fusion":
            raise ValueError("Arquitectura anti-sobreajuste desconocida")
        if self.config.get("input_phases") != list(pdatos.FASES):
            raise ValueError("Se requieren PRE, EARLY y LATE en ese orden")
        if self.config.get("clinical") is not None:
            raise ValueError("El piloto principal no admite variables clínicas")
        if self.config.get("normalization") != "batch":
            raise ValueError("El protocolo utiliza BatchNorm")
        widths = self.config.get("backbone_channels")
        if (not isinstance(widths, list) or len(widths) != 4 or
                any(not isinstance(value, int) or value < 1 for value in widths)):
            raise ValueError("Se requieren cuatro anchos positivos")
        hidden = int(self.config.get("hidden", 64))
        dropout = float(self.config.get("dropout", .35))
        spatial_dropout = float(self.config.get("spatial_dropout", 0))
        if hidden < 1 or not 0 <= dropout < 1 or not 0 <= spatial_dropout < 1:
            raise ValueError("Anchuras o dropout inválidos")
        self.backbone = nn.Sequential(*[
            base.Bloque(input_channels, output_channels, index == 0)
            for index, (input_channels, output_channels) in enumerate(
                zip([3, *widths[:-1]], widths))
        ])
        self.spatial_dropout = nn.Dropout2d(spatial_dropout)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = base.GlobalMaxPool()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(widths[-1]*2, hidden),
            nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        self.apply(imagen.CNNImagenFases._initialize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError(f"Esperado [N,3,256,256], recibido {tuple(x.shape)}")
        z = self.spatial_dropout(self.backbone(x))
        z = torch.cat((self.avgpool(z).flatten(1),
                       self.maxpool(z).flatten(1)), dim=1)
        return self.classifier(z)


def run_path(output: Path, arm: str, loss: str, seed: int, fold: int) -> Path:
    return (Path(output)/f"{EXPERIMENT_NAME}_{arm}"/loss/
            f"seed_{seed}"/f"fold_{fold}")


def verificar() -> dict:
    control = configuracion("control")
    candidate = configuracion("antioverfit")
    return {
        "status": "piloto_preparado_sin_entrenar",
        "arms": list(ARMS), "losses": list(imagen.LOSSES),
        "seeds": list(imagen.SEEDS), "folds": list(PILOT_FOLDS),
        "planned_runs": len(ARMS)*len(imagen.LOSSES)*len(imagen.SEEDS)*len(PILOT_FOLDS),
        "control_parameters": imagen.numero_parametros(CNNTemporalSimple(control)),
        "candidate_parameters": imagen.numero_parametros(CNNTemporalSimple(candidate)),
        "candidate_patient_batches": candidate["training"]["patients_per_batch"],
        "candidate_equal_patient_weight": candidate["training"]["equal_patient_weight"],
        "target_auc": TARGET_AUC, "test_images_loaded": 0,
    }


def _ensemble(output: Path, arm: str, loss: str, fold: int) -> pd.DataFrame:
    frames = []
    for seed in imagen.SEEDS:
        path = run_path(output, arm, loss, seed, fold)
        summary = json.loads((path/"summary.json").read_text())
        if (summary.get("status") != "complete" or summary.get("smoke") or
                not summary.get("outer_evaluated") or
                base.sha256(path/"outer_patients.csv") != summary.get("outer_sha256")):
            raise ValueError(f"Ejecución incompleta o incompatible: {path}")
        frames.append(pd.read_csv(path/"outer_patients.csv"))
    return imagen._aligned_average(frames)


def _gaps(output: Path, arm: str, loss: str) -> dict:
    selected, final = [], []
    for seed in imagen.SEEDS:
        for fold in PILOT_FOLDS:
            path = run_path(output, arm, loss, seed, fold)
            summary = json.loads((path/"summary.json").read_text())
            history = pd.read_csv(path/"history.csv")
            # Tanto summary.best_epoch como history.epoch empiezan en uno.
            chosen = history.loc[history.epoch.eq(int(summary["best_epoch"]))].iloc[0]
            selected.append(float(chosen.auc_gap))
            final.append(float(history.iloc[-1].auc_gap))
    return {
        "selected_gap_median": float(np.median(selected)),
        "final_gap_median": float(np.median(final)),
        "selected_gaps": selected, "final_gaps": final,
    }


def _paired_bootstrap(candidate: pd.DataFrame, control: pd.DataFrame,
                      repetitions: int = 2000) -> list[float]:
    candidate = candidate.sort_values("patient_id").reset_index(drop=True)
    control = control.sort_values("patient_id").reset_index(drop=True)
    for column in ("patient_id", "label", "fold"):
        if not np.array_equal(candidate[column].to_numpy(), control[column].to_numpy()):
            raise ValueError(f"Brazos no alineados en {column}")
    y = candidate.label.to_numpy(dtype=int)
    folds = candidate.fold.to_numpy(dtype=int)
    pc = candidate.probability_raw.to_numpy(dtype=float)
    pr = control.probability_raw.to_numpy(dtype=float)
    strata = [np.flatnonzero((folds == fold) & (y == label))
              for fold in PILOT_FOLDS for label in (0, 1)]
    rng = np.random.default_rng(20261021)
    delta = np.empty(repetitions)
    for index in range(repetitions):
        sample = np.concatenate([rng.choice(group, len(group), replace=True)
                                 for group in strata])
        delta[index] = roc_auc_score(y[sample], pc[sample])-roc_auc_score(y[sample], pr[sample])
    return [float(value) for value in np.quantile(delta, [.025, .975])]


def comparar(output: Path) -> dict:
    output = Path(output)
    report = {"status": "pilot_development_only", "folds": list(PILOT_FOLDS),
              "patients_are_outer_to_each_model": True, "test_images_loaded": 0,
              "warning": "Dos folds de desarrollo: sirve para filtrar, no para demostrar mejora.",
              "losses": {}}
    for loss in imagen.LOSSES:
        frames, metrics = {}, {}
        for arm in ARMS:
            fold_frames = [_ensemble(output, arm, loss, fold) for fold in PILOT_FOLDS]
            frame = pd.concat(fold_frames, ignore_index=True)
            frame.to_csv(output/f"oof_piloto_{arm}_{loss}.csv", index=False)
            fold_auc = [float(roc_auc_score(part.label, part.probability_raw))
                        for part in fold_frames]
            frames[arm] = frame
            metrics[arm] = {
                "fold_auc": fold_auc, "mean_fold_auc": float(np.mean(fold_auc)),
                "pooled_auc": float(roc_auc_score(frame.label, frame.probability_raw)),
                "gaps": _gaps(output, arm, loss),
            }
        fold_delta = np.array(metrics["antioverfit"]["fold_auc"])-np.array(
            metrics["control"]["fold_auc"])
        comparison = {
            "delta_mean_fold_auc": float(fold_delta.mean()),
            "delta_fold_auc": fold_delta.tolist(),
            "delta_pooled_auc_ci95": _paired_bootstrap(
                frames["antioverfit"], frames["control"]),
        }
        comparison["promising_gate"] = bool(
            comparison["delta_mean_fold_auc"] >= .01 and
            float(fold_delta.min()) >= -.02 and
            metrics["antioverfit"]["gaps"]["selected_gap_median"] <=
            metrics["control"]["gaps"]["selected_gap_median"]
        )
        report["losses"][loss] = {**metrics, "comparison": comparison}
    report["any_arm_promising"] = any(
        value["comparison"]["promising_gate"] for value in report["losses"].values())
    base.json_write(output/"comparacion_piloto.json", report)
    lines = ["# Piloto anti-sobreajuste", "",
             "Evaluación exterior en folds 0 y 3; test reservado no cargado.", "",
             "| BCE | AUC control | AUC anti-overfit | Delta | Gate |",
             "|---|---:|---:|---:|---|"]
    for loss, value in report["losses"].items():
        lines.append(
            f"| {loss} | {value['control']['mean_fold_auc']:.4f} | "
            f"{value['antioverfit']['mean_fold_auc']:.4f} | "
            f"{value['comparison']['delta_mean_fold_auc']:+.4f} | "
            f"{value['comparison']['promising_gate']} |")
    lines += ["", "El gate exige +0,01 de AUC media, ningún fold peor en más de "
              "0,02 y una brecha train-selección no mayor. El piloto no autoriza "
              "abrir test ni promocionar un modelo."]
    (output/"INFORME_PILOTO.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", nargs="?", choices=("verificar", "entrenar", "comparar"),
                        default="verificar")
    parser.add_argument("--datos", type=Path, default=DATA_DIR)
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--brazos", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--perdidas", nargs="+", choices=imagen.LOSSES,
                        default=list(imagen.LOSSES))
    parser.add_argument("--semillas", nargs="+", type=int, choices=imagen.SEEDS,
                        default=list(imagen.SEEDS))
    parser.add_argument("--folds", nargs="+", type=int, choices=PILOT_FOLDS,
                        default=list(PILOT_FOLDS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dispositivo", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prueba", action="store_true")
    args = parser.parse_args()
    if args.accion == "verificar":
        print(json.dumps(verificar(), indent=2, ensure_ascii=False))
        return
    if args.accion == "comparar":
        print(json.dumps(comparar(args.salida), indent=2, ensure_ascii=False))
        return
    if args.dispositivo == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA no está disponible")
    device = torch.device(args.dispositivo)
    output = args.salida/"pruebas" if args.prueba else args.salida
    for arm in dict.fromkeys(args.brazos):
        config = configuracion(arm)
        config["training"]["num_workers"] = args.workers
        if args.prueba:
            config["training"].update(epochs=1, min_epochs=1, patience=1,
                                      num_workers=0)
        for loss in dict.fromkeys(args.perdidas):
            for seed in dict.fromkeys(args.semillas):
                for fold in dict.fromkeys(args.folds):
                    imagen.run_training(
                        config, loss, seed, fold, args.datos, output, device,
                        smoke=args.prueba, model_type=CNNTemporalSimple,
                        experiment_name=config["name"],
                        extra_source_files={"entrenamiento_antioverfit.py": Path(__file__)},
                    )


if __name__ == "__main__":
    main()
