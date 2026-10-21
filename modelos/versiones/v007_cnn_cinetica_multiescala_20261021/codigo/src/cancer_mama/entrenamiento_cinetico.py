#!/usr/bin/env python3
"""CNN cinética multiescala solo imagen y con test cerrado.

El modelo combina contexto, región central y núcleo tumoral mediante vistas de
256, 128 y 64 píxeles, todas remuestreadas a 128. En cada escala se apilan seis
mapas con significado explícito (PRE, EARLY, LATE y tres diferencias firmadas)
antes de cualquier convolución. Es una fusión temporal temprana, no RGB.
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
from torch.nn import functional as F

from . import datos as pdatos
from . import entrenamiento as base
from . import entrenamiento_imagen as imagen
from .paths import DATA_DIR, RESULTS_DIR


EXPERIMENT_NAME = "cnn_imagen_cinetica_multiescala"
DEFAULT_OUTPUT = RESULTS_DIR/"04_entrenamiento/cnn_imagen_cinetica_multiescala_20261021"
CONTROL_OOF = RESULTS_DIR/"14_informe_cnn_imagen_20261014/oof_pacientes_weighted.csv"
HISTORICAL_OOF = RESULTS_DIR/"04_entrenamiento/historico/oof_pacientes.csv"


def configuracion() -> dict:
    """Protocolo fijado antes de ejecutar el candidato en GPU."""
    config = deepcopy(imagen.configuracion())
    config["name"] = EXPERIMENT_NAME
    config["model"] = {
        "architecture": "multiscale_early_temporal_kinetic_fusion",
        "input_phases": list(pdatos.FASES),
        "derived_maps": ["EARLY-PRE", "LATE-PRE", "EARLY-LATE"],
        "temporal_channels": ["PRE", "EARLY", "LATE",
                              "EARLY-PRE", "LATE-PRE", "EARLY-LATE"],
        "center_crop_sizes": [256, 128, 64],
        "encoder_input_size": 128,
        "scale_encoder_shared": True,
        "backbone_channels": [24, 48, 96, 160],
        "hidden": 64,
        "dropout": 0.35,
        "spatial_dropout": 0.10,
        "normalization": "batch",
        "clinical": None,
    }
    config["hypothesis"] = (
        "La fusión temporal temprana conserva el inductive bias del modelo histórico "
        "que alcanzó 0.5945 OOF; las vistas centrales reducen el fondo dominante y "
        "los mapas firmados hacen explícita la cinética antes de las convoluciones."
    )
    config["history_plot_title"] = (
        "CNN cinética multiescala: fit frente a selección; outer no interviene"
    )
    return config


class CNNMultiescalaCinetica(nn.Module):
    """CNN 2D propia con fusión temprana temporal/cinética y tres escalas."""

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = deepcopy(config or configuracion()["model"])
        self.config = self.config.get("model", self.config)
        if self.config.get("architecture") != \
                "multiscale_early_temporal_kinetic_fusion":
            raise ValueError("Arquitectura cinética desconocida")
        if self.config.get("input_phases") != list(pdatos.FASES):
            raise ValueError("Las fases deben estar ordenadas como PRE, EARLY y LATE")
        if self.config.get("derived_maps") != ["EARLY-PRE", "LATE-PRE", "EARLY-LATE"]:
            raise ValueError("Los mapas cinéticos no coinciden con el protocolo")
        expected_channels = ["PRE", "EARLY", "LATE",
                             "EARLY-PRE", "LATE-PRE", "EARLY-LATE"]
        if self.config.get("temporal_channels") != expected_channels:
            raise ValueError("Los seis canales temporales no coinciden con el protocolo")
        if self.config.get("scale_encoder_shared") is not True:
            raise ValueError("Las tres escalas requieren el mismo backbone")
        if self.config.get("clinical") is not None:
            raise ValueError("La CNN cinética no admite variables clínicas")
        if self.config.get("normalization") != "batch":
            raise ValueError("El protocolo fijado utiliza BatchNorm")

        widths = self.config.get("backbone_channels")
        if (not isinstance(widths, list) or len(widths) != 4 or
                any(not isinstance(value, int) or value < 1 for value in widths)):
            raise ValueError("Se requieren cuatro anchos positivos para el backbone")
        hidden = int(self.config.get("hidden", 64))
        dropout = float(self.config.get("dropout", 0.35))
        spatial_dropout = float(self.config.get("spatial_dropout", 0.10))
        self.crop_sizes = tuple(self.config.get("center_crop_sizes", []))
        self.encoder_input_size = int(self.config.get("encoder_input_size", 128))
        if (self.crop_sizes != (256, 128, 64) or self.encoder_input_size != 128 or
                hidden < 1 or not 0 <= dropout < 1 or not 0 <= spatial_dropout < 1):
            raise ValueError("Anchuras o dropout inválidos")

        # Replica el patrón de bloques del modelo histórico cercano a 0.60 OOF.
        # Las seis entradas tienen semántica temporal explícita; no son color RGB.
        self.backbone = nn.Sequential(*[
            base.Bloque(entrada, salida, index == 0)
            for index, (entrada, salida) in enumerate(
                zip([6, *widths[:-1]], widths))
        ])
        self.spatial_dropout = nn.Dropout2d(spatial_dropout)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = base.GlobalMaxPool()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(widths[-1]*2*len(self.crop_sizes), hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.apply(imagen.CNNImagenFases._initialize)

    @staticmethod
    def kinetic_maps(x: torch.Tensor) -> torch.Tensor:
        """Calcula realce inicial, tardío y lavado conservando el signo."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Se esperan fases [N,3,H,W]")
        pre, early, late = x[:, 0], x[:, 1], x[:, 2]
        return torch.stack((early-pre, late-pre, early-late), dim=1)

    @classmethod
    def temporal_maps(cls, x: torch.Tensor) -> torch.Tensor:
        """Apila fases crudas y diferencias: [N,6,H,W], en orden prefijado."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Se esperan fases [N,3,H,W]")
        return torch.cat((x, cls.kinetic_maps(x)), dim=1)

    def multiscale_views(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve [N, escala, fase, 128, 128] sin usar etiquetas ni máscaras."""
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError(f"Esperado [N,3,256,256], recibido {tuple(x.shape)}")
        scales = []
        for size in self.crop_sizes:
            start = (256-size)//2
            crop = x[:, :, start:start+size, start:start+size]
            if size != self.encoder_input_size:
                crop = F.interpolate(
                    crop, size=(self.encoder_input_size, self.encoder_input_size),
                    mode="bilinear", align_corners=False,
                )
            scales.append(crop)
        return torch.stack(scales, dim=1)

    def encode_views(self, x: torch.Tensor) -> torch.Tensor:
        scales = self.multiscale_views(x)
        n, n_scales, _, height, width = scales.shape
        scale_batch = scales.reshape(n*n_scales, 3, height, width)
        return self.temporal_maps(scale_batch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.spatial_dropout(self.backbone(self.encode_views(x)))
        z = torch.cat((self.avgpool(z).flatten(1), self.maxpool(z).flatten(1)), dim=1)
        z = z.reshape(len(x), len(self.crop_sizes)*z.shape[1])
        return self.classifier(z)


def comparar_con_control(output: Path, report: dict, n_boot: int = 5000) -> dict | None:
    """Compara de forma pareada con la CNN por fases ya congelada en desarrollo."""
    if not CONTROL_OOF.exists() or not HISTORICAL_OOF.exists():
        return None
    selected_loss = report["selected_loss_for_future_test"]
    candidate_path = Path(output)/"comparacion"/f"oof_pacientes_{selected_loss}.csv"
    candidate = pd.read_csv(candidate_path).sort_values("patient_id").reset_index(drop=True)
    control = pd.read_csv(CONTROL_OOF).sort_values("patient_id").reset_index(drop=True)
    historical = pd.read_csv(HISTORICAL_OOF).sort_values("patient_id").reset_index(drop=True)
    for column in ("patient_id", "label"):
        if not np.array_equal(candidate[column].to_numpy(), control[column].to_numpy()):
            raise ValueError(f"Control y candidato no están alineados en {column}")
        if not np.array_equal(candidate[column].to_numpy(), historical[column].to_numpy()):
            raise ValueError(f"Histórico y candidato no están alineados en {column}")

    y = candidate.label.to_numpy(dtype=int)
    candidate_probability = candidate.probability_raw.to_numpy(dtype=float)
    control_probability = control.probability_raw.to_numpy(dtype=float)
    historical_probability = historical.probability.to_numpy(dtype=float)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(20261021)
    bootstrap = np.empty(n_boot, dtype=float)
    bootstrap_historical = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = np.r_[rng.choice(positive, len(positive), replace=True),
                       rng.choice(negative, len(negative), replace=True)]
        bootstrap[index] = (
            roc_auc_score(y[sample], candidate_probability[sample]) -
            roc_auc_score(y[sample], control_probability[sample])
        )
        bootstrap_historical[index] = (
            roc_auc_score(y[sample], candidate_probability[sample]) -
            roc_auc_score(y[sample], historical_probability[sample])
        )
    candidate_auc = float(roc_auc_score(y, candidate_probability))
    control_auc = float(roc_auc_score(y, control_probability))
    historical_auc = float(roc_auc_score(y, historical_probability))
    low, high = np.quantile(bootstrap, [.025, .975])
    historical_low, historical_high = np.quantile(bootstrap_historical, [.025, .975])
    comparison = {
        "status": "development_paired_comparison",
        "warning": (
            "Comparación adaptativa en desarrollo; no sustituye el test independiente "
            "y el intervalo condiciona a los modelos entrenados."
        ),
        "candidate": EXPERIMENT_NAME,
        "candidate_selected_loss": selected_loss,
        "candidate_pooled_auc": candidate_auc,
        "control": "cnn_imagen_fases_20261014_weighted",
        "control_pooled_auc": control_auc,
        "delta_auc_candidate_minus_control": candidate_auc-control_auc,
        "delta_auc_ci95": [float(low), float(high)],
        "historical": "cnn_imagen_early_fusion_original",
        "historical_pooled_auc": historical_auc,
        "delta_auc_candidate_minus_historical": candidate_auc-historical_auc,
        "delta_auc_historical_ci95": [float(historical_low), float(historical_high)],
        "bootstrap_samples": int(n_boot),
        "target_auc": 0.60,
        "target_reached": bool(candidate_auc >= 0.60),
        "improvement_supported": bool(low > 0),
        "adoption_rule": (
            "candidate_auc_at_least_0.60_and_delta_auc_ci95_low_above_zero"
        ),
        "adoption_supported": bool(candidate_auc >= 0.60 and low > 0),
        "test_images_loaded": 0,
    }
    base.json_write(Path(output)/"comparacion"/"comparacion_control.json", comparison)
    return comparison


def comparar_ejecuciones(root: Path, output: Path) -> dict:
    protocol = configuracion()
    preprocessing = {
        "phases": list(pdatos.FASES),
        "shape": [3, 256, 256],
        "phase_semantics": "ordered DCE times, never RGB",
        "center_crop_sizes": [256, 128, 64],
        "encoder_input_size": [128, 128],
        "scale_weights_shared": True,
        "phase_scaling": "uint8/255, matching the historical early-fusion CNN",
        "kinetic_maps": ["EARLY-PRE", "LATE-PRE", "EARLY-LATE"],
        "kinetic_scaling": "signed differences computed after uint8/255",
        "clinical_required": False,
    }
    report = imagen.comparar_ejecuciones(
        root, output, protocol=protocol, experiment_name=EXPERIMENT_NAME,
        model_class_name="CNNMultiescalaCinetica",
        primary_model="image_only_multiscale_kinetic_cnn_from_scratch",
        preprocessing=preprocessing,
        figure_title="CNN cinética multiescala: mediana e IQR de semillas y folds",
    )
    comparison = comparar_con_control(output, report)
    if comparison is not None:
        report["comparison_with_phase_cnn"] = comparison
        base.json_write(Path(output)/"comparacion"/"seleccion.json", report)
    return report


def verificar() -> dict:
    config = configuracion()
    model = CNNMultiescalaCinetica(config)
    return {
        "status": "preparado_sin_entrenar",
        "configuration": config["name"],
        "architecture": config["model"]["architecture"],
        "parameters": imagen.numero_parametros(model),
        "clinical_variables": False,
        "explicit_kinetic_maps": config["model"]["derived_maps"],
        "center_crop_sizes": config["model"]["center_crop_sizes"],
        "encoder_input_size": config["model"]["encoder_input_size"],
        "early_temporal_fusion": True,
        "patient_aggregation": imagen.PATIENT_AGGREGATION,
        "losses": list(imagen.LOSSES),
        "seeds": list(imagen.SEEDS),
        "outer_folds": list(imagen.FOLDS),
        "planned_runs": len(imagen.LOSSES)*len(imagen.SEEDS)*len(imagen.FOLDS),
        "test_images_loaded": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", nargs="?", choices=("verificar", "entrenar", "comparar"),
                        default="verificar")
    parser.add_argument("--datos", type=Path, default=DATA_DIR)
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--perdidas", nargs="+", choices=imagen.LOSSES,
                        default=list(imagen.LOSSES))
    parser.add_argument("--semillas", nargs="+", type=int, choices=imagen.SEEDS,
                        default=list(imagen.SEEDS))
    parser.add_argument("--folds", nargs="+", type=int, choices=imagen.FOLDS,
                        default=list(imagen.FOLDS))
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
        comparison = report.get("comparison_with_phase_cnn")
        print(json.dumps({
            "selected_loss": report["selected_loss_for_future_test"],
            "delta_auc_vs_phase_cnn": (comparison or {}).get(
                "delta_auc_candidate_minus_control"),
            "target_auc": (comparison or {}).get("target_auc"),
            "target_reached": (comparison or {}).get("target_reached"),
            "adoption_supported": (comparison or {}).get("adoption_supported"),
            "test_images_loaded": 0,
        }, ensure_ascii=False, indent=2))
        return
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.dispositivo == "auto" else args.dispositivo
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA no está disponible")
    config = configuracion()
    config["training"].update(
        epochs=args.epocas, batch_size=args.lote, num_workers=args.workers,
        min_epochs=min(config["training"]["min_epochs"], args.epocas),
    )
    output = args.salida/"pruebas" if args.prueba else args.salida
    for loss_name in dict.fromkeys(args.perdidas):
        for seed in dict.fromkeys(args.semillas):
            for fold in dict.fromkeys(args.folds):
                imagen.run_training(
                    config, loss_name, seed, fold, args.datos, output,
                    torch.device(device_name), smoke=args.prueba,
                    model_type=CNNMultiescalaCinetica, experiment_name=EXPERIMENT_NAME,
                    extra_source_files={"entrenamiento_cinetico.py": Path(__file__)},
                )


if __name__ == "__main__":
    main()
