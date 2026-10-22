"""Inferencia experimental de cuatro modelos por cantidad de cortes en fit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cancer_mama.app_logic import PHASES, uploads_from_paths, validate_phase_uploads
from cancer_mama.entrenamiento_antioverfit import CNNTemporalSimple
from cancer_mama.entrenamiento_imagen import numero_parametros


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(path):
    path = Path(path).resolve()
    if sha(path) != path.with_suffix(".json.sha256").read_text().strip():
        raise ValueError("Manifiesto modificado")
    record = json.loads(path.read_text())
    expected = {(k, seed, fold) for k in (10, 3, 1) for seed in (42, 2026) for fold in (0, 3)}
    actual = {(m["k"], m["seed"], m["fold"]) for m in record["models"]}
    if actual != expected or len(record["models"]) != 12:
        raise ValueError("Se requieren doce pesos identificados")
    if (record["model_class"] != "CNNTemporalSimple" or record["aggregation"] != "mean"
            or record["clinical_required"] is not False or record["threshold"] != .5
            or record["calibration"] != "none" or record["inference_slice_selection"] != "all_available"):
        raise ValueError("Contrato incompatible")
    for entry in record["models"]:
        relative = Path(entry["path"])
        target = (path.parent / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(path.parent):
            raise ValueError("Ruta fuera del paquete")
        if sha(target) != entry["sha256"]:
            raise ValueError("Peso modificado")
    return record


def load(path, entry):
    artifact = torch.load(Path(path).parent / entry["path"], map_location="cpu", weights_only=True)
    if set(artifact) != {"schema_version", "model_config", "state_dict"}:
        raise ValueError("El archivo contiene datos ajenos a inferencia")
    if artifact["model_config"] != entry["model_config"]:
        raise ValueError("Configuración incompatible")
    if not all(bool(torch.isfinite(value).all()) for value in artifact["state_dict"].values()):
        raise ValueError("Pesos no finitos")
    model = CNNTemporalSimple(artifact["model_config"])
    model.load_state_dict(artifact["state_dict"], strict=True)
    return model.eval()


def verify(path):
    record = manifest(path)
    counts = {numero_parametros(load(path, entry)) for entry in record["models"]}
    if counts != {551913}:
        raise ValueError("Número de parámetros incompatible")
    return {"models": 12, "parameters_per_model": 551913, "strict_loading": True,
            "checksums_verified": True, "optimizer_or_rng": False, "experimental": True}


@torch.inference_mode()
def predict(path, k, triples):
    record = manifest(path)
    samples = [validate_phase_uploads(uploads_from_paths(
        {phase: Path(png) for phase, png in zip(PHASES, triple)})) for triple in triples]
    if (not samples or len({sample.patient_id for sample in samples}) != 1
            or len({sample.sample_id for sample in samples}) != len(samples)):
        raise ValueError("Se requieren cortes distintos de una sola paciente")
    x = torch.from_numpy(np.stack([np.stack([sample.arrays[phase] for phase in PHASES])
                                  for sample in samples])).float().div_(255.)
    chosen = [entry for entry in record["models"] if entry["k"] == k]
    if len(chosen) != 4:
        raise ValueError("Se requieren cuatro modelos para el brazo solicitado")
    probabilities = [torch.sigmoid(load(path, entry)(x).squeeze(1)).numpy() for entry in chosen]
    probability = float(np.mean(probabilities))
    return {"fit_k": k, "models": 4, "n_slices": len(samples), "inference_slice_selection": "all_available",
            "probability_raw": probability, "threshold": .5, "pCR": int(probability >= .5),
            "calibration": "none", "experimental": True, "active_model_replaced": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verificar", "predecir"))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifiesto.json"))
    parser.add_argument("--k", type=int, choices=(10, 3, 1), default=10)
    for phase in ("pre", "early", "late"):
        parser.add_argument("--" + phase, nargs="+", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action == "verificar":
        result = verify(args.manifest)
    else:
        if not args.pre or not args.early or not args.late or not len(args.pre) == len(args.early) == len(args.late):
            parser.error("Se requieren listas PRE/EARLY/LATE del mismo tamaño")
        result = predict(args.manifest, args.k, list(zip(args.pre, args.early, args.late)))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
