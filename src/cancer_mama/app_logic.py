"""Validación e inferencia segura para la aplicación web educativa."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch

from cancer_mama.entrenamiento import CNN, CLINICAL_COLUMNS, apply_calibration, cargar_manifest, clinical_matrix


PHASES = ("PRE", "EARLY", "LATE")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
SAMPLE_PATTERN = re.compile(r"([A-Za-z0-9._-]+)_z([0-9]+)")


class InputValidationError(ValueError):
    """Error de entrada que puede mostrarse directamente a la persona usuaria."""


@dataclass(frozen=True)
class UploadedPhase:
    name: str
    data: bytes


@dataclass(frozen=True)
class ValidatedSample:
    sample_id: str
    patient_id: str
    arrays: dict[str, np.ndarray]
    filenames: dict[str, str]


@dataclass(frozen=True)
class ModelCard:
    manifest_path: Path
    checksum: str
    threshold: float
    aggregation: str
    status: str
    frozen_at: str
    model_count: int
    weights_ready: bool
    missing_weights: tuple[str, ...]
    internal_metrics: dict[str, float]


@dataclass(frozen=True)
class Prediction:
    probability_raw: float
    probability_calibrated: float
    predicted_class: int
    threshold: float
    model_count: int
    model_spread: float
    device: str
    elapsed_ms: float


def _decode_png(upload: UploadedPhase, expected_phase: str) -> tuple[str, np.ndarray]:
    if not upload.name or Path(upload.name).name != upload.name or "/" in upload.name or "\\" in upload.name:
        raise InputValidationError("El nombre del archivo no es válido.")
    if len(upload.data) == 0:
        raise InputValidationError(f"El archivo {expected_phase} está vacío.")
    if len(upload.data) > MAX_UPLOAD_BYTES:
        raise InputValidationError(f"El archivo {expected_phase} supera el límite de 5 MB.")
    path = Path(upload.name)
    if path.suffix.lower() != ".png" or not path.stem.endswith("_" + expected_phase):
        raise InputValidationError(
            f"La fase {expected_phase} debe ser un PNG cuyo nombre termine en _{expected_phase}.png."
        )
    sample_id = path.stem[: -(len(expected_phase) + 1)]
    try:
        with Image.open(BytesIO(upload.data)) as image:
            image.load()
            if image.format != "PNG":
                raise InputValidationError(f"{upload.name} no es un PNG real.")
            if getattr(image, "n_frames", 1) != 1:
                raise InputValidationError(f"{upload.name} debe contener una sola imagen.")
            if image.mode != "L" or image.size != (256, 256):
                raise InputValidationError(
                    f"{upload.name} debe ser monocromo de 8 bits y medir exactamente 256×256 px."
                )
            array = np.asarray(image, dtype=np.uint8).copy()
    except InputValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InputValidationError(f"No se ha podido leer {upload.name}: el archivo parece corrupto.") from exc
    return sample_id, array


def validate_phase_uploads(uploads: dict[str, UploadedPhase | None]) -> ValidatedSample:
    """Valida tres PNG y devuelve matrices uint8 alineadas por fase."""
    if set(uploads) != set(PHASES):
        raise InputValidationError("Deben proporcionarse exactamente las fases PRE, EARLY y LATE.")
    missing = [phase for phase in PHASES if uploads[phase] is None]
    if missing:
        raise InputValidationError("Faltan fases: " + ", ".join(missing) + ".")

    arrays: dict[str, np.ndarray] = {}
    filenames: dict[str, str] = {}
    sample_ids: set[str] = set()
    for phase in PHASES:
        upload = uploads[phase]
        assert upload is not None
        sample_id, array = _decode_png(upload, phase)
        sample_ids.add(sample_id)
        arrays[phase] = array
        filenames[phase] = upload.name

    if len(sample_ids) != 1:
        raise InputValidationError("Las tres fases no pertenecen al mismo corte.")
    sample_id = sample_ids.pop()
    match = SAMPLE_PATTERN.fullmatch(sample_id)
    if match is None:
        raise InputValidationError("El identificador debe terminar en _z seguido del número de corte.")
    return ValidatedSample(
        sample_id=sample_id,
        patient_id=match.group(1),
        arrays=arrays,
        filenames=filenames,
    )


def enhancement_map(sample: ValidatedSample) -> np.ndarray:
    """Crea un mapa RGB firmado de EARLY-PRE sin alterar la inferencia."""
    difference = (
        sample.arrays["EARLY"].astype(np.float32) - sample.arrays["PRE"].astype(np.float32)
    ) / 255.0
    scale = float(np.percentile(np.abs(difference), 99))
    if scale <= 1e-8:
        scale = 1.0
    normalized = np.clip(difference / scale, -1.0, 1.0)
    base = np.array([9.0, 18.0, 31.0], dtype=np.float32)
    positive = np.array([255.0, 126.0, 99.0], dtype=np.float32)
    negative = np.array([68.0, 190.0, 201.0], dtype=np.float32)
    magnitude = np.abs(normalized)[..., None] ** 0.72
    target = np.where((normalized >= 0)[..., None], positive, negative)
    return np.clip(base + magnitude * (target - base), 0, 255).astype(np.uint8)


def discover_local_examples(data_dir: Path, limit: int = 8) -> dict[str, dict[str, Path]]:
    """Descubre ejemplos locales sin aceptar rutas introducidas por el usuario."""
    train_dir = Path(data_dir) / "dataset" / "train"
    if not train_dir.is_dir():
        return {}
    examples: dict[str, dict[str, Path]] = {}
    for pre in train_dir.glob("*/*_PRE.png"):
        sample_id = pre.stem[:-4]
        companions = {
            "PRE": pre,
            "EARLY": pre.with_name(sample_id + "_EARLY.png"),
            "LATE": pre.with_name(sample_id + "_LATE.png"),
        }
        if all(path.is_file() for path in companions.values()):
            examples[sample_id] = companions
        if len(examples) >= limit:
            break
    return dict(sorted(examples.items()))


def uploads_from_paths(paths: dict[str, Path]) -> dict[str, UploadedPhase]:
    return {phase: UploadedPhase(path.name, path.read_bytes()) for phase, path in paths.items()}


def validate_clinical(values: dict[str, float | None]) -> np.ndarray:
    """Valida entradas originales; los ausentes se imputan dentro de cada modelo."""
    if set(values) != set(CLINICAL_COLUMNS):
        raise InputValidationError("Se necesitan edad, volumen tumoral, HR y HER2; pueden indicarse como desconocidos.")
    try:
        return clinical_matrix(pd.DataFrame([values])).astype(np.float32)
    except (ValueError, TypeError) as exc:
        raise InputValidationError(str(exc)) from exc


def prediction_key(sample: ValidatedSample, clinical: dict[str, float | None]) -> str:
    """Invalida resultados al cambiar las imágenes o cualquier dato clínico."""
    digest = sha256(sample.sample_id.encode())
    for phase in PHASES:
        digest.update(sample.arrays[phase].tobytes())
    digest.update(validate_clinical(clinical).tobytes())
    return digest.hexdigest()


def read_model_card(manifest_path: Path) -> ModelCard:
    """Lee la ficha del modelo y comprueba qué pesos están disponibles."""
    manifest_path = Path(manifest_path).resolve()
    raw = manifest_path.read_bytes()
    checksum = sha256(raw).hexdigest()
    checksum_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not checksum_path.is_file():
        raise ValueError("Falta el checksum del manifiesto.")
    expected = checksum_path.read_text(encoding="utf-8").strip()
    canonical = sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    if expected not in {checksum, canonical}:
        raise ValueError("El checksum del manifiesto no coincide.")
    manifest = json.loads(raw)
    missing = tuple(
        item["path"] for item in manifest["models"] if not (manifest_path.parent / item["path"]).is_file()
    )
    selection = manifest.get("selection", {})
    metrics = selection.get("metrics_raw_oof", selection.get("metrics", {}))
    return ModelCard(
        manifest_path=manifest_path,
        checksum=checksum,
        threshold=float(manifest["threshold"]),
        aggregation=str(manifest["aggregation"]),
        status=str(manifest.get("status", "desconocido")),
        frozen_at=str(manifest.get("frozen_at", "")),
        model_count=len(manifest["models"]),
        weights_ready=not missing,
        missing_weights=missing,
        internal_metrics={
            key: float(metrics[key])
            for key in ("roc_auc", "average_precision", "brier", "sensitivity", "specificity")
            if key in metrics
        },
    )


class InferenceEngine:
    """Ensemble verificado y cacheable por Streamlit."""

    def __init__(self, manifest_path: Path, device: str | None = None):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = cargar_manifest(self.manifest_path)
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(requested)
        self.models: list[CNN] = []
        for item in self.manifest["models"]:
            artifact = torch.load(
                self.manifest_path.parent / item["path"],
                map_location=self.device,
                weights_only=True,
            )
            if artifact["model_config"] != item["model_config"]:
                raise ValueError("Configuración de pesos y manifiesto diferentes.")
            model = CNN(artifact["model_config"]).to(self.device)
            model.load_state_dict(artifact["state_dict"], strict=True)
            model.eval()
            self.models.append(model)

    @torch.inference_mode()
    def predict(self, sample: ValidatedSample, clinical: dict[str, float | None]) -> Prediction:
        started = perf_counter()
        clinical_tensor = torch.from_numpy(validate_clinical(clinical)).to(self.device)
        tensor = torch.from_numpy(
            np.stack([sample.arrays[phase] for phase in PHASES]).astype(np.float32) / 255.0
        ).unsqueeze(0).to(self.device)
        values = np.array(
            [torch.sigmoid(model(tensor, clinical_tensor).squeeze()).item() for model in self.models], dtype=np.float64
        )
        # El ensemble promedia modelos por corte; la mediana se aplica entre
        # cortes de una paciente. Aquí solo se recibe un corte.
        raw = float(values.mean())
        calibrated = float(apply_calibration(np.array([raw]), self.manifest["calibration"])[0])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (perf_counter() - started) * 1000.0
        threshold = float(self.manifest["threshold"])
        return Prediction(
            probability_raw=raw,
            probability_calibrated=calibrated,
            predicted_class=int(calibrated >= threshold),
            threshold=threshold,
            model_count=len(self.models),
            model_spread=float(values.std(ddof=0)),
            device=str(self.device),
            elapsed_ms=elapsed_ms,
        )
