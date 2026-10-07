#!/usr/bin/env python3
"""Conserva ensembles completos en versiones independientes y verificables."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "modelos"
CODE_FILES = ("04_entrenamiento.py", "pipeline_datos.py", "requirements.txt")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def version_number(identifier):
    match = re.fullmatch(r"v([0-9]{3,})_[a-z0-9]+(?:_[a-z0-9]+)*", identifier)
    if not match or int(match[1]) < 1:
        raise ValueError("ID inválido: usa v001_nombre_fecha, con minúsculas y guiones bajos")
    return int(match[1])


def inside(base, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Ruta fuera de la versión: {relative}")
    target = (base / relative).resolve()
    if not target.is_relative_to(base.resolve()):
        raise ValueError(f"Ruta fuera de la versión: {relative}")
    return target


def check_manifest(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    expected = path.with_suffix(path.suffix + ".sha256").read_text().strip()
    if expected not in (hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()):
        raise ValueError("El checksum del manifiesto de origen no coincide")
    manifest = json.loads(raw)
    models = manifest["models"]
    if len(models) != 10 or {(m["seed"], m["fold"]) for m in models} != {(s, f) for s in (42, 2026) for f in range(5)}:
        raise ValueError("Se requieren diez pesos: semillas 42/2026 y folds 0-4")
    for item in [*models, *manifest.get("provenance_files", [])]:
        if sha256(path.parent / item["path"]) != item["sha256"]:
            raise ValueError(f"Checksum incorrecto en origen: {item['path']}")
    return manifest


def training_code(root):
    root = Path(root)
    legacy = root / "04_entrenamiento.py"
    packaged = root / "src/cancer_mama/entrenamiento.py"
    if legacy.is_file():
        return legacy
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(f"Falta el código de entrenamiento en {root}")


def code_command(code, action, manifest, arguments=()):
    code = Path(code).resolve()
    environment = os.environ.copy()
    if code.name == "entrenamiento.py" and code.parent.name == "cancer_mama":
        source = str(code.parent.parent)
        environment["PYTHONPATH"] = source + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
        command = [sys.executable, "-B", "-m", "cancer_mama.entrenamiento"]
    else:
        command = [sys.executable, "-B", str(code)]
    return [*command, action, "--manifest", str(manifest), *arguments], environment


def run_verification(code, manifest):
    command, environment = code_command(code, "verificar", manifest)
    result = subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


@contextmanager
def archive_lock(archive):
    archive.mkdir(parents=True, exist_ok=True)
    lock = archive / ".versionar.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("Hay otro archivado en curso; espera a que termine") from None
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


def overview(info):
    keys = ("id", "description", "archived_at", "status", "clinical_required", "n_models", "weights_bytes", "metrics_raw_oof")
    return {key: info[key] for key in keys}


def read_versions(archive=ARCHIVE):
    versions = []
    for folder in (Path(archive) / "versiones").glob("v*"):
        if not folder.is_dir():
            continue
        version_number(folder.name)
        path = folder / "version.json"
        if sha256(path) != path.with_suffix(".json.sha256").read_text().strip():
            raise ValueError(f"Metadatos modificados: {folder.name}")
        info = json.loads(path.read_text())
        if info["id"] != folder.name:
            raise ValueError(f"ID y carpeta diferentes: {folder.name}")
        versions.append(overview(info))
    return sorted(versions, key=lambda row: version_number(row["id"]))


def update_catalog(archive=ARCHIVE):
    archive = Path(archive)
    catalog = {"schema_version": 1, "versions": read_versions(archive)}
    fd, name = tempfile.mkstemp(prefix=".catalogo-", suffix=".tmp", dir=archive)
    try:
        os.close(fd)
        write_json(name, catalog)
        os.replace(name, archive / "catalogo.json")
    finally:
        Path(name).unlink(missing_ok=True)
    return catalog


def verify_version(identifier, archive=ARCHIVE, runtime=True):
    version_number(identifier)
    folder = Path(archive) / "versiones" / identifier
    info_path = folder / "version.json"
    if sha256(info_path) != info_path.with_suffix(".json.sha256").read_text().strip():
        raise ValueError(f"Metadatos modificados: {identifier}")
    info = json.loads(info_path.read_text())
    if info["id"] != identifier:
        raise ValueError("ID de versión incompatible")
    expected = info["files_sha256"]
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()
              and "__pycache__" not in p.parts and p.relative_to(folder).as_posix() not in ("version.json", "version.json.sha256")}
    if actual != set(expected):
        raise ValueError(f"Inventario cambiado: faltan {sorted(set(expected)-actual)}; sobran {sorted(actual-set(expected))}")
    for relative, checksum in expected.items():
        if sha256(inside(folder, relative)) != checksum:
            raise ValueError(f"Archivo modificado en {identifier}: {relative}")
    manifest_path = folder / "manifiesto.json"
    manifest = check_manifest(manifest_path)
    for item in [*manifest["models"], *manifest.get("provenance_files", [])]:
        inside(folder, item["path"])
    model_check = run_verification(training_code(folder / "codigo"), manifest_path) if runtime else None
    return {"id": identifier, "files_verified": len(expected), "models": len(manifest["models"]),
            "checksums_verified": True, "runtime": model_check}


def archive_version(manifest_path, identifier, description, evidence=(), archive=ARCHIVE, source_root=ROOT):
    version_number(identifier)
    archive, source_root = Path(archive).resolve(), Path(source_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    with archive_lock(archive):
        versions = read_versions(archive)
        for previous in versions:
            verify_version(previous["id"], archive=archive, runtime=False)
        if any(version_number(v["id"]) == version_number(identifier) for v in versions):
            raise ValueError("Ese número de versión ya está ocupado; guarda una versión nueva")
        destination = archive / "versiones" / identifier
        if destination.exists():
            raise ValueError("La versión ya existe y no se puede sobrescribir")
        source_manifest = check_manifest(manifest_path)
        source_code = training_code(source_root)
        source_check = run_verification(source_code, manifest_path)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=archive))
        try:
            def copy(source, relative, expected=None):
                source, target = Path(source), inside(staging, relative)
                if expected and sha256(source) != expected:
                    raise ValueError(f"Origen modificado: {source}")
                if target.exists() and sha256(target) != sha256(source):
                    raise ValueError(f"Dos archivos distintos usan la misma ruta: {relative}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                if sha256(target) != (expected or sha256(source)):
                    raise ValueError(f"Copia incompleta: {relative}")
                return Path(relative).as_posix()

            def evidence_path(source):
                source = Path(source).resolve()
                if source.is_relative_to(source_root):
                    return Path("evidencia/repo") / source.relative_to(source_root)
                return Path("evidencia/externa") / source.name

            operative = deepcopy(source_manifest)
            weights_bytes = 0
            for item in operative["models"]:
                original = manifest_path.parent / item["path"]
                weights_bytes += original.stat().st_size
                item["path"] = copy(original, f"pesos/semilla_{item['seed']}_fold_{item['fold']}.pt", item["sha256"])
                for name in ("config.json", "summary.json", "history.csv", "environment.json"):
                    auxiliary = original.parent / name
                    if auxiliary.is_file():
                        copy(auxiliary, evidence_path(auxiliary))
            for item in operative.get("provenance_files", []):
                original = manifest_path.parent / item["path"]
                item["path"] = copy(original, evidence_path(original), item["sha256"])
            if operative.get("test_metrics_path"):
                original = manifest_path.parent / operative["test_metrics_path"]
                operative["test_metrics_path"] = copy(original, evidence_path(original))
            for path in evidence:
                copy(path, evidence_path(path))
            copy(manifest_path, "evidencia/manifiesto_origen.json")
            copy(manifest_path.with_suffix(manifest_path.suffix + ".sha256"), "evidencia/manifiesto_origen.json.sha256")
            if source_code.name == "04_entrenamiento.py":
                for name in CODE_FILES:
                    copy(source_root / name, Path("codigo") / name)
            else:
                for source in sorted((source_root / "src/cancer_mama").rglob("*.py")):
                    copy(source, Path("codigo") / source.relative_to(source_root))
                copy(source_root / "requirements.txt", "codigo/requirements.txt")
                if (source_root / "pyproject.toml").is_file():
                    copy(source_root / "pyproject.toml", "codigo/pyproject.toml")
                    if (source_root / "README.md").is_file():
                        copy(source_root / "README.md", "codigo/README.md")
            write_json(staging / "manifiesto.json", operative)
            (staging / "manifiesto.json.sha256").write_text(sha256(staging / "manifiesto.json") + "\n")
            packages = {}
            for package in ("torch", "numpy", "pandas", "Pillow", "scikit-learn", "matplotlib", "requests"):
                try:
                    packages[package] = metadata.version(package)
                except metadata.PackageNotFoundError:
                    packages[package] = None
            write_json(staging / "entorno.json", {"scope": "entorno compatible usado al archivar; no afirma ser el entorno original de entrenamiento",
                       "python": platform.python_version(), "platform": platform.platform(), "packages": packages})
            frozen_check = run_verification(training_code(staging / "codigo"), staging / "manifiesto.json")
            if frozen_check != source_check:
                raise ValueError("El paquete no coincide con el modelo de origen")
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_root, capture_output=True, text=True)
            metrics = source_manifest.get("selection", {}).get("metrics", source_manifest.get("selection", {}).get("metrics_raw_oof", {}))
            try:
                archived_at = datetime.now(ZoneInfo("Europe/Madrid")).isoformat()
            except ZoneInfoNotFoundError:
                archived_at = datetime.now().astimezone().isoformat()
            info = {"schema_version": 1, "id": identifier, "description": description,
                    "archived_at": archived_at,
                    "archived_from_git_commit": git.stdout.strip() if git.returncode == 0 else None,
                    "source_manifest_sha256": sha256(manifest_path), "status": operative["status"],
                    "clinical_required": operative.get("preprocessing", {}).get("clinical_required", False),
                    "n_models": len(operative["models"]), "weights_bytes": weights_bytes, "metrics_raw_oof": metrics,
                    "runtime_verification": frozen_check,
                    "files_sha256": {p.relative_to(staging).as_posix(): sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()}}
            write_json(staging / "version.json", info)
            (staging / "version.json.sha256").write_text(sha256(staging / "version.json") + "\n")
            # La carpeta definitiva solo aparece cuando todas las copias y verificaciones terminaron.
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(destination)
            update_catalog(archive)
            return overview(info)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("listar")
    commands.add_parser("catalogar")
    save = commands.add_parser("guardar")
    save.add_argument("--manifest", type=Path, required=True)
    save.add_argument("--id", required=True)
    save.add_argument("--descripcion", required=True)
    save.add_argument("--evidencia", type=Path, nargs="*", default=[])
    verify = commands.add_parser("verificar")
    verify.add_argument("version", nargs="?")
    verify.add_argument("--solo-hashes", action="store_true")
    predict = commands.add_parser("predecir")
    predict.add_argument("version")
    predict.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.action == "guardar":
            result = archive_version(args.manifest, args.id, args.descripcion, args.evidencia)
        elif args.action == "listar":
            result = read_versions()
        elif args.action == "catalogar":
            result = update_catalog()
        elif args.action == "verificar":
            identifiers = [args.version] if args.version else [row["id"] for row in read_versions()]
            result = [verify_version(v, runtime=not args.solo_hashes) for v in identifiers]
        else:
            verify_version(args.version, runtime=False)
            folder = ARCHIVE / "versiones" / args.version
            arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
            if any(arg == "--manifest" or arg.startswith("--manifest=") for arg in arguments):
                raise ValueError("El manifiesto se selecciona con la versión; no pases --manifest")
            command, environment = code_command(training_code(folder / "codigo"), "predecir", folder / "manifiesto.json", arguments)
            completed = subprocess.run(command, env=environment)
            return completed.returncode
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
        parser.exit(1, f"Error: {detail}\n")


if __name__ == "__main__":
    raise SystemExit(main())
