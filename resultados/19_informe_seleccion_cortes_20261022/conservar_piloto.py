"""Conserva doce pesos de inferencia como p002 experimental, sin publicar."""
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
import torch

from cancer_mama import entrenamiento as base
from cancer_mama import seleccion_cortes as pilot

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = pilot.OUTPUT
IDENTIFIER = "p002_seleccion_cortes_20261022"


def main():
    assert json.loads((RUN / "estado_piloto.json").read_text())["status"] == "complete"
    baseline = json.loads((RUN / "modelos_antes.json").read_text())
    model_root = ROOT / "modelos"
    assert baseline == {p.relative_to(model_root).as_posix(): base.sha256(p)
                        for p in model_root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    destination = model_root / "pilotos" / IDENTIFIER
    assert not destination.exists(), "No sobrescribir paquetes existentes"
    assert not any(p.name.startswith("p002_") for p in destination.parent.iterdir())
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix=".seleccion-", dir=destination.parent) as temporary:
        folder = Path(temporary)
        entries = []
        for k in pilot.KS:
            for seed in (42, 2026):
                for fold in (0, 3):
                    source = RUN / f"cnn_seleccion_cortes_{k}/weighted/seed_{seed}/fold_{fold}"
                    cfg = json.loads((source / "config.json").read_text())
                    summary = json.loads((source / "summary.json").read_text())
                    relative = f"pesos/k_{k}_semilla_{seed}_fold_{fold}.pt"
                    target = folder / relative
                    target.parent.mkdir(exist_ok=True)
                    shutil.copy2(source / "inference.pt", target)
                    assert base.sha256(target) == summary["inference_sha256"]
                    artifact = torch.load(target, map_location="cpu", weights_only=True)
                    assert set(artifact) == {"schema_version", "model_config", "state_dict"}
                    entries.append({"k": k, "loss": "weighted", "seed": seed, "fold": fold,
                                    "path": relative, "sha256": base.sha256(target),
                                    "model_config": artifact["model_config"], "best_epoch": summary["best_epoch"],
                                    "training_config": cfg})
                    evidence = folder / f"evidencia/ejecuciones/k_{k}_semilla_{seed}_fold_{fold}"
                    evidence.mkdir(parents=True)
                    for name in ("config.json", "summary.json", "history.csv", "environment.json"):
                        shutil.copy2(source / name, evidence / name)
        sources = folder / "codigo/src/cancer_mama"
        sources.mkdir(parents=True)
        for path in (ROOT / "src/cancer_mama").glob("*.py"):
            shutil.copy2(path, sources / path.name)
        for name in ("requirements.txt", "pyproject.toml"):
            shutil.copy2(ROOT / name, folder / "codigo" / name)
        shutil.copy2(OUT / "inferencia_piloto.py", folder / "inferencia.py")
        evidence = folder / "evidencia/informe"
        evidence.mkdir(parents=True)
        for path in OUT.iterdir():
            if path.suffix in {".md", ".json", ".csv", ".png"}:
                shutil.copy2(path, evidence / path.name)
        record = {"schema_version": 1, "id": IDENTIFIER, "kind": "two_fold_slice_selection_pilot",
                  "source_commit": "d9f46fe105497a77ffbcb9211b7d61e8ca6ac0b2",
                  "status": "experimental_no_adoptado", "recommended_for_deployment": False,
                  "model_class": "CNNTemporalSimple", "models": entries, "fit_k": [10, 3, 1],
                  "folds": [0, 3], "seeds": [42, 2026], "aggregation": "mean",
                  "clinical_required": False, "phases": ["PRE", "EARLY", "LATE"],
                  "shape": [3, 256, 256], "scaling": "uint8/255", "threshold": .5,
                  "calibration": "none", "inference_slice_selection": "all_available",
                  "test_images_loaded": 0,
                  "prediction_scope": "Four models per k for future inputs; OOF uses two seeds of the excluded fold",
                  "metrics": json.loads((OUT / "evaluacion_recalculada.json").read_text())}
        base.json_write(folder / "manifiesto.json", record)
        (folder / "manifiesto.json.sha256").write_text(base.sha256(folder / "manifiesto.json") + "\n")
        (folder / "README.md").write_text(
            "# Piloto experimental de selección de cortes\n\n"
            "Doce pesos: k=10/3/1 × semillas 42/2026 × folds 0/3. BCE ponderada por paciente. "
            "Ningún brazo alcanza AUC media 0,60. No es una versión productiva y no sustituye V002.\n\n"
            "El parámetro k identifica los cortes usados en fit. En inferencia se usan todos los cortes aportados. "
            "Cuatro modelos por k, media de probabilidades crudas entre modelos y cortes, umbral fijo 0,5 y sin calibración. "
            "Las OOF del informe usan solo dos semillas del fold que excluye cada paciente, por lo que no equivalen "
            "a aplicar los cuatro modelos del paquete sobre train. Positivo significa pCR.\n\n"
            "```bash\nPYTHONPATH=codigo/src python -B inferencia.py verificar\n"
            "PYTHONPATH=codigo/src python -B inferencia.py predecir --k 3 "
            "--pre paciente_z000_PRE.png --early paciente_z000_EARLY.png --late paciente_z000_LATE.png\n```\n\n"
            "Solo pesos de inferencia, fuentes congeladas, configuraciones, métricas y hashes. "
            "No se incluyen datasets ni estado de optimizador/RNG.\n", encoding="utf-8")
        spec = importlib.util.spec_from_file_location("slice_pilot_inference", folder / "inferencia.py")
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        verification = runtime.verify(folder / "manifiesto.json")
        with tempfile.TemporaryDirectory(prefix="seleccion-aislada-") as isolated_directory:
            sandbox = Path(isolated_directory)
            isolated = sandbox / IDENTIFIER
            shutil.copytree(folder, isolated)
            assert not (sandbox / "breastdcedl").exists()
            rng = np.random.default_rng(20261022)
            triples, arrays = [], []
            for z in range(2):
                triple, planes = [], []
                for phase in ("PRE", "EARLY", "LATE"):
                    path = sandbox / f"sintetica_z{z:03d}_{phase}.png"
                    pixels = rng.integers(0, 256, (256, 256), dtype=np.uint8)
                    Image.fromarray(pixels).save(path)
                    triple.append(path); planes.append(pixels)
                triples.append(triple); arrays.append(planes)
            x = torch.from_numpy(np.asarray(arrays)).float().div_(255.)
            differences = []
            for k in pilot.KS:
                source = runtime.predict(folder / "manifiesto.json", k, triples)
                with torch.inference_mode():
                    direct = float(np.mean([torch.sigmoid(runtime.load(folder / "manifiesto.json", m)(x)).numpy()
                                            for m in entries if m["k"] == k]))
                assert abs(source["probability_raw"] - direct) <= 2e-6
                args = []
                for i, phase in enumerate(("pre", "early", "late")):
                    args += ["--" + phase, *[str(triple[i]) for triple in triples]]
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(isolated / "codigo/src")
                result = subprocess.run([sys.executable, "-B", str(isolated / "inferencia.py"),
                                         "predecir", "--k", str(k), *args],
                                        cwd=sandbox, env=environment, capture_output=True, text=True, check=True)
                prediction = json.loads(result.stdout)
                delta = abs(source["probability_raw"] - prediction["probability_raw"])
                assert delta <= 2e-6 and source["pCR"] == prediction["pCR"]
                differences.append({"k": k, "source": source, "isolated": prediction,
                                    "direct_tensor_probability": direct, "absolute_difference": delta})
        audit = {"verified_at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
                 "loading": verification, "isolated_inference": differences,
                 "previous_model_files_unchanged": len(baseline), "dataset_needed": False,
                 "test_images_loaded": 0, "optimizer_or_rng_in_exports": False}
        base.json_write(folder / "verificacion_paquete.json", audit)
        base.json_write(evidence / "verificacion_paquete.json", audit)
        inventory = {p.relative_to(folder).as_posix(): base.sha256(p)
                     for p in sorted(folder.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
        metadata = {"id": IDENTIFIER, "created_at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
                    "status": record["status"], "files_sha256": inventory,
                    "weights_bytes": sum((folder / m["path"]).stat().st_size for m in entries),
                    "models": 12, "production_version_created": False}
        base.json_write(folder / "paquete.json", metadata)
        (folder / "paquete.json.sha256").write_text(base.sha256(folder / "paquete.json") + "\n")
        folder.rename(destination)
    base.json_write(OUT / "verificacion_paquete.json", audit)
    base.json_write(OUT / "paquete_preparado.json", {"path": str(destination.relative_to(ROOT)),
                    "metadata_sha256": base.sha256(destination / "paquete.json"), **metadata})
    print(json.dumps({"path": str(destination), "models": 12, "weights_bytes": metadata["weights_bytes"],
                      "isolated_differences": [row["absolute_difference"] for row in differences]}, indent=2))


if __name__ == "__main__":
    main()
