"""Transferencia ImageNet congelada y fusión clínica con selección interna."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from . import entrenamiento as base
from . import experimento_loss as previous
from . import informe_loss as diagnosis
from .paths import PROJECT_ROOT, RESULTS_DIR

DEFAULT_OUTPUT = RESULTS_DIR / "04_entrenamiento/transfer_multimodal_20261010"
REFERENCE = diagnosis.DEFAULT_REPORT
FAMILIES = ("lineal", "no_lineal", "fusion_tardia")
SEED = 20261010


def grid(family, smoke=False):
    if family not in FAMILIES:
        raise ValueError("Familia desconocida")
    values = []
    for k in (16, 64):
        if family == "no_lineal":
            values.extend(dict(family=family, k=k, lr=lr, iterations=n)
                          for lr in (.03, .1) for n in (100, 250))
        elif family == "lineal":
            values.extend(dict(family=family, k=k, C=c, clinical_scale=scale)
                          for c in (.001, .01, .1, 1.) for scale in (1., 4.))
        else:
            values.extend(dict(family=family, k=k, C=c, image_weight=w)
                          for c in (.001, .01, .1, 1.) for w in (0., .25, .5, .75))
    return values[:1] if smoke else values


def fit_clinical(frame):
    values = base.clinical_matrix(frame)
    if np.isnan(values).all(axis=0).any():
        raise ValueError("Variable clínica ausente en fit")
    medians = np.nanmedian(values, axis=0)
    raw = clinical_design(values, medians)
    return {"medians": medians, "scaler": StandardScaler().fit(raw)}


def clinical_design(values, medians):
    missing = np.isnan(values)
    x = np.where(missing, medians, values)
    return np.column_stack((x, missing.astype(float), x[:, 2] * x[:, 3]))


def transform_clinical(frame, state):
    return state["scaler"].transform(clinical_design(base.clinical_matrix(frame), state["medians"]))


def fit_projection(features, smoke=False):
    features = np.asarray(features, dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("Características no finitas")
    scale = StandardScaler().fit(features)
    n = min(64, features.shape[0] - 1, features.shape[1])
    if not smoke and n != 64:
        raise ValueError("Fit insuficiente para 64 componentes")
    pca = PCA(n_components=n, whiten=True, svd_solver="randomized", random_state=SEED)
    pca.fit(scale.transform(features))
    return {"scaler": scale, "pca": pca}


def project(features, state):
    return state["pca"].transform(state["scaler"].transform(np.asarray(features, dtype=np.float64)))


def logistic(c=1.):
    return LogisticRegression(C=c, solver="lbfgs", max_iter=4000, random_state=SEED)


def fit_head(config, visual, clinical, labels):
    z = visual[:, :config["k"]]
    if config["family"] == "fusion_tardia":
        return {"clinical": logistic().fit(clinical, labels),
                "visual": logistic(config["C"]).fit(z, labels)}
    x = np.column_stack((z, clinical * config.get("clinical_scale", 1.)))
    if config["family"] == "lineal":
        return logistic(config["C"]).fit(x, labels)
    return HistGradientBoostingClassifier(learning_rate=config["lr"], max_iter=config["iterations"],
        max_leaf_nodes=4, min_samples_leaf=30, l2_regularization=5., early_stopping=False,
        random_state=SEED).fit(x, labels)


def head_probability(head, config, visual, clinical):
    z = visual[:, :config["k"]]
    if config["family"] == "fusion_tardia":
        w = config["image_weight"]
        return w * head["visual"].predict_proba(z)[:, 1] + (1-w) * head["clinical"].predict_proba(clinical)[:, 1]
    x = np.column_stack((z, clinical * config.get("clinical_scale", 1.)))
    return head.predict_proba(x)[:, 1]


def select_head(family, fit_visual, fit_clinic, fit_labels, select_visual, select_clinic,
                select_labels, smoke=False):
    """No recibe datos exteriores; la selección es completamente interna."""
    records, best = [], None
    for config in grid(family, smoke):
        head = fit_head(config, fit_visual, fit_clinic, fit_labels)
        p = head_probability(head, config, select_visual, select_clinic)
        loss = float(base.log_loss(select_labels, p, labels=[0, 1]))
        if not np.isfinite(loss):
            raise ValueError("Pérdida de selección no finita")
        records.append(dict(config, selection_logloss=loss,
                            selection_auc=float(base.roc_auc_score(select_labels, p))))
        if best is None or loss < best["selection_logloss"]:
            best = {"head": head, "config": deepcopy(config), "selection_logloss": loss}
    return best, records


def patient_features(samples, slice_features):
    """Promedio por paciente sin usar etiquetas, folds ni variables clínicas."""
    if len(samples) != len(slice_features) or not np.isfinite(slice_features).all():
        raise ValueError("Características desalineadas o no finitas")
    if samples.sample_id.duplicated().any():
        raise ValueError("Cortes duplicados")
    ids = sorted(samples.patient_id.unique())
    lookup = {pid: i for i, pid in enumerate(ids)}
    accum = np.zeros((len(ids), slice_features.shape[1]), dtype=np.float64)
    counts = np.zeros(len(ids), dtype=int)
    for pid, feature in zip(samples.patient_id, slice_features):
        accum[lookup[pid]] += feature
        counts[lookup[pid]] += 1
    return ids, (accum / counts[:, None]).astype(np.float32)


def frozen_backbone(path=None, device="cpu"):
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights if path is None else None)
    if path is not None:
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    return model.eval().to(device), weights.transforms()


@torch.inference_mode()
def visual_features(images, model, transform):
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 256, 256):
        raise ValueError("Se esperan tres fases de 256x256")
    x = images.float() / 255. if images.dtype == torch.uint8 else images.float()
    if x.min() < 0 or x.max() > 1:
        raise ValueError("Intensidades fuera de [0,1]")
    views = (x, x[:, :, 64:192, 64:192])
    return torch.cat([model(transform(view)) for view in views], dim=1)


def extract(samples, root, output, device, smoke, progress):
    dest = output / "features.npz"
    meta_path = output / "features.json"
    expected = {"sample_signature": previous.signature(samples), "smoke": smoke,
                "representation": "ResNet50_ImageNet1K_V2_full_and_center128_mean_patient"}
    if dest.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if any(meta[k] != v for k, v in expected.items()) or base.sha256(dest) != meta["sha256"]:
            raise ValueError("Características modificadas o de otra extracción")
        if base.sha256(output / "resnet50_imagenet_v2.pth") != meta["backbone_sha256"]:
            raise ValueError("Pesos preentrenados modificados")
        with np.load(dest) as arrays:
            return arrays["patient_id"].tolist(), arrays["features"].copy()
    from torchvision.models import ResNet50_Weights
    weight_path = output / "resnet50_imagenet_v2.pth"
    if not weight_path.exists():
        state = ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True, check_hash=True)
        previous.save_torch(weight_path, state)
    model, transform = frozen_backbone(weight_path, device)
    chosen = samples
    if smoke:
        ids = samples.drop_duplicates("patient_id").groupby(["fold", "pCR"]).head(8).patient_id
        chosen = samples[samples.patient_id.isin(ids)].groupby("patient_id", group_keys=False).head(2).reset_index(drop=True)
    # Reutiliza solo la caché de desarrollo que ya está validada contra PNG en el protocolo.
    cache = base.CacheTrain(samples, root, previous.DEFAULT_OUTPUT / ".cache", workers=4)
    loader = base.make_loader(chosen, root, 48, 0 if smoke else 4, torch.Generator().manual_seed(SEED), cache=cache)
    collected = []
    started = time.monotonic()
    for step, batch in enumerate(loader):
        x = batch["image"].to(device)
        collected.append(visual_features(x, model, transform).cpu().numpy())
        if step % 20 == 0 or step + 1 == len(loader):
            progress({"stage": "extracting", "batch": step+1, "batches": len(loader)})
            print(f"Extracción {step+1}/{len(loader)}; {time.monotonic()-started:.1f}s", flush=True)
    ids, features = patient_features(chosen, np.concatenate(collected))
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, patient_id=np.asarray(ids, dtype=str), features=features)
    tmp.replace(dest)
    base.json_write(meta_path, dict(expected, sha256=base.sha256(dest), patients=len(ids),
                    backbone_sha256=base.sha256(weight_path), elapsed_seconds=time.monotonic()-started))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ids, features


def prediction_frame(patients, indices, p):
    frame = patients.iloc[indices][["patient_id", "pCR", "fold", "dataset"]].rename(
        columns={"pCR": "label", "dataset": "cohort"}).copy()
    frame["probability"] = p
    return frame.sort_values("patient_id").reset_index(drop=True)


def fit_fold(samples, patients, features, fold, output, smoke=False):
    fit, selection, outer = previous.split_patients(samples, fold)
    sets = [set(x.patient_id) for x in (fit, selection, outer)]
    fit_idx, sel_idx, out_idx = [np.flatnonzero(patients.patient_id.isin(ids)) for ids in sets]
    if min(map(len, (fit_idx, sel_idx, out_idx))) == 0:
        raise ValueError("Partición vacía")
    folder = output / f"fold_{fold}"
    folder.mkdir(exist_ok=True)
    expected = {"fold": fold, "smoke": smoke,
                "partition_signatures": {role: previous.signature(x) for role, x in
                                         (("fit", fit), ("selection", selection), ("outer", outer))}}
    summary_path = folder / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if any(summary[k] != value for k, value in expected.items()):
            raise ValueError("Particiones distintas del resultado conservado")
        for file, digest in summary["files"].items():
            if base.sha256(folder / file) != digest:
                raise ValueError("Resultado conservado modificado")
        return summary
    partition = patients[["patient_id"]].copy()
    partition["role"] = [next(role for role, ids in zip(("fit", "selection", "outer"), sets) if p in ids)
                         for p in partition.patient_id]
    partition.to_csv(folder / "particiones.csv", index=False)
    clinical_state = fit_clinical(patients.iloc[fit_idx])
    projection = fit_projection(features[fit_idx], smoke)
    c = transform_clinical(patients, clinical_state)
    z = project(features, projection)
    y = patients.pCR.to_numpy(dtype=int)
    records, summaries = [], {}
    baseline = logistic().fit(c[fit_idx], y[fit_idx])
    for family in (*FAMILIES, "clinica"):
        if family == "clinica":
            chosen = {"head": baseline, "config": {"family": family}}
            probabilities = baseline.predict_proba(c)[:, 1]
        else:
            chosen, choices = select_head(family, z[fit_idx], c[fit_idx], y[fit_idx],
                                         z[sel_idx], c[sel_idx], y[sel_idx], smoke)
            records.extend(choices)
            probabilities = head_probability(chosen["head"], chosen["config"], z, c)
        inner = prediction_frame(patients, sel_idx, probabilities[sel_idx])
        held_out = prediction_frame(patients, out_idx, probabilities[out_idx])
        calibrated, calibration, threshold = diagnosis.calibrate_inner(inner, held_out)
        inner.to_csv(folder / f"inner_{family}.csv", index=False)
        held_out.to_csv(folder / f"oof_raw_{family}.csv", index=False)
        calibrated.to_csv(folder / f"oof_calibrated_{family}.csv", index=False)
        package = {"schema_version": 1, "family": family, "chosen": chosen, "clinical": clinical_state,
                   "projection": projection, "calibration": calibration, "threshold": threshold,
                   "fit_ids": sorted(sets[0]), "selection_ids": sorted(sets[1]), "outer_ids": sorted(sets[2])}
        joblib.dump(package, folder / f"model_{family}.joblib", compress=3)
        summaries[family] = {"config": chosen["config"], "threshold": threshold,
            "fit": base.binary_metrics(y[fit_idx], probabilities[fit_idx]),
            "selection": base.binary_metrics(y[sel_idx], probabilities[sel_idx]),
            "outer_raw": diagnosis.metrics(held_out), "outer_calibrated": diagnosis.metrics(calibrated)}
        print(f"Fold {fold} {family}: selección {summaries[family]['selection']['log_loss']:.4f}; "
              f"parámetros {chosen['config']}", flush=True)
    base.json_write(folder / "seleccion.json", records)
    files = {p.name: base.sha256(p) for p in folder.iterdir() if p.is_file() and p != summary_path}
    result = dict(expected, models=summaries, files=files)
    base.json_write(summary_path, result)
    return result


def compare(samples, output, repetitions=2000):
    patients = samples.drop_duplicates("patient_id").sort_values("patient_id").reset_index(drop=True)
    reference = pd.read_csv(REFERENCE / "oof_calibrated_inner_auc.csv")
    secondary = pd.read_csv(REFERENCE / "oof_calibrated_inner_logloss.csv")
    keys = ["patient_id", "label", "fold", "cohort"]
    roster = patients.rename(columns={"pCR": "label", "dataset": "cohort"})[keys]
    if not reference[keys].equals(roster) or not secondary[keys].equals(roster):
        raise ValueError("El control no corresponde a estas pacientes")
    dest = output / "comparacion"
    dest.mkdir(exist_ok=True)
    metrics = [{"candidate": "control_auc_calibrated", **diagnosis.metrics(reference)},
               {"candidate": "control_logloss_calibrated", **diagnosis.metrics(secondary)}]
    comparisons = {}
    fold_rows = []
    for family in (*FAMILIES, "clinica"):
        for scale in ("raw", "calibrated"):
            rows = []
            for fold in range(5):
                folder = output / f"fold_{fold}"
                summary = json.loads((folder / "summary.json").read_text())
                if summary["smoke"]:
                    raise ValueError("Una prueba no es una comparación válida")
                path = folder / f"oof_{scale}_{family}.csv"
                if base.sha256(path) != summary["files"][path.name]:
                    raise ValueError("OOF modificada")
                rows.append(pd.read_csv(path))
            oof = pd.concat(rows).sort_values("patient_id").reset_index(drop=True)
            if not oof[keys].equals(roster):
                raise ValueError("Cobertura OOF incorrecta")
            oof.to_csv(dest / f"oof_{scale}_{family}.csv", index=False)
            metrics.append({"candidate": f"{family}_{scale}", **diagnosis.metrics(oof)})
            for fold, data in oof.groupby("fold"):
                fold_rows.append({"candidate": f"{family}_{scale}", "fold": int(fold), **diagnosis.metrics(data)})
            if scale == "calibrated" and family in FAMILIES:
                intervals = previous.paired_intervals(reference, oof, repetitions)
                intervals_v003 = previous.paired_intervals(secondary, oof, repetitions)
                delta = {k: diagnosis.metrics(oof)[k] - diagnosis.metrics(reference)[k]
                         for k in ("log_loss", "brier", "roc_auc", "mean_fold_auc")}
                uses_images = any(json.loads((output / f"fold_{f}/summary.json").read_text())["models"][family]["config"].get("image_weight", 1) > 0 for f in range(5))
                comparisons[family] = {"delta_vs_control": delta, "intervals_98_33": intervals,
                    "intervals_vs_v003_98_33": intervals_v003, "uses_images": uses_images,
                    "promising_internal_result": intervals["log_loss"]["upper"] < 0
                    and intervals["mean_fold_auc"]["lower"] >= -.01 and delta["brier"] <= 0 and uses_images}
                print(f"Comparación completada {family}: {comparisons[family]['promising_internal_result']}", flush=True)
    report = {"metrics": metrics, "comparisons": comparisons, "bootstrap_repetitions": repetitions,
              "test_images_loaded": 0, "automatic_promotion": False,
              "warning": "Desarrollo interno adaptativo; no confirmación externa ni comparación equiparada con AUC histórica v002."}
    base.json_write(dest / "resumen.json", report)
    pd.DataFrame(metrics).drop(columns="fold_auc").to_csv(dest / "metricas.csv", index=False)
    pd.DataFrame(fold_rows).drop(columns="fold_auc").to_csv(dest / "metricas_folds.csv", index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", choices=("ejecutar", "extraer", "comparar"), nargs="?", default="ejecutar")
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dispositivo", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--prueba", action="store_true")
    args = parser.parse_args()
    output = args.salida / "pruebas" if args.prueba else args.salida
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / "runner.lock").open("a") as lock, threadpool_limits(limits=4):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"status": "running", "pid": os.getpid(), "started_at": previous.now(), "smoke": args.prueba}
        def progress(values):
            state.update(values, updated_at=previous.now())
            base.json_write(output / "estado.json", state)
        progress({"stage": "verifying"})
        try:
            torch.set_num_threads(8)
            base.seed_everything(SEED)
            samples, old_protocol = diagnosis.verify_inputs(base.pdatos.DATOS, previous.DEFAULT_OUTPUT)
            protected = previous.protected_snapshot()
            import torchvision
            sources = {str(Path(p).relative_to(PROJECT_ROOT)): base.sha256(p) for p in
                       (__file__, base.__file__, previous.__file__, diagnosis.__file__)}
            references = {p.name: base.sha256(p) for p in REFERENCE.glob("oof_*.csv")}
            params = {"sources": sources, "data_signature": previous.signature(samples),
                "images_sha256": old_protocol["parameters"]["train_png_sha256"], "protected": protected,
                "reference_sha256": references, "grid": {f: grid(f, args.prueba) for f in FAMILIES},
                "smoke": args.prueba, "torch": torch.__version__, "torchvision": torchvision.__version__,
                "environment": base.environment(torch.device(args.dispositivo))}
            params["environment"].pop("git_commit", None)
            fingerprint = base.config_hash(params)
            protocol = output / "protocolo.json"
            if protocol.exists() and json.loads(protocol.read_text())["fingerprint"] != fingerprint:
                raise ValueError("Protocolo, fuentes o entorno distintos: usar una salida nueva")
            if not protocol.exists():
                base.json_write(protocol, {"fingerprint": fingerprint, "parameters": params, "created_at": previous.now()})
            if args.accion != "comparar":
                ids, features = extract(samples, base.pdatos.DATOS, output, torch.device(args.dispositivo), args.prueba, progress)
                if args.accion == "ejecutar":
                    patients = samples.drop_duplicates("patient_id").set_index("patient_id").loc[ids].reset_index()
                    for fold in ([0] if args.prueba else range(5)):
                        progress({"stage": "training_heads", "fold": fold})
                        fit_fold(samples, patients, features, fold, output, args.prueba)
            if args.accion != "extraer" and not args.prueba:
                progress({"stage": "comparison"})
                report = compare(samples, output)
                print(json.dumps(report["metrics"], indent=2), flush=True)
            if previous.protected_snapshot() != protected:
                raise RuntimeError("Cambió un archivo protegido")
            progress({"status": "complete", "stage": "complete", "protected_unchanged": True, "finished_at": previous.now()})
        except BaseException as exc:
            progress({"status": "failed", "error": str(exc), "finished_at": previous.now()})
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
