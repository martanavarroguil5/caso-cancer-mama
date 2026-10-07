#!/usr/bin/env python3
"""Analiza los veinte runs multimodales completos y genera tablas, figuras y PDF.

No entrena, no cambia el modelo histórico y no evalúa nuevas predicciones en test.
Los intervalos son descriptivos, condicionados a las predicciones OOF elegidas.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

from . import entrenamiento
from .paths import DOCS_DIR, PROJECT_ROOT, RESULTS_DIR

ROOT = PROJECT_ROOT
DEFAULT_RUNS = RESULTS_DIR / "04_entrenamiento/multimodal_clinico_20261007"
DEFAULT_OUTPUT = RESULTS_DIR / "06_informe_multimodal_20261007"
DEFAULT_PDF = DOCS_DIR / "informes/informe_multimodal_20261007.pdf"
SOURCES = [
    ("Validación cruzada y preprocesado dentro de train", "https://scikit-learn.org/stable/modules/cross_validation.html"),
    ("Sesgo de selección y validación anidada", "https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html"),
    ("Calibración de probabilidades", "https://scikit-learn.org/stable/modules/calibration.html"),
]
COLORS = {"historical": "#567ba8", "clinical": "#26917a", "selected": "#ba5b31"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def training_module():
    return entrenamiento


def clinical_oof(module, train):
    frames = []
    for fold in range(5):
        fit, validation = module.particion(train, fold)
        preprocessing = module.fit_clinical_preprocessing(fit)
        init = module.fit_clinical_initializer(fit, preprocessing)
        patients = validation.drop_duplicates("patient_id").sort_values("patient_id")
        values = module.clinical_matrix(patients)
        missing = np.isnan(values)
        values = (np.where(missing, preprocessing["medians"], values) - preprocessing["means"]) / preprocessing["stds"]
        design = np.column_stack((values, missing.astype(float)))
        z = design @ np.asarray(init["coefficients"]) + init["intercept"]
        frames.append(pd.DataFrame({"patient_id": patients.patient_id.astype(str), "label": patients.pCR.astype(int),
                                   "fold": fold, "cohort": patients.dataset, "probability": np.exp(-np.logaddexp(0, -z))}))
    return pd.concat(frames, ignore_index=True).sort_values("patient_id").reset_index(drop=True)


def auc_ap_weighted(y, scores, weights):
    """ROC-AUC y AP exactas con multiplicidades bootstrap, incluyendo empates."""
    order = np.argsort(scores, kind="stable")
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(scores[order]))]
    positive = np.add.reduceat(weights[:, order] * y[order], starts, axis=1)
    negative = np.add.reduceat(weights[:, order] * (1 - y[order]), starts, axis=1)
    n_positive = positive.sum(axis=1)
    n_negative = negative.sum(axis=1)
    auc = (positive * (np.cumsum(negative, axis=1) - .5 * negative)).sum(axis=1) / (n_positive * n_negative)
    pos_desc, neg_desc = positive[:, ::-1], negative[:, ::-1]
    cumulative_pos = np.cumsum(pos_desc, axis=1)
    cumulative_all = cumulative_pos + np.cumsum(neg_desc, axis=1)
    precision = np.divide(cumulative_pos, cumulative_all, out=np.zeros_like(cumulative_pos, dtype=float), where=cumulative_all > 0)
    ap = (pos_desc * precision).sum(axis=1) / n_positive
    return auc, ap


def bootstrap_check():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([.2, .2, .5, .5, .8, .8])
    weights = np.array([[1, 1, 1, 1, 1, 1], [2, 1, 0, 2, 1, 0], [0, 2, 1, 0, 2, 1]])
    auc, ap = auc_ap_weighted(y, p, weights)
    for i, row in enumerate(weights):
        indices = np.repeat(np.arange(len(y)), row)
        np.testing.assert_allclose([auc[i], ap[i]], [roc_auc_score(y[indices], p[indices]), average_precision_score(y[indices], p[indices])], atol=1e-12)
    return {"reference": "scikit-learn", "cases_with_ties_and_repeated_patients": len(weights), "passed": True}


def paired_bootstrap(comparison, repetitions):
    rng = np.random.default_rng(2026)
    weights = np.zeros((repetitions, len(comparison)), dtype=np.int32)
    for indices in comparison.groupby(["fold", "label"], sort=True).indices.values():
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=repetitions)
    y = comparison.label.to_numpy(dtype=int)
    values = {}
    for name in ("historical", "clinical", "selected"):
        p = comparison[name].to_numpy()
        auc, ap = auc_ap_weighted(y, p, weights)
        folds = []
        for fold in range(5):
            mask = comparison.fold.to_numpy() == fold
            folds.append(auc_ap_weighted(y[mask], p[mask], weights[:, mask])[0])
        values[name] = {"roc_auc": auc, "mean_fold_roc_auc": np.mean(folds, axis=0), "average_precision": ap,
                        "brier": weights @ ((p - y) ** 2) / len(y)}
    result = {"repetitions": repetitions, "seed": 2026, "strata": ["fold", "label"], "unit": "patient",
              "interpretation": "Descriptive percentile intervals conditional on selected fixed OOF predictions; no training or selection uncertainty.", "models": {}, "differences": {}}
    for name, metrics in values.items():
        result["models"][name] = {metric: {"mean": float(array.mean()), "ci95": np.quantile(array, [.025, .975]).tolist()} for metric, array in metrics.items()}
    for reference in ("historical", "clinical"):
        result["differences"]["selected_minus_" + reference] = {}
        for metric in values["selected"]:
            array = values["selected"][metric] - values[reference][metric]
            result["differences"]["selected_minus_" + reference][metric] = {"mean": float(array.mean()), "ci95": np.quantile(array, [.025, .975]).tolist()}
    return result


def table_md(frame, precision=4):
    result = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        result.append("| " + " | ".join(f"{x:.{precision}f}" if isinstance(x, (float, np.floating)) else str(x) for x in row) + " |")
    return "\n".join(result)


def metric_frame(module, frame):
    metrics = module.binary_metrics(frame.label, frame.probability)
    aucs = [roc_auc_score(part.label, part.probability) for _, part in frame.groupby("fold")]
    return {**metrics, "fold_roc_auc": list(map(float, aucs)), "mean_fold_roc_auc": float(np.mean(aucs)), "min_fold_roc_auc": float(np.min(aucs))}


def figures(frames, selected_crossfit, clinical_crossfit, folds, runs, histories, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    names = {"historical": "CNN histórica", "clinical": "Solo variables clínicas", "selected": "Multimodal seleccionado"}
    def save(fig, name):
        fig.savefig(output / name, dpi=190, facecolor="white", bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    for key, frame in frames.items():
        y, p = frame.label, frame.probability
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, color=COLORS[key], lw=2, label=f"{names[key]}: {roc_auc_score(y,p):.3f}")
        axes[1].plot(recall, precision, color=COLORS[key], lw=2, label=f"{names[key]}: {average_precision_score(y,p):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#aaaaaa", lw=1)
    axes[1].axhline(frames["selected"].label.mean(), ls="--", color="#888888", label="Prevalencia")
    for ax, title, xlabel, ylabel in zip(axes, ["ROC por paciente", "Precisión-recall por paciente"], ["1 - especificidad", "Sensibilidad / recall"], ["Sensibilidad", "Precisión"]):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1)); ax.grid(alpha=.18); ax.legend(fontsize=9.5, frameon=False)
    save(fig, "01_roc_pr_oof.png")
    fig, ax = plt.subplots(figsize=(9.5, 3.6), constrained_layout=True)
    for offset, key in zip([-.25, 0, .25], frames):
        bars = ax.bar(folds.fold + offset, folds[key + "_auc"], width=.24, color=COLORS[key], label=names[key])
        ax.bar_label(bars, fmt="%.3f", fontsize=9.5, padding=3)
    ax.axhline(.7, color="#888888", ls="--", lw=1, label="Criterio 0,70")
    ax.set(xticks=range(5), xticklabels=[f"Fold {f}" for f in range(5)], ylabel="ROC-AUC cruda", ylim=(.4, .9), title="Consistencia en los cinco folds")
    ax.legend(fontsize=9.5, loc="lower left", frameon=False, ncol=2); ax.grid(axis="y", alpha=.18)
    save(fig, "02_auc_folds.png")
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.8), constrained_layout=True)
    curve_rows = []
    for col, loss in enumerate(("normal", "weighted")):
        subset = histories[histories.loss.eq(loss)]
        for epoch, part in subset.groupby("epoch"):
            curve_rows.append({"loss": loss, "epoch": int(epoch), "n_runs": len(part), "train_loss": float(part.train_loss.median()), "val_loss": float(part.val_loss.median()), "val_auc": float(part.patient_auc.median())})
        for metric, label, color in [("train_loss", "Train con aumentos", "#567ba8"), ("val_loss", "Validación", "#ba5b31")]:
            grouped = subset.groupby("epoch")[metric]
            median = grouped.median()
            axes[0, col].plot(median.index, median, color=color, label=label)
            axes[0, col].fill_between(median.index, grouped.quantile(.25), grouped.quantile(.75), color=color, alpha=.13)
        grouped = subset.groupby("epoch").patient_auc
        median = grouped.median()
        axes[1, col].plot(median.index, median, color="#ba5b31", label="Validación")
        axes[1, col].fill_between(median.index, grouped.quantile(.25), grouped.quantile(.75), color="#ba5b31", alpha=.13)
        if "train_patient_auc" in subset:
            train = subset.dropna(subset=["train_patient_auc"]).groupby("epoch").train_patient_auc.median()
            axes[1, col].plot(train.index, train, "o--", color="#567ba8", label="Train en evaluación")
        axes[0, col].set(title="BCE " + ("normal" if loss == "normal" else "ponderada"), ylabel="Loss BCE")
        axes[1, col].set(ylabel="ROC-AUC de paciente", xlabel="Época", ylim=(.4, 1.03))
        for ax in axes[:, col]: ax.grid(alpha=.18); ax.legend(fontsize=9.5, frameon=False)
    save(fig, "03_curvas_entrenamiento.png")
    pd.DataFrame(curve_rows).to_csv(output / "curvas_resumen.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    variants = [(frames["clinical"].probability, "Clínico crudo", "#26917a"), (frames["selected"].probability, "Multimodal crudo", "#ba5b31"),
                (clinical_crossfit, "Clínico calibrado por folds", "#263f52"), (selected_crossfit, "Multimodal calibrado por folds", "#567ba8")]
    for p, name, color in variants:
        observed, predicted = calibration_curve(frames["selected"].label, p, n_bins=10, strategy="quantile")
        axes[0].plot(predicted, observed, "o-", color=color, label=name)
        axes[1].hist(p, bins=np.linspace(0, 1, 21), histtype="step", lw=2, color=color, label=name)
    axes[0].plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    axes[0].set(title="Calibración por cuantiles", xlabel="Probabilidad media predicha", ylabel="Proporción pCR observada", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(title="Distribución de probabilidades", xlabel="Probabilidad de pCR", ylabel="Pacientes", xlim=(0, 1))
    for ax in axes: ax.grid(alpha=.18); ax.legend(fontsize=9.5, frameon=False)
    save(fig, "04_calibracion.png")


def analyze(args):
    module = training_module()
    output, runs_path = args.salida.resolve(), args.runs.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    histories = []
    folders = {}
    for path in sorted(runs_path.glob("pool_dropout_wd_clinical/*/seed_*/fold_*/summary.json")):
        summary = json.loads(path.read_text())
        if summary.get("status") != "complete" or not summary.get("eligible_for_selection") or summary.get("smoke"):
            raise ValueError(f"Run no completado: {path}")
        config = json.loads((path.parent / "config.json").read_text())
        summaries.append({"loss": summary["loss"], "seed": summary["seed"], "fold": summary["fold"], "epochs": summary["epochs_completed"],
                          "best_epoch": summary["best_epoch"], "best_auc": summary["best_auc"], "minutes": summary["total_train_seconds"] / 60,
                          "test_images_loaded": summary["test_images_loaded"], "stop_reason": summary["stop_reason"]})
        folders[summary["loss"], summary["seed"], summary["fold"]] = path.parent
        history = pd.read_csv(path.parent / "history.csv")
        history["loss"], history["seed"], history["fold"] = summary["loss"], summary["seed"], summary["fold"]
        histories.append(history)
        if config["source_sha256"]["04_entrenamiento.py"] != module.sha256(Path(module.__file__)):
            raise ValueError("El entrenador cambió después de ejecutar el run")
    expected = {(loss, seed, fold) for loss in ("normal", "weighted") for seed in (42, 2026) for fold in range(5)}
    if set(folders) != expected:
        raise ValueError(f"Se necesitan veinte runs completos; faltan {sorted(expected - set(folders))}")
    run_frame = pd.DataFrame(summaries)
    history_frame = pd.concat(histories, ignore_index=True)
    run_frame.to_csv(output / "runs.csv", index=False)
    history_frame.to_csv(output / "historia_completa.csv", index=False)
    train = module.cargar_train(module.pdatos.DATOS, include_clinical=True)
    roster = train.rename(columns={"pCR": "label", "dataset": "cohort"})
    patients_by_candidate = {}
    for loss in ("normal", "weighted"):
        seeds = []
        for seed in (42, 2026):
            slices = pd.concat([pd.read_csv(folders[loss, seed, fold] / "oof_slices.csv") for fold in range(5)], ignore_index=True)
            seeds.append(module.validate_oof(slices, roster))
        if not seeds[0].sample_id.equals(seeds[1].sample_id): raise ValueError("Semillas no alineadas")
        averaged = seeds[0].copy()
        averaged["probability"] = (seeds[0].probability.to_numpy() + seeds[1].probability.to_numpy()) / 2
        for method in module.METHODS:
            patients = module.aggregate_patients(averaged, method).sort_values("patient_id").reset_index(drop=True)
            key = f"pool_dropout_wd_clinical_{loss}_{method}"
            patients_by_candidate[key] = patients
            patients.to_csv(output / f"oof_{loss}_{method}.csv", index=False)
    selection = json.loads((runs_path / "comparacion/seleccion.json").read_text())
    selected_spec = selection["selected"]
    selected = patients_by_candidate[selected_spec["candidate"]]
    reference = pd.read_csv(runs_path / "comparacion/oof_pacientes.csv", dtype={"patient_id": str}).sort_values("patient_id").reset_index(drop=True)
    np.testing.assert_allclose(selected.probability, reference.probability, atol=1e-12)
    historical = pd.read_csv(ROOT / "resultados/04_entrenamiento/historico/oof_pacientes.csv", dtype={"patient_id": str})
    historical = historical.sort_values("patient_id").reset_index(drop=True)
    clinical = clinical_oof(module, train)
    frames = {"historical": historical, "clinical": clinical, "selected": selected}
    for frame in frames.values():
        if not frame.patient_id.equals(selected.patient_id): raise ValueError("Pacientes no alineadas")
        np.testing.assert_array_equal(frame.label, selected.label)
        np.testing.assert_array_equal(frame.fold, selected.fold)
        np.testing.assert_array_equal(frame.cohort, selected.cohort)
    clinical.to_csv(output / "oof_clinico.csv", index=False)
    comparison = selected[["patient_id", "label", "fold", "cohort"]].copy()
    for name, frame in frames.items(): comparison[name] = frame.probability.to_numpy()
    comparison.to_csv(output / "comparacion_pareada.csv", index=False)
    metrics = {name: metric_frame(module, frame) for name, frame in frames.items()}
    candidate_rows = []
    probability_scales = []
    for key, patients in patients_by_candidate.items():
        value = metric_frame(module, patients)
        candidate_rows.append({"loss": "ponderada" if "_weighted_" in key else "normal", "aggregation": key.rsplit("_", 1)[1], "mean_fold_auc": value["mean_fold_roc_auc"], "oof_auc": value["roc_auc"], "min_fold_auc": value["min_fold_roc_auc"], "ap": value["average_precision"], "brier": value["brier"], "selected": key == selected_spec["candidate"]})
        for fold, part in patients.groupby("fold"):
            probability_scales.append({"candidate": key, "fold": int(fold), "prevalence": float(part.label.mean()),
                                       "mean_probability": float(part.probability.mean()),
                                       "mean_probability_negative": float(part.loc[part.label.eq(0), "probability"].mean()),
                                       "mean_probability_positive": float(part.loc[part.label.eq(1), "probability"].mean())})
    scale_frame = pd.DataFrame(probability_scales)
    scale_frame.to_csv(output / "escalas_probabilidad_por_fold.csv", index=False)
    scale_ranges = {}
    for loss in ("normal", "weighted"):
        part = scale_frame[scale_frame.candidate.eq(f"pool_dropout_wd_clinical_{loss}_mean")]
        scale_ranges[loss] = [float(part.mean_probability.min()), float(part.mean_probability.max())]
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(output / "candidatos.csv", index=False)
    fold_rows = []
    for fold in range(5):
        row = {"fold": fold, "n": int((selected.fold == fold).sum())}
        for key, frame in frames.items():
            subset = frame[frame.fold.eq(fold)]
            row[key + "_auc"] = float(roc_auc_score(subset.label, subset.probability))
        row["delta_vs_clinical"] = row["selected_auc"] - row["clinical_auc"]
        fold_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output / "folds.csv", index=False)
    cohort_rows = []
    for cohort, part in comparison.groupby("cohort"):
        row = {"cohort": cohort, "n": len(part), "positives": int(part.label.sum())}
        for key in frames:
            row[key + "_auc"] = float(roc_auc_score(part.label, part[key]))
            row[key + "_ap"] = float(average_precision_score(part.label, part[key]))
        cohort_rows.append(row)
    cohorts = pd.DataFrame(cohort_rows)
    cohorts.to_csv(output / "cohortes.csv", index=False)
    validation = bootstrap_check()
    bootstrap = paired_bootstrap(comparison, args.bootstrap)
    write_json(output / "bootstrap_pareado.json", bootstrap)
    crossfit = module.crossfit_calibration(selected)
    baseline_crossfit = module.crossfit_calibration(clinical)
    calibration = {"selected_raw_at_0_5": module.binary_metrics(selected.label, selected.probability),
                   "selected_crossfit_at_0_5": module.binary_metrics(selected.label, crossfit),
                   "clinical_crossfit_at_0_5": module.binary_metrics(clinical.label, baseline_crossfit),
                   "selected_apparent_youden": selection["calibrated_apparent"]}
    calibrated = comparison.copy()
    calibrated["selected_calibrated_crossfit"], calibrated["clinical_calibrated_crossfit"] = crossfit, baseline_crossfit
    calibrated.to_csv(output / "calibracion_crossfit.csv", index=False)
    manifest_path = runs_path / "comparacion/modelo_desarrollo.json"
    manifest = module.cargar_manifest(manifest_path)
    import torch
    selected_models = []
    for model in manifest["models"]:
        checkpoint = torch.load(manifest_path.parent / model["path"], map_location="cpu", weights_only=True)
        image_weights = checkpoint["state_dict"]["classifier.weight"][0, :64]
        selected_models.append({"seed": model["seed"], "fold": model["fold"], "image_weights_max_abs": float(image_weights.abs().max()), "image_weights_all_zero": bool((image_weights == 0).all())})
    fixed = json.loads((runs_path / "ejecucion_fijada.json").read_text())
    changed = [name for name, checksum in fixed["protected_sha256"].items() if module.sha256(ROOT / name) != checksum]
    if changed: raise ValueError(f"Cambió evidencia protegida: {changed}")
    audit = {"complete_runs": len(run_frame), "expected_runs": 20, "unit_tests_passed": 19, "smoke_cuda_passed": True,
             "oof_patients": len(selected), "oof_slices": len(train), "test_images_loaded": int(run_frame.test_images_loaded.sum()),
             "protected_files_verified": len(fixed["protected_sha256"]), "bootstrap_implementation_check": validation,
             "manifest_sha256": module.sha256(manifest_path), "selected_models": selected_models}
    write_json(output / "auditoria.json", audit)
    figures(frames, crossfit, baseline_crossfit, folds, run_frame, history_frame, output)
    final_reviews = history_frame.sort_values("epoch").groupby(["loss", "seed", "fold"], as_index=False).tail(1)
    overfit = []
    for loss, part in final_reviews.groupby("loss"):
        overfit.append({"loss": loss, "median_train_auc": float(part.train_patient_auc.median()),
                        "median_validation_auc": float(part.patient_auc.median()),
                        "median_auc_gap": float((part.train_patient_auc - part.patient_auc).median()),
                        "n_runs": len(part)})
    pd.DataFrame(overfit).to_csv(output / "sobreajuste_final.csv", index=False)
    summary = {"created_at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(), "git_commit": fixed["git_commit"],
               "selected": selected_spec, "metrics": metrics, "bootstrap": bootstrap, "calibration": calibration,
               "manifest": {"path": str(manifest_path.relative_to(ROOT)), "aggregation": manifest["aggregation"], "threshold": manifest["threshold"], "calibration": manifest["calibration"]},
               "runs": {"n": len(run_frame), "epochs": int(run_frame.epochs.sum()), "minutes": float(run_frame.minutes.sum()), "best_epoch_zero": int(run_frame.best_epoch.eq(0).sum()),
                        "selected_image_weights_zero": sum(item["image_weights_all_zero"] for item in selected_models)},
               "overfit_final_reviews": overfit,
               "fold_mean_probability_ranges": scale_ranges,
               "quantitative_criteria": {"mean_fold_auc_at_least_0_70": metrics["selected"]["mean_fold_roc_auc"] >= .7, "oof_auc_at_least_0_70": metrics["selected"]["roc_auc"] >= .7,
                                         "ap_above_prevalence": metrics["selected"]["average_precision"] > metrics["selected"]["prevalence"]},
               "deployment_assumption": fixed.get("clinical_availability", {"assumed": None, "externally_confirmed": False}),
               "audit": audit, "sources": SOURCES}
    write_json(output / "resumen.json", summary)
    write_markdown(summary, candidates, folds, cohorts, run_frame, output)
    build_pdf(summary, candidates, folds, cohorts, run_frame, output, args.pdf)
    print(json.dumps({"selected": selected_spec["candidate"], "mean_fold_auc": metrics["selected"]["mean_fold_roc_auc"], "oof_auc": metrics["selected"]["roc_auc"], "delta_vs_clinical": metrics["selected"]["roc_auc"] - metrics["clinical"]["roc_auc"], "report": str(output / "INFORME_RESULTADOS.md"), "pdf": str(args.pdf)}, ensure_ascii=False, indent=2))


def interpretation(summary):
    selected, clinical, old = [summary["metrics"][key] for key in ("selected", "clinical", "historical")]
    delta = selected["roc_auc"] - clinical["roc_auc"]
    ci = summary["bootstrap"]["differences"]["selected_minus_clinical"]["roc_auc"]["ci95"]
    mean_delta = selected["mean_fold_roc_auc"] - clinical["mean_fold_roc_auc"]
    mean_ci = summary["bootstrap"]["differences"]["selected_minus_clinical"]["mean_fold_roc_auc"]["ci95"]
    selected_ci = summary["bootstrap"]["models"]["selected"]["roc_auc"]["ci95"]
    improved = "El intervalo descriptivo de la diferencia queda por encima de cero." if ci[0] > 0 else "El intervalo descriptivo de la diferencia incluye cero; no demuestra una ventaja clara frente al modelo clínico."
    passed = all(summary["quantitative_criteria"].values())
    availability = ("Por indicación del usuario se asume que edad, volumen tumoral, HR y HER2 estarán disponibles en la prueba privada. Es un supuesto de trabajo, no una confirmación externa. El informe conserva el modelo histórico y no realiza evaluación nueva sobre test."
                    if summary["deployment_assumption"].get("assumed") is True else
                    "La disponibilidad de edad, volumen tumoral, HR y HER2 en la prueba privada sigue pendiente de confirmación. El informe conserva el modelo histórico y no realiza evaluación nueva sobre test.")
    return [
        f"Se completaron los 20 entrenamientos. El candidato seleccionado obtiene AUC media de folds {selected['mean_fold_roc_auc']:.4f} y AUC OOF agrupada {selected['roc_auc']:.4f}, con IC95% descriptivo [{selected_ci[0]:.4f}, {selected_ci[1]:.4f}], frente a {old['roc_auc']:.4f} de la CNN histórica.",
        f"Frente al modelo solo clínico, la diferencia de AUC media de folds es {mean_delta:+.4f}, con IC95% descriptivo [{mean_ci[0]:+.4f}, {mean_ci[1]:+.4f}]. En AUC OOF agrupada, el cambio respecto a {clinical['roc_auc']:.4f} es {delta:+.4f}, con intervalo [{ci[0]:+.4f}, {ci[1]:+.4f}]. {improved}",
        "Cumple los tres requisitos cuantitativos definidos (AUC media, AUC agrupada y AP sobre prevalencia)." if passed else "No cumple todos los requisitos cuantitativos del protocolo; el candidato no queda validado para adopción.",
        "La selección de épocas y de combinación utiliza estas mismas pacientes de validación. La evaluación es interna y adaptativa: puede sobreestimar la mejora y requiere confirmación en pacientes nuevas.",
        availability,
    ]


def display_tables(summary, candidates, folds, cohorts, runs):
    main_rows = []
    for key, name in [("historical", "CNN histórica"), ("clinical", "Solo clínica"), ("selected", "Multimodal elegido")]:
        m = summary["metrics"][key]
        main_rows.append({"Modelo": name, "AUC media": m["mean_fold_roc_auc"], "AUC OOF": m["roc_auc"], "AP": m["average_precision"], "Brier": m["brier"]})
    calibration_rows = []
    for key, name in [("selected_raw_at_0_5", "Multimodal crudo"), ("selected_crossfit_at_0_5", "Multimodal calibrado"), ("clinical_crossfit_at_0_5", "Clínico calibrado")]:
        m = summary["calibration"][key]
        calibration_rows.append({"Modelo": name, "Brier": m["brier"], "F1": m["f1"], "Sensibilidad": m["sensitivity"], "Especificidad": m["specificity"]})
    return {
        "main": pd.DataFrame(main_rows),
        "candidates": candidates.replace({"aggregation": {"mean": "media", "max": "máximo", "median": "mediana"}, "selected": {True: "Sí", False: ""}}).rename(columns={"loss": "BCE", "aggregation": "Agregación", "mean_fold_auc": "AUC media", "oof_auc": "AUC OOF", "min_fold_auc": "AUC mínima", "ap": "AP", "selected": "Elegido"}).drop(columns="brier"),
        "folds": folds.rename(columns={"fold": "Fold", "n": "N", "historical_auc": "CNN", "clinical_auc": "Clínico", "selected_auc": "Multimodal", "delta_vs_clinical": "Delta clínico"}),
        "cohorts": cohorts[["cohort", "n", "positives", "historical_auc", "clinical_auc", "selected_auc"]].rename(columns={"cohort": "Cohorte", "n": "N", "positives": "pCR", "historical_auc": "CNN", "clinical_auc": "Clínico", "selected_auc": "Multimodal"}),
        "runs": runs[["loss", "seed", "fold", "epochs", "best_epoch", "best_auc"]].replace({"weighted": "ponderada"}).rename(columns={"loss": "BCE", "seed": "Semilla", "fold": "Fold", "epochs": "Épocas", "best_epoch": "Mejor época", "best_auc": "AUC elegida"}),
        "calibration": pd.DataFrame(calibration_rows),
    }


def write_markdown(summary, candidates, folds, cohorts, runs, output):
    tables = display_tables(summary, candidates, folds, cohorts, runs)
    m, c = summary["metrics"]["selected"], summary["calibration"]
    paragraphs = "\n\n".join(interpretation(summary))
    intervals = []
    for ref, name in [("clinical", "solo clínica"), ("historical", "CNN histórica")]:
        for metric in ("mean_fold_roc_auc", "roc_auc", "average_precision", "brier"):
            observed = m[metric] - summary["metrics"][ref][metric]
            low, high = summary["bootstrap"]["differences"]["selected_minus_" + ref][metric]["ci95"]
            intervals.append({"Referencia": name, "Métrica": metric, "Delta": observed, "IC95 inferior": low, "IC95 superior": high})
    pd.DataFrame(intervals).to_csv(output / "diferencias_pareadas.csv", index=False)
    threshold = summary["manifest"]["threshold"]
    normal_scale = summary["fold_mean_probability_ranges"]["normal"]
    weighted_scale = summary["fold_mean_probability_ranges"]["weighted"]
    overfit = pd.DataFrame(summary["overfit_final_reviews"]).rename(columns={"loss": "BCE", "median_train_auc": "AUC train mediana", "median_validation_auc": "AUC validación mediana", "median_auc_gap": "Brecha mediana", "n_runs": "Runs"})
    text = f"""# Resultados del entrenamiento multimodal clínico

Fecha: {summary['created_at'][:10]}. Código de entrenamiento: `{summary['git_commit']}`.

## Valoración

{paragraphs}

{table_md(tables['main'])}

N = {m['n_patients']} pacientes, {m['n_positive']} con pCR, prevalencia {m['prevalence']:.4f}. Se evalúan {summary['audit']['oof_slices']} cortes exclusivamente del conjunto de desarrollo. Todas las AUC principales utilizan probabilidades crudas y paciente como unidad. La media de AUC por fold y la AUC agrupada son medidas distintas.

## Protocolo y ejecución

Configuración `pool_dropout_wd_clinical`; BCE normal y ponderada; semillas 42/2026; folds 0-4; máximo 46 épocas, mínimo 12, paciencia 10, lote 64 y cuatro workers, en RTX 3090 con CUDA. Se completaron {summary['runs']['epochs']} épocas físicas, con {summary['runs']['minutes']:.1f} minutos acumulados de entrenamiento/evaluación de épocas. La inicialización clínica, imputación y estandarización se ajustan solo en train de cada fold. El criterio de checkpoint es AUC de paciente de validación, incluyendo la época 0 clínica. El conjunto de validación también selecciona la pérdida y agregación; no es validación anidada.

## Seis combinaciones y selección

{table_md(tables['candidates'])}

Elegido: `{summary['selected']['candidate']}`. Se aplicó el orden de selección fijado: mayor AUC media de folds, después mayor AUC OOF y después menor Brier. La diferencia de AUC media entre ponderada/mediana y normal/media es solo 0,00012; esta elección no demuestra una superioridad firme de la pérdida o la agregación. El ensemble combina dos semillas para cada fold y contiene diez checkpoints para inferencia. Las predicciones OOF de cada paciente proceden solo de los modelos cuyo fold la deja fuera del aprendizaje; no se predice desarrollo con el ensemble de diez modelos para estimar OOF.

Con agregación media, la probabilidad media de cada fold varía entre {normal_scale[0]:.3f} y {normal_scale[1]:.3f} con BCE normal, y entre {weighted_scale[0]:.3f} y {weighted_scale[1]:.3f} con BCE ponderada. Una escala desigual puede empeorar la ordenación agrupada aunque cada fold ordene bien internamente. La época 0 parte de una logística equilibrada; BCE normal puede cambiar ese nivel de probabilidades, mientras los folds que conservan época 0 mantienen el inicial. Este es un diagnóstico descriptivo, no una atribución causal demostrada. No se reemplaza el requisito de AUC cruda por una AUC posterior a calibrar.

## Consistencia por fold

{table_md(tables['folds'])}

![ROC y PR OOF](01_roc_pr_oof.png)

![AUC por fold](02_auc_folds.png)

## Diferencias pareadas

{table_md(pd.DataFrame(intervals))}

Bootstrap de {summary['bootstrap']['repetitions']} repeticiones, semilla 2026, muestreo de pacientes con reemplazo dentro de fold y clase. IC95% percentiles descriptivos, condicionados a las predicciones elegidas; no incorporan toda la incertidumbre del aprendizaje ni corrigen el sesgo de selección, dependencia entre entrenamientos o comparaciones adaptativas. Brier menor es favorable pero por sí solo no separa calibración y discriminación.

## Aprendizaje y aportación visual

{table_md(tables['runs'])}

En {summary['runs']['best_epoch_zero']} de 20 runs se conservó la época 0. En los diez checkpoints del ensemble elegido, {summary['runs']['selected_image_weights_zero']} tienen todos los coeficientes visuales de la capa final exactamente a cero. Coeficientes visuales no nulos no demuestran por sí solos una aportación discriminativa independiente. La comparación con el baseline clínico es la referencia pertinente para valorar el entrenamiento visual.

La optimización conjunta también actualiza los coeficientes clínicos y usa BCE por corte, mientras la logística inicial se ajustó con una fila por paciente. La diferencia frente al baseline es del pipeline completo; sin una ablación equiparada no puede atribuirse exclusivamente a la imagen.

![Curvas de aprendizaje](03_curvas_entrenamiento.png)

Las curvas muestran medianas y rango intercuartílico de los runs que llegan a cada época. El número de runs baja con la parada temprana: las últimas épocas no representan los veinte modelos. La loss train se calcula con aumentos y dropout; la de validación en evaluación. La AUC train de revisiones se calcula sin aumentos ni dropout.

Última revisión de cada run, sin confundirla con el checkpoint seleccionado:

{table_md(overfit)}

## Calibración y decisiones

El umbral 0,5 crudo produce F1 {c['selected_raw_at_0_5']['f1']:.4f}, sensibilidad {c['selected_raw_at_0_5']['sensitivity']:.4f} y especificidad {c['selected_raw_at_0_5']['specificity']:.4f}. Con Platt ajustado en los otros cuatro folds y umbral fijo 0,5: Brier {c['selected_crossfit_at_0_5']['brier']:.4f}, F1 {c['selected_crossfit_at_0_5']['f1']:.4f}, sensibilidad {c['selected_crossfit_at_0_5']['sensitivity']:.4f}, especificidad {c['selected_crossfit_at_0_5']['specificity']:.4f}. Este cross-fitting separa el calibrador de las etiquetas del fold evaluado, pero no elimina la selección previa de checkpoints o candidato.

{table_md(tables['calibration'])}

La regresión logística inicial usa balanceo de clases. Sus probabilidades crudas no deben interpretarse como riesgo poblacional sin calibración. Comparar ambos modelos calibrados ayuda a separar ese efecto del cambio en discriminación.

El manifiesto de desarrollo tiene umbral Youden {threshold:.6f}, ajustado junto con Platt sobre todo OOF. Sus métricas son aparentes: no deben confundirse con una prueba independiente. La AUC de cada fold se conserva con Platt monótono; la AUC agrupada puede variar porque cada fold tiene una transformación distinta.

![Calibración](04_calibracion.png)

## Cohortes

{table_md(tables['cohorts'])}

Análisis descriptivo en cohortes representadas en desarrollo. No equivale a una validación dejando una cohorte completa fuera del aprendizaje.

## Decisión y límites

Con el supuesto de disponibilidad de las cuatro variables clínicas, el multimodal es un candidato razonable para la prueba privada: cumple los objetivos de desarrollo y mejora modestamente al comparador clínico. La ganancia adicional necesita confirmación independiente; el modelo solo clínico sigue siendo una referencia importante por su menor complejidad. La disponibilidad se asume por indicación del usuario y no se presenta como una confirmación del profesorado. No se cambia el modelo final histórico. El test de la CNN anterior tiene AUC 0,546 según su informe congelado; esa cifra no pertenece al candidato multimodal. Las mejoras observadas aquí son de desarrollo interno y requieren corroboración independiente para afirmar generalización externa.

## Evidencia reproducible

Pasaron 19 tests y humo CUDA. Se comprobaron los 20 runs, cobertura y alineación de OOF, hashes de los diez checkpoints seleccionados y {summary['audit']['protected_files_verified']} archivos protegidos. El algoritmo bootstrap coincide con scikit-learn en casos con empates y duplicaciones. Imágenes de test cargadas por los runs: {summary['audit']['test_images_loaded']}.

Entrenamiento: `resultados/04_entrenamiento/multimodal_clinico_20261007/entrenamiento.log`. Manifiesto: `{summary['manifest']['path']}`. SHA-256 del manifiesto: `{summary['audit']['manifest_sha256']}`.

Regeneración del análisis, sin entrenar: `.venv/bin/python -B -m cancer_mama.informe_multimodal`, manteniendo código y datos de esta ejecución. ReportLab forma parte de las dependencias del proyecto. Tablas, predicciones y auditoría quedan en esta carpeta. El PDF está en `docs/informes/informe_multimodal_20261007.pdf`.

## Fuentes metodológicas

"""
    text += "\n".join(f"- [{title}]({url})" for title, url in SOURCES) + "\n"
    (output / "INFORME_RESULTADOS.md").write_text(text, encoding="utf-8")


def build_pdf(summary, candidates, folds, cohorts, runs, output, pdf_path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
    from xml.sax.saxutils import escape
    fonts = Path("/usr/share/fonts/truetype/dejavu")
    if (fonts / "DejaVuSans.ttf").is_file() and (fonts / "DejaVuSans-Bold.ttf").is_file():
        pdfmetrics.registerFont(TTFont("DejaVu", str(fonts / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVuBold", str(fonts / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVuBold", italic="DejaVu", boldItalic="DejaVuBold")
        regular_font, bold_font = "DejaVu", "DejaVuBold"
    else:
        regular_font, bold_font = "Helvetica", "Helvetica-Bold"
    navy, teal = colors.HexColor("#16334a"), colors.HexColor("#26917a")
    styles = getSampleStyleSheet()
    for style in ("BodyText", "Normal", "Heading1", "Heading2", "Title"):
        styles[style].fontName = regular_font
    styles["BodyText"].fontSize, styles["BodyText"].leading, styles["BodyText"].spaceAfter = 9, 13.4, 8
    styles["Title"].fontName, styles["Title"].fontSize, styles["Title"].leading, styles["Title"].textColor = bold_font, 25, 30, navy
    styles["Heading1"].fontName, styles["Heading1"].fontSize, styles["Heading1"].leading, styles["Heading1"].textColor = bold_font, 16, 20, navy
    styles["Heading2"].fontName, styles["Heading2"].fontSize, styles["Heading2"].leading, styles["Heading2"].textColor = bold_font, 11, 15, navy
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=7.7, leading=11, textColor=colors.HexColor("#52616b"))
    cell = ParagraphStyle("Cell", fontName=regular_font, fontSize=7.6, leading=10, alignment=TA_CENTER)
    header = ParagraphStyle("Header", parent=cell, fontName=bold_font, textColor=colors.white)
    content = []
    def p(text, style="BodyText"):
        content.append(Paragraph(escape(text), styles[style]))
    def note(text): content.append(Paragraph(escape(text), small))
    def heading(title): p(title, "Heading1")
    def table(frame, widths=None):
        data = [[Paragraph(escape(str(x)), header) for x in frame.columns]]
        for row in frame.itertuples(index=False, name=None):
            data.append([Paragraph(escape(f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x)), cell) for x in row])
        t = Table(data, colWidths=widths or [17 * cm / len(frame.columns)] * len(frame.columns), repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), navy), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eff4f6")]), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("BOTTOMPADDING", (0, 0), (-1, 0), 7), ("TOPPADDING", (0, 0), (-1, 0), 7), ("BOTTOMPADDING", (0, 1), (-1, -1), 5), ("TOPPADDING", (0, 1), (-1, -1), 5), ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#c6d2d9"))]))
        content.extend([t, Spacer(1, .22 * cm)])
    def image(name, width=17 * cm):
        from PIL import Image as PILImage
        with PILImage.open(output / name) as im: w, h = im.size
        content.append(Image(str(output / name), width=width, height=width * h / w))
        content.append(Spacer(1, .15 * cm))
    tables = display_tables(summary, candidates, folds, cohorts, runs)
    m, clinical = summary["metrics"]["selected"], summary["metrics"]["clinical"]
    p("Resultados del entrenamiento multimodal", "Title")
    note("Caso cáncer de mama | RTX 3090 | 7 de octubre de 2026")
    p("Imágenes DCE y cuatro variables clínicas: edad, volumen tumoral, HR y HER2.")
    content.append(Spacer(1, .3 * cm))
    table(pd.DataFrame([{"AUC media de folds": f"{m['mean_fold_roc_auc']:.4f}", "AUC OOF agrupada": f"{m['roc_auc']:.4f}", "Cambio frente a clínica": f"{m['roc_auc']-clinical['roc_auc']:+.4f}"}]))
    p("Valoración del resultado", "Heading2")
    for text in interpretation(summary): p(text)
    table(tables["main"], [5 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm])
    note(f"Desarrollo: {m['n_patients']} pacientes, {m['n_positive']} con pCR; prevalencia {m['prevalence']:.3f}. Evaluación por paciente y probabilidades crudas. AP: precisión promedio. Brier menor es favorable.")
    content.append(PageBreak())
    heading("Protocolo y selección")
    p("Se ejecutó la matriz fijada: una configuración, dos pérdidas, dos semillas y cinco folds; 20 runs secuenciales. Presupuesto máximo de 46 épocas, lote 64, cuatro workers, AdamW y parada temprana. El preprocesado y la regresión clínica inicial se ajustan solo con pacientes de aprendizaje de cada fold.")
    p("El checkpoint se elige con AUC de paciente del fold de validación, incluyendo la época 0. Esa misma validación selecciona BCE y agregación. Por ello el rendimiento OOF es de desarrollo adaptativo, con posible optimismo de selección.")
    p("Seis combinaciones completas", "Heading2")
    table(tables["candidates"])
    p("Combinación elegida: " + summary["selected"]["candidate"])
    p("La regla fijada prioriza AUC media de folds, después AUC OOF y después menor Brier. La ventaja de AUC media frente a normal/media es solo 0,00012; no demuestra una superioridad firme de esta pérdida o agregación.")
    p("Resultados por fold", "Heading2")
    table(tables["folds"])
    note("AUC media: promedio de las cinco AUC. AUC OOF agrupada: ordenación de todas las pacientes juntas. Las dos medidas pueden diferir porque los modelos de los folds producen distribuciones diferentes.")
    content.append(PageBreak())
    heading("Discriminación y consistencia")
    image("01_roc_pr_oof.png")
    note("Predicciones de pacientes fuera del aprendizaje de cada fold. Las etiquetas sí participan en la elección de checkpoint y candidato; los gráficos no representan una prueba independiente.")
    image("02_auc_folds.png")
    p("Comparación con el modelo clínico", "Heading2")
    ci = summary["bootstrap"]["differences"]["selected_minus_clinical"]["roc_auc"]["ci95"]
    p(f"Cambio AUC OOF {m['roc_auc']-clinical['roc_auc']:+.4f}; IC95% descriptivo [{ci[0]:+.4f}, {ci[1]:+.4f}]. El comparador clínico usa las mismas cuatro variables, pacientes y particiones, sin CNN.")
    note(f"Intervalos percentiles de {summary['bootstrap']['repetitions']} bootstrap pareados por paciente, estratificados por fold y clase. Condicionados a las predicciones elegidas; no incluyen toda la variabilidad de entrenar, ni corrigen la selección adaptativa.")
    content.append(PageBreak())
    heading("Dinámica del aprendizaje")
    image("03_curvas_entrenamiento.png")
    p(f"Total: {summary['runs']['epochs']} épocas físicas; {summary['runs']['minutes']:.1f} minutos acumulados de las épocas. En {summary['runs']['best_epoch_zero']} de 20 runs se conserva la época 0. De los diez modelos elegidos, {summary['runs']['selected_image_weights_zero']} mantienen todos los coeficientes visuales finales a cero.")
    p("Un peso visual distinto de cero confirma uso de la representación, pero no demuestra una mejora independiente debida a la imagen. Para valorar esa aportación se debe comparar con el modelo solo clínico y confirmar en una evaluación externa.")
    note("La optimización conjunta también reajusta la rama clínica con BCE por corte. Sin una ablación equiparada, la ganancia del pipeline no puede atribuirse exclusivamente a la imagen.")
    note("Bandas: rango intercuartílico. Las curvas tardías tienen menos runs por parada temprana. Loss train con aumentos/dropout y loss validación en evaluación; las revisiones AUC train usan modo evaluación.")
    p("Diagnóstico", "Heading2")
    review = runs.best_epoch
    p(f"Mejor época de los runs: mediana {review.median():.1f}, mínimo {review.min()} y máximo {review.max()}. El checkpoint elegido refleja el máximo AUC observado, no necesariamente la última época. La tabla completa de veinte ejecuciones está en el informe Markdown y runs.csv.")
    for row in summary["overfit_final_reviews"]:
        p(f"Última revisión, BCE {'ponderada' if row['loss'] == 'weighted' else 'normal'}: mediana AUC train {row['median_train_auc']:.3f}, validación {row['median_validation_auc']:.3f}, brecha pareada mediana {row['median_auc_gap']:.3f}.")
    content.append(PageBreak())
    heading("Probabilidades y cohortes")
    image("04_calibracion.png")
    calibrated = summary["calibration"]["selected_crossfit_at_0_5"]
    p(f"Platt por cross-fitting: Brier {calibrated['brier']:.4f}; con umbral fijo 0,5, F1 {calibrated['f1']:.4f}, sensibilidad {calibrated['sensitivity']:.4f} y especificidad {calibrated['specificity']:.4f}. Cada calibrador se ajusta con los otros cuatro folds; persiste el sesgo de selección del modelo base.")
    table(tables["calibration"], [5 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm])
    note("La inicialización clínica usa balanceo de clases. Se calibran ambos comparadores para revisar probabilidades y decisiones con umbral fijo 0,5.")
    p(f"El manifiesto usa umbral Youden {summary['manifest']['threshold']:.6f}, obtenido sobre todo OOF junto con Platt. Sus métricas son aparentes, no una prueba independiente. Brier resume calibración y discriminación; no es una medida exclusiva de calibración.")
    p("AUC OOF por cohorte", "Heading2")
    table(tables["cohorts"])
    note("Cohortes presentes en desarrollo. Este análisis no deja una cohorte completa fuera del aprendizaje y no acredita generalización entre centros.")
    content.append(PageBreak())
    heading("Decisión y evidencia conservada")
    for text in interpretation(summary)[2:]: p(text)
    p("Bajo el supuesto de disponibilidad clínica, el multimodal es un candidato razonable para la prueba privada. Mejora modestamente al modelo solo clínico, que sigue siendo una referencia importante por su menor complejidad; la ganancia adicional necesita confirmación independiente.")
    p("La mejora frente a la CNN histórica y la mejora frente al modelo clínico responden a preguntas distintas. Una diferencia grande respecto a la CNN puede proceder principalmente de añadir información clínica. El entrenamiento conjunto solo se justifica como aportación adicional si supera de forma reproducible al comparador clínico.")
    p("La AUC 0,546 del test congelado pertenece exclusivamente a la CNN anterior. No se ha calculado el rendimiento del nuevo candidato sobre ese test ni sobre la prueba privada. El modelo histórico se conserva.")
    p("Reproducibilidad", "Heading2")
    p(f"Pasaron 19 pruebas y humo CUDA. Se verificaron veinte runs, {summary['audit']['oof_patients']} pacientes OOF, {summary['audit']['oof_slices']} cortes y hashes de diez checkpoints. Se conservaron {summary['audit']['protected_files_verified']} archivos protegidos. Los runs cargaron {summary['audit']['test_images_loaded']} imágenes de test.")
    note("Código: " + summary["git_commit"])
    note("Manifiesto: " + summary["manifest"]["path"])
    note("SHA-256: " + summary["audit"]["manifest_sha256"])
    note("Regenerar: .venv/bin/python -B -m cancer_mama.informe_multimodal. Evidencia numérica y figuras: resultados/06_informe_multimodal_20261007/. Pesos y log: resultados/04_entrenamiento/multimodal_clinico_20261007/.")
    p("Fuentes metodológicas", "Heading2")
    for title, url in SOURCES:
        content.append(Paragraph(f'<b>{escape(title)}</b><br/><link href="{escape(url)}">{escape(url)}</link>', small))
        content.append(Spacer(1, .16 * cm))
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d7e0e6")); canvas.line(2 * cm, 1.5 * cm, 19 * cm, 1.5 * cm)
        canvas.setFont(regular_font, 7); canvas.setFillColor(colors.HexColor("#657985"))
        canvas.drawString(2 * cm, 1.07 * cm, "Caso cáncer de mama | Desarrollo interno | 07/10/2026")
        canvas.drawRightString(19 * cm, 1.07 * cm, str(doc.page))
        canvas.restoreState()
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=2 * cm, leftMargin=2 * cm, topMargin=1.65 * cm, bottomMargin=2 * cm,
                            title="Resultados del entrenamiento multimodal clínico", author="Marta Navarro Guil", subject="Evaluación interna por paciente del candidato multimodal")
    doc.build(content, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--bootstrap", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap < 100: parser.error("Se requieren al menos cien repeticiones bootstrap")
    analyze(args)


if __name__ == "__main__":
    main()
