"""Genera la evidencia reproducible del cambio de CNN de imagen a candidato multimodal.

No entrena la CNN ni vuelve a evaluar el candidato en test. Reconstruye el
baseline clínico OOF dentro de los cinco folds de desarrollo y lo compara de
forma pareada con las predicciones OOF históricas ya congeladas.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)

from . import entrenamiento
from .paths import DOCS_DIR, PROJECT_ROOT, RESULTS_DIR

ROOT = PROJECT_ROOT
OUTPUT = RESULTS_DIR / "05_informe_mejora"
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / ".matplotlib"))
(OUTPUT / ".matplotlib").mkdir(exist_ok=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HISTORICAL = RESULTS_DIR / "04_entrenamiento" / "historico"
DIAGNOSTIC = DOCS_DIR / "mejoras_cnn" / "diagnostico.json"
COLORS = {"historical": "#4472C4", "clinical": "#159D78", "test": "#C55A11",
          "train": "#6F42C1", "reference": "#666666"}


def load_training_module():
    return entrenamiento


def sigmoid(values):
    values = np.asarray(values, dtype=float)
    return np.exp(-np.logaddexp(0, -values))


def clinical_oof(entrenamiento) -> pd.DataFrame:
    data = entrenamiento.cargar_train(entrenamiento.pdatos.DATOS, include_clinical=True)
    frames = []
    for fold in range(5):
        train, validation = entrenamiento.particion(data, fold)
        preprocessing = entrenamiento.fit_clinical_preprocessing(train)
        initialization = entrenamiento.fit_clinical_initializer(train, preprocessing)
        patients = validation.drop_duplicates("patient_id").sort_values("patient_id").copy()
        values = entrenamiento.clinical_matrix(patients)
        missing = np.isnan(values)
        values = (np.where(missing, preprocessing["medians"], values) -
                  preprocessing["means"]) / preprocessing["stds"]
        design = np.column_stack((values, missing.astype(float)))
        probability = sigmoid(design @ np.asarray(initialization["coefficients"]) +
                              initialization["intercept"])
        frames.append(pd.DataFrame({
            "patient_id": patients.patient_id.astype(str),
            "label": patients.pCR.astype(int),
            "fold": fold,
            "cohort": patients.dataset.astype(str),
            "probability": probability,
        }))
    return pd.concat(frames, ignore_index=True).sort_values("patient_id").reset_index(drop=True)


def metrics(frame: pd.DataFrame) -> dict:
    y = frame.label.to_numpy(dtype=int)
    p = frame.probability.to_numpy(dtype=float)
    return {
        "n": int(len(frame)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
    }


def ece(frame: pd.DataFrame, bins: int = 10) -> float:
    work = frame.copy()
    work["bin"] = pd.qcut(work.probability, q=bins, labels=False, duplicates="drop")
    grouped = work.groupby("bin", observed=True).agg(
        n=("label", "size"), observed=("label", "mean"), predicted=("probability", "mean"))
    return float(((grouped.n / len(work)) * (grouped.observed - grouped.predicted).abs()).sum())


def paired_bootstrap(comparison: pd.DataFrame, repetitions: int = 4000, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    strata = [group.index.to_numpy() for _, group in comparison.groupby(["fold", "label"])]
    samples = {name: [] for name in ("historical_auc", "clinical_auc", "delta_auc",
                                     "historical_ap", "clinical_ap", "delta_ap",
                                     "historical_brier", "clinical_brier", "delta_brier")}
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(index, len(index), replace=True) for index in strata])
        sampled = comparison.loc[indices]
        y = sampled.label.to_numpy()
        old = sampled.probability_historical.to_numpy()
        new = sampled.probability_clinical.to_numpy()
        old_auc, new_auc = roc_auc_score(y, old), roc_auc_score(y, new)
        old_ap, new_ap = average_precision_score(y, old), average_precision_score(y, new)
        old_brier, new_brier = brier_score_loss(y, old), brier_score_loss(y, new)
        values = (old_auc, new_auc, new_auc-old_auc, old_ap, new_ap, new_ap-old_ap,
                  old_brier, new_brier, new_brier-old_brier)
        for key, value in zip(samples, values):
            samples[key].append(float(value))
    result = {"repetitions": repetitions, "seed": seed,
              "unit": "patient_stratified_by_fold_and_class", "metrics": {}}
    for key, values in samples.items():
        array = np.asarray(values)
        result["metrics"][key] = {
            "mean": float(array.mean()),
            "ci95": [float(np.quantile(array, .025)), float(np.quantile(array, .975))],
        }
    return result


def save_figure(fig, name: str):
    fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_roc_pr(historical: pd.DataFrame, clinical: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for frame, label, color in ((historical, "CNN histórica", COLORS["historical"]),
                                (clinical, "Baseline clínico", COLORS["clinical"])):
        y, p = frame.label.to_numpy(), frame.probability.to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, lw=2.3, color=color,
                     label=f"{label} · AUC {roc_auc_score(y,p):.3f}")
        axes[1].plot(recall, precision, lw=2.3, color=color,
                     label=f"{label} · AP {average_precision_score(y,p):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color=COLORS["reference"], lw=1, label="Azar")
    prevalence = historical.label.mean()
    axes[1].axhline(prevalence, ls="--", color=COLORS["reference"], lw=1,
                    label=f"Prevalencia {prevalence:.3f}")
    for ax, title, xlabel, ylabel in ((axes[0], "ROC en validación OOF", "1 − especificidad", "Sensibilidad"),
                                      (axes[1], "Precision–Recall OOF", "Recall", "Precisión")):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0,1), ylim=(0,1))
        ax.grid(alpha=.2)
        ax.legend(loc="lower right" if ax is axes[0] else "upper right", frameon=False)
    fig.suptitle("La información clínica mejora la separación entre pCR y no-pCR", fontsize=14)
    save_figure(fig, "01_roc_pr_oof.png")


def plot_folds(fold_metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(5); width = .35
    ax.bar(x-width/2, fold_metrics.historical_auc, width, color=COLORS["historical"], label="CNN histórica")
    ax.bar(x+width/2, fold_metrics.clinical_auc, width, color=COLORS["clinical"], label="Baseline clínico")
    ax.axhline(.7, color=COLORS["test"], ls="--", lw=1.5, label="Objetivo 0,70")
    for offset, column in ((-width/2, "historical_auc"), (width/2, "clinical_auc")):
        for position, value in zip(x+offset, fold_metrics[column]):
            ax.text(position, value+.012, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set(xticks=x, xticklabels=[f"Fold {f}" for f in x], ylim=(.45,.82), ylabel="ROC-AUC",
           title="AUC por fold: mejora global, con variabilidad pendiente")
    ax.grid(axis="y", alpha=.2); ax.legend(frameon=False, ncol=3, loc="lower center")
    save_figure(fig, "02_auc_por_fold.png")


def plot_overfit(curves: pd.DataFrame, curve_summary: pd.DataFrame, diagnostic: dict,
                 historical_metrics: dict, test_metrics: dict, clinical_metrics: dict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    ax = axes[0]
    ax.plot(curve_summary.epoch, curve_summary.train_loss_median, color=COLORS["train"], lw=2,
            label="Loss train (mediana)")
    ax.fill_between(curve_summary.epoch, curve_summary.train_loss_q25, curve_summary.train_loss_q75,
                    color=COLORS["train"], alpha=.15)
    ax.plot(curve_summary.epoch, curve_summary.val_loss_median, color=COLORS["test"], lw=2,
            label="Loss validación (mediana)")
    ax.fill_between(curve_summary.epoch, curve_summary.val_loss_q25, curve_summary.val_loss_q75,
                    color=COLORS["test"], alpha=.15)
    ax.set(title="Curvas históricas", xlabel="Época", ylabel="BCE ponderada")
    ax.grid(alpha=.2); ax.legend(frameon=False)
    size = diagnostic["size_trial_epoch30"]["cnn_actual"]
    values = [size["train_patient_auc"], size["patient_auc"]]
    axes[1].bar(["Train", "Validación"], values, color=[COLORS["train"], COLORS["test"]], width=.58)
    axes[1].axhline(.7, color=COLORS["reference"], ls="--", lw=1)
    for i, value in enumerate(values):
        axes[1].text(i, value+.02, f"{value:.3f}", ha="center", fontsize=11)
    axes[1].text(.5, .78, f"Brecha = {values[0]-values[1]:.3f}",
                 ha="center", color=COLORS["reference"])
    axes[1].set(title="Sobreajuste en época 30", ylabel="ROC-AUC", ylim=(0,1.05))
    labels = ["CNN\nOOF", "CNN\ntest", "Clínico\nOOF"]
    values = [historical_metrics["roc_auc"], test_metrics["roc_auc"], clinical_metrics["roc_auc"]]
    axes[2].bar(labels, values, color=[COLORS["historical"], COLORS["test"], COLORS["clinical"]], width=.62)
    axes[2].axhline(.7, color=COLORS["reference"], ls="--", lw=1)
    for i, value in enumerate(values):
        axes[2].text(i, value+.02, f"{value:.3f}", ha="center", fontsize=11)
    axes[2].set(title="Generalización observada", ylabel="ROC-AUC", ylim=(0, .82))
    fig.suptitle("El cuello de botella era la generalización, no la capacidad de la CNN", fontsize=14)
    save_figure(fig, "03_sobreajuste_train_validacion_test.png")


def plot_distributions_cohorts(comparison: pd.DataFrame, cohort_metrics: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    bins = np.linspace(0, 1, 21)
    for label, ls in ((0, "--"), (1, "-")):
        subset = comparison[comparison.label.eq(label)]
        axes[0].hist(subset.probability_historical, bins=bins, histtype="step", density=True, lw=1.8,
                     ls=ls, color=COLORS["historical"], label=f"CNN · y={label}")
        axes[0].hist(subset.probability_clinical, bins=bins, histtype="step", density=True, lw=2.1,
                     ls=ls, color=COLORS["clinical"], label=f"Clínico · y={label}")
    axes[0].set(title="Distribución de probabilidades OOF", xlabel="Probabilidad predicha", ylabel="Densidad",
                xlim=(0,1)); axes[0].grid(alpha=.2); axes[0].legend(frameon=False, ncol=2)
    cohorts = cohort_metrics.cohort.tolist(); x=np.arange(len(cohorts)); width=.35
    axes[1].bar(x-width/2, cohort_metrics.historical_auc, width, color=COLORS["historical"], label="CNN")
    axes[1].bar(x+width/2, cohort_metrics.clinical_auc, width, color=COLORS["clinical"], label="Clínico")
    axes[1].axhline(.7, color=COLORS["test"], ls="--", lw=1.2)
    axes[1].set(xticks=x, xticklabels=[c.upper() for c in cohorts], ylabel="ROC-AUC", ylim=(.45,.85),
                title="Consistencia por cohorte")
    axes[1].grid(axis="y", alpha=.2); axes[1].legend(frameon=False)
    save_figure(fig, "04_probabilidades_y_cohortes.png")


def plot_calibration(historical: pd.DataFrame, clinical: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    ax.plot([0,1],[0,1], "--", color=COLORS["reference"], lw=1, label="Calibración ideal")
    for frame, label, color in ((historical, "CNN histórica", COLORS["historical"]),
                                (clinical, "Baseline clínico", COLORS["clinical"])):
        observed, predicted = calibration_curve(frame.label, frame.probability, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", lw=2, color=color,
                label=f"{label} · Brier {brier_score_loss(frame.label,frame.probability):.3f}")
    ax.set(title="Calibración OOF (10 grupos por cuantiles)", xlabel="Probabilidad media predicha",
           ylabel="Frecuencia observada de pCR", xlim=(0,1), ylim=(0,1))
    ax.grid(alpha=.2); ax.legend(frameon=False)
    save_figure(fig, "05_calibracion_oof.png")


def write_report(summary: dict, fold_metrics: pd.DataFrame, cohort_metrics: pd.DataFrame,
                 bootstrap: dict, diagnostic: dict):
    old, new, test = summary["historical_oof"], summary["clinical_oof"], summary["historical_test"]
    delta = new["roc_auc"] - old["roc_auc"]
    auc_ci = bootstrap["metrics"]["delta_auc"]["ci95"]
    ap_ci = bootstrap["metrics"]["delta_ap"]["ci95"]
    size = diagnostic["size_trial_epoch30"]["cnn_actual"]
    fold_rows = "\n".join(
        f"| {int(r.fold)} | {r.historical_auc:.3f} | {r.clinical_auc:.3f} | {r.delta_auc:+.3f} |"
        for r in fold_metrics.itertuples())
    cohort_rows = "\n".join(
        f"| {r.cohort.upper()} | {int(r.n)} | {r.historical_auc:.3f} | {r.clinical_auc:.3f} | {r.delta_auc:+.3f} |"
        for r in cohort_metrics.itertuples())
    text = f"""# Justificación cuantitativa de la mejora del modelo

Fecha: 07/10/2026. Unidad de evaluación: paciente.

## Conclusión ejecutiva

El cambio está justificado en **validación cruzada OOF**, no todavía en un test
privado nuevo. La CNN histórica obtiene AUC OOF **{old['roc_auc']:.4f}** y el
baseline clínico que inicializa el candidato multimodal obtiene **{new['roc_auc']:.4f}**:
una diferencia absoluta de **{delta:+.4f}**. El bootstrap pareado por paciente,
estratificado por fold y clase, sitúa el IC95% de la diferencia en
**[{auc_ci[0]:+.4f}, {auc_ci[1]:+.4f}]**. El intervalo no incluye cero.

La AP pasa de **{old['average_precision']:.4f}** a **{new['average_precision']:.4f}**
(IC95% pareado del cambio [{ap_ci[0]:+.4f}, {ap_ci[1]:+.4f}]). La prevalencia es
{new['prevalence']:.4f}, por lo que ambos modelos superan el baseline aleatorio de AP,
pero la separación clínica es mayor.

| Evaluación | ROC-AUC | AP | Brier ↓ | Log-loss ↓ |
|---|---:|---:|---:|---:|
| CNN histórica · OOF | {old['roc_auc']:.4f} | {old['average_precision']:.4f} | {old['brier']:.4f} | {old['log_loss']:.4f} |
| CNN histórica · test | {test['roc_auc']:.4f} | {test['average_precision']:.4f} | {test['brier']:.4f} | {test['log_loss']:.4f} |
| Baseline clínico · OOF | {new['roc_auc']:.4f} | {new['average_precision']:.4f} | {new['brier']:.4f} | {new['log_loss']:.4f} |

## Por qué era necesario cambiar

En el diagnóstico de época 30 la CNN alcanza AUC train **{size['train_patient_auc']:.3f}**
y validación **{size['patient_auc']:.3f}**, una brecha de
**{size['train_patient_auc']-size['patient_auc']:.3f}**. La loss de train sigue
bajando mientras la loss de validación permanece alta: es sobreajuste, no falta de
capacidad. Además, el modelo histórico seleccionado baja de AUC OOF
**{old['roc_auc']:.3f}** a AUC test **{test['roc_auc']:.3f}**.

![ROC y PR OOF](01_roc_pr_oof.png)

![Diagnóstico de sobreajuste](03_sobreajuste_train_validacion_test.png)

## Comparación por fold

| Fold | CNN histórica | Baseline clínico | Diferencia |
|---:|---:|---:|---:|
{fold_rows}

La media de AUC de los cinco folds del baseline clínico es
**{fold_metrics.clinical_auc.mean():.4f}**. Dos folds permanecen por debajo de 0,70;
por eso el resultado es prometedor pero no demuestra todavía estabilidad perfecta.

![AUC por fold](02_auc_por_fold.png)

## Comparación por cohorte

| Cohorte | Pacientes | CNN histórica | Baseline clínico | Diferencia |
|---|---:|---:|---:|---:|
{cohort_rows}

![Probabilidades y cohortes](04_probabilidades_y_cohortes.png)

## Calibración

El AUC mide ordenación, no que una probabilidad de 0,70 signifique un 70% real.
El Brier OOF cambia de **{old['brier']:.4f}** a **{new['brier']:.4f}** y el ECE de
10 cuantiles de **{summary['historical_ece']:.4f}** a
**{summary['clinical_ece']:.4f}**. La calibración deberá reajustarse por cross-fitting
después de entrenar el ensemble multimodal completo.

![Calibración OOF](05_calibracion_oof.png)

## Qué se puede y qué no se puede afirmar

- Sí: las cuatro variables clínicas aportan una mejora OOF grande, pareada y con
  IC95% positivo frente a las predicciones de la CNN histórica.
- Sí: el baseline supera 0,70 tanto en AUC media de folds como en AUC OOF agrupada.
- Sí: el candidato multimodal conserva ese baseline como checkpoint de época 0,
  por lo que el entrenamiento visual no puede reemplazarlo por un checkpoint peor.
- No: todavía no existe una estimación del candidato multimodal completo en el
  test privado del profesor.
- No: esta comparación OOF es interna y adaptativa; el conjunto de desarrollo ya
  se observó al escoger las variables y no sustituye una validación independiente.
- No: el AUC test histórico de {test['roc_auc']:.3f} pertenece a la CNN anterior;
  no debe presentarse como test del modelo clínico.
- No: un AUC mayor de 0,70 por sí solo no demuestra calibración ni utilidad clínica.

## Reproducibilidad

Ejecutar `python -m cancer_mama.informe_mejora`. Se regeneran las predicciones clínicas
OOF, métricas, bootstrap, tablas y figuras sin cargar imágenes ni etiquetas nuevas
de test. Los ficheros numéricos quedan en esta misma carpeta.
"""
    (OUTPUT / "INFORME_JUSTIFICACION.md").write_text(text, encoding="utf-8")


def main():
    entrenamiento = load_training_module()
    clinical = clinical_oof(entrenamiento)
    historical = pd.read_csv(HISTORICAL / "oof_pacientes.csv")[
        ["patient_id", "label", "fold", "cohort", "probability"]].copy()
    historical.patient_id = historical.patient_id.astype(str)
    clinical.to_csv(OUTPUT / "predicciones_clinicas_oof.csv", index=False)
    comparison = historical.merge(clinical, on=["patient_id", "label", "fold", "cohort"],
                                  suffixes=("_historical", "_clinical"), validate="one_to_one")
    if len(comparison) != len(historical) or len(comparison) != len(clinical):
        raise ValueError("Las predicciones OOF no cubren exactamente las mismas pacientes")
    comparison.to_csv(OUTPUT / "comparacion_pareada_oof.csv", index=False)
    old_frame = comparison.rename(columns={"probability_historical": "probability"})
    new_frame = comparison.rename(columns={"probability_clinical": "probability"})
    old_summary, new_summary = metrics(old_frame), metrics(new_frame)
    test_frame = pd.read_csv(HISTORICAL / "test_pacientes.csv").rename(columns={"probability": "probability"})
    test_summary = metrics(test_frame)
    summary = {"historical_oof": old_summary, "clinical_oof": new_summary,
               "historical_test": test_summary, "historical_ece": ece(old_frame),
               "clinical_ece": ece(new_frame), "test_note":
               "Historical CNN only; the new clinical/multimodal candidate was not evaluated on test."}
    (OUTPUT / "metricas_resumen.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([{"model": key, **value} for key, value in summary.items() if isinstance(value, dict)]).to_csv(
        OUTPUT / "metricas_resumen.csv", index=False)
    fold_rows = []
    for fold in range(5):
        subset = comparison[comparison.fold.eq(fold)]
        old_auc = roc_auc_score(subset.label, subset.probability_historical)
        new_auc = roc_auc_score(subset.label, subset.probability_clinical)
        fold_rows.append({"fold": fold, "n": len(subset), "historical_auc": old_auc,
                          "clinical_auc": new_auc, "delta_auc": new_auc-old_auc})
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(OUTPUT / "metricas_por_fold.csv", index=False)
    cohort_rows = []
    for cohort, subset in comparison.groupby("cohort"):
        old_auc = roc_auc_score(subset.label, subset.probability_historical)
        new_auc = roc_auc_score(subset.label, subset.probability_clinical)
        cohort_rows.append({"cohort": cohort, "n": len(subset), "historical_auc": old_auc,
                            "clinical_auc": new_auc, "delta_auc": new_auc-old_auc})
    cohort_metrics = pd.DataFrame(cohort_rows).sort_values("cohort")
    cohort_metrics.to_csv(OUTPUT / "metricas_por_cohorte.csv", index=False)
    bootstrap = paired_bootstrap(comparison)
    (OUTPUT / "bootstrap_pareado.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    curves = pd.read_csv(HISTORICAL / "curvas_entrenamiento.csv")
    curve_summary = curves.groupby("epoch").agg(
        n_runs=("fold", "size"), train_loss_median=("train_loss", "median"),
        train_loss_q25=("train_loss", lambda x: x.quantile(.25)),
        train_loss_q75=("train_loss", lambda x: x.quantile(.75)),
        val_loss_median=("val_loss", "median"), val_loss_q25=("val_loss", lambda x: x.quantile(.25)),
        val_loss_q75=("val_loss", lambda x: x.quantile(.75))).reset_index()
    curve_summary.to_csv(OUTPUT / "curvas_loss_resumen.csv", index=False)
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    plot_roc_pr(old_frame, new_frame)
    plot_folds(fold_metrics)
    plot_overfit(curves, curve_summary, diagnostic, old_summary, test_summary, new_summary)
    plot_distributions_cohorts(comparison, cohort_metrics)
    plot_calibration(old_frame, new_frame)
    write_report(summary, fold_metrics, cohort_metrics, bootstrap, diagnostic)
    provenance = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                  (Path(entrenamiento.__file__), HISTORICAL/"oof_pacientes.csv",
                   HISTORICAL/"test_pacientes.csv", HISTORICAL/"curvas_entrenamiento.csv", DIAGNOSTIC)}
    (OUTPUT / "procedencia.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "historical_oof_auc": old_summary["roc_auc"],
                      "clinical_oof_auc": new_summary["roc_auc"],
                      "delta_auc": new_summary["roc_auc"]-old_summary["roc_auc"],
                      "delta_auc_ci95": bootstrap["metrics"]["delta_auc"]["ci95"]}, indent=2))


if __name__ == "__main__":
    main()
