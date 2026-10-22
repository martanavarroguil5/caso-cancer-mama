"""Recalcula y documenta el piloto 10/3/1, sin entrenar ni publicar."""
from datetime import datetime
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

from cancer_mama import entrenamiento as base
from cancer_mama import entrenamiento_imagen as image
from cancer_mama import seleccion_cortes as pilot

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = pilot.OUTPUT
COLORS = {10: "#2764c4", 3: "#dd7f18", 1: "#25856a"}


def slice_label(k):
    return f"{k} corte" + ("s" if k != 1 else "")


def auc(y, p):
    positive = y == 1
    n = positive.sum()
    return float((rankdata(p)[positive].sum() - n * (n + 1) / 2) / (n * (~positive).sum()))


def main():
    state = json.loads((RUN / "estado_piloto.json").read_text())
    assert state["status"] == "complete" and state["verified_runs"] == 12
    assert not (RUN / "error.json").exists()
    original = json.loads((RUN / "comparacion.json").read_text())
    prep = json.loads((RUN / "preparacion.json").read_text())
    for name in ("lanzamiento.json", "estado_piloto.json", "protocolo_ejecutado.json",
                 "comparacion.json", "verificacion_final.json", "preparacion.json",
                 "verificacion_particiones.json"):
        shutil.copy2(RUN / name, OUT / name)
    frames, metrics, histories, configs = {}, {}, [], []
    for k in pilot.KS:
        frame = pd.read_csv(RUN / f"oof_{k}.csv").sort_values("patient_id").reset_index(drop=True)
        assert not frame.patient_id.duplicated().any()
        frame.to_csv(OUT / f"oof_{k}.csv", index=False)
        frames[k] = frame
        m = image.metricas_binarias(frame.label, frame.probability_raw, .5)
        assert np.isclose(m["roc_auc"], auc(frame.label.to_numpy(), frame.probability_raw.to_numpy()))
        m["sensitivity"] = m["recall_pcr"]
        m["specificity"] = m["recall_no_pcr"]
        m["fold_auc"] = {str(f): float(roc_auc_score(g.label, g.probability_raw)) for f, g in frame.groupby("fold")}
        m["mean_fold_auc"] = float(np.mean(list(m["fold_auc"].values())))
        assert np.isclose(m["mean_fold_auc"], original["results"][str(k)]["mean_fold_auc"])
        selected_rows, final_rows, epochs = [], [], []
        for seed in (42, 2026):
            for fold in (0, 3):
                path = RUN / f"cnn_seleccion_cortes_{k}/weighted/seed_{seed}/fold_{fold}"
                cfg = json.loads((path / "config.json").read_text())
                summary = json.loads((path / "summary.json").read_text())
                assert summary["status"] == "complete" and not summary["smoke"]
                assert summary["config_hash"] == base.config_hash(cfg)
                assert base.sha256(path / "inference.pt") == summary["inference_sha256"]
                h = pd.read_csv(path / "history.csv")
                selected_rows.append(h.loc[h.epoch.eq(summary["best_epoch"])].iloc[0].to_dict())
                final_rows.append(h.iloc[-1].to_dict())
                epochs.append(summary["best_epoch"])
                histories.append(h.assign(k=k, seed=seed, fold=fold, selected_epoch=summary["best_epoch"]))
                configs.append({"k": k, "seed": seed, "fold": fold, "config": cfg, "summary": summary,
                                "environment": json.loads((path / "environment.json").read_text())})
        for column, key in (("auc_gap", "gap"), ("fit_patient_auc", "fit_auc"),
                            ("selection_patient_auc", "selection_auc")):
            m[f"selected_{key}_median"] = float(np.median([row[column] for row in selected_rows]))
            m[f"final_{key}_median"] = float(np.median([row[column] for row in final_rows]))
        m["best_epochs"] = epochs
        m["probability_quantiles"] = dict(zip(("min", "q10", "median", "q90", "max"),
                                             np.quantile(frame.probability_raw, [0, .1, .5, .9, 1]).tolist()))
        metrics[k] = m
    reference = frames[10]
    assert len(reference) == 439
    for frame in frames.values():
        assert frame[["patient_id", "label", "fold", "cohort", "n_slices"]].equals(
            reference[["patient_id", "label", "fold", "cohort", "n_slices"]])
    y, folds = reference.label.to_numpy(), reference.fold.to_numpy()
    probabilities = {k: f.probability_raw.to_numpy() for k, f in frames.items()}
    strata = list(reference.groupby(["fold", "label"]).indices.values())
    rng = np.random.default_rng(20261022)
    boot = {k: {"pooled": [], "mean_fold": []} for k in pilot.KS}
    for _ in range(2000):
        idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        yy, ff = y[idx], folds[idx]
        for k in pilot.KS:
            pp = probabilities[k][idx]
            boot[k]["pooled"].append(auc(yy, pp))
            boot[k]["mean_fold"].append(np.mean([auc(yy[ff == f], pp[ff == f]) for f in (0, 3)]))
    comparisons = {}
    for k in pilot.KS:
        for metric in ("pooled", "mean_fold"):
            metrics[k][f"{metric}_auc_ci95"] = np.quantile(boot[k][metric], [.025, .975]).tolist()
        if k != 10:
            c = dict(original["results"][str(k)])
            c.pop("metrics")
            for metric in ("pooled", "mean_fold"):
                delta = np.array(boot[k][metric]) - np.array(boot[10][metric])
                c[f"delta_{metric}_auc_ci95"] = np.quantile(delta, [.025, .975]).tolist()
            c["delta_fold_auc"] = {str(f): metrics[k]["fold_auc"][str(f)] - metrics[10]["fold_auc"][str(f)] for f in (0, 3)}
            c["gate_components"] = {
                "mean_gain_at_least_0_01": c["delta_mean_auc_vs_10"] >= .01,
                "no_fold_drop_larger_than_0_02": min(c["delta_fold_auc"].values()) >= -.02,
                "selected_signed_gap_not_larger": metrics[k]["selected_gap_median"] <= metrics[10]["selected_gap_median"]}
            comparisons[k] = c
    elapsed = (datetime.fromisoformat(state["finished_at"]) - datetime.fromisoformat(state["started_at"])).total_seconds()
    result = {"status": "exploratory_two_fold_pilot_not_adopted", "metrics": metrics,
              "comparisons": comparisons, "patients": len(reference), "positive_pcr": int(y.sum()),
              "negative_no_pcr": int((y == 0).sum()), "primary_metric": "mean_raw_auc_per_outer_fold",
              "raw_fixed_threshold": .5, "calibration": "none", "bootstrap_samples": 2000,
              "bootstrap_seed": 20261022, "bootstrap_strata": ["fold", "class"],
              "bootstrap_limits": "Conditional on these fitted models, no retraining/search uncertainty",
              "runtime_seconds": elapsed, "physical_epochs": sum(len(h) for h in histories),
              "test_images_loaded": 0, "active_model": "v002_multimodal_clinico_20261007"}
    base.json_write(OUT / "evaluacion_recalculada.json", result)
    base.json_write(OUT / "configuraciones_y_resumenes.json", configs)
    pd.concat(histories, ignore_index=True).to_csv(OUT / "historias.csv", index=False)
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for k, frame in frames.items():
        fpr, tpr, _ = roc_curve(frame.label, frame.probability_raw)
        axes[0].plot(fpr, tpr, color=COLORS[k], label=f"{slice_label(k)} · AUC {metrics[k]['roc_auc']:.3f}")
        precision, recall, _ = precision_recall_curve(frame.label, frame.probability_raw)
        axes[1].plot(recall, precision, color=COLORS[k], label=f"{slice_label(k)} · AP {metrics[k]['average_precision']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].axhline(y.mean(), linestyle="--", color="gray", label=f"Prevalencia {y.mean():.3f}")
    axes[0].set(title="ROC exterior agrupada", xlabel="Tasa de falsos positivos", ylabel="Sensibilidad pCR")
    axes[1].set(title="Precisión y sensibilidad", xlabel="Sensibilidad pCR", ylabel="Precisión pCR")
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle("Piloto de selección de cortes · 439 pacientes exteriores · folds 0 y 3")
    fig.tight_layout(); fig.savefig(OUT / "01_roc_pr.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for index, k in enumerate(pilot.KS):
        axes[0].bar(np.arange(2) + (index - 1) * .23,
                    [metrics[k]["fold_auc"][str(f)] for f in (0, 3)], width=.23,
                    color=COLORS[k], label=slice_label(k))
        lo, hi = metrics[k]["mean_fold_auc_ci95"]
        center = metrics[k]["mean_fold_auc"]
        axes[1].errorbar(index, center, yerr=[[center-lo], [hi-center]], fmt="o", capsize=6, color=COLORS[k])
    axes[0].set(xticks=[0, 1], xticklabels=["Fold 0", "Fold 3"], title="AUC por fold · dos semillas", ylabel="AUC cruda")
    axes[0].legend()
    axes[1].set(xticks=[0, 1, 2], xticklabels=["10 cortes", "3 cortes", "1 corte"], title="Media de folds e IC95 % bootstrap")
    for ax in axes:
        ax.axhline(.60, linestyle="--", color="gray", label="0,60")
        ax.set_ylim(.35, .75); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "02_auc_folds_intervalos.png", dpi=150); plt.close(fig)
    for kind, filename in (("auc", "03_auc_entrenamiento.png"), ("loss", "04_perdidas_entrenamiento.png")):
        fig, axes = plt.subplots(3, 4, figsize=(18, 11))
        for i, k in enumerate(pilot.KS):
            for j, (seed, fold) in enumerate((s, f) for s in (42, 2026) for f in (0, 3)):
                h = next(h for h in histories if h.k.iloc[0] == k and h.seed.iloc[0] == seed and h.fold.iloc[0] == fold)
                ax = axes[i, j]
                fit, inner = ("fit_patient_auc", "selection_patient_auc") if kind == "auc" else ("fit_eval_loss", "selection_loss")
                ax.plot(h.epoch, h[fit], label="Fit sin aumentos")
                ax.plot(h.epoch, h[inner], label="Selección interna")
                ax.axvline(h.selected_epoch.iloc[0], linestyle="--", color="gray", label="Checkpoint elegido")
                ax.set(title=f"{slice_label(k)} · semilla {seed} · fold {fold}", xlabel="Época")
                if kind == "auc": ax.set_ylim(0, 1)
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("AUC por paciente" if kind == "auc" else "BCE ponderada por corte; no es log-loss exterior sin ponderar")
        fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(OUT / filename, dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, k in zip(axes, pilot.KS):
        m = metrics[k]; matrix = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=300)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > 150 else "black", fontsize=15)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No pCR", "pCR"], yticklabels=["No pCR", "pCR"],
               xlabel="Predicción", ylabel="Resultado real", title=f"{slice_label(k)} · umbral crudo 0,5")
    fig.tight_layout(); fig.savefig(OUT / "05_matrices_confusion.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for k, frame in frames.items():
        ordered = frame.sort_values("probability_raw")
        bins = [ordered.iloc[idx] for idx in np.array_split(np.arange(len(ordered)), 5)]
        axes[0].plot([b.probability_raw.mean() for b in bins], [b.label.mean() for b in bins], "o-",
                     color=COLORS[k], label=f"{slice_label(k)} · Brier {metrics[k]['brier']:.3f}")
        axes[1].hist(frame.probability_raw, bins=np.linspace(0, 1, 31), histtype="step", linewidth=2,
                     color=COLORS[k], label=slice_label(k))
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Probabilidad media cruda", ylabel="Proporción pCR observada",
                title="Fiabilidad descriptiva · cinco grupos por cuantiles")
    axes[1].axvline(.5, linestyle="--", color="gray")
    axes[1].set(xlim=(0, 1), xlabel="Probabilidad cruda por paciente", ylabel="Pacientes", title="Distribución de probabilidades")
    for ax in axes: ax.legend(fontsize=9); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "06_probabilidades_fiabilidad.png", dpi=150); plt.close(fig)
    best = max(pilot.KS, key=lambda k: metrics[k]["mean_fold_auc"])
    reaches = metrics[best]["mean_fold_auc"] >= .60
    lines = ["# Piloto de selección de cortes: resultados", "", "22/10/2026. Informe local; no se realiza commit ni push.", "",
             f"Se completaron 12 entrenamientos y {result['physical_epochs']} épocas en {elapsed/60:.2f} minutos, "
             f"incluyendo preparación y auditoría. Evaluación exterior: 439 pacientes, {int(y.sum())} con pCR y {int((y==0).sum())} sin pCR. "
             "Dos semillas por fold y dos folds exteriores (0/3). Test y validación privada cerrados.", "",
             f"**Resultado:** la mayor AUC media puntual es {metrics[best]['mean_fold_auc']:.4f}, con {slice_label(best)}. "
             + ("Supera 0,60 en este piloto, sin confirmación independiente." if reaches else "Ningún brazo alcanza la AUC media 0,60; tampoco se alcanza 0,70.")
             + " V002 permanece activo. No se inicia otro entrenamiento.", "", "## Comparación principal", "",
             "| Cortes en fit | AUC fold 0 | AUC fold 3 | Media AUC (principal) | IC95 % media | Delta frente a 10 | IC95 % delta pareada | Filtro |",
             "|---|---:|---:|---:|---|---:|---|---|"]
    for k in pilot.KS:
        m = metrics[k]; lo, hi = m["mean_fold_auc_ci95"]
        if k == 10: delta, ci, gate = "Referencia", "—", "—"
        else:
            c = comparisons[k]; a, b = c["delta_mean_fold_auc_ci95"]
            delta, ci, gate = f"{c['delta_mean_auc_vs_10']:+.4f}", f"[{a:+.4f}; {b:+.4f}]", str(c['exploratory_gate'])
        lines.append(f"| {k} | {m['fold_auc']['0']:.4f} | {m['fold_auc']['3']:.4f} | {m['mean_fold_auc']:.4f} | [{lo:.4f}; {hi:.4f}] | {delta} | {ci} | {gate} |")
    lines += ["", "El filtro prefijado exige +0,01 de AUC media, ningún fold peor en más de 0,02 y mediana de brecha "
              "fit−selección del checkpoint no mayor. Es exploratorio: dos comparaciones y antecedentes de búsqueda en train "
              "impiden tratarlo como confirmación. Incluso un filtro superado no autoriza abrir test, promocionar modelo ni iniciar otra tanda.", "",
              "Tres cortes mejora la media puntual en 0,0113 y mejora ligeramente ambos folds, pero no pasa la regla de brecha "
              "prefijada: −0,0109 frente a −0,0174 con diez. Ambas brechas son negativas; este incumplimiento no demuestra "
              "por sí solo mayor sobreajuste. Un corte empeora la media en 0,0224. La reducción de cortes no aporta una mejora "
              "suficiente y confirmada en este piloto.", "",
              "Los IC usan 2.000 remuestreos pareados por fold y clase, semilla 20261022. Condicionan a los modelos ya entrenados; "
              "no recogen la variación de nuevas semillas, folds o todo el desarrollo adaptativo. No se corrigen por comparaciones múltiples.", "",
              "## Métricas y errores", "", "AUC agrupada y AP usan probabilidades crudas. La métrica principal es la media por fold; "
              "agrupar mezcla escalas de modelos distintos. Log-loss y Brier se calculan sin ponderar por paciente. "
              "No se calibra ni se elige umbral con las pacientes exteriores.", "",
              "| Cortes | AUC agrupada | IC95 % | AP | Log-loss | Brier | Sensibilidad | Especificidad | Precisión pCR | F1 pCR |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in pilot.KS:
        m = metrics[k]; lo, hi = m['pooled_auc_ci95']
        lines.append(f"| {k} | {m['roc_auc']:.4f} | [{lo:.4f}; {hi:.4f}] | {m['average_precision']:.4f} | {m['log_loss']:.4f} | {m['brier']:.4f} | {m['sensitivity']:.2%} | {m['specificity']:.2%} | {m['precision_pcr']:.2%} | {m['f1_pcr']:.4f} |")
    lines += ["", "Los errores usan el umbral fijo 0,5 crudo. Positivo significa respuesta patológica completa (pCR), no presencia de cáncer. "
              "FP: predice pCR sin alcanzarla; FN: predice no pCR aunque se alcanza. Cambiar de FP a FN no constituye una mejora simultánea.", "",
              "| Cortes | TN | FP | FN | TP |", "|---|---:|---:|---:|---:|"]
    for k in pilot.KS:
        m = metrics[k]; lines.append(f"| {k} | {m['tn']} | {m['fp']} | {m['fn']} | {m['tp']} |")
    lines += ["", "## Sobreajuste y aprendizaje", "",
              "| Cortes | Brecha elegida mediana | Brecha final mediana | AUC fit elegida mediana | AUC selección elegida mediana | Épocas elegidas (42f0,42f3,2026f0,2026f3) |",
              "|---|---:|---:|---:|---:|---|"]
    for k in pilot.KS:
        m = metrics[k]
        lines.append(f"| {k} | {m['selected_gap_median']:+.4f} | {m['final_gap_median']:+.4f} | {m['selected_fit_auc_median']:.4f} | {m['selected_selection_auc_median']:.4f} | {m['best_epochs']} |")
    lines += ["", "La brecha se calcula en los checkpoints elegidos exclusivamente por AUC interna. Esa selección favorece métricas "
              "internas optimistas; una brecha pequeña o negativa no demuestra generalización. Hay que considerar también la AUC de fit, "
              "la evolución de ambas curvas y las AUC exteriores. No se interrumpieron ejecuciones por ruido de pocas épocas: "
              "se respetaron parada temprana y máximo prefijados. No se ajustan parámetros después de ver estas OOF.", "",
              "El problema observado no se explica solo por una brecha grande: las medianas de AUC de fit elegidas "
              "son 0,5668, 0,5611 y 0,5398. El modelo también discrimina poco en los datos de ajuste evaluados sin aumentos. "
              "Esto apunta a aprendizaje predictivo limitado en esta receta, sin identificar por sí solo si la causa "
              "es la representación de las imágenes, la optimización o la señal disponible. Con tres cortes se reduce "
              "la brecha final mediana de 0,0474 a 0,0059, pero esa reducción no se traduce en AUC exterior suficiente.", "",
              "Las probabilidades de diez y tres cortes están muy agrupadas alrededor de 0,5; con uno ninguna llega "
              "a 0,5 y se predice no pCR en las 439 pacientes. La BCE ponderada no garantiza probabilidades calibradas. "
              "Las curvas de fiabilidad son descriptivas; no se reajustan con estas pacientes. La especificidad del 100 % "
              "del brazo de un corte debe leerse junto a sensibilidad del 0 % y AUC media 0,4992.", "",
              "## Qué se ha probado", "", "- Misma CNNTemporalSimple de 551.913 parámetros, desde cero, tres fases PRE/EARLY/LATE, una escala 256×256, sin clínica.",
              "- Tres brazos nuevos y comparables: hasta 10, 3 o 1 corte por paciente solo en fit. Selección interna y exterior usan todos.",
              "- BCE ponderada por pacientes, con idéntico pos_weight y pacientes en los tres brazos de cada fold. "
              "No se reutiliza el piloto anterior como control, porque calculaba el peso de clase por corte.",
              "- AdamW LR 0,0008, WD 0,001, máximo 15 épocas, mínimo 4, paciencia 4; lotes de seis pacientes, mismo peso total por paciente. "
              "Semillas 42/2026, separación por paciente/cohorte/clase. Sin cambio de arquitectura ni protocolo durante la ejecución.",
              "- El selector fijo utiliza un medoide de las tres fases reducidas a 32×32; para tres cortes añade diversidad y separación axial. "
              "No usa pCR, máscaras ni una regla de área tumoral. Se prepara por paciente, sin aprender parámetros entre pacientes.",
              f"- Correlación mediana de descriptores en desarrollo: 10 cortes {prep['median_patient_pair_correlation']['10']:.4f}; "
              f"3 cortes {prep['median_patient_pair_correlation']['3']:.4f}. Con uno no se define correlación por pares. "
              "La reducción de correlación de píxeles no garantiza más información predictiva.",
              "- Reducir cortes cambia imágenes por lote, BatchNorm y variabilidad del gradiente. El estudio no aísla solo redundancia.",
              "- 66 tests y tres pruebas técnicas GPU pasados. Las pruebas técnicas y cualquier piloto CPU se excluyen de esta comparación.",
              "- Finitud, pasos AdamW/AMP, hashes y modelos previos auditados. No se modifica V002 ni ninguna versión o piloto anterior.", "",
              "## Entrega y límites", "", "Se conservan doce pesos exclusivos de inferencia como piloto p002 en modelos/pilotos/, "
              "con configuraciones reales, fuentes congeladas y hashes. No es una versión productiva ni un ensemble de cinco folds. "
              "Para entradas futuras hay cuatro modelos por k; estas métricas usan solo las dos semillas del fold que excluye cada paciente. "
              "No se utilizan datos del test para métricas, calibración o umbrales. No se incluyen imágenes ni checkpoints con optimizador/RNG.", "",
              "Las cifras históricas de V002 o de otros experimentos con clínica, arquitectura, entrenamiento o folds distintos "
              "no constituyen controles equivalentes. La comparación válida de esta tanda es entre sus tres brazos nuevos.", "",
              "- [Métricas completas e intervalos](evaluacion_recalculada.json).",
              "- [Configuraciones, resúmenes y entornos](configuraciones_y_resumenes.json).",
              "- [Protocolo congelado y hashes](protocolo_ejecutado.json).",
              "- [Preparación del selector](preparacion.json).",
              "- [Auditoría de pacientes y pesos](verificacion_particiones.json).",
              "- [Auditoría final](verificacion_final.json).",
              "- [Verificación del paquete aislado](verificacion_paquete.json).", ""]
    for name, label in (("01_roc_pr.png", "ROC y precisión-sensibilidad"), ("02_auc_folds_intervalos.png", "AUC por fold e intervalos"),
                        ("03_auc_entrenamiento.png", "AUC por época"), ("04_perdidas_entrenamiento.png", "Pérdidas por época"),
                        ("05_matrices_confusion.png", "Matrices de confusión"), ("06_probabilidades_fiabilidad.png", "Probabilidades y fiabilidad")):
        lines += [f"![{label}]({name})", ""]
    (OUT / "INFORME_RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "comparisons": comparisons, "runtime_minutes": elapsed/60}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
