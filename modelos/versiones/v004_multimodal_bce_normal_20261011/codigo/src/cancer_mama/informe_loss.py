"""Análisis y manifiesto candidato de LR actual con selección por log-loss.

Reutiliza checkpoints congelados. La calibración de evaluación solo se ajusta
con selección interna; la calibración operativa usa OOF y se etiqueta aparente.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import entrenamiento as base
from . import experimento_loss as experiment
from .paths import PROJECT_ROOT, RESULTS_DIR

DEFAULT_REPORT = RESULTS_DIR / "08_informe_loss_20261010"


def calibrate_inner(selection, outer):
    """No usa etiquetas exteriores para ajustar calibración ni umbral."""
    if selection.patient_id.duplicated().any() or outer.patient_id.duplicated().any():
        raise ValueError("Predicciones duplicadas por paciente")
    if set(selection.patient_id) & set(outer.patient_id):
        raise ValueError("Solape entre selección y exterior")
    calibration = base.fit_platt(selection.label.to_numpy(), selection.probability.to_numpy())
    calibration["fit_data"] = "inner_selection_only_after_checkpoint_selection"
    inner_calibrated = base.apply_calibration(selection.probability.to_numpy(), calibration)
    threshold = base.youden_threshold(selection.label.to_numpy(), inner_calibrated)
    result = outer.copy()
    result["probability_raw"] = result.probability
    result["probability"] = base.apply_calibration(result.probability.to_numpy(), calibration)
    result["threshold"] = threshold
    return result, calibration, threshold


def metrics(frame):
    result = base.binary_metrics(frame.label, frame.probability)
    values = [base.roc_auc_score(f.label, f.probability) for _, f in frame.groupby("fold")]
    return {key: result[key] for key in ("n_patients", "roc_auc", "average_precision", "log_loss", "brier")} | {
        "mean_fold_auc": float(np.mean(values)), "fold_auc": values}


def verify_inputs(root, source):
    protocol = json.loads((source / "protocolo.json").read_text())
    parameters = protocol["parameters"]
    if base.config_hash(parameters) != protocol["fingerprint"]:
        raise ValueError("Protocolo modificado")
    for name, path in (("experiment", Path(experiment.__file__)), ("training", Path(base.__file__)),
                       ("data", Path(base.pdatos.__file__))):
        if base.sha256(path) != parameters["sources"][name]:
            raise ValueError("El análisis requiere el código conservado del ensayo")
    for relative, digest in parameters["protected"].items():
        if base.sha256(PROJECT_ROOT / relative) != digest:
            raise ValueError(f"Archivo protegido cambiado: {relative}")
    samples = base.cargar_train(root, include_clinical=True)
    if experiment.signature(samples) != parameters["data_signature"]:
        raise ValueError("Datos diferentes del ensayo")
    if experiment.input_image_signature(samples, root) != parameters["train_png_sha256"]:
        raise ValueError("Imágenes de desarrollo modificadas")
    return samples, protocol


def analyze(root, source, output, device):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    samples, protocol = verify_inputs(root, source)
    base.seed_everything(20261010)
    torch.set_num_threads(8)
    cache = base.CacheTrain(samples, root, source / ".cache", workers=4)
    raw = {name: pd.read_csv(source / f"comparacion/oof_lr_actual_{name}.csv")
           for name in experiment.CRITERIA}
    calibrated, calibration_records, epochs, models = {name: [] for name in raw}, [], [], []
    evidence = [source / "protocolo.json", source / "estado.json", source / "comparacion/resumen.json"]
    input_hashes = {str(path.relative_to(PROJECT_ROOT)): base.sha256(path) for path in evidence}
    selection_frames = []
    for fold in range(5):
        predictions = {name: [] for name in raw}
        selection_reference = None
        for seed in (42, 2026):
            folder = source / f"lr_actual/seed_{seed}/fold_{fold}"
            config = json.loads((folder / "config.json").read_text())
            summary = json.loads((folder / "summary.json").read_text())
            if (summary["status"] != "complete" or not summary["eligible_for_comparison"]
                    or summary["config_hash"] != base.config_hash(config)
                    or config["data_signature"] != experiment.signature(samples)):
                raise ValueError("Run incompleto, de prueba o modificado")
            for filename, digest in summary["files"].items():
                if base.sha256(folder / filename) != digest:
                    raise ValueError(f"Artefacto cambiado: {folder / filename}")
                input_hashes[str((folder / filename).relative_to(PROJECT_ROOT))] = digest
            for filename in ("config.json", "summary.json"):
                input_hashes[str((folder / filename).relative_to(PROJECT_ROOT))] = base.sha256(folder / filename)
            fit, selection, outer = experiment.split_patients(samples, fold, config["inner_fraction"], config["split_seed"])
            for role, frame in (("fit", fit), ("selection", selection), ("outer", outer)):
                if experiment.signature(frame) != config["partition_signatures"][role]:
                    raise ValueError("Partición distinta del entrenamiento")
            if selection_reference is not None and not selection.sample_id.equals(selection_reference.sample_id):
                raise ValueError("Las semillas tienen selecciones diferentes")
            selection_reference = selection
            loader = base.make_loader(selection, root, 64, 0, torch.Generator().manual_seed(seed), cache=cache, clinical=True)
            history = pd.read_csv(folder / "history.csv").set_index("epoch")
            for name in raw:
                artifact = torch.load(folder / f"best_{name}.pt", map_location="cpu", weights_only=True)
                if artifact["model_config"] != config["model"]:
                    raise ValueError("Configuración del peso distinta del run")
                model = base.CNN(artifact["model_config"]).to(device)
                model.load_state_dict(artifact["state_dict"], strict=True)
                probabilities, _ = base.predict(model, loader, device, False)
                predictions[name].append(probabilities)
                epoch = summary["results"][name]["selection"]["best_epoch"]
                chosen, last = history.loc[epoch], history.iloc[-1]
                epochs.append({"seed": seed, "fold": fold, "criterion": name, "chosen_epoch": epoch,
                    "final_epoch": int(last.name), "chosen_fit_loss": float(chosen.fit_patient_logloss),
                    "chosen_selection_loss": float(chosen.selection_patient_logloss),
                    "final_fit_loss": float(last.fit_patient_logloss),
                    "final_selection_loss": float(last.selection_patient_logloss)})
                if name == "logloss":
                    weight = folder / "best_logloss.pt"
                    models.append({"seed": seed, "fold": fold, "path": os.path.relpath(weight.resolve(), output.resolve()),
                                   "sha256": base.sha256(weight), "model_config": artifact["model_config"]})
                del model
        for name in raw:
            inner = experiment.patient_frame(selection_reference, np.mean(predictions[name], axis=0))
            selection_frames.append(inner.assign(criterion=name, outer_fold=fold))
            held_out = raw[name][raw[name].fold.eq(fold)].copy()
            result, calibration, threshold = calibrate_inner(inner, held_out)
            calibrated[name].append(result)
            calibration_records.append({"fold": fold, "criterion": name, "calibration": calibration,
                                        "threshold": threshold, "n_inner_patients": len(inner)})
        print(f"Calibración interna y diagnóstico completados: fold {fold}", flush=True)
    calibrated = {key: pd.concat(value).sort_values("patient_id").reset_index(drop=True)
                  for key, value in calibrated.items()}
    # Reconstruye OOF desde los runs verificados y comprueba los CSV agregados usados.
    roster = samples.rename(columns={"pCR": "label", "dataset": "cohort"})
    for name in raw:
        seeds = []
        for seed in (42, 2026):
            slices = pd.concat([pd.read_csv(source / f"lr_actual/seed_{seed}/fold_{fold}/oof_{name}.csv")
                               for fold in range(5)], ignore_index=True).assign(split="train")
            seeds.append(base.validate_oof(slices, roster))
        slices = seeds[0].copy()
        slices["probability"] = (seeds[0].probability.to_numpy() + seeds[1].probability.to_numpy()) / 2
        rebuilt = base.aggregate_patients(slices, "median")
        pd.testing.assert_frame_equal(rebuilt, raw[name], check_exact=False, atol=1e-14, rtol=1e-14)
        raw[name].to_csv(output / f"oof_raw_{name}.csv", index=False)
        calibrated[name].to_csv(output / f"oof_calibrated_inner_{name}.csv", index=False)
    pd.DataFrame(epochs).to_csv(output / "epocas.csv", index=False)
    pd.concat(selection_frames, ignore_index=True).to_csv(output / "predicciones_seleccion.csv", index=False)
    base.json_write(output / "calibradores_internos.json", calibration_records)
    fold_rows = []
    for scale, frames in (("raw", raw), ("calibrated_inner", calibrated)):
        for name, frame in frames.items():
            for fold, group in frame.groupby("fold"):
                fold_rows.append({"scale": scale, "criterion": name, "fold": int(fold), **metrics(group)})
    pd.DataFrame(fold_rows).drop(columns="fold_auc").to_csv(output / "metricas_folds.csv", index=False)
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "raw": {key: metrics(frame) for key, frame in raw.items()},
              "calibrated_inner": {key: metrics(frame) for key, frame in calibrated.items()},
              "calibrated_intervals_98_33": experiment.paired_intervals(calibrated["auc"], calibrated["logloss"]),
              "raw_comparison": json.loads((source / "comparacion/resumen.json").read_text())["comparisons"]["lr_actual_logloss"],
              "training_commit": protocol["git_commit"], "test_images_loaded": 0,
              "warning": "Análisis secundario después de elegir candidato sobre desarrollo observado. "
                         "Calibradores de evaluación ajustados solo con selección interna; "
                         "los intervalos condicionan a predicciones fijas. No es confirmación externa.",
              "candidate_only": True, "active_version": "v002_multimodal_clinico_20261007"}
    # Manifiesto operativo separado: Platt y Youden sobre OOF completas, no métrica independiente.
    candidate = raw["logloss"]
    operational_calibration = base.fit_platt(candidate.label.to_numpy(), candidate.probability.to_numpy())
    operational_calibration["fit_data"] = "all_outer_oof_for_candidate_deployment"
    probability = base.apply_calibration(candidate.probability.to_numpy(), operational_calibration)
    threshold = base.youden_threshold(candidate.label.to_numpy(), probability)
    report["operational_calibration_apparent"] = base.binary_metrics(candidate.label, probability, threshold)
    report["candidate_threshold"] = threshold
    manifest = {"schema_version": 1, "status": "candidato_desarrollo_sin_test", "models": models,
                "aggregation": "median", "calibration": operational_calibration, "threshold": threshold,
                "decision_policy": {"strategy": "youden", "fit_data": "all_outer_oof_apparent"},
                "test_evaluated_once": False,
                "selection": {"candidate": "lr_actual_logloss", "lr": 0.0008,
                              "checkpoint_metric": "inner_patient_logloss", "metrics": metrics(candidate)},
                "preprocessing": {"phases": ["PRE", "EARLY", "LATE"], "shape": [3, 256, 256],
                                  "model_representation": "raw", "network_channels": 3,
                                  "scaling": "uint8 / 255; CNN: 2*x-1", "clinical_required": True,
                                  "clinical_raw_columns": list(base.CLINICAL_COLUMNS)},
                "evaluation_warning": report["warning"]}
    base.json_write(output / "resumen.json", report)
    base.json_write(output / "entradas_sha256.json", input_hashes)
    base.json_write(output / "modelo_candidato.json", manifest)
    (output / "modelo_candidato.json.sha256").write_text(base.sha256(output / "modelo_candidato.json") + "\n")
    base.verificar_modelo_final(output / "modelo_candidato.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=base.pdatos.DATOS)
    parser.add_argument("--origen", type=Path, default=experiment.DEFAULT_OUTPUT)
    parser.add_argument("--salida", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dispositivo", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--solo-presentacion", action="store_true", help="Regenera gráficos e informe desde métricas existentes")
    args = parser.parse_args()
    report = (json.loads((args.salida / "resumen.json").read_text()) if args.solo_presentacion
              else analyze(args.datos, args.origen, args.salida, torch.device(args.dispositivo)))
    render_report(args.origen, args.salida)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def render_report(source, output):
    source, output = Path(source), Path(output)
    report = json.loads((output / "resumen.json").read_text())
    folds = pd.read_csv(output / "metricas_folds.csv")
    epochs = pd.read_csv(output / "epocas.csv")
    os.environ.setdefault("MPLCONFIGDIR", str(source / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for scale, offset, color, label in (("raw", -.17, "#267b89", "Probabilidades crudas"),
                                      ("calibrated_inner", .17, "#c65a3e", "Calibración interna")):
        part = folds[folds.scale.eq(scale)]
        a = part[part.criterion.eq("auc")].set_index("fold").sort_index()
        b = part[part.criterion.eq("logloss")].set_index("fold").sort_index()
        for ax, metric in zip(axes, ("log_loss", "roc_auc")):
            ax.bar(np.arange(5)+offset, b[metric]-a[metric], width=.32, color=color, label=label)
    for ax, title in zip(axes, ("Cambio de log-loss · negativo favorece al candidato",
                                "Cambio de AUC · positivo favorece al candidato")):
        ax.axhline(0, color="#444444", linewidth=.8)
        ax.set(title=title, xlabel="Fold exterior", xticks=np.arange(5))
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=9)
    fig.suptitle("LR 0,0008: selección por pérdida frente a selección por AUC", fontsize=14)
    fig.savefig(output / "01_comparacion_folds.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.9), sharey=True, constrained_layout=True)
    for row, seed in enumerate((42, 2026)):
        for fold in range(5):
            ax = axes[row, fold]
            history = pd.read_csv(source / f"lr_actual/seed_{seed}/fold_{fold}/history.csv")
            ax.plot(history.epoch, history.fit_patient_logloss, color="#267b89", label="Fit en evaluación")
            ax.plot(history.epoch, history.selection_patient_logloss, color="#c65a3e", label="Selección interna")
            for criterion, marker, color in (("auc", "o", "#645b91"), ("logloss", "D", "#202d39")):
                best = int(epochs[(epochs.seed.eq(seed)) & (epochs.fold.eq(fold)) &
                                  (epochs.criterion.eq(criterion))].chosen_epoch.iloc[0])
                value = history.loc[history.epoch.eq(best), "selection_patient_logloss"].iloc[0]
                ax.scatter([best], [value], marker=marker, edgecolors=color, facecolors="none", s=75,
                           linewidths=1.5, zorder=4, label="Elegido por " + criterion)
            ax.set(title=f"Semilla {seed} · fold {fold}", xlabel="Época")
            ax.grid(alpha=.15)
            if fold == 0:
                ax.set_ylabel("Log-loss por paciente")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Misma trayectoria; dos criterios eligen distintos checkpoints\n"
                 "FP32, sin aumentos ni dropout · época 0 incluida", fontsize=14)
    fig.savefig(output / "02_curvas_y_seleccion.png", dpi=150)
    plt.close(fig)
    interval = report["calibrated_intervals_98_33"]["log_loss"]
    raw_delta = report["raw_comparison"]["delta"]["log_loss"]
    calibrated_delta = report["calibrated_inner"]["logloss"]["log_loss"] - report["calibrated_inner"]["auc"]["log_loss"]
    if interval["upper"] < 0:
        interpretation = "La reducción de log-loss también se mantiene después de calibrar en este análisis secundario."
    elif interval["lower"] > 0:
        interpretation = "Después de calibrar, el candidato empeora la log-loss en este análisis secundario."
    else:
        interpretation = "Después de calibrar, el intervalo incluye cero: no se demuestra una ventaja adicional en log-loss."
    lines = ["# Candidato multimodal: LR actual y selección por pérdida", "",
             "Se reutilizaron los diez checkpoints ya entrenados con LR 0,0008 y selección interna por log-loss.",
             "El candidato queda preparado y versionable; v002 continúa como base activa.", "",
             "## Comparación equiparada", "",
             "El control y el candidato comparten datos, inicialización y trayectoria. Solo cambia el checkpoint elegido.",
             "Las métricas corresponden a 1.097 pacientes de folds exteriores que no eligieron épocas ni calibración.", "",
             "| Escala | Selector | AUC media folds | AUC OOF | AP | Log-loss | Brier |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for scale in ("raw", "calibrated_inner"):
        for name in ("auc", "logloss"):
            m = report[scale][name]
            lines.append(f"| {scale} | {name} | {m['mean_fold_auc']:.6f} | {m['roc_auc']:.6f} | "
                         f"{m['average_precision']:.6f} | {m['log_loss']:.6f} | {m['brier']:.6f} |")
    lines += ["", f"Delta log-loss cruda: {raw_delta:+.6f}. Tras calibración interna: {calibrated_delta:+.6f}; "
              f"IC98,33% condicionado [{interval['lower']:+.6f}, {interval['upper']:+.6f}].", "", interpretation,
              "", "![Diferencias por fold](01_comparacion_folds.png)", "",
              "## Qué aportó seleccionar por pérdida", "",
              "La pérdida cruda mejora en tres folds y coincide en dos, donde se eligen los mismos pesos. "
              "Seis de los diez checkpoints cambian: el selector AUC conserva seis épocas 0 y el de pérdida conserva dos.",
              "La AUC media cruda aumenta en estimación puntual, pero su intervalo de diferencia incluye cero.",
              "", "Las curvas de ambos selectores son la misma trayectoria. Elegir otra época mejora el modelo conservado, "
              "pero no elimina el sobreajuste que aparece al prolongar el aprendizaje.",
              "", "![Curvas y checkpoints](02_curvas_y_seleccion.png)", "",
              "## Calibración y preparación de la versión", "",
              "Para evaluar calibración, se promedian las dos semillas por corte de selección interna y se agrega por mediana. "
              "Platt y Youden se ajustan con esas pacientes y se aplican al fold exterior. Las etiquetas exteriores no ajustan "
              "el calibrador ni el umbral. El calibrador utiliza el mismo subconjunto interno que eligió el checkpoint; "
              "la evaluación exterior mantiene esa selección separada.",
              "", "El manifiesto operativo tiene una calibración distinta: Platt y Youden ajustados con todas las OOF. "
              "Sus métricas son aparentes y están separadas en resumen.json. No se presentan como evaluación independiente.",
              "", f"Umbral operativo candidato: {report['candidate_threshold']:.10f}. "
              "La inferencia usa diez modelos; cada predicción OOF se calculó con los dos modelos de su fold. "
              "La transferencia de calibración al ensemble completo necesita confirmación externa.",
              "", "## Decisión", "",
              "Conservar LR 0,0008 y selección por pérdida como candidato reproducible. No sustituir v002 por comparar "
              "su AUC histórica de 0,724568 con esta AUC: el protocolo y la fracción de aprendizaje son diferentes. "
              "La evidencia disponible no demuestra superioridad sobre v002 en pacientes nuevas.",
              "", report["warning"], "No se abrieron imágenes de test ni validación privada.",
              "", "## Reproducción y evidencia", "",
              "`.venv/bin/python -B -m cancer_mama.informe_loss` reconstruye el análisis desde la evidencia local.",
              "`--solo-presentacion` regenera únicamente el informe y las figuras. "
              "Los archivos entradas_sha256.json, epocas.csv, metricas_folds.csv, calibradores_internos.json "
              "y las OOF conservan la trazabilidad del diagnóstico."]
    (output / "INFORME_RESULTADOS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
