#!/usr/bin/env python3
"""Informe de diez runs por paciente frente al multimodal previamente seleccionado."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

from . import entrenamiento as e
from .informe_multimodal import auc_ap_weighted, bootstrap_check, metric_frame, table_md
from .paths import PROJECT_ROOT as ROOT, RESULTS_DIR

DEFAULT_RUNS = RESULTS_DIR / '04_entrenamiento/patient_level_clinical_20261007'
DEFAULT_OUTPUT = RESULTS_DIR / '07_informe_paciente_20261007'
PREVIOUS = RESULTS_DIR / '06_informe_multimodal_20261007'
NAMES = {'clinical': 'Solo clínica', 'previous': 'Multimodal anterior', 'new': 'Entrenamiento por paciente'}
COLORS = {'clinical': '#26917a', 'previous': '#567ba8', 'new': '#ba5b31'}


def crossfit_policy(frame, target=.90):
    """Calibración y umbral ajustados sin etiquetas del fold al que se aplican."""
    y, p = frame.label.to_numpy(), frame.probability.to_numpy()
    probabilities = np.empty(len(frame))
    decisions = np.empty(len(frame), dtype=int)
    policies = []
    if set(frame.fold) != set(range(5)):
        raise ValueError('Se requieren cinco folds')
    for fold in range(5):
        held_out = frame.fold.to_numpy() == fold
        calibration = e.fit_platt(y[~held_out], p[~held_out])
        fit_scores = e.apply_calibration(p[~held_out], calibration)
        threshold = e.threshold_for_sensitivity(y[~held_out], fit_scores, target)
        probabilities[held_out] = e.apply_calibration(p[held_out], calibration)
        decisions[held_out] = probabilities[held_out] >= threshold
        policies.append({'fold': fold, 'threshold': threshold, 'calibration': calibration})
    metrics = e.binary_metrics(y, decisions.astype(float))
    # El ranking se calcula sobre probabilidades, no sobre decisiones binarias.
    for key in ('roc_auc', 'average_precision', 'brier', 'log_loss'):
        metrics[key] = e.binary_metrics(y, probabilities)[key]
    metrics.pop('threshold')
    return probabilities, decisions, metrics, policies


def paired_intervals(comparison, repetitions):
    rng = np.random.default_rng(2026)
    weights = np.zeros((repetitions, len(comparison)), dtype=np.int32)
    for indices in comparison.groupby(['fold', 'label'], sort=True).indices.values():
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=repetitions)
    y = comparison.label.to_numpy(dtype=int)
    scores = {}
    for name in NAMES:
        auc, ap = auc_ap_weighted(y, comparison[name].to_numpy(), weights)
        fold_aucs = []
        for fold in range(5):
            mask = comparison.fold.to_numpy() == fold
            fold_aucs.append(auc_ap_weighted(y[mask], comparison.loc[mask, name].to_numpy(), weights[:, mask])[0])
        scores[name] = {'roc_auc': auc, 'mean_fold_roc_auc': np.mean(fold_aucs, axis=0), 'average_precision': ap}
    differences = {}
    for reference in ('previous', 'clinical'):
        differences[reference] = {}
        for metric in scores['new']:
            values = scores['new'][metric] - scores[reference][metric]
            differences[reference][metric] = {'mean': float(values.mean()), 'ci95': np.quantile(values, [.025, .975]).tolist()}
    return {'repetitions': repetitions, 'seed': 2026, 'strata': ['fold', 'label'],
            'interpretation': 'Intervalos descriptivos condicionados a las predicciones OOF seleccionadas; no incluyen incertidumbre de entrenamiento ni selección.',
            'new_minus_reference': differences}


def plots(frames, folds, histories, policies, output):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.codex_tmp/matplotlib_paciente'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for key, frame in frames.items():
        fpr, tpr, _ = roc_curve(frame.label, frame.probability)
        precision, recall, _ = precision_recall_curve(frame.label, frame.probability)
        axes[0].plot(fpr, tpr, label=f'{NAMES[key]}: {roc_auc_score(frame.label,frame.probability):.3f}', color=COLORS[key])
        axes[1].plot(recall, precision, label=NAMES[key], color=COLORS[key])
    axes[0].plot([0, 1], [0, 1], '--', color='#aaaaaa')
    axes[0].set(title='ROC por paciente', xlabel='1 − especificidad', ylabel='Sensibilidad')
    axes[1].axhline(frames['new'].label.mean(), ls='--', color='#aaaaaa')
    axes[1].set(title='Precisión y recall por paciente', xlabel='Sensibilidad', ylabel='Precisión')
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.grid(alpha=.2); ax.legend(fontsize=8.5)
    fig.savefig(output / '01_roc_pr.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for offset, key in zip([-.24, 0, .24], NAMES):
        bars = axes[0].bar(folds.fold + offset, folds[key + '_auc'], width=.23, label=NAMES[key], color=COLORS[key])
        axes[0].bar_label(bars, fmt='%.3f', fontsize=8, padding=3, rotation=90)
    axes[0].set(title='AUC en cada fold', xticks=range(5), ylabel='ROC AUC', ylim=(.4, .9))
    axes[0].legend(fontsize=8.5)
    for _, history in histories.groupby(['seed', 'fold']):
        axes[1].plot(history.epoch, history.patient_auc, alpha=.35, lw=1, color=COLORS['new'])
    axes[1].plot(histories.groupby('epoch').patient_auc.median(), color=COLORS['new'], lw=2, label='Mediana validación')
    reviewed = histories.dropna(subset=['train_patient_auc'])
    axes[1].plot(reviewed.groupby('epoch').train_patient_auc.median(), 'o--', color=COLORS['previous'], label='Mediana train en revisión')
    axes[1].set(title='Evolución durante el entrenamiento', xlabel='Época', ylabel='ROC AUC por paciente', ylim=(.4, 1.02))
    axes[1].legend(fontsize=8.5)
    for ax in axes: ax.grid(axis='y', alpha=.2)
    fig.savefig(output / '02_folds_entrenamiento.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for col, scope in enumerate(('apparent', 'crossfit')):
        x = np.arange(3)
        selected = policies[policies.scope.eq(scope)].set_index('model').loc[list(NAMES)]
        axes[col].bar(x - .18, selected.sensitivity, width=.35, label='Sensibilidad', color='#26917a')
        axes[col].bar(x + .18, selected.specificity, width=.35, label='Especificidad', color='#567ba8')
        axes[col].axhline(.9, ls='--', color='#888888', lw=1)
        axes[col].set(title='Ajuste aparente al 90 %' if scope == 'apparent' else 'Umbral trasladado entre folds',
                      xticks=x, xticklabels=['Solo clínica', 'Anterior', 'Por paciente'], ylim=(0, 1.02))
        axes[col].grid(axis='y', alpha=.2); axes[col].legend(fontsize=9)
    fig.savefig(output / '03_sensibilidad_especificidad.png', dpi=180); plt.close(fig)


def analyze(args):
    runs, output = args.runs.resolve(), args.salida.resolve()
    fixed = json.loads((runs / 'ejecucion_fijada.json').read_text())
    if fixed['status'] != 'completo' or not fixed['protected_files_unchanged']:
        raise ValueError('La ejecución debe estar completa y verificada')
    for name in ('oof_clinico.csv', 'oof_weighted_median.csv'):
        path = PREVIOUS / name
        if e.sha256(path) != fixed['protected_files_sha256'][str(path.relative_to(ROOT))]:
            raise ValueError('Las predicciones de referencia cambiaron después del protocolo')
    for name, digest in fixed['data_metadata_sha256'].items():
        if e.sha256(ROOT / name) != digest: raise ValueError('Los metadatos de datos cambiaron')
    expected = {(seed, fold) for seed in (42, 2026) for fold in range(5)}
    folders, rows, histories = {}, [], []
    for path in sorted(runs.glob('patient_level_clinical/weighted/seed_*/fold_*/summary.json')):
        summary, config = json.loads(path.read_text()), json.loads((path.parent / 'config.json').read_text())
        if summary['status'] != 'complete' or not summary['eligible_for_selection'] or summary['smoke']:
            raise ValueError(f'Run no seleccionable: {path}')
        if summary['config_hash'] != e.config_hash(config): raise ValueError('Configuración modificada')
        if config['source_sha256']['04_entrenamiento.py'] != e.sha256(Path(e.__file__)):
            raise ValueError('El entrenador cambió después de ejecutar el run')
        if config['source_sha256']['pipeline_datos.py'] != e.sha256(Path(e.pdatos.__file__)):
            raise ValueError('El pipeline cambió después de ejecutar el run')
        if config['training'].get('loss_unit') != 'patient' or summary['loss'] != 'weighted':
            raise ValueError('El informe solo admite la nueva variante por paciente')
        for file, key in (('inference.pt', 'inference_sha256'), ('oof_slices.csv', 'oof_sha256')):
            if e.sha256(path.parent / file) != summary[key]: raise ValueError(f'Artefacto modificado: {file}')
        key = summary['seed'], summary['fold']
        if key in folders: raise ValueError('Run duplicado')
        folders[key] = path.parent
        rows.append({'seed': key[0], 'fold': key[1], 'epochs': summary['epochs_completed'], 'best_epoch': summary['best_epoch'],
                     'best_auc': summary['best_auc'], 'minutes': summary['total_train_seconds'] / 60,
                     'test_images_loaded': summary['test_images_loaded'], 'stop_reason': summary['stop_reason'],
                     'peak_vram_gib': summary['peak_vram_gib']})
        history = pd.read_csv(path.parent / 'history.csv'); history['seed'], history['fold'] = key
        histories.append(history)
    if set(folders) != expected: raise ValueError(f'Se requieren diez runs; faltan {expected - set(folders)}')
    output.mkdir(parents=True, exist_ok=True)
    run_frame, history_frame = pd.DataFrame(rows), pd.concat(histories, ignore_index=True)
    train = e.cargar_train(e.pdatos.DATOS, include_clinical=True)
    roster = train.rename(columns={'pCR': 'label', 'dataset': 'cohort'})
    seeds = [e.validate_oof(pd.concat([pd.read_csv(folders[seed, fold] / 'oof_slices.csv') for fold in range(5)], ignore_index=True), roster)
             for seed in (42, 2026)]
    if not seeds[0].sample_id.equals(seeds[1].sample_id): raise ValueError('Semillas no alineadas')
    averaged = seeds[0].copy(); averaged['probability'] = (seeds[0].probability + seeds[1].probability) / 2
    candidates = {method: e.aggregate_patients(averaged, method).sort_values('patient_id').reset_index(drop=True) for method in e.METHODS}
    selection = json.loads((runs / 'comparacion/seleccion.json').read_text())
    manifest_path = runs / 'comparacion/modelo_desarrollo.json'
    runtime = e.verificar_modelo_final(manifest_path)
    manifest = e.cargar_manifest(manifest_path)
    selected = candidates[selection['selected']['aggregation']]
    saved = pd.read_csv(runs / 'comparacion/oof_pacientes.csv').sort_values('patient_id').reset_index(drop=True)
    np.testing.assert_array_equal(selected.patient_id, saved.patient_id)
    np.testing.assert_allclose(selected.probability, saved.probability, atol=1e-12, rtol=0)
    frames = {'clinical': pd.read_csv(PREVIOUS / 'oof_clinico.csv'),
              'previous': pd.read_csv(PREVIOUS / 'oof_weighted_median.csv'), 'new': selected}
    for key, frame in frames.items():
        frame = frame.sort_values('patient_id').reset_index(drop=True); frames[key] = frame
        for column in ('patient_id', 'label', 'fold', 'cohort'):
            np.testing.assert_array_equal(frame[column], selected[column])
    old_config = ROOT / 'modelos/versiones/v002_multimodal_clinico_20261007/evidencia/repo/resultados/04_entrenamiento/multimodal_clinico_20261007/pool_dropout_wd_clinical/weighted/seed_42/fold_0/config.json'
    if json.loads((folders[42, 0] / 'config.json').read_text())['clinical_signature'] != json.loads(old_config.read_text())['clinical_signature']:
        raise ValueError('La clínica cambió respecto al modelo anterior')
    metrics = {name: metric_frame(e, frame) for name, frame in frames.items()}
    previous_metrics = json.loads((ROOT / 'modelos/versiones/v002_multimodal_clinico_20261007/version.json').read_text())['metrics_raw_oof']
    for key in ('roc_auc', 'mean_fold_roc_auc', 'average_precision'):
        np.testing.assert_allclose(metrics['previous'][key], previous_metrics[key], atol=1e-12)
    comparison = selected[['patient_id', 'label', 'fold', 'cohort']].copy()
    policy_rows, transfer = [], {}
    for name, frame in frames.items():
        comparison[name] = frame.probability
        calibration = e.fit_platt(frame.label.to_numpy(), frame.probability.to_numpy())
        calibrated = e.apply_calibration(frame.probability.to_numpy(), calibration)
        threshold = e.threshold_for_sensitivity(frame.label.to_numpy(), calibrated, .9)
        apparent = e.binary_metrics(frame.label, calibrated, threshold)
        if name == 'new':
            np.testing.assert_allclose(threshold, manifest['threshold'], atol=1e-12)
            np.testing.assert_allclose(calibrated, saved.calibrated_fit, atol=1e-12)
        probabilities, decisions, transferred, rules = crossfit_policy(frame)
        comparison[name + '_calibrated_crossfit'] = probabilities
        comparison[name + '_decision_crossfit'] = decisions
        transfer[name] = {'metrics': transferred, 'policies': rules}
        for scope, value in (('apparent', apparent), ('crossfit', transferred)):
            policy_rows.append({'model': name, 'scope': scope, **{key: value[key] for key in ('sensitivity', 'specificity', 'precision', 'f1', 'accuracy', 'fn', 'fp', 'tn', 'tp')}})
    policies = pd.DataFrame(policy_rows)
    fold_frame = pd.DataFrame([{'fold': fold, 'n': int(selected.fold.eq(fold).sum()),
                               **{name + '_auc': float(roc_auc_score(frame.loc[frame.fold.eq(fold), 'label'], frame.loc[frame.fold.eq(fold), 'probability'])) for name, frame in frames.items()}}
                              for fold in range(5)])
    candidates_frame = pd.DataFrame([{'aggregation': method, 'selected': method == manifest['aggregation'],
                                     **{key: value for key, value in metric_frame(e, frame).items() if key in ('roc_auc', 'mean_fold_roc_auc', 'min_fold_roc_auc', 'average_precision', 'brier')}}
                                    for method, frame in candidates.items()])
    bootstrap = paired_intervals(comparison, args.bootstrap)
    validation = bootstrap_check()
    image_branch = []
    for item in manifest['models']:
        weight = e.torch.load(manifest_path.parent / item['path'], map_location='cpu', weights_only=True)['state_dict']['classifier.weight'][0, :64]
        image_branch.append({'seed': item['seed'], 'fold': item['fold'], 'all_zero': bool((weight == 0).all()), 'max_abs': float(weight.abs().max())})
    final = history_frame.sort_values('epoch').groupby(['seed', 'fold']).tail(1)
    training = {'n_runs': 10, 'epochs': int(run_frame.epochs.sum()), 'minutes': float(run_frame.minutes.sum()),
                'wall_minutes': (datetime.fromisoformat(fixed['finished_at']) - datetime.fromisoformat(fixed['started_at'])).total_seconds() / 60,
                'best_epoch_zero': int(run_frame.best_epoch.eq(0).sum()), 'image_branch_zero': sum(item['all_zero'] for item in image_branch),
                'median_final_train_auc': float(final.train_patient_auc.median()),
                'median_final_validation_auc': float(final.patient_auc.median()),
                'median_final_auc_gap': float((final.train_patient_auc - final.patient_auc).median())}
    apparent = policies[policies.scope.eq('apparent')].set_index('model')
    delta_auc = metrics['new']['roc_auc'] - metrics['previous']['roc_auc']
    delta_fold = metrics['new']['mean_fold_roc_auc'] - metrics['previous']['mean_fold_roc_auc']
    delta_spec = float(apparent.loc['new', 'specificity'] - apparent.loc['previous', 'specificity'])
    useful = delta_auc > 0 and delta_fold > 0 and delta_spec >= 0
    decision = ('Conservar el candidato como una nueva versión de desarrollo, manteniendo la anterior.' if useful else
                'Mantener v002 como referencia. La nueva variante no mejora simultáneamente la AUC media, la AUC agrupada y la especificidad al mismo objetivo de sensibilidad; conservar sus resultados y checkpoints locales sin promoverla como sustituta.')
    summary = {'created_at': datetime.now(ZoneInfo('Europe/Madrid')).isoformat(), 'source_git_commit': fixed['source_git_commit'],
               'selected': selection['selected'], 'metrics_raw_oof': metrics, 'training': training,
               'clinical_availability_assumption': fixed['clinical_availability_assumption'], 'bootstrap': bootstrap,
               'crossfit_decision_policies': transfer, 'apparent_90_percent': policies[policies.scope.eq('apparent')].to_dict('records'),
               'manifest_local_path': str(manifest_path.relative_to(ROOT)), 'runtime_verification': runtime,
               'deltas_new_minus_previous': {'roc_auc': delta_auc, 'mean_fold_roc_auc': delta_fold, 'specificity_at_apparent_sensitivity_90': delta_spec},
               'versioning_recommended': useful, 'decision': decision,
               'limits': ['Checkpoints y agregación seleccionados con las mismas validaciones OOF; no validación anidada.',
                          'El 90 % aparente se ajusta con etiquetas OOF y no garantiza sensibilidad en pacientes nuevas.',
                          'El traslado de calibración y umbral entre folds no elimina el sesgo previo de selección de pesos.',
                          'El baseline clínico y su preprocesado se ajustan con una fila por paciente dentro del train del fold.',
                          'Train loss es BCE por paciente; val loss es BCE por corte y sus magnitudes no son directamente comparables.']}
    for name, frame in [('runs.csv', run_frame), ('historia_completa.csv', history_frame), ('comparacion_pareada.csv', comparison),
                        ('folds.csv', fold_frame), ('candidatos.csv', candidates_frame), ('politicas_decision.csv', policies)]:
        frame.to_csv(output / name, index=False)
    for method, frame in candidates.items(): frame.to_csv(output / f'oof_{method}.csv', index=False)
    e.json_write(output / 'resumen.json', summary)
    e.json_write(output / 'bootstrap_pareado.json', bootstrap)
    for folder in folders.values():
        target = output / 'evidencia' / folder.relative_to(runs)
        target.mkdir(parents=True, exist_ok=True)
        for name in ('config.json', 'summary.json', 'history.csv', 'environment.json', 'oof_slices.csv'):
            shutil.copy2(folder / name, target / name)
    shutil.copy2(runs / 'ejecucion_fijada.json', output / 'ejecucion_fijada.json')
    shutil.copy2(runs / 'comparacion/seleccion.json', output / 'seleccion.json')
    shutil.copy2(manifest_path, output / 'modelo_desarrollo_origen.json')
    audit = {'expected_runs': 10, 'complete_runs': len(run_frame), 'test_images_loaded': int(run_frame.test_images_loaded.sum()),
             'oof_patients': len(selected), 'oof_slices': len(train), 'protected_files_verified_at_training_end': len(fixed['protected_files_sha256']),
             'data_metadata_sha256': fixed['data_metadata_sha256'], 'runtime_verification': runtime,
             'bootstrap_implementation_check': validation, 'image_branch': image_branch,
             'source_sha256': {str(path.relative_to(ROOT)): e.sha256(path) for path in (Path(e.__file__), Path(e.pdatos.__file__), Path(__file__))}}
    e.json_write(output / 'auditoria.json', audit)
    plots(frames, fold_frame, history_frame, policies, output)
    write_report(summary, policies, candidates_frame, fold_frame, run_frame, output)
    inventory = {str(path.relative_to(output)): e.sha256(path) for path in sorted(output.rglob('*'))
                 if path.is_file() and path.name != 'archivos_sha256.json'}
    e.json_write(output / 'archivos_sha256.json', inventory)
    return summary


def write_report(summary, policies, candidates, folds, runs, output):
    metrics, training = summary['metrics_raw_oof'], summary['training']
    rows = pd.DataFrame([{'Modelo': NAMES[key], 'AUC media folds': value['mean_fold_roc_auc'], 'AUC OOF': value['roc_auc'],
                          'AP': value['average_precision'], 'Brier crudo': value['brier']} for key, value in metrics.items()])
    def policy_table(scope):
        frame = policies[policies.scope.eq(scope)].copy()
        frame['model'] = frame.model.map(NAMES)
        return table_md(frame[['model', 'sensitivity', 'specificity', 'precision', 'f1', 'fn', 'fp']])
    intervals = summary['bootstrap']['new_minus_reference']['previous']['roc_auc']['ci95']
    delta = summary['deltas_new_minus_previous']
    apparent = policies[policies.scope.eq('apparent')].set_index('model')
    fp_change = int(apparent.loc['new', 'fp'] - apparent.loc['previous', 'fp'])
    comparison_sentence = (f'A igual sensibilidad aparente, la diferencia de falsos positivos del nuevo frente al anterior es {fp_change:+d}.'
                           if np.isclose(apparent.loc['new', 'sensitivity'], apparent.loc['previous', 'sensitivity']) else
                           'Las sensibilidades aparentes difieren por los saltos de los umbrales; hay que comparar ambas junto con sus falsos positivos.')
    interval_sentence = ('El intervalo de la diferencia AUC frente al anterior incluye cero, de modo que no establece una diferencia concluyente entre estas predicciones.'
                         if intervals[0] <= 0 <= intervals[1] else
                         'El intervalo de la diferencia AUC frente al anterior no incluye cero; su alcance se limita a estas predicciones seleccionadas.')
    text = f'''# Resultados del entrenamiento multimodal por paciente

La variante por paciente obtiene AUC OOF {metrics['new']['roc_auc']:.4f}, frente a {metrics['previous']['roc_auc']:.4f} del multimodal anterior. La diferencia es {delta['roc_auc']:+.4f}; el intervalo descriptivo del 95 % es [{intervals[0]:+.4f}, {intervals[1]:+.4f}]. {summary['decision']}

## Protocolo y alcance

Se ejecutó únicamente `patient_level_clinical`, con BCE ponderada, semillas 42 y 2026 y cinco folds. Los diez runs usaron hasta 46 épocas, mínimo 12, paciencia 10, seis pacientes completos por lote, AdamW, dropout 0,35 y weight decay 0,001. El argumento `--lote 64` se usa para evaluación; el entrenamiento agrupa seis pacientes con todos sus cortes y tamaño variable. Las imágenes PRE, EARLY y LATE y las variables edad, volumen tumoral, HR y HER2 conservan la arquitectura de 551.921 parámetros del multimodal anterior.

La BCE conjunta usa la media de las probabilidades de los cortes de cada paciente y cada paciente pesa una vez por época. El peso de clase se calcula por paciente en el train de cada fold. La inicialización clínica y su preprocesado usan una fila por paciente dentro de ese train, igual que en el multimodal anterior. Se asume que las cuatro variables clínicas estarán disponibles en la prueba privada, según el criterio acordado para el proyecto.

La comparación cubre {metrics['new']['n_patients']} pacientes de desarrollo, {metrics['new']['n_positive']} con pCR y {summary['training']['n_runs']} runs. No se cargaron imágenes del test ni se generaron nuevas predicciones de test. Las imputaciones y escalas se ajustan solo en train, como exige la [validación cruzada con preprocesado separado](https://scikit-learn.org/stable/modules/cross_validation.html).

## Discriminación en desarrollo

{table_md(rows)}

La media por fold cambia {delta['mean_fold_roc_auc']:+.4f}. AP se compara con la prevalencia {metrics['new']['prevalence']:.4f}. Las predicciones de los modelos anteriores se reutilizan; no se repitieron sus entrenamientos. Las pacientes, etiquetas, folds y cohortes están alineadas para la comparación pareada. En {sum(auc < .70 for auc in metrics['new']['fold_roc_auc'])} de los cinco folds la nueva variante queda por debajo de AUC 0,70.

![ROC y precisión recall](01_roc_pr.png)

{table_md(folds)}

## Sensibilidad y falsos positivos al mismo objetivo

Cada modelo se calibra y se elige su mayor especificidad con sensibilidad aparente de al menos el 90 %. Esta tabla usa las mismas etiquetas OOF para ajustar y medir el umbral, por lo que describe desarrollo. La decisión original de v002 usaba Youden y no es directamente comparable con la nueva regla del 90 %.

{policy_table('apparent')}

La especificidad de la nueva variante frente a la anterior cambia {delta['specificity_at_apparent_sensitivity_90']:+.4f} al aplicar el mismo objetivo. {comparison_sentence} FN cuenta pacientes con pCR clasificadas como negativas; FP cuenta pacientes sin pCR clasificadas como positivas. Una sensibilidad alta obtenida moviendo el umbral debe juzgarse junto a su especificidad, como explica el [ajuste del umbral de decisión](https://scikit-learn.org/stable/modules/classification_threshold.html).

La siguiente comprobación ajusta calibración y umbral en cuatro folds OOF y los aplica al quinto. No usa directamente las etiquetas del fold destinatario para fijar su umbral, pero los pesos y la agregación ya fueron seleccionados con las validaciones originales; sigue sin ser una evaluación anidada completa.

{policy_table('crossfit')}

![Sensibilidad y especificidad](03_sensibilidad_especificidad.png)

## Entrenamiento y selección

Los diez runs completaron {training['epochs']} épocas en {training['minutes']:.1f} minutos de cómputo registrado y {training['wall_minutes']:.1f} minutos de tiempo total de ejecución, incluyendo inicializaciones, exportaciones y comparación. En {training['best_epoch_zero']} runs se conservó el baseline clínico de época cero; {training['image_branch_zero']} de los diez pesos seleccionados mantienen la contribución visual final a cero. Un peso visual distinto de cero no demuestra por sí solo que las imágenes aporten una mejora generalizable.

La última revisión registra medianas AUC train {training['median_final_train_auc']:.4f} y validación {training['median_final_validation_auc']:.4f}, con brecha mediana {training['median_final_auc_gap']:.4f}. La loss de train es BCE por paciente y la de validación se registra por corte; no se interpretan sus valores como una brecha homogénea.

![Folds y evolución del entrenamiento](02_folds_entrenamiento.png)

La mediana de cada época usa solo los runs que alcanzan esa época. La parada temprana reduce el número de runs disponibles y cambia su composición, por lo que una caída de la mediana tardía no demuestra por sí sola que todos los modelos hayan empeorado. Las líneas individuales y la tabla de runs permiten distinguir ambas situaciones.

El comparador promedia primero las semillas por corte y valora media, máximo y mediana por paciente. Selecciona por AUC media de los cinco folds, seguida de AUC agrupada, Brier y preferencia por media en empate. La agregación elegida es `{summary['selected']['aggregation']}`.

{table_md(candidates)}

{table_md(runs)}

## Interpretación y conservación

{summary['decision']}

La selección de épocas y agregación sobre estas validaciones introduce optimismo. Los {summary['bootstrap']['repetitions']} remuestreos pareados por paciente, estratificados por fold y etiqueta, describen diferencias entre predicciones fijas; no incluyen la variabilidad del entrenamiento ni el sesgo de selección. {interval_sentence} El 90 % aparente no garantiza esa sensibilidad en pacientes nuevas. La [calibración de probabilidades](https://scikit-learn.org/stable/modules/calibration.html) y la selección del umbral necesitan confirmación independiente.

Los pesos y checkpoints de trabajo permanecen en `resultados/04_entrenamiento/patient_level_clinical_20261007/`. `modelo_desarrollo_origen.json` conserva los metadatos del manifiesto local y sus rutas originales; no incluye pesos ni constituye un paquete portátil. El historial publicado en `modelos/versiones/` conserva las versiones previas.

## Reproducibilidad

El commit de partida es `{summary['source_git_commit']}`. `ejecucion_fijada.json` conserva el protocolo previo, el entorno y los hashes protegidos; `evidencia/` conserva configuración, resumen, historia y OOF originales de cada run. `resumen.json`, `politicas_decision.csv`, `bootstrap_pareado.json` y `archivos_sha256.json` permiten comprobar las cifras.

Con los datos y los pesos locales disponibles, regenerar con:

```bash
.venv/bin/python -B -m cancer_mama.informe_paciente
```
'''
    (output / 'INFORME_RESULTADOS.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=DEFAULT_RUNS)
    parser.add_argument('--salida', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--bootstrap', type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap < 100: parser.error('bootstrap debe ser al menos 100')
    summary = analyze(args)
    print(json.dumps({'selected': summary['selected']['candidate'], 'metrics': summary['metrics_raw_oof']['new'],
                      'decision': summary['decision']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
