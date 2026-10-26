"""Audita políticas de V007 sin entrenar ni abrir test/validación privada.

Las políticas se fijan solo con selección interna de cada pareja, antes de
consultar sus OOF exteriores archivadas. No recalibra el ensemble de diez
redes usando pacientes vistas por sus componentes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit
from sklearn.metrics import roc_auc_score, roc_curve
import torch

from . import entrenamiento as base
from . import entrenamiento_imagen as image
from . import cierre_cinetico as utils
from . import inferencia_imagen as inference
from .paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

OUTPUT = RESULTS_DIR/'29_auditoria_politica_v007_20261026'
SOURCE = RESULTS_DIR/'04_entrenamiento/ajuste_cnn_cinetica_20261021'
VERSION = PROJECT_ROOT/'modelos/versiones/v007_cnn_cinetica_multiescala_20261021'
RULES = ('macro_f1', 'specificity_90', 'specificity_95', 'raw_050', 'calibrated_050', 'always_no_pcr')
write, now = utils.write, utils.now


def inverse_threshold(calibration, threshold):
    if calibration['slope'] <= 0 or not 0 < threshold < 1:
        raise ValueError('Se requiere Platt estrictamente creciente y umbral interior')
    return float(expit((base.logit(np.array([threshold]))[0]-calibration['intercept'])/calibration['slope']))


def checked_selection(frame, expected, outer_fold):
    frame = frame.sort_values('patient_id').reset_index(drop=True)
    counts = expected.groupby('patient_id').size()
    expected = expected[['patient_id', 'pCR', 'dataset', 'fold']].drop_duplicates().sort_values('patient_id').reset_index(drop=True)
    if frame.patient_id.duplicated().any() or not frame.patient_id.equals(expected.patient_id):
        raise ValueError('Pacientes de selección incorrectas o duplicadas')
    for left, right in [('label', 'pCR'), ('cohort', 'dataset'), ('fold', 'fold')]:
        if not np.array_equal(frame[left], expected[right]): raise ValueError('Metadatos de selección inconsistentes')
    if outer_fold in set(frame.fold) or not frame.probability_raw.between(0, 1).all():
        raise ValueError('Exterior dentro de selección o puntuaciones inválidas')
    if not np.array_equal(frame.n_slices, frame.patient_id.map(counts)):
        raise ValueError('Número de cortes de selección incorrecto')
    return frame


def build_policies(selection):
    calibration = base.fit_platt(selection.label.to_numpy(), selection.probability_raw.to_numpy())
    calibration['fit_data'] = 'same_two_seed_pair_inner_selection_only'
    p = base.apply_calibration(selection.probability_raw.to_numpy(), calibration)
    return {'calibration': calibration, 'thresholds': {
        'macro_f1': image.threshold_macro_f1(selection.label, p),
        'specificity_90': image.threshold_min_specificity(selection.label, p, .90),
        'specificity_95': image.threshold_min_specificity(selection.label, p, .95),
        'raw_050': .5, 'calibrated_050': .5, 'always_no_pcr': None}}


def classify(frame, policy):
    p = frame.probability_raw.to_numpy()
    calibrated = base.apply_calibration(p, policy['calibration'])
    result = frame.copy(); result['probability_pair_calibrated'] = calibrated
    for rule in RULES:
        score = p if rule == 'raw_050' else calibrated
        result[f'prediction_{rule}'] = (score >= policy['thresholds'][rule]).astype(int) if rule != 'always_no_pcr' else 0
    return result


def decision_metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=int), np.asarray(prediction, dtype=int)
    if len(y) == 0 or len(y) != len(prediction) or not set(y).issubset({0, 1}) or not set(prediction).issubset({0, 1}):
        raise ValueError('Etiquetas o predicciones inválidas')
    tn, fp = int(((y == 0)&(prediction == 0)).sum()), int(((y == 0)&(prediction == 1)).sum())
    fn, tp = int(((y == 1)&(prediction == 0)).sum()), int(((y == 1)&(prediction == 1)).sum())
    div = lambda a, b: float(a/b) if b else 0.
    f0, f1 = div(2*tn, 2*tn+fp+fn), div(2*tp, 2*tp+fp+fn)
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'patients': len(y),
            'sensitivity': div(tp, tp+fn), 'specificity': div(tn, tn+fp),
            'precision_pcr': div(tp, tp+fp), 'f1_pcr': f1, 'f1_no_pcr': f0,
            'macro_f1': (f0+f1)/2, 'balanced_accuracy': (div(tp, tp+fn)+div(tn, tn+fp))/2,
            'accuracy': div(tp+tn, len(y))}


def paired_bootstrap(frame, n=2000):
    rng = np.random.default_rng(20261026)
    groups = [g.index.to_numpy() for _, g in frame.groupby(['fold', 'label'])]
    y = frame.label.to_numpy(); a = frame.prediction_macro_f1.to_numpy(); b = frame.prediction_specificity_90.to_numpy()
    values = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        ma, mb = decision_metrics(y[idx], a[idx]), decision_metrics(y[idx], b[idx])
        values.append([ma[k]-mb[k] for k in ('macro_f1', 'sensitivity', 'specificity')])
    return {'n': n, 'stratification': 'paciente pareada, fold×clase',
            'delta_macro_minus_spec90_ci95': {k: np.quantile(np.array(values)[:, i], [.025, .975]).tolist()
                for i, k in enumerate(('macro_f1', 'sensitivity', 'specificity'))},
            'limitation': 'Condicionado a estos modelos/pacientes; no incluye búsqueda histórica o toda variabilidad de entrenamiento'}


def prepare(output=OUTPUT):
    output = Path(output)
    if output.exists(): raise ValueError('Salida existente: no sobrescribir')
    manifest = inference.cargar_manifest(VERSION/'manifiesto.json')
    samples = base.cargar_train(DATA_DIR, include_clinical=False)
    frozen = json.loads((SOURCE/'seleccion_congelada.json').read_text())
    output.mkdir(parents=True); (output/'selection').mkdir()
    fits, selections, choices, records = {}, {}, [], []
    all_patients = set(samples.patient_id)
    files = [VERSION/'manifiesto.json', VERSION/'version.json', SOURCE/'seleccion_congelada.json',
             SOURCE/'oof_pacientes_selected.csv', DATA_DIR/'metadata/samples.csv', DATA_DIR/'metadata/patients.csv']
    for fold in frozen['folds']:
        k = fold['fold']; selected = fold['selected']; frames = []; model_entries = []
        for entry in selected['models']:
            path = SOURCE/entry['path']; cfg = json.loads((path/'config.json').read_text())
            if hashlib.sha256(samples.to_csv(index=False).encode()).hexdigest() != cfg['data_signature']:
                raise ValueError('Datos diferentes de la ejecución original')
            packaged = next(m for m in manifest['models'] if m['fold'] == k and m['seed'] == entry['seed'])
            if entry['sha256'] != packaged['sha256'] or base.sha256(path/'inference.pt') != packaged['sha256']:
                raise ValueError('Los pesos no son exactamente V007')
            if cfg['model'] != packaged['model_config'] or cfg['loss'] != selected['loss']:
                raise ValueError('Configuración no coincide con V007')
            fit, selection, outer = image.particion_anidada(samples, k, cfg)
            fits[k], selections[k] = set(fit.patient_id), set(selection.patient_id)
            frame = checked_selection(pd.read_csv(path/'selection_patients.csv'), selection, k)
            frames.append(frame)
            model_entries.append({'path': str((VERSION/packaged['path']).relative_to(PROJECT_ROOT)),
                                  'sha256': packaged['sha256'], 'seed': packaged['seed']})
            files += [path/n for n in ('config.json', 'summary.json', 'selection_patients.csv', 'inference.pt')]
        inner = image._aligned_average(frames)
        inner.to_csv(output/'selection'/f'fold_{k}.csv', index=False)
        policy = build_policies(inner)
        choices.append({'fold': k, 'loss': selected['loss'], 'trial': selected['trial'],
                        'models': model_entries, **policy})
        roles = []
        for role, frame in [('fit', fit), ('selection', selection), ('outer', outer)]:
            q = frame[['patient_id', 'pCR', 'dataset', 'fold']].drop_duplicates().copy(); q['role'] = role; roles.append(q)
        pd.concat(roles).sort_values('patient_id').to_csv(output/f'roles_fold_{k}.csv', index=False)
        records.append({'fold': k, 'loss': selected['loss'], 'trial': selected['trial'],
                        'selection_patients': len(inner), 'selection_raw_mean': float(inner.probability_raw.mean()),
                        'selection_raw_std': float(inner.probability_raw.std()),
                        'selection_auc': float(roc_auc_score(inner.label, inner.probability_raw))})
    availability = []
    for pid in sorted(all_patients):
        availability.append({'patient_id': pid, 'models_fit_on_patient': 2*sum(pid in fits[k] for k in range(5)),
                             'pairs_using_patient_in_selection': sum(pid in selections[k] for k in range(5))})
    table = pd.DataFrame(availability); table.to_csv(output/'disponibilidad_validacion.csv', index=False)
    common = table[table.models_fit_on_patient.eq(0)]
    sources = [Path(m.__file__) for m in (base, image, utils, inference)] + [Path(__file__)]
    files += sources + list(output.glob('roles_fold_*.csv')) + list((output/'selection').glob('*.csv'))
    protocol = {'prepared_at': now(), 'authorization': 'Auditar V007 y comparar políticas solo con validación',
        'unit': 'paciente, media de todos los cortes, pareja de dos semillas por fold',
        'training_started': False, 'model_weights_modified': False, 'primary_change': 'macroF1 máximo en selección interna',
        'baseline': 'Política original por pareja: sensibilidad máxima con especificidad>=90% en selección',
        'secondary_rules': ['especificidad95%', 'raw0.5', 'calibrada0.5', 'siempre no pCR'],
        'frozen_weights': 'Las diez redes V007; no se eligen nuevos checkpoints ni configuraciones',
        'policy_guard': 'Cinco calibraciones/umbrales fijados con selección interna antes de leer OOF para evaluar políticas',
        'deployment_guard': 'Estas políticas pertenecen a parejas de dos redes; NO son una calibración validada del ensemble de diez',
        'evaluation': 'OOF de1097 pacientes de train, históricamente observadas: desarrollo adaptativo, no confirmación nueva',
        'primary_comparison': 'MacroF1 macro-policy frente a spec90 con IC95% pareado; informar también sensibilidad/especificidad/FP/FN',
        'calibration_figure': 'Kernel gaussiano ancho0.06, dominio percentiles1–99, bandas exploratorias200 bootstrap fold/clase; evaluación descriptiva, no nueva calibración aplicada',
        'auc': 'Umbral no cambia AUC cruda; calibración creciente preserva AUC dentro de cada fold. Pooled calibrada es otra cantidad',
        'ten_model_reserve': {'patients_never_used_for_fit_any_network': len(common),
                              'patients_never_used_for_fit_or_selection_any_network': int(((table.models_fit_on_patient == 0)&(table.pairs_using_patient_in_selection == 0)).sum()),
                              'conclusion': 'No hay reserva etiquetada suficiente para validar una recalibración del ensemble completo'},
        'selection_summary': records,
        'sources': {str(f.relative_to(PROJECT_ROOT)): base.sha256(f) for f in files},
        'versions_before': utils.previous_versions(), 'test_images_loaded': 0, 'private_images_loaded': 0,
        'limits': ['No usar el test histórico para escoger umbral ni volver a medir sus resultados',
                   'Selection se usó para época/hiperparámetros; sus métricas son optimistas',
                   'No atribuir los0TP históricos exclusivamente a la transferencia2→10; no se prueba causalidad',
                   'No inferir con diez redes sobre train para ajustar política: la mayoría vio esas pacientes',
                   'No hay entrenamiento ni activación ni modificación de versiones ni push en esta auditoría']}
    write(output/'protocolo.json', protocol)
    (output/'protocolo.json.sha256').write_text(base.sha256(output/'protocolo.json')+'\n')
    write(output/'politicas_congeladas.json', {'frozen_at': now(), 'deployable': False,
        'scope': 'five_outer_fold_two_seed_pairs_only', 'folds': choices, 'protocol_sha256': base.sha256(output/'protocolo.json')})
    (output/'politicas_congeladas.json.sha256').write_text(base.sha256(output/'politicas_congeladas.json')+'\n')
    print(json.dumps({'status': 'prepared_policies_frozen', 'folds': 5, 'ten_model_reserve': protocol['ten_model_reserve']}), flush=True)
    return protocol


def verify(output):
    output = Path(output)
    for name in ('protocolo.json', 'politicas_congeladas.json'):
        if base.sha256(output/name) != (output/(name+'.sha256')).read_text().strip(): raise ValueError('Protocolo/políticas modificados')
    protocol = json.loads((output/'protocolo.json').read_text())
    for name, digest in {**protocol['sources'], **protocol['versions_before']}.items():
        if base.sha256(PROJECT_ROOT/name) != digest: raise ValueError(f'Fuente o versión modificada:{name}')
    return protocol


def synthetic_runtime_check():
    torch.set_num_threads(4)
    manifest_path = VERSION/'manifiesto.json'; manifest = inference.cargar_manifest(manifest_path)
    yy, xx = np.mgrid[:256, :256]; arrays = []
    with tempfile.TemporaryDirectory(prefix='v007_policy_synthetic_', dir=PROJECT_ROOT/'.codex_tmp') as tmp:
        triples = []
        for i in range(3):
            mask = (xx-(100+i*12))**2+(yy-128)**2 < (25+i*3)**2
            phases = [np.full((256, 256), 40, dtype=np.uint8) for _ in range(3)]
            phases[1][mask] = 140; phases[2][mask] = 95
            arrays.append(np.stack(phases)); triple = []
            for phase, array in zip(('PRE', 'EARLY', 'LATE'), phases):
                path = Path(tmp)/f'AUDIT_SYNTHETIC_z{i+1:03d}_{phase}.png'
                Image.fromarray(array).save(path); triple.append(path)
            triples.append(triple)
        actual = inference.predecir_cortes(manifest_path, triples)
        x = torch.tensor(np.stack(arrays), dtype=torch.float32)/255; probabilities = []
        for item in manifest['models']:
            model = inference.cargar_modelo(manifest_path, item)
            with torch.inference_mode(): probabilities.append(model(x).sigmoid().flatten().numpy())
        raw = float(np.mean(probabilities, axis=0).mean())
        calibrated = float(base.apply_calibration(np.array([raw]), manifest['calibration'])[0])
        if abs(actual['probability_raw']-raw) > 1e-7 or abs(actual['probability_calibrated']-calibrated) > 1e-7:
            raise RuntimeError('La inferencia no reproduce media+calibración del manifiesto')
    return {'synthetic_only': True, 'dataset_images_loaded': 0, 'n_slices': 3, 'models': 10,
            'manual_raw': raw, 'runtime_raw': actual['probability_raw'], 'calibration_agrees': True,
            'phase_scaling_and_mean_implementation_verified': True, 'not_a_performance_evaluation': True}


def analyze(output=OUTPUT):
    output = Path(output); protocol = verify(output)
    if (output/'metricas.json').exists(): raise ValueError('Auditoría ya ejecutada: no volver a ajustar')
    frozen = json.loads((output/'politicas_congeladas.json').read_text())
    oof = pd.read_csv(SOURCE/'oof_pacientes_selected.csv').sort_values('patient_id').reset_index(drop=True)
    samples = base.cargar_train(DATA_DIR, include_clinical=False)
    expected = samples[['patient_id', 'pCR', 'dataset', 'fold']].drop_duplicates().sort_values('patient_id').reset_index(drop=True)
    if len(oof) != 1097 or oof.patient_id.duplicated().any() or not oof.patient_id.equals(expected.patient_id):
        raise ValueError('OOF incompletas o pacientes incorrectas')
    for a, b in [('label', 'pCR'), ('cohort', 'dataset'), ('fold', 'fold')]:
        if not np.array_equal(oof[a], expected[b]): raise ValueError('OOF no corresponden a train')
    if not np.array_equal(oof.n_slices, oof.patient_id.map(samples.groupby('patient_id').size())):
        raise ValueError('OOF no conserva todos los cortes por paciente')
    frames = []
    for policy in frozen['folds']:
        frame = oof[oof.fold.eq(policy['fold'])].copy()
        classified = classify(frame, policy)
        # Reproducir la política exterior histórica por pareja es un control obligatorio.
        if not np.array_equal(classified.prediction_specificity_90, frame.prediction):
            raise RuntimeError('No reproduce decisiones históricas por pareja')
        np.testing.assert_allclose(classified.probability_pair_calibrated, frame.probability_calibrated, atol=1e-12, rtol=0)
        frames.append(classified)
    frame = pd.concat(frames).sort_values('patient_id').reset_index(drop=True)
    frame.to_csv(output/'comparacion_pacientes.csv', index=False)
    manifest = inference.cargar_manifest(VERSION/'manifiesto.json')
    global_fit = base.fit_platt(frame.label.to_numpy(), frame.probability_raw.to_numpy())
    np.testing.assert_allclose([global_fit['slope'], global_fit['intercept']],
                               [manifest['calibration']['slope'], manifest['calibration']['intercept']], atol=1e-9, rtol=0)
    global_probability = base.apply_calibration(frame.probability_raw.to_numpy(), global_fit)
    candidates = {'specificity_90': image.threshold_min_specificity(frame.label, global_probability, .90),
                  'specificity_95': image.threshold_min_specificity(frame.label, global_probability, .95),
                  'macro_f1': image.threshold_macro_f1(frame.label, global_probability),
                  'youden': base.youden_threshold(frame.label.to_numpy(), global_probability)}
    threshold_audit = {'archived_calibration': manifest['calibration'], 'calibration_reconstructed': True,
        'archived_threshold': manifest['threshold'], 'equivalent_raw_threshold': inverse_threshold(manifest['calibration'], manifest['threshold']),
        'threshold_reconstruction_on_development_oof_only': candidates,
        'matching_strategies': [k for k, v in candidates.items() if abs(v-manifest['threshold']) < 1e-10],
        'apparent_not_validation': decision_metrics(frame.label, global_probability >= manifest['threshold']),
        'warning': 'Calibración/umbral globales se ajustaron con estas mismas OOF: sus decisiones aquí son aparentes, nunca evaluación independiente'}
    scales = []
    for fold, g in frame.groupby('fold'):
        choice = next(x for x in frozen['folds'] if x['fold'] == fold)
        scales.append({'fold': int(fold), 'loss': choice['loss'], 'trial': choice['trial'], 'patients': len(g),
                       'raw_mean': float(g.probability_raw.mean()), 'raw_sd': float(g.probability_raw.std()),
                       'raw_q10_q50_q90': np.quantile(g.probability_raw, [.1, .5, .9]).tolist(),
                       'fraction_above_archived_equivalent_raw_threshold': float(g.probability_raw.ge(threshold_audit['equivalent_raw_threshold']).mean()),
                       'pair_calibrated_mean': float(g.probability_pair_calibrated.mean()),
                       'calibration': choice['calibration']})
    policies = {r: decision_metrics(frame.label, frame[f'prediction_{r}']) for r in RULES}
    quality = {}
    for name, probabilities in [('raw_pair_oof', frame.probability_raw.to_numpy()),
                               ('calibrated_pair_oof', frame.probability_pair_calibrated.to_numpy()),
                               ('global_fit_apparent_only', global_probability)]:
        probabilities = np.clip(probabilities, 1e-7, 1-1e-7); y = frame.label.to_numpy()
        quality[name] = {'brier': float(np.mean((probabilities-y)**2)),
                         'logloss': float(np.mean(-y*np.log(probabilities)-(1-y)*np.log1p(-probabilities))),
                         'mean_prediction': float(probabilities.mean()), 'prevalence': float(y.mean())}
    fold_auc = [float(roc_auc_score(g.label, g.probability_raw)) for _, g in frame.groupby('fold')]
    calibrated_fold_auc = [float(roc_auc_score(g.label, g.probability_pair_calibrated)) for _, g in frame.groupby('fold')]
    np.testing.assert_allclose(fold_auc, calibrated_fold_auc, atol=1e-14, rtol=0)
    result = {'analyzed_at': now(), 'scope': 'Solo train1097, OOF históricas, parejas de dos redes; no validación del ensemble10',
        'metrics': policies, 'auc_raw_unchanged_by_threshold': float(roc_auc_score(frame.label, frame.probability_raw)),
        'mean_fold_raw_auc_unchanged': float(np.mean(fold_auc)), 'sd_fold_auc': float(np.std(fold_auc, ddof=1)),
        'pooled_pair_calibrated_auc_secondary': float(roc_auc_score(frame.label, frame.probability_pair_calibrated)),
        'fold_auc': fold_auc, 'bootstrap': paired_bootstrap(frame), 'threshold_audit': threshold_audit, 'score_scales': scales,
        'probability_quality': quality,
        'by_cohort': {c: {r: decision_metrics(g.label, g[f'prediction_{r}']) for r in RULES} for c, g in frame.groupby('cohort')},
        'synthetic_runtime_check': synthetic_runtime_check(), 'test_images_loaded': 0, 'private_images_loaded': 0,
        'training_started': False, 'deployable': False,
        'conclusion': 'Las políticas de pares no son directamente portables al ensemble10; no hay reserva común suficiente para validarlo con estos pesos'}
    write(output/'metricas.json', result)
    report(output, result, frame)
    verify(output)
    write(output/'verificacion_final.json', {'verified_at': now(), 'versions_unchanged': True, 'original_weights_unchanged': True,
        'same_oof_scores': True, 'historical_pair_decisions_reproduced': True, 'policies_frozen_before_evaluation': True,
        'test_images_loaded': 0, 'private_images_loaded': 0, 'no_training': True, 'no_push': True,
        'files_sha256': {str(f.relative_to(output)): base.sha256(f) for f in output.rglob('*') if f.is_file() and f.name != 'verificacion_final.json'}})
    print(json.dumps({'metrics': policies, 'auc': result['mean_fold_raw_auc_unchanged'], 'bootstrap': result['bootstrap']}), flush=True)
    return result


def report(output, result, frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    protocol = json.loads((output/'protocolo.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'normal': '#176b91', 'weighted': '#d0691c'}
    for item in result['score_scales']:
        group = frame[frame.fold.eq(item['fold'])]
        axes[0].hist(group.probability_raw, bins=np.linspace(0, 1, 31), histtype='step', density=True,
                     label=f"Fold{item['fold']} {item['loss']}", color=colors[item['loss']], alpha=.8)
    axes[0].axvline(result['threshold_audit']['equivalent_raw_threshold'], color='black', linestyle=':', label='Umbral global equivalente')
    axes[0].set(title='Puntuaciones OOF de parejas: escalas distintas', xlabel='Probabilidad cruda', ylabel='Densidad'); axes[0].legend(fontsize=8)
    vals = result['metrics']; rules = list(RULES); x = np.arange(len(rules))
    axes[1].plot(x, [vals[r]['sensitivity'] for r in rules], 'o-', label='Sensibilidad')
    axes[1].plot(x, [vals[r]['specificity'] for r in rules], 'o-', label='Especificidad')
    axes[1].plot(x, [vals[r]['macro_f1'] for r in rules], 'o-', label='MacroF1')
    axes[1].set(xticks=x, xticklabels=['MacroF1', 'Esp90', 'Esp95', 'Cruda.5', 'Cal.5', 'No pCR'], ylim=(0, 1), title='Políticas elegidas en selección, evaluadas en OOF'); axes[1].tick_params(axis='x', rotation=35); axes[1].legend()
    for label, column in [('Cruda OOF, secundaria', 'probability_raw'), ('Calibrada por pareja, secundaria', 'probability_pair_calibrated')]:
        fpr, tpr, _ = roc_curve(frame.label, frame[column]); axes[2].plot(fpr, tpr, label=label)
    axes[2].plot([0, 1], [0, 1], ':', color='grey'); axes[2].set(title='Umbral no cambia la ROC de una puntuación', xlabel='1−Especificidad', ylabel='Sensibilidad'); axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/'01_politicas_y_escalas.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, rule in zip(axes, ('specificity_90', 'macro_f1', 'specificity_95')):
        m = vals[rule]; matrix = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
        ax.imshow(matrix, cmap='Blues'); ax.set(title=rule, xticks=[0, 1], xticklabels=['No pCR', 'pCR'], yticks=[0, 1], yticklabels=['No pCR', 'pCR'], xlabel='Predicción', ylabel='Real')
        for (i, j), value in np.ndenumerate(matrix): ax.text(j, i, str(value), ha='center', va='center', color='black')
    fig.suptitle('Solo desarrollo por parejas; no nuevos resultados de test'); fig.tight_layout(); fig.savefig(output/'02_matrices.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    groups = [g.index.to_numpy() for _, g in frame.groupby(['fold', 'label'])]
    for ax, column, title in zip(axes, ['probability_raw', 'probability_pair_calibrated'], ['Cruda por pareja', 'Calibrada en selección de la misma pareja']):
        x = frame[column].to_numpy(); y = frame.label.to_numpy()
        grid = np.linspace(*np.quantile(x, [.01, .99]), 80)
        weights = np.exp(-.5*((grid[:, None]-x[None, :])/.06)**2)
        smooth = (weights@y)/weights.sum(1)
        rng = np.random.default_rng(20261026); curves = []
        for _ in range(200):
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            counts = np.bincount(idx, minlength=len(x))
            curves.append((weights@(y*counts))/(weights@counts))
        low, high = np.quantile(curves, [.025, .975], axis=0)
        ax.plot([0, 1], [0, 1], ':', color='grey'); ax.plot(grid, smooth, label='Curva descriptiva, kernel0.06')
        ax.fill_between(grid, low, high, alpha=.2, label='Banda bootstrap exploratoria')
        ax.plot(x, np.full(len(x), -.015), '|', color='black', alpha=.06)
        ax.set(title=title, xlabel='Probabilidad predicha', ylabel='Frecuencia pCR suavizada', xlim=(0, 1), ylim=(-.03, 1)); ax.legend(fontsize=8)
    fig.suptitle('Solo OOF de parejas; bandas no incluyen búsqueda/entrenamiento'); fig.tight_layout(); fig.savefig(output/'03_calibracion_parejas.png', dpi=140); plt.close(fig)
    rows = ['| Política | Sensibilidad | Especificidad | F1pCR | MacroF1 | FP | FN |', '|---|---:|---:|---:|---:|---:|---:|']
    for rule in RULES:
        m = vals[rule]; rows.append(f"|{rule}|{m['sensitivity']:.2%}|{m['specificity']:.2%}|{m['f1_pcr']:.4f}|{m['macro_f1']:.4f}|{m['fp']}|{m['fn']}|")
    delta = result['bootstrap']['delta_macro_minus_spec90_ci95']; audit = result['threshold_audit']
    text = f"""# Auditoría de calibración y política de V007

26/10/2026. Sin entrenamiento ni inferencia de pacientes; solo metadatos de train, selección y OOF históricas. No se abren test ni validación privada. Los diez pesos y todas las versiones permanecen intactos. La inferencia se comprueba aparte con PNG sintéticos.

## Qué está confirmado

V007 despliega la media de diez redes, pero ajustó calibración/umbral global sobre OOF de parejas de dos redes. Además, las parejas seleccionadas mezclan BCE normal/ponderada y configuraciones distintas. Las puntuaciones crudas cambian de escala entre folds. La implementación de PNG/255, media de todos los cortes/modelos y calibración coincide con el manifiesto en una prueba sintética; no se ha detectado un error aritmético de inferencia.

La calibración archivada se reproduce desde sus OOF: pendiente{audit['archived_calibration']['slope']:.6f}, intercepto{audit['archived_calibration']['intercept']:.6f}; umbral calibrado{audit['archived_threshold']:.6f}, equivalente crudo{audit['equivalent_raw_threshold']:.6f}. Estrategias que reproducen ese umbral en desarrollo:{audit['matching_strategies']}. La etiqueta del manifiesto test_evaluated_once=False es una instantánea archivada previa a la evaluación histórica ya documentada; no autoriza repetir test y no se modifica esa versión.

Esto confirma una diferencia entre cómo se obtuvieron las puntuaciones de calibración y las puntuaciones de despliegue, no demuestra que cause por sí sola la sensibilidad nula observada históricamente. Tampoco elimina la débil discriminación ni el sobreajuste de la red.

## Comparación con las mismas parejas y pacientes

Cinco calibraciones y umbrales se fijaron con selección interna de cada pareja antes de evaluar sus OOF archivadas. La política principal nueva maximiza macroF1 en selección; la referencia reproduce exactamente la política original por pareja con especificidad objetivo90%. La política95% y los umbrales0,5 son secundarios prefijados, no se escoge la política más favorable después de ver exterior. Selection ya eligió checkpoints/configuraciones: tiene optimismo; las OOF tienen historia adaptativa y no son una nueva confirmación independiente.

{chr(10).join(rows)}

Delta macro-policy−spec90, IC95% pareado: macroF1{delta['macro_f1']}; sensibilidad{delta['sensitivity']}; especificidad{delta['specificity']}. Bootstrap2000 por fold/clase, condicionado a estos modelos. Ganar sensibilidad puede incrementar falsos positivos; macroF1 es un criterio académico, no una preferencia clínica validada.

AUC cruda por paciente **sin cambio**: media de folds{result['mean_fold_raw_auc_unchanged']:.6f} ±{result['sd_fold_auc']:.6f}, agrupada{result['auc_raw_unchanged_by_threshold']:.6f}. La calibración creciente conserva AUC dentro de cada fold; agrupada calibrada{result['pooled_pair_calibrated_auc_secondary']:.6f} es un análisis secundario de puntuaciones diferentes, no mejora de discriminación dentro de cada modelo.

![Escalas y políticas](01_politicas_y_escalas.png)
![Matrices](02_matrices.png)
![Calibración por parejas](03_calibracion_parejas.png)

Calidad probabilística: {result['probability_quality']}. Brier/logloss mezclan calibración y discriminación, no prueban por sí solos buena calibración. Curvas de evaluación con kernel gaussiano de ancho0,06 prefijado y dominio percentiles1–99; bandas exploratorias200 bootstrap por fold/clase, condicionadas a estas puntuaciones y sujetas al sesgo del suavizado. No se utilizan para recalibrar ningún modelo ni para estimar la calibración del ensemble10.

## Qué se puede utilizar

politicas_congeladas.json pertenece únicamente al procedimiento de cinco parejas y está marcado deployable=False. No sustituye la calibración/umbral del ensemble completo ni se conecta a la app. No se calculan nuevas métricas de test con ningún umbral.

Intersección de pacientes nunca usadas para fit por ninguna red:{protocol['ten_model_reserve']['patients_never_used_for_fit_any_network']}; nunca usadas para fit ni selección:{protocol['ten_model_reserve']['patients_never_used_for_fit_or_selection_any_network']}. No hay una reserva suficiente para validar con estos pesos una recalibración del ensemble completo. No se predice con las diez redes sobre train para presentar resultados contaminados como validación.

Para un paquete futuro coherente habría que fijar una pareja y reservar pacientes para su propia política antes de entrenar, o disponer de nuevas pacientes etiquetadas permitidas y no usadas para desarrollo. El diseño932fit/165selección-pólítica ya está preparado, pero no se ejecuta: no hemos iniciado otro entrenamiento ni prometido conservar el AUC histórico de V007. La confirmación final corresponde al profesor con evaluación independiente permitida. No se modifican versiones ni aplicaciones ni se hace push.

[Umbrales, ROC y validación](https://scikit-learn.org/stable/modules/classification_threshold.html). [Calibración y ensembles](https://scikit-learn.org/stable/modules/calibration.html). Los indicadores Brier/logloss no aíslan por sí solos la calibración. Las decisiones y AUC se reportan por separado.
"""
    (output/'INFORME_AUDITORIA.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preparar', 'analizar'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.action == 'preparar': prepare(args.output)
    else: analyze(args.output)


if __name__ == '__main__': main()
