"""Consolida la repetición y prepara el candidato solo si confirma el criterio."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from . import entrenamiento as base
from . import confirma_bce_normal as confirmation
from . import experimento_bce_mixup as experiment
from . import informe_loss as diagnosis
from .paths import PROJECT_ROOT, RESULTS_DIR

OUTPUT = RESULTS_DIR/'11_confirmacion_bce_normal_20261011'


def verify_metrics(frame, expected):
    actual = diagnosis.metrics(frame)
    for key in ('roc_auc', 'mean_fold_auc', 'average_precision', 'log_loss', 'brier'):
        if not np.isclose(actual[key], expected[key], rtol=1e-10, atol=1e-12):
            raise ValueError(f'Métrica {key} no coincide con las OOF')
    return actual


def main():
    source = confirmation.OUTPUT
    state = json.loads((source/'estado.json').read_text())
    if state['status'] != 'complete' or state['smoke'] or not state.get('protected_unchanged'):
        raise ValueError('La repetición aún no está completa y verificada')
    report = json.loads((source/'comparacion/resumen.json').read_text())
    first = json.loads((experiment.DEFAULT_OUTPUT/'comparacion/resumen.json').read_text())
    protocol = json.loads((source/'protocolo.json').read_text())
    if base.config_hash(protocol['parameters']) != protocol['fingerprint']:
        raise ValueError('Protocolo alterado')
    if base.sha256(experiment.DEFAULT_OUTPUT/'comparacion/resumen.json') != protocol['parameters']['original_search_result_sha256']:
        raise ValueError('Resultado de búsqueda cambiado')
    for path, digest in protocol['parameters']['protected'].items():
        if base.sha256(PROJECT_ROOT/path) != digest:
            raise ValueError('Archivo protegido cambiado antes de consolidar')
    evidence = json.loads((source/'comparacion/entradas_sha256.json').read_text())
    for path, digest in evidence.items():
        if base.sha256(PROJECT_ROOT/path) != digest:
            raise ValueError('Evidencia de repetición modificada')
    agreed = confirmation.decision(report['intervals_99_58'], report['delta_vs_control'], report['seed_deltas'])
    if agreed != report['replicated_internal_result']:
        raise ValueError('Decisión inconsistente con las métricas')
    agreed = agreed and first['comparisons']['bce_normal']['promising_internal_result']
    for row in report['metrics']:
        tag = str(row['seeds'][0]) if len(row['seeds']) == 1 else 'ensemble'
        verify_metrics(pd.read_csv(source/f"comparacion/oof_{row['candidate']}_{tag}.csv"), row)
    initial = next(row for row in first['metrics'] if row['candidate'] == 'bce_normal')
    original_oof = pd.read_csv(experiment.DEFAULT_OUTPUT/'comparacion/oof_raw_bce_normal.csv')
    verify_metrics(original_oof, initial)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for pattern in ('*.csv', '*.json'):
        for path in (source/'comparacion').glob(pattern):
            shutil.copy2(path, OUTPUT/path.name)
    for name in ('protocolo.json', 'estado.json'):
        shutil.copy2(source/name, OUTPUT/name)

    models, checkpoints = [], []
    for seed in experiment.SEEDS:
        for fold in range(5):
            folder = experiment.DEFAULT_OUTPUT/f'bce_normal/seed_{seed}/fold_{fold}'
            config = json.loads((folder/'config.json').read_text())
            summary = json.loads((folder/'summary.json').read_text())
            if config['smoke'] or summary['config_hash'] != base.config_hash(config):
                raise ValueError('Candidata de prueba o con configuración alterada')
            for name, digest in summary['files'].items():
                if base.sha256(folder/name) != digest:
                    raise ValueError('Candidata modificada')
                evidence[str((folder/name).relative_to(PROJECT_ROOT))] = digest
            payload = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
            if payload['model_config'] != config['model']:
                raise ValueError('Configuración de inferencia inconsistente')
            weight = folder/'best.pt'
            models.append({'seed': seed, 'fold': fold, 'path': os.path.relpath(weight, OUTPUT),
                           'sha256': base.sha256(weight), 'model_config': payload['model_config']})
            checkpoints.append({'stage': 'search', 'candidate': 'normal', 'seed': seed, 'fold': fold,
                'best_epoch': summary['tracker']['best_epoch'],
                'visual_weight_norm': float(payload['state_dict']['classifier.weight'][0, :config['model']['hidden']].norm())})
    # Las curvas y metadatos de repetición se conservan sin duplicar pesos pesados.
    for seed in confirmation.SEEDS:
        for fold in range(5):
            for candidate, relative in (('normal', 'candidate/bce_normal'), ('weighted', 'control/lr_actual')):
                folder = source/f'{relative}/seed_{seed}/fold_{fold}'
                summary = json.loads((folder/'summary.json').read_text())
                target = OUTPUT/f'{candidate}/seed_{seed}_fold_{fold}'
                target.mkdir(parents=True, exist_ok=True)
                for name in ('config.json', 'summary.json', 'history.csv', 'environment.json'):
                    shutil.copy2(folder/name, target/name)
                tracker = summary['tracker'] if candidate == 'normal' else summary['results']['auc']['selection']
                checkpoints.append({'stage': 'repeat', 'candidate': candidate, 'seed': seed, 'fold': fold,
                                    'best_epoch': tracker['best_epoch'], 'visual_weight_norm': None})
    pd.DataFrame(checkpoints).to_csv(OUTPUT/'checkpoints.csv', index=False)
    base.json_write(OUTPUT/'entradas_sha256.json', evidence)

    os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT/'.codex_tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    comparisons = [('Búsqueda · semillas 42/2026', first['comparisons']['bce_normal']),
                   ('Repetición · semillas 73/1701', report)]
    for ax, key, title in zip(axes, ('log_loss', 'mean_fold_auc'), ('Cambio de pérdida · menor es mejor', 'Cambio de AUC media · mayor es mejor')):
        for y, (_, comparison) in enumerate(comparisons):
            ci = comparison['intervals_99_58'][key]
            ax.plot([ci['lower'], ci['upper']], [y, y], color='#267b89', linewidth=2)
            ax.scatter([comparison['delta_vs_control'][key]], [y], color='#267b89', s=45, zorder=3)
        ax.set_yticks(range(2), [name for name, _ in comparisons])
        ax.set_ylim(-.5, 1.5)
        ax.axvline(0, color='#555', linestyle='--')
        ax.set_title(title)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Repetición del cambio a BCE normal · misma arquitectura y mismas pacientes · IC 99,58%')
    fig.savefig(OUTPUT/'comparacion_repeticion.png', dpi=150)
    plt.close(fig)

    lines = ['# Repetición de la mejora con BCE normal', '',
        'La reducción de pérdida se reproduce según el criterio predefinido.' if agreed else
        'La repetición no cumple todos los criterios predefinidos. No se declara confirmada la mejora.', '',
        'Se volvieron a entrenar candidata y control con semillas 73 y 1701 en los cinco folds. '
        'No se modificaron hiperparámetros, particiones ni decisiones de calibración. '
        'Es repetición sobre el mismo desarrollo, no validación externa.', '',
        '| Semillas | Modelo | AUC media folds | AUC OOF | Log-loss | Brier |',
        '|---|---|---:|---:|---:|---:|']
    for row in report['metrics']:
        lines.append(f"| {'/'.join(map(str,row['seeds']))} | {row['candidate']} | {row['mean_fold_auc']:.4f} | {row['roc_auc']:.4f} | {row['log_loss']:.4f} | {row['brier']:.4f} |")
    lines += ['', '![Repetición](comparacion_repeticion.png)', '',
        'El control principal es BCE ponderada con selección por AUC y Platt interno. '
        'El control seleccionado por pérdida se incluye como diagnóstico. La candidata usa BCE normal, '
        'inicialización clínica no ponderada, selección por pérdida y probabilidades sin calibración posterior.', '',
        '## Qué demuestra y qué queda abierto', '',
        'La búsqueda original obtuvo log-loss 0,5463 frente a 0,5615 y AUC media 0,7107 frente a 0,7047. '
        'La mejora principal es probabilística; no se afirma un aumento concluyente de AUC. '
        'La AUC histórica 0,7246 de v002 procede de otro protocolo y no debe compararse directamente con estas cifras.', '',
        'Siete de los diez checkpoints originales seleccionan época 0, equivalente al inicializador clínico. '
        'Por tanto, las curvas aún muestran sobreajuste y no se ha probado que la rama visual aporte una mejora independiente. '
        'Se conserva el mejor checkpoint, no el último.', '',
        'Las pacientes exteriores de cada fold no se usan para elegir épocas ni ajustar calibración/umbral. '
        'Sin embargo, la investigación previa es adaptativa y se observa el mismo desarrollo repetidamente. '
        'Cambiar semillas comprueba estabilidad de entrenamiento, no generalización a otras pacientes o centros. '
        'El test reservado y la validación privada siguen sin abrirse.', '']
    if agreed:
        threshold = base.youden_threshold(original_oof.label.to_numpy(), original_oof.probability.to_numpy())
        calibration = {'method': 'identity', 'slope': 1., 'intercept': 0., 'clip_epsilon': base.EPS,
                       'fit_data': 'none; raw probabilities from unweighted BCE'}
        metrics = diagnosis.metrics(original_oof)
        provenance = []
        for path in (source/'protocolo.json', source/'comparacion/resumen.json',
                     experiment.DEFAULT_OUTPUT/'protocolo.json', experiment.DEFAULT_OUTPUT/'comparacion/resumen.json',
                     experiment.DEFAULT_OUTPUT/'comparacion/oof_raw_bce_normal.csv'):
            provenance.append({'path': os.path.relpath(path, OUTPUT), 'sha256': base.sha256(path)})
        manifest = {'schema_version': 1, 'status': 'mejora_interna_replicada_sin_test',
            'models': models, 'aggregation': 'median', 'calibration': calibration, 'threshold': threshold,
            'decision_policy': {'strategy': 'youden', 'fit_data': 'original_outer_oof_apparent_for_deployment_only'},
            'test_evaluated_once': False,
            'training': {'loss': 'normal', 'criterion': 'BCEWithLogitsLoss', 'pos_weight': None,
                         'checkpoint_metric': 'inner_patient_logloss', 'lr': .0008, 'max_epochs': 46,
                         'batch_size': 64, 'weight_decay': .001, 'dropout': .35},
            'selection': {'candidate': 'bce_normal', 'loss': 'normal', 'lr': .0008,
                          'checkpoint_metric': 'inner_patient_logloss', 'metrics': metrics,
                          'confirmation_seeds': list(confirmation.SEEDS)},
            'preprocessing': {'phases': ['PRE', 'EARLY', 'LATE'], 'shape': [3,256,256],
                              'model_representation': 'raw', 'network_channels': 3,
                              'scaling': 'uint8 / 255; CNN: 2*x-1', 'clinical_required': True,
                              'clinical_raw_columns': list(base.CLINICAL_COLUMNS)},
            'provenance_files': provenance,
            'evaluation_warning': 'Internal adaptive development; seed replication is not external validation. '
                                  'Seven of ten original checkpoints are clinical initializers. '
                                  'Operational Youden threshold is fit on OOF and its performance is apparent.'}
        base.json_write(OUTPUT/'modelo_candidato.json', manifest)
        (OUTPUT/'modelo_candidato.json.sha256').write_text(base.sha256(OUTPUT/'modelo_candidato.json')+'\n')
        base.verificar_modelo_final(OUTPUT/'modelo_candidato.json')
        lines += ['## Paquete candidato', '',
            'Se conservan los diez pesos originales de las semillas 42/2026, como se fijó antes de repetir. '
            'No se elige el par de semillas con mejor resultado. La calibración es identidad: no añade un Platt posterior.', '',
            f'El umbral operativo Youden es {threshold:.10f}, ajustado sobre las OOF originales. '
            'Es una decisión para inferencia; no se usa para afirmar una mejora independiente de sensibilidad/especificidad. '
            'La evaluación interna agrega varios cortes; usar un solo corte, como en la aplicación docente, puede cambiar el rendimiento.', '']
    (OUTPUT/'INFORME_RESULTADOS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report': str(OUTPUT), 'confirmed': agreed, 'candidate_manifest': agreed}, ensure_ascii=False))


if __name__ == '__main__':
    main()
