"""Informe de la comparación docente de BCE y mixup, solo tras completar runs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from . import entrenamiento as base
from . import experimento_bce_mixup as experiment
from . import informe_loss as diagnosis
from .paths import PROJECT_ROOT, RESULTS_DIR

OUTPUT = RESULTS_DIR/'10_informe_bce_mixup_20261011'
NAMES = {'weighted_auc_control': 'BCE ponderada · selección AUC',
         'weighted_logloss_control': 'BCE ponderada · selección pérdida',
         'bce_normal': 'BCE normal', 'bce_mixup': 'BCE normal + mixup',
         'bce_normal_platt_diagnostic': 'BCE normal + Platt (secundario)',
         'bce_mixup_platt_diagnostic': 'Mixup + Platt (secundario)'}


def main():
    source = experiment.DEFAULT_OUTPUT
    state = json.loads((source/'estado.json').read_text())
    if state['status'] != 'complete' or not state.get('protected_unchanged') or state['smoke']:
        raise ValueError('Informe final requiere la tanda completa y verificada')
    protocol = json.loads((source/'protocolo.json').read_text())
    for name, digest in protocol['parameters']['protected'].items():
        if base.sha256(PROJECT_ROOT/name) != digest:
            raise ValueError('Cambió un archivo protegido')
    report = json.loads((source/'comparacion/resumen.json').read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for pattern in ('*.csv', '*.json'):
        for path in (source/'comparacion').glob(pattern):
            shutil.copy2(path, OUTPUT/path.name)
    for name in ('estado.json', 'protocolo.json'):
        shutil.copy2(source/name, OUTPUT/name)
    records, curves, evidence, seed_metrics = [], [], {}, []
    for arm in experiment.ARMS:
        for seed in experiment.SEEDS:
            predictions = []
            for fold in range(5):
                folder = source/arm/f'seed_{seed}'/f'fold_{fold}'
                config = json.loads((folder/'config.json').read_text())
                summary = json.loads((folder/'summary.json').read_text())
                if summary['smoke'] or summary['config_hash'] != base.config_hash(config):
                    raise ValueError('Configuración inconsistente')
                for name, digest in summary['files'].items():
                    if base.sha256(folder/name) != digest:
                        raise ValueError('Artefacto cambiado')
                    evidence[str((folder/name).relative_to(PROJECT_ROOT))] = digest
                payload = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
                hidden = payload['model_config']['hidden']
                visual_norm = float(payload['state_dict']['classifier.weight'][0, :hidden].norm())
                history = pd.read_csv(folder/'history.csv')
                best = summary['tracker']['best_epoch']
                row = history.loc[history.epoch.eq(best)].iloc[0]
                records.append({'arm': arm, 'seed': seed, 'fold': fold, 'best_epoch': best,
                                'last_epoch': int(history.epoch.max()), 'visual_weight_norm': visual_norm,
                                'fit_log_loss': row.fit_log_loss, 'selection_log_loss': row.selection_log_loss,
                                'selected_gap': row.selection_log_loss-row.fit_log_loss,
                                'seconds_epochs': summary['seconds_epochs'],
                                'peak_allocated_bytes': summary['peak_allocated_bytes']})
                curves.append((arm, seed, fold, history, best))
                predictions.append(pd.read_csv(folder/'oof.csv'))
                target = OUTPUT/arm/f'seed_{seed}_fold_{fold}'
                target.mkdir(parents=True, exist_ok=True)
                for name in ('config.json', 'history.csv', 'summary.json', 'environment.json'):
                    shutil.copy2(folder/name, target/name)
            patient_frame = experiment.average_slices([pd.concat(predictions, ignore_index=True)])
            seed_metrics.append({'arm': arm, 'seed': seed, **diagnosis.metrics(patient_frame)})
    epochs = pd.DataFrame(records)
    epochs.to_csv(OUTPUT/'checkpoints.csv', index=False)
    base.json_write(OUTPUT/'entradas_sha256.json', evidence)
    base.json_write(OUTPUT/'metricas_semillas.json', seed_metrics)
    metrics = pd.DataFrame(report['metrics'])
    metrics.drop(columns='fold_auc').to_csv(OUTPUT/'metricas.csv', index=False)

    os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT/'.codex_tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for arm in experiment.ARMS:
        fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True, constrained_layout=True)
        for item_arm, seed, fold, history, best in curves:
            if item_arm != arm:
                continue
            ax = axes[0 if seed == 42 else 1, fold]
            ax.plot(history.epoch, history.fit_log_loss, color='#267b89', label='Fit')
            ax.plot(history.epoch, history.selection_log_loss, color='#c65a3e', label='Selección interna')
            ax.axvline(best, color='#555', linestyle=':', label='Checkpoint elegido')
            ax.set(title=f'Semilla {seed} · fold {fold}', xlabel='Época')
            ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_ylabel('Log-loss por paciente')
        axes[1, 0].set_ylabel('Log-loss por paciente')
        fig.suptitle(NAMES[arm]+' · evaluación FP32 sin aumentos ni dropout; época 0 incluida')
        fig.savefig(OUTPUT/f'curvas_{arm}.png', dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, key, title in zip(axes, ('log_loss', 'mean_fold_auc'),
                               ('Cambio de pérdida · menor es mejor', 'Cambio de AUC media · mayor es mejor')):
        for y, arm in enumerate(experiment.ARMS):
            decision = report['comparisons'][arm]
            ci = decision['intervals_99_58'][key]
            ax.plot([ci['lower'], ci['upper']], [y, y], color='#267b89', linewidth=2)
            ax.scatter([decision['delta_vs_control'][key]], [y], color='#267b89', s=45, zorder=3)
        ax.axvline(0, color='#555', linestyle='--')
        ax.set_yticks(range(2), [NAMES[arm] for arm in experiment.ARMS])
        ax.set_title(title)
        ax.set_ylim(-.5, 1.5)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Diferencias pareadas frente al control · IC 99,58% · mismas 1.097 pacientes')
    fig.savefig(OUTPUT/'comparacion.png', dpi=150)
    plt.close(fig)
    favorable = [arm for arm, decision in report['comparisons'].items() if decision['promising_internal_result']]
    lines = ['# CNN desde cero: BCE normal y mixup', '',
        'La tanda conserva la arquitectura, las cuatro variables clínicas y el contrato PNG de v002. '
        'Entrena veinte modelos con BCE normal y BCE normal + mixup, comparados con los controles ponderados ya entrenados. '
        'No se cargaron imágenes del test. La versión activa no se modifica.', '',
        ('Señal interna favorable según el criterio predefinido: '+', '.join(NAMES[a] for a in favorable)+'. '
         'Requiere comprobar estabilidad antes de promoverla.') if favorable else
        'Ninguna variante supera el criterio de mejora predefinido; no se ha demostrado una mejora del original.', '',
        '| Variante | AUC media folds | AUC OOF | AP | Log-loss | Brier |',
        '|---|---:|---:|---:|---:|---:|']
    for row in report['metrics']:
        lines.append(f"| {NAMES[row['candidate']]} | {row['mean_fold_auc']:.4f} | {row['roc_auc']:.4f} | {row['average_precision']:.4f} | {row['log_loss']:.4f} | {row['brier']:.4f} |")
    lines += ['', '![Comparación](comparacion.png)', '',
        'La comparación principal usa las probabilidades crudas de BCE normal/mixup y el control ponderado con Platt interno. '
        'Se informa también el control elegido por pérdida; las calibraciones nuevas son diagnósticos secundarios, '
        'no alternativas seleccionadas después de observar su resultado.', '',
        '## Curvas y contribución visual', '']
    for arm in experiment.ARMS:
        group = epochs[epochs.arm.eq(arm)]
        trained = int(group.visual_weight_norm.gt(1e-10).sum())
        lines += [f"{NAMES[arm]}: {trained}/10 checkpoints conservan una contribución visual distinta de cero; "
                  f"{int(group.best_epoch.eq(0).sum())}/10 seleccionan época 0. "
                  'Una época 0 equivale al inicializador clínico y no demuestra aprendizaje visual.', '',
                  f'![Curvas {arm}](curvas_{arm}.png)', '']
    lines += ['## Reproducción y límites', '',
        'La sección 7 del PDF exige CNN 2D desde cero, un logit y BCEWithLogitsLoss. Esta tanda cumple esas restricciones. '
        'Mixup interpola imágenes, clínica imputada/estandarizada y etiquetas conjuntamente; no altera el formato de inferencia.', '',
        'Las decisiones de checkpoint, umbral y calibración se ajustan solo con selección interna por paciente. '
        'No se usan etiquetas exteriores para esas decisiones. El bootstrap pareado condiciona a OOF fijas; '
        'la corrección por doce familias no elimina el sesgo de toda la búsqueda adaptativa previa. '
        'Una señal favorable sería evidencia interna, no confirmación clínica ni externa.', '',
        'Se conservan configuración, entorno, hashes, curvas, métricas por semilla y checkpoints seleccionados. '
        'Los pesos y estados completos permanecen en el directorio local de entrenamiento. '
        'El informe se regenera con `.venv/bin/python -B -m cancer_mama.informe_bce_mixup`.', '',
        '[Mixup, ICLR 2018](https://arxiv.org/abs/1710.09412) · '
        '[Calibración, ICML 2022](https://proceedings.mlr.press/v162/zhang22f.html) · '
        '[Límites con calibración posterior, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_On_the_Pitfall_of_Mixup_for_Uncertainty_Calibration_CVPR_2023_paper.html)']
    precision_path = OUTPUT/'sensibilidad_bootstrap_10000.json'
    if precision_path.exists():
        precision = json.loads(precision_path.read_text())
        ci = precision['intervals']['log_loss']
        lines += ['', '## Comprobación de precisión del bootstrap', '',
            f"Con 10.000 réplicas y el mismo alpha=0,05/12, el intervalo de delta log-loss es "
            f"[{ci['lower']:.6f}; {ci['upper']:.6f}]. Se mantiene por debajo de cero. "
            'Es una comprobación numérica adicional; no añade pacientes ni corrige toda la búsqueda adaptativa.']
    visual_path = OUTPUT/'diagnostico_contribucion_visual.json'
    if visual_path.exists():
        visual = json.loads(visual_path.read_text())
        clinical = visual['clinical_initialization_metrics']
        lines += ['', '## Diagnóstico de la contribución visual', '',
            f"El inicializador clínico solo obtiene log-loss {clinical['log_loss']:.6f} y AUC media "
            f"{clinical['mean_fold_auc']:.6f}; la CNN seleccionada obtiene 0,546255 y 0,710686. "
            'Es una comparación descriptiva posterior, que no cambió las decisiones ni el protocolo de repetición. '
            'No constituye por sí sola una demostración del valor adicional de la imagen.']
    audit_path = OUTPUT/'auditoria_reproducibilidad_y_un_corte.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        single = audit['single_cut_metrics']
        lines += ['', '## Reproducción y formato de un corte', '',
            f"Se comprobaron las diez particiones y una predicción guardada por checkpoint con inferencia CPU. "
            f"El error absoluto máximo respecto a la OOF GPU fue {audit['maximum_absolute_error']:.2e}; "
            'los pesos cargados con la clase original y la del ensayo dan exactamente la misma salida en esas entradas.', '',
            f"Tomando un corte por paciente con una regla hash fija e independiente de las etiquetas, la pérdida "
            f"es {single['normal']['log_loss']:.6f} con BCE normal y {single['weighted_control']['log_loss']:.6f} "
            f"con el control. La AUC media es {single['normal']['mean_fold_auc']:.6f} y "
            f"{single['weighted_control']['mean_fold_auc']:.6f}, respectivamente. "
            'Es un diagnóstico posterior sobre desarrollo; no prueba el comportamiento con cualquier corte o paciente nuevo.']
    (OUTPUT/'INFORME_RESULTADOS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report': str(OUTPUT), 'promising': favorable, 'runs': len(records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
