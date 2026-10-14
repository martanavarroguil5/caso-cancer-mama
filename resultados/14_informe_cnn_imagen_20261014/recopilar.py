#!/usr/bin/env python3
"""Recopila el ensayo ya terminado; no entrena ni abre el dataset."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (brier_score_loss, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def recopilar(runs, output):
    output.mkdir(parents=True, exist_ok=True)
    state = json.loads((runs/'estado.json').read_text())
    assert state['status'] == 'complete' and state['protected_versions_unchanged']
    for source, name in ((runs/'estado.json', 'estado.json'),
                         (runs/'comparacion/seleccion.json', 'seleccion.json'),
                         (runs/'comparacion/comparacion_perdidas.csv', 'comparacion_perdidas.csv'),
                         (runs/'verificacion_final.json', 'verificacion_entrenamiento.json')):
        shutil.copy2(source, output/name)
    histories, records, diagnostics, patients, slices = [], [], [], [], []
    for loss in ('normal', 'weighted'):
        for seed in (42, 2026):
            for fold in range(5):
                directory = runs/f'cnn_imagen_fases/{loss}/seed_{seed}/fold_{fold}'
                config = json.loads((directory/'config.json').read_text())
                summary = json.loads((directory/'summary.json').read_text())
                assert summary['status'] == 'complete' and summary['eligible_for_selection']
                assert not config['smoke'] and summary['test_images_loaded'] == 0
                for filename, key in (('inference.pt', 'inference_sha256'),
                                      ('outer_patients.csv', 'outer_sha256'),
                                      ('selection_patients.csv', 'selection_sha256')):
                    assert digest(directory/filename) == summary[key]
                record = {'loss': loss, 'seed': seed, 'fold': fold}
                records.append({**record, 'config': config, 'summary': summary,
                                'environment': json.loads((directory/'environment.json').read_text())})
                history = pd.read_csv(directory/'history.csv').assign(**record)
                histories.append(history)
                best = history.loc[history.epoch.eq(summary['best_epoch'])].iloc[0]
                last = history.iloc[-1]
                diagnostics.append({**record, 'epochs': len(history),
                    'best_epoch': summary['best_epoch'],
                    'selected_fit_auc': float(best.fit_patient_auc),
                    'selected_internal_auc': float(best.selection_patient_auc),
                    'selected_gap': float(best.auc_gap),
                    'final_fit_auc': float(last.fit_patient_auc),
                    'final_internal_auc': float(last.selection_patient_auc),
                    'final_gap': float(last.auc_gap),
                    'selection_loss_best': float(best.selection_loss),
                    'selection_loss_last': float(last.selection_loss),
                    'epoch_seconds_median': float(history.epoch_seconds.median())})
                for partition in ('selection', 'outer'):
                    patients.append(pd.read_csv(directory/f'{partition}_patients.csv')
                                    .assign(**record, partition=partition))
                slices.append(pd.read_csv(directory/'outer_slices.csv').assign(**record))
                destination = output/f'curvas_por_ejecucion/{loss}_seed_{seed}_fold_{fold}.png'
                destination.parent.mkdir(exist_ok=True)
                shutil.copy2(directory/'curvas_train_seleccion.png', destination)
    history = pd.concat(histories, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostics)
    write(output/'ejecuciones.json', records)
    history.to_csv(output/'historias.csv', index=False)
    diagnostics.to_csv(output/'diagnostico_por_ejecucion.csv', index=False)
    pd.concat(patients, ignore_index=True).to_csv(output/'predicciones_pacientes_por_ejecucion.csv', index=False)
    pd.concat(slices, ignore_index=True).to_csv(output/'predicciones_cortes_outer.csv', index=False)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    colors = {'normal': '#2867ac', 'weighted': '#c26713'}
    verified = []
    for loss in ('normal', 'weighted'):
        source = runs/f'comparacion/oof_pacientes_{loss}.csv'
        shutil.copy2(source, output/source.name)
        frame = pd.read_csv(source)
        auc = roc_auc_score(frame.label, frame.probability_raw)
        candidate = next(c for c in json.loads((output/'seleccion.json').read_text())['candidates'] if c['loss'] == loss)
        assert np.isclose(auc, candidate['pooled_raw_auc_descriptive'], atol=1e-12)
        assert np.isclose(log_loss(frame.label, frame.probability_calibrated), candidate['crossfit_calibrated_log_loss'], atol=1e-12)
        assert np.isclose(brier_score_loss(frame.label, frame.probability_calibrated), candidate['crossfit_calibrated_brier'], atol=1e-12)
        tn, fp, fn, tp = confusion_matrix(frame.label, frame.prediction, labels=[0, 1]).ravel()
        assert [int(tn), int(fp), int(fn), int(tp)] == [candidate['crossfit_primary_decision'][k] for k in ('tn', 'fp', 'fn', 'tp')]
        fpr, tpr, _ = roc_curve(frame.label, frame.probability_raw)
        ax.plot(fpr, tpr, color=colors[loss], label=f'BCE {loss}: AUC OOF global {auc:.3f}')
        verified.append({'loss': loss, 'n_patients': len(frame), 'roc_auc_raw': auc,
                         'matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]})
    ax.plot([0, 1], [0, 1], '--', color='#777777', label='Azar: AUC 0,500')
    ax.set(title='CNN solo imagen: ROC de desarrollo OOF', xlabel='Tasa de falsos positivos',
           ylabel='Sensibilidad para pCR', xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.2)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(output/'01_roc_oof.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, loss in enumerate(('normal', 'weighted')):
        current = history[history.loss.eq(loss)]
        # Solo las épocas comunes a las diez ejecuciones, sin sesgo de supervivencia.
        limit = int(current.groupby(['seed', 'fold']).epoch.max().min())
        current = current[current.epoch.le(limit)]
        for column, metric, ylabel in ((0, 'loss', 'BCE por corte'), (1, 'auc', 'ROC-AUC por paciente')):
            ax = axes[row, column]
            for source, label, color in (('fit', 'Fit, evaluación', '#2867ac'),
                                          ('selection', 'Selección interna', '#c26713')):
                key = ('fit_eval_loss' if source == 'fit' else 'selection_loss') if metric == 'loss' else ('fit_patient_auc' if source == 'fit' else 'selection_patient_auc')
                grouped = current.groupby('epoch')[key]
                median, q1, q3 = grouped.median(), grouped.quantile(.25), grouped.quantile(.75)
                ax.plot(median.index, median, label=label, color=color)
                ax.fill_between(median.index, q1, q3, alpha=.16, color=color)
            ax.set(title=f'BCE {loss}: mediana e IQR, 10 ejecuciones', xlabel='Época', ylabel=ylabel)
            if metric == 'auc':
                ax.set_ylim(.35, .85)
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle('Curvas en evaluación FP32; solo épocas comunes antes de las paradas')
    fig.tight_layout()
    fig.savefig(output/'02_curvas_fit_seleccion.png', dpi=170)
    plt.close(fig)
    seconds = (datetime.fromisoformat(state['finished_at'])-datetime.fromisoformat(state['started_at'])).total_seconds()
    write(output/'procedencia.json', {
        'training_commit': state['source_commit'], 'training_source_sha256': state['source_sha256'],
        'training_started_at': state['started_at'], 'completed_at': state['finished_at'],
        'duration_seconds': seconds, 'runs': len(records), 'epochs': len(history),
        'test_images_loaded': 0, 'protected_versions_unchanged': True,
        'oof_metrics_recomputed': verified,
        'source_files_sha256': {str(path.relative_to(runs)): digest(path)
            for path in sorted(runs.rglob('*')) if path.is_file() and path.suffix in ('.json', '.csv', '.pt')},
    })
    print(json.dumps({'runs': len(records), 'epochs': len(history), 'minutes': seconds/60,
                      'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('resultados/04_entrenamiento/cnn_imagen_fases_20261014'))
    parser.add_argument('--salida', type=Path, default=Path('resultados/14_informe_cnn_imagen_20261014'))
    args = parser.parse_args()
    recopilar(args.runs, args.salida)
