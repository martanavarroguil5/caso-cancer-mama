"""Regenera las figuras de FiLM desde predicciones/historias guardadas; no entrena."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

out = Path(__file__).resolve().parent
root = out.parent.parent
summary = json.loads((out / 'resumen.json').read_text())
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})
colors = {'original': '#153a58', 'control_weighted': '#b46617', 'film_weighted': '#008b82'}
fig, ax = plt.subplots(figsize=(8.7, 7.2))
original = pd.read_csv(root / 'resultados/06_informe_multimodal_20261007/comparacion_pareada.csv')
series = [('V002 original (histórico)', original, 'selected', colors['original'])]
for arm, label in [('control_weighted', 'Control reentrenado'), ('film_weighted', 'FiLM ponderado')]:
    series.append((label, pd.read_csv(out / f'oof_{arm}.csv'), 'probability', colors[arm]))
for label, data, prob, color in series:
    fpr, tpr, _ = roc_curve(data.label, data[prob])
    auc = roc_auc_score(data.label, data[prob])
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{label}: AUC {auc:.4f}')
ax.plot([0, 1], [0, 1], color='#b3b3b3', ls='--', lw=1)
ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel='Tasa de falsos positivos',
       ylabel='Sensibilidad', title='FiLM no mejora la discriminación')
ax.grid(alpha=.15)
ax.legend(loc='lower right', frameon=True, fontsize=10)
fig.text(.12, .045, '1.097 pacientes · probabilidades OOF sin calibrar\n'
         'V002 usa selección histórica distinta: su comparación es descriptiva.', fontsize=10)
fig.tight_layout(rect=(0, .10, 1, 1))
fig.savefig(out / '01_roc.png', dpi=170)
plt.close(fig)

histories = pd.read_csv(out / 'historias.csv')
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.3))
for arm, label in [('control_weighted', 'Control'), ('film_weighted', 'FiLM')]:
    h = histories[(histories.arm == arm) & (histories.epoch <= 12)]
    assert h.groupby('epoch').size().eq(10).all() and h.epoch.nunique() == 13
    for ax, metric in zip(axes, ['slice_bce', 'patient_auc']):
        for partition, partlabel, style in [('fit', 'Entrenamiento', '-'), ('selection', 'Validación interna', '--')]:
            pivot = h.pivot(index='epoch', columns=['seed', 'fold'], values=f'{partition}_{metric}')
            ax.plot(pivot.index, pivot.median(axis=1), style, color=colors[arm], lw=2,
                    label=f'{label} · {partlabel}')
            ax.fill_between(pivot.index, pivot.quantile(.25, axis=1), pivot.quantile(.75, axis=1),
                            color=colors[arm], alpha=.08)
        ax.set(xlabel='Época', xlim=(0, 12), xticks=range(0, 13, 2))
        ax.grid(alpha=.15)
axes[0].set(title='Pérdida BCE ponderada por corte', ylabel='Pérdida (menor es mejor)')
axes[1].set(title='AUC por paciente', ylabel='AUC (mayor es mejor)', ylim=(.5, .95))
axes[0].legend(fontsize=8.5, loc='upper left')
fig.suptitle('FiLM mantiene la separación entre entrenamiento y validación', fontsize=15, fontweight='bold')
fig.text(.07, .035, 'Mediana y rango intercuartílico de 10 ejecuciones por brazo; evaluación FP32 sin aumentos.\n'
         'Se muestran las épocas 0–12, comunes a todas. Son curvas de entrenamiento, no métricas del checkpoint elegido.', fontsize=10)
fig.tight_layout(rect=(0, .12, 1, .94))
fig.savefig(out / '02_curvas.png', dpi=170)
plt.close(fig)

rows = []
original_metrics = summary['historical_v002']['saved_calibration_threshold']
rows.append({'modelo': 'original_v002', 'roc_auc_raw': original_metrics['roc_auc'],
             'average_precision_raw': original_metrics['average_precision'],
             **{k: original_metrics[k] for k in ['log_loss','brier','tn','fp','fn','tp','sensitivity','specificity','accuracy','precision','f1']},
             'protocol': 'historical_selection_and_apparent_oof_calibration_threshold'})
for arm, m in summary['metrics'].items():
    rows.append({'modelo': arm, 'roc_auc_raw': m['raw']['roc_auc'],
                 'average_precision_raw': m['raw']['average_precision'],
                 **{k: m['calibrated'][k] for k in ['log_loss','brier']},
                 **m['inner_youden'], 'protocol': 'inner_selection_calibration_threshold_outer_oof'})
pd.DataFrame(rows).to_csv(out / 'comparacion_metricas.csv', index=False)
print('Figuras y tabla de métricas generadas desde evidencia guardada.')
