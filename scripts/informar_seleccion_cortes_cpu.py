from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=Path(__file__).resolve().parents[1]
o=r/'resultados/18_piloto_seleccion_cortes_20261021'
report=json.loads((o/'comparacion_cpu.json').read_text())
history=pd.read_csv(o/'historias_cpu.csv')
frames={k:pd.read_csv(o/f'oof_{k}_cpu.csv').sort_values('patient_id').reset_index(drop=True) for k in (10,3,1)}
ref=frames[10];y=ref.label.to_numpy(); strata=[np.flatnonzero(y==label) for label in (0,1)]
rng=np.random.default_rng(20261021);deltas={3:[],1:[]}
for _ in range(2000):
 idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in strata])
 baseline=roc_auc_score(y[idx],ref.probability_raw.to_numpy()[idx])
 for k in (3,1):
  assert frames[k].patient_id.equals(ref.patient_id)
  deltas[k].append(roc_auc_score(y[idx],frames[k].probability_raw.to_numpy()[idx])-baseline)
for k in (3,1):report['results'][str(k)]['paired_delta_auc_ci95']=np.quantile(deltas[k],[.025,.975]).tolist()
(o/'comparacion_cpu_con_intervalos.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
rows=[]
for i,k in enumerate((10,3,1)):
 h=history[history.k.eq(k)]
 summary=json.loads((o/f'cnn_seleccion_cortes_{k}_cpu/weighted/seed_42/fold_0/summary.json').read_text())
 cfg=json.loads((o/f'cnn_seleccion_cortes_{k}_cpu/weighted/seed_42/fold_0/config.json').read_text())
 for ax,a,b,ylabel in [(axes[i,0],'fit_eval_loss','selection_loss','BCE ponderada'),(axes[i,1],'fit_patient_auc','selection_patient_auc','AUC por paciente')]:
  ax.plot(h.epoch,h[a],label='Fit sin aumentos');ax.plot(h.epoch,h[b],label='Selección interna')
  ax.axvline(summary['best_epoch'],color='gray',ls='--',label='Checkpoint elegido')
  ax.set(title=f'{k} cortes en fit',ylabel=ylabel,xlabel='Época');ax.grid(alpha=.2);ax.legend(fontsize=8)
 axes[i,1].set_ylim(0,1)
 m=report['results'][str(k)]['metrics']
 rows.append(f"| {k} | {cfg['counts']['fit_slices']} | {summary['best_epoch']} | {summary['best_inner_patient_auc']:.4f} | {m['roc_auc']:.4f} | {m['log_loss']:.4f} | {m['tn']}/{m['fp']}/{m['fn']}/{m['tp']} |")
fig.suptitle('Prueba CPU: 30 fit / 18 selección / 44 exterior; semilla 42, fold 0')
fig.tight_layout();fig.savefig(o/'curvas_cpu.png',dpi=150);plt.close(fig)
prep=json.loads((o/'preparacion.json').read_text())
text='''# Prueba CPU de selección de cortes — 21/10/2026

Tres brazos con la misma red ligera de 30.393 parámetros, BCE ponderada por paciente,
lotes de seis pacientes, seis épocas, semilla 42, fold exterior 0. Mismos pacientes,
inicialización, aumentos y selección interna. 30 pacientes en fit, 18 en selección y
44 exteriores. La selección interna y exterior mantienen todos sus cortes. El test
no se abre. El mejor checkpoint se elige exclusivamente por AUC interna.

| Cortes fit/paciente | Cortes fit totales | Época elegida | AUC interna | AUC exterior | Log-loss exterior sin ponderar | TN/FP/FN/TP a 0,5 |
|---|---:|---:|---:|---:|---:|---|
'''+ '\n'.join(rows)+'\n\n'
for k in (3,1):
 d=report['results'][str(k)];lo,hi=d['paired_delta_auc_ci95']
 text+=f"- {k} frente a 10: delta AUC {d['delta_mean_auc_vs_10']:+.4f}; IC95 % bootstrap pareado [{lo:+.4f}, {hi:+.4f}].\n"
text+=f"\nCorrelación mediana por paciente de descriptores visuales en todo desarrollo: 10 cortes {prep['median_patient_pair_correlation']['10']:.4f}; 3 cortes {prep['median_patient_pair_correlation']['3']:.4f}. Con uno no se define correlación por pares.\n"
text+='''
Los intervalos usan 2.000 remuestreos estratificados por clase de las mismas 44
pacientes; no recogen variación de inicialización, folds o elección de muestras.
No se debe promocionar un modelo ni ajustar el selector tras observar estas métricas.
La red pequeña y el tamaño de muestra hacen que el resultado no se pueda trasladar
directamente al modelo grande. Menor correlación no garantiza mejor AUC.

Selección sin pCR ni máscaras: un corte representativo (medoide de descriptores de
las tres fases); para tres, se añaden cortes diversos y separados en orden axial.
No se afirma que sean los cortes con más tumor o mayor valor predictivo.

![Curvas por época](curvas_cpu.png)

![ROC y AUC exterior](roc_folds_cpu.png)
'''
(o/'INFORME_CPU.md').write_text(text,encoding='utf-8')
print(text)
