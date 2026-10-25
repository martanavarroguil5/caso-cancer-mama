#!/usr/bin/env python3
"""Informe descriptivo; no cambia modelos, calibración ni umbral."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score

from cancer_mama import cierre_cinetico as closure
from cancer_mama import entrenamiento as base


def main():
    output=closure.OUTPUT;protocol=closure.verify(output)
    result=json.loads((output/'resultado.json').read_text());legacy=json.loads((output/'auditoria_v007.json').read_text())
    choice=json.loads((output/'seleccion_congelada.json').read_text())
    frame=pd.read_csv(output/'comprobacion_cortes.csv');one=frame[frame.selected_one_slice].copy()
    plt.rcParams.update({'font.size':10,'figure.dpi':160})
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    fpr,tpr,_=roc_curve(one.pCR,one.probability_calibrated)
    axes[0].plot(fpr,tpr,label=f"Un corte: AUC {result['primary_single_slice']['roc_auc']:.4f}")
    axes[0].plot([0,1],[0,1],'k--',alpha=.4)
    precision,recall,_=precision_recall_curve(one.pCR,one.probability_calibrated)
    axes[1].plot(recall,precision,label=f"AP {result['primary_single_slice']['average_precision']:.4f}")
    axes[1].axhline(one.pCR.mean(),color='k',linestyle='--',alpha=.4,label='Prevalencia')
    axes[0].set(xlabel='Tasa de falsos positivos',ylabel='Sensibilidad',title='ROC · comprobación interna de 165 pacientes')
    axes[1].set(xlabel='Sensibilidad',ylabel='Precisión',title='PR · un corte por paciente')
    for ax in axes:ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(output/'01_roc_pr_un_corte.png');plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,6.5))
    gaps=[];final_gaps=[];selected_rows=[]
    for j,run in enumerate(result['runs']):
        history=pd.read_csv(output/run['path']/'history.csv');epoch=run['selected_epoch']
        selected=history[history.epoch.eq(epoch)].iloc[0]
        gaps.append(float(selected.gap));final_gaps.append(float(history.iloc[-1].gap))
        selected_rows.append({'loss':run['loss'],'seed':run['seed'],'selected_epoch':epoch,**selected.to_dict()})
        axes[0,j].plot(history.epoch,history.fit_bce,label='Fit')
        axes[0,j].plot(history.epoch,history.selection_bce,label='Selección')
        axes[0,j].set(title=f"{run['loss']} · semilla {run['seed']}",ylabel='BCE sin ponderar clase; peso por paciente')
        axes[1,j].plot(history.epoch,history.fit_auc,label='Fit')
        axes[1,j].plot(history.epoch,history.selection_auc,label='Selección')
        axes[1,j].set(xlabel='Época',ylabel='AUC de corte, peso por paciente',ylim=(0,1))
        for ax in axes[:,j]:ax.axvline(epoch,color='k',linestyle=':',alpha=.5);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'02_aprendizaje.png');plt.close(fig)
    pd.DataFrame(selected_rows).to_csv(output/'diagnostico_checkpoints.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(8,3.8))
    for ax,key,title in zip(axes,['raw_at_0_5','primary_single_slice'],['Umbral 0,50 crudo','Política calibrada congelada']):
        m=result[key];matrix=np.array(m['confusion_matrix'])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=117)
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>60 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['no pCR','pCR'],yticklabels=['no pCR','pCR'],xlabel='Predicción',ylabel='Real',title=title)
    fig.tight_layout();fig.savefig(output/'03_matrices.png');plt.close(fig)
    legacy_frame=pd.read_csv(output/'auditoria_v007_cortes.csv');folds=[]
    for fold,g in legacy_frame.groupby('fold'):
        g=g.reset_index(drop=True);single=g.iloc[closure.one_indices(g)]
        mean=g.groupby('patient_id').agg(pCR=('pCR','first'),probability_raw=('probability_raw','mean'))
        folds.append({'fold':int(fold),'patients':len(single),'single_auc':float(roc_auc_score(single.pCR,single.probability_raw)),
                      'mean_slices_auc':float(roc_auc_score(mean.pCR,mean.probability_raw))})
    closure.write(output/'auditoria_v007_por_fold.json',{'folds':folds,'mean_fold_single_auc':float(np.mean([x['single_auc'] for x in folds])),
        'mean_fold_all_slices_auc':float(np.mean([x['mean_slices_auc'] for x in folds]))})
    m=result['primary_single_slice'];raw=result['raw_at_0_5'];lo,hi=result['auc_ci95']
    text=f'''# Cierre de la CNN cinética para una tripleta de un solo corte

25/10/2026. Los cuatro pasos autorizados se ejecutan con un protocolo único y
cuatro entrenamientos. **Resultado interno insuficiente para acreditar AUC 0,60.**
El candidato se conserva como V008 experimental; V007 original permanece intacto.

## 1. Auditoría del formato de uso

La demostración recibe una tripleta PRE/EARLY/LATE de un solo corte, 3×256×256,
uint8/255. No recibe clínica ni promedia otros cortes que el profesor no entrega.
El AUC 0,6084 anterior de V007 usó todos los cortes de 176 pacientes de test:
**no es el rendimiento demostrado de esta aplicación ni de V008**.

Se auditaron las parejas históricas V007 que excluían a cada paciente en {legacy['patients']}
pacientes de las nuevas particiones fit+selection. No se abrieron para esta
auditoría las imágenes de calibration/check ni test/validation. AUC agrupada
cruda con un corte: {legacy['one_fixed_slice_auc']:.4f}; con media de todos los
cortes: {legacy['patient_mean_all_slices_auc']:.4f}. Cada fold puede tener otra
escala de salida por la pérdida elegida: las medias de AUC por fold están en
`auditoria_v007_por_fold.json`. Son componentes históricos de desarrollo, no el
ensemble completo ni una evaluación independiente.

Las diez redes antiguas, en conjunto, usaron todas las pacientes de train. No
había un grupo de train excluido de todas ellas para recalibrar directamente
el ensemble final de diez redes. Por ello se hizo la tanda limitada siguiente.

## 2. Tanda limitada y separación de responsabilidades

Misma clase CNNMultiescalaCinetica, **593.521 parámetros**, mismos filtros,
recortes, mapas y dropout que V007; inicialización desde cero. LR 0,0003,
weight decay 0,01: opción modal de las elecciones internas históricas, sin usar
test para escogerla. AdamW, AMP, lote 32, coseno, máximo 40 épocas, mínimo 8,
paciencia 8. BCEWithLogitsLoss normal/ponderada y semillas 42/2026.

| Rol disjunto | Pacientes | Uso |
|---|---:|---|
| Fit | 657 | Gradientes; todos sus cortes, peso total uno por paciente |
| Selection | 110 | Checkpoint y elección normal/ponderada |
| Calibration | 165 | Platt y umbral del ensemble seleccionado |
| Check | 165 | Una comprobación después de congelar pesos y política |

Se estratifica por cohorte y clase. Las semillas y el corte pseudouniforme por
paciente se fijan en `protocolo.json`, sin seleccionar cortes por etiqueta o
predicción. La selección usa AUC de cortes con peso total uno por paciente:
evalúa el contrato de un corte, no la media de diez cortes. El lote en GPU
conserva los mismos uint8 y la escala /255; la caché no ajusta estadísticas.

| Pérdida | AUC del ensemble en selection |
|---|---:|
| Normal, elegida | {choice['candidates'][0]['selection_auc']:.4f} |
| Ponderada | {choice['candidates'][1]['selection_auc']:.4f} |

Se completaron {sum(x['epochs'] for x in result['runs'])} épocas y
{sum(x['optimizer_steps'] for x in result['runs']):,} pasos AdamW, sin pasos
omitidos. Los checkpoints elegidos son las épocas 9 y 4 de BCE normal. Los
pesos seleccionados se guardaron antes de abrir calibration y las imágenes de
check se cargaron después de congelar `politica_congelada.json`.

## 3. Calibración, umbral y resultado de comprobación

La calibración usa exactamente la media de las dos semillas que ejecutará la
aplicación sobre una tripleta. Platt con C=1 y pendiente no negativa, solo con
las 165 pacientes de calibration; todos sus cortes pesan en total uno por
paciente. El umbral maximiza balanced accuracy en calibration. Es un criterio
docente declarado, no una regla clínica ni una obligación de especificidad 90 %.

**Umbral congelado: {m['threshold']:.9f}.** Pendiente Platt
{result['policy']['calibration']['slope']:.6f}, intercepto
{result['policy']['calibration']['intercept']:.6f}.

La comprobación primaria usa una tripleta prefijada por cada una de las 165
pacientes (48 pCR). IC95 % mediante 2.000 bootstrap por paciente, estratificados
por cohorte y clase, condicionales a los pesos y al corte elegido.

| Métrica | Crudo, umbral 0,50 | Calibrado, umbral congelado |
|---|---:|---:|
| AUC | {raw['roc_auc']:.4f} | **{m['roc_auc']:.4f}**, IC95 % [{lo:.4f}; {hi:.4f}] |
| Sensibilidad pCR | {raw['recall_pcr']:.2%} | **{m['recall_pcr']:.2%}** |
| Especificidad | {raw['recall_no_pcr']:.2%} | **{m['recall_no_pcr']:.2%}** |
| TN / FP / FN / TP | {raw['tn']} / {raw['fp']} / {raw['fn']} / {raw['tp']} | **{m['tn']} / {m['fp']} / {m['fn']} / {m['tp']}** |
| Balanced accuracy | {raw['balanced_accuracy']:.4f} | {m['balanced_accuracy']:.4f} |
| Brier | {raw['brier']:.4f} | {m['brier']:.4f} |
| Logloss | {raw['log_loss']:.4f} | {m['log_loss']:.4f} |

![ROC y PR](01_roc_pr_un_corte.png)
![Matrices](03_matrices.png)

La política detecta más pCR, a costa de muchos FP. La transformación monótona
y el umbral **no aumentan la AUC**. No se ajusta otra vez después de ver check.
Como análisis secundario, AUC de todos los cortes con peso por paciente:
{result['secondary_weighted_slice_auc']:.4f}. La media de cortes por paciente
da {result['diagnostic_all_slices_mean_raw_auc']:.4f}, pero cambia la entrada y
no se elige esa métrica como principal tras observar el resultado.

## Sobreajuste y límites

Brecha mediana fit−selection en los cuatro checkpoints elegidos: {np.median(gaps):.4f};
en las últimas épocas: {np.median(final_gaps):.4f}. La selección interna del
ensemble fue 0,6884 y la comprobación de un corte queda en 0,5598. Las métricas
de selección usan distribución de cortes ponderada y la primaria de check un
corte por paciente; incluso comparando la secundaria de check, 0,5407, el
resultado interno no se sostiene. No se declara eliminado el sobreajuste.

![Aprendizaje y checkpoints](02_aprendizaje.png)

Las 1.097 pacientes de train tienen antecedentes de búsqueda adaptativa en el
proyecto. Aunque los cuatro roles estén separados para este procedimiento,
check sigue siendo desarrollo interno. El test de 176 pacientes y la validación
privada permanecen cerrados en esta tanda. V007 y V008 no se comparan como si
tuvieran los mismos pesos, pacientes o número de cortes de entrada.

## 4. Paquete y aplicación

Se conserva el candidato elegido en selection como
`v008_cnn_cinetica_un_corte_20261025`, dos pesos de inferencia, calibración,
umbral, fuentes, configuración y hashes. No incluye optimizer, RNG ni dataset.
Es experimental e insuficiente: no sustituye V007 ni demuestra AUC 0,60.

La aplicación `app_imagen.py` recibe exclusivamente la tripleta de un corte,
valida formato, tamaño, fase y coincidencia de identificador, muestra el realce,
probabilidad, clase, umbral, versión/checksum, dispositivo y latencia. No ofrece
un deslizador para cambiar el umbral con las muestras del profesor. La evaluación
privada solo corresponde al profesor y no se ejecuta aquí.

```bash
.venv/bin/python -m streamlit run app_imagen.py
```

El despliegue público y la evaluación docente posterior son pasos separados;
esta entrega prepara y verifica la aplicación local, sin publicar automáticamente.

## Evidencia y decisión

- `protocolo.json` y SHA: configuración y particiones fijadas antes de entrenar.
- `particiones.csv`: cuatro roles por paciente; solo train.
- `seleccion_congelada.json`, `politica_congelada.json` y SHA: orden de congelación.
- `resultado.json`, `comprobacion_cortes.csv` y `calibracion_cortes.csv`.
- `diagnostico_checkpoints.csv`, `entorno.json` y registro de entrenamiento.
- Las historias y configuraciones seleccionadas se incluyen en la versión.

**Decisión:** conservar resultados y candidato, preparar la demostración de un
corte y detener los ajustes. El resultado no justifica otra búsqueda contra
estas mismas 165 pacientes ni modificar la política mirando el test antiguo.
'''
    (output/'INFORME_RESULTADOS.md').write_text(text)
    closure.verify(output)
    print(json.dumps({'report':str(output/'INFORME_RESULTADOS.md'),'auc':m['roc_auc'],'gap_selected':float(np.median(gaps))}))


if __name__=='__main__':main()
