"""Informe verificable y paquete de inferencia del ensayo exclusivamente imagen."""
from pathlib import Path
import json,shutil,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,log_loss,confusion_matrix
from cancer_mama import imagen_mil as m
from cancer_mama import entrenamiento as base

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'modelos/imagen_sola_20261025'


def bootstrap(oof,repetitions=2000):
    y=oof.pCR.to_numpy();p=oof.probability.to_numpy();folds=oof.fold.to_numpy()
    groups=[np.flatnonzero((y==label)&(folds==fold)) for fold in range(5) for label in [0,1]]
    rng=np.random.default_rng(20261025)
    pooled=[];mean_fold=[]
    for _ in range(repetitions):
        indices=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        pooled.append(roc_auc_score(y[indices],p[indices]))
        mean_fold.append(np.mean([roc_auc_score(y[ix],p[ix]) for ix in [indices[folds[indices]==f] for f in range(5)]]))
    return dict(repetitions=repetitions,seed=20261025,strata='fold and patient label',
                pooled_auc_ci95=np.quantile(pooled,[.025,.975]).tolist(),
                mean_fold_auc_ci95=np.quantile(mean_fold,[.025,.975]).tolist(),
                limitation='Conditional on fitted models; excludes uncertainty of adaptive historical search')


def package(choices,metrics):
    PACKAGE.mkdir(parents=True,exist_ok=True)
    weights=PACKAGE/'pesos';weights.mkdir(exist_ok=True)
    manifest=dict(schema_version=1,name='imagen_sola_20261025',
        status='experimental_development_validation',predictors=['PRE','EARLY','LATE'],
        models=[],groups=[],source_files={},test_images_loaded=0,metrics=metrics,
        aggregation='mean of five internally calibrated fold ensembles; three seeds per group')
    for choice in choices:
        indices=[]
        for entry in choice['checkpoints']:
            dest=weights/f'fold_{choice["fold"]}_seed_{entry["seed"]}.pt'
            if base.sha256(entry['file'])!=entry['sha256']:
                raise ValueError('Peso modificado desde el cierre de selección')
            shutil.copy2(entry['file'],dest)
            indices.append(len(manifest['models']))
            manifest['models'].append(dict(file=str(dest.relative_to(PACKAGE)),sha256=base.sha256(dest),
                fold=choice['fold'],seed=entry['seed'],family=choice['family']))
        manifest['groups'].append(dict(fold=choice['fold'],family=choice['family'],model_indices=indices,
                                      calibration=choice['calibration']))
    code=PACKAGE/'codigo'
    shutil.copytree(ROOT/'src/cancer_mama',code/'src/cancer_mama',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'requirements.txt',code/'requirements.txt')
    for file in code.rglob('*'):
        if file.is_file():
            manifest['source_files'][str(file.relative_to(PACKAGE))]=base.sha256(file)
    m.write_json(PACKAGE/'manifest.json',manifest)
    for file in ['final_metrics.json','final_plan_locked.json','final_choices_locked.json','bootstrap.json']:
        shutil.copy2(m.OUTPUT/file,PACKAGE/file)
    readme=f'''# Modelo exclusivamente de imágenes — 25/10/2026

Quince pesos entrenados, cinco grupos de tres semillas. Cada grupo utiliza una
CNN 2D propia desde cero y descriptores derivados únicamente de los PNG. Ninguna
entrada clínica, cohorte, identificador o dato de adquisición entra en el modelo.
Los identificadores se usan solo para separar pacientes y auditar resultados.

AUC OOF calibrada de desarrollo: **{metrics['calibrated']['auc']:.4f}**.
Media de AUC por fold: **{metrics['mean_fold_auc']:.4f} ± {metrics['sd_fold_auc']:.4f}**.
La búsqueda es adaptativa; estas cifras no son una evaluación externa. Test cerrado.
No se ha demostrado utilidad clínica. El objetivo ideal 0,70 se informa según
el resultado real, sin cambiar etiquetas, signos ni seleccionar con test.

## Inferencia sin dataset ni clínica

Desde la raíz del proyecto, usando el código congelado:

```bash
PYTHONPATH=modelos/imagen_sola_20261025/codigo/src .venv/bin/python -m cancer_mama.inferencia_mil \\
  --package modelos/imagen_sola_20261025 \\
  --pre paciente_z000_PRE.png paciente_z001_PRE.png \\
  --early paciente_z000_EARLY.png paciente_z001_EARLY.png \\
  --late paciente_z000_LATE.png paciente_z001_LATE.png
```

Los PNG deben ser monocromos de 8 bits y 256 × 256, entre uno y diez cortes
alineados de la misma paciente, en orden axial. Las tres listas deben tener
igual longitud. La evaluación usa los cortes disponibles; un único corte tiene
menos información y su rendimiento no se ha evaluado por separado.

Se reduce a 128 px con BOX; cada modelo replica sus descriptores de intensidad,
realce y/o textura. El escalado guardado procede solo del fit del fold. Las
probabilidades de tres semillas se promedian antes de aplicar su calibración
Platt interna; se promedian después los cinco grupos. El manifest incluye los
SHA-256 de todos los pesos y fuentes. La CLI comprueba los hashes de los pesos.

La ROC publicada usa únicamente el grupo que excluye a cada paciente. La media
de los cinco grupos sirve para pacientes nuevas y no debe reevaluarse sobre
train como si fuera independiente.

Informe: `resultados/20_busqueda_imagen_20261025/INFORME_RESULTADOS.md`.
'''
    (PACKAGE/'README.md').write_text(readme)
    return manifest


def main():
    output=m.OUTPUT
    metrics=json.loads((output/'final_metrics.json').read_text())
    choices=json.loads((output/'final_choices_locked.json').read_text())['choices']
    oof=pd.read_csv(output/'final_oof_patients.csv')
    ci=bootstrap(oof);m.write_json(output/'bootstrap.json',ci)
    # Curva ROC exterior: no se usa para escoger pesos ni umbrales.
    fig,ax=plt.subplots(figsize=(7,6))
    for fold,f in oof.groupby('fold'):
        fpr,tpr,_=roc_curve(f.pCR,f.probability)
        ax.plot(fpr,tpr,lw=1,alpha=.45,label=f'Fold {fold}: {roc_auc_score(f.pCR,f.probability):.3f}')
    fpr,tpr,_=roc_curve(oof.pCR,oof.probability)
    ax.plot(fpr,tpr,lw=2.5,color='#087e8b',label=f'OOF: {metrics["calibrated"]["auc"]:.3f}')
    ax.plot([0,1],[0,1],'--',color='gray',lw=1)
    ax.set(xlabel='1 − especificidad',ylabel='Sensibilidad',title='ROC por paciente — validación de desarrollo')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'01_roc_exterior.png',dpi=160);plt.close(fig)
    histories=[];selected_rows=[]
    for kind in ['auc','bce']:
        fig,axes=plt.subplots(1,5,figsize=(19,4),sharey=True)
        for choice,ax in zip(choices,axes):
            for entry in choice['checkpoints']:
                run=Path(entry['file']).parent
                s=json.loads((run/'summary.json').read_text());h=pd.read_csv(run/'history.csv')
                if kind=='auc':
                    h['fold']=choice['fold'];h['seed']=entry['seed'];h['family']=choice['family'];histories.append(h)
                    row=h[h.epoch.eq(s['best_epoch'])].iloc[0].to_dict();selected_rows.append(row)
                ax.plot(h.epoch,h[f'fit_{kind}'],color='#2878a5',alpha=.45)
                ax.plot(h.epoch,h[f'selection_{kind}'],color='#db7c21',alpha=.45)
                chosen=h[h.epoch.eq(s['best_epoch'])].iloc[0]
                ax.scatter([chosen.epoch],[chosen[f'selection_{kind}']],color='#db7c21',s=18)
            ax.set(title=f'Fold {choice["fold"]}',xlabel='Época');ax.grid(alpha=.2)
        axes[0].set_ylabel('AUC por paciente' if kind=='auc' else 'BCE sin aumentos')
        axes[0].plot([],[],color='#2878a5',label='Fit');axes[0].plot([],[],color='#db7c21',label='Selección interna');axes[0].legend()
        fig.suptitle('Cada punto marca el checkpoint conservado; outer no interviene')
        fig.tight_layout();fig.savefig(output/('02_curvas_auc.png' if kind=='auc' else '03_curvas_perdida.png'),dpi=140);plt.close(fig)
    pd.concat(histories).to_csv(output/'final_selected_histories.csv',index=False)
    pd.DataFrame(selected_rows).to_csv(output/'final_selected_epochs.csv',index=False)
    # Baseline de prevalencia global (solo comparador descriptivo de pérdidas).
    y=oof.pCR.to_numpy();p=oof.probability.to_numpy();cm=confusion_matrix(y,p>=.5,labels=[0,1])
    tn,fp,fn,tp=map(int,cm.ravel())
    runs=[]
    for path in (output/'runs').glob('*/*/*/summary.json'):
        s=json.loads(path.read_text())
        if not s['smoke']:
            runs.append(s)
    training_time=sum(s['seconds'] for s in runs)
    n_epochs=sum(s['epochs'] for s in runs)
    selection=pd.DataFrame(selected_rows)
    outer_gaps=[c['fit_auc']-metrics['folds'][c['fold']]['auc'] for c in choices]
    m.write_json(output/'execution_totals.json',dict(runs=len(runs),epochs=n_epochs,training_seconds=training_time,test_images_loaded=0))
    manifest=package(choices,metrics)
    fold_table='\n'.join(f'| {f["fold"]} | {choices[f["fold"]]["family"]} | {f["auc"]:.4f} | {f["bce"]:.4f} | {outer_gaps[f["fold"]]:+.4f} |' for f in metrics['folds'])
    cohort_table='\n'.join(f'| {f["cohort"]} | {f["patients"]} | {f["auc"]:.4f} |' for f in metrics['cohorts'])
    ci_auc=ci['pooled_auc_ci95'];ci_fold=ci['mean_fold_auc_ci95']
    target=metrics['calibrated']['auc']>=.70 and metrics['mean_fold_auc']>=.70
    text=f'''# Red neuronal exclusivamente de imágenes — resultados

25/10/2026. **{'Se alcanza 0,70 en ambas estimaciones principales.' if target else 'No se ha alcanzado el objetivo ideal de AUC 0,70.'}**

Se completaron **{len(runs)} entrenamientos y {n_epochs} épocas**, con
{training_time/60:.1f} minutos acumulados de entrenamiento medido. Son pilotos,
repeticiones internas y búsqueda anidada; no se suman como muestras independientes.
Se conservan quince pesos seleccionados y un paquete de inferencia independiente.

## Resultado exterior de desarrollo

- **AUC OOF calibrada: {metrics['calibrated']['auc']:.4f}**, IC95 % [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}].
- **Media de AUC por fold: {metrics['mean_fold_auc']:.4f} ± {metrics['sd_fold_auc']:.4f}**, IC95 % condicional [{ci_fold[0]:.4f}, {ci_fold[1]:.4f}].
- AUC OOF cruda: **{metrics['raw']['auc']:.4f}**. La calibración no mejora el ranking dentro de cada fold; alinea escalas entre modelos.
- BCE calibrada: **{metrics['calibrated']['bce']:.4f}**; Brier: **{metrics['calibrated']['brier']:.4f}**; AP: **{metrics['calibrated']['ap']:.4f}**.
- 1.097 pacientes, 322 pCR, una predicción exterior por paciente. **Cero imágenes de test cargadas**.

| Fold | Receta elegida internamente | AUC exterior | BCE exterior | Brecha AUC fit−outer |
|---|---|---:|---:|---:|
{fold_table}

![ROC exterior](01_roc_exterior.png)

## Sobreajuste y elección de pesos

Las CNN son propias y parten de pesos aleatorios. Cada paciente contribuye una
observación a la pérdida. Los aumentos se comparten entre fases y cortes, se
utiliza dropout, AdamW, mixup o ranking auxiliar según la receta y parada interna.
Los descriptores de intensidad, realce y textura se calculan solo de píxeles;
no se usan variables clínicas, cohorte, IDs ni adquisición como predictores.

La mediana de brecha AUC fit−selección de los quince checkpoints es
**{selection.auc_gap.median():.4f}** (rango {selection.auc_gap.min():.4f} a {selection.auc_gap.max():.4f}).
La restricción final se aplica a cada ensemble interno de tres semillas (<=0,12),
no a cada semilla aislada. Los puntos en las gráficas señalan los pesos guardados.
Se conservan también las épocas posteriores para mostrar si aparece divergencia.
Una brecha pequeña no demuestra por sí sola ausencia de sobreajuste: también
puede indicar poco aprendizaje, y la búsqueda puede sobreajustar la validación.
La mediana de brecha entre el ensemble fit y su fold exterior es
**{np.median(outer_gaps):.4f}**. {'Persiste una señal de sobreajuste o cambio de distribución; no se declara resuelto.' if np.median(outer_gaps)>.10 else 'No prueba ausencia de sobreajuste en otra población.'}

![AUC ajuste y selección](02_curvas_auc.png)

![Pérdidas ajuste y selección](03_curvas_perdida.png)

Se descartó la promoción del pico 0,66 de una red lineal muy penalizada: oscilaba
con el optimizador y elegía máximos de validación poco estables. El control
convergido y los pesos promediados revelaron ese problema. La agregación axial
Conv1d y las BCE ponderadas tampoco justificaron ampliar esos ensayos.

## Protocolo y límites

Se fijaron tres recetas antes de outer. En cada fold se compararon internamente
sus tres semillas; se eligió AUC con brecha fit−selección <=0,12 y desempate por
BCE. **Las cinco decisiones y los hashes se guardaron antes de calcular cualquier
predicción exterior.** Calibración Platt únicamente interna, sin invertir signos.
El test entregado y la validación privada siguen cerrados.

Estas cifras son **validación de desarrollo adaptativo**, no confirmación externa:
el historial del proyecto y la búsqueda de arquitecturas han explorado train.
Cada paciente queda fuera de fit, checkpoint, selección de receta y calibración
de su grupo exterior, pero eso no borra el historial global de decisiones.
Los IC usan 2.000 bootstrap por paciente, estratificados por fold y etiqueta;
condicionan a estos pesos y no recogen toda la búsqueda ni futuros reentrenamientos.

| Cohorte (solo auditoría) | Pacientes | AUC |
|---|---:|---:|
{cohort_table}

A umbral fijo 0,5: TN={tn}, FP={fp}, FN={fn}, TP={tp}.
No se ha reajustado el umbral mirando outer. ROC-AUC mide ordenación, no garantiza
buenas decisiones a ese umbral ni calibración en otra población.

## Artefactos y reproducción

- [Registro completo de experimentos](../../docs/mejoras_cnn/EXPERIMENTOS_IMAGEN_20261025.md).
- [Plan previo a outer](final_plan_locked.json) y [elecciones bloqueadas](final_choices_locked.json).
- [OOF por paciente](final_oof_patients.csv), [métricas](final_metrics.json) e [intervalos](bootstrap.json).
- [Candidatos internos de cada fold](final_inner_candidates.csv), [historias](final_selected_histories.csv) y [épocas conservadas](final_selected_epochs.csv).
- [Auditoría de datos](data_audit.json): sin tripletas idénticas entre pacientes distintas de train a 128 px.
- [Verificación CPU/GPU de inferencia](inference_verification_internal.json).
- [Paquete de quince pesos, código congelado y uso](../../modelos/imagen_sola_20261025/README.md).

Preparación y pilotos: `scripts/experimentar_imagen_mil.py`,
`pilotos_imagen_roi.py`, `pilotos_imagen_regularizada.py`, `confirmar_imagen_mil.py`,
`pilotos_imagen_estable.py` y `pilotos_imagen_axial.py`.
Validación final: `scripts/validar_imagen_anidada.py`.
Informe y paquete: `scripts/informar_imagen_mil.py`.
Las configuraciones exactas, particiones, hashes de código y entornos están en
cada ejecución. La caché local de píxeles se excluye de Git y puede regenerarse.

Fuentes metodológicas: [MIL de Ilse et al.](https://proceedings.mlr.press/v80/ilse18a.html),
[Group Normalization](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.php),
[mixup](https://arxiv.org/abs/1710.09412) y
[repositorio original BreastDCEDL](https://github.com/naomifridman/BreastDCEDL).
Sus pesos preentrenados no se han utilizado. Uso educativo.
'''
    (output/'INFORME_RESULTADOS.md').write_text(text)
    print(json.dumps(dict(package=str(PACKAGE),report=str(output/'INFORME_RESULTADOS.md'),metrics=metrics,bootstrap=ci),indent=2))

if __name__=='__main__':
    main()
