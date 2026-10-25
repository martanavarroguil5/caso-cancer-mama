"""Informe de la comparación congelada, sin cargar imágenes ni ajustar políticas."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score,roc_curve,precision_recall_curve,
                             confusion_matrix,precision_recall_fscore_support)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cancer_mama import robustez_adquisicion as run
from cancer_mama import entrenamiento_imagen as image
from cancer_mama import entrenamiento as base


def classify(y,pred):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
    return {'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'accuracy':float((tn+tp)/len(y)),
        'sensitivity':float(recall[1]),'specificity':float(recall[0]),'precision_pcr':float(precision[1]),
        'precision_no_pcr':float(precision[0]),'f1_pcr':float(f1[1]),'f1_no_pcr':float(f1[0]),
        'macro_f1':float(np.mean(f1)),'balanced_accuracy':float(np.mean(recall))}


def load_selected(output,result):
    frames={}
    for arm in run.ARMS:
        rows=[]
        for choice in result['selection']['choices']:
            if choice['arm']==arm:
                fold=choice['fold'];loss=choice['selected']['loss']
                rows.append(pd.read_csv(output/f'outer_{fold}_{arm}_{loss}.csv'))
        frames[arm]=pd.concat(rows).sort_values('patient_id').reset_index(drop=True)
    a,b=frames.values()
    for column in ('patient_id','label','fold','cohort'):
        if not a[column].equals(b[column]):raise ValueError('Pacientes de comparación distintos')
    if a.patient_id.duplicated().any() or len(a)!=439:raise ValueError('Población exterior incorrecta')
    return frames


def paired_bootstrap(frames,n=4000):
    a,b=frames.values();y=a.label.to_numpy();fold=a.fold.to_numpy()
    pa,pb=a.probability_raw.to_numpy(),b.probability_raw.to_numpy()
    groups=[g.index.to_numpy() for _,g in a.groupby(['fold','cohort','label'])]
    rng=np.random.default_rng(20261025);values=np.zeros((n,3))
    for i in range(n):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        ma=[];mb=[]
        for f in run.FOLDS:
            z=ix[fold[ix]==f];ma.append(roc_auc_score(y[z],pa[z]));mb.append(roc_auc_score(y[z],pb[z]))
        values[i]=np.mean(ma),np.mean(mb),np.mean(mb)-np.mean(ma)
    intervals=np.quantile(values,[.025,.975],axis=0)
    return {'replicates':n,'strata':['fold','cohort','label'],'seed':20261025,
        'control_mean_fold_auc_ci95':intervals[:,0].tolist(),'campo_mean_fold_auc_ci95':intervals[:,1].tolist(),
        'delta_mean_fold_auc_ci95':intervals[:,2].tolist(),
        'limits':'Intervalos condicionales a pesos fijos, dos folds y desarrollo adaptativo; no corrigen búsquedas históricas'}


def summarize(output,result,frames):
    summaries={}
    for arm,frame in frames.items():
        folds={str(f):float(roc_auc_score(g.label,g.probability_raw)) for f,g in frame.groupby('fold')}
        choices={c['fold']:c['selected']['loss'] for c in result['selection']['choices'] if c['arm']==arm}
        chosen=[r for r in result['runs'] if r['arm']==arm and r['loss']==choices[r['fold']]]
        cohorts={c:{'n':len(g),'pcr':int(g.label.sum()),'raw_auc':float(roc_auc_score(g.label,g.probability_raw)),
            'calibrated_auc':float(roc_auc_score(g.label,g.probability_calibrated))} for c,g in frame.groupby('cohort')}
        summaries[arm]={'fold_auc':folds,'mean_fold_auc':float(np.mean(list(folds.values()))),'std_fold_auc':float(np.std(list(folds.values()),ddof=1)),
            'pooled_raw':image.metricas_binarias(frame.label,frame.probability_raw),
            'pooled_calibrated':image.metricas_binarias(frame.label,frame.probability_calibrated),
            'frozen_policy':classify(frame.label,frame.prediction),'median_selected_gap':float(np.median([r['selected_gap'] for r in chosen])),
            'median_final_gap':float(np.median([r['final_gap'] for r in chosen])),'selected_loss_by_fold':choices,'cohorts':cohorts}
    intervals=paired_bootstrap(frames)
    control,candidate=summaries['control'],summaries['campo_suave'];delta=candidate['mean_fold_auc']-control['mean_fold_auc']
    conditions={'mean_auc_at_least_060':candidate['mean_fold_auc']>=.60,'delta_at_least_002':delta>=.02,
        'both_folds_improve':all(candidate['fold_auc'][str(f)]>control['fold_auc'][str(f)] for f in run.FOLDS),
        'gap_increase_at_most_001':candidate['median_selected_gap']<=control['median_selected_gap']+.01,
        'paired_ci95_lower_positive':intervals['delta_mean_fold_auc_ci95'][0]>0}
    gate={'passed':all(conditions.values()),'conditions':conditions,'delta_mean_auc':delta,
        'meaning':'Filtro exploratorio para ampliar internamente; nunca éxito independiente ni permiso para abrir test'}
    normal_weighted=[]
    for arm in run.ARMS:
        for loss in run.LOSSES:
            metrics=[o for o in result['outcomes'] if o['arm']==arm and o['loss']==loss]
            normal_weighted.append({'arm':arm,'loss':loss,'mean_fold_auc':float(np.mean([m['raw']['roc_auc'] for m in metrics])),
                'fold_auc':{str(m['fold']):m['raw']['roc_auc'] for m in metrics},
                'sensitivity_mean_frozen_policy':float(np.mean([m['policy']['recall_pcr'] for m in metrics])),
                'specificity_mean_frozen_policy':float(np.mean([m['policy']['recall_no_pcr'] for m in metrics]))})
    return {'completed_at':run.now(),'scope':'Piloto interno adaptativo, dos folds; NO test nuevo','unit':'patient_mean_all_slices',
        'n_patients':439,'positive':int(frames['control'].label.sum()),'summaries':summaries,'intervals':intervals,'gate':gate,
        'normal_weighted':normal_weighted,'always_no_pcr':classify(frames['control'].label,np.zeros(439,dtype=int)),
        'finite_all':all(r['finite'] for r in result['runs']),'optimizer_steps':sum(r['optimizer_steps'] for r in result['runs']),
        'skipped_steps':sum(r['skipped'] for r in result['runs']),'training_seconds':sum(r['seconds'] for r in result['runs']),
        'test_images_loaded':0,'private_images_loaded':0,'versions_unchanged':result['versions_unchanged']}


def plots(output,result,frames):
    colors={'control':'#28648f','campo_suave':'#b65a19'}
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for arm,frame in frames.items():
        for fold,g in frame.groupby('fold'):
            fpr,tpr,_=roc_curve(g.label,g.probability_raw)
            axes[0].plot(fpr,tpr,color=colors[arm],linestyle='-' if fold==0 else '--',label=f'{arm} fold{fold}: {roc_auc_score(g.label,g.probability_raw):.3f}')
        precision,recall,_=precision_recall_curve(frame.label,frame.probability_calibrated)
        axes[1].step(recall,precision,where='post',color=colors[arm],label=arm)
    axes[0].plot([0,1],[0,1],':',color='gray');axes[0].set(xlabel='Falsos positivos',ylabel='Sensibilidad',title='ROC cruda por fold · métrica principal')
    axes[1].axhline(frames['control'].label.mean(),color='gray',linestyle=':');axes[1].set(xlabel='Recall',ylabel='Precisión',title='PR agrupada calibrada · secundaria')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('439 pacientes de train · comprobación interna adaptativa, sin test');fig.tight_layout();fig.savefig(output/'03_roc_pr.png',dpi=150);plt.close(fig)
    for fold in run.FOLDS:
        fig,axes=plt.subplots(4,4,figsize=(16,12))
        for row,(arm,loss) in enumerate((a,l) for a in run.ARMS for l in run.LOSSES):
            for col,seed in enumerate(run.SEEDS):
                path=run.run_path(output,fold,arm,loss,seed);h=pd.read_csv(path/'history.csv');s=json.loads((path/'summary.json').read_text())
                for kind in (0,1):
                    ax=axes[row,2*col+kind]
                    fit,val=('fit_auc','selection_auc') if kind==0 else ('fit_logloss','selection_logloss')
                    ax.plot(h.epoch,h[fit],label='fit');ax.plot(h.epoch,h[val],label='selección');ax.axvline(s['selected_epoch'],linestyle=':',color='gray')
                    ax.set_title(f'{arm} {loss} seed{seed} · '+('AUC paciente' if kind==0 else 'Logloss paciente'),fontsize=8)
                    if kind==0:ax.set_ylim(.3,1.)
                    ax.legend(fontsize=6);ax.grid(alpha=.15);ax.set_xlabel('Época')
        fig.suptitle(f'Fold{fold}: época seleccionada marcada; pérdidas en pacientes sin aumento, misma escala en fit/selección')
        fig.tight_layout(rect=(0,0,1,.96));fig.savefig(output/f'04_aprendizaje_fold_{fold}.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,(name,y,pred) in zip(axes,[(arm,f.label,f.prediction) for arm,f in frames.items()]+[('Siempre no pCR',frames['control'].label,np.zeros(439,dtype=int))]):
        cm=confusion_matrix(y,pred,labels=[0,1]);ax.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black',fontsize=15)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['no pCR','pCR'],yticklabels=['no pCR','pCR'],xlabel='Predicción',ylabel='Real',title=name)
    fig.suptitle('Políticas congeladas en calibración, especificidad mínima95 % allí; no garantizada fuera')
    fig.tight_layout();fig.savefig(output/'05_matrices.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    for arm,f in frames.items():
        observed,predicted=calibration_curve(f.label,f.probability_calibrated,n_bins=5,strategy='quantile')
        ax.plot(predicted,observed,'o-',label=arm,color=colors[arm])
    ax.plot([0,1],[0,1],':',color='gray');ax.set(xlabel='Probabilidad predicha',ylabel='Proporción pCR',title='Calibración exterior interna · cinco grupos');ax.legend();ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(output/'06_calibracion.png',dpi=150);plt.close(fig)


def report(output,result,summary):
    a=summary['summaries'];ci=summary['intervals'];audit=json.loads((output/'auditoria/auditoria.json').read_text())
    rows=[]
    for arm,s in a.items():
        p=s['frozen_policy'];rows.append(f'| {arm} | {s["fold_auc"]["0"]:.4f} | {s["fold_auc"]["3"]:.4f} | **{s["mean_fold_auc"]:.4f} ± {s["std_fold_auc"]:.4f}** | {s["pooled_raw"]["roc_auc"]:.4f} | {s["pooled_calibrated"]["roc_auc"]:.4f} | {p["macro_f1"]:.4f} |')
    text=f'''# Robustez de adquisición: piloto de la CNN cinética por paciente

{run.now()}. Solo train. CNN propia 2D V007, **593.521 parámetros**, desde cero y sin clínica. Dieciséis entrenamientos completados; no se ha abierto test ni validation privada.

## Resultado principal

| Brazo | AUC fold0 | AUC fold3 | Media ± DE entre dos folds | AUC agrupada cruda | AUC agrupada calibrada | Macro-F1 política |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

La unidad es la paciente: media de todos sus cortes, y ensemble de dos semillas. Pérdida elegida únicamente en selección interna. Son **439 pacientes, {summary['positive']} pCR**. La AUC agrupada cruda mezcla escalas de folds y queda como secundaria.

Δ media campo−control = **{summary['gate']['delta_mean_auc']:+.4f}**, IC95 % pareado **[{ci['delta_mean_fold_auc_ci95'][0]:+.4f}; {ci['delta_mean_fold_auc_ci95'][1]:+.4f}]**. Intervalos condicionales a modelos fijos, 4.000 bootstrap por fold/cohorte/clase; no corrigen todo el historial de búsqueda. No se comparan estas AUC directamente con el 0,6084 del test histórico.

**Filtro prefijado para ampliar: {'APROBADO, solo habilita réplica interna' if summary['gate']['passed'] else 'NO APROBADO; no ampliar esta hipótesis'}.**

{chr(10).join('- '+k+': '+str(v) for k,v in summary['gate']['conditions'].items())}

No se declara logrado un AUC independiente de0,60/0,70 por este piloto. Una cifra de selección o de un fold no constituye ese resultado. No se activa ningún candidato ni se reajusta después de comprobar.

## Datos entendidos antes de entrenar

Se auditaron los índices de1.097 pacientes de train y se inspeccionó un corte central disponible por paciente, sin convertirlo en unidad de evaluación. El realce medio EARLY−PRE fue positivo en **{audit['early_gt_pre_fraction']:.2%}**; LATE disminuyó respecto a EARLY en **{audit['washout_fraction']:.2%}** de esos cortes. La máscara de foreground es solo una selección de píxeles no negros: no delimita el tumor.

Las prevalencias fueron Duke21,05 %, I-SPY1 25 % e I-SPY2 32,14 %. La AUC OOF histórica agrupada de V007 fue0,5264 y la de un diagnóstico basado únicamente en tasas de cohorte de otros folds fue{audit['cohort_only_oof_auc']:.4f}. La similitud no demuestra causalmente que la CNN use atajos. La media de folds histórica fue0,5766; no confundirla con la agrupada.

![Errores OOF históricos y fases](auditoria/01_errores_oof.png)

Se ven diferencias de adquisición, algunos desplazamientos entre fases y casos con LATE especialmente brillante. Son observaciones exploratorias; no se excluyen pacientes ni se renormalizan fases por separado. No hay contornos radiológicos para afirmar si la red atiende correctamente al tumor.

## Único cambio experimental

Control frente a campo suave de intensidad compartido. Una rejilla aleatoria4×4 por paciente/época se interpola bicúbicamente; el campo se mantiene en[0,85;1] y se aplica con probabilidad0,5 a las tres fases y todos sus cortes. Después se aplica la geometría habitual. No cambia la arquitectura, ni LR, WD, agregación, etiquetas o datos. La atenuación evita añadir saturación; preserva EARLY′−PRE′=B·(EARLY−PRE). Se midió un error máximo numérico de{audit['augmentation_check']['enhancement_max_error']:.2g} en esa identidad.

Es una adaptación aleatoria inspirada en [Chen et al., MICCAI2020](https://arxiv.org/abs/2006.13322), que estudian segmentación de RM cardíaca y campos adversariales. No es una reproducción de ese método ni evidencia previa de mejorar pCR.

![Comprobación del aumento](auditoria/02_campo_compartido.png)

## Protocolo, entrenamiento y separación

Folds originales0/3 como comprobación de cada modelo. En fold0:614 fit,132 selección,132 calibración,219 comprobación. En fold3:613/132/132/220. Roles disjuntos por paciente, estratificados por cohorte/clase. Entre folds una paciente puede entrenar el modelo de otro fold, como en validación cruzada; nunca el que genera su predicción de comprobación.

LR0,0003, WD0,01, AdamW, lote32, máximo40 épocas, mínimo8, paciencia8 y margen0,001. Semillas42/2026 y mismas inicializaciones y aumentos geométricos en brazos emparejados. BCEWithLogitsLoss normal y ponderada con N0/N1 de pacientes de fit; BCE por corte con contribución total igual por paciente y normalizador fijo. La selección de época usa AUC por paciente sin aumento, no AUC por corte.

Se congelaron las dieciséis épocas seleccionadas y las cuatro elecciones de pérdida antes de calibración. Calibración PlattC1 con pendiente no negativa en pacientes separados. Se congelaron las ocho políticas antes de comprobar. Los hashes se verifican al terminar. El script de informe solo lee resultados; no ejecuta inferencia ni decide otro umbral.

Pasos AdamW reales: **{summary['optimizer_steps']}**; pasos omitidos AMP: **{summary['skipped_steps']}**. Pesos/pérdidas finitos: **{summary['finite_all']}**. Tiempo acumulado de épocas: **{summary['training_seconds']/60:.1f} minutos**. GPU, memoria, versión de entorno y fuentes están en entorno.json/protocolo.json. Las99 pruebas pasaron antes de entrenar. Se corrigió previamente una incompatibilidad de pandas3 en el script de auditoría; no se modificó una ejecución en marcha.

## Sobreajuste

'''
    for arm,s in a.items():text+=f'- {arm}: brecha fit−selección mediana en épocas elegidas **{s["median_selected_gap"]:+.4f}**; al final **{s["median_final_gap"]:+.4f}**.\n'
    text+='''
Las pérdidas de las figuras son logloss de probabilidades medias por paciente, evaluadas sin aumento y con la misma definición en fit/selección. La BCE ponderada de optimización se conserva aparte en history.csv y no se presenta como directamente comparable con una BCE normal. La línea vertical marca la época elegida; una brecha pequeña no basta para afirmar buen aprendizaje.

![Curvas fold0](04_aprendizaje_fold_0.png)
![Curvas fold3](04_aprendizaje_fold_3.png)

## Normal frente a ponderada

| Brazo | BCE | AUC media folds | Sensibilidad media con política congelada | Especificidad media |
|---|---|---:|---:|---:|
'''
    for row in summary['normal_weighted']:text+=f'| {row["arm"]} | {row["loss"]} | {row["mean_fold_auc"]:.4f} | {row["sensitivity_mean_frozen_policy"]:.2%} | {row["specificity_mean_frozen_policy"]:.2%} |\n'
    text+='''
La política de cada receta se ajusta por separado solo en sus132 pacientes de calibración. Estas filas no se usan para escoger otro ganador después de ver la comprobación. No asumir que pos_weight mejora sensibilidad al mantener otro umbral: entrenamiento y decisión son pasos diferentes.

## Umbral, F1 y utilidad académica

La regla principal maximiza sensibilidad entre umbrales con especificidad≥95 % en calibración. Busca reducir falsos optimismos pCR en un uso hipotético, sin autorizar decisiones clínicas. Esa especificidad no está garantizada en pacientes nuevos. La regla secundaria de máximo macro-F1 también quedó fijada en calibración y se conserva por transparencia; no sustituye a la principal según el resultado exterior.

| Política | TN | FP | FN | TP | Sensibilidad | Especificidad | Precisión pCR | F1 pCR | F1 no pCR | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    policies={arm:s['frozen_policy'] for arm,s in a.items()};policies['Siempre no pCR']=summary['always_no_pcr']
    for arm,p in policies.items():text+=f'| {arm} | {p["tn"]} | {p["fp"]} | {p["fn"]} | {p["tp"]} | {p["sensitivity"]:.2%} | {p["specificity"]:.2%} | {p["precision_pcr"]:.2%} | {p["f1_pcr"]:.4f} | {p["f1_no_pcr"]:.4f} | {p["macro_f1"]:.4f} |\n'
    text+='''
![Matrices](05_matrices.png)

## Discriminación y calibración

| Brazo | AP cruda / calibrada | Brier crudo / calibrado | Logloss crudo / calibrado |
|---|---:|---:|---:|
'''
    for arm,s in a.items():
        raw,cal=s['pooled_raw'],s['pooled_calibrated'];text+=f'| {arm} | {raw["average_precision"]:.4f} / {cal["average_precision"]:.4f} | {raw["brier"]:.4f} / {cal["brier"]:.4f} | {raw["log_loss"]:.4f} / {cal["log_loss"]:.4f} |\n'
    text+='''
![ROC y PR](03_roc_pr.png)
![Calibración](06_calibracion.png)

Por cohorte, la siguiente tabla es exploratoria, con muestras pequeñas y escalas de dos folds; no sirve para excluir una cohorte ni crear modelos específicos tras verla.

| Brazo | Cohorte | N | pCR | AUC cruda / calibrada |
|---|---|---:|---:|---:|
'''
    for arm,s in a.items():
        for cohort,c in s['cohorts'].items():text+=f'| {arm} | {cohort} | {c["n"]} | {c["pcr"]} | {c["raw_auc"]:.4f} / {c["calibrated_auc"]:.4f} |\n'
    text+='''
## Límites y entrega

Esta comprobación interna reutiliza train con amplio historial de desarrollo. Dos folds no completan la validación cruzada de cinco folds exigida para una entrega final y no prueban generalización independiente. El test histórico no se ha vuelto a cargar ni ha elegido ningún parámetro, umbral, exclusión o siguiente entrenamiento.

V007 y V008, versiones anteriores y aplicación quedan intactos. Los pesos de trabajo y checkpoints permanecen locales, fuera de Git; el código, configuración, hashes, curvas e informe permiten revisar el ensayo. No se han publicado ramas ni resultados remotos. El resultado insuficiente se conserva como tal; no se fuerza un0,60 ajustando sobre la comprobación.
'''
    (output/'INFORME_RESULTADOS.md').write_text(text)


def main():
    output=run.OUTPUT;run.verify(output);result=json.loads((output/'resultado.json').read_text())
    configs=[];histories=[]
    for item in result['runs']:
        path=run.run_path(output,item['fold'],item['arm'],item['loss'],item['seed'])
        cfg=json.loads((path/'config.json').read_text())
        if base.sha256(path/'config.json')!=item['config_sha256']:raise ValueError('Configuración alterada')
        configs.append({'summary':item,'config':cfg})
        h=pd.read_csv(path/'history.csv')
        for key in ('fold','arm','loss','seed'):h[key]=item[key]
        histories.append(h)
    run.write(output/'ejecuciones.json',configs);pd.concat(histories).to_csv(output/'historias.csv',index=False)
    frames=load_selected(output,result);summary=summarize(output,result,frames)
    run.write(output/'comparacion.json',summary);plots(output,result,frames);report(output,result,summary)
    files=[p for p in output.rglob('*') if p.is_file() and 'runs' not in p.parts and p.name!='indice_entrega.json']
    run.write(output/'indice_entrega.json',{'created_at':run.now(),'files':{str(p.relative_to(output)):base.sha256(p) for p in files},'weights':{str((run.run_path(output,r['fold'],r['arm'],r['loss'],r['seed'])/'inference.pt').relative_to(output)):r['weights_sha256'] for r in result['runs']}})
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
