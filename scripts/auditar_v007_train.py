"""Inspección de train y errores OOF históricos, sin inferencia sobre test."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from cancer_mama import robustez_adquisicion as run
from cancer_mama import cierre_cinetico as close
from cancer_mama import entrenamiento as base
from cancer_mama.paths import DATA_DIR, PROJECT_ROOT


def main():
    torch.set_num_threads(4)
    output=run.OUTPUT/'auditoria';output.mkdir(parents=True,exist_ok=True)
    samples=base.cargar_train(DATA_DIR,include_clinical=False).sort_values(['patient_id','sample_id']).reset_index(drop=True)
    source=PROJECT_ROOT/'resultados/04_entrenamiento/ajuste_cnn_cinetica_20261021/oof_pacientes_selected.csv'
    oof=pd.read_csv(source).sort_values('patient_id').reset_index(drop=True)
    patients=samples[['patient_id','pCR','dataset','fold']].drop_duplicates().sort_values('patient_id').reset_index(drop=True)
    if not oof.patient_id.equals(patients.patient_id) or not np.array_equal(oof.label,patients.pCR) or not np.array_equal(oof.fold,patients.fold):raise ValueError('OOF históricas no corresponden a train')
    if len(patients)!=1097:raise ValueError('Población de train distinta')
    # Un corte central disponible por paciente para inspección, nunca como evaluación principal.
    middle=samples.groupby('patient_id').cumcount().eq(samples.groupby('patient_id').patient_id.transform('size')//2)
    representatives=samples[middle].reset_index(drop=True)
    pixels=close.load_pixels(representatives,torch.device('cpu')).float()/255
    mask=(pixels.amax(1)>.05).float();den=mask.sum((1,2)).clamp_min(1)
    means=(pixels*mask[:,None]).sum((2,3))/den[:,None]
    stats=representatives[['patient_id','sample_id','pCR','dataset','fold']].copy()
    for i,phase in enumerate(('pre','early','late')):stats['mean_'+phase]=means[:,i].numpy()
    stats['early_minus_pre']=stats.mean_early-stats.mean_pre
    stats['late_minus_early']=stats.mean_late-stats.mean_early
    stats['foreground_fraction']=mask.mean((1,2)).numpy()
    stats['zero_fraction']=(pixels==0).float().mean((1,2,3)).numpy()
    stats['one_fraction']=(pixels==1).float().mean((1,2,3)).numpy()
    stats=stats.merge(oof[['patient_id','probability_raw','prediction','threshold']],on='patient_id',validate='one_to_one')
    stats.to_csv(output/'inspeccion_train.csv',index=False)
    cohort=[];priors=np.zeros(len(patients))
    for fold in range(5):
        fit=patients[patients.fold.ne(fold)];counts=fit.groupby('dataset').pCR.agg(['sum','count'])
        for i in np.flatnonzero(patients.fold.eq(fold)):
            row=counts.loc[patients.iloc[i].dataset];priors[i]=(row['sum']+1)/(row['count']+2)
    for name,g in oof.groupby('cohort'):
        s=stats[stats.dataset.eq(name)]
        cohort.append({'cohort':name,'patients':len(g),'pcr':int(g.label.sum()),'prevalence':float(g.label.mean()),
            'oof_raw_auc':float(roc_auc_score(g.label,g.probability_raw)),
            'mean_score':float(g.probability_raw.mean()),'early_gt_pre_fraction':float(s.early_minus_pre.gt(0).mean()),
            'median_enhancement':float(s.early_minus_pre.median()),'median_foreground':float(s.foreground_fraction.median())})
    summary={'created_at':run.now(),'scope':'Solo1097 pacientes de train; corte central disponible para inspección, OOF históricas para métricas',
        'source':str(source.relative_to(PROJECT_ROOT)),'source_sha256':base.sha256(source),'patients':1097,'slices_metadata':len(samples),
        'oof_pooled_raw_auc':float(roc_auc_score(oof.label,oof.probability_raw)),
        'oof_mean_fold_auc':float(np.mean([roc_auc_score(g.label,g.probability_raw) for _,g in oof.groupby('fold')])),
        'cohort_only_oof_auc':float(roc_auc_score(patients.pCR,priors)),
        'cohort_only_note':'Diagnóstico descriptivo, tasa de cohorte calculada en otros cuatro folds; no modelo principal ni prueba causal de atajos',
        'cohorts':cohort,'early_gt_pre_fraction':float(stats.early_minus_pre.gt(0).mean()),
        'washout_fraction':float(stats.late_minus_early.lt(0).mean()),'test_images_loaded':0,'private_images_loaded':0,
        'limits':['No hay segmentación tumoral en estos PNG; el foreground no es una máscara de tumor',
            'La normalización mincrop y las OOF tienen antecedentes adaptativos','Las imágenes no permiten atribuir por sí solas la causa de un fallo']}
    run.write(output/'auditoria.json',summary)
    # Un falso positivo y un falso negativo de cada cohorte, sin inventar diagnósticos visuales.
    chosen=[]
    for cohort_name,g in oof.groupby('cohort'):
        for label,prediction in ((0,1),(1,0)):
            failed=g[g.label.eq(label)&g.prediction.eq(prediction)].copy()
            failed['severity']=np.abs(failed.label-failed.probability_calibrated)
            if len(failed):chosen.append(failed.sort_values(['severity','patient_id'],ascending=[False,True]).iloc[0])
    chosen=pd.DataFrame(chosen);chosen.to_csv(output/'casos_fallados.csv',index=False)
    fig,axes=plt.subplots(len(chosen),4,figsize=(11,2.7*len(chosen)),squeeze=False)
    index={pid:i for i,pid in enumerate(representatives.patient_id)}
    for row,case in enumerate(chosen.itertuples()):
        x=pixels[index[case.patient_id]].numpy()
        for col,(array,title) in enumerate(zip([*x,x[1]-x[0]],['PRE','EARLY','LATE','EARLY−PRE'])):
            axes[row,col].imshow(array,cmap='gray' if col<3 else 'coolwarm',vmin=0 if col<3 else -.5,vmax=1 if col<3 else .5)
            axes[row,col].axis('off');axes[row,col].set_title(title,fontsize=9)
        axes[row,0].text(-.03,.5,f'{case.cohort}\n{case.patient_id}\nreal={case.label}; pred={case.prediction}',transform=axes[row,0].transAxes,ha='right',va='center',fontsize=8)
    fig.suptitle('Errores OOF de V007: corte central disponible, no mapa de tumor',fontsize=13)
    fig.text(.5,.008,'Datos BreastDCEDL MinCrop · CC BY-NC 4.0 · Uso educativo · Solo train',ha='center',fontsize=8)
    fig.tight_layout(rect=(.13,.02,1,.97));fig.savefig(output/'01_errores_oof.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,6));x=pixels[0:1];field=run.fields(1,256,torch.Generator().manual_seed(42),probability=1);perturbed=x*field
    for i,phase in enumerate(('PRE','EARLY','LATE')):
        for j,tensor in enumerate((x,perturbed)):
            axes[j,i].imshow(tensor[0,i],cmap='gray',vmin=0,vmax=1);axes[j,i].axis('off');axes[j,i].set_title(phase+(' original' if j==0 else ' mismo campo'))
    fig.suptitle('Aumento prefijado: mismo campo suave para todas las fases; sin clipping de imágenes')
    fig.text(.5,.01,'BreastDCEDL MinCrop · CC BY-NC 4.0 · Solo train · No representa respuesta al tratamiento',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,.95));fig.savefig(output/'02_campo_compartido.png',dpi=140);plt.close(fig)
    expected=(x[:,1]-x[:,0])*field[:,0]
    summary['augmentation_check']={'field_min':float(field.min()),'field_max':float(field.max()),'enhancement_max_error':float(((perturbed[:,1]-perturbed[:,0])-expected).abs().max()),'new_saturation':0}
    run.write(output/'auditoria.json',summary)
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
