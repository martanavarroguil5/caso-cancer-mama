"""Validación final de una búsqueda interna prefijada; outer se abre al final."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from cancer_mama import imagen_mil as m
from cancer_mama import entrenamiento as base

FAMILIES=['hybrid_v1','hybrid_rank_v2','hybrid_mlp_stable_v4']
SEEDS=[42,2026,3407]


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-30,30)))


def logit(p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6)
    return np.log(p/(1-p))


def fit_calibration(p,y):
    z=logit(p); mean=float(z.mean());std=max(float(z.std()),1e-4)
    reg=LogisticRegression(C=1.,solver='lbfgs').fit(((z-mean)/std)[:,None],y)
    # No se permite invertir la señal con la etiqueta de validación.
    return dict(mean=mean,std=std,coef=max(float(reg.coef_[0,0]),0.),intercept=float(reg.intercept_[0]))


def calibrate(p,c):
    return sigmoid(c['intercept']+c['coef']*(logit(p)-c['mean'])/c['std'])


def average_frames(paths):
    frames=[pd.read_csv(path).sort_values('patient_id').reset_index(drop=True) for path in paths]
    for f in frames[1:]:
        for col in ['patient_id','pCR','fold']:
            if not f[col].equals(frames[0][col]):
                raise ValueError('Predicciones no alineadas')
    out=frames[0].copy();out['probability']=np.mean([f.probability for f in frames],axis=0)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=m.OUTPUT)
    parser.add_argument('--reference',type=Path,default=m.OUTPUT)
    args=parser.parse_args()
    output=args.output
    configs={name:json.loads((args.reference/'runs'/name/'fold_0/seed_42/config.json').read_text()) for name in FAMILIES}
    m.prepare(output);m.prepare_roi_features(output)
    from cancer_mama import imagen_textura
    imagen_textura.prepare(output)
    plan=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),families=FAMILIES,
              seeds=SEEDS,folds=list(range(5)),configurations=configs,
              selector='highest inner ensemble AUC among fit-minus-selection AUC gap <= 0.12; tie lower BCE',
              gap_limit=.12,calibration='Platt, inner selection only, standardized ensemble logit, C=1, nonnegative slope',
              outer_policy='all training and all five choices fixed before any outer predictions',
              test_images_loaded=0,source_sha256=base.sha256(Path(m.__file__)))
    freeze=output/'final_plan_locked.json'
    if freeze.exists():
        old=json.loads(freeze.read_text())
        for key in ['families','seeds','configurations','source_sha256']:
            if old[key]!=plan[key]:
                raise ValueError('Plan final ya fijado e incompatible')
        plan=old
    else:
        m.write_json(freeze,plan)
    data={}
    for cfg in configs.values():
        key=cfg.get('feature_file') or ('features_roi_v2.npy' if cfg.get('roi_features') else 'features.npy')
        if key not in data:
            data[key]=m.load_cache(feature_file=key)
    # TODAS las decisiones internas preceden a TODAS las evaluaciones exteriores.
    for fold in range(5):
        for name,cfg in configs.items():
            key=cfg.get('feature_file') or ('features_roi_v2.npy' if cfg.get('roi_features') else 'features.npy')
            for seed in SEEDS:
                m.train(data[key],cfg,name,fold,seed,evaluate_outer=False)
    choices=[];candidate_rows=[]
    for fold in range(5):
        candidates=[]
        for name,cfg in configs.items():
            runs=[output/'runs'/name/f'fold_{fold}'/f'seed_{seed}' for seed in SEEDS]
            f=average_frames([run/'fit_predictions.csv' for run in runs])
            v=average_frames([run/'selection_predictions.csv' for run in runs])
            fit_metrics=m.metrics(f.pCR,f.probability);val_metrics=m.metrics(v.pCR,v.probability)
            gap=fit_metrics['auc']-val_metrics['auc']
            row=dict(fold=fold,family=name,fit_auc=fit_metrics['auc'],selection_auc=val_metrics['auc'],
                     selection_bce=val_metrics['bce'],auc_gap=gap,eligible=bool(gap<=plan['gap_limit']))
            candidates.append(row);candidate_rows.append(row)
        eligible=[c for c in candidates if c['eligible']]
        if not eligible:
            raise RuntimeError(f'Fold {fold}: ningún candidato supera la restricción de sobreajuste')
        chosen=sorted(eligible,key=lambda r:(-r['selection_auc'],r['selection_bce']))[0]
        runs=[output/'runs'/chosen['family']/f'fold_{fold}'/f'seed_{seed}' for seed in SEEDS]
        v=average_frames([run/'selection_predictions.csv' for run in runs])
        chosen['calibration']=fit_calibration(v.probability,v.pCR)
        chosen['checkpoints']=[dict(file=str(run/'best.pt'),sha256=base.sha256(run/'best.pt'),seed=seed)
                               for run,seed in zip(runs,SEEDS)]
        choices.append(chosen)
    pd.DataFrame(candidate_rows).to_csv(output/'final_inner_candidates.csv',index=False)
    m.write_json(output/'final_choices_locked.json',dict(plan_sha256=base.sha256(freeze),choices=choices,
                 locked_utc=datetime.now(timezone.utc).isoformat(),test_images_loaded=0))
    print('DECISIONES FIJADAS ANTES DE OUTER:',json.dumps(choices,indent=2),flush=True)
    outer_frames=[]
    for chosen in choices:
        fold,name=chosen['fold'],chosen['family'];cfg=configs[name]
        key=cfg.get('feature_file') or ('features_roi_v2.npy' if cfg.get('roi_features') else 'features.npy')
        patients,images,masks,features=data[key]
        _,_,outer=m.split_indices(patients,fold)
        predictions=[]
        for entry in chosen['checkpoints']:
            if base.sha256(entry['file'])!=entry['sha256']:
                raise ValueError('Checkpoint modificado tras el cierre de selección')
            ckpt=torch.load(entry['file'],map_location=features.device,weights_only=True)
            model=m.PatientNet(ckpt['config'],ckpt['n_features']).to(features.device)
            model.load_state_dict(ckpt['state_dict'])
            p=m.predict(model,images,masks,features,outer)
            predictions.append(p)
            f=patients.iloc[outer].copy();f['probability']=p
            f.to_csv(output/f'outer_{name}_fold{fold}_seed{entry["seed"]}.csv',index=False)
        frame=patients.iloc[outer].copy();frame['probability_raw']=np.mean(predictions,axis=0)
        frame['probability']=calibrate(frame.probability_raw,chosen['calibration'])
        frame['selected_family']=name
        outer_frames.append(frame)
        print(f'OUTER {fold} {name}: {m.metrics(frame.pCR,frame.probability)}',flush=True)
    oof=pd.concat(outer_frames).sort_values('patient_id').reset_index(drop=True)
    if oof.patient_id.duplicated().any() or len(oof)!=1097:
        raise ValueError('OOF incompletas o pacientes duplicadas')
    oof.to_csv(output/'final_oof_patients.csv',index=False)
    result=dict(status='complete',patients=len(oof),test_images_loaded=0,
        calibrated=m.metrics(oof.pCR,oof.probability),raw=m.metrics(oof.pCR,oof.probability_raw),
        folds=[dict(fold=int(fold),**m.metrics(f.pCR,f.probability)) for fold,f in oof.groupby('fold')],
        cohorts=[dict(cohort=cohort,patients=len(f),**m.metrics(f.pCR,f.probability)) for cohort,f in oof.groupby('dataset')],
        choices_sha256=base.sha256(output/'final_choices_locked.json'),oof_sha256=base.sha256(output/'final_oof_patients.csv'))
    result['mean_fold_auc']=float(np.mean([f['auc'] for f in result['folds']]))
    result['sd_fold_auc']=float(np.std([f['auc'] for f in result['folds']],ddof=1))
    m.write_json(output/'final_metrics.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    main()
