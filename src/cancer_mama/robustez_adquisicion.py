"""Piloto de una hipótesis de adquisición; CNN V007 y media por paciente.

Solo train. Control/campo suave, ambas BCE, dos semillas, folds 0/3.
Selección, calibración y comprobación tienen pacientes separados por modelo.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch import nn
from torch.nn import functional as F

from . import entrenamiento as base
from . import entrenamiento_cinetico as cnn
from . import entrenamiento_imagen as image
from . import cierre_cinetico as close
from .paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

OUTPUT = RESULTS_DIR/'25_robustez_adquisicion_20261025'
ARMS, LOSSES, SEEDS, FOLDS = ('control','campo_suave'), ('normal','weighted'), (42,2026), (0,3)
STRENGTH, PROBABILITY, GRID, SPLIT_SEED = .15, .5, 4, 20261025
MIN_SPECIFICITY = .95


def now(): return close.now()
def write(path, value): base.json_write(Path(path), value)


def fields(n, size, generator, strength=STRENGTH, probability=PROBABILITY, grid=GRID):
    """Campo positivo [1-strength,1], sin clipping de imágenes ni cambio por fase."""
    if n < 1 or size < 2 or not 0 <= strength < 1 or not 0 <= probability <= 1 or grid < 2:
        raise ValueError('Configuración de campo inválida')
    controls = torch.rand(n,1,grid,grid,generator=generator)
    smooth = F.interpolate(controls,size=(size,size),mode='bicubic',align_corners=False).clamp(0,1)
    active = (torch.rand(n,1,1,1,generator=generator) < probability).float()
    return 1-strength*smooth*active


def aggregate(frame, probabilities):
    if frame.groupby('patient_id')[['pCR','dataset','fold']].nunique().gt(1).any().any():
        raise ValueError('Paciente con etiqueta, cohorte o fold inconsistentes')
    return image.patient_predictions(frame,np.asarray(probabilities))


def split_roles(samples, outer_fold):
    if not samples.split.eq('train').all() or outer_fold not in range(5):
        raise ValueError('Solo train y folds 0–4')
    patients = samples[['patient_id','pCR','dataset','fold']].drop_duplicates().sort_values('patient_id').reset_index(drop=True)
    if patients.patient_id.duplicated().any(): raise ValueError('Metadatos de paciente inconsistentes')
    outer = patients[patients.fold.eq(outer_fold)].copy(); outer['role']='outer'
    remaining = patients[patients.fold.ne(outer_fold)].copy().reset_index(drop=True)
    count = round(len(remaining)*.15); parts=[outer]
    for offset,role in enumerate(('selection','calibration')):
        strata=remaining.dataset.astype(str)+'|'+remaining.pCR.astype(str)
        a,b=next(StratifiedShuffleSplit(n_splits=1,test_size=count,random_state=SPLIT_SEED+outer_fold*10+offset).split(remaining.patient_id,strata))
        part=remaining.iloc[b].copy();part['role']=role;parts.append(part)
        remaining=remaining.iloc[a].reset_index(drop=True)
    remaining['role']='fit';parts.append(remaining)
    result=pd.concat(parts).sort_values('patient_id').reset_index(drop=True)
    if len(result)!=len(patients) or result.patient_id.duplicated().any():raise ValueError('Roles solapados')
    if any(g.pCR.nunique()!=2 for _,g in result.groupby('role')):raise ValueError('Clase ausente')
    return result


def config():
    cfg=cnn.configuracion()
    cfg['training'].update(lr=.0003,weight_decay=.01,num_workers=0,cpu_threads=4)
    cfg['selection_metric']='patient_mean_raw_auc'
    return cfg


def state_hash(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare(output):
    output=Path(output)
    if output.exists():raise ValueError('Salida existente: no cambiar un protocolo')
    samples=base.cargar_train(DATA_DIR,include_clinical=False)
    cfg=config();manifest=json.loads((close.V007/'manifiesto.json').read_text())
    if any(m['model_config']!=cfg['model'] for m in manifest['models']):raise ValueError('Arquitectura distinta de V007')
    if image.numero_parametros(cnn.CNNMultiescalaCinetica(cfg))!=593521:raise ValueError('Número de parámetros distinto')
    output.mkdir(parents=True)
    inventories=close.previous_versions()
    sources={str(Path(m.__file__).relative_to(PROJECT_ROOT)):base.sha256(m.__file__) for m in (base,cnn,image,close,__import__(__name__,fromlist=['_']))}
    for name in ('src/cancer_mama/datos.py','breastdcedl/metadata/samples.csv','breastdcedl/metadata/patients.csv'):
        sources[name]=base.sha256(PROJECT_ROOT/name)
    splits={}
    for fold in FOLDS:
        path=output/f'roles_fold_{fold}.csv';roles=split_roles(samples,fold);roles.to_csv(path,index=False)
        splits[path.name]={'sha256':base.sha256(path),'counts':roles.groupby('role').size().to_dict()}
    protocol={'created_at':now(),'config':cfg,'parameters':593521,'arms':list(ARMS),'losses':list(LOSSES),'seeds':list(SEEDS),'folds':list(FOLDS),'planned_runs':16,
        'augmentation':{'strength':STRENGTH,'probability_per_patient_epoch':PROBABILITY,'control_grid':GRID,'range':[1-STRENGTH,1.],
            'shared':'Una realización por paciente y época, idéntica en fases y cortes; después geometría habitual por corte',
            'implementation':'U[0,1] en rejilla4x4, bicúbica y clamp del CAMPO a[0,1], B=1-.15*campo. Nunca clipping de imágenes'},
        'primary_metric':'Media de AUC cruda por paciente de folds0/3; media de probabilidades de todos los cortes y dos semillas',
        'loss_selection':'Por AUC ensemble en selection de cada brazo/fold; empate normal, antes de calibración y outer',
        'training_objective':'BCEWithLogitsLoss por corte, ponderación1/n_cortes y normalizador fijo; pos_weight por pacientes de fit',
        'calibration':'Platt C=1 y pendiente no negativa; pacientes calibration, después de congelar épocas/pérdidas',
        'threshold':{'primary':'Máxima sensibilidad con especificidad>=.95 en calibration; empates precisión, especificidad, umbral mayor',
            'secondary':'Máximo macroF1 en calibration, solo descriptivo; no seleccionar política con outer',
            'rationale':'Uso hipotético académico: evitar optimismo falso pCR; no recomendación clínica'},
        'checkpoint':'Máximo AUC selection por paciente; paciencia8, mínimo8, máximo40; ninguna selección según outer',
        'replication_gate':{'mean_raw_auc_min':.60,'delta_mean_min':.02,'improve_both_folds':True,'median_selected_gap_increase_max':.01,'paired_ci95_lower_gt':0.,
            'action':'Solo habilita ampliación interna prefijada a cinco folds; no éxito independiente ni apertura de test'},
        'limits':['Todo train tiene antecedentes adaptativos; outer no es test ciego nuevo','No cambiar arquitectura, LR, WD, campo ni política después de resultados',
            'No abrir test/validation ni activar versión/app por este piloto','Sin push ni versiones históricas modificadas'],
        'sources':sources,'previous_versions':inventories,'splits':splits,'test_images_loaded':0,'private_images_loaded':0}
    write(output/'protocolo.json',protocol)
    (output/'protocolo.json.sha256').write_text(base.sha256(output/'protocolo.json')+'\n')
    write(output/'estado.json',{'status':'prepared','completed':0,'planned':16,'updated_at':now()})
    print(json.dumps({'status':'prepared','roles':splits,'parameters':593521},ensure_ascii=False),flush=True)


def verify(output):
    output=Path(output);path=output/'protocolo.json'
    if base.sha256(path)!=(output/'protocolo.json.sha256').read_text().strip():raise ValueError('Protocolo cambiado')
    p=json.loads(path.read_text())
    for name,digest in {**p['sources'],**p['previous_versions']}.items():
        if base.sha256(PROJECT_ROOT/name)!=digest:raise ValueError('Fuente/versión modificada: '+name)
    for name,entry in p['splits'].items():
        if base.sha256(output/name)!=entry['sha256']:raise ValueError('Roles modificados')
    return p


def role_frames(samples,roles):
    return {role:samples[samples.patient_id.isin(g.patient_id)].sort_values(['patient_id','sample_id']).reset_index(drop=True) for role,g in roles.groupby('role')}


def train_run(fit,selection,fit_x,sel_x,cfg,arm,loss_name,seed,fold,path,device,state_path=None):
    if (not fit.split.eq('train').all() or not selection.split.eq('train').all() or
            set(fit.patient_id)&set(selection.patient_id) or arm not in ARMS or loss_name not in LOSSES):
        raise ValueError('Roles de entrenamiento o brazo inválidos')
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    cfg=deepcopy(cfg);cfg.update(arm=arm,loss=loss_name,seed=seed,fold=fold)
    cfg['counts']={'fit_patients':int(fit.patient_id.nunique()),'selection_patients':int(selection.patient_id.nunique()),'fit_slices':len(fit)}
    cfg['bias_field']={'enabled':arm=='campo_suave','strength':STRENGTH,'probability':PROBABILITY,'grid':GRID,'patient_shared':True,'at_inference':False}
    patients=fit.drop_duplicates('patient_id')
    cfg['pos_weight']=float(patients.pCR.eq(0).sum()/patients.pCR.eq(1).sum()) if loss_name=='weighted' else 1.
    if (path/'config.json').exists() and json.loads((path/'config.json').read_text())!=cfg:raise ValueError('Configuración de run cambiada')
    write(path/'config.json',cfg)
    if (path/'summary.json').exists():
        r=json.loads((path/'summary.json').read_text())
        if r['weights_sha256']!=base.sha256(path/'inference.pt'):raise ValueError('Peso alterado')
        return r
    t=cfg['training'];base.seed_everything(seed,True)
    gen=torch.Generator().manual_seed(seed);aug=torch.Generator().manual_seed(seed+1)
    model=cnn.CNNMultiescalaCinetica(cfg).to(device);initial_hash=state_hash(model)
    opt=torch.optim.AdamW(model.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=t['epochs'],eta_min=t['min_lr'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and t['amp'],init_scale=1024.)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg['pos_weight'],device=device),reduction='none')
    y=torch.tensor(fit.pCR.to_numpy(),dtype=torch.float32,device=device)
    weights=torch.tensor(close.patient_weights(fit),dtype=torch.float32,device=device)
    patient_codes=torch.tensor(pd.factorize(fit.patient_id,sort=True)[0],device=device)
    normalization=len(fit)/len(patients)
    best,best_epoch,bad,start,history=-1.,-1,0,0,[]
    if (path/'last.pt').exists():
        ck=base.load_checkpoint(path/'last.pt',model,opt,scheduler,scaler,gen,aug)
        if base.config_hash(ck['config'])!=base.config_hash(cfg):raise ValueError('Checkpoint incompatible')
        best,best_epoch,bad,start,history=ck['best_auc'],ck['best_epoch'],ck['bad_epochs'],ck['epoch']+1,ck['history']
    started=time.monotonic()
    for epoch in range(start,t['epochs']):
        if epoch>=t['min_epochs'] and bad>=t['patience']:break
        tic=time.monotonic();model.train();order=torch.randperm(len(fit),generator=gen)
        field=None
        if arm=='campo_suave':field=fields(len(patients),256,torch.Generator().manual_seed(seed+900000+epoch)).to(device)
        total=0.;attempts=0;skipped=0
        for idx in order.split(t['batch_size']):
            idx=idx.to(device);x=fit_x[idx].float()/255
            if field is not None:x=x*field[patient_codes[idx]]
            x=base.aumentar_geometria(x,aug,**cfg['augmentation']);opt.zero_grad(set_to_none=True)
            with base.amp_context(device,scaler.is_enabled()):
                losses=criterion(model(x).flatten(),y[idx]);loss=(losses*weights[idx]).mean()*normalization
            if not torch.isfinite(loss):raise FloatingPointError('BCE no finita')
            scaler.scale(loss).backward();scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),t['clip_grad_norm'],error_if_nonfinite=not scaler.is_enabled())
            old_scale=scaler.get_scale();scaler.step(opt);scaler.update()
            skipped+=int(scaler.get_scale()<old_scale);attempts+=1;total+=float(loss.detach())*len(idx)
        del field
        fit_p=close.predict(model,fit_x);sel_p=close.predict(model,sel_x)
        fa,sa=aggregate(fit,fit_p),aggregate(selection,sel_p)
        fm=image.metricas_binarias(fa.label,fa.probability_raw);sm=image.metricas_binarias(sa.label,sa.probability_raw)
        substantial=sm['roc_auc']>best+t['min_delta'];bad=0 if substantial else bad+1
        if sm['roc_auc']>best+1e-12:
            best,best_epoch=sm['roc_auc'],epoch;close.save_inference(path/'inference.pt',model,cfg)
        scheduler.step()
        row={'epoch':epoch+1,'train_bce_objective':total/len(fit),'fit_auc':fm['roc_auc'],'selection_auc':sm['roc_auc'],
            'fit_logloss':fm['log_loss'],'selection_logloss':sm['log_loss'],'gap':fm['roc_auc']-sm['roc_auc'],
            'attempts':attempts,'skipped':skipped,'seconds':time.monotonic()-tic,'lr':opt.param_groups[0]['lr']}
        history.append(row);pd.DataFrame(history).to_csv(path/'history.csv',index=False)
        base.save_checkpoint(path/'last.pt',model,opt,scheduler,scaler,epoch,best,bad,cfg,gen,aug,history,best_epoch)
        if state_path:
            state=json.loads(Path(state_path).read_text());state.update(epoch=epoch+1,selection_auc=sm['roc_auc'],gap=row['gap'],epoch_seconds=row['seconds'],updated_at=now());write(state_path,state)
        print(f'fold={fold} {arm} {loss_name} seed={seed} epoch={epoch+1} fit={fm["roc_auc"]:.4f} sel={sm["roc_auc"]:.4f} gap={row["gap"]:+.4f} {row["seconds"]:.1f}s',flush=True)
    artifact=torch.load(path/'inference.pt',map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in artifact['state_dict'].values()):raise FloatingPointError('Peso no finito')
    model.load_state_dict(artifact['state_dict']);selected=aggregate(selection,close.predict(model,sel_x));selected.to_csv(path/'selection.csv',index=False)
    attempted=sum(r['attempts'] for r in history);skipped=sum(r['skipped'] for r in history)
    steps={int(v['step'].item()) for v in opt.state.values()}
    if steps!={attempted-skipped} or skipped>attempted*.01:raise RuntimeError('AdamW/AMP no progresa correctamente')
    result={'status':'complete','arm':arm,'loss':loss_name,'seed':seed,'fold':fold,'epochs':len(history),'selected_epoch':best_epoch+1,
        'selected_auc':best,'selected_gap':history[best_epoch]['gap'],'final_gap':history[-1]['gap'],'seconds':sum(r['seconds'] for r in history),
        'optimizer_steps':attempted-skipped,'skipped':skipped,'finite':True,'initial_state_sha256':initial_hash,
        'weights_sha256':base.sha256(path/'inference.pt'),'config_sha256':base.sha256(path/'config.json')}
    write(path/'summary.json',result)
    del model,opt,scheduler,scaler
    return result


def run_path(output,fold,arm,loss,seed):return Path(output)/'runs'/f'fold_{fold}'/arm/loss/f'seed_{seed}'


def freeze_choices(output):
    choices=[]
    for fold in FOLDS:
        for arm in ARMS:
            candidates=[]
            for loss in LOSSES:
                paths=[run_path(output,fold,arm,loss,seed) for seed in SEEDS]
                frames=[pd.read_csv(p/'selection.csv') for p in paths]
                if not frames[0].patient_id.equals(frames[1].patient_id):raise ValueError('Ensemble desalineado')
                mean=np.mean([f.probability_raw for f in frames],axis=0)
                metric=float(roc_auc_score(frames[0].label,mean))
                candidates.append({'loss':loss,'selection_auc':metric,'models':{str((p/'inference.pt').relative_to(output)):base.sha256(p/'inference.pt') for p in paths}})
            choices.append({'fold':fold,'arm':arm,'candidates':candidates,'selected':max(candidates,key=lambda r:r['selection_auc'])})
    frozen={'frozen_at':now(),'protocol_sha256':base.sha256(Path(output)/'protocolo.json'),'choices':choices,'outer_used':False}
    write(Path(output)/'seleccion_congelada.json',frozen)
    return frozen


def fit_policy(patients):
    calibration=base.fit_platt(patients.label.to_numpy(),patients.probability_raw.to_numpy())
    cal=base.apply_calibration(patients.probability_raw.to_numpy(),calibration)
    threshold=image.threshold_min_specificity(patients.label,cal,MIN_SPECIFICITY)
    secondary=image.threshold_macro_f1(patients.label,cal)
    return {'calibration':calibration,'threshold':threshold,'macro_f1_threshold':secondary,'min_specificity':MIN_SPECIFICITY,
        'fit_data':'calibration_patients_only_after_checkpoint_and_loss_frozen','calibration_n':len(patients),
        'apparent_calibration_metrics':image.metricas_binarias(patients.label,cal,threshold)}


def predict_ensemble(paths,frame,pixels,device):
    probabilities=[]
    for path in paths:
        artifact=torch.load(path,map_location='cpu',weights_only=True)
        model=cnn.CNNMultiescalaCinetica(artifact['model_config']).to(device);model.load_state_dict(artifact['state_dict'])
        probabilities.append(close.predict(model,pixels));del model
    return aggregate(frame,np.mean(probabilities,axis=0))


def execute(output):
    output=Path(output);protocol=verify(output)
    if (output/'resultado.json').exists():raise ValueError('Piloto terminado: no repetir ni ajustar contra outer')
    if not torch.cuda.is_available():raise RuntimeError('GPU requerida')
    device=torch.device('cuda');torch.set_num_threads(4)
    samples=base.cargar_train(DATA_DIR,include_clinical=False).sort_values(['patient_id','sample_id']).reset_index(drop=True)
    state={'status':'training','pid':os.getpid(),'started_at':now(),'completed':0,'planned':16,'test_images_loaded':0,'private_images_loaded':0}
    write(output/'estado.json',state);write(output/'entorno.json',base.environment(device));runs=[]
    for fold in FOLDS:
        frames=role_frames(samples,pd.read_csv(output/f'roles_fold_{fold}.csv'))
        fit_x=close.load_pixels(frames['fit'],device);sel_x=close.load_pixels(frames['selection'],device)
        for arm in ARMS:
            for loss in LOSSES:
                for seed in SEEDS:
                    state.update(fold=fold,arm=arm,loss=loss,seed=seed,epoch=0,updated_at=now());write(output/'estado.json',state)
                    r=train_run(frames['fit'],frames['selection'],fit_x,sel_x,protocol['config'],arm,loss,seed,fold,run_path(output,fold,arm,loss,seed),device,output/'estado.json')
                    runs.append(r);state['completed']+=1;write(output/'estado.json',state)
        del fit_x,sel_x
    for fold in FOLDS:
        for loss in LOSSES:
            for seed in SEEDS:
                a=[r for r in runs if r['fold']==fold and r['loss']==loss and r['seed']==seed]
                if len({r['initial_state_sha256'] for r in a})!=1:raise ValueError('Inicializaciones no emparejadas')
    frozen=freeze_choices(output);verify(output)
    frozen_weights={name:digest for choice in frozen['choices'] for candidate in choice['candidates'] for name,digest in candidate['models'].items()}
    def check_weights():
        for name,digest in frozen_weights.items():
            if base.sha256(output/name)!=digest:raise ValueError('Peso cambiado después de selección: '+name)
    check_weights()
    state.update(status='calibrating',updated_at=now());write(output/'estado.json',state)
    policies=[]
    for fold in FOLDS:
        frames=role_frames(samples,pd.read_csv(output/f'roles_fold_{fold}.csv'));cal_x=close.load_pixels(frames['calibration'],device)
        for arm in ARMS:
            for loss in LOSSES:
                paths=[run_path(output,fold,arm,loss,s)/'inference.pt' for s in SEEDS]
                patients=predict_ensemble(paths,frames['calibration'],cal_x,device)
                policy=fit_policy(patients);policy.update(fold=fold,arm=arm,loss=loss)
                policies.append(policy);patients.to_csv(output/f'calibration_{fold}_{arm}_{loss}.csv',index=False)
        del cal_x
    write(output/'politicas_congeladas.json',{'frozen_at':now(),'policies':policies,'selection_sha256':base.sha256(output/'seleccion_congelada.json')})
    policy_hash=base.sha256(output/'politicas_congeladas.json');(output/'politicas_congeladas.json.sha256').write_text(policy_hash+'\n')
    state.update(status='checking',updated_at=now());write(output/'estado.json',state);outcomes=[]
    check_weights()
    for fold in FOLDS:
        frames=role_frames(samples,pd.read_csv(output/f'roles_fold_{fold}.csv'));outer_x=close.load_pixels(frames['outer'],device)
        for arm in ARMS:
            selected=next(c['selected']['loss'] for c in frozen['choices'] if c['fold']==fold and c['arm']==arm)
            for loss in LOSSES:
                paths=[run_path(output,fold,arm,loss,s)/'inference.pt' for s in SEEDS]
                patients=predict_ensemble(paths,frames['outer'],outer_x,device)
                policy=next(p for p in policies if p['fold']==fold and p['arm']==arm and p['loss']==loss)
                patients['probability_calibrated']=base.apply_calibration(patients.probability_raw.to_numpy(),policy['calibration'])
                patients['prediction']=(patients.probability_calibrated>=policy['threshold']).astype(int)
                patients['threshold']=policy['threshold'];patients['fold']=fold;patients['arm']=arm;patients['loss']=loss
                patients['selected_loss']=loss==selected
                patients.to_csv(output/f'outer_{fold}_{arm}_{loss}.csv',index=False)
                outcomes.append({'fold':fold,'arm':arm,'loss':loss,'selected':loss==selected,
                    'raw':image.metricas_binarias(patients.label,patients.probability_raw,.5),
                    'policy':image.metricas_binarias(patients.label,patients.probability_calibrated,policy['threshold']),
                    'macro_f1_policy':image.metricas_binarias(patients.label,patients.probability_calibrated,policy['macro_f1_threshold'])})
        del outer_x
    if base.sha256(output/'politicas_congeladas.json')!=policy_hash:raise ValueError('Política alterada durante outer')
    verify(output)
    check_weights()
    result={'completed_at':now(),'runs':runs,'outcomes':outcomes,'selection':frozen,'policies':policies,'versions_unchanged':True,
        'test_images_loaded':0,'private_images_loaded':0,'evaluation_scope':'Desarrollo adaptativo en train, no test independiente nuevo'}
    write(output/'resultado.json',result);state.update(status='complete',updated_at=now());write(output/'estado.json',state)
    print('PILOTO COMPLETO',json.dumps({'completed':16,'outcomes':outcomes},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['preparar','entrenar']);parser.add_argument('--salida',type=Path,default=OUTPUT);args=parser.parse_args()
    try:
        (prepare if args.action=='preparar' else execute)(args.salida)
    except Exception:
        if args.salida.exists():write(args.salida/'error.json',{'at':now(),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
