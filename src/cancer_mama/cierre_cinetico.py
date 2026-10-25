"""Cierre acotado de la CNN cinética: cuatro roles de pacientes y un solo corte.

No abre test ni validation. Arquitectura idéntica a V007, cuatro entrenamientos
prefijados y una sola comprobación del paquete completo, sin búsqueda posterior.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch import nn

from . import entrenamiento as base
from . import entrenamiento_cinetico as cnn
from . import entrenamiento_imagen as image
from .paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

OUTPUT = RESULTS_DIR/'24_cierre_cinetico_un_corte_20261025'
V007 = PROJECT_ROOT/'modelos/versiones/v007_cnn_cinetica_multiescala_20261021'
SPLIT_SEED, SLICE_SEED = 20261025, 734921
SEEDS, LOSSES = (42,2026), ('normal','weighted')


def now(): return datetime.now(ZoneInfo('Europe/Madrid')).isoformat()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def split_patients(samples,seed=SPLIT_SEED):
    patients=samples[['patient_id','pCR','dataset']].drop_duplicates().sort_values('patient_id').reset_index(drop=True)
    if patients.patient_id.duplicated().any() or not samples.split.eq('train').all():
        raise ValueError('Solo train; una etiqueta/cohorte por paciente')
    n=len(patients);remaining=patients.copy();parts=[]
    for offset,(role,size) in enumerate([('check',round(n*.15)),('calibration',round(n*.15)),('selection',round(n*.10))]):
        strata=remaining.dataset.astype(str)+'_'+remaining.pCR.astype(str)
        train_idx,held_idx=next(StratifiedShuffleSplit(n_splits=1,test_size=size,random_state=seed+offset).split(remaining.patient_id,strata))
        part=remaining.iloc[held_idx].copy();part['role']=role;parts.append(part)
        remaining=remaining.iloc[train_idx].reset_index(drop=True)
    remaining['role']='fit';parts.append(remaining)
    result=pd.concat(parts).sort_values('patient_id').reset_index(drop=True)
    if len(result)!=n or result.patient_id.duplicated().any():raise ValueError('Particiones solapadas')
    return result


def one_indices(frame,seed=SLICE_SEED):
    """Un corte pseudouniforme por paciente, sin consultar etiqueta ni predicción."""
    result=[]
    for pid,g in frame.reset_index(drop=True).groupby('patient_id',sort=True):
        ordered=g.sort_values('sample_id')
        pick=int.from_bytes(hashlib.sha256(f'{seed}:{pid}'.encode()).digest()[:8],'big')%len(g)
        result.append(int(ordered.index[pick]))
    return np.array(result,dtype=int)


def patient_weights(frame):
    return 1/frame.groupby('patient_id').patient_id.transform('size').to_numpy(dtype=float)


def weighted_auc(frame,p):
    return float(roc_auc_score(frame.pCR,p,sample_weight=patient_weights(frame)))


def source_hashes():
    modules=[base,cnn,image,__import__(__name__,fromlist=['_'])]
    paths=[Path(m.__file__) for m in modules]+[PROJECT_ROOT/'src/cancer_mama/datos.py',
             DATA_DIR/'metadata/samples.csv',DATA_DIR/'metadata/patients.csv']
    return {str(p.relative_to(PROJECT_ROOT)):base.sha256(p) for p in paths}


def previous_versions():
    root=PROJECT_ROOT/'modelos/versiones'
    return {str(p.relative_to(PROJECT_ROOT)):base.sha256(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def prepare(output):
    output=Path(output)
    if output.exists():raise ValueError('La salida existe: no sobrescribir protocolos')
    samples=base.cargar_train(DATA_DIR,include_clinical=False)
    parts=split_patients(samples)
    manifest=json.loads((V007/'manifiesto.json').read_text())
    cfg=cnn.configuracion()
    cfg['training'].update(lr=3e-4,weight_decay=.01,cpu_threads=4,num_workers=0)
    if any(entry['model_config']!=cfg['model'] for entry in manifest['models']):raise ValueError('Arquitectura diferente de V007')
    if image.numero_parametros(cnn.CNNMultiescalaCinetica(cfg))!=593521:raise ValueError('Parámetros distintos')
    output.mkdir(parents=True)
    parts.to_csv(output/'particiones.csv',index=False)
    protocol={'created_at':now(),'source_sha256':source_hashes(),'previous_versions':previous_versions(),
        'split_sha256':base.sha256(output/'particiones.csv'),'split_seed':SPLIT_SEED,'single_slice_seed':SLICE_SEED,
        'config':cfg,'runs':4,'losses':list(LOSSES),'seeds':list(SEEDS),'parameters':593521,
        'recipe_reason':'LR/WD modal entre las cinco elecciones internas originales de V007; no selección por test',
        'roles':parts.groupby('role').size().to_dict(),
        'checkpoint_and_loss_selection':'AUC de corte con peso total uno por paciente; ensemble de dos semillas para elegir pérdida; empate BCE normal',
        'training_unit':'BCEWithLogitsLoss por corte con peso 1/n_cortes; pos_weight por pacientes de fit',
        'calibration':'Platt C=1, pendiente no negativa; solo calibration, mismo ensemble de dos semillas y entrada de un corte',
        'threshold':'Máxima balanced accuracy en calibration, ponderada por paciente; sin objetivo clínico',
        'check_primary':'Un corte prefijado por paciente; pesos/política congelados antes de leer sus imágenes',
        'check_secondary':'Todos los cortes ponderados por paciente; media de cortes solo como diagnóstico separado',
        'legacy_audit':'Solo fit+selection, predicciones de la pareja V007 cuyo fold excluyó a esa paciente; comparación un corte/media',
        'limits':['Train tiene antecedentes de desarrollo adaptativo: check no es confirmación externa nueva',
                  'No abrir test ni validation, no volver a ajustar después de check, no publicar ni desplegar automáticamente',
                  'Se archiva la receta elegida en selection aunque check sea insuficiente'],
        'test_images_loaded':0,'private_validation_images_loaded':0}
    write(output/'protocolo.json',protocol)
    (output/'protocolo.json.sha256').write_text(base.sha256(output/'protocolo.json')+'\n')
    write(output/'estado.json',{'status':'prepared','runs_completed':0,'planned_runs':4,'updated_at':now()})
    print(json.dumps({'prepared':True,'patients':protocol['roles'],'runs':4,'parameters':593521}),flush=True)


def verify(output):
    output=Path(output);path=output/'protocolo.json'
    if base.sha256(path)!=path.with_suffix('.json.sha256').read_text().strip():raise ValueError('Protocolo modificado')
    p=json.loads(path.read_text())
    for name,digest in {**p['source_sha256'],**p['previous_versions']}.items():
        if base.sha256(PROJECT_ROOT/name)!=digest:raise ValueError(f'Fuente o versión modificada: {name}')
    if base.sha256(output/'particiones.csv')!=p['split_sha256']:raise ValueError('Particiones modificadas')
    return p


def load_pixels(frame,device):
    if not frame.split.eq('train').all():raise ValueError('Solo se pueden cargar imágenes de train')
    def read_row(row):
        arrays=[]
        for phase in ('pre','early','late'):
            p=DATA_DIR/getattr(row,'path_'+phase)
            if not p.resolve().is_relative_to((DATA_DIR/'dataset/train').resolve()):raise ValueError('Ruta fuera de train')
            with Image.open(p) as im:
                if im.mode!='L' or im.size!=(256,256) or im.format!='PNG':raise ValueError('PNG incompatible')
                arrays.append(np.asarray(im).copy())
        return np.stack(arrays)
    with ThreadPoolExecutor(max_workers=8) as pool:
        array=np.stack(list(pool.map(read_row,frame.itertuples())))
    return torch.from_numpy(array).to(device)


@torch.inference_mode()
def predict(model,pixels,batch_size=64):
    model.eval()
    return np.concatenate([model(x.float()/255).sigmoid().flatten().cpu().numpy() for x in pixels.split(batch_size)])


def save_inference(path,model,cfg):
    torch.save({'schema_version':1,'model_config':deepcopy(cfg['model']),
                'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}},path)


def train_run(frame,selection,fit_x,sel_x,cfg,loss_name,seed,run_dir,device):
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    cfg=deepcopy(cfg);cfg.update(loss=loss_name,seed=seed,selection_metric='patient_balanced_single_slice_auc')
    cfg['counts']={'fit_patients':int(frame.patient_id.nunique()),'fit_slices':len(frame),
                   'selection_patients':int(selection.patient_id.nunique()),'selection_slices':len(selection)}
    patients=frame.drop_duplicates('patient_id')
    pos_weight=float((patients.pCR==0).sum()/(patients.pCR==1).sum()) if loss_name=='weighted' else 1.
    cfg['pos_weight']=pos_weight;write(run_dir/'config.json',cfg)
    if (run_dir/'summary.json').exists():
        result=json.loads((run_dir/'summary.json').read_text())
        if result['config_sha256']!=base.sha256(run_dir/'config.json'):raise ValueError('Run distinto')
        if result['weights_sha256']!=base.sha256(run_dir/'inference.pt'):raise ValueError('Peso modificado')
        return result
    t=cfg['training'];base.seed_everything(seed,True)
    generator=torch.Generator().manual_seed(seed);aug_gen=torch.Generator().manual_seed(seed+1)
    model=cnn.CNNMultiescalaCinetica(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=t['epochs'],eta_min=t['min_lr'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and t['amp'],init_scale=1024.)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=device),reduction='none')
    y=torch.tensor(frame.pCR.to_numpy(),device=device,dtype=torch.float32)
    w=torch.tensor(patient_weights(frame),device=device,dtype=torch.float32)
    best,best_epoch,bad,history,start=-1.,-1,0,[],0
    if (run_dir/'last.pt').exists():
        ck=base.load_checkpoint(run_dir/'last.pt',model,optimizer,scheduler,scaler,generator,aug_gen)
        if base.config_hash(ck['config'])!=base.config_hash(cfg):raise ValueError('Checkpoint incompatible')
        best,best_epoch,bad,history,start=ck['best_auc'],ck['best_epoch'],ck['bad_epochs'],ck['history'],ck['epoch']+1
    started=time.monotonic()
    for epoch in range(start,t['epochs']):
        if epoch>=t['min_epochs'] and bad>=t['patience']:break
        tic=time.monotonic();model.train();order=torch.randperm(len(frame),generator=generator)
        total,den=0.,0.
        for idx in order.split(t['batch_size']):
            idx=idx.to(device);x=base.aumentar_geometria(fit_x[idx].float()/255,aug_gen,**cfg['augmentation'])
            optimizer.zero_grad(set_to_none=True)
            with base.amp_context(device,scaler.is_enabled()):
                losses=criterion(model(x).flatten(),y[idx]);loss=(losses*w[idx]).sum()/w[idx].sum()
            if not torch.isfinite(loss):raise FloatingPointError('BCE no finita')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),t['clip_grad_norm'],error_if_nonfinite=not scaler.is_enabled())
            scaler.step(optimizer);scaler.update()
            total+=float(loss.detach())*float(w[idx].sum());den+=float(w[idx].sum())
        fit_p,sel_p=predict(model,fit_x),predict(model,sel_x)
        fit_auc,sel_auc=weighted_auc(frame,fit_p),weighted_auc(selection,sel_p)
        fit_bce=float(nn.functional.binary_cross_entropy(torch.tensor(fit_p),torch.tensor(frame.pCR.to_numpy(),dtype=torch.float32),weight=torch.tensor(patient_weights(frame),dtype=torch.float32),reduction='sum')/frame.patient_id.nunique())
        sel_bce=float(nn.functional.binary_cross_entropy(torch.tensor(sel_p),torch.tensor(selection.pCR.to_numpy(),dtype=torch.float32),weight=torch.tensor(patient_weights(selection),dtype=torch.float32),reduction='sum')/selection.patient_id.nunique())
        substantial=sel_auc>best+t['min_delta'];bad=0 if substantial else bad+1
        if sel_auc>best+1e-12:
            best,best_epoch=sel_auc,epoch;save_inference(run_dir/'inference.pt',model,cfg)
        scheduler.step()
        row={'epoch':epoch+1,'train_loss':total/den,'fit_auc':fit_auc,'selection_auc':sel_auc,
             'fit_bce':fit_bce,'selection_bce':sel_bce,'gap':fit_auc-sel_auc,'lr':optimizer.param_groups[0]['lr'],'seconds':time.monotonic()-tic}
        history.append(row);pd.DataFrame(history).to_csv(run_dir/'history.csv',index=False)
        base.save_checkpoint(run_dir/'last.pt',model,optimizer,scheduler,scaler,epoch,best,bad,cfg,generator,aug_gen,history,best_epoch)
        print(f"{loss_name} seed={seed} época={epoch+1}: AUC selección={sel_auc:.4f}, brecha={fit_auc-sel_auc:+.4f}, {row['seconds']:.1f}s",flush=True)
    artifact=torch.load(run_dir/'inference.pt',map_location=device,weights_only=True)
    model.load_state_dict(artifact['state_dict']);selected=predict(model,sel_x)
    out=selection[['sample_id','patient_id','pCR','dataset']].copy();out['probability_raw']=selected
    out.to_csv(run_dir/'selection.csv',index=False)
    steps={int(state['step'].item()) for state in optimizer.state.values()}
    expected=len(history)*int(np.ceil(len(frame)/t['batch_size']))
    if steps!={expected}:raise RuntimeError(f'Pasos AdamW omitidos: {steps}, esperado {expected}')
    if any(not torch.isfinite(t).all() for t in artifact['state_dict'].values()):raise FloatingPointError('Peso no finito')
    result={'status':'complete','epochs':len(history),'selected_epoch':best_epoch+1,'selection_auc':best,
            'optimizer_steps':expected,'skipped_steps':0,'seconds':time.monotonic()-started,
            'weights_sha256':base.sha256(run_dir/'inference.pt'),'config_sha256':base.sha256(run_dir/'config.json')}
    write(run_dir/'summary.json',result)
    del model,optimizer,scheduler,scaler
    return result


def legacy_audit(output,frames,pixels,device):
    path=output/'auditoria_v007.json'
    if path.exists():return
    manifest=json.loads((V007/'manifiesto.json').read_text());outputs=[]
    for role in ('fit','selection'):
        frame=frames[role];p=np.zeros(len(frame))
        for fold in range(5):
            indices=np.flatnonzero(frame.fold.eq(fold))
            pair=[]
            for entry in manifest['models']:
                if entry['fold']!=fold:continue
                weight=V007/entry['path']
                if base.sha256(weight)!=entry['sha256']:raise ValueError('Peso histórico modificado')
                artifact=torch.load(weight,map_location='cpu',weights_only=True)
                model=cnn.CNNMultiescalaCinetica(artifact['model_config']).to(device)
                model.load_state_dict(artifact['state_dict']);pair.append(predict(model,pixels[role][torch.tensor(indices,device=device)]));del model
            if len(pair)!=2:raise ValueError('Pareja exterior incompleta')
            p[indices]=np.mean(pair,axis=0)
        out=frame[['sample_id','patient_id','pCR','dataset','fold']].copy();out['probability_raw']=p;outputs.append(out)
    frame=pd.concat(outputs).sort_values(['patient_id','sample_id']).reset_index(drop=True)
    one=frame.iloc[one_indices(frame)];means=frame.groupby('patient_id').agg(pCR=('pCR','first'),probability_raw=('probability_raw','mean'))
    result={'patients':int(frame.patient_id.nunique()),'scope':'fit+selection solamente; sin imágenes de calibration/check/test/validation',
        'one_fixed_slice_auc':float(roc_auc_score(one.pCR,one.probability_raw)),
        'patient_mean_all_slices_auc':float(roc_auc_score(means.pCR,means.probability_raw)),
        'patient_balanced_single_slice_distribution_auc':weighted_auc(frame,frame.probability_raw),
        'unbiased_components':True,'full_ten_model_ensemble_evaluated':False,
        'warning':'Desarrollo histórico: las parejas son las que excluyeron a la paciente; no mide el ensemble final de diez redes'}
    frame.to_csv(output/'auditoria_v007_cortes.csv',index=False);write(path,result)
    print('AUDITORÍA',json.dumps(result,ensure_ascii=False),flush=True)


def aligned_ensemble(paths,frame,pixels,device):
    probabilities=[]
    for path in paths:
        artifact=torch.load(path,map_location='cpu',weights_only=True)
        model=cnn.CNNMultiescalaCinetica(artifact['model_config']).to(device)
        model.load_state_dict(artifact['state_dict']);probabilities.append(predict(model,pixels));del model
    p=np.mean(probabilities,axis=0)
    if not np.isfinite(p).all():raise FloatingPointError('Probabilidades no finitas')
    return p


def fit_policy(frame,p):
    w=patient_weights(frame);x=base.logit(np.asarray(p)).reshape(-1,1)
    fit=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,random_state=0).fit(x,frame.pCR,sample_weight=w)
    slope,intercept=float(fit.coef_[0,0]),float(fit.intercept_[0]);boundary=slope<0
    if boundary:slope,intercept=0.,float(base.logit(np.array([np.average(frame.pCR,weights=w)]))[0])
    calibration={'method':'regularized_platt','slope':slope,'intercept':intercept,'C':1.,'clip_epsilon':base.EPS,
        'scope':'single_slice','input':'logit_single_slice_ensemble_probability','fit_data':'calibration_patients_only',
        'nonnegative_slope_constraint':True,'boundary_solution':boundary,'sample_weight':'1/n_slices_per_patient'}
    calibrated=base.apply_calibration(np.asarray(p),calibration)
    fpr,tpr,thresholds=roc_curve(frame.pCR,calibrated,sample_weight=w,drop_intermediate=False)
    thresholds=np.minimum(thresholds,1.);scores=(tpr+1-fpr)/2
    eligible=np.flatnonzero(scores>=scores.max()-1e-12)
    best=max(eligible,key=lambda i:(min(tpr[i],1-fpr[i]),thresholds[i]))
    return {'calibration':calibration,'threshold':float(thresholds[best]),
            'threshold_strategy':'max_patient_balanced_accuracy_calibration_only',
            'fitted_at':now(),'calibration_patients':int(frame.patient_id.nunique())}


def check_metrics(frame,p,policy):
    calibrated=base.apply_calibration(p,policy['calibration']);idx=one_indices(frame)
    one=frame.iloc[idx].reset_index(drop=True)
    primary=image.metricas_binarias(one.pCR,calibrated[idx],policy['threshold'])
    raw=image.metricas_binarias(one.pCR,p[idx],.5)
    rng=np.random.default_rng(SPLIT_SEED);strata=[]
    for _,g in one.groupby(['dataset','pCR']):strata.append(g.index.to_numpy())
    boot=[]
    for _ in range(2000):
        s=np.concatenate([rng.choice(g,len(g),replace=True) for g in strata])
        boot.append(roc_auc_score(one.pCR.to_numpy()[s],calibrated[idx][s]))
    out=frame[['sample_id','patient_id','pCR','dataset']].copy()
    out['probability_raw']=p;out['probability_calibrated']=calibrated
    out['selected_one_slice']=False;out.loc[idx,'selected_one_slice']=True
    means=out.groupby('patient_id').agg(pCR=('pCR','first'),probability_raw=('probability_raw','mean'))
    return {'primary_single_slice':primary,'raw_at_0_5':raw,'auc_ci95':[float(v) for v in np.quantile(boot,[.025,.975])],
            'secondary_weighted_slice_auc':weighted_auc(frame,calibrated),
            'diagnostic_all_slices_mean_raw_auc':float(roc_auc_score(means.pCR,means.probability_raw)),
            'test_opened':False,'private_validation_opened':False,
            'interpretation':'Comprobación interna del procedimiento congelado; train tiene historia adaptativa, no confirmación externa'},out


def execute(output):
    output=Path(output);protocol=verify(output)
    if (output/'resultado.json').exists():raise ValueError('Ensayo terminado: no repetir ni ajustar contra check')
    if not torch.cuda.is_available():raise RuntimeError('CUDA requerida para esta tanda')
    device=torch.device('cuda');torch.set_num_threads(4)
    samples=base.cargar_train(DATA_DIR,include_clinical=False).sort_values(['patient_id','sample_id']).reset_index(drop=True)
    parts=pd.read_csv(output/'particiones.csv')
    frames={role:samples[samples.patient_id.isin(parts.loc[parts.role.eq(role),'patient_id'])].reset_index(drop=True)
            for role in ('fit','selection','calibration','check')}
    state={'status':'training','pid':os.getpid(),'started_at':now(),'runs_completed':0,'planned_runs':4}
    write(output/'estado.json',state);write(output/'entorno.json',base.environment(device))
    pixels={role:load_pixels(frames[role],device) for role in ('fit','selection')}
    legacy_audit(output,frames,pixels,device)
    runs=[]
    for loss in LOSSES:
        for seed in SEEDS:
            state.update(loss=loss,seed=seed,updated_at=now());write(output/'estado.json',state)
            path=output/'runs'/loss/f'seed_{seed}'
            summary=train_run(frames['fit'],frames['selection'],pixels['fit'],pixels['selection'],protocol['config'],loss,seed,path,device)
            runs.append({'loss':loss,'seed':seed,'path':str(path.relative_to(output)),**summary})
            state['runs_completed']+=1;write(output/'estado.json',state)
    candidates=[]
    for loss in LOSSES:
        predictions=[pd.read_csv(output/'runs'/loss/f'seed_{seed}'/'selection.csv') for seed in SEEDS]
        if not predictions[0].sample_id.equals(predictions[1].sample_id):raise ValueError('Semillas no alineadas')
        p=np.mean([f.probability_raw.to_numpy() for f in predictions],axis=0)
        candidates.append({'loss':loss,'selection_auc':weighted_auc(frames['selection'],p)})
    selected=max(candidates,key=lambda c:c['selection_auc'])
    paths=[output/'runs'/selected['loss']/f'seed_{seed}'/'inference.pt' for seed in SEEDS]
    choice={'selected':selected,'candidates':candidates,'frozen_at':now(),
            'weights':{str(p.relative_to(output)):base.sha256(p) for p in paths}}
    write(output/'seleccion_congelada.json',choice)
    del pixels
    state.update(status='calibrating',updated_at=now());write(output/'estado.json',state)
    cal_x=load_pixels(frames['calibration'],device)
    cal_p=aligned_ensemble(paths,frames['calibration'],cal_x,device);del cal_x
    policy=fit_policy(frames['calibration'],cal_p)
    policy.update(selected=selected,weights=choice['weights'],protocol_sha256=base.sha256(output/'protocolo.json'))
    write(output/'politica_congelada.json',policy)
    (output/'politica_congelada.json.sha256').write_text(base.sha256(output/'politica_congelada.json')+'\n')
    cal_frame=frames['calibration'][['sample_id','patient_id','pCR','dataset']].copy();cal_frame['probability_raw']=cal_p
    cal_frame['probability_calibrated']=base.apply_calibration(cal_p,policy['calibration'])
    cal_frame.to_csv(output/'calibracion_cortes.csv',index=False)
    state.update(status='checking_frozen_policy',updated_at=now());write(output/'estado.json',state)
    verify(output)
    check_x=load_pixels(frames['check'],device)
    check_p=aligned_ensemble(paths,frames['check'],check_x,device);del check_x
    result,out=check_metrics(frames['check'],check_p,policy)
    out.to_csv(output/'comprobacion_cortes.csv',index=False)
    result.update(completed_at=now(),selected=selected,runs=runs,policy=policy)
    write(output/'resultado.json',result)
    candidate=output/'candidato';candidate.mkdir(exist_ok=True)
    models=[]
    for seed,path in zip(SEEDS,paths):
        import shutil
        dest=candidate/f'seed_{seed}.pt';shutil.copy2(path,dest)
        models.append({'seed':seed,'fold':0,'path':dest.name,'sha256':base.sha256(dest),'model_config':protocol['config']['model']})
    manifest={'schema_version':2,'evaluation_protocol':'train_only_four_way_single_slice',
        'model_class':'CNNMultiescalaCinetica','status':'candidato_cierre_interno_sin_confirmacion_privada',
        'inference_unit':'single_slice','models':models,'ensemble':'mean probabilities of two fixed seeds',
        'slice_aggregation':'single','calibration':policy['calibration'],'threshold':policy['threshold'],
        'threshold_policy':policy['threshold_strategy'],'test_evaluated_once':False,
        'preprocessing':json.loads((V007/'manifiesto.json').read_text())['preprocessing'],
        'training':{**protocol['config']['training'],'loss':selected['loss'],'patient_balanced_loss':True},
        'selection':{'loss':selected['loss'],'method':'selection_patients_only','selection_auc':selected['selection_auc'],
                     'metrics_internal_check':result['primary_single_slice'],'auc_ci95_internal_check':result['auc_ci95']},
        'provenance_files':[]}
    for name in ('protocolo.json','particiones.csv','seleccion_congelada.json','politica_congelada.json','resultado.json'):
        import shutil
        dest=candidate/name;shutil.copy2(output/name,dest)
        manifest['provenance_files'].append({'path':name,'sha256':base.sha256(dest)})
    write(candidate/'manifiesto.json',manifest)
    (candidate/'manifiesto.json.sha256').write_text(base.sha256(candidate/'manifiesto.json')+'\n')
    verify(output)
    state.update(status='complete',updated_at=now(),selected_loss=selected['loss'],check_auc=result['primary_single_slice']['roc_auc'])
    write(output/'estado.json',state)
    print('RESULTADO',json.dumps(result,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preparar','ejecutar'])
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.action=='preparar':prepare(args.output)
    else:
        try:execute(args.output)
        except Exception:
            write(args.output/'error.json',{'at':now(),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
