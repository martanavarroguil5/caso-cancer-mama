"""Refit prefijado a igual número de pasos, sin selección con outer."""
from datetime import datetime,timezone
from copy import deepcopy
from pathlib import Path
import json,math,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from cancer_mama import imagen_mil as m
from cancer_mama import datos
from cancer_mama import entrenamiento as base
import validar_imagen_anidada as validation


def refit(cache,source,fold,seed,output):
    patients,images,masks,features=cache;device=features.device
    ckpt=torch.load(source,map_location=device,weights_only=True);cfg=ckpt['config']
    original_fit,_,_=m.split_indices(patients,fold)
    fit=np.flatnonzero(patients.fold.ne(fold));outer=np.flatnonzero(patients.fold.eq(fold))
    if set(fit)&set(outer):raise ValueError('Fuga de partición')
    source_steps=math.ceil(len(original_fit)/cfg['batch_size'])
    fixed_steps=int(ckpt['epoch'])*source_steps
    directory=output/'refit'/f'fold_{fold}'/f'seed_{seed}'
    if (directory/'summary.json').exists():return json.loads((directory/'summary.json').read_text())
    directory.mkdir(parents=True,exist_ok=True)
    datos.fijar_semilla(seed+1000*fold);torch.set_num_threads(6)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=m.PatientNet(cfg,features.shape[1]).to(device)
    model.feature_mean.copy_(features[fit].mean(0))
    model.feature_std.copy_(features[fit].std(0).clamp_min(cfg.get('feature_std_min',.02)))
    averaged=deepcopy(model).eval() if cfg.get('ema_decay') else None
    if averaged is not None:
        for parameter in averaged.parameters():parameter.requires_grad_(False)
    y=torch.tensor(patients.pCR.to_numpy(),device=device,dtype=torch.float32)
    pos_weight=(1-y[fit].mean())/y[fit].mean() if cfg['weighted'] else torch.tensor(1.,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    total_schedule_steps=cfg['epochs']*source_steps
    step=0;epoch=0;history=[];start=time.monotonic()
    while step<fixed_steps:
        epoch+=1;model.train();losses=[]
        for ix in torch.as_tensor(np.random.permutation(fit),device=device).split(cfg['batch_size']):
            target=y[ix];mask=masks[ix];feat=features[ix]
            scores=torch.rand(mask.shape,device=device).masked_fill(~mask,2.)
            selected=scores.argsort(1)[:,:cfg['train_slices']]
            if cfg.get('slice_pool')=='sequence':selected=selected.sort(1).values
            x=m.augment(images[ix[:,None],selected].float()/255);mask=mask.gather(1,selected)
            target=target*(1-cfg['label_smoothing'])+cfg['label_smoothing']*.5
            if cfg['mixup']>0:
                lam=float(np.random.beta(cfg['mixup'],cfg['mixup']));other=torch.randperm(len(ix),device=device)
                feat=lam*feat+(1-lam)*feat[other];x=lam*x+(1-lam)*x[other]
                mask=mask|mask[other];target=lam*target+(1-lam)*target[other]
            # Reproduce la LR de la época interna correspondiente al mismo paso.
            source_epoch=step//source_steps
            lr=cfg['lr']*.05+(cfg['lr']-cfg['lr']*.05)*.5*(1+math.cos(math.pi*source_epoch/cfg['epochs']))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,enabled=device.type=='cuda',dtype=torch.float16):
                logit=model(x,mask,feat)
                loss=F.binary_cross_entropy_with_logits(logit,target,pos_weight=pos_weight)
                if cfg.get('rank_weight',0)>0:
                    positive=logit[y[ix]>.5];negative=logit[y[ix]<.5]
                    if positive.numel() and negative.numel():
                        loss=loss+cfg['rank_weight']*F.softplus(-(positive[:,None]-negative[None,:])).mean()
                if cfg.get('feature_l2',0)>0 and model.linear_features:
                    loss=loss+cfg['feature_l2']*model.feature_net.weight.square().sum()
            if not torch.isfinite(loss):raise RuntimeError('Pérdida no finita')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),2.)
            scaler.step(optimizer);scaler.update()
            if averaged is not None:
                with torch.no_grad():
                    for avg,current in zip(averaged.parameters(),model.parameters()):avg.lerp_(current,1-cfg['ema_decay'])
                    for avg,current in zip(averaged.buffers(),model.buffers()):
                        if avg.is_floating_point():avg.lerp_(current,1-cfg['ema_decay'])
                        else:avg.copy_(current)
            losses.append(float(loss.detach()));step+=1
            if step>=fixed_steps:break
        history.append(dict(epoch=epoch,steps=step,train_augmented_loss=float(np.mean(losses))))
        pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
        print(f'refit f{fold} s{seed}: {step}/{fixed_steps} pasos, pérdida aumentada={np.mean(losses):.4f}',flush=True)
    evaluated=averaged if averaged is not None else model
    torch.save(dict(state_dict=evaluated.state_dict(),config=cfg,n_features=features.shape[1],
                    epoch=epoch,seed=seed,fold=fold,steps=step,preprocessing=m.CACHE_VERSION),directory/'best.pt')
    p=m.predict(evaluated,images,masks,features,fit)
    result=dict(status='complete',source_checkpoint=str(source),source_sha256=base.sha256(source),
        config=cfg,fold=fold,seed=seed,source_best_epoch=ckpt['epoch'],fixed_steps=fixed_steps,steps=step,
        fit_patients=len(fit),fit_metrics=m.metrics(patients.iloc[fit].pCR,p),seconds=time.monotonic()-start,
        inference_file=str(directory/'best.pt'),inference_sha256=base.sha256(directory/'best.pt'),test_images_loaded=0,
        calibration='unchanged calibration from internally selected source ensemble; no refit labels used for calibration')
    m.write_json(directory/'summary.json',result)
    return result


def main():
    output=m.OUTPUT;plan_path=output/'refit_plan_locked.json'
    if not plan_path.exists():
        if list(output.glob('outer_*.csv')):raise RuntimeError('El refit debe prefijarse antes de outer')
        m.write_json(plan_path,dict(created_utc=datetime.now(timezone.utc).isoformat(),
            strategy='all non-outer patients; exactly source best epoch times original fit batches optimizer steps',
            lr='same source-epoch LR at each optimizer step',calibration='unchanged original inner calibration',
            selector='same per-fold family and seeds already chosen by the inner search',test_images_loaded=0,
            source_sha256=base.sha256(Path(__file__))))
    while not (output/'final_metrics.json').exists():time.sleep(2)
    choices=json.loads((output/'final_choices_locked.json').read_text())['choices'];data={};refitted=[]
    for choice in choices:
        new=deepcopy(choice);entries=[]
        for entry in choice['checkpoints']:
            cfg=torch.load(entry['file'],map_location='cpu',weights_only=True)['config']
            key=cfg.get('feature_file') or ('features_roi_v2.npy' if cfg.get('roi_features') else 'features.npy')
            if key not in data:data[key]=m.load_cache(feature_file=key)
            result=refit(data[key],entry['file'],choice['fold'],entry['seed'],output)
            entries.append(dict(file=result['inference_file'],sha256=result['inference_sha256'],seed=entry['seed']))
        new['checkpoints']=entries;refitted.append(new)
    m.write_json(output/'refit_choices_locked.json',dict(choices=refitted,plan_sha256=base.sha256(plan_path),
                   locked_utc=datetime.now(timezone.utc).isoformat()))
    frames=[]
    for choice in refitted:
        predictions=[]
        for entry in choice['checkpoints']:
            ckpt=torch.load(entry['file'],map_location='cuda',weights_only=True);cfg=ckpt['config']
            key=cfg.get('feature_file') or ('features_roi_v2.npy' if cfg.get('roi_features') else 'features.npy')
            patients,images,masks,features=data[key];outer=np.flatnonzero(patients.fold.eq(choice['fold']))
            model=m.PatientNet(cfg,ckpt['n_features']).cuda();model.load_state_dict(ckpt['state_dict'])
            predictions.append(m.predict(model,images,masks,features,outer))
        frame=patients.iloc[outer].copy();frame['probability_raw']=np.mean(predictions,axis=0)
        frame['probability']=validation.calibrate(frame.probability_raw,choice['calibration'])
        frame['selected_family']=choice['family'];frames.append(frame)
        print('REFIT OUTER',choice['fold'],m.metrics(frame.pCR,frame.probability),flush=True)
    oof=pd.concat(frames).sort_values('patient_id').reset_index(drop=True)
    oof.to_csv(output/'refit_oof_patients.csv',index=False)
    result=dict(status='complete',strategy='refit_fixed_optimizer_steps',patients=len(oof),test_images_loaded=0,
        calibrated=m.metrics(oof.pCR,oof.probability),raw=m.metrics(oof.pCR,oof.probability_raw),
        folds=[dict(fold=int(fold),**m.metrics(f.pCR,f.probability)) for fold,f in oof.groupby('fold')],
        cohorts=[dict(cohort=cohort,patients=len(f),**m.metrics(f.pCR,f.probability)) for cohort,f in oof.groupby('dataset')],
        choices_sha256=base.sha256(output/'refit_choices_locked.json'),oof_sha256=base.sha256(output/'refit_oof_patients.csv'))
    result['mean_fold_auc']=float(np.mean([f['auc'] for f in result['folds']]))
    result['sd_fold_auc']=float(np.std([f['auc'] for f in result['folds']],ddof=1))
    m.write_json(output/'refit_metrics.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
