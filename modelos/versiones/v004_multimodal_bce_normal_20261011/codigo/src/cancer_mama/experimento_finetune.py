"""Ajuste parcial de ResNet18 con L2-SP y corrección visual de clínica fija."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn

from . import entrenamiento as base
from . import experimento_loss as previous
from . import experimento_transfer as transfer
from . import experimento_local as local
from . import informe_loss as diagnosis
from .paths import PROJECT_ROOT, RESULTS_DIR

DEFAULT_OUTPUT = RESULTS_DIR / '04_entrenamiento/finetune_multimodal_20261010'


def settings(smoke=False):
    return dict(epochs=1 if smoke else 24, min_epochs=1 if smoke else 8,
        patience=6, min_delta=.001, patients_per_batch=16, slices_per_patient=4,
        tail_lr=1e-5, head_lr=.001, head_weight_decay=.01, l2sp=.005,
        image_logit_limit=.5, dropout=.5, seed_list=[42,2026], smoke=smoke)


def clinical_design(patients, prep):
    x=base.clinical_matrix(patients); missing=np.isnan(x)
    z=(np.where(missing,prep['medians'],x)-prep['means'])/prep['stds']
    return np.column_stack((z,missing.astype(float)))


def clinical_state(fit):
    patients=fit.drop_duplicates('patient_id').sort_values('patient_id')
    prep=base.fit_clinical_preprocessing(patients)
    model=LogisticRegression(C=1.,max_iter=4000,random_state=0).fit(clinical_design(patients,prep),patients.pCR)
    return dict(preprocessing=prep,coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]))


def clinical_logits(patients,state):
    return (clinical_design(patients,state['preprocessing'])@np.asarray(state['coef'])+state['intercept']).astype(np.float32)


class ResidualModel(nn.Module):
    def __init__(self,backbone_state,config):
        super().__init__()
        from torchvision.models import resnet18
        network=resnet18(weights=None);network.load_state_dict(backbone_state)
        self.tail=network.layer4
        self.head=nn.Sequential(nn.Dropout(config['dropout']),nn.Linear(512,32),nn.ReLU(),
                                nn.Dropout(config['dropout']),nn.Linear(32,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.limit=config['image_logit_limit']
        self.train()

    def train(self,mode=True):
        super().train(mode)
        for layer in self.tail.modules():
            if isinstance(layer,nn.BatchNorm2d):layer.eval()
        return self

    def embedding(self,x):
        return self.tail(x).mean(dim=(-2,-1))

    def from_embedding(self,z,clinical):
        return clinical+self.limit*torch.tanh(self.head(z).squeeze(1))

    def forward(self,x,clinical):
        batch,slices=x.shape[:2]
        embedding=self.embedding(x.flatten(0,1)).reshape(batch,slices,512).mean(1)
        return self.from_embedding(embedding,clinical)


def l2sp(model,anchor):
    return sum((p-anchor[name]).square().sum() for name,p in model.tail.named_parameters())


@torch.inference_mode()
def prefix_cache(samples,root,folder,device,progress):
    from torchvision.models import ResNet18_Weights,resnet18
    folder.mkdir(parents=True,exist_ok=True)
    weights_path=folder/'resnet18_imagenet.pth';data_path=folder/'prefix.npy';meta_path=folder/'metadata.json'
    fingerprint={'data':previous.signature(samples),'code':base.sha256(__file__),
                 'representation':'ResNet18_ImageNet1K_V1_layer3_FP32_original_hflip'}
    if meta_path.exists() and data_path.exists():
        meta=json.loads(meta_path.read_text())
        if any(meta[k]!=v for k,v in fingerprint.items()) or base.sha256(weights_path)!=meta['weights_sha256']:
            raise ValueError('Caché incompatible o pesos alterados')
        if base.sha256(data_path)!=meta['data_sha256']:raise ValueError('Activaciones alteradas')
        return np.load(data_path,mmap_mode='r'),weights_path
    weights=ResNet18_Weights.IMAGENET1K_V1
    network=resnet18(weights=weights).eval().requires_grad_(False)
    previous.save_torch(weights_path,network.state_dict())
    prefix=nn.Sequential(network.conv1,network.bn1,network.relu,network.maxpool,
                         network.layer1,network.layer2,network.layer3).eval().to(device)
    tmp=data_path.with_suffix('.tmp.npy')
    memory=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(samples),2,256,14,14))
    cache=base.CacheTrain(samples,root,previous.DEFAULT_OUTPUT/'.cache',workers=4)
    loader=base.make_loader(samples,root,64,4,torch.Generator().manual_seed(transfer.SEED),cache=cache)
    offset=0
    for step,batch in enumerate(loader):
        x,_=base.batch_device(batch,device)
        for view in (0,1):
            memory[offset:offset+len(x),view]=prefix(weights.transforms()(x if view==0 else x.flip(-1))).cpu().numpy()
        offset+=len(x)
        if step%30==0 or step+1==len(loader):
            print(f'Prefijo ResNet18 {step+1}/{len(loader)}',flush=True)
            progress({'stage':'cache','batch':step+1,'batches':len(loader)})
    memory.flush();del memory;tmp.replace(data_path)
    base.json_write(meta_path,dict(fingerprint,weights_sha256=base.sha256(weights_path),data_sha256=base.sha256(data_path)))
    return np.load(data_path,mmap_mode='r'),weights_path


def patient_slice_indices(samples,patients):
    groups=samples.groupby('patient_id',sort=False).indices
    return [np.asarray(groups[p],dtype=int) for p in patients.patient_id]


def sample_bags(cache,groups,patient_indices,slices):
    selected=[];views=[]
    for i in patient_indices:
        selected.append(np.random.choice(groups[i],slices,replace=len(groups[i])<slices))
        views.append(np.random.randint(2))
    selected=np.stack(selected);views=np.asarray(views)[:,None]
    return torch.from_numpy(np.array(cache[selected,views],copy=True))


@torch.inference_mode()
def evaluate(model,cache,groups,patient_indices,logits,labels,device):
    model.eval()
    slice_ids=np.concatenate([groups[i] for i in patient_indices])
    owner=np.repeat(np.arange(len(patient_indices)),[len(groups[i]) for i in patient_indices])
    sums=np.zeros((len(patient_indices),512),dtype=np.float64)
    for start in range(0,len(slice_ids),128):
        ids=slice_ids[start:start+128]
        x=torch.from_numpy(np.array(cache[ids,0],copy=True)).to(device)
        z=model.embedding(x).cpu().numpy()
        np.add.at(sums,owner[start:start+128],z)
    means=(sums/np.asarray([len(groups[i]) for i in patient_indices])[:,None]).astype(np.float32)
    prediction=torch.sigmoid(model.from_embedding(torch.from_numpy(means).to(device),
        torch.from_numpy(logits[patient_indices]).to(device))).cpu().numpy()
    return base.binary_metrics(labels[patient_indices],prediction),prediction


def run_one(samples,cache,backbone_state,output,seed,fold,config,device,progress,until_epoch=None):
    base.seed_everything(seed)
    fit,selection,outer=previous.split_patients(samples,fold)
    if config['smoke']:
        fit,selection,outer=[x[x.patient_id.isin(x.drop_duplicates('patient_id').groupby('pCR').head(3).patient_id)].copy() for x in (fit,selection,outer)]
    patients=samples.drop_duplicates('patient_id').sort_values('patient_id').reset_index(drop=True)
    groups=patient_slice_indices(samples,patients)
    ids=[np.flatnonzero(patients.patient_id.isin(set(x.patient_id))) for x in (fit,selection,outer)]
    fi,si,oi=ids;clinical=clinical_state(fit)
    logits=clinical_logits(patients,clinical);labels=patients.pCR.to_numpy(dtype=np.float32)
    record=dict(config,seed=seed,fold=fold,clinical=clinical,
        partitions={name:previous.signature(x) for name,x in (('fit',fit),('selection',selection),('outer',outer))})
    folder=output/f'seed_{seed}'/f'fold_{fold}';folder.mkdir(parents=True,exist_ok=True)
    config_hash=base.config_hash(record);config_path=folder/'config.json'
    if config_path.exists() and base.config_hash(json.loads(config_path.read_text()))!=config_hash:
        raise ValueError('Configuración de otro entrenamiento')
    base.json_write(config_path,record)
    if (folder/'summary.json').exists():
        summary=json.loads((folder/'summary.json').read_text())
        if summary['config_hash']!=config_hash:raise ValueError('Resumen incompatible')
        for name,sha in summary['files'].items():
            if base.sha256(folder/name)!=sha:raise ValueError('Resultado cambiado')
        return summary
    model=ResidualModel(backbone_state,config).to(device)
    anchor={name:p.detach().clone() for name,p in model.tail.named_parameters()}
    optimizer=torch.optim.AdamW([{'params':model.tail.parameters(),'lr':config['tail_lr'],'weight_decay':0},
        {'params':model.head.parameters(),'lr':config['head_lr'],'weight_decay':config['head_weight_decay']}])
    generator=torch.Generator().manual_seed(seed);aug_generator=torch.Generator().manual_seed(seed+1)
    history=[];start=0;checkpoint=folder/'last.pt'
    def measure(epoch,optimized=None,penalty=None):
        fm,_=evaluate(model,cache,groups,fi,logits,labels,device)
        sm,_=evaluate(model,cache,groups,si,logits,labels,device)
        return {'epoch':epoch,'optimized_bce':optimized,'l2sp_penalty':penalty,
                **{'fit_'+k:v for k,v in fm.items() if k in ('log_loss','roc_auc','brier')},
                **{'selection_'+k:v for k,v in sm.items() if k in ('log_loss','roc_auc','brier')}}
    if checkpoint.exists():
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload['config_hash']!=config_hash:raise ValueError('Checkpoint incompatible')
        model.load_state_dict(payload['state_dict']);optimizer.load_state_dict(payload['optimizer'])
        history=payload['history'];tracker=payload['tracker'];best_state=payload['best_state'];start=payload['epoch']
        base.restore_rng(payload['rng'],generator,aug_generator)
    else:
        row=measure(0);history.append(row)
        tracker=previous.new_tracker('logloss',row['selection_log_loss']);best_state=previous.cpu_state(model)
    for epoch in range(start+1,min(config['epochs'],until_epoch or config['epochs'])+1):
        if tracker['stopped']:break
        model.train();bces=[];penalties=[]
        order=np.random.permutation(fi)
        for begin in range(0,len(order),config['patients_per_batch']):
            idx=order[begin:begin+config['patients_per_batch']]
            x=sample_bags(cache,groups,idx,config['slices_per_patient']).to(device)
            c=torch.from_numpy(logits[idx]).to(device);y=torch.from_numpy(labels[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            bce=nn.functional.binary_cross_entropy_with_logits(model(x,c),y)
            penalty=config['l2sp']*l2sp(model,anchor)
            (bce+penalty).backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            bces.append(float(bce.detach())*len(idx));penalties.append(float(penalty.detach())*len(idx))
        row=measure(epoch,sum(bces)/len(fi),sum(penalties)/len(fi));history.append(row)
        if previous.update_tracker(tracker,row['selection_log_loss'],epoch,config['min_epochs'],config['patience'],config['min_delta']):
            best_state=previous.cpu_state(model)
        previous.save_torch(checkpoint,{'config_hash':config_hash,'epoch':epoch,'state_dict':previous.cpu_state(model),
            'optimizer':optimizer.state_dict(),'history':history,'tracker':tracker,'best_state':best_state,
            'rng':base.capture_rng(generator,aug_generator)})
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        progress({'stage':'training','seed':seed,'fold':fold,'epoch':epoch,'best_epoch':tracker['best_epoch'],
                  'fit_loss':row['fit_log_loss'],'selection_loss':row['selection_log_loss']})
        print(f"seed={seed} fold={fold} epoch={epoch} fit={row['fit_log_loss']:.4f} selection={row['selection_log_loss']:.4f} best={tracker['best_epoch']}",flush=True)
    if until_epoch is not None and until_epoch<config['epochs'] and not tracker['stopped']:return None
    # Exterior solo después de congelar el estado elegido.
    model.load_state_dict(best_state);model.eval()
    previous.save_torch(folder/'best.pt',{'state_dict':best_state,'config':record,'best_epoch':tracker['best_epoch']})
    for name,index in (('inner',si),('oof',oi)):
        _,p=evaluate(model,cache,groups,index,logits,labels,device)
        transfer.prediction_frame(patients,index,p).to_csv(folder/f'{name}.csv',index=False)
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    summary={'config_hash':config_hash,'smoke':config['smoke'],'tracker':tracker,
             'files':{name:base.sha256(folder/name) for name in ('best.pt','history.csv','inner.csv','oof.csv','config.json')}}
    base.json_write(folder/'summary.json',summary)
    return summary


def compare(samples,output):
    rows=[];inners=[];calibrated=[];calibrators=[]
    for fold in range(5):
        pairs={name:[] for name in ('inner','oof')}
        for seed in (42,2026):
            folder=output/f'seed_{seed}'/f'fold_{fold}'
            summary=json.loads((folder/'summary.json').read_text())
            if summary['smoke']:raise ValueError('Prueba excluida')
            for name,sha in summary['files'].items():
                if base.sha256(folder/name)!=sha:raise ValueError('Resultado modificado')
            for name in pairs:pairs[name].append(pd.read_csv(folder/f'{name}.csv'))
        averaged={}
        keys=['patient_id','label','fold','cohort']
        for name,pair in pairs.items():
            if not pair[0][keys].equals(pair[1][keys]):raise ValueError('Semillas desalineadas')
            frame=pair[0].copy();frame['probability']=(pair[0].probability+pair[1].probability)/2;averaged[name]=frame
        rows.append(averaged['oof']);inners.append(averaged['inner'].assign(outer_fold=fold))
        result,cal,threshold=diagnosis.calibrate_inner(averaged['inner'],averaged['oof'])
        calibrated.append(result);calibrators.append({'fold':fold,'calibration':cal,'threshold':threshold})
    raw=pd.concat(rows).sort_values('patient_id').reset_index(drop=True)
    cal=pd.concat(calibrated).sort_values('patient_id').reset_index(drop=True)
    reference=pd.read_csv(transfer.REFERENCE/'oof_calibrated_inner_auc.csv')
    secondary=pd.read_csv(transfer.REFERENCE/'oof_calibrated_inner_logloss.csv')
    keys=['patient_id','label','fold','cohort']
    roster=samples.drop_duplicates('patient_id').sort_values('patient_id').rename(columns={'pCR':'label','dataset':'cohort'}).reset_index(drop=True)
    if not raw[keys].equals(roster[keys]):raise ValueError('OOF incompletas')
    dest=output/'comparacion';dest.mkdir(exist_ok=True)
    raw.to_csv(dest/'oof_raw.csv',index=False);cal.to_csv(dest/'oof_calibrated.csv',index=False)
    pd.concat(inners).to_csv(dest/'inner.csv',index=False);base.json_write(dest/'calibradores.json',calibrators)
    ci=local.intervals(reference,cal,alpha=.05/7);ci_secondary=local.intervals(secondary,cal,alpha=.05/7)
    delta={k:diagnosis.metrics(cal)[k]-diagnosis.metrics(reference)[k] for k in ('log_loss','brier','roc_auc','mean_fold_auc')}
    report={'raw':diagnosis.metrics(raw),'calibrated':diagnosis.metrics(cal),'delta_vs_control':delta,
            'intervals':ci,'intervals_vs_v003':ci_secondary,'alpha':.05/7,'test_images_loaded':0,
            'promising_internal_result':ci['log_loss']['upper']<0 and ci['mean_fold_auc']['lower']>=-.01 and delta['brier']<=0}
    base.json_write(dest/'resumen.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--salida',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--prueba',action='store_true');args=parser.parse_args()
    output=args.salida/'pruebas' if args.prueba else args.salida;output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (output/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'status':'running','started_at':previous.now(),'pid':os.getpid(),'completed_runs':0,'total_runs':1 if args.prueba else 10}
        def progress(data):state.update(data,updated_at=previous.now());base.json_write(output/'estado.json',state)
        progress({'stage':'verifying'})
        try:
            torch.set_num_threads(8);base.seed_everything(transfer.SEED)
            samples,old=diagnosis.verify_inputs(base.pdatos.DATOS,previous.DEFAULT_OUTPUT)
            protected=previous.protected_snapshot();config=settings(args.prueba)
            sources={Path(p).name:base.sha256(p) for p in (__file__,base.__file__,previous.__file__,transfer.__file__,local.__file__,diagnosis.__file__)}
            env=base.environment(torch.device('cuda'));env.pop('git_commit',None)
            params={'sources':sources,'config':config,'protected':protected,'data':previous.signature(samples),
                'images':old['parameters']['train_png_sha256'],'environment':env,
                'references':{p.name:base.sha256(p) for p in transfer.REFERENCE.glob('oof_*.csv')}}
            fingerprint=base.config_hash(params);protocol=output/'protocolo.json'
            if protocol.exists() and json.loads(protocol.read_text())['fingerprint']!=fingerprint:raise ValueError('Protocolo distinto')
            if not protocol.exists():base.json_write(protocol,{'fingerprint':fingerprint,'parameters':params,'created_at':previous.now()})
            cache,weights=prefix_cache(samples,base.pdatos.DATOS,args.salida/'cache',torch.device('cuda'),progress)
            backbone=torch.load(weights,map_location='cpu',weights_only=True)
            for fold in ([0] if args.prueba else range(5)):
                for seed in ([42] if args.prueba else (42,2026)):
                    progress({'stage':'training','fold':fold,'seed':seed,'epoch':0})
                    run_one(samples,cache,backbone,output,seed,fold,config,torch.device('cuda'),progress)
                    progress({'completed_runs':state['completed_runs']+1})
                    if previous.protected_snapshot()!=protected:raise RuntimeError('Archivo protegido cambiado')
            if not args.prueba:
                progress({'stage':'comparison'});report=compare(samples,output);print(json.dumps(report,indent=2),flush=True)
            progress({'status':'complete','stage':'complete','finished_at':previous.now(),'protected_unchanged':True})
        except BaseException as exc:
            progress({'status':'failed','error':str(exc)});traceback.print_exc();raise


if __name__=='__main__':main()
