"""Descriptores explícitos de los PNG DCE y fusión clínica regularizada."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
import torch
from torch.nn import functional as F

from . import entrenamiento as base
from . import experimento_transfer as transfer
from . import experimento_local as local
from . import experimento_loss as previous
from . import informe_loss as diagnosis
from .paths import RESULTS_DIR

DEFAULT_OUTPUT=RESULTS_DIR/'04_entrenamiento/textura_multimodal_20261010'
FAMILIES=('textura_lineal','textura_svm','textura_arboles')


def grid(family,smoke=False):
    result=[]
    for k in (8,32):
        if family=='textura_arboles':
            result.extend(dict(family=family,k=k,lr=lr,iterations=n) for lr in (.03,.1) for n in (100,250))
        else:
            result.extend(dict(family=family,k=k,C=c,gamma=gamma) for c in (.01,.1,1.,10.)
                          for gamma in ((.1,1.) if family.endswith('svm') else (None,)))
    if family not in FAMILIES:raise ValueError('Familia desconocida')
    return result[:1] if smoke else result


@torch.inference_mode()
def descriptors(images):
    if images.ndim!=4 or tuple(images.shape[1:])!=(3,256,256):raise ValueError('Tres fases 256x256 requeridas')
    x=images.float()/255. if images.dtype==torch.uint8 else images.float()
    if not torch.isfinite(x).all() or x.min()<0 or x.max()>1:raise ValueError('Intensidades no válidas')
    outputs=[];batch=len(x)
    ii,jj=torch.meshgrid(torch.arange(8,device=x.device),torch.arange(8,device=x.device),indexing='ij')
    ii=ii.flatten().float();jj=jj.flatten().float();difference=(ii-jj).square()
    for width in (256,128,64):
        start=(256-width)//2
        v=F.adaptive_avg_pool2d(x[:,:,start:start+width,start:start+width],(64,64))
        pre,early,late=v[:,0:1],v[:,1:2],v[:,2:3]
        maps=torch.cat((v,(early-pre+1)/2,(late-pre+1)/2,(late-early+1)/2),1)
        z=maps.flatten(0,1);flat=z.flatten(1)
        stats=[flat.mean(1),flat.std(1,unbiased=False),(z[:,:,1:]-z[:,:,:-1]).abs().mean((1,2)),
               (z[:,1:,:]-z[:,:-1,:]).abs().mean((1,2))]
        lap=z[:,1:-1,:-2]+z[:,1:-1,2:]+z[:,:-2,1:-1]+z[:,2:,1:-1]-4*z[:,1:-1,1:-1]
        stats.append(lap.square().mean((1,2)))
        # Estabiliza fronteras exactas de bin frente al orden de suma del pooling.
        bins=(flat*16+1e-5).long().clamp(0,15)
        hist=torch.zeros((len(flat),16),device=x.device,dtype=torch.int64)
        hist.scatter_add_(1,bins,torch.ones_like(bins));hist=hist.float()/flat.shape[1]
        stats.append(-(hist*hist.clamp_min(1e-10).log()).sum(1)/np.log(16))
        cdf=hist.cumsum(1)
        for quantile in (.1,.25,.5,.75,.9):
            stats.append(((cdf>=quantile).to(torch.int64).argmax(1).float()+.5)/16)
        quant=(z*8+1e-5).long().clamp(0,7)
        a=torch.cat((quant[:,:,:-1].flatten(1),quant[:,:-1,:].flatten(1)),1)
        b=torch.cat((quant[:,:,1:].flatten(1),quant[:,1:,:].flatten(1)),1)
        pairs=torch.cat((8*a+b,8*b+a),1)
        counts=torch.zeros((len(flat),64),device=x.device,dtype=torch.int64)
        counts.scatter_add_(1,pairs,torch.ones_like(pairs));p=counts.float()/pairs.shape[1]
        mu_i=(p*ii).sum(1);mu_j=(p*jj).sum(1)
        var_i=(p*(ii[None]-mu_i[:,None]).square()).sum(1)
        var_j=(p*(jj[None]-mu_j[:,None]).square()).sum(1)
        corr=(p*(ii[None]-mu_i[:,None])*(jj[None]-mu_j[:,None])).sum(1)/(var_i*var_j).sqrt().clamp_min(1e-8)
        stats.extend(((p*difference).sum(1)/49,(p/(1+difference)).sum(1),p.square().sum(1).sqrt(),
                      -(p*p.clamp_min(1e-10).log()).sum(1)/np.log(64),corr))
        outputs.append(torch.stack(stats,1).reshape(batch,-1))
    return torch.cat(outputs,1)


def extract(samples,root,output,smoke,progress):
    path=output/'features.npz';meta_path=output/'features.json'
    signature=previous.signature(samples)
    if path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['data_signature']!=signature or meta['sha256']!=base.sha256(path) or meta['smoke']!=smoke:
            raise ValueError('Características incompatibles')
        with np.load(path) as data:return data['patient_id'].tolist(),data['features'].copy()
    selected=samples
    if smoke:
        ids=samples.drop_duplicates('patient_id').groupby(['fold','pCR']).head(8).patient_id
        selected=samples[samples.patient_id.isin(ids)].groupby('patient_id',group_keys=False).head(2).reset_index(drop=True)
    cache=base.CacheTrain(samples,root,previous.DEFAULT_OUTPUT/'.cache',workers=4)
    loader=base.make_loader(selected,root,64,0 if smoke else 4,torch.Generator().manual_seed(transfer.SEED),cache=cache)
    features=[]
    for step,batch in enumerate(loader):
        features.append(descriptors(batch['image'].cuda()).cpu().numpy())
        if step%30==0 or step+1==len(loader):
            print(f'Texturas {step+1}/{len(loader)}',flush=True)
            progress({'stage':'extracting','batch':step+1,'batches':len(loader)})
    z=np.concatenate(features)
    ids,mean=transfer.patient_features(selected,z)
    groups=selected.groupby('patient_id').indices
    std=np.stack([z[groups[pid]].std(axis=0,dtype=np.float64) for pid in ids]).astype(np.float32)
    result=np.column_stack((mean,std))
    if not np.isfinite(result).all():raise ValueError('Descriptor no finito')
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,patient_id=np.asarray(ids,dtype=str),features=result);tmp.replace(path)
    base.json_write(meta_path,{'sha256':base.sha256(path),'data_signature':signature,'smoke':smoke,'shape':list(result.shape)})
    return ids,result


def fit_selected(family,features,patients,fi,si,smoke=False):
    y=patients.pCR.to_numpy();prep=local.fit_preprocessing(features[fi],patients.iloc[fi],y[fi])
    best=None;best_key=None;records=[]
    for config in grid(family,smoke):
        xf=local.transform(features[fi],patients.iloc[fi],prep,config['k'])
        xs=local.transform(features[si],patients.iloc[si],prep,config['k'])
        if family.endswith('svm'):model=SVC(C=config['C'],gamma=config['gamma'],class_weight='balanced',random_state=transfer.SEED)
        elif family.endswith('lineal'):model=LogisticRegression(C=config['C'],max_iter=4000,random_state=transfer.SEED)
        else:model=HistGradientBoostingClassifier(learning_rate=config['lr'],max_iter=config['iterations'],max_leaf_nodes=4,
            min_samples_leaf=30,l2_regularization=5.,early_stopping=False,random_state=transfer.SEED)
        model.fit(xf,y[fi]);p=local.probability(model,xs,family)
        loss=float(base.log_loss(y[si],p,labels=[0,1]));auc=float(base.roc_auc_score(y[si],p))
        key=(-auc,loss) if family.endswith('svm') else (loss,)
        records.append(dict(config,selection_logloss=loss,selection_auc=auc))
        if best_key is None or key<best_key:best_key=key;best={'config':config,'model':model,'preprocessing':prep}
    return best,records


def fit_fold(samples,patients,features,fold,output,smoke):
    split=previous.split_patients(samples,fold);sets=[set(x.patient_id) for x in split]
    fi,si,oi=[np.flatnonzero(patients.patient_id.isin(ids)) for ids in sets]
    folder=output/f'fold_{fold}';folder.mkdir(exist_ok=True);path=folder/'summary.json'
    expected={'fold':fold,'smoke':smoke,'partitions':{k:previous.signature(v) for k,v in zip(('fit','selection','outer'),split)}}
    if path.exists():
        record=json.loads(path.read_text())
        if any(record[k]!=v for k,v in expected.items()):raise ValueError('Particiones cambiadas')
        for name,sha in record['files'].items():
            if base.sha256(folder/name)!=sha:raise ValueError('Archivo cambiado')
        return record
    summaries={}
    for family in FAMILIES:
        chosen,records=fit_selected(family,features,patients,fi,si,smoke)
        x=local.transform(features,patients,chosen['preprocessing'],chosen['config']['k'])
        p=local.probability(chosen['model'],x,family)
        inner=transfer.prediction_frame(patients,si,p[si]);outer=transfer.prediction_frame(patients,oi,p[oi])
        cal,calibration,threshold=diagnosis.calibrate_inner(inner,outer)
        for scale,frame in (('raw',outer),('calibrated',cal)):frame.to_csv(folder/f'oof_{scale}_{family}.csv',index=False)
        inner.to_csv(folder/f'inner_{family}.csv',index=False)
        chosen.update(calibration=calibration,threshold=threshold,fit_ids=sorted(sets[0]),selection_ids=sorted(sets[1]),outer_ids=sorted(sets[2]))
        joblib.dump(chosen,folder/f'model_{family}.joblib',compress=3)
        base.json_write(folder/f'seleccion_{family}.json',records)
        summaries[family]={'config':chosen['config'],'outer_raw':diagnosis.metrics(outer),'outer_calibrated':diagnosis.metrics(cal)}
        print(f'Fold {fold} {family}: {chosen["config"]}',flush=True)
    record=dict(expected,models=summaries,files={p.name:base.sha256(p) for p in folder.iterdir() if p.is_file()})
    base.json_write(path,record);return record


def compare(samples,output):
    reference=pd.read_csv(transfer.REFERENCE/'oof_calibrated_inner_auc.csv')
    roster=samples.drop_duplicates('patient_id').sort_values('patient_id').rename(columns={'pCR':'label','dataset':'cohort'}).reset_index(drop=True)
    keys=['patient_id','label','fold','cohort']
    if not reference[keys].equals(roster[keys]):raise ValueError('Control desalineado')
    dest=output/'comparacion';dest.mkdir(exist_ok=True);metrics=[];comparisons={}
    for family in FAMILIES:
        for scale in ('raw','calibrated'):
            pieces=[]
            for fold in range(5):
                folder=output/f'fold_{fold}';summary=json.loads((folder/'summary.json').read_text());path=folder/f'oof_{scale}_{family}.csv'
                if summary['smoke'] or base.sha256(path)!=summary['files'][path.name]:raise ValueError('OOF inválida')
                pieces.append(pd.read_csv(path))
            x=pd.concat(pieces).sort_values('patient_id').reset_index(drop=True)
            if not x[keys].equals(reference[keys]):raise ValueError('Pacientes distintas')
            x.to_csv(dest/f'oof_{scale}_{family}.csv',index=False);m=diagnosis.metrics(x)
            metrics.append({'candidate':f'{family}_{scale}',**m})
            if scale=='calibrated':
                ci=local.intervals(reference,x,alpha=.05/10)
                delta={k:m[k]-diagnosis.metrics(reference)[k] for k in ('log_loss','brier','roc_auc','mean_fold_auc')}
                comparisons[family]={'delta_vs_control':delta,'intervals_99_5':ci,
                    'promising_internal_result':ci['log_loss']['upper']<0 and ci['mean_fold_auc']['lower']>=-.01 and delta['brier']<=0}
                print(f'Comparado {family}: {comparisons[family]["promising_internal_result"]}',flush=True)
    result={'metrics':metrics,'comparisons':comparisons,'alpha':.05/10,'bootstrap_repetitions':2000,'test_images_loaded':0,'adaptive_round':4}
    base.json_write(dest/'resumen.json',result);pd.DataFrame(metrics).drop(columns='fold_auc').to_csv(dest/'metricas.csv',index=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--salida',type=Path,default=DEFAULT_OUTPUT);parser.add_argument('--prueba',action='store_true');args=parser.parse_args()
    output=args.salida/'pruebas' if args.prueba else args.salida;output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (output/'runner.lock').open('a') as lock,threadpool_limits(limits=4):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'status':'running','pid':os.getpid(),'started_at':previous.now(),'smoke':args.prueba}
        def progress(data):state.update(data,updated_at=previous.now());base.json_write(output/'estado.json',state)
        progress({'stage':'verifying'})
        try:
            torch.set_num_threads(8);base.seed_everything(transfer.SEED)
            samples,old=diagnosis.verify_inputs(base.pdatos.DATOS,previous.DEFAULT_OUTPUT);protected=previous.protected_snapshot()
            env=base.environment(torch.device('cuda'));env.pop('git_commit',None)
            params={'sources':{Path(p).name:base.sha256(p) for p in (__file__,base.__file__,previous.__file__,transfer.__file__,local.__file__,diagnosis.__file__)},
                'protected':protected,'data':previous.signature(samples),'images':old['parameters']['train_png_sha256'],
                'grid':{f:grid(f,args.prueba) for f in FAMILIES},'smoke':args.prueba,'environment':env,
                'reference':base.sha256(transfer.REFERENCE/'oof_calibrated_inner_auc.csv')}
            fingerprint=base.config_hash(params);protocol=output/'protocolo.json'
            if protocol.exists() and json.loads(protocol.read_text())['fingerprint']!=fingerprint:raise ValueError('Protocolo distinto')
            if not protocol.exists():base.json_write(protocol,{'fingerprint':fingerprint,'parameters':params,'created_at':previous.now()})
            ids,z=extract(samples,base.pdatos.DATOS,output,args.prueba,progress)
            patients=samples.drop_duplicates('patient_id').set_index('patient_id').loc[ids].reset_index()
            for fold in ([0] if args.prueba else range(5)):
                progress({'stage':'training','fold':fold});fit_fold(samples,patients,z,fold,output,args.prueba)
            if not args.prueba:progress({'stage':'comparison'});result=compare(samples,output);print(json.dumps(result['metrics'],indent=2),flush=True)
            if previous.protected_snapshot()!=protected:raise RuntimeError('Archivos protegidos cambiados')
            progress({'status':'complete','stage':'complete','finished_at':previous.now(),'protected_unchanged':True})
        except BaseException as exc:progress({'status':'failed','error':str(exc)});traceback.print_exc();raise


if __name__=='__main__':main()
