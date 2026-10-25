"""Experimentos de imagen por paciente; no carga las imágenes del test.

CNN propia desde cero y/o red sobre descriptores calculados solo de píxeles.
Los identificadores y la cohorte sirven exclusivamente para particiones/auditoría.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score, brier_score_loss
import torch
from torch import nn
from torch.nn import functional as F

from . import entrenamiento as base
from . import entrenamiento_imagen as original
from . import datos
from .paths import DATA_DIR, RESULTS_DIR

OUTPUT = RESULTS_DIR / '20_busqueda_imagen_20261025'
CACHE_VERSION = 'pixels128_box_v1_descriptors_v1'


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def prepare(output=OUTPUT, root=DATA_DIR):
    """Caché sin ajuste estadístico; las rutas proceden solo de train."""
    output, root = Path(output), Path(root)
    cache = output / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    samples = base.cargar_train(root, include_clinical=False)
    signature = hashlib.sha256(samples.to_csv(index=False).encode()).hexdigest()
    if (cache/'ready.json').exists():
        meta = json.loads((cache/'ready.json').read_text())
        if meta['signature'] != signature or meta['version'] != CACHE_VERSION:
            raise ValueError('Caché incompatible: usar una carpeta nueva')
        return
    patients = samples.drop_duplicates('patient_id')[['patient_id','pCR','fold','dataset']]
    patients = patients.sort_values('patient_id').reset_index(drop=True)
    ordered = [samples[samples.patient_id.eq(pid)].sort_values('slice_index')
               for pid in patients.patient_id]
    arrays = np.lib.format.open_memmap(cache/'images.npy', mode='w+', dtype=np.uint8,
                                    shape=(len(patients), 10, 3, 128, 128))
    arrays[:] = 0
    masks = np.zeros((len(patients),10), dtype=bool)

    def load_patient(frame):
        if len(frame) > 10 or not frame.split.eq('train').all():
            raise ValueError('Solo train, como máximo diez cortes')
        images = []
        for row in frame.itertuples():
            planes = []
            for name in datos.COLUMNAS_RUTA:
                path = root/getattr(row,name)
                if 'test' in path.parts:
                    raise ValueError('Ruta de test prohibida')
                with Image.open(path) as im:
                    if im.size != (256,256) or im.mode != 'L':
                        raise ValueError(f'PNG inesperado: {path}')
                    planes.append(np.asarray(im.resize((128,128),Image.Resampling.BOX)))
            images.append(np.stack(planes))
        return np.stack(images)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, arr in enumerate(pool.map(load_patient, ordered)):
            arrays[i,:len(arr)] = arr
            masks[i,:len(arr)] = True
            if (i+1)%100 == 0:
                print(f'cache {i+1}/{len(patients)}',flush=True)
    arrays.flush()
    patients.to_csv(cache/'patients.csv', index=False)
    np.save(cache/'mask.npy',masks)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = []
    for i in range(0,len(patients),12):
        x = torch.tensor(np.array(arrays[i:i+12]), device=device).float()/255
        b,s,c,h,w=x.shape
        per_slice = pixel_descriptors(x.flatten(0,1)).reshape(b,s,-1)
        mask = torch.tensor(masks[i:i+12],device=device)
        weights=mask[:,:,None].float()
        mean=(per_slice*weights).sum(1)/weights.sum(1)
        std=(((per_slice-mean[:,None])**2*weights).sum(1)/weights.sum(1)+1e-8).sqrt()
        features.append(torch.cat([mean,std],1).cpu().numpy())
    np.save(cache/'features.npy',np.concatenate(features).astype('float32'))
    write_json(cache/'ready.json',{'version':CACHE_VERSION,'signature':signature,
                                 'patients':len(patients),'test_images_loaded':0,
                                 'predictors':'only PRE EARLY LATE pixels'})


@torch.no_grad()
def pixel_descriptors(x):
    """Resúmenes de intensidad, realce y textura; sin máscaras externas."""
    pre,early,late=x[:,0:1],x[:,1:2],x[:,2:3]
    d=early-pre
    maps=torch.cat([pre,early,late,d,late-pre,late-early,
                    (d/(pre+.1)).clamp(-2,4)/4,
                    ((late-early)/(d.abs()+.1)).clamp(-2,2)/2],1)
    vectors=[]
    # Posiciones prefijadas, no coordenadas de tumor ni datos clínicos.
    for margin in (0,32,48):
        z=maps if margin==0 else maps[:,:,margin:-margin,margin:-margin]
        z=z.flatten(2)
        qs=torch.quantile(z,torch.tensor([.1,.25,.5,.75,.9],device=x.device),dim=-1)
        vectors.extend([z.mean(-1),z.std(-1),*qs.unbind(0)])
    # Textura local a distintas escalas, sin biblioteca externa ni etiquetas.
    for kernel in (3,9):
        local=F.avg_pool2d(maps,kernel,1,kernel//2)
        vectors.extend([(maps-local).abs().mean((-1,-2)),
                        (maps-local).square().mean((-1,-2)).sqrt()])
    for threshold in (.05,.15,.30):
        selected=d>threshold
        den=selected.sum((-1,-2)).clamp_min(1)
        vectors.extend([selected.float().mean((-1,-2)),
                        (maps*selected).sum((-1,-2))/den])
    return torch.cat(vectors,1)


def enhancement_crop(x, size=64, prior_strength=1.5):
    """ROI heurística de píxeles; nunca utiliza la máscara o clínica original.

    Máximo de realce medio local con prior espacial central suave, prefijado.
    No se presenta como segmentación de tumor.
    """
    if x.shape[-2:] != (128,128):
        raise ValueError('La ROI espera imágenes de 128 x 128')
    d=(x[:,1:2]-x[:,0:1]).clamp_min(0)
    score=F.avg_pool2d(d,25,1,12).squeeze(1)
    coords=torch.linspace(-1,1,128,device=x.device)
    prior=torch.exp(-prior_strength*(coords[:,None].square()+coords[None,:].square()))
    score=score*prior
    # Recorte sin extrapolación: centros confinados al interior válido.
    margin=size//2
    score=score[:,margin:128-margin+1,margin:128-margin+1]
    width=score.shape[-1]
    index=score.flatten(1).argmax(1)
    row=index//width;col=index%width
    delta=torch.arange(size,device=x.device)
    roi=x[torch.arange(len(x),device=x.device)[:,None,None,None],
          torch.arange(3,device=x.device)[None,:,None,None],
          row[:,None,None,None]+delta[None,None,:,None],
          col[:,None,None,None]+delta[None,None,None,:]]
    return F.interpolate(roi,(128,128),mode='bilinear',align_corners=False)


def prepare_roi_features(output=OUTPUT):
    cache=Path(output)/'cache'
    destination=cache/'features_roi_v2.npy'
    if destination.exists():
        return
    torch.set_num_threads(6)
    arrays=np.load(cache/'images.npy',mmap_mode='r')
    masks=np.load(cache/'mask.npy')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    features=[]
    for start in range(0,len(arrays),12):
        x=torch.tensor(np.array(arrays[start:start+12]),device=device).float()/255
        b,s=x.shape[:2]
        roi=enhancement_crop(x.flatten(0,1))
        per_slice=pixel_descriptors(roi).reshape(b,s,-1)
        weights=torch.tensor(masks[start:start+12,:,None],device=device).float()
        mean=(per_slice*weights).sum(1)/weights.sum(1)
        std=(((per_slice-mean[:,None]).square()*weights).sum(1)/weights.sum(1)+1e-8).sqrt()
        features.append(torch.cat([mean,std],1).cpu().numpy())
    joined=np.concatenate([np.load(cache/'features.npy'),np.concatenate(features)],1)
    np.save(destination,joined.astype('float32'))


def split_indices(patients, outer_fold):
    # Reutiliza la separación estratificada por cohorte y etiqueta del proyecto.
    stub=patients.copy()
    stub['split']='train'
    cfg=original.configuracion()
    cfg['evaluation']['inner_selection_fraction']=.22
    cfg['evaluation']['inner_split_seed']=20261025
    fit,selection,outer=original.particion_anidada(stub,outer_fold,cfg)
    return tuple(np.flatnonzero(patients.patient_id.isin(f.patient_id))
                 for f in (fit,selection,outer))


class Block(nn.Module):
    def __init__(self,cin,cout,norm='group'):
        super().__init__()
        normalization=lambda: (nn.GroupNorm(4,cout) if norm=='group' else nn.BatchNorm2d(cout))
        self.net=nn.Sequential(nn.Conv2d(cin,cout,3,stride=2,padding=1,bias=False),
                               normalization(),nn.SiLU(),
                               nn.Conv2d(cout,cout,3,padding=1,bias=False),
                               normalization(),nn.SiLU())
    def forward(self,x):
        return self.net(x)


class PatientNet(nn.Module):
    def __init__(self,config,n_features):
        super().__init__()
        self.config=deepcopy(config)
        mode=config['mode']; widths=config.get('widths',[12,24,48,64])
        spatial_size=16 if config.get('spatial_pool')=='grid4' else 2
        view_count=2 if config.get('view')=='multires_native' else 1
        slice_features=widths[-1]*spatial_size*view_count
        self.register_buffer('feature_mean',torch.zeros(n_features))
        self.register_buffer('feature_std',torch.ones(n_features))
        self.feature_net=None
        feature_hidden=int(config.get('feature_hidden',32))
        self.linear_features=config.get('feature_head')=='linear'
        if mode in ('features','hybrid'):
            if self.linear_features:
                self.feature_net=nn.Linear(n_features,1)
                nn.init.zeros_(self.feature_net.weight)
                nn.init.constant_(self.feature_net.bias,-.8)
            else:
                self.feature_net=nn.Sequential(nn.Dropout(config['dropout']),
                    nn.Linear(n_features,feature_hidden),nn.SiLU(),nn.Dropout(config['dropout']))
        self.encoder=None
        if mode in ('cnn','hybrid'):
            channels=6 if config.get('kinetics',True) else 3
            self.encoder=nn.Sequential(*[Block(a,b,config.get('norm','group'))
                for a,b in zip([channels,*widths[:-1]],widths)])
            self.attention=nn.Sequential(nn.Linear(slice_features,32),nn.Tanh(),nn.Linear(32,1))
            nn.init.zeros_(self.attention[-1].weight)
            nn.init.zeros_(self.attention[-1].bias)
            if config.get('slice_pool')=='sequence':
                self.axial_conv1=nn.Conv1d(widths[-1]*4,32,3,padding=1)
                self.axial_conv2=nn.Conv1d(32,32,3,padding=1)
        cnn_features=(64 if config.get('slice_pool')=='sequence' else slice_features*2)
        n=(cnn_features if self.encoder is not None else 0)+(feature_hidden if self.feature_net is not None and not self.linear_features else 0)
        self.classifier=(nn.Sequential(nn.Dropout(config['dropout']),nn.Linear(n,32),
                                      nn.SiLU(),nn.Dropout(config['dropout']),nn.Linear(32,1))
                         if n else None)

    def forward(self,x,mask,features):
        parts=[]
        linear_logit=None
        if self.feature_net is not None:
            z=((features-self.feature_mean)/self.feature_std).clamp(-6,6)
            if self.linear_features:
                linear_logit=self.feature_net(z).squeeze(1)
            else:
                parts.append(self.feature_net(z))
        if self.encoder is not None:
            b,s,c,h,w=x.shape
            im=x.flatten(0,1)
            views=1
            if self.config.get('view')=='multires_native':
                if (h,w)!=(256,256):raise ValueError('La CNN nativa requiere 256 px')
                im=torch.cat([F.avg_pool2d(im,2),im[:,:,64:192,64:192]],0)
                views=2
            if self.config.get('view')=='roi':
                im=enhancement_crop(im,prior_strength=self.config.get('prior_strength',1.5))
            if self.config.get('kinetics',True):
                im=torch.cat([2*im-1,3*(im[:,1:2]-im[:,0:1]),
                              3*(im[:,2:3]-im[:,0:1]),3*(im[:,2:3]-im[:,1:2])],1)
            else:
                im=2*im-1
            z=self.encoder(im)
            if self.config.get('slice_pool')=='sequence':
                seq=F.adaptive_avg_pool2d(z,2).flatten(1).reshape(b,s,-1).transpose(1,2)
                seq=seq*mask[:,None]
                seq=F.silu(self.axial_conv1(seq))*mask[:,None]
                seq=F.silu(self.axial_conv2(seq))*mask[:,None]
                pooled=seq.sum(2)/mask.sum(1)[:,None].clamp_min(1)
                second=seq.masked_fill(~mask[:,None],-1e4).amax(2)
            else:
                if self.config.get('spatial_pool')=='grid4':
                    z=F.adaptive_avg_pool2d(z,4).flatten(1)
                else:
                    z=torch.cat([z.mean((-1,-2)),z.amax((-1,-2))],1)
                if views==2:z=torch.cat([z[:b*s],z[b*s:]],1)
                z=z.reshape(b,s,-1)
                scores=self.attention(z).squeeze(-1).masked_fill(~mask,-1e4)
                pooled=(scores.softmax(1)[:,:,None]*z).sum(1)
                if self.config.get('slice_pool')=='attention_mean':
                    second=(z*mask[:,:,None]).sum(1)/mask.sum(1)[:,None].clamp_min(1)
                else:
                    second=z.masked_fill(~mask[:,:,None],-1e4).amax(1)
            parts.extend([pooled,second])
        out=self.classifier(torch.cat(parts,1)).squeeze(1) if self.classifier is not None else None
        if linear_logit is not None:
            return linear_logit if out is None else out+linear_logit
        return out


def configuration(mode='hybrid',**updates):
    cfg=dict(mode=mode,dropout=.5,widths=[12,24,48,64],norm='group',kinetics=True,
             lr=3e-4,weight_decay=.03,epochs=45,patience=10,min_epochs=10,
             batch_size=24,train_slices=4,mixup=.2,weighted=False,
             selection_metric='inner_bce_loss',label_smoothing=.04)
    cfg.update(updates)
    return cfg


def make_model(config,n_features):
    if config.get('model_kind') not in (None, 'cnn_2d'):
        raise ValueError('Este entrenador solo admite la CNN 2D propia')
    return PatientNet(config,n_features)


def augment(x):
    """Misma geometría y ganancia para todas las fases y cortes de la paciente."""
    b,s,c,h,w=x.shape
    flip=torch.rand(b,device=x.device)<.5
    x[flip]=x[flip].flip(-1)
    flip=torch.rand(b,device=x.device)<.5
    x[flip]=x[flip].flip(-2)
    theta=torch.zeros(b,2,3,device=x.device)
    angle=(torch.rand(b,device=x.device)-.5)*.35
    scale=.92+.16*torch.rand(b,device=x.device)
    theta[:,0,0]=theta[:,1,1]=torch.cos(angle)*scale
    theta[:,0,1]=-torch.sin(angle)*scale
    theta[:,1,0]=torch.sin(angle)*scale
    theta[:,:,2]=(torch.rand(b,2,device=x.device)-.5)*.08
    theta=theta[:,None].expand(-1,s,-1,-1).reshape(b*s,2,3)
    grid=F.affine_grid(theta,(b*s,c,h,w),align_corners=False)
    x=F.grid_sample(x.flatten(0,1),grid,align_corners=False).reshape(b,s,c,h,w)
    gain=(.9+.2*torch.rand(b,1,1,1,1,device=x.device))
    return (x*gain).clamp(0,1)


def metrics(y,p):
    return {'auc':float(roc_auc_score(y,p)),
            'bce':float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])),
            'ap':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p))}


def load_cache(output=OUTPUT,device='cuda',roi_features=False,feature_file=None,image_file=None):
    cache=Path(output)/'cache'
    if json.loads((cache/'ready.json').read_text())['version']!=CACHE_VERSION:
        raise ValueError('Caché incompatible')
    return (pd.read_csv(cache/'patients.csv'),
            torch.from_numpy(np.load(cache/(image_file or 'images.npy'))).to(device),
            torch.from_numpy(np.load(cache/'mask.npy')).to(device),
            torch.from_numpy(np.load(cache/(feature_file or ('features_roi_v2.npy' if roi_features else 'features.npy')))).to(device))


@torch.no_grad()
def predict(model,images,masks,features,indices,batch_size=12):
    model.eval(); ps=[]
    device=features.device
    for start in range(0,len(indices),batch_size):
        ix=torch.as_tensor(indices[start:start+batch_size],device=device)
        x=images[ix].float()/255 if model.encoder is not None else None
        with torch.autocast(device.type,enabled=device.type=='cuda',dtype=torch.float16):
            logit=model(x,masks[ix],features[ix])
        ps.extend(logit.float().sigmoid().cpu().tolist())
    return np.asarray(ps)


def plot_history(run_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h=pd.read_csv(Path(run_dir)/'history.csv')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for split in ('fit','selection'):
        axs[0].plot(h.epoch,h[f'{split}_auc'],label=split)
        axs[1].plot(h.epoch,h[f'{split}_bce'],label=split)
    axs[0].set(ylabel='ROC-AUC por paciente',ylim=(.4,1))
    axs[1].set(ylabel='BCE sin aumentos')
    for ax in axs:
        ax.set_xlabel('Época');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(Path(run_dir)/'curvas.png',dpi=140);plt.close(fig)


def train(cache,config,name,fold=0,seed=42,output=OUTPUT,evaluate_outer=False,smoke=False):
    patients,images,masks,features=cache
    device=features.device
    fit,selection,outer=split_indices(patients,fold)
    if smoke:
        fit=np.concatenate([fit[patients.iloc[fit].pCR.to_numpy()==v][:8] for v in (0,1)])
        selection=np.concatenate([selection[patients.iloc[selection].pCR.to_numpy()==v][:4] for v in (0,1)])
    run_dir=Path(output)/'runs'/name/f'fold_{fold}'/f'seed_{seed}'
    if (run_dir/'summary.json').exists():
        summary=json.loads((run_dir/'summary.json').read_text())
        if summary['config']!=config or summary['smoke']!=smoke:
            raise ValueError('Configuración cambiada, usar otro nombre')
        return summary
    run_dir.mkdir(parents=True,exist_ok=True)
    datos.fijar_semilla(seed+1000*fold)
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    model=make_model(config,features.shape[1]).to(device)
    model.feature_mean.copy_(features[fit].mean(0))
    model.feature_std.copy_(features[fit].std(0).clamp_min(config.get('feature_std_min',.02)))
    averaged=deepcopy(model).eval() if config.get('ema_decay') else None
    if averaged is not None:
        for parameter in averaged.parameters():
            parameter.requires_grad_(False)
    y=torch.tensor(patients.pCR.to_numpy(),device=device,dtype=torch.float32)
    pos_weight=(1-y[fit].mean())/y[fit].mean() if config['weighted'] else torch.tensor(1.,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['epochs'],eta_min=config['lr']*.05)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    write_json(run_dir/'config.json',config)
    write_json(run_dir/'provenance.json',{'source_sha256':base.sha256(Path(__file__)),
        'cache':json.loads((Path(output)/'cache/ready.json').read_text()),
        'features_sha256':base.sha256(Path(output)/'cache'/(config.get('feature_file') or ('features_roi_v2.npy' if config.get('roi_features') else 'features.npy'))),
        'environment':base.environment(device)})
    (run_dir/'training_source.py').write_text(Path(__file__).read_text())
    write_json(run_dir/'partitions.json',{k:patients.iloc[v].patient_id.tolist()
                                        for k,v in zip(('fit','selection','outer'),(fit,selection,outer))})
    best=float('inf');bad=0;history=[];start=time.monotonic()
    best_epoch=0
    for epoch in range(1,(2 if smoke else config['epochs'])+1):
        tic=time.monotonic(); model.train(); train_losses=[]
        order=torch.as_tensor(np.random.permutation(fit),device=device)
        for ix in order.split(config['batch_size']):
            target=y[ix]; mask=masks[ix]; feat=features[ix]
            if model.encoder is not None:
                # Muestra cortes reales uniformemente, sin dar peso extra a pacientes.
                scores=torch.rand(mask.shape,device=device).masked_fill(~mask,2.)
                selected=scores.argsort(1)[:,:config['train_slices']]
                if config.get('slice_pool')=='sequence':
                    selected=selected.sort(1).values
                x=images[ix[:,None],selected].float()/255
                mask=mask.gather(1,selected)
                x=augment(x)
            else:
                x=None
            target=target*(1-config['label_smoothing'])+config['label_smoothing']*.5
            if config['mixup']>0:
                lam=float(np.random.beta(config['mixup'],config['mixup']))
                other=torch.randperm(len(ix),device=device)
                feat=lam*feat+(1-lam)*feat[other]
                if x is not None:
                    x=lam*x+(1-lam)*x[other]
                    mask=mask|mask[other]
                target=lam*target+(1-lam)*target[other]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,enabled=device.type=='cuda',dtype=torch.float16):
                logit=model(x,mask,feat)
                loss=F.binary_cross_entropy_with_logits(logit,target,pos_weight=pos_weight)
                if config.get('rank_weight',0)>0:
                    # Pares de clases originales; se desactiva mixup en esta receta.
                    positives=logit[y[ix]>.5];negatives=logit[y[ix]<.5]
                    if positives.numel() and negatives.numel():
                        rank=F.softplus(-(positives[:,None]-negatives[None,:])).mean()
                        loss=loss+config['rank_weight']*rank
                if config.get('feature_l2',0)>0 and model.linear_features:
                    loss=loss+config['feature_l2']*model.feature_net.weight.square().sum()
            if not torch.isfinite(loss):
                raise RuntimeError('Pérdida no finita')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),2.)
            scaler.step(optimizer);scaler.update()
            if averaged is not None:
                with torch.no_grad():
                    decay=config['ema_decay']
                    for avg,current in zip(averaged.parameters(),model.parameters()):
                        avg.lerp_(current,1-decay)
                    for avg,current in zip(averaged.buffers(),model.buffers()):
                        # Normalización y escalado pertenecen al modelo evaluado.
                        if avg.is_floating_point():
                            avg.lerp_(current,1-decay)
                        else:
                            avg.copy_(current)
            train_losses.append(float(loss.detach()))
        scheduler.step()
        evaluated=averaged if averaged is not None else model
        pf=predict(evaluated,images,masks,features,fit)
        ps=predict(evaluated,images,masks,features,selection)
        mf=metrics(patients.iloc[fit].pCR,pf); ms=metrics(patients.iloc[selection].pCR,ps)
        # BCE normal permite evaluar calibración aunque el objetivo esté ponderado.
        # Para selección ponderada se utiliza BCE equilibrada, fijada a priori.
        selection_objective=ms['bce']
        if config['weighted']:
            ys=patients.iloc[selection].pCR.to_numpy()
            selection_objective=float(-(float(pos_weight)*ys*np.log(ps.clip(1e-7,1))+(1-ys)*np.log((1-ps).clip(1e-7,1))).mean())
        if config['selection_metric']=='inner_auc':
            selection_objective=-ms['auc']
        allowed=mf['auc']-ms['auc']<=config.get('max_auc_gap',float('inf'))
        improved=(selection_objective<best-1e-4 and allowed) or best_epoch==0
        if improved:
            best=selection_objective;bad=0;best_epoch=epoch
            torch.save({'state_dict':evaluated.state_dict(),'config':config,
                        'n_features':features.shape[1],'epoch':epoch,'seed':seed,'fold':fold,
                        'preprocessing':CACHE_VERSION},run_dir/'best.pt')
        else:
            bad+=1
        row=dict(epoch=epoch,train_augmented_loss=float(np.mean(train_losses)),
                 fit_auc=mf['auc'],selection_auc=ms['auc'],fit_bce=mf['bce'],selection_bce=ms['bce'],
                 selection_objective=selection_objective,auc_gap=mf['auc']-ms['auc'],
                 seconds=time.monotonic()-tic,best_epoch=best_epoch)
        history.append(row);pd.DataFrame(history).to_csv(run_dir/'history.csv',index=False)
        if epoch%3==0 or epoch==1:
            plot_history(run_dir)
        print(f'{name} f{fold} s{seed} e{epoch}: fit={mf["auc"]:.4f} val={ms["auc"]:.4f} '
              f'BCE={ms["bce"]:.4f} gap={row["auc_gap"]:+.3f} best={best_epoch} '
              f'{row["seconds"]:.1f}s',flush=True)
        if epoch>=config['min_epochs'] and bad>=config['patience']:
            break
    ckpt=torch.load(run_dir/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(ckpt['state_dict'])
    result=dict(status='complete',name=name,config=config,fold=fold,seed=seed,smoke=smoke,
                best_epoch=best_epoch,epochs=len(history),seconds=time.monotonic()-start,
                parameters=sum(p.numel() for p in model.parameters()),test_images_loaded=0,
                inference_sha256=base.sha256(run_dir/'best.pt'),splits={})
    chosen=history[best_epoch-1]
    result['selection_constraint_satisfied']=chosen['auc_gap']<=config.get('max_auc_gap',float('inf'))
    for split,ix in [('fit',fit),('selection',selection)]+([('outer',outer)] if evaluate_outer else []):
        p=predict(model,images,masks,features,ix)
        result['splits'][split]=metrics(patients.iloc[ix].pCR,p)
        frame=patients.iloc[ix].copy();frame['probability']=p
        frame.to_csv(run_dir/f'{split}_predictions.csv',index=False)
    plot_history(run_dir)
    write_json(run_dir/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','pilot','run'])
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--name',default='hybrid_v1')
    parser.add_argument('--mode',choices=['features','cnn','hybrid'],default='hybrid')
    parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--outer',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    prepare(args.output)
    if args.action=='prepare':
        return
    cfg=json.loads(args.config.read_text()) if args.config else configuration(args.mode)
    cache=load_cache(args.output,roi_features=cfg.get('roi_features',False),
                     feature_file=cfg.get('feature_file'),image_file=cfg.get('image_file'))
    print(json.dumps(train(cache,cfg,args.name,args.fold,args.seed,args.output,args.outer,args.smoke),indent=2))


if __name__=='__main__':
    main()
