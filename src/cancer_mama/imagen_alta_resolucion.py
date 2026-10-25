"""Representación de imagen de alta resolución, sin predictores de metadatos."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from . import imagen_mil as mil
from .imagen_textura import texture_descriptors
from . import entrenamiento as base
from .paths import RESULTS_DIR, DATA_DIR

OUTPUT=RESULTS_DIR/'21_imagen_textura_alta_resolucion_20261025'
VERSION='native256_center128_center64_lbp_glcm_v1'


@torch.no_grad()
def lbp_descriptors(x):
    maps=torch.cat([x,x[:,1:2]-x[:,0:1]],1)
    values=[]
    for region in [maps,maps[:,:,64:-64,64:-64]]:
        for radius in [1,2]:
            center=region[:,:,radius:-radius,radius:-radius]
            h,w=center.shape[-2:]
            bits=[]
            for dy,dx in [(-radius,-radius),(-radius,0),(-radius,radius),(0,radius),
                          (radius,radius),(radius,0),(radius,-radius),(0,-radius)]:
                neighbor=region[:,:,radius+dy:radius+dy+h,radius+dx:radius+dx+w]
                bits.append(neighbor>=center)
            bits=torch.stack(bits,2)
            transitions=(bits!=bits.roll(1,2)).sum(2)
            code=bits.sum(2).masked_fill(transitions>2,9)
            tissue=region[:,:3,radius:-radius,radius:-radius].amax(1,keepdim=True)>.03
            weights=tissue.expand_as(code).float().flatten(2)
            bins=torch.zeros(len(x),4,10,device=x.device)
            bins.scatter_add_(2,code.flatten(2),weights)
            values.append((bins/bins.sum(2,keepdim=True).clamp_min(1)).flatten(1))
    return torch.cat(values,1)


@torch.no_grad()
def descriptors(x):
    if x.shape[1:]!=(3,256,256):raise ValueError('Se requieren fases originales 256 px')
    center128=x[:,:,64:192,64:192]
    center64=F.interpolate(x[:,:,96:160,96:160],size=(128,128),mode='nearest')
    parts=[mil.pixel_descriptors(x),mil.pixel_descriptors(center128),
           mil.pixel_descriptors(center64),texture_descriptors(x),
           texture_descriptors(center128),texture_descriptors(center64),lbp_descriptors(x)]
    maps=torch.cat([x,x[:,1:2]-x[:,0:1]],1)
    for kernel in [3,7,15]:
        residual=maps-F.avg_pool2d(maps,kernel,1,kernel//2)
        for margin in [0,64]:
            z=residual if margin==0 else residual[:,:,margin:-margin,margin:-margin]
            parts.extend([z.abs().mean((-1,-2)),z.square().mean((-1,-2)).sqrt(),
                          z.abs().amax((-1,-2))])
    return torch.cat(parts,1)


def aggregate(per_slice):
    mean=per_slice.mean(0)
    std=((per_slice-mean).square().mean(0)+1e-8).sqrt()
    return torch.cat([mean,std])


def prepare(output=OUTPUT,root=DATA_DIR,device='cuda'):
    output,root=Path(output),Path(root);cache=output/'cache';cache.mkdir(parents=True,exist_ok=True)
    for name in ['images.npy','mask.npy','patients.csv','features.npy','ready.json']:
        dest=cache/name
        if not dest.exists():os.link(mil.OUTPUT/'cache'/name,dest)
    if (cache/'features_native_v1.npy').exists():return
    rows=base.cargar_train(root,include_clinical=False)
    import pandas as pd
    patients=pd.read_csv(cache/'patients.csv')
    groups=[rows[rows.patient_id.eq(pid)].sort_values('slice_index') for pid in patients.patient_id]
    def load(frame):
        images=[]
        for row in frame.itertuples():
            phases=[]
            for column in ['path_pre','path_early','path_late']:
                path=root/getattr(row,column)
                if 'test' in path.parts:raise ValueError('Test prohibido')
                with Image.open(path) as im:
                    if im.mode!='L' or im.size!=(256,256):raise ValueError('PNG inesperado')
                    phases.append(np.asarray(im))
            images.append(np.stack(phases))
        return np.stack(images)
    torch.set_num_threads(4);features=[]
    # Lecturas en paralelo; extracción por paciente no aprende estadísticas.
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,arr in enumerate(pool.map(load,groups)):
            x=torch.tensor(arr,device=device,dtype=torch.float32)/255
            features.append(aggregate(descriptors(x)).cpu().numpy())
            if (i+1)%100==0:print(f'Alta resolución {i+1}/{len(groups)}',flush=True)
    features=np.stack(features).astype('float32')
    if not np.isfinite(features).all():raise ValueError('Descriptores no finitos')
    np.save(cache/'features_native_v1.npy',features)
    mil.write_json(cache/'native_provenance.json',dict(version=VERSION,
        patients=len(patients),features=features.shape[1],source_sha256=base.sha256(Path(__file__)),
        features_sha256=base.sha256(cache/'features_native_v1.npy'),test_images_loaded=0))


if __name__=='__main__':prepare()
