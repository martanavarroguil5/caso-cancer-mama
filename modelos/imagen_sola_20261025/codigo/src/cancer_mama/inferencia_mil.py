"""Inferencia del candidato MIL usando únicamente PNG PRE/EARLY/LATE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .imagen_mil import PatientNet, pixel_descriptors, enhancement_crop
from .imagen_textura import texture_descriptors


@torch.no_grad()
def features_from_images(images,config):
    """Replica exactamente los descriptores de caché, sin ajustar estadísticas."""
    def aggregate(per_slice):
        mean=per_slice.mean(0)
        std=((per_slice-mean).square().mean(0)+1e-8).sqrt()
        return torch.cat([mean,std])
    full=aggregate(pixel_descriptors(images))
    if config.get('feature_file')=='features_texture_v3.npy':
        roi=enhancement_crop(images,prior_strength=6.)
        per_slice=torch.cat([pixel_descriptors(roi),texture_descriptors(images),
                             texture_descriptors(roi)],1)
        return torch.cat([full,aggregate(per_slice)])
    if config.get('roi_features'):
        return torch.cat([full,aggregate(pixel_descriptors(enhancement_crop(images)))])
    return full


def load_pngs(pre,early,late,device='cpu'):
    if not 1<=len(pre)<=10 or not len(pre)==len(early)==len(late):
        raise ValueError('Se requieren de 1 a 10 cortes alineados de cada fase')
    images=[]
    for triplet in zip(pre,early,late):
        phases=[]
        for path in triplet:
            with Image.open(path) as im:
                if im.mode!='L' or im.size!=(256,256):
                    raise ValueError('Se requieren PNG monocromos de 8 bits y 256 x 256')
                phases.append(np.asarray(im.resize((128,128),Image.Resampling.BOX)))
        images.append(np.stack(phases))
    return torch.tensor(np.stack(images),device=device,dtype=torch.float32)/255


@torch.no_grad()
def predict_checkpoint(path,images,device='cpu'):
    ckpt=torch.load(path,map_location=device,weights_only=True)
    model=PatientNet(ckpt['config'],ckpt['n_features']).to(device).eval()
    model.load_state_dict(ckpt['state_dict'])
    features=features_from_images(images,ckpt['config'])[None]
    mask=torch.ones(1,len(images),device=device,dtype=torch.bool)
    return float(model(images[None],mask,features).sigmoid()[0])


def predict_package(package,pre,early,late,device='cpu'):
    package=Path(package)
    manifest=json.loads((package/'manifest.json').read_text())
    from .entrenamiento import sha256
    torch.set_num_threads(4)
    images=load_pngs(pre,early,late,device)
    predictions=[]
    for entry in manifest['models']:
        path=package/entry['file']
        if sha256(path)!=entry['sha256']:
            raise ValueError(f'Checkpoint modificado: {path}')
        predictions.append(predict_checkpoint(path,images,device))
    group_probabilities=[]
    if manifest.get('groups'):
        for group in manifest['groups']:
            raw=float(np.mean([predictions[i] for i in group['model_indices']]))
            c=group['calibration']
            log_odds=float(np.log(np.clip(raw,1e-6,1-1e-6)/(1-np.clip(raw,1e-6,1-1e-6))))
            calibrated=1/(1+np.exp(-np.clip(c['intercept']+c['coef']*(log_odds-c['mean'])/c['std'],-30,30)))
            group_probabilities.append(float(calibrated))
    probability=float(np.mean(group_probabilities or predictions))
    return {'probability_pcr':probability,
            'model_probabilities':predictions,'models':len(predictions),
            'fold_group_probabilities':group_probabilities,
            'slices':len(images),'predictors':'PNG PRE/EARLY/LATE exclusivamente',
            'educational_only':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    for phase in ('pre','early','late'):
        parser.add_argument('--'+phase,nargs='+',type=Path,required=True)
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    args=parser.parse_args()
    print(json.dumps(predict_package(args.package,args.pre,args.early,args.late,args.device),indent=2))


if __name__=='__main__':
    main()
