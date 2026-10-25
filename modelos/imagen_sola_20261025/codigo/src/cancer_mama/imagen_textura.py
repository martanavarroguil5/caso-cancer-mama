"""Descriptores de textura fijos derivados de los PNG, sin aprendizaje externo."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import imagen_mil as mil


@torch.no_grad()
def texture_descriptors(x):
    """GLCM simétrica promediada sobre direcciones, ocho niveles fijos.

    Dos regiones, tres distancias y seis propiedades por cuatro mapas. Las
    intensidades, distancias y regiones se fijan antes de ver las etiquetas.
    """
    x=F.avg_pool2d(x,2)
    maps=torch.cat([x,x[:,1:2]-x[:,0:1]],1)
    maps[:,3]=((maps[:,3]+1)/2)
    maps=maps.clamp(0,1)
    vectors=[]
    grid=torch.arange(8,device=x.device,dtype=x.dtype)
    ii,jj=torch.meshgrid(grid,grid,indexing='ij')
    delta=(ii-jj).abs()
    for margin in (0,16):
        region=maps if margin==0 else maps[:,:,margin:-margin,margin:-margin]
        bins=(region*7.999).long()
        n,c,h,w=bins.shape
        flat=bins.flatten(0,1)
        for distance in (1,2,4):
            counts=torch.zeros(n*c,64,device=x.device)
            for dy,dx in ((0,distance),(distance,0),(distance,distance),(distance,-distance)):
                y0,y1=max(0,-dy),min(h,h-dy)
                x0,x1=max(0,-dx),min(w,w-dx)
                a=flat[:,y0:y1,x0:x1].flatten(1)
                b=flat[:,y0+dy:y1+dy,x0+dx:x1+dx].flatten(1)
                valid=((a+b)>0).float()
                counts.scatter_add_(1,8*a+b,valid)
                counts.scatter_add_(1,8*b+a,valid)
            p=(counts/counts.sum(1,keepdim=True).clamp_min(1)).reshape(n,c,8,8)
            mui=(p*ii).sum((-1,-2));muj=(p*jj).sum((-1,-2))
            vi=(p*(ii-mui[:,:,None,None]).square()).sum((-1,-2))
            vj=(p*(jj-muj[:,:,None,None]).square()).sum((-1,-2))
            corr=(p*(ii-mui[:,:,None,None])*(jj-muj[:,:,None,None])).sum((-1,-2))/(vi*vj+1e-8).sqrt()
            vectors.extend([(p*delta.square()).sum((-1,-2))/49,
                            (p*delta).sum((-1,-2))/7,
                            (p/(1+delta.square())).sum((-1,-2)),
                            p.square().sum((-1,-2)),
                            -(p*p.clamp_min(1e-8).log()).sum((-1,-2))/4.2,
                            corr])
    return torch.cat(vectors,1)


def prepare(output=mil.OUTPUT):
    cache=Path(output)/'cache'
    destination=cache/'features_texture_v3.npy'
    if destination.exists():
        return
    torch.set_num_threads(6)
    arrays=np.load(cache/'images.npy',mmap_mode='r')
    masks=np.load(cache/'mask.npy')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    features=[]
    for start in range(0,len(arrays),8):
        x=torch.tensor(np.array(arrays[start:start+8]),device=device).float()/255
        b,s=x.shape[:2]
        raw=x.flatten(0,1)
        roi=mil.enhancement_crop(raw,prior_strength=6.)
        per_slice=torch.cat([mil.pixel_descriptors(roi),texture_descriptors(raw),texture_descriptors(roi)],1).reshape(b,s,-1)
        weights=torch.tensor(masks[start:start+8,:,None],device=device).float()
        mean=(per_slice*weights).sum(1)/weights.sum(1)
        std=(((per_slice-mean[:,None]).square()*weights).sum(1)/weights.sum(1)+1e-8).sqrt()
        features.append(torch.cat([mean,std],1).cpu().numpy())
        if start%160==0:
            print(f'Textura: {start}/{len(arrays)}',flush=True)
    joined=np.concatenate([np.load(cache/'features.npy'),np.concatenate(features)],1)
    np.save(destination,joined.astype('float32'))


if __name__=='__main__':
    prepare()
