"""Inferencia cacheable de la aplicación docente de una sola tripleta PNG."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .app_logic import PHASES, ValidatedSample
from . import entrenamiento as base
from . import inferencia_imagen as runtime
from .paths import PROJECT_ROOT

DEFAULT_IMAGE_MANIFEST = PROJECT_ROOT/'modelos/versiones/v008_cnn_cinetica_un_corte_20261025/manifiesto.json'


def image_key(sample: ValidatedSample) -> str:
    digest=sha256()
    for phase in PHASES:digest.update(sample.arrays[phase].tobytes())
    return digest.hexdigest()


class SingleSliceEngine:
    def __init__(self,manifest_path: Path,device: str | None=None):
        self.path=Path(manifest_path).resolve()
        self.manifest=runtime.cargar_manifest(self.path)
        if self.manifest.get('inference_unit')!='single_slice':
            raise ValueError('La aplicación necesita un paquete validado para un solo corte')
        self.checksum=base.sha256(self.path)
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.models=[runtime.cargar_modelo(self.path,item).to(self.device).eval().requires_grad_(False)
                     for item in self.manifest['models']]

    @torch.inference_mode()
    def predict(self,sample: ValidatedSample) -> dict:
        started=perf_counter()
        x=torch.from_numpy(np.stack([sample.arrays[p] for p in PHASES])).float()[None].to(self.device)/255
        predictions=np.array([float(model(x).sigmoid().flatten()[0]) for model in self.models])
        if not np.isfinite(predictions).all():raise FloatingPointError('Probabilidad no finita')
        raw=float(predictions.mean())
        calibrated=float(base.apply_calibration(np.array([raw]),self.manifest['calibration'])[0])
        if self.device.type=='cuda':torch.cuda.synchronize()
        return {'sample_id':sample.sample_id,'input_sha256':image_key(sample),
                'version':self.path.parent.name,'manifest_sha256':self.checksum,
                'probability_raw':raw,'probability_pcr':calibrated,
                'threshold':float(self.manifest['threshold']),
                'predicted_class':int(calibrated>=self.manifest['threshold']),
                'positive_class':'pCR','inference_unit':'single_slice','models':len(self.models),
                'device':str(self.device),'elapsed_ms':(perf_counter()-started)*1000,
                'educational_only':True,'clinical_inputs_used':False}

    def card(self) -> dict:
        selection=self.manifest['selection']
        return {'version':self.path.parent.name,'checksum':self.checksum,
                'threshold':self.manifest['threshold'],'models':len(self.models),
                'loss':self.manifest['training']['loss'],
                'training':self.manifest['training'],
                'internal_check':selection['metrics_internal_check'],
                'auc_ci95':selection['auc_ci95_internal_check'],
                'status':self.manifest['status']}
