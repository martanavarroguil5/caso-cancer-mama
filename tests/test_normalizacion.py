"""GroupNorm: compatibilidad histórica, pareado y reanudación con norma real."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from test_generalizacion import roster

spec=importlib.util.spec_from_file_location('ent_norm',Path(__file__).parents[1]/'04_entrenamiento.py')
ent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ent;spec.loader.exec_module(ent)


class TinyNormCNN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        norm=(torch.nn.GroupNorm(8,8) if config['model']['normalization']=='group' else torch.nn.BatchNorm2d(8))
        self.layers=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(8),torch.nn.Conv2d(3,8,1,bias=False),
            norm,torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),
            torch.nn.Dropout(.35),torch.nn.Linear(8,1))
    def forward(self,x):return self.layers(x)


class TestNormalizacion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_arquitectura_inicializacion_y_compatibilidad_batchnorm(self):
        configs=ent.configuraciones_normalizacion()
        torch.manual_seed(42);bn=ent.CNN(configs[0])
        torch.manual_seed(42);gn=ent.CNN(configs[1])
        self.assertEqual(ent.numero_parametros(bn),551913)
        self.assertEqual(ent.numero_parametros(gn),551913)
        for name,param in bn.named_parameters():self.assertTrue(torch.equal(param,dict(gn.named_parameters())[name]),name)
        norms=[m for m in gn.modules() if isinstance(m,torch.nn.GroupNorm)]
        self.assertEqual(len(norms),8)
        self.assertEqual(sum(isinstance(m,torch.nn.BatchNorm2d) for m in gn.modules()),0)
        self.assertTrue(all(m.num_groups==8 and m.eps==1e-5 and m.affine for m in norms))
        default=json.loads(json.dumps(configs[0]));default['model'].pop('normalization');default['model'].pop('norm_groups')
        torch.manual_seed(42);legacy=ent.CNN(default)
        legacy.load_state_dict(bn.state_dict(),strict=True)
        x=torch.rand(2,3,256,256)
        bn.eval();legacy.eval()
        with torch.no_grad():self.assertTrue(torch.equal(bn(x),legacy(x)))

    def test_groupnorm_independiente_de_otras_muestras_y_modo(self):
        model=ent.CNN(ent.configuraciones_normalizacion()[1])
        x=torch.randn(2,3,256,256)
        # La cabeza tiene dropout; se contrasta el extractor, donde solo cambia la norma.
        model.features.train()
        with torch.no_grad():
            single=model.features(x[:1]);together=model.features(x)[:1]
            model.features.eval();evaluation=model.features(x[:1])
        torch.testing.assert_close(single,together,rtol=1e-5,atol=1e-5)
        self.assertTrue(torch.equal(single,evaluation))

    def test_rechaza_normalizacion_y_grupos_invalidos(self):
        for norm,groups in [('layer',8),('group',0),('group',7),('group',2.5)]:
            with self.assertRaises(ValueError):ent.CNN({'normalization':norm,'norm_groups':groups})

    def test_config_solo_cambia_normalizacion(self):
        configs=ent.configuraciones_normalizacion()
        for c in configs:
            c.pop('name');c['model'].pop('normalization')
        self.assertEqual(*configs)
        self.assertEqual(configs[0]['training']['batch_policy'],'pacientes_completas')
        self.assertEqual(configs[0]['model']['norm_groups'],8)

    def test_reanudacion_con_groupnorm_y_rechazo_de_lotes_distintos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';(root/'metadata').mkdir(parents=True)
            samples=roster();samples.to_csv(root/'metadata/samples.csv',index=False)
            for number,row in enumerate(samples.itertuples()):
                for column in ent.pdatos.COLUMNAS_RUTA:
                    path=root/getattr(row,column);path.parent.mkdir(parents=True,exist_ok=True)
                    pixels=np.full((256,256),15+number%210,dtype=np.uint8);pixels[:128,:128]+=10
                    Image.fromarray(pixels).save(path)
            configs=ent.configuraciones_normalizacion(2,2,0)
            for c in configs:c['training']['cpu_threads']=2
            output=Path(tmp)/'paired';resumed=Path(tmp)/'resumed'
            with patch.object(ent,'CNN',TinyNormCNN),patch.object(ent,'graficar_historia'):
                for config in configs:ent.run_training(config,'weighted',42,0,root,output/'ejecuciones',torch.device('cpu'))
                original=ent.save_checkpoint
                def crash(*args,**kwargs):
                    original(*args,**kwargs)
                    if args[0].name=='last.pt' and args[5]==0:raise RuntimeError('interrupción real')
                with patch.object(ent,'save_checkpoint',side_effect=crash),self.assertRaisesRegex(RuntimeError,'interrupción'):
                    ent.run_training(configs[1],'weighted',42,0,root,resumed,torch.device('cpu'))
                ent.run_training(configs[1],'weighted',42,0,root,resumed,torch.device('cpu'))
            report=ent.verificar_lotes_pareados(output,configs,seeds=(42,),folds=(0,),normalization=True)
            self.assertEqual(report['status'],'passed')
            relative=Path('groupnorm/weighted/seed_42/fold_0')
            a=torch.load(output/'ejecuciones'/relative/'last.pt',weights_only=False)
            b=torch.load(resumed/relative/'last.pt',weights_only=False)
            for name,value in a['state_dict'].items():self.assertTrue(torch.equal(value,b['state_dict'][name]),name)
            for key in ('torch','loader','augmentation'):self.assertTrue(torch.equal(a['rng'][key],b['rng'][key]),key)
            exported=torch.load(output/'ejecuciones'/relative/'inference.pt',weights_only=True)
            restored=TinyNormCNN({'model':exported['model_config']});restored.load_state_dict(exported['state_dict'],strict=True)
            audit=output/'ejecuciones'/relative/'lotes_001.json'
            changed=json.loads(audit.read_text());changed['batch_order_sha256']='tampered';ent.json_write(audit,changed)
            with self.assertRaisesRegex(ValueError,'Composición/orden'):
                ent.verificar_lotes_pareados(output,configs,seeds=(42,),folds=(0,),normalization=True)

if __name__=='__main__':unittest.main()
