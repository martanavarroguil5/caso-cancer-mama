from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from cancer_mama import entrenamiento as base
from cancer_mama import entrenamiento_cinetico as cnn
from cancer_mama import inferencia_imagen as runtime
from cancer_mama.app_logic import PHASES,UploadedPhase,uploads_from_paths,validate_phase_uploads
from cancer_mama.app_imagen_logic import SingleSliceEngine,image_key

spec=importlib.util.spec_from_file_location('versionar_cierre',Path(__file__).parents[1]/'modelos/versionar.py')
versionar=importlib.util.module_from_spec(spec);spec.loader.exec_module(versionar)


class TestAppImagen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.path=self.root/'manifiesto.json'
        cfg=cnn.configuracion()['model'];cfg.update(backbone_channels=[2]*4,hidden=2,dropout=0.,spatial_dropout=0.)
        model=cnn.CNNMultiescalaCinetica(cfg)
        with torch.no_grad():
            for parameter in model.parameters():parameter.zero_()
            model.classifier[-1].bias.fill_(.4)
        models=[]
        for seed in (42,2026):
            p=self.root/f'seed_{seed}.pt'
            torch.save({'schema_version':1,'model_config':cfg,'state_dict':model.state_dict()},p)
            models.append({'seed':seed,'fold':0,'path':p.name,'sha256':base.sha256(p),'model_config':cfg})
        prep={'clinical_required':False,'phases':list(PHASES),'shape':[3,256,256],
              'center_crop_sizes':[256,128,64],'encoder_input_size':[128,128],
              'scale_weights_shared':True,'kinetic_maps':['EARLY-PRE','LATE-PRE','EARLY-LATE']}
        self.manifest={'schema_version':2,'evaluation_protocol':'train_only_four_way_single_slice',
            'model_class':'CNNMultiescalaCinetica','inference_unit':'single_slice','slice_aggregation':'single',
            'status':'experimental','models':models,'preprocessing':prep,'threshold':.55,
            'calibration':{'slope':1.,'intercept':0.,'clip_epsilon':base.EPS},'test_evaluated_once':False,
            'training':{'loss':'normal'},'selection':{'metrics_internal_check':{'n_patients':4,'roc_auc':.5},
                                                     'auc_ci95_internal_check':[.1,.9]}}
        self.write()
        paths={}
        for phase,value in zip(PHASES,[50,80,70]):
            p=self.root/f'SYNTHETIC_z000_{phase}.png';Image.fromarray(np.full((256,256),value,np.uint8)).save(p);paths[phase]=p
        self.paths=paths;self.sample=validate_phase_uploads(uploads_from_paths(paths))

    def write(self):
        base.json_write(self.path,self.manifest)
        self.path.with_suffix('.json.sha256').write_text(base.sha256(self.path)+'\n')

    def test_motor_un_corte_coincide_con_cli_y_no_requiere_clinica(self):
        engine=SingleSliceEngine(self.path,'cpu');prediction=engine.predict(self.sample)
        cli=runtime.predecir_cortes(self.path,[[self.paths[p] for p in PHASES]])
        self.assertAlmostEqual(prediction['probability_pcr'],cli['probability_calibrated'],places=7)
        self.assertEqual(prediction['predicted_class'],1)
        self.assertFalse(prediction['clinical_inputs_used'])
        self.assertTrue(all(not m.training for m in engine.models))
        self.assertTrue(all(not p.requires_grad for m in engine.models for p in m.parameters()))
        self.assertEqual(engine.card()['internal_check']['n_patients'],4)
        self.assertEqual(runtime.verificar_modelo_final(self.path)['models'],2)

    def test_rechaza_multiples_cortes_en_paquete_nuevo(self):
        triple=[self.paths[p] for p in PHASES]
        with self.assertRaisesRegex(ValueError,'exactamente un corte'):
            runtime.predecir_cortes(self.path,[triple,triple])

    def test_no_relaja_el_contrato_original_de_diez_modelos(self):
        self.manifest['schema_version']=1;self.write()
        with self.assertRaises(ValueError):runtime.cargar_manifest(self.path)
        with self.assertRaises(ValueError):versionar.check_manifest(self.path)

    def test_no_acepta_semillas_duplicadas_ni_pesos_modificados(self):
        self.manifest['models'][1]['seed']=42;self.write()
        with self.assertRaises(ValueError):runtime.cargar_manifest(self.path)
        self.manifest['models'][1]['seed']=2026;self.write()
        (self.root/self.manifest['models'][0]['path']).write_bytes(b'modificado')
        with self.assertRaisesRegex(ValueError,'Artefacto modificado'):SingleSliceEngine(self.path,'cpu')

    def test_clave_cambia_al_cambiar_una_fase(self):
        key=image_key(self.sample);self.sample.arrays['LATE'][0,0]+=1
        self.assertNotEqual(key,image_key(self.sample))

    def test_archivo_independiente_con_metricas_internas_sin_llamarlas_oof(self):
        source=self.root/'source';package=source/'src/cancer_mama';package.mkdir(parents=True)
        (source/'requirements.txt').write_text('numpy\n')
        for name in ['__init__.py','inferencia_imagen.py','entrenamiento.py']:(package/name).write_text('# fixture\n')
        archive=self.root/'archive'
        with patch.object(versionar,'run_verification',return_value={'models':2,'checksums_verified':True}):
            row=versionar.archive_version(self.path,'v008_cierre_fixture','Cierre de un corte',archive=archive,source_root=source)
        self.assertEqual(row['n_models'],2)
        self.assertEqual(row['metrics_raw_oof'],{})
        self.assertEqual(row['metrics_internal_check']['n_patients'],4)
        self.assertIn('not OOF',row['evaluation_scope'])
        self.assertEqual(versionar.verify_version(row['id'],archive=archive,runtime=False)['models'],2)


if __name__=='__main__':unittest.main()
