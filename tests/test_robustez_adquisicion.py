import tempfile
from copy import deepcopy
from pathlib import Path
import unittest
import importlib.util

import numpy as np
import pandas as pd
import torch

from cancer_mama import robustez_adquisicion as r

spec=importlib.util.spec_from_file_location('informe_robustez',r.PROJECT_ROOT/'scripts/informar_robustez_adquisicion.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def samples(n=300,cuts=2):
    return pd.DataFrame([dict(patient_id=f'P{i:04}',sample_id=f'P{i:04}_{z}',pCR=(i//5)%2,
        dataset=('duke','spy1','spy2')[(i//10)%3],fold=i%5,split='train') for i in range(n) for z in range(cuts)])


class TestRobustez(unittest.TestCase):
    def test_matriz_y_macro_f1_no_ocultan_clase_pcr(self):
        out=report.classify(np.array([0,0,1,1]),np.array([0,0,0,0]))
        self.assertEqual(out['f1_pcr'],0.)
        self.assertAlmostEqual(out['f1_no_pcr'],2/3)
        self.assertAlmostEqual(out['macro_f1'],1/3)
        self.assertEqual(out['sensitivity'],0.)

    def test_bootstrap_pareado_no_crea_diferencia_en_predicciones_identicas(self):
        f=pd.DataFrame({'label':np.arange(60)%2,'fold':np.where(np.arange(60)<30,0,3),
            'cohort':['duke']*60,'probability_raw':np.where(np.arange(60)%2,.8,.2)})
        out=report.paired_bootstrap({'control':f,'campo_suave':f.copy()},n=20)
        self.assertEqual(out['delta_mean_fold_auc_ci95'],[0.,0.])

    def test_campo_conserva_realce_fases_y_rangos_sin_saturacion(self):
        field=r.fields(3,32,torch.Generator().manual_seed(42),probability=1)
        x=torch.rand(3,3,32,32);transformed=x*field
        self.assertGreaterEqual(float(field.min()),.85)
        self.assertLessEqual(float(field.max()),1.)
        torch.testing.assert_close(transformed[:,1]-transformed[:,0],field[:,0]*(x[:,1]-x[:,0]))
        torch.testing.assert_close(transformed[:,2]-transformed[:,0],field[:,0]*(x[:,2]-x[:,0]))
        self.assertTrue(torch.equal(torch.sign(transformed[:,1]-transformed[:,0]),torch.sign(x[:,1]-x[:,0])))
        self.assertLessEqual(float(transformed.max()),1.)
        self.assertEqual(float((torch.zeros_like(x)*field).max()),0.)

    def test_control_y_reproducibilidad_sin_consumir_rng_global(self):
        torch.manual_seed(31);before=torch.get_rng_state().clone()
        a=r.fields(2,32,torch.Generator().manual_seed(5))
        b=r.fields(2,32,torch.Generator().manual_seed(5))
        torch.testing.assert_close(a,b);self.assertTrue(torch.equal(before,torch.get_rng_state()))
        torch.testing.assert_close(r.fields(2,32,torch.Generator().manual_seed(5),strength=0),torch.ones(2,1,32,32))

    def test_roles_estables_paciente_y_cohorte_sin_test(self):
        frame=samples();a=r.split_roles(frame,0);b=r.split_roles(frame.sample(frac=1,random_state=1),0)
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(len(a),300);self.assertFalse(a.patient_id.duplicated().any())
        self.assertTrue(a.loc[a.role.eq('outer'),'fold'].eq(0).all())
        self.assertFalse(a.loc[a.role.ne('outer'),'fold'].eq(0).any())
        for _,g in a.groupby('role'):
            self.assertEqual(g.pCR.nunique(),2);self.assertEqual(g.dataset.nunique(),3)
        frame.loc[0,'split']='test'
        with self.assertRaises(ValueError):r.split_roles(frame,0)

    def test_agrega_probabilidades_no_logits_y_no_sobrerrepresenta_cortes(self):
        frame=samples(4,2).iloc[1:].reset_index(drop=True)
        p=np.array([.8,.1,.3,.2,.4,.7,.9])
        mean=r.aggregate(frame,p)
        self.assertEqual(len(mean),4);np.testing.assert_allclose(mean.probability_raw,[.8,.2,.3,.8])
        frame.loc[1,'pCR']=1-frame.loc[1,'pCR']
        with self.assertRaises(ValueError):r.aggregate(frame,p)

    def test_politica_constante_no_inventa_sensibilidad_y_respeta_especificidad(self):
        frame=pd.DataFrame({'label':[0,0,0,0,1,1], 'probability_raw':[.4]*6})
        policy=r.fit_policy(frame)
        self.assertEqual(policy['apparent_calibration_metrics']['recall_no_pcr'],1.)
        self.assertEqual(policy['apparent_calibration_metrics']['recall_pcr'],0.)

    def test_entrenamiento_paciente_adamw_y_artefacto_seguro(self):
        torch.set_num_threads(2)
        fit=samples(12,1);fit['pCR']=np.arange(12)%2
        selection=fit.copy();selection.patient_id='S'+selection.patient_id
        x=torch.randint(0,256,(12,3,256,256),dtype=torch.uint8)
        cfg=deepcopy(r.config());cfg['training'].update(epochs=1,min_epochs=1,batch_size=4,amp=False)
        with tempfile.TemporaryDirectory() as tmp:
            out=r.train_run(fit,selection,x,x,cfg,'campo_suave','weighted',42,0,Path(tmp),torch.device('cpu'))
            self.assertEqual(out['optimizer_steps'],3);self.assertEqual(out['skipped'],0)
            artifact=torch.load(Path(tmp)/'inference.pt',map_location='cpu',weights_only=True)
            self.assertEqual(set(artifact),{'schema_version','model_config','state_dict'})
            self.assertEqual(artifact['model_config'],cfg['model'])


if __name__=='__main__':unittest.main()
