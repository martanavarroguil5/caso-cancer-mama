import tempfile
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from cancer_mama import cierre_cinetico as close
from cancer_mama import entrenamiento_cinetico as cnn


def frame(n=180,cuts=2):
    rows=[]
    for i in range(n):
        for z in range(cuts):
            rows.append(dict(patient_id=f'P{i:04}',sample_id=f'P{i:04}_z{z:03}',
                pCR=i%2,dataset=('duke','spy1','spy2')[(i//2)%3],split='train',fold=i%5))
    return pd.DataFrame(rows)


class TestCierreCinetico(unittest.TestCase):
    def test_cuatro_roles_sin_solapamiento_y_reproducibles(self):
        samples=frame();a=close.split_patients(samples);b=close.split_patients(samples.sample(frac=1,random_state=3))
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(a.patient_id.nunique(),180)
        self.assertEqual(set(a.role),{'fit','selection','calibration','check'})
        for _,part in a.groupby('role'):
            self.assertEqual(set(part.pCR),{0,1})
            self.assertEqual(set(part.dataset),{'duke','spy1','spy2'})
        self.assertFalse(a.patient_id.duplicated().any())

    def test_no_admite_test_en_particiones(self):
        samples=frame();samples.loc[0,'split']='test'
        with self.assertRaises(ValueError):close.split_patients(samples)

    def test_corte_fijo_no_depende_de_etiquetas_ni_orden(self):
        samples=frame(12,7);selected=set(samples.iloc[close.one_indices(samples)].sample_id)
        samples=samples.sample(frac=1,random_state=4).reset_index(drop=True);samples['pCR']=1-samples.pCR
        self.assertEqual(selected,set(samples.iloc[close.one_indices(samples)].sample_id))
        self.assertEqual(len(selected),12)

    def test_peso_total_por_paciente_independiente_de_numero_de_cortes(self):
        samples=frame(4,3).iloc[1:].reset_index(drop=True)
        samples['weight']=close.patient_weights(samples)
        np.testing.assert_allclose(samples.groupby('patient_id').weight.sum(),1.)

    def test_calibracion_no_invierte_una_red_con_ordenacion_inversa(self):
        samples=frame(4,1);samples['pCR']=[0,0,1,1]
        policy=close.fit_policy(samples,np.array([.9,.8,.2,.1]))
        self.assertTrue(policy['calibration']['boundary_solution'])
        self.assertEqual(policy['calibration']['slope'],0.)

    def test_comprobacion_cuenta_pacientes_no_cortes(self):
        samples=frame(12,2);p=np.where(samples.pCR,.8,.2)
        policy=close.fit_policy(samples,p)
        result,out=close.check_metrics(samples,p,policy)
        self.assertEqual(result['primary_single_slice']['n_patients'],12)
        self.assertEqual(int(out.selected_one_slice.sum()),12)
        self.assertEqual(result['primary_single_slice']['roc_auc'],1.)

    def test_entrena_adam_y_conserva_checkpoint_seguro_sin_optim_en_inferencia(self):
        torch.set_num_threads(2)
        fit,selection=frame(8),frame(4)
        fit_x=torch.randint(0,256,(len(fit),3,256,256),dtype=torch.uint8)
        sel_x=torch.randint(0,256,(len(selection),3,256,256),dtype=torch.uint8)
        cfg=deepcopy(cnn.configuracion());cfg['training'].update(epochs=2,min_epochs=2,batch_size=4,amp=False)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            result=close.train_run(fit,selection,fit_x,sel_x,cfg,'normal',42,path,torch.device('cpu'))
            self.assertEqual(result['optimizer_steps'],8)
            artifact=torch.load(path/'inference.pt',weights_only=True,map_location='cpu')
            self.assertNotIn('optimizer',artifact);self.assertNotIn('rng',artifact)
            self.assertEqual(artifact['model_config'],cfg['model'])
            self.assertEqual(result,close.train_run(fit,selection,fit_x,sel_x,cfg,'normal',42,path,torch.device('cpu')))


if __name__=='__main__':unittest.main()
