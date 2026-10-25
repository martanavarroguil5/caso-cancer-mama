import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter,shift

from cancer_mama import revision_train as r


def row():
    return SimpleNamespace(patient_id='P1',sample_id='P1_z002',dataset='spy1',pCR=1,fold=0,slice_index=2,split='train',
        path_pre='dataset/train/P1/P1_z002_PRE.png',path_early='dataset/train/P1/P1_z002_EARLY.png',path_late='dataset/train/P1/P1_z002_LATE.png')


def metadata(root,n=300):
    rows=[];patients=[]
    for i in range(n):
        pid=f'P{i:04}';label=(i//5)%2;cohort=('duke','spy1','spy2')[(i//10)%3]
        rows.append(dict(patient_id=pid,sample_id=pid+'_z001',pCR=label,fold=i%5,slice_index=1,split='train',
            path_pre=f'dataset/train/{pid}/{pid}_z001_PRE.png',path_early=f'dataset/train/{pid}/{pid}_z001_EARLY.png',path_late=f'dataset/train/{pid}/{pid}_z001_LATE.png'))
        patients.append(dict(pid=pid,pCR=label,split='train',dataset=cohort))
    (root/'metadata').mkdir();pd.DataFrame(rows).to_csv(root/'metadata/samples.csv',index=False);pd.DataFrame(patients).to_csv(root/'metadata/patients.csv',index=False)


class TestRevisionTrain(unittest.TestCase):
    def test_paths_cerrados_a_paciente_fase_y_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);record=row()
            self.assertEqual(r.checked_path(root,record,'path_pre','PRE'),root/record.path_pre)
            for invalid in ['dataset/test/P1/P1_z002_PRE.png','../otro/P1_z002_PRE.png','dataset/train/P1/P1_z002_EARLY.png']:
                record.path_pre=invalid
                with self.assertRaises(ValueError):r.checked_path(root,record,'path_pre','PRE')

    def test_lectura_valida_y_hashes_sin_conversion_silenciosa(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);record=row();(root/'dataset/train/P1').mkdir(parents=True)
            for value,name in zip([30,60,90],r.PATHS):Image.fromarray(np.full((256,256),value,dtype=np.uint8)).save(root/getattr(record,name))
            stats,files,hist=r.inspect_triplet(root,record)
            self.assertEqual(stats['error'],'');self.assertEqual(len(files),3)
            self.assertAlmostEqual(stats['early_minus_pre'],30/255,places=5)
            self.assertEqual([int(h.sum()) for h in hist],[65536]*3)
            self.assertEqual(len(stats['triplet_sha256']),64)
            Image.fromarray(np.zeros((256,256,3),dtype=np.uint8)).save(root/record.path_early)
            invalid,_,_=r.inspect_triplet(root,record)
            self.assertIn('PNG L',invalid['error'])

    def test_traslacion_conocida_y_signo(self):
        rng=np.random.default_rng(42);a=gaussian_filter(rng.random((256,256)),2)
        b=shift(a,(8,-12),order=1,mode='constant')
        d=r.translation_diagnostic(a,b)
        self.assertTrue(d['estimable'])
        self.assertAlmostEqual(d['dy'],-8,delta=4);self.assertAlmostEqual(d['dx'],12,delta=4)
        self.assertGreater(d['ncc_best'],d['ncc_zero'])

    def test_intensidad_afin_no_simula_movimiento_y_plano_no_estima(self):
        a=gaussian_filter(np.random.default_rng(2).random((256,256)),2)
        d=r.translation_diagnostic(a,a*1.4+.2)
        self.assertEqual((d['dy'],d['dx']),(0,0));self.assertFalse(d['flag'])
        self.assertFalse(r.translation_diagnostic(np.ones((256,256)),np.ones((256,256)))['estimable'])

    def test_metadatos_detectan_etiqueta_incoherente(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);metadata(root)
            samples,patients=r.train_metadata(root);self.assertEqual(len(samples),300)
            p=pd.read_csv(root/'metadata/patients.csv');p.loc[0,'pCR']=1-p.loc[0,'pCR'];p.to_csv(root/'metadata/patients.csv',index=False)
            with self.assertRaises(ValueError):r.train_metadata(root)

    def test_cinco_roles_oof_y_final_sin_entrenar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);metadata(root)
            output=root/'out';protocol=r.prepare_protocol(root,output)
            self.assertFalse(protocol['training_started']);self.assertEqual(protocol['reference_cv_runs'],20)
            validation_ids=[]
            for fold in range(5):
                roles=pd.read_csv(output/'protocolo_cinco_folds'/f'roles_fold_{fold}.csv')
                self.assertEqual(len(roles),300);self.assertFalse(roles.patient_id.duplicated().any())
                self.assertEqual(set(roles.role),{'fit','selection','outer'})
                self.assertTrue(roles.loc[roles.role.eq('outer'),'fold'].eq(fold).all())
                validation_ids.extend(roles.loc[roles.role.eq('outer'),'patient_id'])
            self.assertEqual(len(set(validation_ids)),300);self.assertEqual(len(validation_ids),300)
            final=pd.read_csv(output/'protocolo_cinco_folds/roles_modelo_final.csv')
            self.assertEqual(final.role.eq('fit').sum(),255)
            self.assertEqual(final.role.eq('selection_and_policy').sum(),45)
            with self.assertRaises(ValueError):r.prepare_protocol(root,output)


if __name__=='__main__':unittest.main()
