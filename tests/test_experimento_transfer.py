import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

from cancer_mama import experimento_transfer as ex


class TestTransfer(unittest.TestCase):
    def test_patient_aggregation_does_not_use_outcomes(self):
        frame = pd.DataFrame({'sample_id':['a0','a1','b0'], 'patient_id':['a','a','b'], 'pCR':[0,0,1]})
        x = np.array([[2.,4.],[4.,6.],[9.,1.]])
        ids, z = ex.patient_features(frame, x)
        self.assertEqual(ids, ['a','b'])
        np.testing.assert_array_equal(z, [[3.,5.],[9.,1.]])
        _, changed = ex.patient_features(frame.assign(pCR=1-frame.pCR), x)
        np.testing.assert_array_equal(z, changed)
        with self.assertRaises(ValueError):
            ex.patient_features(pd.concat([frame, frame.iloc[:1]]), np.vstack([x,x[:1]]))

    def test_preprocessing_is_fit_only(self):
        fit = pd.DataFrame({'age':[30.,50.,np.nan], 'tum_vol':[1.,2.,3.], 'HR':[0.,1.,1.], 'HER2':[0.,1.,0.]})
        state = ex.fit_clinical(fit)
        self.assertEqual(state['medians'][0], 40.)
        means = state['scaler'].mean_.copy()
        transformed = ex.transform_clinical(fit.assign(age=10000.), state)
        self.assertGreater(transformed[0,0], 100.)
        np.testing.assert_array_equal(state['scaler'].mean_, means)
        self.assertEqual(transformed.shape[1], 9)

    def test_views_preserve_phase_order_and_intensity_scale(self):
        x = torch.zeros(2,3,256,256,dtype=torch.uint8)
        x[:,0] = 51
        x[:,1] = 102
        x[:,2] = 204
        model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())
        got = ex.visual_features(x, model, lambda value: value)
        torch.testing.assert_close(got, torch.tensor([[.2,.4,.8,.2,.4,.8]]).repeat(2,1))

    def test_outer_labels_cannot_change_selected_model_or_calibration(self):
        rng = np.random.default_rng(92)
        n = 80
        frame = pd.DataFrame({'patient_id':[f'p{i:03}' for i in range(n)],
            'sample_id':[f'p{i:03}_z000' for i in range(n)], 'pCR':np.arange(n)%2,
            'fold':np.arange(n)//16, 'dataset':'synthetic', 'split':'train',
            'age':rng.uniform(25,75,n), 'tum_vol':rng.uniform(1,20,n),
            'HR':rng.integers(0,2,n), 'HER2':rng.integers(0,2,n)})
        features = rng.normal(size=(n,32)).astype(np.float32)
        def split(samples, fold):
            return samples.iloc[32:].copy(),samples.iloc[16:32].copy(),samples.iloc[:16].copy()
        changed = frame.copy()
        changed.loc[:15,'pCR'] = 1-changed.loc[:15,'pCR']
        with tempfile.TemporaryDirectory() as tmp, patch.object(ex.previous,'split_patients',side_effect=split), threadpool_limits(limits=1):
            paths = [Path(tmp)/'original',Path(tmp)/'changed']
            for path, data in zip(paths,(frame,changed)):
                path.mkdir()
                ex.fit_fold(data,data,features,0,path,smoke=True)
            for family in (*ex.FAMILIES,'clinica'):
                a,b = [joblib.load(p/'fold_0'/f'model_{family}.joblib') for p in paths]
                self.assertEqual(a['chosen']['config'],b['chosen']['config'])
                self.assertEqual(a['calibration'],b['calibration'])
                self.assertEqual(a['threshold'],b['threshold'])
                pa,pb = [pd.read_csv(p/'fold_0'/f'oof_calibrated_{family}.csv').probability for p in paths]
                np.testing.assert_allclose(pa,pb,atol=0,rtol=0)


if __name__ == '__main__':
    unittest.main()
