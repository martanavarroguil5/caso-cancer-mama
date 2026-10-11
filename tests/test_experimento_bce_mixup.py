from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from cancer_mama import experimento_bce_mixup as ex


def synthetic_data():
    rng = np.random.default_rng(4)
    n = 24
    return pd.DataFrame({'patient_id': [f'p{i:02}' for i in range(n)],
        'sample_id': [f'p{i:02}_z000' for i in range(n)], 'pCR': np.arange(n)%2,
        'fold': np.arange(n)//8, 'split': 'train', 'dataset': 'synthetic',
        'age': rng.uniform(30, 70, n), 'tum_vol': rng.uniform(1, 20, n),
        'HR': np.arange(n)%2, 'HER2': (np.arange(n)%3 == 0).astype(int)})


class TestBCEMixup(unittest.TestCase):
    def test_mixing_keeps_phase_clinical_and_target_associated(self):
        images = torch.stack([torch.full((3, 4, 4), 2.), torch.full((3, 4, 4), 6.)])
        clinical = torch.tensor([[2., 1.], [6., 0.]])
        labels = torch.tensor([0., 1.])
        with patch.object(np.random, 'beta', return_value=.25), patch.object(torch, 'randperm', return_value=torch.tensor([1, 0])):
            x, c, y = ex.mix_batch(images, clinical, labels, .2)
        torch.testing.assert_close(x[:, 0, 0, 0], torch.tensor([5., 3.]))
        torch.testing.assert_close(x[:, 0], x[:, 1])
        torch.testing.assert_close(x[:, 1], x[:, 2])
        torch.testing.assert_close(c[:, 0], x[:, 0, 0, 0])
        torch.testing.assert_close(c[:, 1], torch.tensor([.25, .75]))
        torch.testing.assert_close(y, torch.tensor([.75, .25]))

    def test_missing_clinical_is_imputed_before_mixing_and_export_matches(self):
        torch.set_num_threads(2)
        data = synthetic_data()
        config = ex.settings()['model']
        prep = ex.base.fit_clinical_preprocessing(data)
        config['clinical'].update(preprocessing=prep, initialization=ex.initializer(data, prep))
        model = ex.MixupCNN(config).eval()
        x = torch.rand(2, 3, 256, 256)
        c = torch.tensor([[float('nan'), 1., 1., 0.], [42., 2., 0., 1.]])
        with torch.inference_mode():
            prepared = model.preprocess_clinical(c)
            self.assertTrue(torch.isfinite(prepared).all())
            self.assertEqual(prepared[0, 4].item(), 1.)
            _, mixed, _ = ex.mix_batch(x, prepared, torch.tensor([0., 1.]), .2)
            self.assertTrue(torch.isfinite(mixed).all())
            torch.testing.assert_close(model.forward_prepared(x, prepared), model(x, c), rtol=0, atol=0)
            restored = ex.base.CNN(config).eval()
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(model(x, c), restored(x, c), rtol=0, atol=0)

    def test_resume_matches_continuous_and_outer_is_not_used_for_selection(self):
        torch.set_num_threads(2)
        data = synthetic_data()
        memory = torch.from_numpy(np.random.default_rng(3).integers(0, 256, (len(data), 3, 256, 256), dtype=np.uint8))
        config = ex.settings(True)
        config['training'].update(epochs=2, min_epochs=2, amp=False)
        config['model'].update(channels=[2, 2, 2, 2], hidden=2)
        def split(*args):
            return data.iloc[:8].copy(), data.iloc[8:16].copy(), data.iloc[16:].copy()
        with tempfile.TemporaryDirectory() as tmp, patch.object(ex.previous, 'split_patients', side_effect=split):
            a, b = Path(tmp)/'continuous', Path(tmp)/'resumed'
            ex.run_one(data, memory, a, 'bce_mixup', 42, 0, config, torch.device('cpu'), lambda _: None)
            with patch.object(ex, 'evaluate', wraps=ex.evaluate) as evaluate:
                ex.run_one(data, memory, b, 'bce_mixup', 42, 0, config, torch.device('cpu'), lambda _: None, until_epoch=1)
                for call in evaluate.call_args_list:
                    self.assertTrue(set(call.args[5].patient_id).isdisjoint(set(data.iloc[16:].patient_id)))
            ex.run_one(data, memory, b, 'bce_mixup', 42, 0, config, torch.device('cpu'), lambda _: None)
            aa, bb = [torch.load(p/'bce_mixup/seed_42/fold_0/last.pt', weights_only=False) for p in (a, b)]
            for result in (aa, bb):
                for row in result['history']:
                    row.pop('seconds')
            self.assertEqual(aa['history'], bb['history'])
            self.assertEqual(aa['tracker'], bb['tracker'])
            for key in aa['state_dict']:
                torch.testing.assert_close(aa['state_dict'][key], bb['state_dict'][key], rtol=0, atol=0)

    def test_seed_average_precedes_patient_median_and_checks_alignment(self):
        frame = pd.DataFrame({'sample_id': ['a', 'b', 'c'], 'patient_id': ['p']*3,
            'pCR': [1]*3, 'fold': [0]*3, 'dataset': ['cohort']*3, 'probability': [.1, .2, .9]})
        other = frame.copy()
        other.probability = [.9, .2, .1]
        result = ex.average_slices([frame, other.iloc[::-1]])
        self.assertAlmostEqual(result.probability.iloc[0], .5)
        other.loc[0, 'pCR'] = 0
        with self.assertRaises(ValueError):
            ex.average_slices([frame, other])


if __name__ == '__main__':
    unittest.main()
