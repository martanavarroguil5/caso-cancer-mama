from copy import deepcopy
import unittest

from cancer_mama import confirma_bce_normal as ex


class TestConfirmationGate(unittest.TestCase):
    def test_requires_loss_interval_auc_noninferiority_brier_and_both_seeds(self):
        interval = {'log_loss': {'upper': -.001}, 'mean_fold_auc': {'lower': -.009}}
        delta = {'brier': -.001}
        seeds = [{'log_loss': -.01, 'mean_fold_auc': 0.}, {'log_loss': -.01, 'mean_fold_auc': 0.}]
        self.assertTrue(ex.decision(interval, delta, seeds))
        for key, metric, value in [('log_loss', 'upper', 0.), ('mean_fold_auc', 'lower', -.0101)]:
            bad = deepcopy(interval)
            bad[key][metric] = value
            self.assertFalse(ex.decision(bad, delta, seeds))
        self.assertFalse(ex.decision(interval, {'brier': .001}, seeds))
        self.assertFalse(ex.decision(interval, delta, seeds[:1]))
        bad = deepcopy(seeds)
        bad[1]['log_loss'] = .001
        self.assertFalse(ex.decision(interval, delta, bad))
        bad = deepcopy(seeds)
        bad[0]['mean_fold_auc'] = -.0101
        self.assertFalse(ex.decision(interval, delta, bad))


if __name__ == '__main__':
    unittest.main()
