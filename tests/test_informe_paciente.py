import unittest

import numpy as np
import pandas as pd

from cancer_mama.informe_paciente import crossfit_policy, paired_intervals


class TestInformePaciente(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'fold': np.repeat(np.arange(5), 4),
                                   'label': np.tile([0, 1, 0, 1], 5),
                                   'probability': np.tile([.2, .4, .6, .8], 5)})

    def test_el_umbral_no_usa_las_etiquetas_del_fold_destinatario(self):
        scores, decisions, _, policies = crossfit_policy(self.frame)
        changed = self.frame.copy()
        changed.loc[changed.fold.eq(0), 'label'] = 1 - changed.loc[changed.fold.eq(0), 'label']
        new_scores, new_decisions, _, new_policies = crossfit_policy(changed)
        self.assertEqual(policies[0], new_policies[0])
        np.testing.assert_array_equal(scores[:4], new_scores[:4])
        np.testing.assert_array_equal(decisions[:4], new_decisions[:4])

    def test_diferencias_pareadas_de_predicciones_identicas_son_cero(self):
        comparison = self.frame[['fold', 'label']].copy()
        for name in ('new', 'previous', 'clinical'):
            comparison[name] = self.frame.probability
        result = paired_intervals(comparison, 100)
        for reference in result['new_minus_reference'].values():
            for metric in reference.values():
                self.assertEqual(metric['mean'], 0)
                self.assertEqual(metric['ci95'], [0, 0])


if __name__ == '__main__':
    unittest.main()
