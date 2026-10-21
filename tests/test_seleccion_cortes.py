"""Selección determinista, sin etiquetas, con conservación de pacientes."""
import unittest
import numpy as np
from cancer_mama import seleccion_cortes as sc


class TestSeleccionCortes(unittest.TestCase):
    def test_medoid_avoids_single_extreme_outlier(self):
        x=np.array([[0.],[.1],[.2],[.3],[100.]])
        self.assertEqual(sc.select_indices(x,1),[3])

    def test_three_spaced_and_deterministic(self):
        x=np.arange(10,dtype=float).reshape(-1,1)
        selected=sc.select_indices(x,3)
        self.assertEqual(len(set(selected)),3)
        self.assertEqual(selected,sc.select_indices(x,3))
        self.assertTrue(all(abs(i-j)>=3 for i,j in zip(selected,selected[1:])))

    def test_short_patient_and_all_arm(self):
        self.assertEqual(sc.select_indices(np.zeros((2,3)),3),[0,1])
        self.assertEqual(sc.select_indices(np.zeros((6,3)),10),list(range(6)))
        self.assertIsNone(sc.mean_pair_correlation(np.ones((2,3)),[0]))

    def test_training_identical_apart_from_selection(self):
        configs=[sc.configuration(k,'hash') for k in sc.KS]
        for cfg in configs:
            self.assertEqual(cfg['training'],configs[0]['training'])
            self.assertEqual(cfg['model'],configs[0]['model'])
            self.assertEqual(cfg['augmentation'],configs[0]['augmentation'])
            self.assertEqual(cfg['training']['loss_weight_unit'],'patient')
            self.assertTrue(cfg['slice_selection']['fit_only'])


if __name__=='__main__':unittest.main()
