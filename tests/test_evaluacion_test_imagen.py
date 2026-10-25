"""Comprueba el cálculo estadístico sin abrir el dataset docente."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location('evaluar_test_imagen',Path(__file__).resolve().parents[1]/'scripts/evaluar_test_imagen.py')
evaluation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class TestEvaluacionCongelada(unittest.TestCase):
    def test_discriminacion_y_matriz_con_valores_conocidos(self):
        y=np.array([0,0,1,1]);p=np.array([.1,.4,.35,.8])
        result=evaluation.metrics(y,p)
        self.assertEqual(result['auc'],.75)
        self.assertEqual(result['at_0_5']['tn'],2)
        self.assertEqual(result['at_0_5']['fp'],0)
        self.assertEqual(result['at_0_5']['fn'],1)
        self.assertEqual(result['at_0_5']['tp'],1)
        self.assertEqual(result['at_0_5']['sensitivity'],.5)

    def test_bootstrap_realmente_pareado_para_predicciones_iguales(self):
        y=np.array([0,1,0,1,0,1,0,1])
        p=np.array([.1,.3,.6,.5,.2,.8,.3,.7])
        cohorts=np.array(['a']*4+['b']*4)
        result=evaluation.bootstrap(y,{'uno':p,'dos':p.copy()},cohorts,repetitions=100)
        self.assertEqual(result['paired_differences'][0]['ci95'],[0.,0.])
        self.assertEqual(result['paired_differences'][0]['delta_auc'],0.)
        self.assertEqual(result['auc_ci95']['uno'],result['auc_ci95']['dos'])

    def test_bootstrap_distingue_ordenacion_opuesta_y_es_reproducible(self):
        y=np.array([0,1,0,1,0,1,0,1]);p=np.array([.1,.9,.2,.8,.3,.7,.4,.6])
        cohorts=np.array(['a']*4+['b']*4)
        kwargs=dict(repetitions=50,seed=15)
        first=evaluation.bootstrap(y,{'buena':p,'inversa':1-p},cohorts,**kwargs)
        second=evaluation.bootstrap(y,{'buena':p,'inversa':1-p},cohorts,**kwargs)
        self.assertEqual(first,second)
        self.assertEqual(first['paired_differences'][0]['ci95'],[1.,1.])


if __name__=='__main__':unittest.main()
