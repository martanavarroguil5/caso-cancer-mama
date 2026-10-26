import unittest
import numpy as np
import pandas as pd
from cancer_mama import auditoria_politica as a


class TestAuditoriaPolitica(unittest.TestCase):
    def test_seleccion_rechaza_exterior_y_recuentos_incorrectos(self):
        expected = pd.DataFrame({'patient_id':['a','a','a','b'], 'pCR':[0,0,0,1],
                                 'dataset':['x']*4, 'fold':[0,0,0,1]})
        frame = pd.DataFrame({'patient_id':['a','b'], 'label':[0,1], 'cohort':['x','x'],
                              'fold':[0,1], 'n_slices':[3,1], 'probability_raw':[.2,.8]})
        self.assertEqual(len(a.checked_selection(frame,expected,2)),2)
        with self.assertRaises(ValueError): a.checked_selection(frame,expected,1)
        changed=frame.copy();changed.loc[0,'n_slices']=2
        with self.assertRaises(ValueError): a.checked_selection(changed,expected,2)
        changed=frame.copy();changed.loc[0,'label']=1
        with self.assertRaises(ValueError): a.checked_selection(changed,expected,2)
        with self.assertRaises(ValueError): a.checked_selection(pd.concat([frame,frame.iloc[:1]]),expected,2)

    def test_macro_f1_y_especificidad_muestran_el_intercambio(self):
        f=pd.DataFrame({'label':[0]*6+[1]*6,
                        'probability_raw':[.05,.1,.15,.25,.4,.65,.2,.3,.45,.55,.7,.9]})
        policy=a.build_policies(f);classified=a.classify(f,policy)
        macro=a.decision_metrics(f.label,classified.prediction_macro_f1)
        strict=a.decision_metrics(f.label,classified.prediction_specificity_90)
        self.assertGreater(macro['sensitivity'],strict['sensitivity'])
        self.assertLess(macro['specificity'],strict['specificity'])
        self.assertGreaterEqual(macro['macro_f1'],strict['macro_f1'])
        self.assertEqual(set(policy['thresholds']),set(a.RULES))

    def test_matriz_y_macro_f1_ambas_clases(self):
        m=a.decision_metrics([0,0,1,1],[0,1,0,1])
        self.assertEqual({k:m[k] for k in ('tn','fp','fn','tp')},{'tn':1,'fp':1,'fn':1,'tp':1})
        self.assertAlmostEqual(m['macro_f1'],.5)
        baseline=a.decision_metrics([0,0,1,1],[0]*4)
        self.assertAlmostEqual(baseline['macro_f1'],1/3)
        self.assertEqual(baseline['sensitivity'],0)
        self.assertEqual(baseline['specificity'],1)
        with self.assertRaises(ValueError): a.decision_metrics([0,1],[0])
        with self.assertRaises(ValueError): a.decision_metrics([0,2],[0,1])

    def test_umbral_inverso_reproduce_probabilidad_y_no_invierte_ranking(self):
        calibration={'slope':.8,'intercept':-.7}
        raw=a.inverse_threshold(calibration,.3)
        self.assertAlmostEqual(a.base.apply_calibration(np.array([raw]),calibration)[0],.3,places=14)
        p=np.linspace(.01,.99,100)
        np.testing.assert_array_equal(p>=raw,a.base.apply_calibration(p,calibration)>=.3)
        with self.assertRaises(ValueError): a.inverse_threshold({'slope':0,'intercept':0},.3)

    def test_bootstrap_es_pareado_con_diferencia_nula_si_decisiones_iguales(self):
        f=pd.DataFrame({'fold':[0]*4+[1]*4, 'label':[0,0,1,1]*2,
                        'prediction_macro_f1':[0,1,0,1]*2,
                        'prediction_specificity_90':[0,1,0,1]*2})
        intervals=a.paired_bootstrap(f,n=40)['delta_macro_minus_spec90_ci95']
        for interval in intervals.values():np.testing.assert_array_equal(interval,[0.,0.])


if __name__=='__main__':unittest.main()
