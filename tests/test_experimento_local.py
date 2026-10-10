import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cancer_mama import experimento_local as ex


class TestLocalFeatures(unittest.TestCase):
    def data(self):
        rng=np.random.default_rng(31);n=80
        x=rng.normal(size=(n,180))
        frame=pd.DataFrame({'patient_id':[f'p{i:03}' for i in range(n)],'pCR':np.arange(n)%2,
            'age':rng.uniform(30,70,n),'tum_vol':rng.uniform(1,20,n),
            'HR':rng.integers(0,2,n),'HER2':rng.integers(0,2,n)})
        x[:,0]=frame.pCR.to_numpy()*3+rng.normal(size=n)*.1
        x[:,1]=0
        return x,frame

    def test_supervised_selection_uses_fit_and_handles_constants(self):
        x,frame=self.data();idx=np.arange(50)
        state=ex.fit_preprocessing(x[idx],frame.iloc[idx],frame.iloc[idx].pCR)
        self.assertEqual(state['indices'][0],0)
        self.assertTrue(np.isfinite(state['scores']).all())
        selected=state['indices'].copy()
        transformed=ex.transform(x[50:]*100,frame.iloc[50:],state,32)
        self.assertEqual(transformed.shape,(30,40))
        np.testing.assert_array_equal(selected,state['indices'])

    def test_outer_changes_cannot_affect_selection(self):
        x,frame=self.data();fi=np.arange(48);si=np.arange(48,64)
        changed=frame.copy();changed.loc[64:,'pCR']=1-changed.loc[64:,'pCR']
        x_changed=x.copy();x_changed[64:]*=1000
        with threadpool_limits(limits=1):
            for family in ex.FAMILIES:
                a,ar=ex.fit_selected(family,x,frame,fi,si,True)
                b,br=ex.fit_selected(family,x_changed,changed,fi,si,True)
                self.assertEqual(a['config'],b['config']);self.assertEqual(ar,br)
                np.testing.assert_array_equal(a['preprocessing']['indices'],b['preprocessing']['indices'])

    def test_bootstrap_identical_predictions_gives_zero_difference(self):
        frame=pd.DataFrame({'patient_id':[f'p{i}' for i in range(20)],'label':np.arange(20)%2,
            'fold':np.arange(20)//4,'cohort':'synthetic','probability':np.linspace(.1,.9,20)})
        for interval in ex.intervals(frame,frame,repetitions=12).values():
            self.assertAlmostEqual(interval['lower'],0)
            self.assertAlmostEqual(interval['upper'],0)


if __name__=='__main__':unittest.main()
