import unittest
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from cancer_mama import experimento_textura as ex


class TestTextures(unittest.TestCase):
    def test_constant_images_are_finite_and_texture_is_reflection_invariant(self):
        torch.set_num_threads(2)
        constant=torch.full((2,3,256,256),128,dtype=torch.uint8)
        features=ex.descriptors(constant)
        self.assertEqual(features.shape,(2,288));self.assertTrue(torch.isfinite(features).all())
        torch.testing.assert_close(features[0],features[1],rtol=0,atol=0)
        generator=torch.Generator().manual_seed(13)
        image=torch.randint(0,256,(2,3,256,256),dtype=torch.uint8,generator=generator)
        torch.testing.assert_close(ex.descriptors(image),ex.descriptors(image.flip(-1)),rtol=1e-5,atol=1e-6)

    def test_outer_labels_do_not_select_features_or_model(self):
        rng=np.random.default_rng(9);n=80
        frame=pd.DataFrame({'patient_id':[f'p{i}' for i in range(n)],'pCR':np.arange(n)%2,
            'age':rng.uniform(30,70,n),'tum_vol':rng.uniform(1,20,n),'HR':rng.integers(0,2,n),'HER2':rng.integers(0,2,n)})
        z=rng.normal(size=(n,160)).astype(np.float32);changed=frame.copy();changed.loc[64:,'pCR']=1-changed.loc[64:,'pCR']
        with threadpool_limits(limits=1):
            for family in ex.FAMILIES:
                a,ar=ex.fit_selected(family,z,frame,np.arange(48),np.arange(48,64),True)
                b,br=ex.fit_selected(family,z,changed,np.arange(48),np.arange(48,64),True)
                self.assertEqual(a['config'],b['config']);self.assertEqual(ar,br)
                np.testing.assert_array_equal(a['preprocessing']['indices'],b['preprocessing']['indices'])


if __name__=='__main__':unittest.main()
