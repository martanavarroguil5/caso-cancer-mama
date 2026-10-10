from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from cancer_mama import experimento_finetune as ex


class TinyResidual(ex.ResidualModel):
    def __init__(self,backbone,config):
        nn.Module.__init__(self)
        self.tail=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(256,512))
        self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(512,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.limit=.5
    def embedding(self,x):return self.tail(x)


class TestPartialFineTune(unittest.TestCase):
    def test_initial_prediction_preserves_clinical_and_correction_is_bounded(self):
        from torchvision.models import resnet18
        torch.set_num_threads(2)
        model=ex.ResidualModel(resnet18(weights=None).state_dict(),ex.settings()).eval()
        z=torch.randn(5,512);clinical=torch.linspace(-2,2,5)
        torch.testing.assert_close(model.from_embedding(z,clinical),clinical,rtol=0,atol=0)
        with torch.no_grad():model.head[-1].bias.fill_(100)
        self.assertLessEqual(float((model.from_embedding(z,clinical)-clinical).abs().max()),.500001)
        model.train()
        self.assertTrue(all(not m.training for m in model.tail.modules() if isinstance(m,nn.BatchNorm2d)))

    def test_bag_sampling_does_not_mix_patient_or_orientation(self):
        cache=np.zeros((8,2,1,2,2),dtype=np.float32)
        for i in range(8):
            for view in range(2):cache[i,view]=10*i+view
        groups=[np.array([0,1]),np.array([4,5,6])]
        bags=ex.sample_bags(cache,groups,[0,1],4).numpy()[:,:,0,0,0]
        for bag,allowed in zip(bags,groups):
            self.assertTrue(set((bag//10).astype(int)).issubset(set(allowed)))
            self.assertEqual(len(set(bag%10)),1)

    def test_resume_matches_continuous_training(self):
        torch.set_num_threads(2);rng=np.random.default_rng(42);n=24
        frame=pd.DataFrame({'patient_id':[f'p{i:03}' for i in range(n)],'sample_id':[f'p{i:03}_z000' for i in range(n)],
            'pCR':np.arange(n)%2,'fold':np.arange(n)//8,'split':'train','dataset':'synthetic',
            'age':rng.uniform(30,70,n),'tum_vol':rng.uniform(1,20,n),'HR':np.arange(n)%2,'HER2':np.arange(n)%3==0})
        frame['HER2']=frame.HER2.astype(int)
        cache=rng.normal(size=(n,2,256,2,2)).astype(np.float32)
        config=ex.settings(True);config.update(epochs=2,min_epochs=2,patients_per_batch=4)
        def split(data,fold):return data.iloc[:8].copy(),data.iloc[8:16].copy(),data.iloc[16:].copy()
        with tempfile.TemporaryDirectory() as tmp,patch.object(ex,'ResidualModel',TinyResidual),patch.object(ex.previous,'split_patients',side_effect=split):
            paths=[Path(tmp)/'continuous',Path(tmp)/'resume']
            ex.run_one(frame,cache,{},paths[0],42,0,config,torch.device('cpu'),lambda _:None)
            ex.run_one(frame,cache,{},paths[1],42,0,config,torch.device('cpu'),lambda _:None,until_epoch=1)
            ex.run_one(frame,cache,{},paths[1],42,0,config,torch.device('cpu'),lambda _:None)
            a,b=[torch.load(p/'seed_42/fold_0/last.pt',weights_only=False) for p in paths]
            self.assertEqual(a['history'],b['history']);self.assertEqual(a['tracker'],b['tracker'])
            for key in a['state_dict']:
                torch.testing.assert_close(a['state_dict'][key],b['state_dict'][key],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
