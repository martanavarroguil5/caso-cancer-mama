"""Contratos que protegen la evaluación por paciente y los píxeles exclusivos."""
import unittest
from unittest.mock import patch
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cancer_mama import imagen_mil as mil


class TestImagenMIL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_masked_slices_cannot_change_prediction(self):
        model=mil.PatientNet(mil.configuration('hybrid'),16).eval()
        x=torch.rand(2,3,3,128,128)
        mask=torch.tensor([[True,True,False],[True,False,False]])
        features=torch.rand(2,16)
        with torch.no_grad():
            expected=model(x,mask,features)
            x[~mask]=torch.rand_like(x[~mask])*100
            actual=model(x,mask,features)
        torch.testing.assert_close(actual,expected)

    def test_patient_pooling_is_invariant_to_slice_order(self):
        model=mil.PatientNet(mil.configuration('cnn'),16).eval()
        x=torch.rand(2,3,3,128,128)
        mask=torch.tensor([[True,True,False],[True,True,True]])
        feat=torch.rand(2,16)
        with torch.no_grad():
            a=model(x,mask,feat)
            b=model(x[:,[2,0,1]],mask[:,[2,0,1]],feat)
        torch.testing.assert_close(a,b)

    def test_splits_are_disjoint_and_cover_patients(self):
        frame=pd.DataFrame([dict(patient_id=f'{fold}-{cohort}-{y}-{i}',fold=fold,
                                 dataset=cohort,pCR=y)
            for fold in range(5) for cohort in ['a','b'] for y in [0,1] for i in range(8)])
        fit,selection,outer=mil.split_indices(frame,2)
        self.assertFalse(set(fit)&set(selection))
        self.assertFalse(set(fit)&set(outer))
        self.assertFalse(set(selection)&set(outer))
        self.assertEqual(set(np.concatenate([fit,selection,outer])),set(range(len(frame))))
        self.assertTrue(frame.iloc[outer].fold.eq(2).all())

    def test_joint_augmentation_preserves_equal_phases_and_slices(self):
        x=torch.rand(3,1,1,128,128).repeat(1,4,3,1,1)
        augmented=mil.augment(x)
        torch.testing.assert_close(augmented[:,:,0],augmented[:,:,2])
        torch.testing.assert_close(augmented[:,0],augmented[:,3])

    def test_descriptors_constant_images_are_finite_and_phase_sensitive(self):
        x=torch.zeros(2,3,128,128)
        f=mil.pixel_descriptors(x)
        self.assertTrue(torch.isfinite(f).all())
        x[:,1]=.5
        changed=mil.pixel_descriptors(x)
        self.assertEqual(f.shape,changed.shape)
        self.assertGreater(float((f-changed).abs().sum()),0)

    def test_linear_branch_is_regularizable_and_trains(self):
        model=mil.PatientNet(mil.configuration('hybrid',feature_head='linear'),16)
        x=torch.rand(2,3,3,128,128)
        out=model(x,torch.ones(2,3,dtype=torch.bool),torch.rand(2,16))
        out.sum().backward()
        self.assertTrue(torch.isfinite(model.feature_net.weight.grad).all())
        self.assertIsNotNone(model.encoder[0].net[0].weight.grad)

    def test_texture_on_blank_images_is_finite(self):
        from cancer_mama.imagen_textura import texture_descriptors
        f=texture_descriptors(torch.zeros(2,3,128,128))
        self.assertEqual(tuple(f.shape),(2,144))
        self.assertTrue(torch.isfinite(f).all())

    def test_inference_descriptors_match_cached_definition(self):
        from cancer_mama.inferencia_mil import features_from_images
        from cancer_mama.imagen_textura import texture_descriptors
        torch.manual_seed(43)
        images=torch.rand(5,3,128,128)
        def reference(z):
            mean=z.mean(0)
            return torch.cat([mean,((z-mean).square().mean(0)+1e-8).sqrt()])
        cfg=mil.configuration('hybrid',feature_file='features_texture_v3.npy')
        roi=mil.enhancement_crop(images,prior_strength=6.)
        expected=torch.cat([reference(mil.pixel_descriptors(images)),
             reference(torch.cat([mil.pixel_descriptors(roi),texture_descriptors(images),texture_descriptors(roi)],1))])
        actual=features_from_images(images,cfg)
        torch.testing.assert_close(actual,expected)
        permuted=features_from_images(images[[2,0,4,3,1]],cfg)
        torch.testing.assert_close(actual,permuted)

    def test_axial_aggregation_receives_gradients(self):
        model=mil.PatientNet(mil.configuration('cnn',slice_pool='sequence'),16)
        x=torch.rand(2,5,3,128,128)
        model(x,torch.ones(2,5,dtype=torch.bool),torch.rand(2,16)).sum().backward()
        self.assertIsNotNone(model.axial_conv1.weight.grad)
        self.assertGreater(float(model.axial_conv1.weight.grad.abs().sum()),0)

    def test_native_two_view_cnn_has_valid_shape_and_mask(self):
        config=mil.configuration('cnn',widths=[4,8],view='multires_native',spatial_pool='grid4')
        model=mil.PatientNet(config,16).eval()
        x=torch.rand(2,3,3,256,256);mask=torch.tensor([[True,True,False],[True,True,True]])
        with torch.no_grad():
            expected=model(x,mask,torch.zeros(2,16))
            x[0,2]=50
            actual=model(x,mask,torch.zeros(2,16))
        self.assertEqual(tuple(actual.shape),(2,))
        torch.testing.assert_close(actual,expected)

    def test_inference_averages_calibrated_fold_groups(self):
        from cancer_mama import inferencia_mil as infer
        from cancer_mama.entrenamiento import sha256
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            models=[]
            for i in range(3):
                file=folder/f'{i}.pt';file.write_text(str(i))
                models.append({'file':file.name,'sha256':sha256(file)})
            calibration={'mean':0.,'std':1.,'coef':1.,'intercept':0.}
            (folder/'manifest.json').write_text(json.dumps({'models':models,'groups':[
                {'model_indices':[0,1],'calibration':calibration},
                {'model_indices':[2],'calibration':calibration}]}))
            with patch.object(infer,'load_pngs',return_value=torch.zeros(5,3,128,128)), \
                 patch.object(infer,'predict_checkpoint',side_effect=[.2,.4,.9]):
                result=infer.predict_package(folder,[],[],[])
            self.assertAlmostEqual(result['probability_pcr'],.6)


if __name__=='__main__':
    unittest.main()
