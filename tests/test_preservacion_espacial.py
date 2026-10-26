import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn

from cancer_mama import preservacion_espacial as c


def samples():
    return pd.DataFrame({'sample_id': ['a1', 'a2', 'a3', 'b1'],
        'patient_id': ['a', 'a', 'a', 'b'], 'pCR': [0, 0, 0, 1],
        'dataset': ['x']*4, 'fold': [0]*4, 'split': ['train']*4})


class TinyCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x.mean((2, 3)))


class TestPreservacionEspacial(unittest.TestCase):
    def test_nativo256_cuadricula16_y_un_logit_por_corte(self):
        torch.set_num_threads(1)
        for profile in c.PROFILES:
            model=c.make_model(c.model_config(profile)).eval()
            x=torch.rand(2,3,256,256)
            with torch.inference_mode():
                self.assertEqual(model.encode(x).shape,(2,96,16,16))
                y=model(x);separate=torch.cat([model(v[None]) for v in x])
            self.assertEqual(y.shape,(2,1))
            torch.testing.assert_close(y,separate,rtol=1e-5,atol=1e-6)
            with self.assertRaises(ValueError):model(torch.rand(1,2,3,256,256))
            with self.assertRaises(ValueError):model(torch.rand(1,3,128,128))

    def test_inicializacion_comun_y_uniformidad_al_inicio(self):
        c.base.seed_everything(42,True);a=c.make_model(c.model_config('uniforme')).eval()
        c.base.seed_everything(42,True);b=c.make_model(c.model_config('atencion')).eval()
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        for key,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[key],rtol=0,atol=0)
        self.assertFalse(a.gate.weight.requires_grad);self.assertTrue(b.gate.weight.requires_grad)
        self.assertEqual(c.image.numero_parametros(b)-c.image.numero_parametros(a),96)
        x=torch.rand(2,3,256,256)
        with torch.inference_mode():torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)

    def test_atencion_acotada_normalizada_y_gradiente(self):
        model=c.make_model(c.model_config('atencion'))
        z=torch.randn(2,96,16,16,requires_grad=True)
        weights=model.spatial_weights(z)
        torch.testing.assert_close(weights,torch.full_like(weights,1/256))
        torch.testing.assert_close(weights.sum(-1),torch.ones(2,1))
        loss=(z.flatten(2)*weights).sum(-1).square().mean();loss.backward()
        self.assertTrue(torch.isfinite(model.gate.weight.grad).all())
        self.assertGreater(model.gate.weight.grad.abs().sum().item(),0)
        with torch.no_grad():model.gate.weight.fill_(100)
        weights=model.spatial_weights(z)
        self.assertGreaterEqual(weights.min().item(),.5/256)
        self.assertLessEqual(weights.max().item(),.5+.5/256+1e-6)
        torch.testing.assert_close(weights.sum(-1),torch.ones(2,1))

    def test_filtro_fijo_y_dimensiones_de_submuestreo(self):
        layer=c.SmoothDownsample(3)
        self.assertEqual(sum(v.numel() for v in layer.parameters()),0)
        torch.testing.assert_close(layer.filter.sum((1,2,3)),torch.ones(3))
        x=torch.ones(2,3,256,256,requires_grad=True)
        y=layer(x);self.assertEqual(y.shape,(2,3,128,128))
        torch.testing.assert_close(y[:,:,1:-1,1:-1],torch.ones_like(y[:,:,1:-1,1:-1]))
        y.square().mean().backward();self.assertTrue(torch.isfinite(x.grad).all())

    def test_temporal_sin_recortes_y_configuracion_estricta(self):
        model=c.make_model(c.model_config()).eval();captured=[]
        hook=model.blocks[0][0].register_forward_pre_hook(lambda m,a:captured.append(a[0].detach().clone()))
        x=torch.rand(1,3,256,256)
        with torch.inference_mode():model(x)
        hook.remove();maps=captured[0]
        self.assertEqual(maps.shape,(1,6,256,256))
        torch.testing.assert_close(maps[:,:3],x)
        torch.testing.assert_close(maps[:,3],x[:,1]-x[:,0])
        torch.testing.assert_close(maps[:,4],x[:,2]-x[:,0])
        cfg=c.model_config();cfg['clinical']=['age']
        with self.assertRaises(ValueError):c.make_model(cfg)

    def test_guardia_familiar_antes_de_cualquier_imagen(self):
        # No permite ejecutar con protocolo ausente; no llega a cargar píxeles.
        with tempfile.TemporaryDirectory() as tmp,patch.object(c.cache,'load_pixels',side_effect=AssertionError('No se deben cargar imágenes')):
            with self.assertRaises(FileNotFoundError):c.execute(Path(tmp))

    def test_bce_y_auc_despues_de_media_por_paciente(self):
        m, patients = c.patient_metrics(samples(), [.1, .2, .6, .8], pos_weight=2.)
        p0 = .9/3
        self.assertEqual(len(patients), 2)
        self.assertAlmostEqual(m['bce'], (-np.log(1-p0)-2*np.log(.8))/2)
        self.assertAlmostEqual(m['auc'], 1.)
        # Cambiar el número de cortes idénticos de una paciente no cambia su peso.
        f = samples().iloc[[0, 0, 0, 3]].copy(); f['sample_id'] = ['x1', 'x2', 'x3', 'b1']
        m, _ = c.patient_metrics(f, [.1, .1, .1, .8])
        self.assertAlmostEqual(m['bce'], (-np.log(.9)-np.log(.8))/2)

    def test_cohorte_aprende_solo_fit_y_no_pesos_por_corte(self):
        fit = samples(); out = pd.DataFrame({'dataset': ['x', 'x'], 'pCR': [1, 1]})
        p, rates = c.cohort_predictions(fit, out)
        np.testing.assert_array_equal(p, [.5, .5])
        self.assertEqual(rates, {'x': .5})
        out.pCR = 0
        np.testing.assert_array_equal(c.cohort_predictions(fit, out)[0], p)
        with self.assertRaises(ValueError):
            c.cohort_predictions(fit, pd.DataFrame({'dataset': ['unknown']}))

    def test_no_forzar_curvas_conjuntas_por_llegar_a_14(self):
        good = [{'epoch': i, 'fit_bce': 1-i*.01, 'selection_bce': 1-i*.005} for i in range(1, 15)]
        self.assertTrue(c.curve_diagnostic(good)['joint_improvement'])
        bad = [dict(r, selection_bce=1+r['epoch']*.005) for r in good]
        self.assertFalse(c.curve_diagnostic(bad)['joint_improvement'])
        self.assertFalse(c.curve_diagnostic(good[:12])['observed_through_epoch_14'])

    def test_guardia_roles_y_etiquetas(self):
        f = pd.concat([samples(), samples().assign(patient_id=['c', 'c', 'c', 'd'],
            sample_id=['c1', 'c2', 'c3', 'd1'], fold=1)], ignore_index=True)
        roles = f[['patient_id', 'dataset', 'pCR', 'fold']].drop_duplicates()
        roles['role'] = ['fit', 'selection', 'outer', 'fit']
        parts = c.roles_for(f, roles)
        self.assertEqual(sum(x.patient_id.nunique() for x in parts.values()), 4)
        with self.assertRaises(ValueError): c.roles_for(f, pd.concat([roles, roles.iloc[:1]]))
        roles.loc[roles.patient_id.eq('a'), 'pCR'] = 1
        with self.assertRaises(ValueError): c.roles_for(f, roles)

    def test_auc_mann_whitney_con_empates(self):
        rng = np.random.default_rng(42)
        y = np.array([0]*40+[1]*20)
        for _ in range(5):
            p = rng.integers(0, 5, len(y))
            self.assertAlmostEqual(c.fast_auc(y, p), roc_auc_score(y, p))

    def test_bootstrap_pareado_identico_tiene_delta_cero(self):
        f = pd.DataFrame({'fold': [0]*4+[1]*4, 'label': [0, 0, 1, 1]*2,
                          'probability_raw': [.1, .4, .3, .7]*2,
                          'cohort_probability': [.1, .4, .3, .7]*2})
        np.testing.assert_array_equal(c.bootstrap(f, n=20)['delta_cohort_ci95'], [0., 0.])

    def test_reanudar_reproduce_pesos_metricas_y_checkpoint_temprano(self):
        torch.set_num_threads(1)
        cfg = c.cnn.configuracion()
        cfg['training'].update(epochs=4, min_epochs=4, patience=8, batch_size=2, amp=False)
        f = samples(); sel = f.assign(sample_id=['s1', 's2', 's3', 's4'], patient_id=['s', 's', 's', 't'])
        pixels = torch.stack([torch.full((3, 256, 256), v, dtype=torch.uint8) for v in [30, 40, 50, 200]])
        original_save = c.base.save_checkpoint
        called = 0

        def interrupt_after_checkpoint(*args, **kwargs):
            nonlocal called
            original_save(*args, **kwargs)
            called += 1
            if called == 2: raise InterruptedError('interrupción sintética tras guardar época2')

        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'make_model', TinyCNN):
            a, b = Path(tmp)/'a'/'runs'/'fold_0'/'normal'/'seed_42', Path(tmp)/'b'/'runs'/'fold_0'/'normal'/'seed_42'
            expected = c.train_run(f, sel, pixels, pixels, cfg, 'normal', 42, a, torch.device('cpu'))
            with patch.object(c.base, 'save_checkpoint', interrupt_after_checkpoint):
                with self.assertRaises(InterruptedError):
                    c.train_run(f, sel, pixels, pixels, cfg, 'normal', 42, b, torch.device('cpu'))
            actual = c.train_run(f, sel, pixels, pixels, cfg, 'normal', 42, b, torch.device('cpu'))
            self.assertEqual(expected['selected'], actual['selected'])
            self.assertEqual(expected['optimizer_steps'], 8)
            self.assertEqual(actual['skipped_steps'], 0)
            for rule in c.RULES:
                s1 = torch.load(a/f'{rule}.pt', weights_only=True)['state_dict']
                s2 = torch.load(b/f'{rule}.pt', weights_only=True)['state_dict']
                for name in s1: torch.testing.assert_close(s1[name], s2[name], rtol=0, atol=0)
            # Los resultados existentes deben verificarse y no sobrescribirse.
            before = (b/'summary.json').read_bytes()
            c.train_run(f, sel, pixels, pixels, cfg, 'normal', 42, b, torch.device('cpu'))
            self.assertEqual(before, (b/'summary.json').read_bytes())


if __name__ == '__main__': unittest.main()
