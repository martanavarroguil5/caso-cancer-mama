"""Integridad y contrato de las versiones archivadas de CNN solo imagen."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from cancer_mama import entrenamiento as base
from cancer_mama import entrenamiento_imagen as imagen
from cancer_mama import inferencia_imagen as runtime


class TestInferenciaImagen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = imagen.configuracion()['model']
        config.update(stem_channels=[2, 2], fusion_channels=2,
                      trunk_channels=[2, 2], hidden=2, dropout=0.0)
        model = imagen.CNNImagenFases(config)
        self.weight = self.root/'peso.pt'
        torch.save({'schema_version': 1, 'model_config': config,
                    'state_dict': model.state_dict()}, self.weight)
        models = [{'seed': seed, 'fold': fold, 'path': self.weight.name,
                   'sha256': base.sha256(self.weight), 'model_config': deepcopy(config)}
                  for seed in (42, 2026) for fold in range(5)]
        self.manifest = self.root/'manifiesto.json'
        self.value = {'model_class': 'CNNImagenFases', 'models': models,
                      'status': 'resultado_insuficiente_conservado',
                      'slice_aggregation': 'mean', 'threshold': 1.0,
                      'calibration': {'slope': 0.0, 'intercept': -1.0,
                                      'clip_epsilon': base.EPS},
                      'preprocessing': {'clinical_required': False,
                                        'phases': list(runtime.PHASES),
                                        'shape': [3, 256, 256]},
                      'test_evaluated_once': False}
        self.write()

    def write(self):
        base.json_write(self.manifest, self.value)
        self.manifest.with_suffix('.json.sha256').write_text(base.sha256(self.manifest)+'\n')

    def triples(self):
        result = []
        for index, value in enumerate((51, 77, 230)):
            triple = []
            for phase in runtime.PHASES:
                path = self.root/f'paciente_z{index:03d}_{phase}.png'
                Image.fromarray(np.full((256, 256), value, dtype=np.uint8)).save(path)
                triple.append(path)
            result.append(triple)
        return result

    def test_carga_estricta_con_pesos_y_calibracion_constante(self):
        result = runtime.verificar_modelo_final(self.manifest)
        self.assertEqual(result['models'], 10)
        self.assertTrue(result['calibration_is_constant'])
        self.assertFalse(result['test_evaluated_once'])

    def test_detecta_pesos_corruptos_y_no_admite_manifest_multimodal(self):
        self.weight.write_bytes(b'corrupto')
        with self.assertRaisesRegex(ValueError, 'Artefacto modificado'):
            runtime.cargar_manifest(self.manifest)
        self.value['model_class'] = 'CNN'
        self.write()
        with self.assertRaisesRegex(ValueError, 'Contrato incompatible'):
            runtime.cargar_manifest(self.manifest)

    def test_media_de_cortes_y_constante_se_conservan_sin_clinica(self):
        class Example(torch.nn.Module):
            def forward(self, x):
                return x[:, 0].mean(dim=(1, 2)).unsqueeze(1)
        triples = self.triples()
        with patch.object(runtime, 'cargar_modelo', return_value=Example()):
            result = runtime.predecir_cortes(self.manifest, triples)
        expected = float(torch.sigmoid(torch.tensor([51., 77., 230.])/255).mean())
        self.assertAlmostEqual(result['probability_raw'], expected, places=6)
        self.assertAlmostEqual(result['probability_calibrated'], 1/(1+np.exp(1)), places=6)
        self.assertEqual(result['pCR'], 0)
        self.assertEqual(result['n_slices'], 3)
        self.assertFalse(result['clinical_used'])
        with self.assertRaisesRegex(ValueError, 'Corte duplicado'):
            runtime.predecir_cortes(self.manifest, [triples[0], triples[0]])


if __name__ == '__main__':
    unittest.main()
