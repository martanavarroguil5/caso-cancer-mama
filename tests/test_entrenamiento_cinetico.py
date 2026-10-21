"""Contratos del candidato cinético solo imagen."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from cancer_mama import entrenamiento_cinetico as cinetico
from cancer_mama import entrenamiento_imagen as imagen


class TestEntrenamientoCinetico(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_mapas_cineticos_explicitos_conservan_signo(self):
        x = torch.tensor([[[[.2]], [[.7]], [[.5]]]])
        maps = cinetico.CNNMultiescalaCinetica.kinetic_maps(x)
        expected = torch.tensor([[[[.5]], [[.3]], [[.2]]]])
        torch.testing.assert_close(maps, expected)
        reverse = cinetico.CNNMultiescalaCinetica.kinetic_maps(x[:, [0, 2, 1]])
        self.assertLess(float(reverse[0, 2, 0, 0]), 0)
        temporal = cinetico.CNNMultiescalaCinetica.temporal_maps(x)
        self.assertEqual(tuple(temporal.shape), (1, 6, 1, 1))
        torch.testing.assert_close(temporal[:, :3], x)
        torch.testing.assert_close(temporal[:, 3:], expected)

    def test_fusion_temprana_y_salida_binaria(self):
        config = cinetico.configuracion()
        config["model"].update(backbone_channels=[2, 2, 2, 2],
                               hidden=2, dropout=0.0)
        model = cinetico.CNNMultiescalaCinetica(config)
        self.assertEqual(model.backbone[0][0].in_channels, 6)
        self.assertIsNone(config["model"]["clinical"])
        logits = model(torch.rand(2, 3, 256, 256))
        self.assertEqual(tuple(logits.shape), (2, 1))
        logits.sum().backward()
        self.assertIsNotNone(model.backbone[0][0].weight.grad)

    def test_vistas_multiescala_preservan_contexto_centro_y_nucleo(self):
        config = cinetico.configuracion()
        config["model"].update(backbone_channels=[2, 2, 2, 2],
                               hidden=2, dropout=0.0)
        model = cinetico.CNNMultiescalaCinetica(config)
        x = torch.arange(256*256, dtype=torch.float32).reshape(1, 1, 256, 256)
        x = x.repeat(1, 3, 1, 1)/(256*256-1)
        views = model.multiscale_views(x)
        self.assertEqual(tuple(views.shape), (1, 3, 3, 128, 128))
        torch.testing.assert_close(views[:, 1], x[:, :, 64:192, 64:192])
        center = x[:, :, 96:160, 96:160]
        expected_core = torch.nn.functional.interpolate(
            center, size=(128, 128), mode="bilinear", align_corners=False)
        torch.testing.assert_close(views[:, 2], expected_core)

    def test_protocolo_solo_cambia_la_arquitectura(self):
        control = imagen.configuracion()
        candidate = cinetico.configuracion()
        for section in ("training", "augmentation", "evaluation"):
            self.assertEqual(candidate[section], control[section])
        self.assertEqual(candidate["model"]["input_phases"], ["PRE", "EARLY", "LATE"])
        self.assertEqual(candidate["model"]["derived_maps"],
                         ["EARLY-PRE", "LATE-PRE", "EARLY-LATE"])
        self.assertEqual(candidate["model"]["center_crop_sizes"], [256, 128, 64])
        self.assertEqual(candidate["model"]["encoder_input_size"], 128)
        self.assertEqual(candidate["model"]["backbone_channels"], [24, 48, 96, 160])
        self.assertEqual(candidate["model"]["spatial_dropout"], .10)
        self.assertIsNone(candidate["model"]["clinical"])
        status = cinetico.verificar()
        self.assertEqual(status["planned_runs"], 20)
        self.assertEqual(status["test_images_loaded"], 0)

    def test_comparacion_pareada_exige_mismos_pacientes(self):
        destination = self.root/"comparacion"
        destination.mkdir()
        patient_id = [f"p{i:02d}" for i in range(20)]
        label = np.array([0]*10+[1]*10)
        candidate = pd.DataFrame({
            "patient_id": patient_id,
            "label": label,
            "probability_raw": np.r_[np.linspace(.05, .25, 10),
                                      np.linspace(.75, .95, 10)],
        })
        control = candidate.copy()
        control["probability_raw"] = 1-candidate["probability_raw"]
        candidate.to_csv(destination/"oof_pacientes_weighted.csv", index=False)
        control_path = self.root/"control.csv"
        control.to_csv(control_path, index=False)
        historical = control.rename(columns={"probability_raw": "probability"})
        historical_path = self.root/"historical.csv"
        historical.to_csv(historical_path, index=False)
        report = {"selected_loss_for_future_test": "weighted"}
        with (patch.object(cinetico, "CONTROL_OOF", control_path),
              patch.object(cinetico, "HISTORICAL_OOF", historical_path)):
            comparison = cinetico.comparar_con_control(self.root, report, n_boot=200)
        self.assertTrue(comparison["adoption_supported"])
        self.assertTrue(comparison["target_reached"])
        self.assertEqual(comparison["target_auc"], .60)
        self.assertEqual(comparison["test_images_loaded"], 0)
        saved = json.loads((destination/"comparacion_control.json").read_text())
        self.assertGreater(saved["delta_auc_ci95"][0], 0)
        self.assertGreater(saved["delta_auc_historical_ci95"][0], 0)


if __name__ == "__main__":
    unittest.main()
