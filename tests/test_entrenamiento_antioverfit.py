"""Contratos del piloto anti-sobreajuste."""
import unittest
import json
from pathlib import Path
import tempfile

import pandas as pd
import torch

from cancer_mama import entrenamiento as base
from cancer_mama import entrenamiento_antioverfit as pilot


class TestEntrenamientoAntioverfit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sampler_agrupa_pacientes_y_cubre_cada_corte(self):
        samples = pd.DataFrame({
            "patient_id": ["p1", "p1", "p2", "p2", "p2", "p3"],
            "sample_id": [f"s{i}" for i in range(6)],
        })
        sampler = base.PatientBatchSampler(
            samples, patients_per_batch=2, generator=torch.Generator().manual_seed(7))
        batches = list(sampler)
        self.assertEqual(len(batches), 2)
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(6)))
        for batch in batches:
            self.assertLessEqual(samples.iloc[batch].patient_id.nunique(), 2)
            for patient in samples.iloc[batch].patient_id.unique():
                expected = set(samples.index[samples.patient_id.eq(patient)])
                self.assertTrue(expected.issubset(set(batch)))

    def test_aumentos_son_identicos_entre_fases(self):
        plane = torch.linspace(0, 1, 32*32).reshape(1, 1, 32, 32)
        x = plane.repeat(4, 3, 1, 1)
        augmented = base.aumentar_geometria(
            x, torch.Generator().manual_seed(11), hflip=.5, rot90=True,
            translate=.03, scale_min=.95, scale_max=1.05,
            gain_min=.9, gain_max=1.1, gamma_min=.9, gamma_max=1.1,
            noise_std=.01)
        torch.testing.assert_close(augmented[:, 0], augmented[:, 1])
        torch.testing.assert_close(augmented[:, 1], augmented[:, 2])
        self.assertGreaterEqual(float(augmented.min()), 0)
        self.assertLessEqual(float(augmented.max()), 1)

    def test_modelo_simple_salida_y_gradiente(self):
        config = pilot.configuracion("antioverfit")
        config["model"].update(backbone_channels=[2, 2, 2, 2], hidden=2,
                               dropout=0.0, spatial_dropout=0.0)
        model = pilot.CNNTemporalSimple(config)
        self.assertEqual(model.backbone[0][0].in_channels, 3)
        logits = model(torch.rand(2, 3, 256, 256))
        self.assertEqual(tuple(logits.shape), (2, 1))
        logits.sum().backward()
        self.assertIsNotNone(model.backbone[0][0].weight.grad)

    def test_protocolo_difiere_solo_en_paquete_prefijado(self):
        control = pilot.configuracion("control")
        candidate = pilot.configuracion("antioverfit")
        self.assertEqual(control["model"]["backbone_channels"], [24, 48, 96, 160])
        self.assertEqual(candidate["model"]["backbone_channels"], [24, 48, 96, 160])
        self.assertEqual(control["model"]["spatial_dropout"], 0)
        self.assertEqual(candidate["model"]["spatial_dropout"], .10)
        self.assertIsNone(control["training"]["patients_per_batch"])
        self.assertEqual(candidate["training"]["patients_per_batch"], 6)
        self.assertFalse(control["training"]["equal_patient_weight"])
        self.assertTrue(candidate["training"]["equal_patient_weight"])
        self.assertNotIn("translate", control["augmentation"])
        self.assertEqual(candidate["augmentation"]["translate"], .03)
        status = pilot.verificar()
        self.assertEqual(status["planned_runs"], 16)
        self.assertEqual(status["control_parameters"], status["candidate_parameters"])
        self.assertEqual(status["test_images_loaded"], 0)

    def test_brecha_usa_la_epoca_elegida_incluso_si_es_la_ultima(self):
        for best_epoch, expected in ((1, .10), (2, .20), (3, .90)):
            with self.subTest(best_epoch=best_epoch), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                for seed in (42, 2026):
                    for fold in (0, 3):
                        run = pilot.run_path(output, "control", "normal", seed, fold)
                        run.mkdir(parents=True)
                        (run/"summary.json").write_text(json.dumps({"best_epoch": best_epoch}))
                        pd.DataFrame({"epoch": [1, 2, 3], "auc_gap": [.10, .20, .90]}).to_csv(
                            run/"history.csv", index=False)
                gaps = pilot._gaps(output, "control", "normal")
                self.assertEqual(gaps["selected_gaps"], [expected]*4)
                self.assertEqual(gaps["final_gaps"], [.90]*4)


if __name__ == "__main__":
    unittest.main()
