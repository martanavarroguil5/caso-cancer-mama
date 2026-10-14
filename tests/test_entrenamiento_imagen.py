"""Contratos del candidato principal solo imagen."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from cancer_mama import entrenamiento as base
from cancer_mama import entrenamiento_imagen as imagen


def metadata_sintetica(root: Path) -> pd.DataFrame:
    (root/"metadata").mkdir(parents=True)
    rows = []
    for fold in range(5):
        for label in (0, 1):
            patient = f"p{fold}{label}"
            for slice_index in (0, 1):
                sample = f"{patient}_z{slice_index:03d}"
                row = {"sample_id": sample, "patient_id": patient, "fold": fold,
                       "pCR": label, "split": "train", "dataset": "spy1",
                       "slice_index": slice_index}
                for phase in ("PRE", "EARLY", "LATE"):
                    row[f"path_{phase.lower()}"] = str(
                        Path("dataset/train")/patient/f"{sample}_{phase}.png")
                rows.append(row)
    test = rows[0].copy()
    test.update(sample_id="test_z000", patient_id="test", split="test", fold=-1,
                pCR="NO_LEER")
    rows.append(test)
    frame = pd.DataFrame(rows)
    frame.to_csv(root/"metadata/samples.csv", index=False)
    return frame


class TestEntrenamientoImagen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_encoder_monocanal_compartido_y_fusion_ordenada(self):
        config = imagen.configuracion()
        config["model"].update(stem_channels=[2, 2], fusion_channels=2,
                               trunk_channels=[2, 2], hidden=2, dropout=0.0)
        model = imagen.CNNImagenFases(config)
        self.assertEqual(model.phase_encoder[0][0].in_channels, 1)
        self.assertIsNone(config["model"]["clinical"])
        encoded = torch.zeros(1, 3, 1, 2, 2)
        encoded[:, 0], encoded[:, 1], encoded[:, 2] = 1, 3, 2
        expected = torch.tensor([1., 3., 2., 2., 1., -1.])
        torch.testing.assert_close(model.fusion_views(encoded)[0, :, 0, 0], expected)
        logits = model(torch.rand(2, 3, 256, 256))
        self.assertEqual(tuple(logits.shape), (2, 1))
        logits.sum().backward()
        self.assertGreater(float(model.phase_encoder[0][0].weight.grad.abs().sum()), 0)

    def test_particion_interna_no_toca_outer_y_es_fija_entre_semillas(self):
        metadata_sintetica(self.root)
        samples = base.cargar_train(self.root)
        first = imagen.particion_anidada(samples, 2)
        second = imagen.particion_anidada(samples, 2)
        fit, selection, outer = first
        self.assertEqual(set(outer.fold), {2})
        self.assertNotIn(2, set(fit.fold)|set(selection.fold))
        patient_sets = [set(frame.patient_id) for frame in first]
        self.assertFalse(patient_sets[0]&patient_sets[1])
        self.assertFalse(patient_sets[0]&patient_sets[2])
        self.assertFalse(patient_sets[1]&patient_sets[2])
        self.assertEqual(set(selection.patient_id), set(second[1].patient_id))
        self.assertEqual(set(fit.patient_id)|set(selection.patient_id)|set(outer.patient_id),
                         set(samples.patient_id))

    def test_umbral_conservador_y_metricas_de_ambas_clases(self):
        y = np.array([0]*10+[1]*4)
        probability = np.array([.1,.2,.3,.4,.5,.6,.7,.8,.85,.9,.65,.75,.88,.95])
        threshold = imagen.threshold_min_specificity(y, probability, .90)
        self.assertAlmostEqual(threshold, .88)
        metrics = imagen.metricas_binarias(y, probability, threshold)
        self.assertGreaterEqual(metrics["recall_no_pcr"], .90)
        self.assertEqual(metrics["recall_pcr"], .5)
        self.assertIn("f1_no_pcr", metrics)
        self.assertIn("f1_pcr", metrics)
        self.assertIn("macro_f1", metrics)
        baseline = imagen.metricas_binarias(y, np.zeros(len(y)), .5)
        self.assertEqual(baseline["f1_pcr"], 0)
        self.assertEqual(baseline["balanced_accuracy"], .5)

    def test_comparador_exige_veinte_runs_y_mantiene_test_cerrado(self):
        metadata_sintetica(self.root)
        train = base.cargar_train(self.root)
        signature = hashlib.sha256(train.to_csv(index=False).encode()).hexdigest()
        output = self.root/"runs"
        for loss_name in imagen.LOSSES:
            for seed in imagen.SEEDS:
                for fold in imagen.FOLDS:
                    fit, selection, outer = imagen.particion_anidada(train, fold)
                    run_dir = output/imagen.EXPERIMENT_NAME/loss_name/f"seed_{seed}"/f"fold_{fold}"
                    run_dir.mkdir(parents=True)
                    config = imagen.configuracion()
                    config.update(loss=loss_name, seed=seed, outer_fold=fold, smoke=False,
                                  device_type="cpu", data_signature=signature,
                                  source_sha256={"fixture": "test"}, counts={}, pos_weight=1.0)
                    base.json_write(run_dir/"config.json", config)
                    for name, frame in (("selection", selection), ("outer", outer)):
                        patients = frame.drop_duplicates("patient_id").sort_values("patient_id")
                        positive = .9 if loss_name == "normal" else .1
                        negative = .1 if loss_name == "normal" else .9
                        result = pd.DataFrame({
                            "patient_id": patients.patient_id,
                            "label": patients.pCR.astype(int),
                            "fold": patients.fold.astype(int),
                            "cohort": patients.dataset,
                            "probability_raw": np.where(patients.pCR.eq(1), positive, negative),
                            "n_slices": 2,
                        })
                        result.to_csv(run_dir/f"{name}_patients.csv", index=False)
                    torch.save({"schema_version": 1, "state_dict": {},
                                "model_config": config["model"]}, run_dir/"inference.pt")
                    pd.DataFrame([{"epoch": 1, "selection_loss": .5,
                                   "selection_patient_auc": 1.0 if loss_name == "normal" else 0.0}]
                                 ).to_csv(run_dir/"history.csv", index=False)
                    summary = {
                        "status": "complete", "eligible_for_selection": True,
                        "config_hash": base.config_hash(config),
                        "inference_sha256": base.sha256(run_dir/"inference.pt"),
                        "outer_sha256": base.sha256(run_dir/"outer_patients.csv"),
                        "selection_sha256": base.sha256(run_dir/"selection_patients.csv"),
                    }
                    base.json_write(run_dir/"summary.json", summary)
        report = imagen.comparar_ejecuciones(self.root, output)
        self.assertEqual(report["selected_loss_for_future_test"], "normal")
        self.assertFalse(report["clinical_variables_used"])
        self.assertEqual(report["patient_aggregation"], "mean")
        self.assertEqual(report["test_images_loaded"], 0)
        manifest = json.loads((output/"comparacion/modelo_desarrollo_imagen.json").read_text())
        self.assertFalse(manifest["preprocessing"]["clinical_required"])
        self.assertEqual(manifest["slice_aggregation"], "mean")


if __name__ == "__main__":
    unittest.main()
