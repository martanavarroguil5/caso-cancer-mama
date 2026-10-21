"""El ajuste mantiene la red y no puede usar el fold exterior al seleccionar."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from cancer_mama import ajuste_cinetico as ajuste
from cancer_mama import entrenamiento_imagen as imagen
from test_entrenamiento_imagen import metadata_sintetica


class TestAjusteCinetico(unittest.TestCase):
    def test_arquitectura_fija_en_todos_los_candidatos(self):
        plan = ajuste.protocol()
        self.assertEqual(plan["parameters"], 593521)
        self.assertEqual(plan["planned_runs"], 60)
        for trial in ajuste.TRIALS:
            config = ajuste.training_config(trial)
            self.assertEqual(config["model"], plan["architecture"])
            self.assertEqual(config["augmentation"], imagen.configuracion()["augmentation"])

    def test_seleccion_usa_solo_probabilidades_internas_y_ambas_semillas(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for trial in ajuste.TRIALS:
                for loss in imagen.LOSSES:
                    for seed in imagen.SEEDS:
                        path = ajuste.run_path(output, trial, loss, seed, 0)
                        path.mkdir(parents=True)
                        (path/"inference.pt").write_bytes(b"fixture")
                        # lr_menor normal es perfecto; los demás invierten el orden.
                        probability = [.1, .2, .8, .9]
                        if (trial, loss) != ("lr_menor", "normal"):
                            probability = probability[::-1]
                        pd.DataFrame({"patient_id": list("abcd"), "label": [0,0,1,1],
                                      "fold": [1,2,3,4], "cohort": "spy1", "n_slices": 2,
                                      "probability_raw": probability}).to_csv(
                                          path/"selection_patients.csv", index=False)
            with patch.object(ajuste, "check_run", return_value={"finite_weights": True}):
                choice = ajuste.choose_inner(output, 0)
                self.assertEqual(choice["selected"]["trial"], "lr_menor")
                self.assertEqual(choice["selected"]["loss"], "normal")
                self.assertEqual(choice["selected"]["inner_auc"], 1.0)
                self.assertEqual(len(choice["selected"]["models"]), 2)
                # Inyectar pacientes exteriores en el fichero interno debe fallar.
                for seed in imagen.SEEDS:
                    path = ajuste.run_path(output, "referencia", "normal", seed, 0)
                    frame = pd.read_csv(path/"selection_patients.csv")
                    frame.loc[0, "fold"] = 0
                    frame.to_csv(path/"selection_patients.csv", index=False)
                with self.assertRaisesRegex(ValueError, "exterior"):
                    ajuste.choose_inner(output, 0)

    def test_outer_no_se_evalua_sin_elecciones_congeladas(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "congeladas"):
                ajuste.evaluate_choice(Path(tmp), Path(tmp), {}, 0, torch.device("cpu"))

    def test_entrenamiento_interno_no_carga_outer_y_reanuda_sin_epoca_extra(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = metadata_sintetica(root)
            for _, row in rows[rows.split.eq("train")].iterrows():
                for phase in ("pre", "early", "late"):
                    path = root/row[f"path_{phase}"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.full((256,256), 60+int(row.pCR)*80,
                                            dtype=np.uint8)).save(path)
            config = imagen.configuracion()
            config["model"].update(stem_channels=[2,2], fusion_channels=2,
                                    trunk_channels=[2,2], hidden=2, dropout=0.)
            config["training"].update(epochs=4, min_epochs=1, patience=1,
                                      batch_size=4, num_workers=0, cpu_threads=2, amp=False)
            original_loader = imagen._loader

            def protected_loader(samples, *args, **kwargs):
                if 0 in set(samples.fold):
                    raise AssertionError("Se intentó cargar el fold exterior")
                return original_loader(samples, *args, **kwargs)

            output = root/"runs"
            with patch.object(imagen, "_loader", side_effect=protected_loader), \
                    patch.object(imagen, "_patient_auc", return_value=.5):
                summary = imagen.run_training(config, "normal", 42, 0, root,
                                               output, torch.device("cpu"), evaluate_outer=False)
            path = output/imagen.EXPERIMENT_NAME/"normal/seed_42/fold_0"
            self.assertEqual(summary["epochs_completed"], 2)
            self.assertFalse(summary["outer_evaluated"])
            self.assertFalse(summary["eligible_for_selection"])
            self.assertFalse((path/"outer_patients.csv").exists())
            # Simula caída tras last.pt y antes de terminar los artefactos.
            (path/"summary.json").unlink()
            with patch.object(imagen, "_loader", side_effect=protected_loader), \
                    patch.object(imagen, "train_epoch", side_effect=AssertionError("Época extra")):
                repeated = imagen.run_training(config, "normal", 42, 0, root,
                                                output, torch.device("cpu"), evaluate_outer=False)
            self.assertEqual(repeated["epochs_completed"], 2)
            saved = json.loads((path/"config.json").read_text())
            self.assertTrue(saved["selection_only"])


if __name__ == "__main__":
    unittest.main()
