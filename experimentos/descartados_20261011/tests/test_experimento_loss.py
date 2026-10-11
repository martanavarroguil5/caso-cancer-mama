"""Separación externa, selectores, comparación OOF y reanudación del ensayo."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from cancer_mama import experimento_loss as exp


def samples_fixture(root, images=False):
    rows = []
    rng = np.random.default_rng(4)
    for fold in range(5):
        for label in (0, 1):
            for repeat in range(2):
                patient = f"p{fold}{label}{repeat}"
                for z in range(2):
                    sample = f"{patient}_z{z:03d}"
                    row = {"sample_id": sample, "patient_id": patient, "slice_index": z,
                           "split": "train", "fold": fold, "pCR": label, "dataset": "cohort_a",
                           "age": 40. + fold + repeat, "tum_vol": 5. + fold,
                           "HR": float(label), "HER2": float(repeat)}
                    for phase in exp.base.pdatos.FASES:
                        path = Path("dataset/train") / patient / f"{sample}_{phase}.png"
                        row["path_" + phase.lower()] = str(path)
                        if images:
                            (root / path).parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(rng.integers(0, 255, (256, 256), dtype=np.uint8)).save(root / path)
                    rows.append(row)
    return pd.DataFrame(rows)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.set_num_threads(2)

    def test_outer_labels_do_not_change_fit_or_selection(self):
        samples = samples_fixture(self.root)
        first = exp.split_patients(samples, 0)
        samples.loc[samples.fold.eq(0), "pCR"] = 1 - samples.loc[samples.fold.eq(0), "pCR"]
        second = exp.split_patients(samples, 0)
        for before, after in zip(first[:2], second[:2]):
            pd.testing.assert_frame_equal(before, after)
        sets = [set(frame.patient_id) for frame in first]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(samples.patient_id))

    def test_patient_logloss_weights_patients_once_and_uses_median(self):
        samples = samples_fixture(self.root)
        frame = samples.iloc[[0, 0, 1, 4]].reset_index(drop=True)
        patients = exp.patient_frame(frame, np.array([.1, .2, .9, .8]))
        self.assertEqual(len(patients), 2)
        np.testing.assert_allclose(patients.probability, [.2, .8])
        value = exp.base.binary_metrics(patients.label, patients.probability)["log_loss"]
        self.assertAlmostEqual(value, -np.log(.8))
        with self.assertRaisesRegex(ValueError, "no finitas"):
            exp.patient_frame(frame, np.array([np.nan, .2, .9, .8]))

    def test_selectors_can_keep_different_epochs_and_freeze_independently(self):
        auc = exp.new_tracker("auc", .7)
        loss = exp.new_tracker("logloss", .6)
        for epoch, (a, b) in enumerate(((.72, .61), (.71, .58), (.70, .57)), 1):
            exp.update_tracker(auc, a, epoch, 1, 2, .001)
            exp.update_tracker(loss, b, epoch, 1, 2, .001)
        self.assertTrue(auc["stopped"])
        self.assertEqual(auc["best_epoch"], 1)
        self.assertFalse(loss["stopped"])
        self.assertEqual(loss["best_epoch"], 3)
        self.assertFalse(exp.update_tracker(auc, .99, 4, 1, 2, .001))
        self.assertEqual(auc["best_epoch"], 1)

    def test_small_improvements_accumulate_for_patience(self):
        tracker = exp.new_tracker("auc", .7)
        exp.update_tracker(tracker, .7006, 1, 1, 3, .001)
        exp.update_tracker(tracker, .7012, 2, 1, 3, .001)
        self.assertEqual(tracker["bad_epochs"], 0)
        self.assertEqual(tracker["best_epoch"], 2)

    def test_baseline_epoch_zero_is_kept_when_training_worsens(self):
        for criterion, baseline, worse in (("auc", .7, .69), ("logloss", .6, .61)):
            tracker = exp.new_tracker(criterion, baseline)
            for epoch in range(1, 4):
                exp.update_tracker(tracker, worse, epoch, 2, 2, .001)
            self.assertEqual(tracker["best_epoch"], 0)
            self.assertEqual(tracker["stop_epoch"], 2)

    def test_resume_matches_continuous_and_outer_is_only_evaluated_at_end(self):
        samples = samples_fixture(self.root, images=True)
        config = exp.settings(workers=0)
        config["model"].update(channels=[2, 2, 2, 2], hidden=4)
        config["training"].update(epochs=3, min_epochs=2, cpu_threads=2, batch_size=8, amp=False)
        a, b = self.root / "continuous", self.root / "resumed"
        original_evaluate = exp.evaluate
        outer_calls = []
        def guarded_evaluate(model, loader, frame, criterion, device):
            if frame.fold.eq(0).all():
                outer_calls.append(1)
                active = b if b.exists() else a
                history = pd.read_csv(active / "lr_actual/seed_42/fold_0/history.csv")
                self.assertEqual(int(history.epoch.max()), 3)
            return original_evaluate(model, loader, frame, criterion, device)
        with patch.object(exp, "plot_history"), patch.object(exp, "evaluate", side_effect=guarded_evaluate):
            exp.run_one(samples, self.root, a, "lr_actual", 42, 0, config, torch.device("cpu"))
            paused = exp.run_one(samples, self.root, b, "lr_actual", 42, 0, config,
                                 torch.device("cpu"), until_epoch=1)
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(len(outer_calls), 2)
            exp.run_one(samples, self.root, b, "lr_actual", 42, 0, config, torch.device("cpu"))
        self.assertEqual(len(outer_calls), 4)
        folders = [p / "lr_actual/seed_42/fold_0" for p in (a, b)]
        histories = [pd.read_csv(p / "history.csv").drop(columns="seconds") for p in folders]
        self.assertEqual(int(histories[0].epoch.min()), 0)
        pd.testing.assert_frame_equal(*histories, check_exact=True)
        fit_ids = set(pd.read_csv(folders[0] / "particiones.csv").query('role == "fit"').patient_id)
        for criterion in exp.CRITERIA:
            artifacts = [torch.load(p / f"best_{criterion}.pt", weights_only=True) for p in folders]
            self.assertEqual(artifacts[0]["model_config"]["clinical"]["preprocessing"]["n_patients"], len(fit_ids))
            for key in artifacts[0]["state_dict"]:
                torch.testing.assert_close(artifacts[0]["state_dict"][key], artifacts[1]["state_dict"][key], rtol=0, atol=0)
            pd.testing.assert_frame_equal(*[pd.read_csv(p / f"oof_{criterion}.csv") for p in folders])
        changed = deepcopy(config)
        changed["training"]["weight_decay"] *= 2
        with self.assertRaisesRegex(ValueError, "Cambió"):
            exp.run_one(samples, self.root, b, "lr_actual", 42, 0, changed, torch.device("cpu"))

    def fake_matrix(self):
        samples = samples_fixture(self.root)
        for arm in exp.ARMS:
            for seed in (42, 2026):
                for fold in range(5):
                    folder = self.root / arm / f"seed_{seed}" / f"fold_{fold}"
                    folder.mkdir(parents=True)
                    config = exp.settings()
                    config["model"]["clinical"].update(preprocessing={}, initialization={})
                    config.update(arm=arm, seed=seed, fold=fold, lr=exp.ARMS[arm],
                                  data_signature=exp.signature(samples), partition_signatures={})
                    exp.base.json_write(folder / "config.json", config)
                    frame = samples[samples.fold.eq(fold)][["sample_id", "patient_id", "pCR", "fold", "dataset"]].copy()
                    frame["probability"] = .25 + .5 * frame.pCR
                    for criterion in exp.CRITERIA:
                        frame.to_csv(folder / f"oof_{criterion}.csv", index=False)
                        (folder / f"best_{criterion}.pt").write_bytes(b"comparison fixture; not used for inference")
                    for name in ("history.csv", "particiones.csv"):
                        (folder / name).write_text("fixture\n")
                    exp.base.json_write(folder / "summary.json", {
                        "status": "complete", "eligible_for_comparison": True,
                        "config_hash": exp.base.config_hash(config),
                        "files": {p.name: exp.base.sha256(p) for p in folder.iterdir() if p.name != "config.json"}})
        return samples

    def test_complete_comparison_identical_candidates_have_zero_deltas(self):
        samples = self.fake_matrix()
        report = exp.compare(samples, self.root, repetitions=5)
        self.assertEqual(len(report["metrics"]), 4)
        self.assertEqual(report["n_runs"], 20)
        for row in report["comparisons"].values():
            self.assertTrue(all(value == 0 for value in row["delta"].values()))
            self.assertFalse(row["promising_internal_result"])
        (self.root / "lr_reducido/seed_2026/fold_4/summary.json").unlink()
        with self.assertRaises(FileNotFoundError):
            exp.compare(samples, self.root, repetitions=5)

    def test_comparison_rejects_changed_oof_and_smoke(self):
        samples = self.fake_matrix()
        folder = self.root / "lr_actual/seed_42/fold_0"
        summary = json.loads((folder / "summary.json").read_text())
        summary["eligible_for_comparison"] = False
        exp.base.json_write(folder / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "20 runs"):
            exp.compare(samples, self.root, repetitions=5)
        summary["eligible_for_comparison"] = True
        exp.base.json_write(folder / "summary.json", summary)
        with (folder / "oof_auc.csv").open("a") as stream:
            stream.write("modified\n")
        with self.assertRaisesRegex(ValueError, "Artefacto cambiado"):
            exp.compare(samples, self.root, repetitions=5)


if __name__ == "__main__":
    unittest.main()
