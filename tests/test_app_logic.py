import io
import unittest

import numpy as np
from PIL import Image
import torch

from cancer_mama.app_logic import (
    InputValidationError,
    InferenceEngine,
    UploadedPhase,
    enhancement_map,
    validate_phase_uploads,
    validate_clinical,
    prediction_key,
)


def png_bytes(value=80, size=(256, 256), mode="L"):
    image = Image.new(mode, size, value if mode == "L" else (value, value, value))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


class AppLogicTests(unittest.TestCase):
    def valid_uploads(self):
        return {
            phase: UploadedPhase(f"PATIENT-01_z004_{phase}.png", png_bytes(40 + index * 30))
            for index, phase in enumerate(("PRE", "EARLY", "LATE"))
        }

    def test_validates_aligned_three_phase_sample(self):
        sample = validate_phase_uploads(self.valid_uploads())
        self.assertEqual(sample.patient_id, "PATIENT-01")
        self.assertEqual(sample.sample_id, "PATIENT-01_z004")
        self.assertEqual(sample.arrays["PRE"].shape, (256, 256))
        self.assertEqual(sample.arrays["PRE"].dtype, np.uint8)

    def test_rejects_missing_phase(self):
        uploads = self.valid_uploads()
        uploads["LATE"] = None
        with self.assertRaisesRegex(InputValidationError, "Faltan fases: LATE"):
            validate_phase_uploads(uploads)

    def test_rejects_phase_from_another_slice(self):
        uploads = self.valid_uploads()
        uploads["LATE"] = UploadedPhase("PATIENT-01_z005_LATE.png", png_bytes())
        with self.assertRaisesRegex(InputValidationError, "mismo corte"):
            validate_phase_uploads(uploads)

    def test_rejects_wrong_geometry_and_rgb(self):
        uploads = self.valid_uploads()
        uploads["PRE"] = UploadedPhase("PATIENT-01_z004_PRE.png", png_bytes(size=(128, 256)))
        with self.assertRaisesRegex(InputValidationError, "256×256"):
            validate_phase_uploads(uploads)
        uploads = self.valid_uploads()
        uploads["PRE"] = UploadedPhase("PATIENT-01_z004_PRE.png", png_bytes(mode="RGB"))
        with self.assertRaisesRegex(InputValidationError, "monocromo"):
            validate_phase_uploads(uploads)

    def test_rejects_corrupt_payload(self):
        uploads = self.valid_uploads()
        uploads["EARLY"] = UploadedPhase("PATIENT-01_z004_EARLY.png", b"not-a-png")
        with self.assertRaisesRegex(InputValidationError, "corrupto"):
            validate_phase_uploads(uploads)

    def test_enhancement_map_is_rgb_and_signed(self):
        sample = validate_phase_uploads(self.valid_uploads())
        preview = enhancement_map(sample)
        self.assertEqual(preview.shape, (256, 256, 3))
        self.assertEqual(preview.dtype, np.uint8)
        self.assertGreater(float(preview[..., 0].mean()), float(preview[..., 1].mean()))

    def test_clinical_values_and_missing_indicators_input(self):
        values = {"age": 49, "tum_vol": 12, "HR": 1, "HER2": None}
        result = validate_clinical(values)
        np.testing.assert_allclose(result[0, :3], [49, np.log1p(12), 1], rtol=1e-6)
        self.assertTrue(np.isnan(result[0, 3]))
        for change in ({"age": -1, "tum_vol": None}, {"HR": 2}, {"tum_vol": -1},
                       {"age": float("inf")}, {"age": "incorrecto"}):
            with self.subTest(change=change), self.assertRaises(InputValidationError):
                validate_clinical(values | change)

    def test_changed_images_or_clinical_values_invalidate_saved_prediction(self):
        sample = validate_phase_uploads(self.valid_uploads())
        values = {"age": 49, "tum_vol": 12, "HR": 1, "HER2": 0}
        key = prediction_key(sample, values)
        self.assertNotEqual(key, prediction_key(sample, values | {"HR": 0}))
        uploads = self.valid_uploads()
        uploads["PRE"] = UploadedPhase(uploads["PRE"].name, png_bytes(41))
        self.assertNotEqual(key, prediction_key(validate_phase_uploads(uploads), values))

    def test_single_slice_averages_models_before_patient_aggregation(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.device = torch.device("cpu")
        engine.manifest = {"aggregation": "median", "threshold": .5,
                           "calibration": {"slope": 1., "intercept": 0.}}
        class FixedModel:
            def __init__(self, p):
                self.p = p
            def __call__(self, image, clinical):
                self.clinical = clinical.clone()
                return torch.logit(torch.tensor([[self.p]]))
        engine.models = [FixedModel(.1), FixedModel(.2), FixedModel(.9)]
        sample = validate_phase_uploads(self.valid_uploads())
        result = engine.predict(sample, {"age": 49, "tum_vol": 12, "HR": 1, "HER2": 0})
        self.assertAlmostEqual(result.probability_raw, .4, places=6)
        self.assertAlmostEqual(result.probability_calibrated, .4, places=6)
        torch.testing.assert_close(engine.models[0].clinical, torch.tensor([[49., np.log1p(12), 1., 0.]], dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
