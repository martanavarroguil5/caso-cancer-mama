import io
import unittest

import numpy as np
from PIL import Image

from cancer_mama.app_logic import (
    InputValidationError,
    UploadedPhase,
    enhancement_map,
    validate_phase_uploads,
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


if __name__ == "__main__":
    unittest.main()
