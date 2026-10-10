import unittest

import numpy as np
import pandas as pd

from cancer_mama.informe_loss import calibrate_inner


class LossReportTests(unittest.TestCase):
    def frames(self):
        inner = pd.DataFrame({"patient_id": [f"inner_{i}" for i in range(20)],
                              "label": [0, 1] * 10,
                              "probability": [.3, .8, .5, .7, .4, .6, .6, .9, .2, .55] * 2})
        outer = pd.DataFrame({"patient_id": ["outer_0", "outer_1", "outer_2"],
                              "label": [0, 1, 0], "probability": [.25, .65, .8], "fold": [0, 0, 0]})
        return inner, outer

    def test_outer_labels_cannot_change_calibration_or_threshold(self):
        inner, outer = self.frames()
        a, ca, ta = calibrate_inner(inner, outer)
        b, cb, tb = calibrate_inner(inner, outer.assign(label=1-outer.label))
        self.assertEqual(ca, cb)
        self.assertEqual(ta, tb)
        np.testing.assert_array_equal(a.probability, b.probability)
        np.testing.assert_array_equal(a.probability_raw, outer.probability)
        self.assertTrue(a.probability.between(0, 1).all())

    def test_overlap_and_duplicate_patients_are_rejected(self):
        inner, outer = self.frames()
        broken = outer.copy()
        broken.loc[0, "patient_id"] = inner.patient_id.iloc[0]
        with self.assertRaisesRegex(ValueError, "Solape"):
            calibrate_inner(inner, broken)
        with self.assertRaisesRegex(ValueError, "duplicadas"):
            calibrate_inner(pd.concat([inner, inner.iloc[:1]]), outer)


if __name__ == "__main__":
    unittest.main()
