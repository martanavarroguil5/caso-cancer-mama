import unittest

import numpy as np
import pandas as pd
import torch

from pcr_party import offset
from pcr_party.entrenar import RECETAS, entrenar, logit_pcr, predecir
from pcr_party.modelo import PartyNet


def clinica_sintetica(n=200, semilla=0):
    rng = np.random.default_rng(semilla)
    t = pd.DataFrame({
        "pCR": rng.integers(0, 2, n), "cohorte": "spy2", "edad": rng.normal(50, 10, n),
        "log_volumen": rng.normal(3, 1, n), "HR": rng.integers(0, 2, n).astype(float),
        "HER2": rng.integers(0, 2, n).astype(float), "fold": np.arange(n) % 5,
    }, index=[f"P{i:03d}" for i in range(n)])
    t.loc["P000", "HR"] = np.nan
    return t


class TestOffset(unittest.TestCase):
    def test_grupos_de_subtipo(self):
        t = pd.DataFrame({"HR": [1, 0, 1, 0, np.nan], "HER2": [0, 1, 1, 0, 1]})
        self.assertEqual(offset.grupo_subtipo(t).tolist(), [0, 1, 1, 2, -1])

    def test_offset_sin_fugas(self):
        t = clinica_sintetica()
        base = offset.desplazamientos(t, 2)
        self.assertFalse(base.isna().any())
        # Cambiar la etiqueta de una paciente no cambia su propio offset.
        for pid in ("P002", "P003"):  # exterior y desarrollo
            otra = t.copy()
            otra.loc[pid, "pCR"] = 1 - otra.loc[pid, "pCR"]
            self.assertAlmostEqual(offset.desplazamientos(otra, 2)[pid], base[pid], places=10)
        # Las etiquetas del exterior no afectan a ningún offset.
        otra = t.copy()
        otra.loc[otra["fold"] == 2, "pCR"] = 1 - otra.loc[otra["fold"] == 2, "pCR"]
        pd.testing.assert_series_equal(offset.desplazamientos(otra, 2), base)

    def test_una_cabeza_no_cambia_la_red(self):
        claves = set(PartyNet().state_dict())
        self.assertIn("cabeza_pcr.bias", claves)
        self.assertNotIn("sesgo_pcr", claves)
        self.assertNotIn("cabezas", PartyNet().config)

    def test_varias_cabezas_comparten_sesgo(self):
        m = PartyNet(cabezas=3).eval()
        self.assertIsNone(m.cabeza_pcr.bias)
        self.assertEqual(m(torch.rand(2, 3, 256, 256)).shape, (2, 3))
        salida = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        logit = logit_pcr(salida, torch.tensor([0.5, 0.5, 0.5]), torch.tensor([2, 0, -1]))
        torch.testing.assert_close(logit, torch.tensor([3.5, 1.5, 2.5]))

    def test_entrena_con_offset_en_cpu(self):
        receta = dict(RECETAS["presupuesto"], pacientes_diagnostico_fit=4, lote=4)
        receta["modelo"] = dict(receta["modelo"], cabezas=3)
        indice = pd.DataFrame({"patient_id": np.repeat([f"P{i}" for i in range(8)], 2),
                               "pCR": np.repeat([0, 1] * 4, 2), "HR": 1.0, "HER2": 0.0, "tum_vol": 5.0})
        pixeles = torch.randint(0, 255, (16, 3, 256, 256), dtype=torch.uint8)
        o = np.linspace(-1, 1, 16).astype(np.float32)
        g = np.tile([0, 1, 2, -1], 4)
        r = entrenar(receta, pixeles, indice, np.arange(12), np.arange(12, 16), 0, "normal", 0.0,
                     epocas=1, desplazamiento=o, grupo=g)
        self.assertEqual(float(r.estado["sesgo_pcr"].abs().max()) < 1, True)
        m = PartyNet(**receta["modelo"])
        m.load_state_dict(r.estado)
        p, _ = predecir(m, pixeles, np.arange(16), tta=False, desplazamiento=o, grupo=g)
        self.assertTrue(np.isfinite(p).all())
        with self.assertRaises(ValueError):
            entrenar(receta, pixeles, indice, np.arange(12), None, 0, "ponderada", 0.0, epocas=1,
                     desplazamiento=o, grupo=g)


if __name__ == "__main__":
    unittest.main()
