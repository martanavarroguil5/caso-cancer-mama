import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pcr_party import datos, metricas
from pcr_party.entrenar import (RECETA, RECETA_LENTA, aumentar, entrenar, muestrear_cortes,
                                objetivos_auxiliares, predecir)
from pcr_party.modelo import PartyNet, numero_parametros


def indice_sintetico(n_pacientes=12, cortes=3):
    filas = []
    for i in range(n_pacientes):
        for z in range(cortes):
            filas.append({"sample_id": f"P{i}_z{z}", "patient_id": f"P{i}", "pCR": i % 2,
                          "fold": i % 3, "cohorte": ("duke", "spy2")[i % 2 == 0 and i < 6],
                          "HR": float(i % 3 == 0), "HER2": np.nan if i == 1 else float(i % 4 == 0),
                          "tum_vol": 5.0 + i})
    return pd.DataFrame(filas)


class TestModelo(unittest.TestCase):
    def test_un_logit_y_auxiliares(self):
        modelo = PartyNet().eval()
        x = torch.rand(2, 3, 256, 256)
        self.assertEqual(tuple(modelo(x).shape), (2, 1))
        logit, aux = modelo.forward_todo(x)
        self.assertEqual(tuple(aux.shape), (2, 3))
        self.assertTrue(torch.allclose(logit, modelo(x)))
        self.assertLess(numero_parametros(modelo), 700_000)

    def test_rechaza_tamano_incorrecto(self):
        with self.assertRaises(ValueError):
            PartyNet()(torch.rand(1, 3, 128, 128))

    def test_mapas_cineticos_firmados(self):
        x = torch.zeros(1, 3, 4, 4)
        x[:, 0], x[:, 1], x[:, 2] = 0.2, 0.7, 0.5
        c = PartyNet.canales(x)
        self.assertTrue(torch.allclose(c[0, 3], torch.full((4, 4), 0.5)))
        self.assertTrue(torch.allclose(c[0, 4], torch.full((4, 4), 0.3)))
        self.assertTrue(torch.allclose(c[0, 5], torch.full((4, 4), 0.2)))


class TestAumentos(unittest.TestCase):
    def test_fases_siguen_alineadas(self):
        g = torch.Generator().manual_seed(0)
        base = torch.rand(8, 1, 256, 256)
        x = aumentar(base.repeat(1, 3, 1, 1), g, RECETA["afin"])
        self.assertTrue(torch.equal(x[:, 0], x[:, 1]))
        self.assertTrue(torch.equal(x[:, 1], x[:, 2]))

    def test_receta_lenta_mantiene_fases_alineadas(self):
        g = torch.Generator().manual_seed(2)
        base = torch.rand(16, 1, 256, 256) + 0.1
        x = aumentar(base.repeat(1, 3, 1, 1), g, RECETA_LENTA["afin"], RECETA_LENTA["cutout"])
        self.assertTrue(torch.equal(x[:, 0], x[:, 2]))
        self.assertTrue((x == 0).any())

    def test_muestreo_k_cortes_por_paciente(self):
        pacientes = np.repeat(["a", "b", "c"], [10, 2, 5])
        filas = np.arange(len(pacientes))
        elegidas = muestrear_cortes(filas, pacientes, 3, np.random.default_rng(0))
        conteo = pd.Series(pacientes[elegidas]).value_counts()
        self.assertEqual(conteo.to_dict(), {"a": 3, "c": 3, "b": 2})
        self.assertEqual(len(set(elegidas)), len(elegidas))
        self.assertTrue(np.array_equal(muestrear_cortes(filas, pacientes, None, None), filas))

    def test_sin_afin_es_permutacion_de_pixeles(self):
        g = torch.Generator().manual_seed(1)
        x = torch.rand(4, 3, 256, 256)
        y = aumentar(x, g, None)
        for i in range(4):
            self.assertTrue(torch.allclose(x[i].flatten().sort().values, y[i].flatten().sort().values))


class TestDatos(unittest.TestCase):
    def test_roles_disjuntos_y_exterior_es_el_fold(self):
        pacientes = datos.tabla_pacientes(indice_sintetico(60))
        roles = datos.roles_fold(pacientes, 1, 0.2)
        self.assertEqual(set(roles.loc[roles.rol == "exterior", "patient_id"]),
                         set(pacientes.loc[pacientes.fold == 1, "patient_id"]))
        self.assertEqual(roles["patient_id"].nunique(), len(roles))
        self.assertGreater((roles.rol == "seleccion").sum(), 0)
        self.assertTrue(roles.equals(datos.roles_fold(pacientes, 1, 0.2)))

    def test_atributo_variable_en_paciente_falla(self):
        indice = indice_sintetico()
        indice.loc[0, "pCR"] = 1 - indice.loc[0, "pCR"]
        with self.assertRaises(ValueError):
            datos.tabla_pacientes(indice)

    def test_leer_png_valida_formato(self):
        with tempfile.TemporaryDirectory() as tmp:
            bueno, malo = Path(tmp) / "a.png", Path(tmp) / "b.png"
            Image.fromarray(np.zeros((256, 256), np.uint8)).save(bueno)
            Image.fromarray(np.zeros((128, 128, 3), np.uint8)).save(malo)
            self.assertEqual(datos.leer_png(bueno).shape, (256, 256))
            with self.assertRaises(ValueError):
                datos.leer_png(malo)

    def test_auxiliares_estandarizados_con_fit(self):
        indice = indice_sintetico()
        fit = {"P0", "P2", "P4", "P6"}
        aux = objetivos_auxiliares(indice, fit)
        lv = aux[indice.patient_id.isin(fit).to_numpy(), 2]
        self.assertAlmostEqual(float(np.mean(lv)), 0.0, places=5)
        self.assertTrue(np.isnan(aux[indice.patient_id.eq("P1").to_numpy(), 1]).all())

    def test_split_desconocido(self):
        with self.assertRaises(ValueError):
            datos.cargar_indice("validation")


class TestMetricas(unittest.TestCase):
    def test_auc_coincide_con_sklearn(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(0)
        y, p = rng.integers(0, 2, 200), rng.random(200)
        p[:20] = 0.5
        self.assertAlmostEqual(metricas.auc(y, p), roc_auc_score(y, p), places=10)

    def test_umbral_y_binarias(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.35, 0.8])
        r = metricas.binarias(y, p, metricas.umbral_youden(y, p))
        self.assertEqual(r["TP"] + r["FN"], 2)
        self.assertGreaterEqual(r["sensibilidad"] + r["especificidad"], 1.5)

    def test_bootstrap_delta_identico_es_cero(self):
        rng = np.random.default_rng(0)
        tabla = pd.DataFrame({"pCR": rng.integers(0, 2, 300), "fold": rng.integers(0, 3, 300),
                              "a": rng.random(300)})
        tabla["b"] = tabla["a"]
        r = metricas.bootstrap_delta(tabla, "a", "b", n=50)
        self.assertEqual(r["delta"], 0.0)
        self.assertEqual(r["ic95"], [0.0, 0.0])


class TestEntrenamiento(unittest.TestCase):
    def test_una_epoca_cpu_finita_y_reproducible(self):
        indice = indice_sintetico(8, 2)
        pixeles = torch.randint(0, 256, (len(indice), 3, 256, 256), dtype=torch.uint8)
        receta = deepcopy(RECETA)
        receta.update(lote=4, epocas_max=2, pacientes_diagnostico_fit=4)
        receta["modelo"] = dict(receta["modelo"], anchos=[4, 4, 4, 4], oculto=8)
        fit, sel = np.arange(0, 12), np.arange(12, 16)
        a = entrenar(receta, pixeles, indice, fit, sel, 3, "ponderada", 1.0, registro=lambda _: None)
        b = entrenar(receta, pixeles, indice, fit, sel, 3, "ponderada", 1.0, registro=lambda _: None)
        for k in a.estado:
            self.assertTrue(torch.equal(a.estado[k], b.estado[k]), k)
        modelo = PartyNet(**receta["modelo"])
        modelo.load_state_dict(a.estado)
        p, aux = predecir(modelo, pixeles, np.arange(len(indice)))
        self.assertTrue(np.isfinite(p).all() and np.isfinite(aux).all())

    def test_receta_lenta_entrena_en_cpu(self):
        indice = indice_sintetico(8, 4)
        pixeles = torch.randint(0, 256, (len(indice), 3, 256, 256), dtype=torch.uint8)
        receta = deepcopy(RECETA_LENTA)
        receta.update(lote=4, epocas_max=2, epocas_min=1, pacientes_diagnostico_fit=4)
        receta["modelo"] = dict(receta["modelo"], anchos=[4, 4, 4, 4], oculto=8)
        r = entrenar(receta, pixeles, indice, np.arange(0, 24), np.arange(24, 32), 5,
                     "normal", 0.0, registro=lambda _: None)
        self.assertEqual(len(r.historia), 2)
        self.assertIn("sel_bce_obj", r.historia[0])

    def test_final_requiere_epocas(self):
        indice = indice_sintetico(4, 1)
        pixeles = torch.zeros(len(indice), 3, 256, 256, dtype=torch.uint8)
        with self.assertRaises(ValueError):
            entrenar(RECETA, pixeles, indice, np.arange(4), None, 0, "normal", 0.0)


if __name__ == "__main__":
    unittest.main()
