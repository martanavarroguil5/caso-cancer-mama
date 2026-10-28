import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pcr_party import datos, fusion, metricas
try:
    from test_party import indice_sintetico
except ImportError:
    from tests.test_party import indice_sintetico


def tabla_sintetica(n=400, semilla=0):
    rng = np.random.default_rng(semilla)
    t = pd.DataFrame({
        "HR": rng.integers(0, 2, n).astype(float), "HER2": rng.integers(0, 2, n).astype(float),
        "edad": rng.normal(50, 10, n), "log_volumen": rng.normal(2.5, 1, n), "imagen": rng.normal(0, 1, n),
        "fold": rng.integers(0, 5, n)}, index=[f"P{i}" for i in range(n)])
    z = -1 + 1.2 * t.HER2 - 0.8 * t.HR + 0.4 * t.imagen
    t["pCR"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return t


class TestIntegridadFusion(unittest.TestCase):
    def test_manifiesto_portable_sin_aceptar_cambios_de_contenido(self):
        original = '{\n  "nombre": "imagen"\n}\n'.encode("utf-8")
        with tempfile.TemporaryDirectory() as carpeta:
            raiz = Path(carpeta)
            imagen, combinado = raiz / "imagen", raiz / "fusion"
            imagen.mkdir()
            combinado.mkdir()
            (combinado / "paquete.json").write_text(json.dumps({
                "paquete_imagen": "imagen",
                "sha256_paquete_imagen": hashlib.sha256(original).hexdigest()
            }), encoding="utf-8")
            for contenido in (original, original.replace(b"\n", b"\r\n")):
                with self.subTest(contenido=contenido), patch("pcr_party.inferencia.Paquete") as cargar:
                    (imagen / "paquete.json").write_bytes(contenido)
                    fusion.PaqueteFusion(combinado, "cpu")
                    cargar.assert_called_once_with(imagen, "cpu")
            (imagen / "paquete.json").write_bytes(original.replace(b"imagen", b"otro"))
            with patch("pcr_party.inferencia.Paquete") as cargar:
                with self.assertRaisesRegex(RuntimeError, "no coincide"):
                    fusion.PaqueteFusion(combinado, "cpu")
                cargar.assert_not_called()


class TestFusion(unittest.TestCase):
    def test_diagnostico_ajuste_mide_brecha_logistica(self):
        ajuste = tabla_sintetica(250, 2)
        validacion = tabla_sintetica(100, 3)
        modelos = fusion.ajustar_submodelos(ajuste, False)
        d = fusion.diagnostico_ajuste(modelos, ajuste, validacion, 2)
        p, _ = fusion.predecir_patron(modelos, validacion, tuple(fusion.GRUPOS), True)
        self.assertEqual(d["ajuste"]["n"], 250)
        self.assertEqual(d["validacion"]["n"], 100)
        self.assertAlmostEqual(d["validacion"]["auc"], metricas.auc(validacion["pCR"], p))
        self.assertAlmostEqual(d["brecha_auc"], d["ajuste"]["auc"] - d["validacion"]["auc"])
        self.assertAlmostEqual(d["brecha_bce"], d["validacion"]["bce"] - d["ajuste"]["bce"])

    def test_ocho_combinaciones_y_nombres(self):
        c = fusion.combinaciones()
        self.assertEqual(len(c), 8)
        self.assertEqual(c[0], ())
        self.assertEqual(fusion.nombre_submodelo((), True), "imagen")
        self.assertEqual(fusion.nombre_submodelo(("subtipo", "volumen"), True), "subtipo+volumen+imagen")

    def test_subtipo_exige_hr_y_her2(self):
        self.assertEqual(fusion.grupos_disponibles({"HR": 1.0, "HER2": None, "edad": 40}), ("edad",))
        self.assertEqual(fusion.grupos_disponibles({"HR": 0, "HER2": 1, "log_volumen": np.nan}), ("subtipo",))

    def test_logistica_recupera_signos_y_estandariza_con_ajuste(self):
        t = tabla_sintetica()
        params = fusion.ajustar_logistica(t, ["HR", "HER2", "imagen"], ponderada=False)
        self.assertGreater(params["coeficientes"]["HER2"], 0)
        self.assertLess(params["coeficientes"]["HR"], 0)
        self.assertAlmostEqual(params["media"][2], t["imagen"].mean(), places=10)
        p = fusion.aplicar_logistica(params, t)
        self.assertTrue(((p > 0) & (p < 1)).all())

    def test_patron_usa_lo_disponible_sin_imputar(self):
        t = tabla_sintetica()
        sub = fusion.ajustar_submodelos(t, ponderada=False)
        self.assertEqual(len(sub), 15)
        prueba = t.iloc[:3].copy()
        prueba.loc[prueba.index[0], "edad"] = np.nan
        prueba.loc[prueba.index[1], ["HR", "HER2", "edad", "log_volumen"]] = np.nan
        p, usados = fusion.predecir_patron(sub, prueba, ("subtipo", "edad", "volumen"), True)
        self.assertEqual(usados, ["subtipo+volumen+imagen", "imagen", "subtipo+edad+volumen+imagen"])
        self.assertTrue(np.isfinite(p).all())
        p, usados = fusion.predecir_patron(sub, prueba, ("subtipo",), False)
        self.assertTrue(np.isnan(p[1]) and usados[1] == "")

    def test_roles_internos_excluyen_el_exterior(self):
        pacientes = datos.tabla_pacientes(indice_sintetico(60))
        roles = fusion.roles_internos(pacientes, 0, 1, 0.2)
        self.assertEqual(set(roles.loc[roles.rol == "fuera", "fold"]), {0})
        self.assertEqual(set(roles.loc[roles.rol == "exterior", "fold"]), {1})
        self.assertFalse(roles.loc[roles.rol.isin(["fit", "seleccion"]), "fold"].isin([0, 1]).any())
        self.assertGreater((roles.rol == "seleccion").sum(), 0)
        with self.assertRaises(ValueError):
            fusion.roles_internos(pacientes, 2, 2, 0.2)

    def test_interaccion_solo_cambia_con_imagen(self):
        t = tabla_sintetica()
        params = fusion.ajustar_logistica(t, ["HR", "HER2", "edad", "log_volumen", "imagen"], False, True)
        self.assertIn("imagen×HER2", params["coeficientes"])
        self.assertEqual(len(fusion.aplicar_logistica(params, t)), len(t))


if __name__ == "__main__":
    unittest.main()
