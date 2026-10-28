"""Las curvas por fold deben conservar su procedencia y no mezclar semillas de otros folds."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pcr_party import recursos_app


class TestCurvasCNN(unittest.TestCase):
    def test_exporta_perdidas_por_fold_sin_cambiar_metricas(self):
        with tempfile.TemporaryDirectory() as carpeta:
            raiz = Path(carpeta)
            runs, recursos = raiz / "runs", raiz / "recursos"
            recursos.mkdir()
            destino = recursos / "modelo_imagen.json"
            original = {"cv": "cv_prueba", "brazo": "ponderada_aux0", "metricas": {"auc": 0.6}}
            destino.write_text(json.dumps(original), encoding="utf-8")
            for fold in range(2):
                for semilla in range(3):
                    ruta = runs / "cv_prueba" / f"fold{fold}" / "ponderada_aux0" / f"semilla{semilla}" / "historia.csv"
                    ruta.parent.mkdir(parents=True)
                    filas = []
                    for epoca in (1, 2):
                        perdida = 0.7 + fold + 0.1 * semilla - 0.01 * (epoca - 1)
                        filas.append({"epoca": epoca, "fit_bce_obj": perdida, "sel_bce_obj": perdida + 0.05,
                                      "bce_pcr_fit_aumentada": perdida + 0.2, "fit_auc": 0.6, "sel_auc": 0.55})
                    pd.DataFrame(filas).to_csv(ruta, index=False)
            with patch.object(recursos_app, "RUNS", runs), patch.object(recursos_app, "RECURSOS", recursos):
                recursos_app.exportar_curvas_cnn()
            exportado = json.loads(destino.read_text(encoding="utf-8"))
            self.assertEqual(exportado["metricas"], original["metricas"])
            self.assertEqual(set(exportado["curvas_por_fold"]), {"0", "1"})
            self.assertAlmostEqual(exportado["curvas_por_fold"]["0"][0]["fit_bce_obj"], 0.8)
            self.assertAlmostEqual(exportado["curvas_por_fold"]["1"][0]["fit_bce_obj"], 1.8)
            self.assertEqual(exportado["curvas_por_fold"]["0"][0]["n"], 3)
            self.assertEqual(exportado["curvas"][0]["n"], 6)
            self.assertAlmostEqual(exportado["curvas"][0]["fit_bce_obj"], 1.3)

    def test_sin_historicos_no_modifica_el_recurso(self):
        with tempfile.TemporaryDirectory() as carpeta:
            recursos = Path(carpeta)
            destino = recursos / "modelo_imagen.json"
            original = json.dumps({"cv": "ausente", "brazo": "ponderada_aux0"})
            destino.write_text(original, encoding="utf-8")
            with patch.object(recursos_app, "RUNS", recursos / "runs"), patch.object(recursos_app, "RECURSOS", recursos):
                with self.assertRaisesRegex(FileNotFoundError, "Faltan históricos"):
                    recursos_app.exportar_curvas_cnn()
            self.assertEqual(destino.read_text(encoding="utf-8"), original)
