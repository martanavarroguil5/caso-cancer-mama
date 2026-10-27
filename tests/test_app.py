"""Ejecuta cada página de la web sin navegador (requiere app/recursos y los paquetes)."""
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
APP = RAIZ / "app"
LISTO = (APP / "recursos" / "modelo_imagen.json").exists() and (RAIZ / "modelos" / "party_final_v2").exists()


@unittest.skipUnless(LISTO, "faltan recursos de la web o el paquete de imagen")
class TestPaginas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for ruta in (str(RAIZ), str(APP)):
            if ruta not in sys.path:
                sys.path.insert(0, ruta)

    def ejecutar(self, pagina):
        from streamlit.testing.v1 import AppTest
        # El inicio solo funciona dentro de la navegación (enlaces a otras páginas).
        ruta = APP / "inicio.py" if pagina == "resumen.py" else APP / "paginas" / pagina
        at = AppTest.from_file(str(ruta), default_timeout=120)
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    def test_paginas_cargan(self):
        for pagina in ("resumen.py", "red_imagen.py", "combinado.py", "metodologia.py", "prediccion.py"):
            with self.subTest(pagina=pagina):
                self.ejecutar(pagina)

    def test_prediccion_con_paciente_de_ejemplo(self):
        at = self.ejecutar("prediccion.py")
        at.segmented_control[0].set_value("Paciente de ejemplo").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        textos = " ".join(m.value for m in at.markdown)
        self.assertIn("probabilidad estimada de pCR", textos)
        self.assertIn("Submodelo usado", textos)


if __name__ == "__main__":
    unittest.main()
