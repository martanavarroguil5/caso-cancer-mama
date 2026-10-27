"""Web docente de predicción de pCR: streamlit run app/inicio.py (desde la raíz del repo)."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
for ruta in (RAIZ, RAIZ / "app"):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

import streamlit as st  # noqa: E402

from componentes import RECURSOS, estilo  # noqa: E402

st.set_page_config(page_title="PartyNet · pCR", page_icon=str(RECURSOS / "icono.svg"), layout="wide",
                   initial_sidebar_state="collapsed")
st.logo(str(RECURSOS / "logo.svg"), size="large")
estilo()

PAGINAS = Path(__file__).resolve().parent / "paginas"
navegacion = st.navigation([
    st.Page(PAGINAS / "resumen.py", title="Inicio", default=True),
    st.Page(PAGINAS / "prediccion.py", title="Predecir"),
    st.Page(PAGINAS / "red_imagen.py", title="CNN de imagen"),
    st.Page(PAGINAS / "combinado.py", title="Imagen + clínica"),
    st.Page(PAGINAS / "metodologia.py", title="Metodología"),
], position="top")
navegacion.run()
