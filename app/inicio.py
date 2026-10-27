"""Web docente de predicción de pCR: streamlit run app/inicio.py (desde la raíz del repo)."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
for ruta in (RAIZ, RAIZ / "app"):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

import streamlit as st  # noqa: E402

from componentes import estilo  # noqa: E402

st.set_page_config(page_title="pCR · BreastDCEDL", page_icon=":material/biotech:", layout="wide",
                   initial_sidebar_state="expanded")
estilo()

PAGINAS = Path(__file__).resolve().parent / "paginas"
navegacion = st.navigation({
    "Visión general": [st.Page(PAGINAS / "resumen.py", title="Inicio", icon=":material/home:", default=True)],
    "Predicción": [st.Page(PAGINAS / "prediccion.py", title="Predecir una paciente", icon=":material/monitor_heart:")],
    "Modelos": [st.Page(PAGINAS / "red_imagen.py", title="CNN de imagen", icon=":material/neurology:"),
                st.Page(PAGINAS / "combinado.py", title="Imagen + datos clínicos", icon=":material/hub:")],
    "Proyecto": [st.Page(PAGINAS / "metodologia.py", title="Datos y metodología", icon=":material/menu_book:")],
})
with st.sidebar:
    st.caption("BreastDCEDL · CNN 2D propia desde cero")
    st.caption("Uso educativo. No es un dispositivo médico.")
navegacion.run()
