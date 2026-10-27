import streamlit as st

from componentes import (aviso, cabecera, num, pasos, pie, recurso, seccion, tarjetas)

img = recurso("modelo_imagen")
fus = recurso("modelo_fusion")
m = img["metricas"]

cabecera("Proyecto docente · DCE-MRI de mama",
         "Predicción de respuesta patológica completa antes del tratamiento",
         "Una CNN 2D diseñada y entrenada desde cero estima, a partir de la resonancia previa al "
         "tratamiento neoadyuvante, la probabilidad de alcanzar respuesta patológica completa (pCR). "
         "Opcionalmente se combina con cuatro datos clínicos.")

aviso("<b>Uso exclusivamente educativo.</b> Una probabilidad no es una certeza: una estimación alta no "
      "garantiza pCR ni permite evitar la cirugía, y una baja no la descarta.", "atencion")

seccion("1", "El proyecto en cifras")
if fus:
    r = fus["resumen"]["brazos"][fus["resumen"]["alias"]["completa+imagen"]]
    fusion = {"t": "Imagen + clínica", "v": num(r["auc_media_folds"]),
              "d": f"AUC media ± {num(r['auc_de_folds'])} (CV anidada)",
              "ayuda": "Regresión logística sobre la puntuación de la CNN, edad, volumen tumoral, HR y HER2. "
                       "Ningún coeficiente vio el fold que evalúa."}
else:
    fusion = {"t": "Imagen + clínica", "v": "En curso", "d": "Evaluación anidada en marcha",
              "ayuda": "Se está midiendo sin fugas cuánto aporta la imagen sobre los datos clínicos."}
tarjetas([
    {"t": "CNN de imagen", "v": num(m["auc_media_folds"]), "d": f"AUC media ± {num(m['auc_de_folds'])} en 5 folds",
     "ayuda": "Media de la AUC por paciente de los 5 folds exteriores de la validación cruzada. "
              f"Juntando las 1.097 pacientes: {num(m['auc_agrupada'])} (IC95 % {num(m['auc_ic95'][0])}–{num(m['auc_ic95'][1])})."},
    fusion,
    {"t": "Pacientes de desarrollo", "v": "1.097", "d": f"{img['n_pcr']} con pCR ({round(100 * m['prevalencia'])} %)",
     "ayuda": "Train: tres cohortes (I-SPY2, Duke, I-SPY1), ≈10 cortes por paciente. La separación es siempre por paciente."},
    {"t": "Test reservado", "v": "176", "d": "Sin evaluar todavía",
     "ayuda": "El test público se evaluará una sola vez con el modelo congelado. No interviene en ninguna decisión."},
])

seccion("2", "Cómo funciona", "Del estudio de resonancia a una probabilidad y una banda de interpretación.")
pasos([
    ("Tres fases", "PRE, EARLY y LATE del mismo corte, antes del tratamiento"),
    ("CNN propia", f"{img['parametros']:,} parámetros entrenados desde cero; 5 redes en conjunto".replace(",", ".")),
    ("Probabilidad de imagen", "Media de los cortes de la paciente, calibrada con la validación cruzada"),
    ("Datos clínicos (opcional)", "Edad, volumen tumoral, HR y HER2 se combinan en una regresión logística"),
    ("Banda", "Probabilidad baja, zona intermedia o alta, con la pCR observada en cada banda"),
])

seccion("3", "Cómo leer una predicción")
c1, c2 = st.columns(2)
with c1:
    aviso("<b>Falso positivo:</b> el modelo estima pCR alta pero queda tumor residual. Si se usase para "
          "relajar el tratamiento, se infratrataría a la paciente. Por eso nunca debe cambiar la cirugía prevista.",
          "clave")
with c2:
    aviso("<b>Falso negativo:</b> estima pCR baja pero la paciente sí responde por completo. Podría llevar a "
          "intensificar sin necesidad o a generar una preocupación injustificada.", "clave")
aviso("Las <b>bandas</b> dividen a las pacientes de la validación cruzada en tres grupos de igual tamaño según "
      "su probabilidad. Al pasar el ratón por cada banda se ve cuántas tuvieron pCR de verdad: así la "
      "predicción se interpreta con datos observados y no como un sí o un no.", "info")

seccion("4", "Explorar")
c1, c2 = st.columns(2)
c1.page_link("paginas/prediccion.py", label="Predecir una paciente", icon=":material/monitor_heart:")
c1.page_link("paginas/red_imagen.py", label="CNN de imagen", icon=":material/neurology:")
c2.page_link("paginas/combinado.py", label="Imagen + datos clínicos", icon=":material/hub:")
c2.page_link("paginas/metodologia.py", label="Datos y metodología", icon=":material/menu_book:")
pie()
