import streamlit as st

import estado
from componentes import aviso, destinos, num, pasos, pie, portada, recurso, seccion, tarjetas

img = recurso("modelo_imagen")
fus = recurso("modelo_fusion")
m = img["metricas"]

portada(estado.portada(), "Predicción de la respuesta patológica completa en <em>cáncer de mama</em>",
        "Modelo que estima, a partir de la resonancia magnética con contraste (DCE-MRI) previa a la quimioterapia "
        "neoadyuvante, la probabilidad de <b>respuesta patológica completa (pCR)</b>: ausencia de tumor invasivo "
        "residual en la pieza quirúrgica. La red neuronal convolucional se ha diseñado y entrenado desde cero y "
        "puede combinarse con cuatro variables clínicas.",
        [("prediccion", "Realizar una predicción", True), ("red_imagen", "Arquitectura del modelo", False)])

aviso("<b>Uso exclusivamente educativo.</b> Una probabilidad no es una certeza: una estimación alta no garantiza "
      "pCR ni permite evitar la cirugía, y una baja no la descarta.", "atencion")

# Escala de las barras: AUC entre el azar (0,5) y la perfección (1).
escala = lambda auc: (auc - 0.5) / 0.5
seccion("1", "Resultados principales")
if fus:
    r = fus["resumen"]["brazos"][fus["resumen"]["alias"]["completa+imagen"]]
    fusion = {"t": "Imagen + clínica", "v": num(r["auc_media_folds"]), "acento": True,
              "d": f"AUC media ± {num(r['auc_de_folds'])} · CV anidada", "barra": escala(r["auc_media_folds"]),
              "barra_txt": "0,5 → 1", "ayuda": "Regresión logística sobre la puntuación de la CNN, edad, volumen "
              "tumoral, HR y HER2. Ningún coeficiente vio el fold que evalúa."}
else:
    fusion = {"t": "Imagen + clínica", "v": "…", "d": "Evaluación anidada en marcha",
              "ayuda": "Se está midiendo sin fugas cuánto aporta la imagen sobre los datos clínicos."}
tarjetas([
    {"t": "CNN de imagen", "v": num(m["auc_media_folds"]), "d": f"AUC media ± {num(m['auc_de_folds'])} en 5 folds",
     "barra": escala(m["auc_media_folds"]), "barra_txt": "0,5 → 1",
     "ayuda": "Media de la AUC por paciente de los 5 folds exteriores. Juntando las 1.097 pacientes: "
              f"{num(m['auc_agrupada'])} (IC95 % {num(m['auc_ic95'][0])}–{num(m['auc_ic95'][1])})."},
    fusion,
    {"t": "Pacientes de desarrollo", "v": "1.097", "d": f"{img['n_pcr']} con pCR ({round(100 * m['prevalencia'])} %)",
     "barra": m["prevalencia"], "barra_txt": "con pCR",
     "ayuda": "Train: tres cohortes (I-SPY2, Duke, I-SPY1), ≈10 cortes por paciente. La separación es siempre por paciente."},
    {"t": "Test reservado", "v": "176", "d": "Reservado; aún no evaluado",
     "ayuda": "El test público se evaluará una sola vez con el modelo congelado. No interviene en ninguna decisión."},
])

seccion("2", "Flujo de predicción")
pasos([
    ("Tres fases", "PRE, EARLY y LATE del mismo corte, antes del tratamiento"),
    ("CNN propia", f"{img['parametros']:,} parámetros desde cero; 5 redes en conjunto".replace(",", ".")),
    ("Probabilidad", "Media de los cortes de la paciente, calibrada con la validación cruzada"),
    ("Clínica (opcional)", "Edad, volumen tumoral, HR y HER2 en una regresión logística"),
    ("Banda", "Baja, intermedia o alta, con la pCR que se observó en cada una"),
])

seccion("3", "Interpretación de los resultados")
c1, c2 = st.columns(2)
with c1:
    aviso('<div class="error"><div class="mini"><i></i><i class="on"></i><i></i><i></i></div><div><b>Falso positivo.</b> '
          "El modelo estima pCR alta, pero queda tumor residual. Si se usase para relajar el tratamiento, se "
          "infratrataría a la paciente: por eso nunca debe cambiar la cirugía prevista.</div></div>", "clave")
with c2:
    aviso('<div class="error"><div class="mini"><i></i><i></i><i class="on"></i><i></i></div><div><b>Falso negativo.</b> '
          "Estima pCR baja, pero la paciente sí responde por completo. Podría llevar a intensificar sin necesidad "
          "o a una preocupación injustificada.</div></div>", "info")
aviso("Las <b>bandas</b> reparten a las pacientes de la validación cruzada en tres grupos de igual tamaño según su "
      "probabilidad. Al pasar el ratón por cada banda se ve cuántas tuvieron pCR de verdad: así la predicción se "
      "lee con datos observados y no como un sí o un no.", "clave")

seccion("4", "Contenido")
destinos([
    ("prediccion", "Predicción individual", "Carga de las tres fases o paciente de ejemplo, visor de cortes y bandas de probabilidad."),
    ("red_imagen", "CNN de imagen", "Arquitectura, curvas de entrenamiento y capacidad de discriminación."),
    ("combinado", "Imagen + clínica", "Fusión tardía evaluada con validación cruzada anidada."),
    ("metodologia", "Datos y metodología", "Cohortes, separación por paciente, reglas de evaluación y limitaciones."),
])
pie()
