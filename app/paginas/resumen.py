import streamlit as st

import estado
from componentes import aviso, destinos, num, pasos, pie, portada, recurso, seccion, tarjetas

img = recurso("modelo_imagen")
fus = recurso("modelo_fusion")
m = img["metricas"]

portada(estado.portada(), "¿Responderá el tumor <em>antes</em> de operar?",
        "Una CNN 2D diseñada y entrenada desde cero lee la resonancia con contraste previa a la quimioterapia "
        "neoadyuvante y estima la probabilidad de <b>respuesta patológica completa (pCR)</b>: que en la cirugía "
        "ya no quede tumor invasivo. Opcionalmente se combina con cuatro datos clínicos.",
        [("prediccion", "Predecir una paciente", True), ("red_imagen", "Ver la red por dentro", False)])

aviso("<b>Uso exclusivamente educativo.</b> Una probabilidad no es una certeza: una estimación alta no garantiza "
      "pCR ni permite evitar la cirugía, y una baja no la descarta.", "atencion")

# Escala de las barras: AUC entre el azar (0,5) y la perfección (1).
escala = lambda auc: (auc - 0.5) / 0.5
seccion("1", "El proyecto en <em>cifras</em>")
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
    {"t": "Test reservado", "v": "176", "d": "Sellado: sin evaluar todavía",
     "ayuda": "El test público se evaluará una sola vez con el modelo congelado. No interviene en ninguna decisión."},
])

seccion("2", "De la resonancia a una <em>probabilidad</em>")
pasos([
    ("Tres fases", "PRE, EARLY y LATE del mismo corte, antes del tratamiento"),
    ("CNN propia", f"{img['parametros']:,} parámetros desde cero; 5 redes en conjunto".replace(",", ".")),
    ("Probabilidad", "Media de los cortes de la paciente, calibrada con la validación cruzada"),
    ("Clínica (opcional)", "Edad, volumen tumoral, HR y HER2 en una regresión logística"),
    ("Banda", "Baja, intermedia o alta, con la pCR que se observó en cada una"),
])

seccion("3", "Cómo leer una <em>predicción</em>")
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

seccion("4", "Explorar")
destinos([
    ("prediccion", "Predecir una paciente", "Sube las tres fases o usa un ejemplo. Visor de cortes, bandas y la red por dentro."),
    ("red_imagen", "CNN de imagen", "La red neurona a neurona, cómo aprendió época a época y cuánto discrimina."),
    ("combinado", "Imagen + clínica", "Fusión tardía evaluada con CV anidada: qué aporta la imagen sobre la clínica."),
    ("metodologia", "Datos y método", "Cohortes, separación por paciente, reglas de evaluación y límites."),
])
pie()
