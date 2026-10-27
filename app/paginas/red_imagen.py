import pandas as pd
import streamlit as st

import estado
from componentes import (COBRE, COHORTES, GRIS, aviso, bandas_grafica, bloques_3d, cabecera, calibracion,
                         curvas_comparadas, curvas_epoca, grafica, matriz, num, pct, pie, pr, recurso, red_neuronal,
                         roc, seccion, tabla, tarjetas)

img = recurso("modelo_imagen")
m = img["metricas"]
curvas = img["curvas"]
fin = curvas[-1]
sel_min = min(curvas, key=lambda c: c["sel_bce_obj"])
# Meseta: primera época a menos de 0,001 de la mejor AUC media de validación.
auc_pico = next(c for c in curvas if c["sel_auc"] >= max(x["sel_auc"] for x in curvas) - 0.001)

cabecera("Modelo 1 · solo imagen", "CNN de imagen: <em>PartyNet</em>",
         "Una CNN 2D propia, sin pesos preentrenados ni datos clínicos de entrada. Su capacidad de discriminación "
         "es modesta, pero está bien medida, es estable entre folds y no memoriza.",
         meta=[("parámetros", f"{img['parametros']:,}".replace(",", ".")), ("redes", "5 × semilla"),
               ("épocas", "12"), ("entrada", "3 fases · 256²"), ("pacientes", "1.097")])

tarjetas([
    {"t": "AUC media (5 folds)", "v": num(m["auc_media_folds"]), "d": f"± {num(m['auc_de_folds'])} entre folds", "acento": True,
     "ayuda": "Cada paciente se predice con redes que no la vieron. Por fold: "
              + ", ".join(num(a) for a in m["auc_por_fold"]) + "."},
    {"t": "IC95 % de la AUC", "v": f"{num(m['auc_ic95'][0], 2)}–{num(m['auc_ic95'][1], 2)}", "d": "No incluye el 0,5 del azar",
     "ayuda": "Bootstrap estratificado por clase sobre las 1.097 predicciones fuera de fold."},
    {"t": "Precisión media (AP)", "v": num(m["ap"]), "d": f"Azar = {num(m['prevalencia'])}",
     "ayuda": "Área bajo la curva precisión–sensibilidad. Solo con la cohorte: " + num(m["ap_cohorte"]) + "."},
    {"t": "Brecha de memorización", "v": num(fin["fit_auc"] - fin["sel_auc"], 2), "d": "AUC entrenamiento − validación, época 12",
     "ayuda": "Con la receta antigua llegaba a 0,28 y la validación caía al azar."},
])

# ---------------------------------------------------------------- arquitectura
seccion("1", "Recorrido de un corte por la red",
        "Un corte real se propaga por la red de izquierda a derecha. Cada miniatura es el mapa de activación de un "
        "filtro; las líneas representan los pesos efectivos entre los filtros con mayor contribución a la salida. "
        "El selector de vista muestra el mismo filtro sobre el recorte de 256, 128 o 64 px.")
ej = estado.ejemplos()
c1, c2 = st.columns([3, 2])
pid = c1.selectbox("Paciente de ejemplo (train)", ej.index, index=3, format_func=lambda p: (
    f"{p} · {COHORTES.get(ej.loc[p, 'cohorte'], ej.loc[p, 'cohorte'])} · {'pCR' if ej.loc[p, 'pCR'] else 'no pCR'}"))
pixeles, nombres = estado.cortes_ejemplo(pid)
k = c2.select_slider("Corte", options=list(range(len(nombres))), value=len(nombres) // 2,
                     format_func=lambda i: nombres[i]) if len(nombres) > 1 else 0
red_neuronal(estado.recorrido(pixeles[k]), alto=620)

seccion("2", "Arquitectura en bloques",
        "La misma red como volúmenes: la altura de cada bloque es proporcional a la resolución espacial y su "
        "profundidad, al número de canales. Dentro de cada bloque se muestran mapas reales del corte seleccionado "
        "y los conos indican el campo receptivo de la convolución entre etapas.")
bloques_3d(estado.bloques(pixeles[k]))
with st.expander("Tabla de capas", icon=":material/table_rows:"):
    tabla(pd.DataFrame([{"Etapa": c["nombre"], "Salida": c["forma"], "Qué hace": c["detalle"],
                         "Parámetros": c["params"] or "—"} for c in estado.capas_red()]))
    st.caption("Las tres vistas pasan por el mismo tronco convolucional (pesos compartidos). La cabeza auxiliar del "
               "código no se usa en este modelo (λ = 0). Entrenamiento: AdamW, LR lineal hasta 0 en la época 12, EMA "
               "de pesos, BCE ponderada y aumentos geométricos idénticos en las tres fases. En la vista de arriba, el "
               "peso efectivo entre filtros es la suma de |pesos| de las dos convoluciones de cada bloque, escalada por "
               "su BatchNorm y por la activación del filtro de origen.")

# ---------------------------------------------------------------- entrenamiento
seccion("3", "Entrenamiento por épocas",
        "Media de las 15 redes de la validación cruzada (5 folds × 3 semillas); la sombra es el rango "
        "intercuartílico. «Validación» es un 15 % de pacientes que la red no ve al entrenar.")
c1, c2 = st.columns(2, gap="large")
with c1:
    grafica(curvas_epoca(curvas, "fit_bce_obj", "sel_bce_obj", "Pérdida (BCE ponderada)", "pérdida",
                         marca_x=sel_min["epoca"]))
with c2:
    grafica(curvas_epoca(curvas, "fit_auc", "sel_auc", "AUC por paciente", "AUC", dominio=[0.45, 0.7],
                         marca_x=auc_pico["epoca"]))
aviso(f"<b>Diagnóstico.</b> La pérdida de validación toca su mínimo en la <b>época {sel_min['epoca']}</b> y después "
      f"sube muy poco ({num(sel_min['sel_bce_obj'])} → {num(fin['sel_bce_obj'])}): sobreajuste leve. La AUC de "
      f"validación no baja nunca: sube hasta la <b>época {auc_pico['epoca']}</b> ({num(auc_pico['sel_auc'])}) y se "
      f"mantiene. El entrenamiento termina en la época 12 con la tasa de aprendizaje en cero.", "clave")
with st.expander("¿Cómo era antes? Comparación con la receta antigua", icon=":material/compare_arrows:"):
    grafica(curvas_comparadas(img["curvas_control"], curvas))
    st.caption("Con todos los cortes en cada época y una tasa de aprendizaje más alta, la red memorizaba desde la "
               "época 2–3: la brecha crecía hasta 0,28. La receta actual usa 3 cortes al azar por paciente y época "
               "y un presupuesto fijo de 12 épocas.")

# ---------------------------------------------------------------- discriminación
seccion("4", "Capacidad de discriminación",
        "Predicciones fuera de fold de las 1.097 pacientes de desarrollo. El test público no se ha usado.")
c1, c2 = st.columns(2, gap="large")
with c1:
    grafica(roc({"CNN de imagen": img["roc"], "Solo cohorte": img["roc_cohorte"]},
                {"CNN de imagen": COBRE, "Solo cohorte": GRIS}))
with c2:
    grafica(pr(img["pr"], m["prevalencia"]))
grupos = {**{f"Cohorte {COHORTES.get(c, c)}": a for c, a in m["auc_por_cohorte"].items()},
          **{f"Subtipo {s}": a for s, a in m["auc_por_subtipo"].items()}}
with st.expander("AUC por cohorte y por subtipo", icon=":material/groups:"):
    tabla(pd.DataFrame([{"Grupo": g, "AUC": num(a)} for g, a in grupos.items()]))
    st.caption("La imagen discrimina algo mejor en HER2+ y casi nada en HR+/HER2−.")

# ---------------------------------------------------------------- umbral y bandas
seccion("5", "Umbral y bandas de probabilidad",
        "Un sí/no depende de un umbral. Las bandas ofrecen una lectura más honesta: qué proporción de pacientes "
        "con una probabilidad parecida alcanzó pCR.")
c1, c2 = st.columns([1.05, 1], gap="large")
with c1:
    regla = st.segmented_control("Umbral", ["Youden", "Máximo macro-F1"], default="Youden")
    r = img["umbrales"]["youden" if regla != "Máximo macro-F1" else "macro_f1"]
    grafica(matriz(r))
    st.caption(f"Umbral {num(r['umbral'])} sobre la probabilidad calibrada · sensibilidad {pct(r['sensibilidad'], 1)} · "
               f"especificidad {pct(r['especificidad'], 1)} · VPP {pct(r['precision_pcr'], 1)}.")
with c2:
    grafica(bandas_grafica(img["bandas"], "pCR observada por banda"))
    st.caption("Cada barra agrupa un tercio de las pacientes; la línea discontinua es la prevalencia global. Con "
               "solo imagen las bandas se separan poco: la señal es débil.")

with st.expander("Calibración", icon=":material/tune:"):
    grafica(calibracion(img["calibracion"]))
    st.caption("Cada punto agrupa un 10 % de pacientes. Cerca de la diagonal = probabilidades honestas. Que todos "
               "queden entre ≈0,2 y ≈0,35 muestra que el modelo nunca está muy seguro.")

seccion("6", "Conclusión")
aviso("Con imagen sola y una CNN desde cero hay una <b>señal pequeña pero real</b>: el intervalo de la AUC excluye el "
      "azar, la variación entre folds es baja y las curvas muestran que la red no memoriza. Como predictor aislado es "
      "débil, lo que motiva combinarla con los datos clínicos.", "info")
pie()
