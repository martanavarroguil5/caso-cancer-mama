import pandas as pd
import streamlit as st

import motor
from componentes import (AZUL, GRIS, aviso, bandas_grafica, cabecera, calibracion, curvas_comparadas,
                         curvas_epoca, matriz, num, pct, pie, pr, recurso, red_3d, roc, seccion, tabla, tarjetas)

img = recurso("modelo_imagen")
m = img["metricas"]
curvas = img["curvas"]
fin = curvas[-1]
sel_min = min(curvas, key=lambda c: c["sel_bce_obj"])
# Meseta: primera época a menos de 0,001 de la mejor AUC media de validación.
auc_pico = next(c for c in curvas if c["sel_auc"] >= max(x["sel_auc"] for x in curvas) - 0.001)

cabecera("Modelo 1 · solo imagen", "CNN de imagen: PartyNet",
         "Una CNN 2D propia, sin pesos preentrenados ni datos clínicos de entrada. Su capacidad de "
         "discriminación es modesta, pero está bien medida, es estable entre folds y no memoriza.")

tarjetas([
    {"t": "AUC media (5 folds)", "v": num(m["auc_media_folds"]), "d": f"± {num(m['auc_de_folds'])} entre folds",
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
seccion("1", "Arquitectura", f"{img['parametros']:,} parámetros entrenables. ".replace(",", ".") +
        "Cada caja es una etapa: su grosor refleja el número de canales y su tamaño, la resolución espacial.")


@st.cache_data(show_spinner=False)
def capas():
    return motor.capas_red()


red_3d(capas())
with st.expander("Tabla de capas", icon=":material/table_rows:"):
    tabla(pd.DataFrame([{"Etapa": c["nombre"], "Salida": c["forma"], "Qué hace": c["detalle"],
                         "Parámetros": c["params"] or "—"} for c in capas()]))
    st.caption("Las tres vistas pasan por el mismo tronco convolucional (pesos compartidos). La cabeza auxiliar "
               "del código no se usa en este modelo (λ = 0). Entrenamiento: AdamW, LR lineal hasta 0 en la época 12, "
               "EMA de pesos, BCE ponderada, aumentos geométricos idénticos en las tres fases.")

# ---------------------------------------------------------------- entrenamiento
seccion("2", "Entrenamiento época a época",
        "Media de las 15 redes de la validación cruzada (5 folds × 3 semillas); la sombra es el rango "
        "intercuartílico. «Validación» es un 15 % de pacientes que la red no ve al entrenar.")
c1, c2 = st.columns(2)
with c1:
    st.altair_chart(curvas_epoca(curvas, "fit_bce_obj", "sel_bce_obj", "Pérdida (BCE ponderada)", "pérdida",
                                 marca_x=sel_min["epoca"]), width="stretch")
with c2:
    st.altair_chart(curvas_epoca(curvas, "fit_auc", "sel_auc", "AUC por paciente", "AUC", dominio=[0.45, 0.7],
                                 marca_x=auc_pico["epoca"]), width="stretch")
aviso(f"<b>Diagnóstico.</b> La pérdida de validación toca su mínimo en la <b>época {sel_min['epoca']}</b> y "
      f"después sube muy poco ({num(sel_min['sel_bce_obj'])} → {num(fin['sel_bce_obj'])}): sobreajuste leve. La AUC de "
      f"validación no baja nunca: sube hasta la <b>época {auc_pico['epoca']}</b> ({num(auc_pico['sel_auc'])}) y se "
      f"mantiene. El entrenamiento termina en la época 12 con la tasa de aprendizaje en cero.", "clave")
with st.expander("¿Cómo era antes? Comparación con la receta antigua", icon=":material/compare_arrows:"):
    st.altair_chart(curvas_comparadas(img["curvas_control"], curvas), width="stretch")
    st.caption("Con todos los cortes en cada época y una tasa de aprendizaje más alta, la red memorizaba desde la "
               "época 2–3: la brecha crecía hasta 0,28. La receta actual usa 3 cortes al azar por paciente y época "
               "y un presupuesto fijo de 12 épocas.")

# ---------------------------------------------------------------- discriminación
seccion("3", "Capacidad de discriminación",
        "Predicciones fuera de fold de las 1.097 pacientes de desarrollo. El test público no se ha usado.")
c1, c2 = st.columns(2)
with c1:
    st.altair_chart(roc({"CNN de imagen": img["roc"], "Solo cohorte": img["roc_cohorte"]},
                        {"CNN de imagen": AZUL, "Solo cohorte": GRIS}), width="stretch")
with c2:
    st.altair_chart(pr(img["pr"], m["prevalencia"]), width="stretch")
filas = [{"Grupo": k, "AUC": num(v)} for k, v in {**{f"Cohorte {c}": a for c, a in m["auc_por_cohorte"].items()},
                                                    **{f"Subtipo {s}": a for s, a in m["auc_por_subtipo"].items()}}.items()]
with st.expander("AUC por cohorte y por subtipo", icon=":material/groups:"):
    tabla(pd.DataFrame(filas))
    st.caption("La imagen discrimina algo mejor en HER2+ y casi nada en HR+/HER2−.")

# ---------------------------------------------------------------- umbral y bandas
seccion("4", "De la probabilidad a una decisión",
        "Un sí/no depende de un umbral. Las bandas ofrecen una lectura más honesta: qué proporción de "
        "pacientes con una probabilidad parecida alcanzó pCR.")
c1, c2 = st.columns([1.05, 1])
with c1:
    regla = st.segmented_control("Umbral", ["Youden", "Máximo macro-F1"], default="Youden")
    r = img["umbrales"]["youden" if regla != "Máximo macro-F1" else "macro_f1"]
    st.altair_chart(matriz(r), width="stretch")
    st.caption(f"Umbral {num(r['umbral'])} sobre la probabilidad calibrada · sensibilidad {pct(r['sensibilidad'], 1)} · "
               f"especificidad {pct(r['especificidad'], 1)} · VPP {pct(r['precision_pcr'], 1)}.")
with c2:
    st.altair_chart(bandas_grafica(img["bandas"], "pCR observada por banda (terciles)"), width="stretch")
    st.caption("Cada barra agrupa un tercio de las pacientes; la línea discontinua es la prevalencia global. "
               "Con solo imagen las bandas se separan poco: la señal es débil.")

with st.expander("Calibración", icon=":material/tune:"):
    st.altair_chart(calibracion(img["calibracion"]), width="stretch")
    st.caption("Cada punto agrupa un 10 % de pacientes. Cerca de la diagonal = probabilidades honestas. Que todos "
               "queden entre ≈0,2 y ≈0,35 muestra que el modelo nunca está muy seguro.")

seccion("5", "Conclusión")
aviso("Con imagen sola y una CNN desde cero hay una <b>señal pequeña pero real</b>: el intervalo de la AUC excluye "
      "el azar, la variación entre folds es baja y las curvas muestran que la red no memoriza. Como predictor "
      "aislado es débil, lo que motiva combinarla con los datos clínicos.", "info")
pie()
