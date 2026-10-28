import altair as alt
import pandas as pd
import streamlit as st

import estado
from componentes import (COBRE, COHORTES, GRIS, TINTA2, _base, aviso, bandas_grafica, bloques_3d, cabecera, calibracion,
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

# ---------------------------------------------------------------- entrenamiento
st.markdown('<div id="perdida-cnn"></div>', unsafe_allow_html=True)
seccion("1", "Pérdida de la CNN · época a época",
        "Entrenamiento y validación en la misma gráfica. Una pérdida menor indica menos error; "
        "si entrenamiento baja y validación sube, aumenta la brecha de generalización.")
fuentes_curvas = {"Media de los 5 folds": curvas}
for fold_id, historia in sorted(img.get("curvas_por_fold", {}).items(), key=lambda par: int(par[0])):
    if historia:
        fuentes_curvas[f"Fold {int(fold_id) + 1}"] = historia
if len(fuentes_curvas) > 1:
    ambito_curvas = st.segmented_control("Curvas de entrenamiento de la CNN", list(fuentes_curvas),
                                        default="Media de los 5 folds", key="cnn_curvas_ambito") or "Media de los 5 folds"
else:
    ambito_curvas = "Media de los 5 folds"
curvas_vista = fuentes_curvas[ambito_curvas]
epocas_vista = [c["epoca"] for c in curvas_vista]
epoca_elegida = st.select_slider("Época de entrenamiento", options=epocas_vista, value=epocas_vista[-1],
                                key=f"cnn_epoca_{ambito_curvas}")
fila_epoca = next(c for c in curvas_vista if c["epoca"] == epoca_elegida)
minimo_vista = min(curvas_vista, key=lambda c: c["sel_bce_obj"])
final_vista = curvas_vista[-1]
grafica(curvas_epoca(curvas_vista, "fit_bce_obj", "sel_bce_obj",
                     f"Pérdida BCE ponderada · {ambito_curvas.lower()}", "Pérdida BCE ponderada",
                     marca_x=epoca_elegida))
tarjetas([
    {"t": f"Época {epoca_elegida} · entrenamiento", "v": num(fila_epoca["fit_bce_obj"], 4),
     "d": "BCE ponderada, evaluada sin aumentos"},
    {"t": f"Época {epoca_elegida} · validación", "v": num(fila_epoca["sel_bce_obj"], 4),
     "d": "Pacientes de selección no usadas para entrenar", "acento": True},
    {"t": "Brecha de pérdida", "v": f"{fila_epoca['sel_bce_obj'] - fila_epoca['fit_bce_obj']:+.4f}".replace(".", ","),
     "d": "Validación − entrenamiento en esta época"},
])
st.caption(f"Cada punto resume {fila_epoca['n']} redes en la época seleccionada. La sombra es el rango "
           "intercuartílico entre redes, no un intervalo de confianza. «Validación» es el conjunto interno de "
           "selección (15 %); no es el fold exterior utilizado para calcular la AUC OOF.")
st.caption(f"Mínimo de pérdida de validación: época {minimo_vista['epoca']} ({num(minimo_vista['sel_bce_obj'], 4)}). "
           f"Última época: {final_vista['epoca']} ({num(final_vista['sel_bce_obj'], 4)}). "
           "Una subida tras el mínimo es una señal de sobreajuste; se interpreta junto a la magnitud de la brecha y la AUC.")
if len(fuentes_curvas) == 1:
    st.caption("Curva media de 5 folds × 3 semillas. El desglose de pérdida por fold no está disponible en esta versión.")
with st.expander("AUC de entrenamiento y validación por época", icon=":material/show_chart:"):
    grafica(curvas_epoca(curvas_vista, "fit_auc", "sel_auc", f"AUC por época · {ambito_curvas.lower()}", "AUC",
                         marca_x=epoca_elegida))
with st.expander("Valores de pérdida en cada época", icon=":material/table_rows:"):
    tabla(pd.DataFrame([{"Época": c["epoca"], "Entrenamiento": num(c["fit_bce_obj"], 4),
                         "Validación": num(c["sel_bce_obj"], 4),
                         "Brecha": num(c["sel_bce_obj"] - c["fit_bce_obj"], 4), "Redes": c["n"]}
                        for c in curvas_vista]))

with st.expander("¿Cómo era antes? Comparación con la receta antigua", icon=":material/compare_arrows:"):
    grafica(curvas_comparadas(img["curvas_control"], curvas))
    st.caption("Con todos los cortes en cada época y una tasa de aprendizaje más alta, la red memorizaba desde la "
               "época 2–3: la brecha crecía hasta 0,28. La receta actual usa 3 cortes al azar por paciente y época "
               "y un presupuesto fijo de 12 épocas.")

# ---------------------------------------------------------------- arquitectura
seccion("2", "Recorrido de un corte por la red",
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

seccion("3", "Arquitectura en bloques",
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

# ---------------------------------------------------------------- discriminación
seccion("4", "Capacidad de discriminación",
        "Predicciones fuera de fold de las 1.097 pacientes de desarrollo. El test público no se ha usado.")
c1, c2 = st.columns(2, gap="large")
with c1:
    grafica(roc({"CNN de imagen": img["roc"], "Solo cohorte": img["roc_cohorte"]},
                {"CNN de imagen": COBRE, "Solo cohorte": GRIS}))
with c2:
    grafica(pr(img["pr"], m["prevalencia"]))

st.markdown("### CNN · comparación fold a fold")
st.caption("En cada fold se evalúa un grupo distinto de pacientes, excluido del entrenamiento de sus redes. "
           "Los folds son particiones independientes, no épocas consecutivas.")
valores_fold = m["auc_por_fold"]
opciones_fold = ["Todos"] + [f"Fold {i + 1}" for i in range(len(valores_fold))]
fold_elegido = st.segmented_control("Resaltar un fold de la CNN", opciones_fold, default="Todos", key="cnn_fold") or "Todos"
filas_fold = pd.DataFrame([
    {"Fold": f"Fold {i + 1}", "AUC": valor, "Azar": 0.5,
     "Diferencia con la media": valor - m["auc_media_folds"],
     "Resaltado": fold_elegido == "Todos" or fold_elegido == f"Fold {i + 1}"}
    for i, valor in enumerate(valores_fold)
])
base_fold = alt.Chart(filas_fold).encode(
    x=alt.X("Fold:O", sort=opciones_fold[1:], title="Fold exterior · paciente fuera del entrenamiento"))
trazos_fold = base_fold.mark_rule(strokeWidth=3).encode(
    y=alt.Y("Azar:Q", title="AUC · eje ampliado (0,45–0,65)", scale=alt.Scale(domain=[0.45, 0.65], zero=False)),
    y2="AUC:Q", color=alt.condition(alt.datum.Resaltado, alt.value(COBRE), alt.value("#3c4652")))
puntos_fold = base_fold.mark_circle(size=180, stroke="#0d1116", strokeWidth=2).encode(
    y="AUC:Q", color=alt.condition(alt.datum.Resaltado, alt.value(COBRE), alt.value(GRIS)),
    tooltip=["Fold:O", alt.Tooltip("AUC:Q", format=".4f"),
             alt.Tooltip("Diferencia con la media:Q", format="+.4f")])
texto_fold = base_fold.mark_text(dy=-18, fontSize=14, color=TINTA2).encode(
    y="AUC:Q", text=alt.Text("AUC:Q", format=".3f"))
referencias_fold = pd.DataFrame([
    {"AUC": m["auc_media_folds"], "Referencia": "Media de los 5 folds"},
    {"AUC": 0.5, "Referencia": "Azar"},
])
lineas_fold = alt.Chart(referencias_fold).mark_rule(strokeDash=[5, 4], strokeWidth=1).encode(
    y="AUC:Q", color=alt.Color("Referencia:N", scale=alt.Scale(
        domain=["Media de los 5 folds", "Azar"], range=[COBRE, GRIS]), legend=alt.Legend(title=None)),
    tooltip=["Referencia:N", alt.Tooltip("AUC:Q", format=".3f")])
grafica(_base(alt.layer(trazos_fold, lineas_fold, puntos_fold, texto_fold).properties(
    title="Rendimiento de la CNN en cada partición de validación"), 320))
if fold_elegido != "Todos":
    valor_fold = valores_fold[opciones_fold.index(fold_elegido) - 1]
    tarjetas([
        {"t": fold_elegido + " · AUC", "v": num(valor_fold), "d": "Predicciones fuera de fold", "acento": True},
        {"t": "Diferencia con la media", "v": f"{valor_fold - m['auc_media_folds']:+.3f}".replace(".", ","),
         "d": "Media de referencia: " + num(m["auc_media_folds"])},
        {"t": "Diferencia con el azar", "v": f"{valor_fold - 0.5:+.3f}".replace(".", ","), "d": "Azar = 0,500"},
    ])
else:
    tarjetas([
        {"t": "Media de los folds", "v": num(m["auc_media_folds"]), "d": "5 particiones de pacientes", "acento": True},
        {"t": "Variación entre folds", "v": num(m["auc_de_folds"]), "d": "Desviación estándar; no es un IC"},
        {"t": "Rango observado", "v": num(min(valores_fold)) + "–" + num(max(valores_fold)),
         "d": "Separación máximo − mínimo: " + num(max(valores_fold) - min(valores_fold))},
    ])
st.caption("Los cinco valores son superiores a 0,5, con una señal débil y variación pequeña entre particiones. "
           "Sin intervalos por fold, eso no demuestra significación en cada uno ni ausencia de sobreajuste.")

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
