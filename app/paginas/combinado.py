import altair as alt
import pandas as pd
import streamlit as st

from componentes import (AMBAR, CIAN, GRIS, NARANJA, TINTA2, _base, aviso, bandas_grafica, cabecera, grafica, mancuernas,
                         num, pasos, pie, recurso, roc, seccion, tabla, tarjetas)

fus = recurso("modelo_fusion")
img = recurso("modelo_imagen")

cabecera("Modelo 2 · imagen + datos clínicos", "La CNN más <em>cuatro datos</em> clínicos",
         "La CNN de imagen no cambia. Su probabilidad entra como una variable más en una regresión logística junto "
         "con la edad, el volumen tumoral, HR y HER2. Así el peso de la imagen es visible y medible.",
         meta=[("fusión", "tardía"), ("variables", "imagen + 4"), ("submodelos", "por patrón"), ("evaluación", "CV anidada")])

seccion("1", "Diseño")
pasos([
    ("CNN congelada", "La misma red de la página anterior produce la puntuación de imagen"),
    ("Datos clínicos", "Edad, log(1 + volumen), HR y HER2; ni cohorte ni datos del escáner"),
    ("Logística por patrón", "Un submodelo para cada combinación de datos disponibles: nunca se imputa"),
    ("CV anidada", "La puntuación de imagen de las pacientes de ajuste sale de redes que no vieron el fold evaluado"),
])
with st.expander("¿Por qué así y no una red multimodal?", icon=":material/help:"):
    st.markdown(
        "- En el proyecto ya se probaron tres redes entrenadas junto con la clínica (V002, FiLM, SAM). En las tres, "
        "quitar la parte de imagen dejaba la AUC igual o mejor: la red aprendía a **ignorar las imágenes**.\n"
        "- Con la CNN congelada, la aportación de la imagen es un coeficiente y una diferencia pareada de AUC.\n"
        "- Si en la evaluación no hay datos clínicos, la CNN sola sigue siendo un modelo completo.\n"
        "- Con datos incompletos se usa el **submodelo por patrón** (Fletcher Mercaldo y Blume, *Biostatistics* "
        "2020): una logística entrenada con exactamente las variables disponibles.")

if fus is None:
    aviso("<b>Evaluación en curso.</b> Cuando termine la validación cruzada anidada, esta página mostrará la "
          "comparación con la imagen sola y con la clínica sola, los coeficientes y las bandas.", "atencion")
    pie()
    st.stop()

r = fus["resumen"]
a = r["alias"]
b = lambda n: r["brazos"][a[n]]
comp = r["comparaciones"]
principal = comp["principal: completa+imagen − completa"]

seccion("2", "Resultados de la <em>CV anidada</em>", "Mismas 1.097 pacientes y mismos folds para todos los modelos.")
escala = lambda auc: (auc - 0.5) / 0.5
tarjetas([
    {"t": "Imagen + clínica", "v": num(b("completa+imagen")["auc_media_folds"]), "acento": True,
     "d": f"AUC media ± {num(b('completa+imagen')['auc_de_folds'])}", "barra": escala(b("completa+imagen")["auc_media_folds"]),
     "barra_txt": "0,5 → 1", "ayuda": f"AP {num(b('completa+imagen')['ap'])} · Brier {num(b('completa+imagen')['brier'])}"},
    {"t": "Solo clínica", "v": num(b("completa")["auc_media_folds"]), "d": f"AUC media ± {num(b('completa')['auc_de_folds'])}",
     "barra": escala(b("completa")["auc_media_folds"]), "barra_txt": "0,5 → 1", "ayuda": "Edad, volumen, HR y HER2, sin imagen."},
    {"t": "Solo imagen", "v": num(b("imagen")["auc_media_folds"]), "d": "CNN recalibrada",
     "barra": escala(b("imagen")["auc_media_folds"]), "barra_txt": "0,5 → 1", "ayuda": "La misma CNN de la página anterior."},
    {"t": "Aporte de la imagen", "v": f"{principal['delta']:+.3f}".replace(".", ","),
     "d": f"IC95 % [{principal['ic95'][0]:+.3f}; {principal['ic95'][1]:+.3f}]".replace(".", ","),
     "ayuda": "Comparación principal fijada antes de entrenar: diferencia pareada de AUC media por fold "
              "(bootstrap estratificado, 2.000 remuestreos)."},
])
c1, c2 = st.columns([1.1, 1], gap="large")
with c1:
    grafica(roc(fus["roc"], {"completa+imagen": NARANJA, "completa": CIAN, "subtipo": AMBAR, "imagen": GRIS},
                "ROC fuera de fold (anidada)"))
with c2:
    filas = [{"Comparación": n.replace("principal: ", "★ "), "Δ AUC": f"{c['delta']:+.4f}",
              "IC95 %": f"[{c['ic95'][0]:+.4f}; {c['ic95'][1]:+.4f}]", "P(Δ≤0)": f"{c['p_delta_menor_igual_0']:.3f}"}
             for n, c in comp.items()]
    st.markdown("<div style='height:2.6rem'></div>", unsafe_allow_html=True)
    tabla(pd.DataFrame(filas))
    st.caption("★ comparación principal. Un IC que no cruza el 0 indica una diferencia que no se explica por azar.")

seccion("3", "Cuánto pesa <em>cada variable</em>", "Coeficientes estandarizados del modelo completo en cada fold: "
        "cuánto cambia el logit de pCR por cada desviación típica de la variable.")
coef = pd.DataFrame(fus["coeficientes_por_fold"]).melt(id_vars="fold", var_name="variable", value_name="coeficiente")
nombres = {"HR": "HR positivo", "HER2": "HER2 positivo", "edad": "Edad", "log_volumen": "Volumen (log)", "imagen": "Imagen (CNN)"}
coef["variable"] = coef["variable"].map(nombres)
y = alt.Y("variable:N", title=None, sort=list(nombres.values()), axis=alt.Axis(labelFont="Geist", labelFontSize=12))
rango = alt.Chart(coef).mark_rule(strokeWidth=6, opacity=0.25, strokeCap="round").encode(
    y=y, x="min(coeficiente):Q", x2="max(coeficiente):Q",
    color=alt.condition(alt.datum.variable == "Imagen (CNN)", alt.value(NARANJA), alt.value(GRIS)))
puntos = alt.Chart(coef).mark_circle(size=90, opacity=0.95, stroke="#0d1116", strokeWidth=1.5).encode(
    y=y, x=alt.X("coeficiente:Q", title="Coeficiente estandarizado (un punto por fold)"),
    color=alt.condition(alt.datum.variable == "Imagen (CNN)", alt.value(NARANJA), alt.value(TINTA2)),
    tooltip=["variable:N", "fold:Q", alt.Tooltip("coeficiente:Q", format="+.3f")])
cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#6b7480", strokeDash=[2, 3]).encode(x="x:Q")
grafica(_base(alt.layer(cero, rango, puntos), 250))

seccion("4", "Todas las <em>combinaciones</em> de datos", "AUC media de cada submodelo sin imagen (gris) y con "
        "imagen (naranja). La línea es lo que añade la imagen.")
filas = []
for clave, v in r["brazos"].items():
    if clave.endswith("|normal") and not clave.startswith("imagen"):
        base = clave.split("|")[0]
        if base.endswith("+imagen"):
            continue
        con = r["brazos"].get(f"{base}+imagen|normal")
        if con:
            filas.append({"fila": base.replace("+", " + "), "sin": v["auc_media_folds"], "con": con["auc_media_folds"]})
combos = pd.DataFrame(filas).sort_values("sin", ascending=False)
grafica(mancuernas(combos))
with st.expander("Tabla", icon=":material/table_rows:"):
    tabla(pd.DataFrame({"Datos clínicos": combos["fila"], "Sin imagen": combos["sin"].map(num),
                        "Con imagen": combos["con"].map(num),
                        "Δ": (combos["con"] - combos["sin"]).map(lambda d: f"{d:+.3f}".replace(".", ","))}))

seccion("5", "Bandas de <em>probabilidad</em>")
c1, c2 = st.columns(2, gap="large")
with c1:
    grafica(bandas_grafica(img["bandas"], "Solo imagen"))
with c2:
    grafica(bandas_grafica(fus["bandas"]["subtipo+edad+volumen+imagen"], "Imagen + clínica completa"))
st.caption("Cuanto más separadas están las barras, mejor distingue el modelo a las pacientes con y sin pCR.")
pie()
