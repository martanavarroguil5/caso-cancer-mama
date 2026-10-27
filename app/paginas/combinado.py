import altair as alt
import pandas as pd
import streamlit as st

from componentes import (AMARILLO, AZUL, GRIS, NARANJA, TINTA, _base, aviso, bandas_grafica, cabecera, num, pasos,
                         pie, recurso, roc, seccion, tabla, tarjetas)

fus = recurso("modelo_fusion")
img = recurso("modelo_imagen")

cabecera("Modelo 2 · imagen + datos clínicos", "Fusión tardía: la CNN más cuatro datos clínicos",
         "La CNN de imagen no cambia. Su probabilidad entra como una variable más en una regresión logística "
         "junto con la edad, el volumen tumoral, HR y HER2. Así el peso de la imagen es visible y medible.")

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
    aviso("<b>Evaluación en curso.</b> Se están entrenando las 60 redes internas de la validación cruzada anidada. "
          "Cuando termine, esta página mostrará la comparación con la imagen sola y con la clínica sola, los "
          "coeficientes y las bandas de probabilidad.", "atencion")
    pie()
    st.stop()

r = fus["resumen"]
a = r["alias"]
b = lambda n: r["brazos"][a[n]]
comp = r["comparaciones"]
principal = comp["principal: completa+imagen − completa"]

seccion("2", "Resultados de la validación cruzada anidada", "Mismas 1.097 pacientes y mismos folds para todos los modelos.")
tarjetas([
    {"t": "Imagen + clínica", "v": num(b("completa+imagen")["auc_media_folds"]),
     "d": f"AUC media ± {num(b('completa+imagen')['auc_de_folds'])}",
     "ayuda": f"AP {num(b('completa+imagen')['ap'])} · Brier {num(b('completa+imagen')['brier'])}"},
    {"t": "Solo clínica", "v": num(b("completa")["auc_media_folds"]), "d": f"AUC media ± {num(b('completa')['auc_de_folds'])}",
     "ayuda": "Edad, volumen, HR y HER2, sin imagen."},
    {"t": "Solo imagen", "v": num(b("imagen")["auc_media_folds"]), "d": "CNN recalibrada",
     "ayuda": "La misma CNN de la página anterior."},
    {"t": "Aporte de la imagen", "v": f"{principal['delta']:+.3f}".replace(".", ","),
     "d": f"IC95 % [{principal['ic95'][0]:+.3f}; {principal['ic95'][1]:+.3f}]".replace(".", ","),
     "ayuda": "Comparación principal fijada antes de entrenar: diferencia pareada de AUC media por fold "
              "(bootstrap estratificado, 2.000 remuestreos)."},
])
c1, c2 = st.columns([1.1, 1])
with c1:
    st.altair_chart(roc(fus["roc"], {"completa+imagen": NARANJA, "completa": GRIS, "subtipo": AMARILLO,
                                     "imagen": AZUL}, "ROC fuera de fold (anidada)"), width="stretch")
with c2:
    filas = [{"Comparación": n.replace("principal: ", "★ "), "Δ AUC": f"{c['delta']:+.4f}",
              "IC95 %": f"[{c['ic95'][0]:+.4f}; {c['ic95'][1]:+.4f}]", "P(Δ≤0)": f"{c['p_delta_menor_igual_0']:.3f}"}
             for n, c in comp.items()]
    tabla(pd.DataFrame(filas))
    st.caption("★ comparación principal. Un IC que no cruza el 0 indica una diferencia que no se explica por azar.")

seccion("3", "Cuánto pesa cada variable", "Coeficientes estandarizados del modelo completo en cada fold: "
        "cuánto cambia el logit de pCR por cada desviación típica de la variable.")
coef = pd.DataFrame(fus["coeficientes_por_fold"]).melt(id_vars="fold", var_name="variable", value_name="coeficiente")
nombres = {"HR": "HR positivo", "HER2": "HER2 positivo", "edad": "Edad", "log_volumen": "Volumen (log)", "imagen": "Imagen (CNN)"}
coef["variable"] = coef["variable"].map(nombres)
puntos = alt.Chart(coef).mark_circle(size=90, opacity=0.8).encode(
    y=alt.Y("variable:N", title=None, sort=list(nombres.values())), x=alt.X("coeficiente:Q", title="Coeficiente estandarizado"),
    color=alt.condition(alt.datum.variable == "Imagen (CNN)", alt.value(AZUL), alt.value(GRIS)),
    tooltip=["variable:N", "fold:Q", alt.Tooltip("coeficiente:Q", format="+.3f")])
cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=TINTA, strokeDash=[3, 3]).encode(x="x:Q")
st.altair_chart(_base(alt.layer(cero, puntos), 240), width="stretch")

seccion("4", "Todas las combinaciones de datos", "AUC media de cada submodelo con y sin imagen.")
filas = []
for clave, v in r["brazos"].items():
    if clave.endswith("|normal") and not clave.startswith("imagen"):
        base = clave.split("|")[0]
        if base.endswith("+imagen"):
            continue
        con = r["brazos"].get(f"{base}+imagen|normal")
        filas.append({"Datos clínicos": base.replace("+", " + "), "Sin imagen": num(v["auc_media_folds"]),
                      "Con imagen": num(con["auc_media_folds"]) if con else "—",
                      "Δ": f"{con['auc_media_folds'] - v['auc_media_folds']:+.3f}".replace(".", ",") if con else "—"})
tabla(pd.DataFrame(filas))

seccion("5", "Bandas de probabilidad")
c1, c2 = st.columns(2)
with c1:
    st.altair_chart(bandas_grafica(img["bandas"], "Solo imagen"), width="stretch")
with c2:
    st.altair_chart(bandas_grafica(fus["bandas"]["subtipo+edad+volumen+imagen"], "Imagen + clínica completa"),
                    width="stretch")
st.caption("Cuanto más separadas están las barras, mejor distingue el modelo a las pacientes con y sin pCR.")
pie()
