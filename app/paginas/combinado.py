import altair as alt
import pandas as pd
import streamlit as st

from componentes import (COBRE, GRIS, TINTA2, _base, aviso, bandas_grafica, cabecera, calibracion,
                         grafica, matriz, num, pct, pie, pr, recurso, roc, seccion, tabla, tarjetas, estructura_fusion)

fus = recurso("modelo_fusion")
if not fus:
    st.info("Faltan los recursos de evaluación del modelo combinado.")
    st.stop()
r = fus["resumen"]
m = r["brazos"][r["alias"]["completa+imagen"]]
config = fus.get("configuracion", {})
sub = config.get("submodelo")
prevalencia = fus["bandas"]["subtipo+edad+volumen+imagen"]["prevalencia"]

cabecera("Modelo 2 · imagen + clínica", "Evaluación del <em>modelo combinado</em>",
         "Regresión logística regularizada sobre la puntuación de imagen, edad, volumen, HR y HER2. "
         "Resultados por paciente en validación cruzada anidada; test reservado.",
         meta=[("variables", "5"), ("evaluación", "5 folds anidados"),
               ("regularización", "L2 · C = 1"), ("test", "pendiente")])
tarjetas([
    {"t": "Discriminación · AUC", "v": num(m["auc_media_folds"]), "d": f"± {num(m['auc_de_folds'])} entre folds",
     "acento": True, "ayuda": "AUC media en validación anidada. 0,5 = azar; 1 = separación perfecta."},
    {"t": "Precisión media · AP", "v": num(m["ap"]), "d": f"Referencia sin señal: {num(prevalencia)}"},
    {"t": "Error probabilístico · Brier", "v": num(m["brier"]), "d": "Menor es mejor; 0 es perfecto"},
    {"t": "Pérdida · BCE", "v": num(m["bce"]), "d": "Fuera de fold; menor es mejor"},
])


if sub:
    estructura_fusion(sub)

seccion("1", "Discriminación y estabilidad", "Cada paciente se evalúa fuera del ajuste de su fold.")
c1, c2 = st.columns(2, gap="large")
with c1:
    grafica(roc({"Combinado": fus["roc"]["completa+imagen"]}, {"Combinado": COBRE}, "ROC del combinado · OOF anidada"))
with c2:
    df = pd.DataFrame({"Fold": [str(i + 1) for i in range(len(m["auc_por_fold"]))], "AUC": m["auc_por_fold"]})
    barras = alt.Chart(df).mark_bar(color=COBRE, size=32).encode(
        x=alt.X("Fold:O", title="Fold exterior"), y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=["Fold:O", alt.Tooltip("AUC:Q", format=".3f")])
    etiquetas = barras.mark_text(dy=-9, color=TINTA2).encode(text=alt.Text("AUC:Q", format=".3f"))
    azar = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color=GRIS, strokeDash=[3, 3]).encode(y="y:Q")
    grafica(_base(alt.layer(barras, etiquetas, azar).properties(title="AUC en cada fold"), 360))
if fus.get("pr", {}).get("completa+imagen"):
    grafica(pr(fus["pr"]["completa+imagen"], prevalencia))
else:
    st.caption("Curva precisión–sensibilidad pendiente de exportar desde las predicciones OOF originales; AP guardada arriba.")
st.caption("Discriminación moderada. La DE describe variación entre folds; no es un intervalo de confianza ni descarta sobreajuste.")
with st.expander("Rendimiento por cohorte y subtipo", icon=":material/groups:"):
    filas = [{"Grupo": "Cohorte " + k, "AUC del combinado": num(v)} for k, v in m["auc_dentro_cohorte"].items()]
    filas += [{"Grupo": "Subtipo " + k, "AUC del combinado": num(v["completa+imagen"])} for k, v in r["auc_por_subtipo"].items()]
    tabla(pd.DataFrame(filas))

seccion("2", "Errores de clasificación", "Selecciona el umbral: cambia el equilibrio entre falsos positivos y falsos negativos.")
umbrales = {k: v for k, v in fus.get("umbrales", {}).items() if k.endswith("+imagen")}
if umbrales:
    patron = st.selectbox("Datos disponibles del modelo combinado", list(umbrales),
                         index=list(umbrales).index("subtipo+edad+volumen+imagen"))
    regla = st.segmented_control("Umbral de decisión", ["Youden", "Máximo macro-F1"], default="Youden")
    met = umbrales[patron]["macro_f1" if regla == "Máximo macro-F1" else "youden"]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        grafica(matriz(met))
        st.caption(f"Umbral {num(met['umbral'])} · {met['TP'] + met['TN'] + met['FP'] + met['FN']} pacientes.")
    with c2:
        tabla(pd.DataFrame([{"Métrica": nombre, "Valor": pct(met[clave], 1)} for clave, nombre in
             (("sensibilidad", "Sensibilidad"), ("especificidad", "Especificidad"),
              ("precision_pcr", "Valor predictivo positivo"), ("exactitud", "Exactitud"),
              ("f1_pcr", "F1 pCR"), ("macro_f1", "Macro-F1"))]))
    st.caption("Umbrales elegidos con estas mismas OOF: las métricas de clasificación pueden ser optimistas. Test pendiente.")

seccion("3", "Calidad de las probabilidades")
c1, c2 = st.columns(2, gap="large")
with c1:
    grafica(bandas_grafica(fus["bandas"]["subtipo+edad+volumen+imagen"], "pCR observada por banda del combinado"))
with c2:
    ref_brier = prevalencia * (1 - prevalencia)
    df = pd.DataFrame([{"Predicción": "Prevalencia constante", "Brier": ref_brier},
                       {"Predicción": "Combinado OOF", "Brier": m["brier"]}])
    barras = alt.Chart(df).mark_bar(color=COBRE, size=40).encode(
        x=alt.X("Predicción:N", axis=alt.Axis(labelAngle=0), title=None),
        y=alt.Y("Brier:Q", title="Brier · menor es mejor", scale=alt.Scale(domain=[0, 0.25])),
        tooltip=["Predicción:N", alt.Tooltip("Brier:Q", format=".3f")])
    grafica(_base(alt.layer(barras, barras.mark_text(dy=-9, color=TINTA2).encode(text=alt.Text("Brier:Q", format=".3f")))
                 .properties(title="Error de las probabilidades"), 320))
st.caption("La prevalencia constante es una referencia descriptiva calculada en las OOF. "
           "El Brier combina calibración y discriminación; por sí solo no demuestra buena calibración.")
if fus.get("calibracion", {}).get("completa+imagen"):
    grafica(calibracion(fus["calibracion"]["completa+imagen"]))
else:
    st.caption("Calibración por deciles pendiente: faltan las probabilidades individuales OOF en esta copia. "
               "Las bandas muestran tasas observadas por tercil, no una curva de calibración.")

seccion("4", "Complejidad y parámetros de la fusión",
        "Cinco coeficientes y un intercepto en el submodelo completo; la CNN se mantiene congelada.")
if sub:
    tarjetas([
        {"t": "Parámetros logísticos", "v": str(len(sub["coeficientes"]) + 1), "d": "5 coeficientes + intercepto"},
        {"t": "Pacientes del ajuste final", "v": str(sub["n_ajuste"]), "d": f"{sub['pcr_ajuste']} con pCR"},
        {"t": "Penalización", "v": "L2", "d": "C = 1 · sin pesos de clase"},
        {"t": "Interacciones", "v": "0", "d": "Variables estandarizadas"},
    ])
coef = pd.DataFrame(fus["coeficientes_por_fold"]).melt(id_vars="fold", var_name="variable", value_name="coeficiente")
nombres = {"HR": "HR positivo", "HER2": "HER2 positivo", "edad": "Edad", "log_volumen": "Volumen (log)", "imagen": "Puntuación de imagen"}
coef["variable"] = coef["variable"].map(nombres)
base = alt.Chart(coef).encode(y=alt.Y("variable:N", title=None, sort=list(nombres.values())))
rango = base.mark_rule(color=COBRE, strokeWidth=5, opacity=0.3).encode(x="min(coeficiente):Q", x2="max(coeficiente):Q")
puntos = base.mark_point(filled=True, color=COBRE, size=85).encode(
    x=alt.X("coeficiente:Q", title="Coeficiente estandarizado · cada punto es un fold"),
    tooltip=["variable:N", "fold:Q", alt.Tooltip("coeficiente:Q", format="+.3f")])
cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=GRIS, strokeDash=[3, 3]).encode(x="x:Q")
grafica(_base(alt.layer(cero, rango, puntos), 280))
st.caption("L2 reduce la magnitud de los coeficientes; no garantiza ausencia de sobreajuste. "
           "La estabilidad entre folds es un diagnóstico complementario. Los 6 parámetros corresponden a la etapa logística, no a la CNN.")
if sub:
    with st.expander("Parámetros del paquete final", icon=":material/table_rows:"):
        tabla(pd.DataFrame([{"Variable": nombres.get(k, k), "Coeficiente": num(v),
                             "Media": num(sub["media"][i]), "Escala": num(sub["escala"][i])}
                            for i, (k, v) in enumerate(sub["coeficientes"].items())]))
        st.caption(f"Intercepto {num(sub['intercepto'])}. Volumen transformado como log(1 + volumen). "
                   "Cada patrón de datos ausentes tiene su propia logística.")

seccion("5", "¿Hay sobreajuste de la fusión?",
        "Diagnóstico propio: rendimiento en el ajuste logístico frente al fold exterior reservado.")
diag = fus.get("diagnostico_generalizacion", r.get("diagnostico_generalizacion", []))
if diag:
    filas = [{"Fold": str(d["fold"] + 1), "Conjunto": titulo,
              "AUC": d[clave]["auc"], "BCE": d[clave]["bce"]}
             for d in diag for clave, titulo in (("ajuste", "Ajuste"), ("validacion", "Validación exterior"))]
    for metrica in ("AUC", "BCE"):
        grafica(_base(alt.Chart(pd.DataFrame(filas)).mark_line(point=True).encode(
            x=alt.X("Fold:O"), y=alt.Y(f"{metrica}:Q", scale=alt.Scale(zero=False)),
            color="Conjunto:N", tooltip=["Fold:O", "Conjunto:N", alt.Tooltip(f"{metrica}:Q", format=".3f")])
            .properties(title=f"{metrica} · ajuste y validación de la fusión"), 260))
    tabla(pd.DataFrame([{"Fold": d["fold"] + 1, "Brecha AUC (ajuste − validación)": num(d["brecha_auc"]),
                         "Brecha BCE (validación − ajuste)": num(d["brecha_bce"])} for d in diag]))
    st.caption("Brechas positivas indican optimismo del ajuste. Las puntuaciones de imagen del ajuste son cross-fitted. "
               "Este diagnóstico mide la etapa logística, no la memorización de la CNN; no hay un umbral universal que descarte sobreajuste.")
else:
    aviso("<b>No determinado.</b> La copia local no contiene las predicciones internas necesarias para medir "
          "la brecha de ajuste–validación. La AUC anidada mide generalización, pero no cuantifica esa brecha.", "atencion")
    with st.expander("Cómo completar el diagnóstico", icon=":material/terminal:"):
        st.code("python scripts/diagnostico_fusion.py", language="bash")
        st.caption("Ejecutar desde la raíz del proyecto en el equipo que conserva runs/cv_interna_fusion y "
                   "runs/cv_presupuesto. Reajusta solo las logísticas de los folds con el mismo C = 1; "
                   "no entrena CNN, no cambia el paquete final y no lee test.")

seccion("6", "Conclusión")
aviso("<b>Discriminación moderada en validación anidada.</b> El combinado obtiene AUC 0,714 y AP 0,505; "
      "sus probabilidades mejoran el Brier de la referencia constante. Esto no basta para afirmar "
      "buena calibración ni ausencia de sobreajuste. <b>Evaluación final en test pendiente.</b>", "clave")
pie()
