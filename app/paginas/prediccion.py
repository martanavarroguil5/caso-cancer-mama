import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import motor
from componentes import (AZUL, GRIS, aviso, banda_de, barra_bandas, cabecera, chip_banda, num, pct, pie,
                         recurso, seccion, _base)

cabecera("Predicción", "Predecir una paciente",
         "Sube las tres fases de uno o varios cortes de la misma paciente. Los datos clínicos son opcionales: "
         "si los añades, se muestra también la estimación combinada.")


@st.cache_resource(show_spinner="Verificando SHA-256 y cargando las 5 redes…")
def paquete_imagen():
    return motor.cargar_imagen()


@st.cache_resource(show_spinner="Cargando el modelo combinado…")
def paquete_fusion():
    return motor.cargar_fusion()


@st.cache_data(show_spinner=False)
def lista_ejemplos():
    return motor.ejemplos()


img_info = recurso("modelo_imagen")
fus_info = recurso("modelo_fusion")
paquete = paquete_imagen()
fusion = paquete_fusion()

# ---------------------------------------------------------------- 1. imágenes
seccion("1", "Imágenes", "PNG en gris de 256 × 256 por fase. El nombre debe terminar en <code>_PRE</code>, "
        "<code>_EARLY</code> o <code>_LATE</code>; los cortes se agrupan por el resto del nombre.")
origen = st.segmented_control("Origen", ["Subir PNG", "Paciente de ejemplo"], default="Subir PNG",
                              label_visibility="collapsed")
pixeles, nombres, clinica_ejemplo, etiqueta_real = None, [], {}, None
if origen == "Subir PNG":
    archivos = st.file_uploader("Fases PRE, EARLY y LATE", type="png", accept_multiple_files=True)
    if archivos:
        grupos, errores = motor.agrupar_subidas(archivos)
        for e in errores:
            st.warning(e)
        if grupos:
            try:
                pixeles, nombres = motor.leer_cortes(grupos), list(grupos)
            except ValueError as exc:
                st.error(f"Formato no válido: {exc}")
else:
    ej = lista_ejemplos()
    aviso("Estas pacientes son de <b>entrenamiento</b>: el modelo final las vio, así que su predicción es "
          "optimista y no sirve para evaluar. Solo ilustran el funcionamiento.", "atencion")
    pid = st.selectbox("Paciente", ej.index, format_func=lambda p: f"{p} · {ej.loc[p, 'cohorte']} · {ej.loc[p, 'cortes']} cortes")
    pixeles, nombres = motor.cortes_de_ejemplo(pid)
    f = ej.loc[pid]
    clinica_ejemplo = {"edad": f["edad"], "volumen": float(np.expm1(f["log_volumen"])), "HR": f["HR"], "HER2": f["HER2"]}
    etiqueta_real = int(f["pCR"])

if pixeles is not None:
    st.caption(f"{len(pixeles)} corte(s) válidos.")
    i = st.select_slider("Corte para visualizar", options=list(range(len(nombres))),
                         format_func=lambda k: nombres[k]) if len(nombres) > 1 else 0
    cols = st.columns(4)
    for col, fase, k in zip(cols, ("PRE", "EARLY", "LATE"), range(3)):
        col.image(pixeles[i, k], caption=fase, width="stretch", clamp=True)
    cols[3].image(motor.mapa_realce(pixeles[i]), caption="Realce EARLY − PRE (rojo = capta contraste)",
                  width="stretch")

# ---------------------------------------------------------------- 2. clínica
seccion("2", "Datos clínicos (opcional)",
        "Deja en «Desconocido» lo que no tengas: el modelo usa el submodelo entrenado con exactamente "
        "los datos disponibles, sin inventar los que faltan.")


def _opcion(valor):
    return "Desconocido" if valor is None or pd.isna(valor) else ("Positivo" if valor == 1 else "Negativo")


c1, c2, c3, c4 = st.columns(4)
edad_conocida = c1.checkbox("Edad conocida", value=bool(clinica_ejemplo) and not pd.isna(clinica_ejemplo.get("edad")))
edad = c1.number_input("Edad (años)", 18.0, 100.0, float(clinica_ejemplo.get("edad") or 50.0), 1.0,
                       disabled=not edad_conocida)
vol_conocido = c2.checkbox("Volumen conocido", value=bool(clinica_ejemplo) and not pd.isna(clinica_ejemplo.get("volumen")))
volumen = c2.number_input("Volumen tumoral", 0.0, 1000.0, float(clinica_ejemplo.get("volumen") or 10.0), 0.5,
                          disabled=not vol_conocido, help="Mismas unidades que tum_vol en patients.csv (≈ cm³).")
opciones = ["Desconocido", "Positivo", "Negativo"]
hr = c3.selectbox("Receptores hormonales (HR)", opciones, index=opciones.index(_opcion(clinica_ejemplo.get("HR"))))
her2 = c4.selectbox("HER2", opciones, index=opciones.index(_opcion(clinica_ejemplo.get("HER2"))))
a_num = {"Desconocido": None, "Positivo": 1.0, "Negativo": 0.0}
clinica = {"edad": edad if edad_conocida else None, "volumen": volumen if vol_conocido else None,
           "HR": a_num[hr], "HER2": a_num[her2]}
hay_clinica = any(v is not None for v in clinica.values())

# ---------------------------------------------------------------- 3. resultado
seccion("3", "Resultado")
if pixeles is None:
    aviso("Carga imágenes o elige una paciente de ejemplo para ver la predicción.", "info")
    pie()
    st.stop()

pred = motor.predecir_cortes(paquete, pixeles, nombres)
p_img = motor.calibrar(paquete, pred.p_cruda)
banda_img = banda_de(p_img, img_info["bandas"])
umbral_img = paquete.info["politica"]["umbral"]


def tarjeta_resultado(titulo, p, banda, umbral, detalle):
    clase = "pCR" if p >= umbral else "no pCR"
    st.markdown(f'<div class="resultado"><div class="etiqueta">{titulo}</div>'
                f'<div class="p">{pct(p, 1)}</div><div class="sub">probabilidad estimada de pCR</div>'
                f'<div style="margin-top:.8rem">{chip_banda(banda)}'
                f'<span class="chip" style="background:#f1f3f7;color:#4a5568">Umbral {num(umbral, 3)} → {clase}</span>'
                f'</div><div class="sub" style="margin-top:.7rem">{detalle}</div></div>', unsafe_allow_html=True)


col_img, col_fus = st.columns(2)
with col_img:
    dispersion = float(np.std([motor.calibrar(paquete, p) for p in pred.por_red]))
    tarjeta_resultado("Solo imagen · CNN", p_img, banda_img, umbral_img,
                      f"{len(nombres)} corte(s) × 5 redes × 4 reflexiones. Desacuerdo entre redes: ± {num(dispersion, 3)}.")
    barra_bandas(p_img, img_info["bandas"])
with col_fus:
    if fusion is None or fus_info is None:
        st.markdown('<div class="resultado"><div class="etiqueta">Imagen + datos clínicos</div>'
                    '<div class="p" style="color:#b6bfcc">—</div><div class="sub">El modelo combinado se está '
                    'evaluando con validación cruzada anidada; aparecerá aquí cuando esté congelado.</div></div>',
                    unsafe_allow_html=True)
    elif not hay_clinica:
        st.markdown('<div class="resultado"><div class="etiqueta">Imagen + datos clínicos</div>'
                    '<div class="p" style="color:#b6bfcc">—</div><div class="sub">Añade al menos un dato clínico '
                    'para ver la estimación combinada.</div></div>', unsafe_allow_html=True)
    else:
        d = fusion.decidir(pred.p_cruda, clinica)
        sub = d["submodelo"]
        info_b = fus_info["bandas"].get(sub, img_info["bandas"])
        tarjeta_resultado("Imagen + datos clínicos", d["probabilidad_pcr"], banda_de(d["probabilidad_pcr"], info_b),
                          d["umbral"], f"Submodelo usado: <b>{sub}</b> (según los datos disponibles).")
        barra_bandas(d["probabilidad_pcr"], info_b)

if etiqueta_real is not None:
    st.caption(f"Resultado real de esta paciente de entrenamiento: {'pCR' if etiqueta_real else 'no pCR'}.")

with st.expander("Detalle por corte y por red", icon=":material/stacked_bar_chart:"):
    df = pd.DataFrame({"corte": nombres, "probabilidad": [motor.calibrar(paquete, p) for p in pred.por_corte]})
    barras = alt.Chart(df).mark_bar(color=AZUL, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("corte:N", title="Corte", sort=None), y=alt.Y("probabilidad:Q", title="Probabilidad calibrada"),
        tooltip=["corte:N", alt.Tooltip("probabilidad:Q", format=".3f")])
    media = alt.Chart(pd.DataFrame({"y": [p_img]})).mark_rule(color=GRIS, strokeDash=[4, 3]).encode(y="y:Q")
    st.altair_chart(_base(alt.layer(barras, media).properties(title="Probabilidad por corte (línea: media de la paciente)"), 260),
                    width="stretch")
    st.caption("La predicción de la paciente es la media de sus cortes. Cortes muy distintos entre sí indican "
               "que la estimación depende de qué parte del tumor se mire.")

aviso("<b>Cómo interpretarlo.</b> La banda indica en qué tercio de las pacientes de la validación cruzada cae esta "
      "estimación y, al pasar el ratón, qué proporción de ese tercio alcanzó pCR. La zona intermedia es la de "
      "mayor incertidumbre. Ningún resultado debe modificar el tratamiento ni la cirugía previstos.", "clave")
pie()
