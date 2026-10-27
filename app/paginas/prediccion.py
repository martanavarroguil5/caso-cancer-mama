import numpy as np
import pandas as pd
import streamlit as st

import estado
import motor
from componentes import (COHORTES, aviso, cabecera, escapar, lectura, lectura_vacia, num, pct, pie, recurso,
                         red_neuronal, seccion, visor)

cabecera("Predicción", "Predecir <em>una paciente</em>",
         "Sube las tres fases de uno o varios cortes de la misma paciente, o elige un ejemplo. Los datos clínicos "
         "son opcionales: si los añades, aparece también la estimación combinada.")

img_info = recurso("modelo_imagen")
fus_info = recurso("modelo_fusion")
paquete = estado.paquete_imagen()
fusion = estado.paquete_fusion()

# ---------------------------------------------------------------- 1. imágenes
seccion("1", "Imágenes", "PNG en gris de 256 × 256 por fase. El nombre debe terminar en <code>_PRE</code>, "
        "<code>_EARLY</code> o <code>_LATE</code>; los cortes se agrupan por el resto del nombre.")
origen = st.segmented_control("Origen", ["Paciente de ejemplo", "Subir PNG"], default="Paciente de ejemplo",
                              label_visibility="collapsed")
pixeles, nombres, clinica_ejemplo, etiqueta_real, titulo_visor = None, [], {}, None, "subida"
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
    ej = estado.ejemplos()
    pid = st.selectbox("Paciente de ejemplo (train)", ej.index, format_func=lambda p: (
        f"{p} · {COHORTES.get(ej.loc[p, 'cohorte'], ej.loc[p, 'cohorte'])} · {ej.loc[p, 'cortes']} cortes"))
    pixeles, nombres = estado.cortes_ejemplo(pid)
    f = ej.loc[pid]
    clinica_ejemplo = {"edad": f["edad"], "volumen": float(np.expm1(f["log_volumen"])), "HR": f["HR"], "HER2": f["HER2"]}
    etiqueta_real = int(f["pCR"])
    titulo_visor = f"{COHORTES.get(f['cohorte'], f['cohorte'])} · {escapar(pid)}"
    aviso("Estas pacientes son de <b>entrenamiento</b>: el modelo final las vio, así que su predicción es optimista "
          "y no sirve para evaluar. Solo ilustran el funcionamiento.", "atencion")

pred = None
if pixeles is not None:
    pred = estado.prediccion(pixeles, tuple(nombres))
    p_cortes = tuple(motor.calibrar(paquete, p) for p in pred.por_corte)
    visor(estado.visor(pixeles, tuple(nombres), p_cortes), titulo_visor, alto=500,
          inicial=int(np.argsort(p_cortes)[len(p_cortes) // 2]))
    st.caption(f"{len(nombres)} corte(s) válidos. Las ventanas discontinuas son las tres vistas centrales que recibe "
               "la red (256, 128 y 64 px). La probabilidad de cada corte ya está calibrada.")

# ---------------------------------------------------------------- 2. clínica
seccion("2", "Datos clínicos <em>(opcional)</em>",
        "Deja en «Desconocido» lo que no tengas: el modelo usa el submodelo entrenado con exactamente los datos "
        "disponibles, sin inventar los que faltan.")


def _opcion(valor):
    return "Desconocido" if valor is None or pd.isna(valor) else ("Positivo" if valor == 1 else "Negativo")


with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1, 1, 1.45, 1.45], gap="medium")
    edad_conocida = c1.toggle("Edad conocida", value=bool(clinica_ejemplo) and not pd.isna(clinica_ejemplo.get("edad")))
    edad = c1.number_input("Edad (años)", 18.0, 100.0, float(clinica_ejemplo.get("edad") or 50.0), 1.0,
                           disabled=not edad_conocida)
    vol_conocido = c2.toggle("Volumen conocido", value=bool(clinica_ejemplo) and not pd.isna(clinica_ejemplo.get("volumen")))
    volumen = c2.number_input("Volumen tumoral (cm³)", 0.0, 1000.0, float(clinica_ejemplo.get("volumen") or 10.0), 0.5,
                              disabled=not vol_conocido, help="Mismas unidades que tum_vol en patients.csv (≈ cm³).")
    opciones = ["Desconocido", "Positivo", "Negativo"]
    hr = c3.segmented_control("Receptores hormonales (HR)", opciones,
                              default=_opcion(clinica_ejemplo.get("HR")), key=f"hr_{titulo_visor}") or "Desconocido"
    her2 = c4.segmented_control("HER2", opciones, default=_opcion(clinica_ejemplo.get("HER2")),
                                key=f"her2_{titulo_visor}") or "Desconocido"
a_num = {"Desconocido": None, "Positivo": 1.0, "Negativo": 0.0}
clinica = {"edad": edad if edad_conocida else None, "volumen": volumen if vol_conocido else None,
           "HR": a_num[hr], "HER2": a_num[her2]}
hay_clinica = any(v is not None for v in clinica.values())

# ---------------------------------------------------------------- 3. lectura
seccion("3", "Lectura")
if pred is None:
    aviso("Carga imágenes o elige una paciente de ejemplo para ver la predicción.", "info")
    pie()
    st.stop()

p_img = motor.calibrar(paquete, pred.p_cruda)
umbral_img = paquete.info["politica"]["umbral"]
dispersion = float(np.std([motor.calibrar(paquete, p) for p in pred.por_red]))
col_img, col_fus = st.columns(2, gap="medium")
with col_img:
    lectura("Solo imagen · CNN", p_img, img_info["bandas"], umbral_img,
            f"{len(nombres)} corte(s) × 5 redes × 4 reflexiones · desacuerdo entre redes ± {num(dispersion, 3)}")
with col_fus:
    if fusion is None or fus_info is None:
        lectura_vacia("Imagen + clínica", "El modelo combinado aún no está congelado.")
    elif not hay_clinica:
        lectura_vacia("Imagen + clínica", "Añade al menos un dato clínico para ver la estimación combinada.")
    else:
        d = fusion.decidir(pred.p_cruda, clinica)
        sub = d["submodelo"]
        info_b = fus_info["bandas"].get(sub, img_info["bandas"])
        cambio = 100 * (d["probabilidad_pcr"] - p_img)
        lectura("Imagen + clínica", d["probabilidad_pcr"], info_b, d["umbral"],
                f"submodelo {escapar(sub)} · según los datos disponibles",
                extra=f" Con la clínica, la estimación cambia <b>{cambio:+.1f} puntos</b>.".replace(".", ",", 1))

if etiqueta_real is not None:
    st.caption(f"Resultado real de esta paciente de entrenamiento: {'pCR' if etiqueta_real else 'no pCR'}.")

# ---------------------------------------------------------------- 4. dentro de la red
seccion("4", "Dentro de la <em>red</em>",
        "El corte elegido recorre una de las cinco redes. Cada miniatura es un mapa de activación real; las líneas "
        "son los pesos efectivos entre los filtros que más pesan en la salida. Pasa el ratón por cualquier nodo.")
k = st.select_slider("Corte que se propaga", options=list(range(len(nombres))), format_func=lambda i: (
    f"{nombres[i]} · p {pct(p_cortes[i], 1)}"), value=int(np.argsort(p_cortes)[len(p_cortes) // 2])) \
    if len(nombres) > 1 else 0
red_neuronal(estado.recorrido(pixeles[k]))

aviso("<b>Cómo interpretarlo.</b> La banda indica en qué tercio de las pacientes de la validación cruzada cae la "
      "estimación y qué proporción de ese tercio alcanzó pCR (pasa el ratón por la regla). La zona intermedia es la "
      "de mayor incertidumbre. Ningún resultado debe modificar el tratamiento ni la cirugía previstos.", "clave")
pie()
