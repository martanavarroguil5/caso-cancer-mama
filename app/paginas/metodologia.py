import pandas as pd
import streamlit as st

from componentes import aviso, cabecera, pasos, pie, seccion, tabla

cabecera("Proyecto", "Datos y metodología",
         "Qué datos se usan, cómo se separan, cómo se evalúa y qué límites tiene el resultado.")

seccion("1", "Datos: BreastDCEDL MinCrop",
        "Resonancia dinámica con contraste (DCE-MRI) previa al tratamiento neoadyuvante, recortada a 256 × 256 "
        "alrededor del tumor, con unos diez cortes por paciente.")
tabla(pd.DataFrame([
    {"Cohorte": "I-SPY2", "Train": 784, "Test": 99, "pCR en train": "32 %", "Selección de cortes": "Máscara tumoral 3D"},
    {"Cohorte": "Duke", "Train": 209, "Test": 42, "pCR en train": "21 %", "Selección de cortes": "Cortes centrales (sin máscara completa)"},
    {"Cohorte": "I-SPY1", "Train": 104, "Test": 35, "pCR en train": "25 %", "Selección de cortes": "Máscara tumoral 3D"},
]))
st.caption("Las cohortes se procesaron de forma distinta y tienen prevalencias distintas: saber solo la cohorte ya "
           "da una AUC ≈ 0,55. Por eso la cohorte nunca es una entrada y se informa la AUC dentro de cada cohorte.")

seccion("2", "Fases y canales")
pasos([("PRE", "Antes del contraste"), ("EARLY", "Fase postcontraste temprana"), ("LATE", "Fase postcontraste tardía"),
       ("Realces", "EARLY−PRE, LATE−PRE y EARLY−LATE: la cinética del contraste")])
st.caption("Los canales no son colores: no hay normalización de ImageNet ni aumentos de color, y toda "
           "transformación geométrica se aplica a las tres fases a la vez.")

seccion("3", "Separación y evaluación")
pasos([
    ("5 folds por paciente", "Todos los cortes de una paciente van al mismo fold"),
    ("Fit / validación", "Dentro de cada fold de entrenamiento, un 15 % vigila las curvas"),
    ("Predicción fuera de fold", "Cada paciente se predice con redes que no la vieron"),
    ("Paciente", "Media de las probabilidades de sus cortes y de las redes"),
    ("Test único", "El test público se evalúa una vez, con el modelo congelado"),
])
aviso("<b>Reglas.</b> Nada del test ni de la validación privada interviene en la arquitectura, las épocas, la "
      "calibración, los umbrales ni las bandas. La comparación entre modelos usa un bootstrap pareado "
      "estratificado por fold y clase.", "clave")

seccion("4", "Límites")
st.markdown(
    "- **Señal de imagen débil:** AUC ≈ 0,58 con una CNN desde cero y ≈1.100 pacientes. Modelos grandes "
    "preentrenados publicados tampoco funcionan en Duke (AUC ≈ 0,54).\n"
    "- **Desarrollo adaptativo:** el train se ha usado en muchas pruebas; la confirmación honesta es el test "
    "y la validación privada del profesorado.\n"
    "- **Volumen tumoral:** en Duke se midió de otra forma que en I-SPY.\n"
    "- **Datos clínicos en la evaluación privada:** no vienen en <code>patients.csv</code>; si faltan, la web usa "
    "el submodelo correspondiente o la CNN sola.\n"
    "- **Uso:** educativo. No es un dispositivo médico ni puede cambiar decisiones de tratamiento.",
    unsafe_allow_html=True)

seccion("5", "Referencias")
st.markdown(
    "- Fridman et al. *BreastDCEDL: a comprehensive breast cancer DCE-MRI dataset*. arXiv 2506.12190.\n"
    "- Li, Yumer y Ramanan. *Budgeted Training: Rethinking Deep Neural Network Training Under Resource "
    "Constraints*. ICLR 2020.\n"
    "- Fletcher Mercaldo y Blume. *Missing data and prediction: the pattern submodel*. Biostatistics 2020.")
pie()
