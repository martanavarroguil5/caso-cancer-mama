import pandas as pd
import streamlit as st

from componentes import aviso, cabecera, pasos, pie, seccion

cabecera("Proyecto", "Datos y metodología",
         "Qué datos se usan, cómo se separan, cómo se evalúa y qué límites tiene el resultado.",
         meta=[("dataset", "BreastDCEDL MinCrop"), ("cohortes", "3"), ("train", "1.097"), ("test", "176")])

seccion("1", "Datos: BreastDCEDL",
        "Resonancia dinámica con contraste (DCE-MRI) previa al tratamiento neoadyuvante, recortada a 256 × 256 "
        "alrededor del tumor, con unos diez cortes por paciente.")
cohortes = pd.DataFrame([
    {"c": "I-SPY2", "train": 784, "test": 99, "pcr": 0.32, "sel": "Máscara tumoral 3D"},
    {"c": "Duke", "train": 209, "test": 42, "pcr": 0.21, "sel": "Cortes centrales (sin máscara completa)"},
    {"c": "I-SPY1", "train": 104, "test": 35, "pcr": 0.25, "sel": "Máscara tumoral 3D"},
])
maximo = (cohortes["train"] + cohortes["test"]).max()
filas = "".join(
    f'<div class="coh"><div class="nom">{f.c}<small>{f.sel}</small></div>'
    f'<div class="pista"><span class="tr" style="width:{100 * f.train / maximo:.1f}%">'
    f'<i style="width:{100 * f.pcr:.0f}%"></i></span><span class="te" style="width:{100 * f.test / maximo:.1f}%"></span></div>'
    f'<div class="cif">{f.train} <span>+ {f.test} test</span><b>{100 * f.pcr:.0f} % pCR</b></div></div>'
    for f in cohortes.itertuples())
st.markdown(f"""
<style>
.coh{{display:grid;grid-template-columns:240px 1fr 170px;gap:18px;align-items:center;padding:14px 0;border-bottom:1px solid var(--linea)}}
.coh .nom{{font-family:var(--serif);font-size:1.6rem;line-height:1.1}}
.coh .nom small{{display:block;font:400 .74rem var(--sans);color:var(--tenue);margin-top:.25rem}}
.coh .pista{{display:flex;gap:3px;height:16px}}
.coh .tr{{position:relative;background:#26303b;animation:crece 1.1s cubic-bezier(.2,.8,.2,1) both}}
.coh .tr i{{position:absolute;left:0;top:0;bottom:0;background:var(--realce)}}
.coh .te{{background:repeating-linear-gradient(135deg,#3a4552 0 3px,transparent 3px 6px);border:1px solid #3a4552;animation:crece 1.1s .2s cubic-bezier(.2,.8,.2,1) both}}
.coh .cif{{font:400 .82rem var(--mono);color:var(--tinta)}}
.coh .cif span{{color:var(--tenue)}}.coh .cif b{{display:block;font-weight:500;color:var(--realce);font-size:.74rem;margin-top:.2rem}}
.ley-coh{{display:flex;gap:1.4rem;font:400 .7rem var(--mono);color:var(--tenue);margin-top:.8rem;letter-spacing:.06em}}
.ley-coh i{{display:inline-block;width:12px;height:8px;margin-right:.45rem;vertical-align:0}}
.folds{{display:grid;grid-template-columns:110px repeat(5,1fr);gap:4px;margin:.6rem 0 .4rem}}
.folds .r{{font:400 .7rem var(--mono);color:var(--tenue);display:flex;align-items:center;letter-spacing:.06em}}
.folds .c{{height:34px;background:#1a222b;position:relative;transition:transform .15s}}
.folds .c.ev{{background:var(--realce)}}
.folds .c.de::after{{content:"";position:absolute;right:0;top:0;bottom:0;width:15%;background:repeating-linear-gradient(135deg,#5a9ec455 0 2px,transparent 2px 5px)}}
.folds .c:hover{{transform:scaleY(1.15)}}
.folds .h{{font:500 .66rem var(--mono);color:var(--tenue);text-align:center;letter-spacing:.12em}}
@media (max-width:760px){{.coh{{grid-template-columns:1fr}}}}
</style>
{filas}
<div class="ley-coh"><span><i style="background:#26303b"></i>train</span><span><i style="background:var(--realce)"></i>con pCR</span>
<span><i style="background:repeating-linear-gradient(135deg,#3a4552 0 3px,transparent 3px 6px);border:1px solid #3a4552"></i>test reservado</span></div>
""", unsafe_allow_html=True)
st.caption("Las cohortes se procesaron de forma distinta y tienen prevalencias distintas: saber solo la cohorte ya da "
           "una AUC ≈ 0,55. Por eso la cohorte nunca es una entrada y se informa la AUC dentro de cada cohorte.")

seccion("2", "Fases y canales")
pasos([("PRE", "Antes del contraste"), ("EARLY", "Fase postcontraste temprana"), ("LATE", "Fase postcontraste tardía"),
       ("Realces", "EARLY−PRE, LATE−PRE y EARLY−LATE: la cinética del contraste")])
st.caption("Los canales no son colores: no hay normalización de ImageNet ni aumentos de color, y toda transformación "
           "geométrica se aplica a las tres fases a la vez.")

seccion("3", "Separación y evaluación",
        "Cinco folds por paciente. En cada ronda, un fold se predice (naranja) con redes entrenadas en los otros "
        "cuatro; dentro de esos, un 15 % de pacientes (rayado) solo vigila las curvas.")
celdas = "".join(
    f'<div class="r">exterior {k + 1}</div>' + "".join(
        f'<div class="c {"ev" if j == k else "de"}" title="{"Fold evaluado" if j == k else "Desarrollo (fit + 15 % selección)"}"></div>'
        for j in range(5)) for k in range(5))
cab = '<div></div>' + "".join(f'<div class="h">FOLD {j + 1}</div>' for j in range(5))
st.markdown(f'<div class="folds">{cab}{celdas}</div>', unsafe_allow_html=True)
pasos([
    ("Por paciente", "Todos los cortes de una paciente van al mismo fold"),
    ("Fit / validación", "Dentro de cada fold de entrenamiento, un 15 % vigila las curvas"),
    ("Fuera de fold", "Cada paciente se predice con redes que no la vieron"),
    ("Paciente", "Media de las probabilidades de sus cortes y de las redes"),
    ("Test único", "El test público se evalúa una vez, con el modelo congelado"),
])
aviso("<b>Reglas.</b> Nada del test ni de la validación privada interviene en la arquitectura, las épocas, la "
      "calibración, los umbrales ni las bandas. La comparación entre modelos usa un bootstrap pareado estratificado "
      "por fold y clase.", "clave")

seccion("4", "Limitaciones")
st.markdown(
    "- **Señal de imagen débil:** AUC ≈ 0,58 con una CNN desde cero y ≈1.100 pacientes. Modelos grandes "
    "preentrenados publicados tampoco funcionan en Duke (AUC ≈ 0,54).\n"
    "- **Desarrollo adaptativo:** el train se ha usado en muchas pruebas; la confirmación honesta es el test y la "
    "validación privada del profesorado.\n"
    "- **Volumen tumoral:** en Duke se midió de otra forma que en I-SPY.\n"
    "- **Datos clínicos en la evaluación privada:** no vienen en <code>patients.csv</code>; si faltan, la web usa el "
    "submodelo correspondiente o la CNN sola.\n"
    "- **Uso:** educativo. No es un dispositivo médico ni puede cambiar decisiones de tratamiento.",
    unsafe_allow_html=True)

seccion("5", "Referencias")
st.markdown(
    "- Fridman et al. *BreastDCEDL: a comprehensive breast cancer DCE-MRI dataset*. arXiv 2506.12190.\n"
    "- Li, Yumer y Ramanan. *Budgeted Training: Rethinking Deep Neural Network Training Under Resource "
    "Constraints*. ICLR 2020.\n"
    "- Fletcher Mercaldo y Blume. *Missing data and prediction: the pattern submodel*. Biostatistics 2020.")
pie()
