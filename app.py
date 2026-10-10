"""Aplicación web educativa para inferencia pCR con DCE-MRI."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from cancer_mama.app_logic import (
    InferenceEngine,
    InputValidationError,
    PHASES,
    UploadedPhase,
    discover_local_examples,
    enhancement_map,
    read_model_card,
    prediction_key,
    uploads_from_paths,
    validate_phase_uploads,
)
from cancer_mama.paths import DATA_DIR, MULTIMODAL_MANIFEST


MANIFEST_PATH = MULTIMODAL_MANIFEST

st.set_page_config(
    page_title="Breast DCE · pCR Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root { --coral:#ff7e69; --mint:#68d7ca; --ink:#07111f; --panel:#0d1a2b; --muted:#91a3b8; }
.stApp { background:
  radial-gradient(circle at 78% 5%, rgba(255,126,105,.13), transparent 27rem),
  radial-gradient(circle at 8% 28%, rgba(104,215,202,.08), transparent 24rem), #07111f; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }
[data-testid="stSidebar"] { background:rgba(9,20,35,.92); border-right:1px solid rgba(255,255,255,.07); }
[data-testid="stSidebar"] > div { padding-top:1.3rem; }
.block-container { max-width:1220px; padding-top:2rem; padding-bottom:4rem; }
.hero { position:relative; overflow:hidden; padding:2.6rem 2.8rem; border:1px solid rgba(255,255,255,.09);
  border-radius:28px; background:linear-gradient(120deg,rgba(16,31,51,.96),rgba(12,25,43,.86));
  box-shadow:0 22px 70px rgba(0,0,0,.28); margin-bottom:1.4rem; }
.hero:after { content:""; position:absolute; width:260px; height:260px; right:-70px; top:-95px;
  border-radius:50%; border:55px solid rgba(255,126,105,.10); box-shadow:0 0 0 28px rgba(104,215,202,.035); }
.eyebrow { display:inline-flex; gap:.5rem; align-items:center; font-size:.72rem; letter-spacing:.14em;
  text-transform:uppercase; font-weight:700; color:#83ded4; margin-bottom:1rem; }
.dot { width:7px; height:7px; border-radius:50%; background:#68d7ca; box-shadow:0 0 14px #68d7ca; }
.hero h1 { max-width:760px; font-size:clamp(2.25rem,4.8vw,4.1rem); line-height:.98; margin:0 0 1rem;
  color:#f6f8fb; position:relative; z-index:1; }
.hero h1 span { color:#ff8a77; }
.hero p { max-width:650px; color:#aebdcd; font-size:1.03rem; line-height:1.65; margin:0; position:relative; z-index:1; }
.chips { display:flex; gap:.55rem; flex-wrap:wrap; margin-top:1.4rem; position:relative; z-index:1; }
.chip { font-size:.75rem; padding:.42rem .72rem; border-radius:999px; color:#d8e1eb;
  background:rgba(255,255,255,.055); border:1px solid rgba(255,255,255,.08); }
.section-kicker { color:#ff917f; text-transform:uppercase; letter-spacing:.13em; font-size:.7rem; font-weight:700; margin-top:1rem; }
.phase-label { display:flex; justify-content:space-between; align-items:center; color:#e9eef5; font-weight:700;
  margin:.3rem 0 .65rem; }
.phase-label span { color:#7f93a9; font-size:.68rem; font-weight:600; letter-spacing:.08em; }
.result-card { padding:1.6rem; border-radius:22px; background:linear-gradient(145deg,rgba(255,126,105,.13),rgba(104,215,202,.055));
  border:1px solid rgba(255,143,124,.25); min-height:212px; }
.result-label { color:#91a3b8; font-size:.74rem; text-transform:uppercase; letter-spacing:.12em; font-weight:700; }
.result-value { font-family:'Manrope',sans-serif; font-size:3.35rem; line-height:1; font-weight:800; color:#fff; margin:.7rem 0 .35rem; }
.result-class { display:inline-block; padding:.38rem .68rem; border-radius:9px; background:rgba(255,126,105,.18);
  color:#ffad9f; font-size:.8rem; font-weight:700; }
.meter { height:8px; border-radius:99px; background:rgba(255,255,255,.08); overflow:hidden; margin:1.25rem 0 .45rem; }
.meter > div { height:100%; border-radius:99px; background:linear-gradient(90deg,#68d7ca,#ffd28a,#ff7e69); }
.meter-meta { display:flex; justify-content:space-between; color:#8193a8; font-size:.72rem; }
.soft-card { padding:1.1rem 1.2rem; border-radius:17px; border:1px solid rgba(255,255,255,.075);
  background:rgba(255,255,255,.028); color:#9fb0c3; font-size:.86rem; line-height:1.55; }
.model-id { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; color:#83ded4; font-size:.76rem; }
div[data-testid="stFileUploader"] { background:rgba(255,255,255,.025); border-radius:16px; padding:.25rem .55rem; }
div[data-testid="stMetric"] { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07);
  padding:.85rem 1rem; border-radius:16px; }
.stButton > button[kind="primary"] { border:0; background:linear-gradient(100deg,#ff745f,#ff947f); color:#0b1421;
  font-weight:800; box-shadow:0 10px 28px rgba(255,116,95,.22); min-height:3rem; }
.stButton > button[kind="primary"]:hover { color:#07111f; transform:translateY(-1px); }
.footer-note { color:#6f8298; font-size:.73rem; text-align:center; margin-top:2.5rem; }
@media (max-width:700px) { .hero { padding:1.8rem 1.45rem; border-radius:22px; } .block-container { padding-top:1rem; } }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Verificando y cargando el ensemble…")
def load_engine(path: str) -> InferenceEngine:
    return InferenceEngine(Path(path))


@st.cache_data(show_spinner=False)
def local_examples() -> dict[str, dict[str, Path]]:
    return discover_local_examples(DATA_DIR)


def uploaded_phase(upload) -> UploadedPhase | None:
    if upload is None:
        return None
    return UploadedPhase(name=upload.name, data=upload.getvalue())


st.markdown(
    """
<section class="hero">
  <div class="eyebrow"><span class="dot"></span> Inteligencia artificial aplicada a DCE-MRI</div>
  <h1>Explora la respuesta <span>pCR</span> antes del tratamiento.</h1>
  <p>Una demostración educativa que combina las fases PRE, EARLY y LATE con edad,
  volumen tumoral, HR y HER2 para estimar la probabilidad de respuesta patológica completa.</p>
  <div class="chips"><span class="chip">CNN 2D propia</span><span class="chip">3 fases alineadas</span>
  <span class="chip">Ensemble de 10 modelos</span><span class="chip">Inferencia reproducible</span></div>
</section>
""",
    unsafe_allow_html=True,
)

try:
    model_card = read_model_card(MANIFEST_PATH)
    model_card_error = None
except Exception as exc:
    model_card = None
    model_card_error = str(exc)

with st.sidebar:
    st.markdown("### Breast DCE")
    st.caption("Panel de inferencia educativa")
    st.divider()
    st.markdown("**Estado del sistema**")
    if model_card_error:
        st.error("Manifiesto no disponible")
        st.caption(model_card_error)
    elif model_card and model_card.weights_ready:
        st.success("Modelo listo")
        st.caption(f"{model_card.model_count} pesos verificados al ejecutar la inferencia")
    else:
        st.warning("Faltan los pesos finales")
        st.caption("Faltan archivos de la versión multimodal conservada en este proyecto.")
    if model_card:
        st.markdown("**Versión del modelo**")
        st.caption("v002 · Multimodal clínico")
        st.markdown(f'<div class="model-id">sha256:{model_card.checksum[:12]}…</div>', unsafe_allow_html=True)
        st.caption(f"Umbral congelado: {model_card.threshold:.3f}  ·  Agregación: {model_card.aggregation}")
    st.divider()
    st.markdown("**Flujo de trabajo**")
    st.caption("01 · Carga las tres fases\n\n02 · Revisa el realce\n\n03 · Ejecuta la estimación")
    st.divider()
    st.caption("© BreastDCEDL · CC BY-NC 4.0")

st.markdown('<div class="section-kicker">01 · Entrada del estudio</div>', unsafe_allow_html=True)
st.subheader("Carga un corte DCE-MRI")
st.caption("Las tres imágenes deben corresponder al mismo corte, en PNG monocromo de 256×256 px.")

examples = local_examples()
source_options = ["Subir imágenes"] + (["Usar ejemplo local"] if examples else [])
source = st.radio("Origen de las imágenes", source_options, horizontal=True, label_visibility="collapsed")

uploads = None
if source == "Subir imágenes":
    upload_columns = st.columns(3)
    selected = {}
    help_text = {"PRE": "Antes del contraste", "EARLY": "Postcontraste temprana", "LATE": "Postcontraste tardía"}
    for column, phase in zip(upload_columns, PHASES):
        with column:
            st.markdown(f'<div class="phase-label">{phase}<span>{help_text[phase]}</span></div>', unsafe_allow_html=True)
            selected[phase] = st.file_uploader(
                f"Archivo {phase}", type=["png"], key=f"upload_{phase}", label_visibility="collapsed"
            )
    if any(value is not None for value in selected.values()):
        uploads = {phase: uploaded_phase(selected[phase]) for phase in PHASES}
else:
    example_id = st.selectbox("Ejemplo disponible en este equipo", list(examples))
    uploads = uploads_from_paths(examples[example_id])

sample = None
if uploads is not None:
    try:
        sample = validate_phase_uploads(uploads)
    except InputValidationError as exc:
        st.error(str(exc), icon="⚠️")

if sample:
    st.success(f"Estudio validado · paciente {sample.patient_id} · corte {sample.sample_id.rsplit('_z', 1)[1]}", icon="✅")
    st.markdown('<div class="section-kicker">02 · Exploración visual</div>', unsafe_allow_html=True)
    st.subheader("Fases y patrón de realce")
    image_columns = st.columns(4)
    subtitles = {"PRE": "Referencia", "EARLY": "Captación temprana", "LATE": "Evolución tardía"}
    for column, phase in zip(image_columns[:3], PHASES):
        with column:
            st.markdown(f'<div class="phase-label">{phase}<span>{subtitles[phase]}</span></div>', unsafe_allow_html=True)
            st.image(sample.arrays[phase], clamp=True, use_container_width=True)
    with image_columns[3]:
        st.markdown('<div class="phase-label">REALCE<span>EARLY − PRE</span></div>', unsafe_allow_html=True)
        st.image(enhancement_map(sample), use_container_width=True)
        st.caption("Coral: aumento · Turquesa: descenso")

    st.subheader("Datos clínicos de la paciente")
    st.caption("Introduce los datos correspondientes a estas imágenes. Los desconocidos se imputan con las estadísticas guardadas de cada modelo.")
    clinical_columns = st.columns(4)
    clinical = {}
    with clinical_columns[0]:
        clinical["age"] = st.number_input("Edad (años)", min_value=0.0, value=None, step=1.0)
    with clinical_columns[1]:
        clinical["tum_vol"] = st.number_input("Volumen tumoral", min_value=0.0, value=None, step=0.1)
    for column, name in zip(clinical_columns[2:], ("HR", "HER2")):
        with column:
            choice = st.selectbox(name, ("Desconocido", "Negativo", "Positivo"))
            clinical[name] = {"Desconocido": None, "Negativo": 0.0, "Positivo": 1.0}[choice]
    current_key = prediction_key(sample, clinical)
    st.caption("La estimación usa un único corte; la calibración de desarrollo se ajustó con varios cortes por paciente.")

    st.markdown('<div class="section-kicker">03 · Estimación</div>', unsafe_allow_html=True)
    st.subheader("Resultado del ensemble")
    can_predict = bool(model_card and model_card.weights_ready)
    predict_clicked = st.button("Ejecutar predicción", type="primary", use_container_width=True, disabled=not can_predict)
    if not can_predict:
        st.info("La interfaz ya está operativa, pero esta copia no contiene los diez pesos finales necesarios para calcular una predicción real.")
    if predict_clicked:
        try:
            engine = load_engine(str(MANIFEST_PATH))
            with st.spinner("Analizando las tres fases con el ensemble…"):
                st.session_state["prediction"] = (current_key, engine.predict(sample, clinical))
        except Exception as exc:
            st.error(f"No se pudo ejecutar la inferencia: {exc}")

    saved = st.session_state.get("prediction")
    if saved and saved[0] == current_key:
        prediction = saved[1]
        probability = prediction.probability_calibrated
        label = "pCR" if prediction.predicted_class else "no pCR"
        left, right = st.columns([1.05, 1.25])
        with left:
            st.markdown(
                f"""
<div class="result-card">
  <div class="result-label">Probabilidad calibrada de pCR</div>
  <div class="result-value">{probability:.1%}</div>
  <span class="result-class">Clase estimada · {label}</span>
  <div class="meter"><div style="width:{probability * 100:.2f}%"></div></div>
  <div class="meter-meta"><span>0%</span><span>umbral {prediction.threshold:.1%}</span><span>100%</span></div>
</div>
""",
                unsafe_allow_html=True,
            )
        with right:
            metrics = st.columns(2)
            metrics[0].metric("Probabilidad sin calibrar", f"{prediction.probability_raw:.1%}")
            metrics[1].metric("Tiempo de inferencia", f"{prediction.elapsed_ms:.0f} ms")
            metrics[0].metric("Modelos", str(prediction.model_count))
            metrics[1].metric("Dispositivo", prediction.device.upper())
            st.markdown(
                f'<div class="soft-card">La dispersión entre modelos es <strong>{prediction.model_spread:.3f}</strong>. '
                "Es una medida descriptiva del ensemble, no un intervalo clínico de incertidumbre.</div>",
                unsafe_allow_html=True,
            )

st.divider()
details_left, details_right = st.columns(2)
with details_left:
    with st.expander("Arquitectura y entrenamiento"):
        st.markdown(
            """
- **Entrada:** PRE, EARLY y LATE apiladas como `[3, 256, 256]`, edad, volumen tumoral, HR y HER2.
- **Arquitectura:** CNN 2D propia con pooling intermedio y rama clínica.
- **Ensemble:** 2 semillas × 5 folds, diez modelos.
- **Pérdida seleccionada:** BCE ponderada.
- **Inferencia:** `model.eval()`, sin gradientes y escalado `uint8 / 255`.
"""
        )
with details_right:
    with st.expander("Métricas y trazabilidad"):
        if model_card and model_card.internal_metrics:
            metrics = model_card.internal_metrics
            st.markdown(
                f"**Desarrollo OOF**  \nROC-AUC: `{metrics.get('roc_auc', float('nan')):.3f}`  ·  "
                f"AP: `{metrics.get('average_precision', float('nan')):.3f}`  ·  "
                f"Brier: `{metrics.get('brier', float('nan')):.3f}`"
            )
        else:
            st.caption("Las métricas internas se mostrarán al cargar un manifiesto compatible.")
        if model_card:
            st.markdown(f"**Checksum**  \n`{model_card.checksum}`")

st.warning(
    "**Uso exclusivamente educativo.** Esta estimación no es un diagnóstico, no sustituye la anatomía patológica y no debe utilizarse para decidir tratamientos ni evitar una cirugía.",
    icon="🛡️",
)
st.markdown(
    '<div class="footer-note">Breast DCE pCR Explorer · Proyecto docente de Inteligencia Artificial y Deep Learning</div>',
    unsafe_allow_html=True,
)
