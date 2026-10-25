"""Demostración docente de la CNN cinética congelada para un único corte."""
from pathlib import Path
import json

import streamlit as st

from cancer_mama.app_logic import (PHASES,MAX_UPLOAD_BYTES,InputValidationError,UploadedPhase,
    discover_local_examples,enhancement_map,uploads_from_paths,validate_phase_uploads)
from cancer_mama.app_imagen_logic import DEFAULT_IMAGE_MANIFEST,SingleSliceEngine,image_key
from cancer_mama.entrenamiento import sha256
from cancer_mama.paths import DATA_DIR

st.set_page_config(page_title='BreastDCEDL · Un corte, tres fases',page_icon='◫',layout='wide')
st.markdown('''<style>
:root{--papel:#f5f7f8;--tinta:#183244;--placa:#101820;--contraste:#187d86;--borde:rgba(24,50,68,.14)}
.stApp{background:var(--papel);color:var(--tinta)}
.block-container{max-width:1232px;padding-top:2rem;padding-bottom:2rem}
h1,h2,h3{letter-spacing:-.025em;font-family:system-ui,sans-serif}
[data-testid="stMetricValue"]{font-variant-numeric:tabular-nums}
[data-testid="stImage"] img{background:var(--placa);border-radius:4px}
div.stButton>button[kind="primary"]{background:var(--contraste);border:0;color:white;border-radius:5px}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--borde)}
</style>''',unsafe_allow_html=True)

st.caption('BREASTDCEDL / DEMOSTRACIÓN DOCENTE / CNN 2D PROPIA')
st.title('Un corte, tres fases')
st.write('Carga PRE, EARLY y LATE del mismo corte para explorar una estimación de respuesta patológica completa.')
st.info('Uso educativo. Esta estimación no confirma pCR, no equivale a curación y no autoriza decisiones médicas.')


@st.cache_resource(show_spinner='Verificando pesos y cargando la CNN…')
def engine(path: str,checksum: str):
    return SingleSliceEngine(Path(path))


@st.cache_data(show_spinner=False)
def examples():return discover_local_examples(DATA_DIR,limit=8)


try:
    model=engine(str(DEFAULT_IMAGE_MANIFEST),sha256(DEFAULT_IMAGE_MANIFEST))
    card=model.card()
except (OSError,ValueError,KeyError,RuntimeError) as exc:
    st.error('El paquete de cierre todavía no está disponible o no supera la verificación de integridad.')
    st.caption(str(exc));st.stop()

with st.expander('Modelo, entrenamiento y evaluación',expanded=False):
    metric=card['internal_check'];lo,hi=card['auc_ci95']
    st.write(f"**{card['version']}** · {card['models']} redes · BCE {card['loss']}")
    st.write(f"Comprobación interna: {metric['n_patients']} pacientes, un corte por paciente. AUC {metric['roc_auc']:.4f}; IC95 % [{lo:.4f}, {hi:.4f}].")
    st.caption('Resultado interno de desarrollo. No es validación externa ni el AUC histórico de V007 con todos los cortes.')
    st.code(f"SHA-256 {card['checksum']}\nUmbral congelado {card['threshold']:.6f}",language=None)
    st.json(card['training'],expanded=False)

local=examples()
source=st.radio('Entrada',['Subir PNG']+(['Ejemplo de train'] if local else []),horizontal=True,key='image_source')
uploads={p:None for p in PHASES}
if source=='Ejemplo de train':
    selected=st.selectbox('Ejemplo educativo de train',list(local),key='image_example')
    uploads=uploads_from_paths(local[selected])
    st.caption('Ejemplo de desarrollo: su resultado no se interpreta como rendimiento en una paciente nueva.')
else:
    for col,phase in zip(st.columns(3),PHASES):
        with col:
            upload=st.file_uploader(phase,type=['png'],key='image_'+phase,
                help='PNG real, escala de grises de 8 bits, 256 × 256; máximo 5 MB. Nombre terminado en _'+phase+'.png.')
            if upload is not None:
                if upload.size>MAX_UPLOAD_BYTES:
                    st.error(f'{phase} supera el máximo de 5 MB.')
                else:uploads[phase]=UploadedPhase(upload.name,upload.getvalue())

sample=None
if all(uploads.values()):
    try:sample=validate_phase_uploads(uploads)
    except InputValidationError as exc:st.error(str(exc))
elif any(uploads.values()):st.caption('Completa las tres fases del mismo corte.')

key=image_key(sample) if sample else None
previous=st.session_state.get('single_slice_prediction')
if previous and (previous['input_sha256']!=key or previous['manifest_sha256']!=card['checksum']):
    del st.session_state['single_slice_prediction']

if sample:
    st.caption(f'CORTE VALIDADO · {sample.sample_id} · 256 × 256 · PRE → EARLY → LATE')
    for col,phase in zip(st.columns(4),[*PHASES,'Realce EARLY − PRE']):
        with col:
            st.markdown('**'+phase+'**')
            if phase in PHASES:st.image(sample.arrays[phase],clamp=True,width=256)
            else:st.image(enhancement_map(sample),width=256)

if st.button('Calcular estimación',type='primary',disabled=sample is None,key='single_slice_predict'):
    try:
        with st.spinner('Ejecutando el paquete congelado…'):
            st.session_state['single_slice_prediction']=model.predict(sample)
    except (ValueError,RuntimeError,FloatingPointError) as exc:st.error(f'No se pudo completar la inferencia: {exc}')

prediction=st.session_state.get('single_slice_prediction')
if prediction:
    st.divider()
    left,right=st.columns([1,2])
    with left:
        st.metric('Probabilidad estimada de pCR',f"{prediction['probability_pcr']:.1%}")
        st.write('Clase según umbral: **'+('pCR' if prediction['predicted_class'] else 'no pCR')+'**')
    with right:
        st.write(f"Umbral **{prediction['threshold']:.6f}** · dispositivo **{prediction['device']}** · {prediction['elapsed_ms']:.0f} ms")
        st.caption('La probabilidad expresa incertidumbre. Una clase pCR predicha puede ser un falso positivo y una clase no pCR puede ser un falso negativo.')
        st.download_button('Descargar registro de esta predicción',json.dumps(prediction,ensure_ascii=False,indent=2),
            file_name='prediccion_'+prediction['sample_id']+'.json',mime='application/json')

st.caption('Fuente: BreastDCEDL · Fridman y colaboradores · CC BY-NC 4.0. Imágenes desidentificadas para docencia e investigación.')
