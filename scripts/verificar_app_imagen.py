#!/usr/bin/env python3
"""Prueba de integración de Streamlit con un ejemplo de train, sin evaluación."""
import json
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def main():
    at=AppTest.from_file(str(ROOT/'app_imagen.py')).run(timeout=30)
    assert len(at.exception)==0,at.exception
    assert at.button[0].disabled
    at.radio[0].set_value('Ejemplo de train').run(timeout=30)
    assert len(at.exception)==0,at.exception
    assert not at.button[0].disabled
    at.button[0].click().run(timeout=30)
    assert len(at.exception)==0,at.exception
    prediction=at.session_state['single_slice_prediction']
    assert prediction['models']==2 and not prediction['clinical_inputs_used']
    assert prediction['version']=='v008_cnn_cinetica_un_corte_20261025'
    assert len(at.metric)==1
    at.selectbox[0].select_index(1).run(timeout=30)
    try:
        at.session_state['single_slice_prediction']
    except KeyError:
        cleared=True
    else:
        cleared=False
    assert cleared,'Se conserva una predicción de otra imagen'
    result={'app_rendered':True,'example_prediction_passed':True,'result_cleared_after_input_change':True,
            'version':prediction['version'],'device':prediction['device'],'single_slice':True,'clinical_inputs':False}
    out=ROOT/'resultados/24_cierre_cinetico_un_corte_20261025/verificacion_app.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
