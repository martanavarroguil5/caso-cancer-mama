"""Comprobación de inferencia v002: GUI, CPU, CLI y paquete congelado; solo train."""
import json
from pathlib import Path
import subprocess
import sys
import torch
from streamlit.testing.v1 import AppTest
from cancer_mama import entrenamiento as base
from cancer_mama.app_logic import InferenceEngine, discover_local_examples, uploads_from_paths, validate_phase_uploads
from cancer_mama.paths import DATA_DIR, MULTIMODAL_MANIFEST, PROJECT_ROOT

def main():
    torch.set_num_threads(4)
    # Compara CPU/GUI con la precisión FP32 de la evaluación del entrenamiento.
    # TF32 puede introducir pequeñas diferencias aun usando los mismos pesos.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert MULTIMODAL_MANIFEST.parent.name == 'v002_multimodal_clinico_20261007'
    examples = discover_local_examples(DATA_DIR)
    name = next(iter(examples))
    paths = examples[name]
    assert all('/train/' in str(p) for p in paths.values())
    sample = validate_phase_uploads(uploads_from_paths(paths))
    clinical = {'age':None, 'tum_vol':None, 'HR':None, 'HER2':None}
    cpu = InferenceEngine(MULTIMODAL_MANIFEST, device='cpu').predict(sample, clinical)
    direct = base.predecir_cortes(MULTIMODAL_MANIFEST, [[paths[p] for p in ('PRE','EARLY','LATE')]], clinical)
    assert abs(cpu.probability_raw - direct['probability_raw']) < 1e-6
    args = [sys.executable, '-B', str(PROJECT_ROOT/'modelos/versionar.py'), 'predecir', MULTIMODAL_MANIFEST.parent.name, '--']
    for p in ('PRE','EARLY','LATE'):
        args += ['--'+p.lower(), str(paths[p])]
    args += ['--edad', 'nan']
    frozen = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    assert abs(cpu.probability_raw - frozen['probability_raw']) < 1e-6
    assert abs(cpu.probability_calibrated - frozen['probability_calibrated']) < 1e-6
    assert abs(cpu.probability_calibrated - direct['probability_calibrated']) < 1e-6
    assert cpu.predicted_class == frozen['pCR'] == direct['pCR']
    app=AppTest.from_file(str(PROJECT_ROOT/'app.py')).run(timeout=30)
    assert not app.exception, list(app.exception)
    assert any('v002' in str(item.value) for item in app.caption)
    assert any('BCE ponderada' in str(item.value) and 'Platt posterior' in str(item.value) for item in app.markdown)
    app.radio[0].set_value('Usar ejemplo local').run(timeout=30)
    app.selectbox[0].set_value(name).run(timeout=30)
    assert not app.exception, list(app.exception)
    app.button[0].click().run(timeout=60)
    assert not app.exception, list(app.exception)
    assert not app.error, list(app.error)
    ui = app.session_state['prediction'][1]
    assert abs(cpu.probability_raw - ui.probability_raw) < 2e-6
    assert abs(cpu.probability_calibrated - ui.probability_calibrated) < 2e-6
    assert cpu.predicted_class == ui.predicted_class
    assert ui.threshold == .35433316934837994
    assert any('<div class="result-card">' in str(x.value) for x in app.markdown)
    app.number_input[0].set_value(50.).run(timeout=30)
    assert not app.exception, list(app.exception)
    assert not any('<div class="result-card">' in str(x.value) for x in app.markdown)
    result={'version': MULTIMODAL_MANIFEST.parent.name,
            'dataset_scope':'one development/train example, missing clinical inputs',
            'gpu_tf32_disabled_for_parity_check':True,
            'cpu_vs_cli_abs_difference':abs(cpu.probability_raw-direct['probability_raw']),
            'cpu_vs_frozen_abs_difference':abs(cpu.probability_raw-frozen['probability_raw']),
            'cpu_vs_gui_abs_difference':abs(cpu.probability_raw-ui.probability_raw),
            'platt_and_saved_threshold_verified':True,'stale_prediction_hidden_after_input_change':True,
            'test_opened':False}
    out=PROJECT_ROOT/'.codex_tmp/verificacion_multimodal/v002.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
