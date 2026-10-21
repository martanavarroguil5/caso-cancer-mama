"""Verifica versiones, procedencia e inferencia aislada sin acceder al dataset."""
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'resultados/04_entrenamiento/ajuste_cnn_cinetica_20261021'
IDENTIFIER = 'v007_cnn_cinetica_multiescala_20261021'
ARCHIVE = ROOT / 'modelos/versiones'


def main():
    spec = importlib.util.spec_from_file_location('versionar', ROOT / 'modelos/versionar.py')
    versions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(versions)
    baseline = json.loads((RUN / 'versiones_antes.json').read_text())
    old_names = {Path(relative).parts[0] for relative in baseline}
    actual = {path.relative_to(ARCHIVE).as_posix(): versions.sha256(path)
              for name in old_names for path in (ARCHIVE / name).rglob('*')
              if path.is_file() and '__pycache__' not in path.parts}
    assert actual == baseline, 'Las versiones anteriores han cambiado'
    source_hashes = json.loads((OUT / 'protocolo.json').read_text())['source_sha256']
    for module, checksum in source_hashes.items():
        assert versions.sha256(ROOT / 'src/cancer_mama' / (module + '.py')) == checksum
        assert versions.sha256(ARCHIVE / IDENTIFIER / 'codigo/src/cancer_mama' / (module + '.py')) == checksum
    current_source = (ROOT / 'src/cancer_mama/entrenamiento_cinetico.py').read_bytes()
    original_source = subprocess.run(['git', 'show', '39bd951:src/cancer_mama/entrenamiento_cinetico.py'],
                                     cwd=ROOT, capture_output=True, check=True).stdout
    assert current_source == original_source, 'La arquitectura original ha cambiado'
    from cancer_mama.paths import MULTIMODAL_MANIFEST
    expected_active = ARCHIVE / 'v002_multimodal_clinico_20261007/manifiesto.json'
    assert MULTIMODAL_MANIFEST == expected_active
    assert versions.sha256(expected_active) == '59a540c8949827becca7e5e9bf70698cc2faed94dd63b2c1872f47a2854e9292'
    all_versions = [versions.verify_version(row['id']) for row in versions.read_versions()]
    assert len(all_versions) == 7
    source_manifest = OUT / 'manifiesto_candidato.json'
    frozen_manifest = ARCHIVE / IDENTIFIER / 'manifiesto.json'
    source_check = versions.run_verification(versions.training_code(ROOT, json.loads(source_manifest.read_text())), source_manifest)
    assert source_check == all_versions[-1]['runtime']
    from cancer_mama.inferencia_imagen import predecir_cortes
    torch.set_num_threads(4)
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = ''
    environment['OMP_NUM_THREADS'] = '2'
    with tempfile.TemporaryDirectory(prefix='verificacion-v007-') as directory:
        sandbox = Path(directory)
        isolated = sandbox / IDENTIFIER
        shutil.copytree(frozen_manifest.parent, isolated, ignore=shutil.ignore_patterns('__pycache__'))
        assert not (sandbox / 'breastdcedl').exists()
        environment['PYTHONPATH'] = str(isolated / 'codigo/src')
        generator = np.random.default_rng(20261021)
        triples = []
        for index in range(3):
            triple = []
            for phase in ('PRE', 'EARLY', 'LATE'):
                path = sandbox / f'sintetica_z{index:03d}_{phase}.png'
                Image.fromarray(generator.integers(0, 256, (256, 256), dtype=np.uint8)).save(path)
                triple.append(path)
            triples.append(triple)
        source_prediction = predecir_cortes(source_manifest, triples)
        arguments = []
        for index, phase in enumerate(('pre', 'early', 'late')):
            arguments += ['--' + phase, *[str(triple[index]) for triple in triples]]
        result = subprocess.run([sys.executable, '-B', '-m', 'cancer_mama.inferencia_imagen', 'predecir',
                                 '--manifest', str(isolated / 'manifiesto.json'), *arguments],
                                cwd=sandbox, env=environment, capture_output=True, text=True, check=True)
        isolated_prediction = json.loads(result.stdout)
        differences = {key: abs(source_prediction[key] - isolated_prediction[key])
                       for key in ('probability_raw', 'probability_calibrated', 'threshold')}
        assert max(differences.values()) <= 2e-6
        assert source_prediction['pCR'] == isolated_prediction['pCR']
        assert source_prediction['models'] == isolated_prediction['models'] == 10
        assert source_prediction['n_slices'] == isolated_prediction['n_slices'] == 3
    weights = sorted((frozen_manifest.parent / 'pesos').glob('*.pt'))
    assert len(weights) == 10
    for weight in weights:
        artifact = torch.load(weight, map_location='cpu', weights_only=True)
        assert set(artifact) == {'schema_version', 'model_config', 'state_dict'}
        assert weight.stat().st_size < 100_000_000
    documents = [ROOT / 'README.md', ROOT / 'modelos/README.md',
                 ROOT / 'docs/mejoras_cnn/PROTOCOLO_CNN_CINETICA.md',
                 ROOT / 'docs/EXPERIMENTOS_DESCARTADOS.md', OUT / 'INFORME_RESULTADOS.md']
    missing = []
    linked = 0
    for document in documents:
        body = re.sub(r'```.*?```', '', document.read_text(), flags=re.S)
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', body):
            target = target.strip('<>').split('#')[0]
            if not target or re.match(r'^[a-zA-Z]+:', target):
                continue
            path = (document.parent / target).resolve()
            linked += 1
            if path == OUT / 'verificacion_publicacion.json':
                continue  # Este archivo se escribe al completar todas las comprobaciones.
            if not path.exists():
                missing.append([str(document.relative_to(ROOT)), target])
    assert not missing, missing
    test_log = ROOT / '.codex_tmp/tests_cierre_cinetico_20261021.log'
    assert 'Ran 57 tests' in test_log.read_text() and test_log.read_text().rstrip().endswith('OK')
    subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
    report = {'verified_at': datetime.now(ZoneInfo('Europe/Madrid')).isoformat(),
              'version': IDENTIFIER, 'tests': {'passed': 57, 'log_sha256': versions.sha256(test_log)},
              'source_sha256': source_hashes, 'architecture_original_commit': '39bd951',
              'architecture_source_unchanged': True, 'previous_version_files_unchanged': len(baseline),
              'versions': all_versions, 'active_model': str(expected_active.relative_to(ROOT)),
              'active_manifest_sha256': versions.sha256(expected_active),
              'isolated_inference': {'inputs': 'tres cortes sintéticos PNG de tres fases; sin dataset',
                                     'source': source_prediction, 'isolated': isolated_prediction,
                                     'absolute_differences': differences, 'tolerance': 2e-6},
              'inference_weights': {'count': len(weights), 'bytes': sum(p.stat().st_size for p in weights),
                                    'no_optimizer_or_rng': True},
              'markdown_links_checked': linked, 'missing_links': missing,
              'visual_review': {'figures': 4, 'legibility_and_layout_checked': True},
              'test_images_loaded_for_publication': 0,
              'no_additional_training': True}
    (OUT / 'verificacion_publicacion.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('version', 'tests', 'previous_version_files_unchanged',
                      'inference_weights', 'markdown_links_checked', 'missing_links')}, ensure_ascii=False, indent=2))
    print('Diferencias de inferencia:', differences)


if __name__ == '__main__':
    main()
