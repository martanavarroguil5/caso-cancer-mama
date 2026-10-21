"""Conserva dieciséis exportaciones del piloto, sin crear versión de producción."""
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
import torch

from cancer_mama import entrenamiento as base
from cancer_mama import entrenamiento_antioverfit as pilot
from cancer_mama import entrenamiento_imagen as image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = pilot.DEFAULT_OUTPUT
IDENTIFIER = 'p001_antioverfit_20261021'


def write(path, value):
    base.json_write(path, value)


def main():
    assert json.loads((RUN / 'estado_piloto.json').read_text())['status'] == 'complete'
    baseline = json.loads((RUN / 'versiones_antes.json').read_text())
    version_root = ROOT / 'modelos/versiones'
    assert baseline == {p.relative_to(version_root).as_posix(): base.sha256(p)
                        for p in version_root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    destination = ROOT / 'modelos/pilotos' / IDENTIFIER
    assert not destination.exists(), 'No sobrescribir un paquete ya conservado'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.piloto-', dir=destination.parent) as temporary:
        folder = Path(temporary)
        models = []
        for arm in pilot.ARMS:
            for loss in image.LOSSES:
                for seed in image.SEEDS:
                    for fold in pilot.PILOT_FOLDS:
                        source = pilot.run_path(RUN, arm, loss, seed, fold)
                        config = json.loads((source / 'config.json').read_text())
                        summary = json.loads((source / 'summary.json').read_text())
                        relative = f'pesos/{arm}_{loss}_semilla_{seed}_fold_{fold}.pt'
                        target = folder / relative
                        target.parent.mkdir(exist_ok=True)
                        shutil.copy2(source / 'inference.pt', target)
                        assert base.sha256(target) == summary['inference_sha256']
                        artifact = torch.load(target, map_location='cpu', weights_only=True)
                        assert set(artifact) == {'schema_version', 'model_config', 'state_dict'}
                        models.append({'arm': arm, 'loss': loss, 'seed': seed, 'fold': fold,
                                       'path': relative, 'sha256': base.sha256(target),
                                       'model_config': artifact['model_config'],
                                       'best_epoch': summary['best_epoch']})
                        evidence = folder / f'evidencia/ejecuciones/{arm}_{loss}_semilla_{seed}_fold_{fold}'
                        evidence.mkdir(parents=True)
                        for name in ('config.json', 'summary.json', 'history.csv', 'environment.json'):
                            shutil.copy2(source / name, evidence / name)
        sources = folder / 'codigo/src/cancer_mama'
        sources.mkdir(parents=True)
        for path in (ROOT / 'src/cancer_mama').glob('*.py'):
            shutil.copy2(path, sources / path.name)
        for name in ('requirements.txt', 'pyproject.toml'):
            shutil.copy2(ROOT / name, folder / 'codigo' / name)
        shutil.copy2(OUT / 'inferencia_piloto.py', folder / 'inferencia.py')
        evidence = folder / 'evidencia/informe'
        evidence.mkdir(parents=True)
        for path in OUT.iterdir():
            if path.suffix in {'.md', '.json', '.csv', '.png'}:
                shutil.copy2(path, evidence / path.name)
        record = {'schema_version': 1, 'id': IDENTIFIER, 'kind': 'two_fold_pilot',
                  'status': 'experimental_no_adoptado', 'recommended_for_deployment': False,
                  'model_class': 'CNNTemporalSimple', 'models': models,
                  'folds': [0, 3], 'seeds': [42, 2026], 'aggregation': 'mean',
                  'clinical_required': False, 'phases': ['PRE', 'EARLY', 'LATE'],
                  'shape': [3, 256, 256], 'scaling': 'uint8/255', 'threshold': .5,
                  'calibration': 'none', 'test_images_loaded': 0,
                  'prediction_scope': 'four-model ensemble for future inputs; OOF pilot uses two seeds from each excluded fold',
                  'metrics': json.loads((OUT / 'evaluacion_recalculada.json').read_text())}
        write(folder / 'manifiesto.json', record)
        (folder / 'manifiesto.json.sha256').write_text(base.sha256(folder / 'manifiesto.json')+'\n')
        (folder / 'README.md').write_text(
            '# Paquete experimental del piloto anti-sobreajuste\n\n'
            'Dieciséis pesos: control/antioverfit × normal/weighted × semillas 42/2026 × folds 0/3. '
            'No es una versión final de cinco folds, no cambia V002 y no se integra en la aplicación.\n\n'
            'Desde esta carpeta, con las dependencias instaladas:\n\n'
            '```bash\nPYTHONPATH=codigo/src python -B inferencia.py verificar\n'
            'PYTHONPATH=codigo/src python -B inferencia.py predecir --arm antioverfit --loss weighted '
            '--pre paciente_z000_PRE.png --early paciente_z000_EARLY.png --late paciente_z000_LATE.png\n```\n\n'
            'La inferencia promedia cuatro modelos y cortes; usa probabilidad cruda y umbral diagnóstico '
            '0,5, sin calibración. Las OOF del informe usan únicamente las dos semillas del fold que '
            'excluye a cada paciente: no equivalen a la inferencia conjunta del paquete sobre train.\n')
        specification = importlib.util.spec_from_file_location('pilot_inference', folder / 'inferencia.py')
        runtime = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(runtime)
        torch.set_num_threads(2)
        verification = runtime.verify(folder / 'manifiesto.json')
        with tempfile.TemporaryDirectory(prefix='piloto-aislado-') as isolated_directory:
            sandbox = Path(isolated_directory)
            isolated = sandbox / IDENTIFIER
            shutil.copytree(folder, isolated)
            assert not (sandbox / 'breastdcedl').exists()
            generator = np.random.default_rng(20261021)
            triples = []
            for z in range(2):
                triple = []
                for phase in ('PRE', 'EARLY', 'LATE'):
                    path = sandbox / f'sintetica_z{z:03d}_{phase}.png'
                    Image.fromarray(generator.integers(0, 256, (256, 256), dtype=np.uint8)).save(path)
                    triple.append(path)
                triples.append(triple)
            differences = []
            for arm, loss in (('control', 'normal'), ('antioverfit', 'weighted')):
                source = runtime.predict(folder / 'manifiesto.json', arm, loss, triples)
                arguments = []
                for index, phase in enumerate(('pre', 'early', 'late')):
                    arguments += ['--'+phase, *[str(triple[index]) for triple in triples]]
                environment = os.environ.copy()
                environment['PYTHONPATH'] = str(isolated / 'codigo/src')
                result = subprocess.run([sys.executable, '-B', str(isolated / 'inferencia.py'),
                    'predecir', '--arm', arm, '--loss', loss, *arguments],
                    cwd=sandbox, env=environment, capture_output=True, text=True, check=True)
                prediction = json.loads(result.stdout)
                delta = abs(source['probability_raw']-prediction['probability_raw'])
                assert delta <= 2e-6 and source['pCR'] == prediction['pCR']
                differences.append({'arm': arm, 'loss': loss, 'source': source,
                                    'isolated': prediction, 'absolute_difference': delta})
        audit = {'verified_at': datetime.now(ZoneInfo('Europe/Madrid')).isoformat(),
                 'loading': verification, 'isolated_inference': differences,
                 'previous_version_files_unchanged': len(baseline), 'dataset_needed': False,
                 'test_images_loaded': 0, 'optimizer_or_rng_in_exports': False}
        write(folder / 'verificacion_paquete.json', audit)
        write(evidence / 'verificacion_paquete.json', audit)
        inventory = {p.relative_to(folder).as_posix(): base.sha256(p)
                     for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
        metadata = {'id': IDENTIFIER, 'created_at': datetime.now(ZoneInfo('Europe/Madrid')).isoformat(),
                    'status': record['status'], 'files_sha256': inventory,
                    'weights_bytes': sum((folder / m['path']).stat().st_size for m in models),
                    'models': 16, 'production_version_created': False}
        write(folder / 'paquete.json', metadata)
        (folder / 'paquete.json.sha256').write_text(base.sha256(folder / 'paquete.json')+'\n')
        folder.rename(destination)
    write(OUT / 'verificacion_paquete.json', audit)
    write(OUT / 'paquete_preparado.json', {'path': str(destination.relative_to(ROOT)),
          'metadata_sha256': base.sha256(destination / 'paquete.json'), **metadata})
    print(json.dumps({'path': str(destination), 'models': 16, 'weights_bytes': metadata['weights_bytes'],
                      'isolated_differences': [r['absolute_difference'] for r in differences]}, indent=2))


if __name__ == '__main__':
    main()
