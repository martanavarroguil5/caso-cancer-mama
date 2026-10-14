#!/usr/bin/env python3
"""Congela los veinte pesos de inferencia del ensayo, sin volver a entrenar."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from cancer_mama import entrenamiento as base
from cancer_mama import entrenamiento_imagen as image

spec = importlib.util.spec_from_file_location('versionar', ROOT/'modelos/versionar.py')
versionar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(versionar)


def main():
    runs = ROOT/'resultados/04_entrenamiento/cnn_imagen_fases_20261014'
    evidence = Path(__file__).resolve().parent
    temporary = ROOT/'.codex_tmp/manifiestos_imagen_20261014'
    temporary.mkdir(parents=True, exist_ok=True)
    report = json.loads((evidence/'seleccion.json').read_text())
    origin = json.loads((runs/'comparacion/modelo_desarrollo_imagen.json').read_text())
    state = json.loads((runs/'estado.json').read_text())
    for loss, identifier in (
            ('normal', 'v005_cnn_imagen_fases_normal_20261014'),
            ('weighted', 'v006_cnn_imagen_fases_ponderada_20261014')):
        manifest = json.loads(json.dumps(origin))
        manifest.update(status='ensayo_insuficiente_conservado', loss=loss,
                        recommended_for_prediction=False,
                        training_git_commit=state['source_commit'],
                        training_source_sha256=state['source_sha256'])
        models = []
        for seed in (42, 2026):
            for fold in range(5):
                directory = runs/f'cnn_imagen_fases/{loss}/seed_{seed}/fold_{fold}'
                config = json.loads((directory/'config.json').read_text())
                summary = json.loads((directory/'summary.json').read_text())
                weight = directory/'inference.pt'
                assert base.sha256(weight) == summary['inference_sha256']
                assert config['source_sha256']['entrenamiento_imagen.py'] == state['source_sha256']
                models.append({'path': os.path.relpath(weight, temporary),
                               'sha256': summary['inference_sha256'], 'seed': seed,
                               'fold': fold, 'outer_fold': fold, 'model_config': config['model']})
        manifest['models'] = models
        frame = pd.read_csv(evidence/f'oof_pacientes_{loss}.csv')
        candidate = next(c for c in report['candidates'] if c['loss'] == loss)
        metrics = image.metricas_binarias(frame.label, frame.probability_raw, .5)
        metrics.update(mean_fold_auc=candidate['mean_fold_raw_auc'],
                       std_fold_auc=candidate['std_fold_raw_auc'],
                       fold_auc=candidate['fold_raw_auc'])
        manifest['selection'] = {
            'configuration': 'cnn_imagen_fases', 'loss': loss,
            'won_development_loss_comparison': loss == report['selected_loss_for_future_test'],
            'metrics_raw_oof': metrics,
            'crossfit_decision': candidate['crossfit_primary_decision'],
            'promotion': 'none; retained for reproducibility of an insufficient result',
        }
        if loss == 'normal':
            # No se seleccionó una política final para BCE normal. No se ajusta ahora.
            manifest['calibration'] = {'method': 'identity', 'slope': 1.0,
                'intercept': 0.0, 'clip_epsilon': base.EPS,
                'fit_data': 'none; raw archival diagnostic only'}
            manifest['threshold'] = .5
            manifest['threshold_policy'] = {
                'strategy': 'raw_default_not_selected_for_future_test',
                'fit_data': 'none', 'warning': 'Not the crossfit thresholds used in the report'}
        else:
            assert manifest['calibration']['slope'] == 0 and manifest['threshold'] == 1
        manifest['provenance_files'] = [
            {'path': os.path.relpath(path, temporary), 'sha256': base.sha256(path)}
            for path in (evidence/'seleccion.json', evidence/f'oof_pacientes_{loss}.csv',
                         evidence/'procedencia.json', evidence/'verificacion_entrenamiento.json')]
        path = temporary/f'{loss}.json'
        base.json_write(path, manifest)
        path.with_suffix('.json.sha256').write_text(base.sha256(path)+'\n')
        info = versionar.archive_version(
            path, identifier,
            f'CNN solo imagen por fases, BCE {loss}: resultado insuficiente de desarrollo; conservada sin activar',
            evidence=[evidence/'INFORME_RESULTADOS.md', evidence/'01_roc_oof.png',
                      evidence/'02_curvas_fit_seleccion.png',
                      ROOT/'docs/mejoras_cnn/PROTOCOLO_CNN_IMAGEN_FASES.md'])
        print(json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
