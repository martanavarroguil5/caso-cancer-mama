"""Carga las CNN por fases o cinéticas archivadas sin utilizar el dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import entrenamiento as base
from .app_logic import PHASES, uploads_from_paths, validate_phase_uploads
from .entrenamiento_imagen import CNNImagenFases, numero_parametros


ARCHITECTURES = {
    'CNNImagenFases': 'shared_phase_encoder_ordered_temporal_fusion',
    'CNNMultiescalaCinetica': 'multiscale_early_temporal_kinetic_fusion',
}


def cargar_manifest(path):
    path = Path(path).resolve()
    if base.sha256(path) != path.with_suffix(path.suffix + '.sha256').read_text().strip():
        raise ValueError('El manifiesto ha cambiado')
    manifest = json.loads(path.read_text())
    models = manifest['models']
    single = manifest.get('schema_version') == 2
    expected = ({(seed, 0) for seed in (42, 2026)} if single else
                {(seed, fold) for seed in (42, 2026) for fold in range(5)})
    if len(models) != len(expected) or {(m['seed'], m['fold']) for m in models} != expected:
        raise ValueError('Número de modelos, semillas o folds incompatible')
    if single and (manifest.get('evaluation_protocol') != 'train_only_four_way_single_slice' or
                   manifest.get('inference_unit') != 'single_slice' or
                   manifest.get('model_class') != 'CNNMultiescalaCinetica'):
        raise ValueError('Contrato de cierre de un corte inválido')
    prep = manifest['preprocessing']
    if (manifest.get('model_class') not in ARCHITECTURES or
            manifest.get('slice_aggregation') != ('single' if single else 'mean') or
            prep.get('clinical_required') is not False or
            prep.get('phases') != list(PHASES) or prep.get('shape') != [3, 256, 256]):
        raise ValueError('Contrato incompatible con la CNN solo imagen')
    architecture = ARCHITECTURES[manifest['model_class']]
    if any(item['model_config'].get('architecture') != architecture for item in models):
        raise ValueError('La clase del manifiesto no coincide con la arquitectura de los pesos')
    if manifest['model_class'] == 'CNNMultiescalaCinetica':
        if (prep.get('center_crop_sizes') != [256, 128, 64] or
                prep.get('encoder_input_size') != [128, 128] or
                prep.get('scale_weights_shared') is not True or
                prep.get('kinetic_maps') != ['EARLY-PRE', 'LATE-PRE', 'EARLY-LATE']):
            raise ValueError('Preprocesado incompatible con la CNN cinética')
    calibration = manifest['calibration']
    if (calibration.get('clip_epsilon') != base.EPS or calibration['slope'] < 0 or
            not np.isfinite([calibration['slope'], calibration['intercept'],
                             manifest['threshold']]).all() or
            not 0 <= manifest['threshold'] <= 1):
        raise ValueError('Calibración o umbral inválidos')
    for item in [*models, *manifest.get('provenance_files', [])]:
        if not (path.parent / item['path']).resolve().is_relative_to(path.parent):
            raise ValueError('Ruta de artefacto fuera del paquete')
        if base.sha256(path.parent / item['path']) != item['sha256']:
            raise ValueError(f"Artefacto modificado: {item['path']}")
    return manifest


def cargar_modelo(path, item):
    artifact = torch.load(Path(path).parent / item['path'], map_location='cpu',
                          weights_only=True)
    if artifact.get('schema_version') != 1 or artifact['model_config'] != item['model_config']:
        raise ValueError('Configuración de pesos y manifiesto diferentes')
    if not all(bool(torch.isfinite(value).all()) for value in artifact['state_dict'].values()):
        raise ValueError('Los pesos contienen valores no finitos')
    architecture = artifact['model_config'].get('architecture')
    if architecture == ARCHITECTURES['CNNImagenFases']:
        model = CNNImagenFases(artifact['model_config'])
    elif architecture == ARCHITECTURES['CNNMultiescalaCinetica']:
        from .entrenamiento_cinetico import CNNMultiescalaCinetica
        model = CNNMultiescalaCinetica(artifact['model_config'])
    else:
        raise ValueError('Arquitectura de pesos desconocida')
    model.load_state_dict(artifact['state_dict'], strict=True)
    return model.eval()


def verificar_modelo_final(path):
    manifest = cargar_manifest(path)
    counts = {numero_parametros(cargar_modelo(path, item)) for item in manifest['models']}
    if len(counts) != 1:
        raise ValueError('El ensemble contiene arquitecturas diferentes')
    return {'models': len(manifest['models']), 'model_class': manifest['model_class'],
            'parameters_per_model': counts.pop(),
            'checksums_verified': True, 'clinical_required': False,
            'aggregation': manifest['slice_aggregation'], 'threshold': manifest['threshold'],
            'calibration_is_constant': manifest['calibration']['slope'] == 0,
            'test_evaluated_once': manifest['test_evaluated_once'], 'test_repeated': False}


@torch.inference_mode()
def predecir_cortes(path, triples):
    manifest = cargar_manifest(path)
    if manifest.get('inference_unit') == 'single_slice' and len(triples) != 1:
        raise ValueError('Esta versión requiere exactamente un corte alineado')
    samples = []
    for triple in triples:
        if len(triple) != 3:
            raise ValueError('Cada corte requiere PRE, EARLY y LATE')
        paths = {phase: Path(png) for phase, png in zip(PHASES, triple)}
        samples.append(validate_phase_uploads(uploads_from_paths(paths)))
    if not samples or len({sample.patient_id for sample in samples}) != 1:
        raise ValueError('Se necesitan cortes de una sola paciente')
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError('Corte duplicado')
    x = torch.from_numpy(np.stack([
        np.stack([sample.arrays[phase] for phase in PHASES]) for sample in samples
    ])).float().div_(255.0)
    predictions = []
    for item in manifest['models']:
        model = cargar_modelo(path, item)
        probabilities = torch.cat([torch.sigmoid(model(chunk).squeeze(1))
                                   for chunk in x.split(8)]).numpy()
        predictions.append(probabilities)
    raw = float(np.mean(predictions, axis=0).mean())
    calibrated = float(base.apply_calibration(np.array([raw]), manifest['calibration'])[0])
    return {'patient_id': samples[0].patient_id, 'n_slices': len(samples),
            'models': len(predictions), 'probability_raw': raw,
            'probability_calibrated': calibrated, 'threshold': manifest['threshold'],
            'pCR': int(calibrated >= manifest['threshold']), 'clinical_used': False,
            'status': manifest['status'], 'calibration_is_constant': manifest['calibration']['slope'] == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('accion', choices=('verificar', 'predecir'))
    parser.add_argument('--manifest', type=Path, required=True)
    for phase in ('pre', 'early', 'late'):
        parser.add_argument('--' + phase, type=Path, nargs='+')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.accion == 'predecir':
        if not args.pre or not args.early or not args.late or not len(args.pre) == len(args.early) == len(args.late):
            parser.error('Se requiere el mismo número de PRE, EARLY y LATE')
        result = predecir_cortes(args.manifest, list(zip(args.pre, args.early, args.late)))
    else:
        result = verificar_modelo_final(args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
