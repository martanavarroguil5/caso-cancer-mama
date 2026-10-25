"""Repite candidata BCE normal y control ponderado con dos semillas nuevas."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch

from . import entrenamiento as base
from . import experimento_loss as control
from . import experimento_bce_mixup as candidate
from . import informe_loss as diagnosis
from . import incertidumbre as uncertainty
from .paths import PROJECT_ROOT, RESULTS_DIR

OUTPUT = RESULTS_DIR/'04_entrenamiento/confirmacion_bce_normal_20261011'
SEEDS = (73, 1701)
ALPHA = .05/12


def verify_run(folder, samples, weighted=False):
    config = json.loads((folder/'config.json').read_text())
    summary = json.loads((folder/'summary.json').read_text())
    if config['smoke'] or summary['config_hash'] != base.config_hash(config):
        raise ValueError('Run de prueba o configuración modificada')
    data_key = 'data_signature' if weighted else 'data'
    if config[data_key] != control.signature(samples):
        raise ValueError('Datos distintos')
    if config['seed'] not in SEEDS or config['fold'] not in range(5):
        raise ValueError('Semilla o fold ajenos a la confirmación')
    if weighted and (summary['status'] != 'complete' or not summary['eligible_for_comparison']):
        raise ValueError('Control incompleto')
    for name, digest in summary['files'].items():
        if base.sha256(folder/name) != digest:
            raise ValueError('Artefacto cambiado')
    fit, inner, outer = control.split_patients(samples, config['fold'])
    recorded = config['partition_signatures' if weighted else 'partitions']
    for name, frame in (('fit', fit), ('selection', inner), ('outer', outer)):
        if control.signature(frame) != recorded[name]:
            raise ValueError('Particiones distintas')
    return config, summary, inner


def decision(intervals, delta, seed_deltas):
    return bool(intervals['log_loss']['upper'] < 0
                and intervals['mean_fold_auc']['lower'] >= -.01
                and delta['brier'] <= 0
                and len(seed_deltas) == len(SEEDS)
                and all(item['log_loss'] < 0 and item['mean_fold_auc'] >= -.01 for item in seed_deltas))


def compare(samples, cache, output, device, repetitions=2000):
    by_seed = {seed: {name: {'inner': [], 'oof': []} for name in ('normal', 'weighted_auc', 'weighted_logloss')}
               for seed in SEEDS}
    evidence = {}
    for seed in SEEDS:
        for fold in range(5):
            normal = output/f'candidate/bce_normal/seed_{seed}/fold_{fold}'
            weighted = output/f'control/lr_actual/seed_{seed}/fold_{fold}'
            for folder, is_weighted in ((normal, False), (weighted, True)):
                config, summary, inner = verify_run(folder, samples, is_weighted)
                if (config['seed'], config['fold']) != (seed, fold):
                    raise ValueError('Run en carpeta equivocada')
                for name, digest in summary['files'].items():
                    evidence[str((folder/name).relative_to(PROJECT_ROOT))] = digest
            for name in ('inner', 'oof'):
                by_seed[seed]['normal'][name].append(pd.read_csv(normal/f'{name}.csv'))
            for selector in ('auc', 'logloss'):
                artifact = torch.load(weighted/f'best_{selector}.pt', map_location='cpu', weights_only=True)
                model = base.CNN(artifact['model_config']).to(device)
                model.load_state_dict(artifact['state_dict'], strict=True)
                loader = base.make_loader(inner, base.pdatos.DATOS, 64, 0,
                                          torch.Generator().manual_seed(seed), cache=cache, clinical=True)
                probabilities, _ = base.predict(model, loader, device, False)
                frame = inner[['sample_id', 'patient_id', 'pCR', 'fold', 'dataset']].copy()
                frame['probability'] = probabilities
                frame.to_csv(weighted/f'inner_{selector}.csv', index=False)
                evidence[str((weighted/f'inner_{selector}.csv').relative_to(PROJECT_ROOT))] = base.sha256(weighted/f'inner_{selector}.csv')
                by_seed[seed]['weighted_'+selector]['inner'].append(frame)
                by_seed[seed]['weighted_'+selector]['oof'].append(pd.read_csv(weighted/f'oof_{selector}.csv'))
                del model

    dest = output/'comparacion'
    dest.mkdir(exist_ok=True)
    frames, rows, seed_deltas = {}, [], []
    keys = candidate.KEYS
    roster = samples.drop_duplicates('patient_id').sort_values('patient_id').rename(columns={'pCR': 'label', 'dataset': 'cohort'}).reset_index(drop=True)
    # Siempre se promedian semillas por corte antes de la mediana de paciente.
    for seeds in ((SEEDS[0],), (SEEDS[1],), SEEDS):
        tag = str(seeds[0]) if len(seeds) == 1 else 'ensemble'
        current = {}
        for name in ('normal', 'weighted_auc', 'weighted_logloss'):
            results, calibrators = [], []
            for fold in range(5):
                inner, outer = [candidate.average_slices([by_seed[seed][name][split][fold] for seed in seeds])
                                for split in ('inner', 'oof')]
                if name != 'normal':
                    outer, calibration, threshold = diagnosis.calibrate_inner(inner, outer)
                else:
                    calibration = {'method': 'none'}
                    threshold = base.youden_threshold(inner.label.to_numpy(), inner.probability.to_numpy())
                    outer['threshold'] = threshold
                results.append(outer)
                calibrators.append({'fold': fold, 'calibration': calibration, 'threshold': threshold})
            frame = pd.concat(results).sort_values('patient_id').reset_index(drop=True)
            if not frame[keys].equals(roster[keys]):
                raise ValueError('OOF incompletas o desalineadas')
            label = f'{name}_{tag}'
            frames[label] = frame
            current[name] = diagnosis.metrics(frame)
            rows.append({'candidate': name, 'seeds': list(seeds), **current[name]})
            frame.to_csv(dest/f'oof_{label}.csv', index=False)
            base.json_write(dest/f'calibradores_{label}.json', calibrators)
        delta = {k: current['normal'][k]-current['weighted_auc'][k]
                 for k in ('log_loss', 'brier', 'roc_auc', 'mean_fold_auc')}
        if len(seeds) == 1:
            seed_deltas.append({'seed': seeds[0], **delta})
        else:
            ensemble_delta = delta
    intervals = uncertainty.intervals(frames['weighted_auc_ensemble'], frames['normal_ensemble'], repetitions, alpha=ALPHA)
    result = {'seeds': list(SEEDS), 'metrics': rows, 'delta_vs_control': ensemble_delta,
              'seed_deltas': seed_deltas, 'intervals_99_58': intervals,
              'replicated_internal_result': decision(intervals, ensemble_delta, seed_deltas),
              'bootstrap_repetitions': repetitions, 'test_images_loaded': 0,
              'warning': 'Repetición de entrenamiento con semillas nuevas sobre el mismo desarrollo; no validación externa.',
              'automatic_promotion': False}
    base.json_write(dest/'resumen.json', result)
    base.json_write(dest/'entradas_sha256.json', evidence)
    pd.DataFrame(rows).drop(columns='fold_auc').to_csv(dest/'metricas.csv', index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida', type=Path, default=OUTPUT)
    parser.add_argument('--prueba', action='store_true')
    args = parser.parse_args()
    output = args.salida/'pruebas' if args.prueba else args.salida
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'status': 'running', 'pid': os.getpid(), 'started_at': control.now(), 'smoke': args.prueba}
        def progress(data):
            state.update(data, updated_at=control.now())
            base.json_write(output/'estado.json', state)
        progress({'stage': 'verifying'})
        try:
            torch.set_num_threads(8)
            device = torch.device('cuda')
            samples, old = diagnosis.verify_inputs(base.pdatos.DATOS, control.DEFAULT_OUTPUT)
            protected = control.protected_snapshot()
            normal_config = candidate.settings(args.prueba)
            normal_config['seeds'] = list(SEEDS)
            weighted_config = control.settings(args.prueba, workers=4)
            cache = base.CacheTrain(samples, base.pdatos.DATOS, control.DEFAULT_OUTPUT/'.cache', workers=4)
            env = base.environment(device)
            env.pop('git_commit', None)
            params = {'seeds': list(SEEDS), 'normal_config': normal_config, 'weighted_config': weighted_config,
                      'smoke': args.prueba, 'environment': env, 'protected': protected,
                      'sources': {Path(p).name: base.sha256(p) for p in
                                  (__file__, base.__file__, control.__file__, candidate.__file__, diagnosis.__file__, uncertainty.__file__)},
                      'data': control.signature(samples), 'train_png_sha256': old['parameters']['train_png_sha256'],
                      'cache_sha256': base.sha256(cache.path), 'bootstrap_alpha': ALPHA,
                      'original_search_result_sha256': base.sha256(candidate.DEFAULT_OUTPUT/'comparacion/resumen.json')}
            fingerprint = base.config_hash(params)
            protocol = output/'protocolo.json'
            if protocol.exists() and json.loads(protocol.read_text())['fingerprint'] != fingerprint:
                raise ValueError('Protocolo cambiado')
            if not protocol.exists():
                base.json_write(protocol, {'fingerprint': fingerprint, 'parameters': params, 'created_at': control.now()})
            memory = torch.from_numpy(np.array(cache.memory, copy=True)).to(device)
            for seed in ([SEEDS[0]] if args.prueba else SEEDS):
                for fold in ([0] if args.prueba else range(5)):
                    progress({'stage': 'training_control', 'seed': seed, 'fold': fold})
                    control.run_one(samples, base.pdatos.DATOS, output/'control', 'lr_actual', seed, fold,
                                    weighted_config, device, cache=cache,
                                    progress=lambda data: progress(dict(data, stage='training_control')))
                    progress({'stage': 'training_candidate', 'seed': seed, 'fold': fold})
                    candidate.run_one(samples, memory, output/'candidate', 'bce_normal', seed, fold,
                                      normal_config, device,
                                      progress=lambda data: progress(dict(data, stage='training_candidate')))
            if not args.prueba:
                progress({'stage': 'comparison'})
                report = compare(samples, cache, output, device)
                print(json.dumps(report, ensure_ascii=False), flush=True)
            if control.protected_snapshot() != protected:
                raise RuntimeError('Archivos protegidos cambiados')
            progress({'status': 'complete', 'stage': 'complete', 'protected_unchanged': True, 'finished_at': control.now()})
        except BaseException as error:
            progress({'status': 'failed', 'error': str(error)})
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
