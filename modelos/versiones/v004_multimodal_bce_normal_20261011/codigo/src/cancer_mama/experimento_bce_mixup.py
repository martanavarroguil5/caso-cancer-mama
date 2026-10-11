"""CNN original desde cero: BCE normal y mixup multimodal, sin transfer learning."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch

from . import entrenamiento as base
from . import experimento_loss as previous
from . import informe_loss as diagnosis
from . import experimento_local as uncertainty
from .paths import PROJECT_ROOT, RESULTS_DIR

DEFAULT_OUTPUT = RESULTS_DIR / '04_entrenamiento/bce_mixup_multimodal_20261011'
ARMS = {'bce_normal': 0., 'bce_mixup': .2}
SEEDS = (42, 2026)
KEYS = ['patient_id', 'label', 'fold', 'cohort']


def settings(smoke=False):
    config = previous.settings(smoke=smoke, workers=0)
    config.update(learning_rate=.0008, seeds=list(SEEDS),
                  inference='raw_probability_without_posthoc_calibration',
                  mixup_scope='images_and_imputed_standardized_clinical_and_target',
                  bootstrap_alpha=.05/12)
    return config


def initializer(fit, preprocessing):
    patients = fit.drop_duplicates('patient_id').sort_values('patient_id')
    values = base.clinical_matrix(patients)
    missing = np.isnan(values)
    standardized = (np.where(missing, preprocessing['medians'], values)
                    - preprocessing['means']) / preprocessing['stds']
    design = np.column_stack((standardized, missing.astype(float)))
    regression = LogisticRegression(C=1., max_iter=2000, random_state=0).fit(design, patients.pCR)
    return {'method': 'fit_only_unweighted_logistic', 'coefficients': regression.coef_[0].tolist(),
            'intercept': float(regression.intercept_[0]), 'n_patients': len(patients)}


class MixupCNN(base.CNN):
    """Los pesos y la inferencia normal son compatibles con la CNN original.

    La mezcla clínica se aplica después de imputar y escalar. Mezclar NaN en las
    variables crudas borraría información observada de la otra paciente.
    """
    def forward_prepared(self, x, clinical_design):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError('Se requiere [N,3,256,256]')
        if clinical_design.shape != (len(x), 8) or not torch.isfinite(clinical_design).all():
            raise ValueError('Diseño clínico preparado incompatible')
        z = self.features(self.preprocess(x))
        z = torch.cat((self.avgpool(z).flatten(1), self.maxpool(z).flatten(1)), dim=1)
        z = torch.cat((self.image_projection(z), clinical_design), dim=1)
        return self.classifier(z)


def mix_batch(images, clinical, labels, alpha):
    if alpha <= 0:
        return images, clinical, labels
    coefficient = float(np.random.beta(alpha, alpha))
    order = torch.randperm(len(labels), device=images.device)
    def combine(x):
        return coefficient*x + (1-coefficient)*x[order]
    return combine(images), combine(clinical), combine(labels)


def indices(samples, frame):
    positions = {sample: i for i, sample in enumerate(samples.sample_id)}
    return np.asarray([positions[sample] for sample in frame.sample_id])


@torch.inference_mode()
def evaluate(model, memory, clinical, labels, ix, frame, device, batch_size):
    model.eval()
    predictions = []
    total = 0.
    for start in range(0, len(ix), batch_size):
        ids = torch.as_tensor(ix[start:start+batch_size], device=device)
        x = memory[ids].float().div_(255.)
        logits = model(x, clinical[ids]).squeeze(1)
        total += float(torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels[ids], reduction='sum'))
        predictions.append(logits.sigmoid().cpu().numpy())
    probabilities = np.concatenate(predictions)
    patients = previous.patient_frame(frame, probabilities)
    metrics = base.binary_metrics(patients.label.to_numpy(), patients.probability.to_numpy())
    metrics['slice_bce'] = total/len(ix)
    return metrics, probabilities


def run_one(samples, memory, output, arm, seed, fold, config, device, progress, until_epoch=None):
    if arm not in ARMS:
        raise ValueError('Brazo desconocido')
    config = deepcopy(config)
    base.seed_everything(seed, deterministic=config['training']['deterministic'])
    fit, selection, outer = previous.split_patients(samples, fold, config['inner_fraction'], config['split_seed'])
    if config['smoke']:
        fit, selection, outer = [f[f.patient_id.isin(f.drop_duplicates('patient_id').groupby('pCR').head(3).patient_id)].copy()
                                 for f in (fit, selection, outer)]
    prep = base.fit_clinical_preprocessing(fit)
    config['model']['clinical'].update(preprocessing=prep, initialization=initializer(fit, prep))
    config.update(arm=arm, mixup_alpha=ARMS[arm], seed=seed, fold=fold,
                  sources={Path(p).name: base.sha256(p) for p in (__file__, base.__file__, previous.__file__, diagnosis.__file__)},
                  data=previous.signature(samples),
                  partitions={name: previous.signature(f) for name, f in (('fit', fit), ('selection', selection), ('outer', outer))})
    folder = output/arm/f'seed_{seed}'/f'fold_{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = base.config_hash(config)
    if (folder/'config.json').exists() and base.config_hash(json.loads((folder/'config.json').read_text())) != fingerprint:
        raise ValueError('Configuración modificada; no se reutiliza el entrenamiento')
    base.json_write(folder/'config.json', config)
    if (folder/'summary.json').exists():
        summary = json.loads((folder/'summary.json').read_text())
        if summary['config_hash'] != fingerprint:
            raise ValueError('Resumen incompatible')
        for name, digest in summary['files'].items():
            if base.sha256(folder/name) != digest:
                raise ValueError('Artefacto cambiado')
        return summary
    pd.concat([f[['patient_id', 'pCR', 'dataset']].drop_duplicates().assign(role=name)
               for name, f in (('fit', fit), ('selection', selection), ('outer', outer))]).to_csv(folder/'particiones.csv', index=False)
    base.json_write(folder/'environment.json', base.environment(device))
    clinical = torch.as_tensor(base.clinical_matrix(samples), dtype=torch.float32, device=device)
    labels = torch.as_tensor(samples.pCR.to_numpy(dtype=np.float32, copy=True), device=device)
    fi, si, oi = [indices(samples, f) for f in (fit, selection, outer)]
    tc = config['training']
    batch_size = 8 if config['smoke'] else tc['batch_size']
    model = MixupCNN(config['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=tc['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc['epochs'], eta_min=tc['min_lr'])
    amp = device.type == 'cuda' and tc['amp']
    scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=1024.)
    criterion = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed+fold*1000)
    augmentation_generator = torch.Generator().manual_seed(seed+fold*1000+1)
    history = []
    start = 0
    checkpoint = folder/'last.pt'
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    def measure(epoch, optimized=None, seconds=None):
        fm, _ = evaluate(model, memory, clinical, labels, fi, fit, device, batch_size)
        sm, _ = evaluate(model, memory, clinical, labels, si, selection, device, batch_size)
        return {'epoch': epoch, 'optimized_bce': optimized, 'seconds': seconds,
                'lr': optimizer.param_groups[0]['lr'],
                **{'fit_'+k: fm[k] for k in ('log_loss', 'brier', 'roc_auc', 'slice_bce')},
                **{'selection_'+k: sm[k] for k in ('log_loss', 'brier', 'roc_auc', 'slice_bce')}}

    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if payload['config_hash'] != fingerprint:
            raise ValueError('Checkpoint incompatible')
        model.load_state_dict(payload['state_dict'])
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
        scaler.load_state_dict(payload['scaler'])
        history, tracker, best_state, start = (payload[k] for k in ('history', 'tracker', 'best_state', 'epoch'))
        base.restore_rng(payload['rng'], generator, augmentation_generator)
    else:
        row = measure(0)
        history.append(row)
        tracker = previous.new_tracker('logloss', row['selection_log_loss'])
        best_state = previous.cpu_state(model)
    limit = min(tc['epochs'], until_epoch if until_epoch is not None else tc['epochs'])
    for epoch in range(start+1, limit+1):
        if tracker['stopped']:
            break
        started = time.perf_counter()
        model.train()
        order = fi[torch.randperm(len(fi), generator=generator).numpy()]
        total = 0.
        for begin in range(0, len(order), batch_size):
            ids = torch.as_tensor(order[begin:begin+batch_size], device=device)
            x = memory[ids].float().div_(255.)
            x = base.aumentar_geometria(x, augmentation_generator, **config['augmentation'])
            c = model.preprocess_clinical(clinical[ids])
            x, c, y = mix_batch(x, c, labels[ids], ARMS[arm])
            optimizer.zero_grad(set_to_none=True)
            with base.amp_context(device, amp):
                loss = criterion(model.forward_prepared(x, c).squeeze(1), y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Pérdida no finita')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc['clip_grad_norm'], error_if_nonfinite=not amp)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())*len(ids)
        scheduler.step()
        row = measure(epoch, total/len(fi))
        row['seconds'] = time.perf_counter()-started
        history.append(row)
        if previous.update_tracker(tracker, row['selection_log_loss'], epoch, tc['min_epochs'], tc['patience'], config['loss_min_delta']):
            best_state = previous.cpu_state(model)
        previous.save_torch(checkpoint, {'config_hash': fingerprint, 'epoch': epoch,
            'state_dict': previous.cpu_state(model), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'history': history, 'tracker': tracker, 'best_state': best_state,
            'rng': base.capture_rng(generator, augmentation_generator)})
        base.history_write(folder/'history.csv', history)
        progress({'stage': 'training', 'arm': arm, 'seed': seed, 'fold': fold,
                  'epoch': epoch, 'best_epoch': tracker['best_epoch'], 'metrics': row})
        print(f"{arm} seed={seed} fold={fold} epoch={epoch} fit={row['fit_log_loss']:.4f} "
              f"selection={row['selection_log_loss']:.4f} best={tracker['best_epoch']} seconds={row['seconds']:.1f}", flush=True)
    if history[-1]['epoch'] < tc['epochs'] and not tracker['stopped']:
        return None
    model.load_state_dict(best_state)
    previous.save_torch(folder/'best.pt', {'schema_version': 1, 'state_dict': best_state, 'model_config': config['model']})
    # Outer se evalúa solamente después de fijar el checkpoint.
    for name, ix, frame in (('inner', si, selection), ('oof', oi, outer)):
        _, p = evaluate(model, memory, clinical, labels, ix, frame, device, batch_size)
        rows = frame[['sample_id', 'patient_id', 'pCR', 'fold', 'dataset']].copy()
        rows['probability'] = p
        rows.to_csv(folder/f'{name}.csv', index=False)
    base.history_write(folder/'history.csv', history)
    summary = {'config_hash': fingerprint, 'smoke': config['smoke'], 'tracker': tracker,
               'n_parameters': base.numero_parametros(model),
               'seconds_epochs': sum(r['seconds'] or 0 for r in history),
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
               'files': {name: base.sha256(folder/name) for name in ('best.pt', 'history.csv', 'particiones.csv',
                          'config.json', 'environment.json', 'inner.csv', 'oof.csv')}}
    base.json_write(folder/'summary.json', summary)
    return summary


def average_slices(frames):
    keys = ['sample_id', 'patient_id', 'pCR', 'fold', 'dataset']
    frames = [f.sort_values('sample_id').reset_index(drop=True) for f in frames]
    if any(not f[keys].equals(frames[0][keys]) for f in frames[1:]):
        raise ValueError('Predicciones de semillas desalineadas')
    merged = frames[0].copy()
    merged['probability'] = np.mean([f.probability.to_numpy() for f in frames], axis=0)
    return previous.patient_frame(merged, merged.probability.to_numpy())


def compare(samples, output, repetitions=2000):
    dest = output/'comparacion'
    dest.mkdir(exist_ok=True)
    controls = {name: pd.read_csv(diagnosis.DEFAULT_REPORT/f'oof_calibrated_inner_{selector}.csv')
                for name, selector in (('weighted_auc_control', 'auc'), ('weighted_logloss_control', 'logloss'))}
    roster = samples.drop_duplicates('patient_id').sort_values('patient_id').rename(columns={'pCR':'label', 'dataset':'cohort'}).reset_index(drop=True)
    if any(not f[KEYS].equals(roster[KEYS]) for f in controls.values()):
        raise ValueError('Control de otras pacientes')
    metrics = [{'candidate': name, **diagnosis.metrics(f)} for name, f in controls.items()]
    comparisons = {}
    for arm in ARMS:
        raw, calibrated, calibrators = [], [], []
        for fold in range(5):
            predictions = {'inner': [], 'oof': []}
            for seed in SEEDS:
                folder = output/arm/f'seed_{seed}'/f'fold_{fold}'
                config = json.loads((folder/'config.json').read_text())
                summary = json.loads((folder/'summary.json').read_text())
                if summary['smoke'] or summary['config_hash'] != base.config_hash(config) or config['data'] != previous.signature(samples):
                    raise ValueError('Run no apto para comparación')
                if (config['arm'], config['seed'], config['fold']) != (arm, seed, fold):
                    raise ValueError('Run en carpeta equivocada')
                for name, digest in summary['files'].items():
                    if base.sha256(folder/name) != digest:
                        raise ValueError('Artefacto modificado')
                for name in predictions:
                    predictions[name].append(pd.read_csv(folder/f'{name}.csv'))
            inner, outer = [average_slices(predictions[name]) for name in ('inner', 'oof')]
            calibrated_outer, calibration, threshold = diagnosis.calibrate_inner(inner, outer)
            raw.append(outer)
            calibrated.append(calibrated_outer)
            calibrators.append({'fold': fold, 'platt_diagnostic': calibration,
                                'primary_threshold': base.youden_threshold(inner.label.to_numpy(), inner.probability.to_numpy()),
                                'platt_threshold': threshold})
        frame = pd.concat(raw).sort_values('patient_id').reset_index(drop=True)
        if not frame[KEYS].equals(roster[KEYS]):
            raise ValueError('OOF incompletas o desalineadas')
        frame.to_csv(dest/f'oof_raw_{arm}.csv', index=False)
        diagnostic = pd.concat(calibrated).sort_values('patient_id').reset_index(drop=True)
        diagnostic.to_csv(dest/f'oof_calibrated_{arm}.csv', index=False)
        base.json_write(dest/f'calibration_{arm}.json', calibrators)
        row = diagnosis.metrics(frame)
        metrics.extend([{'candidate': arm, **row}, {'candidate': arm+'_platt_diagnostic', **diagnosis.metrics(diagnostic)}])
        control = controls['weighted_auc_control']
        delta = {k: row[k]-diagnosis.metrics(control)[k] for k in ('log_loss', 'brier', 'roc_auc', 'mean_fold_auc')}
        ci = uncertainty.intervals(control, frame, repetitions=repetitions, alpha=.05/12)
        comparisons[arm] = {'delta_vs_control': delta, 'intervals_99_58': ci,
                            'promising_internal_result': ci['log_loss']['upper'] < 0 and ci['mean_fold_auc']['lower'] >= -.01 and delta['brier'] <= 0}
    report = {'metrics': metrics, 'comparisons': comparisons, 'test_images_loaded': 0,
              'primary': 'Raw probabilities for the two new unweighted arms; control uses its inner Platt.',
              'selection': 'Inner patient log-loss; same outer folds and inner partitions as weighted control.',
              'adaptive_search': True, 'automatic_promotion': False, 'bootstrap_repetitions': repetitions}
    base.json_write(dest/'resumen.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--prueba', action='store_true')
    args = parser.parse_args()
    output = args.salida/'pruebas' if args.prueba else args.salida
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'status': 'running', 'pid': os.getpid(), 'smoke': args.prueba, 'started_at': previous.now()}
        def progress(data):
            state.update(data, updated_at=previous.now())
            base.json_write(output/'estado.json', state)
        progress({'stage': 'verifying'})
        try:
            device = torch.device('cuda')
            torch.set_num_threads(8)
            samples, old = diagnosis.verify_inputs(base.pdatos.DATOS, previous.DEFAULT_OUTPUT)
            protected = previous.protected_snapshot()
            config = settings(args.prueba)
            cache = base.CacheTrain(samples, base.pdatos.DATOS, previous.DEFAULT_OUTPUT/'.cache', workers=4)
            env = base.environment(device)
            env.pop('git_commit', None)
            params = {'sources': {Path(p).name: base.sha256(p) for p in (__file__, base.__file__, previous.__file__, diagnosis.__file__, uncertainty.__file__)},
                      'protected': protected, 'data': previous.signature(samples),
                      'train_png_sha256': old['parameters']['train_png_sha256'], 'cache_sha256': base.sha256(cache.path),
                      'config': config, 'arms': ARMS, 'environment': env,
                      'controls': {name: base.sha256(diagnosis.DEFAULT_REPORT/name) for name in ('oof_calibrated_inner_auc.csv', 'oof_calibrated_inner_logloss.csv')}}
            fingerprint = base.config_hash(params)
            path = output/'protocolo.json'
            if path.exists() and json.loads(path.read_text())['fingerprint'] != fingerprint:
                raise ValueError('Protocolo cambiado; no se reanuda')
            if not path.exists():
                base.json_write(path, {'fingerprint': fingerprint, 'parameters': params, 'created_at': previous.now()})
            progress({'stage': 'loading_train_cache'})
            memory = torch.from_numpy(np.array(cache.memory, copy=True)).to(device)
            for arm in ARMS:
                for seed in ([42] if args.prueba else SEEDS):
                    for fold in ([0] if args.prueba else range(5)):
                        run_one(samples, memory, output, arm, seed, fold, config, device, progress)
            if not args.prueba:
                progress({'stage': 'comparison'})
                report = compare(samples, output)
                print(json.dumps(report, ensure_ascii=False), flush=True)
            if previous.protected_snapshot() != protected:
                raise RuntimeError('Archivos protegidos cambiados')
            progress({'status': 'complete', 'stage': 'complete', 'protected_unchanged': True, 'finished_at': previous.now()})
        except BaseException as exc:
            progress({'status': 'failed', 'error': str(exc)})
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
