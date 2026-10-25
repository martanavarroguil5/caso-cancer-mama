"""CNN espacial compacta: un logit por corte, sin agregación aprendida.

Solo train: cinco folds exteriores, mismos roles previamente auditados. Las
épocas 12–14 son un diagnóstico, nunca una condición para ocultar un óptimo
anterior. La referencia de selección por AUC usa la misma trayectoria y es
secundaria, no un segundo experimento independiente.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import torch
from torch import nn
from torch.nn import functional as F

from . import entrenamiento as base
from . import entrenamiento_cinetico as cnn
from . import entrenamiento_imagen as image
from . import cierre_cinetico as cache
from . import revision_train as audit
from .paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

OUTPUT = RESULTS_DIR/'28_cnn_espacial_por_corte_20261025'
CONTROL = RESULTS_DIR/'27_criterio_docente_20261025'
ROLES = audit.OUTPUT/'protocolo_cinco_folds'
SEEDS, LOSSES, RULES = (42, 2026), ('normal', 'weighted'), ('loss', 'auc')
write, now = cache.write, cache.now


def model_config():
    return {'architecture': 'compact_spatial_per_slice_kinetic_v1',
            'input_phases': ['PRE', 'EARLY', 'LATE'],
            'temporal_channels': ['PRE', 'EARLY', 'LATE', 'EARLY-PRE', 'LATE-PRE', 'EARLY-LATE'],
            'input_size': 256, 'encoder_input_size': 128,
            'backbone_channels': [16, 32, 64], 'spatial_grid': 4,
            'spatial_pool': 'avg_kernel4_stride4_on_16x16',
            'hidden': 32, 'dropout': .35, 'spatial_dropout': .1,
            'normalization': 'batch', 'activation': 'relu', 'clinical': None,
            'slice_output': 'one_logit', 'patient_aggregation': 'mean_probabilities_outside_network'}


class CNNSpatialPerSlice(nn.Module):
    """Codifica fases/diferencias, conserva una cuadrícula4×4 y clasifica un corte.

    No recibe IDs, cohorte, clínica, máscaras ni bolsas de cortes. No contiene
    atención, max entre cortes, EMA o label smoothing del antiguo piloto MIL.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = deepcopy(config or model_config())
        self.config = self.config.get('model', self.config)
        if self.config != model_config():
            raise ValueError('Configuración fuera del único candidato espacial prefijado')
        blocks = []
        for cin, cout in zip([6, 16, 32], [16, 32, 64]):
            blocks.extend([nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                           nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                           nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                           nn.BatchNorm2d(cout), nn.ReLU(inplace=True)])
        self.encoder = nn.Sequential(*blocks)
        self.spatial_dropout = nn.Dropout2d(.1)
        # La entrada fija128 produce16×16; esta media4×4 equivale exactamente
        # al pooling adaptativo4×4 y dispone de backward CUDA determinista.
        self.grid = nn.AvgPool2d(4, stride=4)
        self.classifier = nn.Sequential(nn.Dropout(.35), nn.Linear(64*4*4, 32),
                                        nn.ReLU(inplace=True), nn.Dropout(.35), nn.Linear(32, 1))
        self.apply(image.CNNImagenFases._initialize)

    def encode_grid(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError('Se requieren cortes independientes [N,3,256,256]')
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        z = cnn.CNNMultiescalaCinetica.temporal_maps(x)
        return self.grid(self.spatial_dropout(self.encoder(z)))

    def forward(self, x):
        return self.classifier(self.encode_grid(x).flatten(1))


def make_model(config):
    return CNNSpatialPerSlice(config)


def synthetic_learning_check():
    """Solo patrones sintéticos, igual brillo global y distinta localización."""
    base.seed_everything(42, True); torch.set_num_threads(4)
    device = torch.device('cuda')
    model = make_model(model_config()).to(device)
    x = torch.full((8, 3, 256, 256), .15, device=device)
    y = torch.tensor([0, 1]*4, device=device, dtype=torch.float32)
    for i in range(8):
        start = 64 if y[i] == 0 else 160
        x[i, 1, start:start+32, start:start+32] = .75
        x[i, 2, start:start+32, start:start+32] = .45
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    def assess():
        model.eval()
        with torch.inference_mode():
            logits = model(x).flatten()
            return float(F.binary_cross_entropy_with_logits(logits, y)), float(((logits >= 0) == y.bool()).float().mean())
    initial, _ = assess()
    for _ in range(60):
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            loss = F.binary_cross_entropy_with_logits(model(x).flatten(), y)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update()
    final, accuracy = assess()
    steps = {int(v['step'].item()) for v in optimizer.state.values()}
    if not (final < initial*.7 and accuracy == 1 and steps == {60}):
        raise RuntimeError(f'Falla aprendizaje sintético: BCE{initial}->{final}, acc{accuracy}, pasos{steps}')
    return {'synthetic_only': True, 'patient_images_loaded': 0, 'parameters': image.numero_parametros(model),
            'steps': 60, 'initial_bce': initial, 'final_bce': final, 'accuracy': accuracy,
            'production_lr_and_wd': True, 'deterministic': True,
            'note': 'Prueba de funcionamiento, no estimación de generalización'}


def aligned_ensemble(paths, frame, pixels, device):
    predictions = []
    for path in paths:
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        if set(artifact) != {'schema_version', 'model_config', 'state_dict'}:
            raise ValueError('Solo pesos de inferencia sin optimizador/RNG')
        model = make_model(artifact['model_config']).to(device)
        model.load_state_dict(artifact['state_dict'], strict=True)
        predictions.append(cache.predict(model, pixels))
        del model
    p = np.mean(predictions, axis=0)
    if len(p) != len(frame) or not np.isfinite(p).all():
        raise ValueError('Ensemble no alineado o no finito')
    return p


def patient_metrics(frame, probabilities, pos_weight=1.):
    """Media de todos los cortes antes de calcular BCE/AUC por paciente."""
    patients = image.patient_predictions(frame, probabilities)
    y = patients.label.to_numpy(dtype=float)
    p = np.clip(patients.probability_raw.to_numpy(dtype=float), 1e-7, 1-1e-7)
    metrics = {'auc': float(roc_auc_score(y, patients.probability_raw)),
               'bce': float(np.mean(-pos_weight*y*np.log(p)-(1-y)*np.log1p(-p))),
               'bce_unweighted': float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))}
    return metrics, patients


def cohort_predictions(fit_patients, patients):
    """Comparador, nunca entrada de la CNN: prevalencias aprendidas solo en fit."""
    fit_patients = fit_patients.drop_duplicates('patient_id')
    rates = fit_patients.groupby('dataset').pCR.mean()
    p = patients.dataset.map(rates)
    if p.isna().any():
        raise ValueError('Cohorte exterior ausente en fit')
    return p.to_numpy(dtype=float), rates.to_dict()


def curve_diagnostic(history):
    """Tendencias descriptivas prefijadas; no exige monotonía época a época."""
    h = pd.DataFrame(history)
    if len(h) < 14:
        return {'observed_through_epoch_14': False, 'joint_improvement': False}
    h = h.iloc[:14]
    result = {'observed_through_epoch_14': True, 'window': [1, 14],
              'comparison': 'media épocas12–14 frente a1–3 y pendiente lineal1–14; sin suavizar datos'}
    for role in ('fit', 'selection'):
        v = h[f'{role}_bce'].to_numpy()
        result[f'{role}_change'] = float(v[-3:].mean()-v[:3].mean())
        result[f'{role}_slope'] = float(np.polyfit(h.epoch, v, 1)[0])
    result['joint_improvement'] = all(result[f'{r}_{s}'] < 0
                                             for r in ('fit', 'selection') for s in ('change', 'slope'))
    return result


def roles_for(samples, roles):
    patients = samples[['patient_id', 'dataset', 'pCR', 'fold']].drop_duplicates()
    if roles.patient_id.duplicated().any() or set(roles.role) != {'fit', 'selection', 'outer'}:
        raise ValueError('Roles duplicados o incompletos')
    if not patients.sort_values('patient_id').reset_index(drop=True).equals(
            roles[patients.columns].sort_values('patient_id').reset_index(drop=True)):
        raise ValueError('Roles y datos actuales no coinciden')
    if not samples.split.eq('train').all():
        raise ValueError('Solo train')
    return {r: samples[samples.patient_id.isin(roles.loc[roles.role.eq(r), 'patient_id'])]
            .reset_index(drop=True) for r in ('fit', 'selection', 'outer')}


def prepare(output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise ValueError('Salida existente: no sobrescribir')
    samples, _ = audit.train_metadata(DATA_DIR)
    previous = json.loads((ROLES/'protocolo.json').read_text())
    if base.sha256(ROLES/'protocolo.json') != (ROLES/'protocolo.json.sha256').read_text().strip():
        raise ValueError('Protocolo de roles modificado')
    control_protocol = json.loads((CONTROL/'protocolo.json').read_text())
    if base.sha256(CONTROL/'protocolo.json') != (CONTROL/'protocolo.json.sha256').read_text().strip():
        raise ValueError('Protocolo del control modificado')
    cfg = deepcopy(control_protocol['config'])
    cfg['name'] = 'cnn_espacial_compacta_por_corte'
    cfg['model'] = model_config()
    cfg['hypothesis'] = 'Una CNN más compacta con cuadrícula espacial4×4 puede conservar localización sin memorizar tan pronto; un logit por corte y media de probabilidades'
    cfg['history_plot_title'] = 'CNN espacial compacta: fit y selección por paciente'
    if cfg['training'] != control_protocol['config']['training'] or cfg['augmentation'] != control_protocol['config']['augmentation']:
        raise ValueError('Solo se permite cambiar arquitectura')
    parameters = image.numero_parametros(make_model(cfg))
    final_control = json.loads((CONTROL/'verificacion_final.json').read_text())
    for name, digest in final_control['files_sha256'].items():
        if base.sha256(CONTROL/name) != digest:
            raise ValueError(f'Control cerrado modificado: {name}')
    output.mkdir(parents=True)
    (output/'roles').mkdir()
    folds = []
    for k, record in enumerate(previous['folds']):
        src = ROLES/f'roles_fold_{k}.csv'
        if base.sha256(src) != record['sha256']:
            raise ValueError('Roles alterados')
        if base.sha256(src) != base.sha256(CONTROL/'roles'/src.name):
            raise ValueError('Pacientes distintos del control')
        shutil.copyfile(src, output/'roles'/src.name)
        parts = roles_for(samples, pd.read_csv(src))
        p = parts['outer'].drop_duplicates('patient_id').sort_values('patient_id')
        scores, rates = cohort_predictions(parts['fit'], p)
        folds.append({'fold': k, 'counts': {r: int(f.patient_id.nunique()) for r, f in parts.items()},
                      'cohort_rates_fit_only': rates, 'cohort_auc': float(roc_auc_score(p.pCR, scores))})
    sources = [Path(m.__file__) for m in (base, cnn, image, cache, audit)] + [Path(__file__)]
    sources += [PROJECT_ROOT/'src/cancer_mama/datos.py', DATA_DIR/'metadata/samples.csv', DATA_DIR/'metadata/patients.csv']
    sources += list((output/'roles').glob('*.csv'))
    sources += [CONTROL/'protocolo.json', CONTROL/'oof_loss.csv', CONTROL/'verificacion_final.json']
    protocol = {'created_at': now(), 'authorized_by_user': 'venga pues dale a ver: una variante espacial compacta, mismos roles y entrenamiento',
        'config': cfg, 'parameters': parameters, 'runs': 20, 'seeds': list(SEEDS), 'losses': list(LOSSES),
        'control': {'path': str(CONTROL.relative_to(PROJECT_ROOT)), 'code_commit': 'db845626bfca69276217074256fc131c0bffbc01',
                    'mean_auc': .5562698827127401, 'control_fit_retrained': False,
                    'contract': 'Mismos roles, semillas, BCE por corte1/n, paciente media, selección por pérdida, parada, LR/WD, lote, aumentos y política'},
        'only_candidate': 'Cambiar arquitectura: tres bloques16/32/64, vista global128, cuadrícula4×4 y cabeza32; manteniendo dropout.35/.1 y BN/ReLU',
        'historical_spatial_warning': 'El piloto anterior usaba atención/max entre cortes, EMA y label smoothing; sus0.595/0.611 son selección y no validan este nuevo contrato',
        'folds': folds, 'cohort_mean_auc': float(np.mean([f['cohort_auc'] for f in folds])),
        'source_hashes': {str(p.relative_to(PROJECT_ROOT)): base.sha256(p) for p in sources},
        'previous_versions': cache.previous_versions(),
        'checkpoint': 'Principal: BCE por paciente mínima en selection con pos_weight de fit correspondiente; empate época anterior',
        'early_stopping': 'Paciencia8, mejora mínimaBCE0.001, máximo40, observar al menos14; checkpoint puede ser anterior',
        'training_objective': 'BCEWithLogitsLoss por corte, peso1/n_cortes y normalizador constante n_pacientes/n_cortes; pos_weight por pacientes de fit',
        'curve_measurement': 'Fit y selection en eval, sin aumentos/dropout, media de todos los cortes primero; misma BCE y pos_weight',
        'loss_choice': 'AUC por paciente del ensemble de dos semillas en selection, empate normal; nunca comparar BCE normal contra ponderada directamente',
        'policies': 'Platt C1 no negativo, umbral sensibilidad máxima con especificidad>=.95 y secundario macroF1; solo selection de la misma pareja',
        'outer_guard': 'Congelar las cinco elecciones y políticas antes de cargar imágenes exteriores para inferencia',
        'primary': 'Media±DE AUC cruda cinco folds del ensemble de dos semillas con checkpoints por pérdida',
        'secondary': 'Comparación pareada contra control cerrado; regla bestAUC misma trayectoria, BCE fijas, AUC agrupada/calibrada y por cohorte, AP y umbrales',
        'criteria': {'point_estimate': 'mediaAUC>0.54 y delta positivo frente a cohorte en los mismos folds',
                     'evidence_beyond_cohort': 'extremo inferiorIC95% pareado de delta media>0, condicionado a estos modelos',
                     'curves': 'Describir mejoría conjunta1–14 (pendiente negativa y media12–14 menor que1–3), épocas elegidas y brechas; no garantiza generalización'},
        'data_policy': 'Conservar todos los PNG originales; ISPY1_1139 EARLY=LATE y brillo ISPY1_1193 documentados, sin corrección/exclusión inferida',
        'limits': ['Desarrollo adaptativo, train ya observado; no se presenta como confirmación externa',
                   'Selection compartida para época/pérdida/calibración/umbral: métricas internas optimistas',
                   'La pérdida puede mejorar sin superar cohorte; curvas cercanas pueden significar infraajuste',
                   'Comparación AUC/loss misma trayectoria no aísla el efecto de una parada temprana diferente',
                   'No cambiar este candidato/LR/regularización durante la matriz ni para fabricar curvas de14 épocas',
                   'No abrir test/privada, no entrenar refit final, no activar versión ni hacer push',
                   'No ensayos adicionales automáticos después de estas20 ejecuciones'],
        'test_images_loaded': 0, 'private_images_loaded': 0}
    write(output/'protocolo.json', protocol)
    (output/'protocolo.json.sha256').write_text(base.sha256(output/'protocolo.json')+'\n')
    write(output/'estado.json', {'status': 'prepared', 'runs_completed': 0, 'planned_runs': 20, 'updated_at': now()})
    return protocol


def verify(output):
    output = Path(output)
    if base.sha256(output/'protocolo.json') != (output/'protocolo.json.sha256').read_text().strip():
        raise ValueError('Protocolo modificado')
    p = json.loads((output/'protocolo.json').read_text())
    for name, digest in {**p['source_hashes'], **p['previous_versions']}.items():
        if base.sha256(PROJECT_ROOT/name) != digest:
            raise ValueError(f'Fuente, roles, metadatos o versión alterada: {name}')
    return p


def train_run(fit, selection, fit_x, sel_x, cfg, loss_name, seed, path, device):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    cfg = deepcopy(cfg)
    patients = fit.drop_duplicates('patient_id')
    pw = float(patients.pCR.eq(0).sum()/patients.pCR.eq(1).sum()) if loss_name == 'weighted' else 1.
    cfg.update(loss=loss_name, seed=seed, pos_weight=pw, outer_fold=int(path.parent.parent.name.removeprefix('fold_')),
               counts={'fit_patients': len(patients), 'selection_patients': int(selection.patient_id.nunique())})
    if (path/'config.json').exists() and json.loads((path/'config.json').read_text()) != cfg:
        raise ValueError('Configuración de run modificada')
    write(path/'config.json', cfg)
    if (path/'summary.json').exists():
        result = json.loads((path/'summary.json').read_text())
        for name, digest in result['hashes'].items():
            if base.sha256(path/name) != digest: raise ValueError('Artefacto de run alterado')
        return result
    t = cfg['training']; base.seed_everything(seed, True)
    generator = torch.Generator().manual_seed(seed)
    aug = torch.Generator().manual_seed(seed+1)
    model = make_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t['epochs'], eta_min=t['min_lr'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and t['amp'], init_scale=1024.)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device), reduction='none')
    y = torch.tensor(fit.pCR.to_numpy(), device=device, dtype=torch.float32)
    w = torch.tensor(cache.patient_weights(fit), device=device, dtype=torch.float32)
    normalizer = float(w.mean())  # Fijo: no cambiar el objetivo según la composición del lote.
    start, bad, history, best_score, patience_anchor = 0, 0, [], -1e30, -1e30
    best_auc, best_auc_epoch = -1., -1
    if (path/'last.pt').exists():
        ck = base.load_checkpoint(path/'last.pt', model, optimizer, scheduler, scaler, generator, aug)
        if base.config_hash(ck['config']) != base.config_hash(cfg): raise ValueError('Checkpoint incompatible')
        start, bad, history, best_score = ck['epoch']+1, ck['bad_epochs'], ck['history'], ck['best_auc']
        patience_anchor = history[-1]['patience_anchor']
        best_auc = max(r['selection_auc'] for r in history)
        best_auc_epoch = next(r['epoch'] for r in history if r['selection_auc'] == best_auc)
    for epoch in range(start, t['epochs']):
        if epoch >= t['min_epochs'] and bad >= t['patience']: break
        tic = time.monotonic(); lr_used = optimizer.param_groups[0]['lr']; model.train()
        order = torch.randperm(len(fit), generator=generator)
        loss_total = 0.
        for indices in order.split(t['batch_size']):
            idx = indices.to(device)
            x = base.aumentar_geometria(fit_x[idx].float()/255, aug, **cfg['augmentation'])
            optimizer.zero_grad(set_to_none=True)
            with base.amp_context(device, scaler.is_enabled()):
                loss = (criterion(model(x).flatten(), y[idx])*w[idx]).mean()/normalizer
            if not torch.isfinite(loss): raise FloatingPointError('BCE no finita')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), t['clip_grad_norm'], error_if_nonfinite=not scaler.is_enabled())
            scaler.step(optimizer); scaler.update()
            loss_total += float(loss.detach())*len(idx)
        fit_m, _ = patient_metrics(fit, cache.predict(model, fit_x), pw)
        sel_m, _ = patient_metrics(selection, cache.predict(model, sel_x), pw)
        score = -sel_m['bce']
        if score > patience_anchor+t['min_delta']:
            patience_anchor, bad = score, 0
        else: bad += 1
        if score > best_score+1e-12:
            best_score = score; cache.save_inference(path/'loss.pt', model, cfg)
        if sel_m['auc'] > best_auc+1e-12:
            best_auc, best_auc_epoch = sel_m['auc'], epoch+1
            cache.save_inference(path/'auc.pt', model, cfg)
        scheduler.step()
        row = {'epoch': epoch+1, 'train_augmented_bce': loss_total/len(fit),
               **{f'fit_{k}': v for k, v in fit_m.items()},
               **{f'selection_{k}': v for k, v in sel_m.items()},
               'auc_gap': fit_m['auc']-sel_m['auc'], 'lr_used': lr_used,
               'bad_epochs': bad, 'patience_anchor': patience_anchor, 'seconds': time.monotonic()-tic}
        history.append(row)
        best_epoch = int(np.argmin([r['selection_bce'] for r in history]))
        base.save_checkpoint(path/'last.pt', model, optimizer, scheduler, scaler, epoch,
                             best_score, bad, cfg, generator, aug, history, best_epoch)
        pd.DataFrame(history).to_csv(path/'history.csv', index=False)
        print(f"{path.relative_to(path.parents[3])} e{epoch+1}: BCE fit/sel {fit_m['bce']:.4f}/{sel_m['bce']:.4f}; "
              f"AUC {fit_m['auc']:.4f}/{sel_m['auc']:.4f}; {row['seconds']:.2f}s", flush=True)
    steps = {int(state['step'].item()) for state in optimizer.state.values()}
    expected = len(history)*int(np.ceil(len(fit)/t['batch_size']))
    if steps != {expected}: raise RuntimeError(f'Pasos AdamW omitidos: {steps} esperado{expected}')
    selections = {}
    for rule in RULES:
        artifact = torch.load(path/f'{rule}.pt', map_location='cpu', weights_only=True)
        if any(not torch.isfinite(v).all() for v in artifact['state_dict'].values()):
            raise FloatingPointError('Peso no finito')
        model.load_state_dict(artifact['state_dict'])
        m, p = patient_metrics(selection, cache.predict(model, sel_x), pw)
        p.to_csv(path/f'selection_{rule}.csv', index=False)
        idx = int(np.argmin([r['selection_bce'] for r in history])) if rule == 'loss' else best_auc_epoch-1
        selections[rule] = {'epoch': idx+1, **m, 'fit_auc': history[idx]['fit_auc'],
                            'auc_gap': history[idx]['auc_gap'], 'fit_bce': history[idx]['fit_bce']}
        expected_m = history[idx]
        if abs(m['bce']-expected_m['selection_bce']) > 1e-6 or abs(m['auc']-expected_m['selection_auc']) > 1e-6:
            raise RuntimeError('Inferencia no reproduce checkpoint elegido')
    files = ['config.json', 'history.csv']+[f'{r}.{ext}' for r in RULES for ext in ('pt',)]
    files += [f'selection_{r}.csv' for r in RULES]
    result = {'status': 'complete', 'loss': loss_name, 'seed': seed, 'epochs': len(history),
              'selected': selections, 'curve_diagnostic': curve_diagnostic(history),
              'optimizer_steps': expected, 'skipped_steps': 0,
              'seconds': float(sum(r['seconds'] for r in history)),
              'hashes': {name: base.sha256(path/name) for name in files}}
    write(path/'summary.json', result)
    del model, optimizer, scheduler, scaler
    return result


def freeze_fold(output, fold, parts):
    choices = {}
    for rule in RULES:
        candidates = []
        for loss in LOSSES:
            paths = [output/'runs'/f'fold_{fold}'/loss/f'seed_{s}' for s in SEEDS]
            p = image._aligned_average([pd.read_csv(p/f'selection_{rule}.csv') for p in paths])
            cal = base.fit_platt(p.label.to_numpy(), p.probability_raw.to_numpy())
            cal['fit_data'] = 'inner_selection_same_pair_same_patient_means'
            cp = base.apply_calibration(p.probability_raw.to_numpy(), cal)
            candidates.append({'loss': loss, 'selection_auc': float(roc_auc_score(p.label, p.probability_raw)),
                'calibration': cal, 'threshold': image.threshold_min_specificity(p.label, cp, .95),
                'secondary_threshold': image.threshold_macro_f1(p.label, cp),
                'weights': {str((p/f'{rule}.pt').relative_to(output)): base.sha256(p/f'{rule}.pt') for p in paths}})
        chosen = max(candidates, key=lambda c: c['selection_auc'])
        choices[rule] = {'selected_loss': chosen['loss'], 'candidates': candidates}
    return {'fold': fold, 'frozen_at': now(), 'rules': choices,
            'counts': {r: int(f.patient_id.nunique()) for r, f in parts.items()}}


def fast_auc(y, p):
    y = np.asarray(y); n1 = y.sum(); n0 = len(y)-n1
    return float((rankdata(p)[y == 1].sum()-n1*(n1+1)/2)/(n0*n1))


def bootstrap(frame, n=2000):
    rng = np.random.default_rng(20261025)
    groups = [g.index.to_numpy() for _, g in frame.groupby(['fold', 'label'])]
    aucs, deltas, control_deltas = [], [], []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        sample = frame.iloc[idx]
        values, refs, controls = [], [], []
        for _, g in sample.groupby('fold'):
            values.append(fast_auc(g.label.to_numpy(), g.probability_raw.to_numpy()))
            refs.append(fast_auc(g.label.to_numpy(), g.cohort_probability.to_numpy()))
            if 'control_probability' in g:
                controls.append(fast_auc(g.label.to_numpy(), g.control_probability.to_numpy()))
        aucs.append(np.mean(values)); deltas.append(np.mean(values)-np.mean(refs))
        if controls: control_deltas.append(np.mean(values)-np.mean(controls))
    result = {'n_bootstrap': n, 'stratification': 'fold × clase, pacientes pareadas candidato/cohorte/control',
            'mean_auc_ci95': np.quantile(aucs, [.025, .975]).tolist(),
            'delta_cohort_ci95': np.quantile(deltas, [.025, .975]).tolist(),
            'limitation': 'Condiciona a estos modelos; no incorpora selección adaptativa ni toda variabilidad de entrenamiento'}
    if control_deltas:
        result['delta_control_ci95_secondary'] = np.quantile(control_deltas, [.025, .975]).tolist()
    return result


def attach_control(frame):
    control = pd.read_csv(CONTROL/'oof_loss.csv').sort_values('patient_id').reset_index(drop=True)
    if control.patient_id.duplicated().any() or not frame.patient_id.equals(control.patient_id):
        raise ValueError('Control y candidato no tienen las mismas pacientes ordenadas')
    for column in ('label', 'fold', 'cohort', 'n_slices'):
        if not frame[column].equals(control[column]):
            raise ValueError(f'Control no comparable en {column}')
    np.testing.assert_allclose(frame.cohort_probability, control.cohort_probability, atol=1e-15, rtol=0)
    frame['control_probability'] = control.probability_raw.to_numpy()
    return frame


def evaluate(output, protocol, samples, choices, device):
    # El guardia se comprueba antes de leer ninguna imagen exterior de inferencia.
    frozen = output/'elecciones_congeladas.json'
    if not frozen.exists() or len(choices) != 5: raise ValueError('Faltan cinco elecciones congeladas')
    all_results = {(rule, loss): [] for rule in RULES for loss in LOSSES}
    primary = {r: [] for r in RULES}
    for choice in choices:
        fold = choice['fold']
        parts = roles_for(samples, pd.read_csv(output/'roles'/f'roles_fold_{fold}.csv'))
        outer = parts['outer']; pixels = cache.load_pixels(outer, device)
        for rule in RULES:
            for candidate in choice['rules'][rule]['candidates']:
                paths = []
                for name, digest in candidate['weights'].items():
                    path = output/name
                    if base.sha256(path) != digest: raise ValueError('Peso congelado alterado')
                    paths.append(path)
                p = aligned_ensemble(paths, outer, pixels, device)
                _, patients = patient_metrics(outer, p)
                # La alineación explícita evita asignar tasas según orden de cortes.
                patient_rows = outer.drop_duplicates('patient_id').set_index('patient_id').loc[patients.patient_id].reset_index()
                patients['cohort_probability'], _ = cohort_predictions(parts['fit'], patient_rows)
                patients['probability_calibrated'] = base.apply_calibration(patients.probability_raw.to_numpy(), candidate['calibration'])
                patients['threshold'] = candidate['threshold']
                patients['secondary_threshold'] = candidate['secondary_threshold']
                patients['prediction'] = (patients.probability_calibrated >= patients.threshold).astype(int)
                patients['secondary_prediction'] = (patients.probability_calibrated >= patients.secondary_threshold).astype(int)
                all_results[rule, candidate['loss']].append(patients)
                if candidate['loss'] == choice['rules'][rule]['selected_loss']:
                    primary[rule].append(patients)
        del pixels
    summaries = {}
    for rule, frames in primary.items():
        frame = pd.concat(frames).sort_values('patient_id').reset_index(drop=True)
        if frame.patient_id.duplicated().any() or len(frame) != 1097: raise ValueError('OOF inválidas')
        frame = attach_control(frame)
        frame.to_csv(output/f'oof_{rule}.csv', index=False)
        fold_auc = [float(roc_auc_score(g.label, g.probability_raw)) for _, g in frame.groupby('fold')]
        cohort_auc = [float(roc_auc_score(g.label, g.cohort_probability)) for _, g in frame.groupby('fold')]
        control_auc = [float(roc_auc_score(g.label, g.control_probability)) for _, g in frame.groupby('fold')]
        metrics = image.metricas_binarias(frame.label, frame.probability_calibrated, .5)
        # Umbrales diferentes por fold ya congelados; no reutilizar .5 para sus clases.
        classification = image.metricas_binarias(frame.label, frame.prediction.astype(float), .5)
        classification = {k: v for k, v in classification.items() if k not in ('roc_auc', 'average_precision', 'brier', 'log_loss', 'threshold')}
        summaries[rule] = {'fold_auc': fold_auc, 'mean_auc': float(np.mean(fold_auc)), 'sd_auc': float(np.std(fold_auc, ddof=1)),
            'pooled_raw_auc': float(roc_auc_score(frame.label, frame.probability_raw)),
            'cohort_fold_auc': cohort_auc, 'cohort_mean_auc': float(np.mean(cohort_auc)),
            'delta_mean_cohort': float(np.mean(fold_auc)-np.mean(cohort_auc)),
            'control_fold_auc': control_auc, 'control_mean_auc': float(np.mean(control_auc)),
            'delta_mean_control': float(np.mean(fold_auc)-np.mean(control_auc)),
            'calibrated_probability_metrics_at_0_5': metrics, 'frozen_policy_classification': classification,
            'raw_metrics_at_0_5': image.metricas_binarias(frame.label, frame.probability_raw, .5),
            'by_cohort': {c: image.metricas_binarias(g.label, g.probability_raw, .5) for c, g in frame.groupby('cohort')},
            'always_no_pcr': image.metricas_binarias(frame.label, np.zeros(len(frame)), .5)}
        if rule == 'loss':
            summaries[rule].update(bootstrap(frame))
            summaries[rule]['point_criterion_passed'] = bool(summaries[rule]['mean_auc'] > .54 and summaries[rule]['delta_mean_cohort'] > 0)
            summaries[rule]['evidence_beyond_cohort'] = bool(summaries[rule]['delta_cohort_ci95'][0] > 0)
    for (rule, loss), frames in all_results.items():
        frame = pd.concat(frames)
        frame.to_csv(output/f'oof_{rule}_{loss}_fija.csv', index=False)
        summaries[f'{rule}_{loss}_fixed'] = {'mean_auc': float(np.mean([roc_auc_score(g.label, g.probability_raw) for _, g in frame.groupby('fold')]))}
    write(output/'comparacion.json', summaries)
    rows = []
    for (cohort, fold), g in pd.concat(primary['loss']).groupby(['cohort', 'fold']):
        rows.append({'cohort': cohort, 'fold': int(fold), 'patients': len(g), 'pcr': int(g.label.sum()),
                     'auc': float(roc_auc_score(g.label, g.probability_raw))})
    pd.DataFrame(rows).to_csv(output/'auc_por_cohorte_y_fold.csv', index=False)
    return summaries


def report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = json.loads((output/'comparacion.json').read_text())
    choices = json.loads((output/'elecciones_congeladas.json').read_text())['folds']
    fig, axes = plt.subplots(5, 2, figsize=(13, 19))
    records = []
    for fold in range(5):
        choice = choices[fold]['rules']['loss']; loss = choice['selected_loss']
        for col, seed in enumerate(SEEDS):
            path = output/'runs'/f'fold_{fold}'/loss/f'seed_{seed}'
            h = pd.read_csv(path/'history.csv'); summary = json.loads((path/'summary.json').read_text())
            selected = summary['selected']['loss']; ax = axes[fold, col]
            ax.plot(h.epoch, h.fit_bce, label='Fit, eval')
            ax.plot(h.epoch, h.selection_bce, label='Selección, eval')
            ax.axvspan(12, 14, color='grey', alpha=.1); ax.axvline(selected['epoch'], color='black', linestyle=':', label='Checkpoint')
            ax.set(title=f"Fold{fold} · {loss} · semilla{seed} · época{selected['epoch']}", xlabel='Época', ylabel='BCE por paciente')
            ax.legend(fontsize=8)
            records.append({'fold': fold, 'loss': loss, 'seed': seed, **selected, **summary['curve_diagnostic']})
    fig.suptitle('Pérdidas sin aumentos/dropout; misma BCE en fit y selección. Datos sin suavizar.')
    fig.tight_layout(rect=(0, 0, 1, .98)); fig.savefig(output/'01_perdidas.png', dpi=130); plt.close(fig)
    fig, axes = plt.subplots(5, 2, figsize=(13, 19))
    for fold in range(5):
        loss = choices[fold]['rules']['loss']['selected_loss']
        for col, seed in enumerate(SEEDS):
            path = output/'runs'/f'fold_{fold}'/loss/f'seed_{seed}'; h = pd.read_csv(path/'history.csv')
            selected = json.loads((path/'summary.json').read_text())['selected']['loss']['epoch']
            ax = axes[fold, col]; ax.plot(h.epoch, h.fit_auc, label='Fit'); ax.plot(h.epoch, h.selection_auc, label='Selección')
            ax.axvline(selected, linestyle=':', color='black'); ax.set(title=f'Fold{fold} · {loss} · semilla{seed}', xlabel='Época', ylabel='AUC paciente', ylim=(.3, 1)); ax.legend()
    fig.tight_layout(); fig.savefig(output/'02_auc_epocas.png', dpi=130); plt.close(fig)
    frame = pd.read_csv(output/'oof_loss.csv'); fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for fold, g in frame.groupby('fold'):
        x, y, _ = roc_curve(g.label, g.probability_raw); axes[0].plot(x, y, label=f'Fold{fold}: {roc_auc_score(g.label, g.probability_raw):.3f}')
    axes[0].plot([0, 1], [0, 1], ':', color='grey'); axes[0].set(xlabel='1−Especificidad', ylabel='Sensibilidad', title='ROC cruda por fold'); axes[0].legend()
    for column, label in [('probability_raw', 'CNN agrupada, secundaria'), ('cohort_probability', 'Solo cohorte')]:
        precision, recall, _ = precision_recall_curve(frame.label, frame[column]); axes[1].step(recall, precision, where='post', label=label)
    axes[1].axhline(frame.label.mean(), linestyle=':', color='grey'); axes[1].set(xlabel='Recall', ylabel='Precisión', title='PR por paciente, sin interpolar'); axes[1].legend()
    x = np.arange(5); axes[2].plot(x, results['loss']['fold_auc'], 'o-', label='CNN espacial'); axes[2].plot(x, results['loss']['control_fold_auc'], 'o-', label='Control cinético'); axes[2].plot(x, results['loss']['cohort_fold_auc'], 'o-', label='Cohorte'); axes[2].axhline(.54, linestyle=':'); axes[2].set(xlabel='Fold', ylabel='AUC', title='Comparación en las mismas pacientes'); axes[2].legend()
    fig.tight_layout(); fig.savefig(output/'03_roc_pr_cohortes.png', dpi=130); plt.close(fig)
    pd.DataFrame(records).to_csv(output/'diagnostico_curvas_seleccionadas.csv', index=False)
    primary = results['loss']; gaps = [r['auc_gap'] for r in records]
    joined = sum(bool(r['joint_improvement']) for r in records)
    protocol = json.loads((output/'protocolo.json').read_text())
    summaries = [json.loads(p.read_text()) for p in output.glob('runs/*/*/*/summary.json')]
    text = f"""# CNN espacial compacta por corte: comparación prefijada

Nueva CNN propia de **{protocol['parameters']:,} parámetros**, desde cero, PRE/EARLY/LATE y diferencias firmadas. Tres bloques16/32/64, vista global128, cuadrícula4×4, cabeza32 y un logit por corte. Media de probabilidades de todos los cortes fuera de la red. Sin atención/max entre cortes, clínica, EMA, mixup ni label smoothing.20 entrenamientos, cinco folds, dos semillas, BCE normal/ponderada. Datos originales conservados, incluidas incidencias auditadas. Test y validación privada cerrados; desarrollo con antecedentes adaptativos.

Se cambia únicamente la arquitectura, definida completamente en protocolo.json. Roles, semillas, batch32, LR0.0003, WD0.01, dropout.35/.1, BN/ReLU, aumentos geométricos, máximo40/mínimo14/paciencia8, contribución1/n_cortes y normalizador fijo, selección por pérdida, elección de BCE por AUC interna y políticas coinciden con el control cerrado. No se atribuye el cambio a un único componente de la nueva arquitectura. El piloto espacial histórico con atención/EMA tenía otro contrato y sus cifras internas0.595/0.611 no son resultados de este modelo.

La referencia de solo cohorte aprende prevalencias exclusivamente en fit: mediaAUC **{primary['cohort_mean_auc']:.6f}**. El valor0,54 es orientativo; aquí se compara contra la referencia realmente calculada.

La CNN con checkpoint por BCE interna tiene mediaAUC **{primary['mean_auc']:.6f} ± {primary['sd_auc']:.6f}**; IC95% condicionado {primary['mean_auc_ci95']}. Diferencia respecto a cohorte **{primary['delta_mean_cohort']:+.6f}**, IC95% pareado {primary['delta_cohort_ci95']}. Criterio puntual superado: **{primary['point_criterion_passed']}**; evidencia condicionada de mejora sobre cohorte: **{primary['evidence_beyond_cohort']}**. AUC agrupada cruda secundaria {primary['pooled_raw_auc']:.6f}; no sustituye la media por fold.

Control cinético cerrado: media **{primary['control_mean_auc']:.6f}**. Delta espacial−control **{primary['delta_mean_control']:+.6f}**, IC95% pareado secundario {primary['delta_control_ci95_secondary']}. Se compara sobre las mismas pacientes, sin reentrenar ni escoger otra receta del control. La comparación contra cohorte es principal; no se interpretan dos intervalos secundarios como prueba conjunta corregida.

## Curvas y sobreajuste

Ambas pérdidas se calculan en eval, sin aumentos ni dropout, después de la media de todos los cortes por paciente, usando idéntico pos_weight dentro de cada run. La BCE de optimización por corte se conserva separada; no se compara directamente con la pérdida por paciente. Las pérdidas ponderadas y normales no se comparan por magnitud para elegir el brazo.

Se observaron al menos14 épocas; se puede seleccionar una época anterior. **{joined}/10** modelos elegidos muestran pendiente negativa en ambas pérdidas en1–14 y media12–14 inferior a1–3. Brecha AUC fit−selección mediana en los checkpoints **{np.median(gaps):.6f}**. Curvas próximas pueden indicar infraajuste y no demuestran discriminación. Las gráficas conservan cada punto observado; no se exige una bajada monótona ni se altera LR para alcanzar una época estética.

En las20 trayectorias, **{sum(s['curve_diagnostic']['joint_improvement'] for s in summaries)}/20** muestran esa mejora conjunta. Mediana de época elegida en las diez redes principales **{np.median([r['epoch'] for r in records]):.1f}**. Se completaron **{sum(s['epochs'] for s in summaries)} épocas** y **{sum(s['optimizer_steps'] for s in summaries)} pasos AdamW**, sin pasos omitidos. Una mejora conjunta1–14 puede corresponder solo al ajuste de prevalencia: se exige analizar además AUC y su comparación exterior con cohorte.

![Pérdidas](01_perdidas.png)
![AUC](02_auc_epocas.png)
![ROC/PR/cohorte](03_roc_pr_cohortes.png)

## Métricas y comparación

La regla de checkpoint por AUC en las mismas trayectorias obtiene media **{results['auc']['mean_auc']:.6f}**. Es un diagnóstico prefijado, con la misma parada por pérdida: no una comparación causal de dos recetas completas ni una autorización para elegirla después de mirar exterior.

BCE normal fija: {results['loss_normal_fixed']['mean_auc']:.6f}; ponderada fija: {results['loss_weighted_fixed']['mean_auc']:.6f}. La pérdida principal de cada fold se eligió en selección, no exterior. Todas las cinco parejas y políticas se congelaron antes de inferir exterior. Selection se reutiliza para checkpoint, pérdida, calibración y umbral: sus métricas son optimistas. La CV evalúa el procedimiento completo con745–746 pacientes fit y132 de selección por fold.

Política principal, especificidad objetivo95% en selección: TN{primary['frozen_policy_classification']['tn']}, FP{primary['frozen_policy_classification']['fp']}, FN{primary['frozen_policy_classification']['fn']}, TP{primary['frozen_policy_classification']['tp']}; sensibilidad{primary['frozen_policy_classification']['recall_pcr']:.2%}, especificidad{primary['frozen_policy_classification']['recall_no_pcr']:.2%}, F1pCR{primary['frozen_policy_classification']['f1_pcr']:.4f}, macroF1{primary['frozen_policy_classification']['macro_f1']:.4f}. Decir siempre no pCR: macroF1{primary['always_no_pcr']['macro_f1']:.4f}. Esto no constituye una recomendación clínica ni selección de otra política con exterior.

Comparación completa por cohorte, AP, Brier/logloss calibrados, umbral0,5 crudo y política congelada en comparacion.json y oof_loss.csv. La política secundaria macroF1 y sus predicciones también se conservaron sin escogerla mirando exterior. Umbrales de validación no garantizan especificidad95% exterior.

AUC por cohorte y fold en auc_por_cohorte_y_fold.csv; los grupos SPY1 tienen pocos positivos. Las AUC agrupadas por cohorte combinan puntuaciones de modelos distintos y son secundarias, no sustituyen las medidas dentro de cada fold.

## Límites y decisión

No es una nueva evaluación ciega: train y el test histórico ya se han examinado en el proyecto. Bootstrap2000 por fold/clase, pareado por paciente, condicionado a estos modelos; no incorpora toda variabilidad de selección/entrenamiento. Una estimación superior a0,54 no demuestra por sí sola señal tumoral; se informa el comparador de cohorte y AUC dentro de cada cohorte. No se promete0,60. No se activa ni publica un modelo, no se entrena refit y no se inicia otra búsqueda automática.

La preferencia de curvas conjuntamente descendentes se incorpora como diagnóstico comprobable, no como prueba suficiente de generalización. [Cawley y Talbot](https://www.jmlr.org/papers/v11/cawley10a.html) documentan el sesgo por optimizar repetidamente la selección; repetir CV sobre este train no borra su historial.
"""
    (output/'INFORME_RESULTADOS.md').write_text(text)


def execute(output=OUTPUT):
    output = Path(output); protocol = verify(output)
    if (output/'comparacion.json').exists(): raise ValueError('Ensayo terminado: no reabrir exterior')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA requerida')
    device = torch.device('cuda'); torch.set_num_threads(4)
    samples, _ = audit.train_metadata(DATA_DIR)
    state = {'status': 'training', 'pid': os.getpid(), 'started_at': now(), 'runs_completed': 0, 'planned_runs': 20}
    write(output/'estado.json', state); write(output/'entorno.json', base.environment(device))
    choices = []
    for fold in range(5):
        parts = roles_for(samples, pd.read_csv(output/'roles'/f'roles_fold_{fold}.csv'))
        fit_x, sel_x = (cache.load_pixels(parts[r], device) for r in ('fit', 'selection'))
        for loss in LOSSES:
            for seed in SEEDS:
                state.update(fold=fold, loss=loss, seed=seed, updated_at=now()); write(output/'estado.json', state)
                path = output/'runs'/f'fold_{fold}'/loss/f'seed_{seed}'
                train_run(parts['fit'], parts['selection'], fit_x, sel_x, protocol['config'], loss, seed, path, device)
                state['runs_completed'] += 1; state['updated_at'] = now(); write(output/'estado.json', state)
        choices.append(freeze_fold(output, fold, parts))
        write(output/f'eleccion_fold_{fold}.json', choices[-1]); del fit_x, sel_x
    write(output/'elecciones_congeladas.json', {'frozen_at': now(), 'folds': choices})
    verify(output); state.update(status='evaluating_frozen', updated_at=now()); write(output/'estado.json', state)
    result = evaluate(output, protocol, samples, choices, device)
    report(output); verify(output)
    write(output/'verificacion_final.json', {'verified_at': now(), 'sources_and_versions_unchanged': True,
        'patient_roles_disjoint': True, 'runs': 20, 'test_images_loaded': 0, 'private_images_loaded': 0,
        'no_push': True, 'no_model_activation': True,
        'artifacts': {str(p.relative_to(output)): base.sha256(p) for p in output.glob('*') if p.is_file() and p.name not in ('estado.json', 'verificacion_final.json') and p.suffix != '.log'}})
    state.update(status='complete', updated_at=now(), mean_auc=result['loss']['mean_auc'],
                 delta_cohort=result['loss']['delta_mean_cohort']); write(output/'estado.json', state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['comprobar', 'preparar', 'entrenar', 'informe'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        if args.action == 'comprobar': print(json.dumps(synthetic_learning_check(), ensure_ascii=False), flush=True)
        elif args.action == 'preparar': print(json.dumps(prepare(args.output), ensure_ascii=False), flush=True)
        elif args.action == 'entrenar': execute(args.output)
        else: report(args.output)
    except Exception:
        if args.action == 'entrenar': write(args.output/'error.json', {'at': now(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
