"""Regularización de estilo de la CNN cinética, sin pesos preentrenados.

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

from . import entrenamiento as base
from . import entrenamiento_cinetico as cnn
from . import entrenamiento_imagen as image
from . import cierre_cinetico as cache
from . import revision_train as audit
from .paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

OUTPUT = RESULTS_DIR/'30_regularizacion_estilo_20261026'
CONTROL = RESULTS_DIR/'27_criterio_docente_20261025'
ROLES = audit.OUTPUT/'protocolo_cinco_folds'
SEEDS, LOSSES, RULES = (42, 2026), ('normal', 'weighted'), ('loss', 'auc')
write, now = cache.write, cache.now


def style_config():
    return {'p': .5, 'alpha': .1, 'eps': 1e-6, 'blocks': [0, 1],
            'shuffle': 'random_fit_slices_same_scale_shared_across_three_views',
            'statistics': 'fp32_spatial_mean_and_unbiased_variance_detached',
            'inference': 'identity'}


class GroupedMixStyle(nn.Module):
    """MixStyle (Zhou et al., ICLR2021), con las tres vistas acopladas.

    Implementación propia de las ecuaciones del artículo. Conserva la etiqueta
    y el mapa estandarizado de cada corte; solo interpola media/desviación de
    características. No recibe etiquetas, cohorte ni datos ajenos a fit.
    """
    def __init__(self, p=.5, alpha=.1, eps=1e-6):
        super().__init__()
        if not 0 <= p <= 1 or alpha <= 0 or eps <= 0:
            raise ValueError('MixStyle inválido')
        self.p, self.alpha, self.eps = p, alpha, eps

    def mix(self, x, permutation, coefficient):
        if x.ndim != 4 or len(x) % 3:
            raise ValueError('Se requieren tres vistas consecutivas por corte')
        n, c, h, w = len(x)//3, *x.shape[1:]
        if len(permutation) != n or len(coefficient) != n:
            raise ValueError('Una pareja y un coeficiente por corte')
        z = x.float().reshape(n, 3, c, h, w)
        mu = z.mean((-2, -1), keepdim=True).detach()
        sig = (z.var((-2, -1), keepdim=True, correction=1)+self.eps).sqrt().detach()
        lam = coefficient.to(device=x.device, dtype=torch.float32).reshape(n, 1, 1, 1, 1)
        permutation = permutation.to(x.device)
        new_mu = lam*mu+(1-lam)*mu[permutation]
        new_sig = lam*sig+(1-lam)*sig[permutation]
        return (((z-mu)/sig)*new_sig+new_mu).reshape_as(x).to(x.dtype)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        if x.ndim != 4 or len(x) % 3:
            raise ValueError('Agrupación de vistas inválida')
        n = len(x)//3
        if n < 2 or torch.rand(()).item() >= self.p:
            return x
        # RNG CPU guardado por save_checkpoint; no altera RNG CUDA del dropout.
        permutation = torch.randperm(n)
        coefficient = torch.distributions.Beta(self.alpha, self.alpha).sample((n,))
        return self.mix(x, permutation, coefficient)


class CNNMixStyle(cnn.CNNMultiescalaCinetica):
    """Mismos593521 parámetros y state_dict que la CNN original; solo regulariza fit."""
    def __init__(self, config=None):
        super().__init__(config)
        self.style1, self.style2 = GroupedMixStyle(), GroupedMixStyle()

    def forward(self, x):
        z = self.encode_views(x)
        for k, block in enumerate(self.backbone):
            z = block(z)
            if k == 0: z = self.style1(z)
            elif k == 1: z = self.style2(z)
        z = self.spatial_dropout(z)
        z = torch.cat((self.avgpool(z).flatten(1), self.maxpool(z).flatten(1)), dim=1)
        return self.classifier(z.reshape(len(x), len(self.crop_sizes)*z.shape[1]))


def make_model(config):
    return CNNMixStyle(config)


def synthetic_learning_check():
    """Comprobación CUDA/AMP/BCE y exportación con datos enteramente sintéticos."""
    base.seed_everything(42, True); torch.set_num_threads(4)
    device = torch.device('cuda')
    cfg = cnn.configuracion(); model = make_model(cfg).to(device)
    x = torch.rand(4, 3, 256, 256, device=device)
    y = torch.tensor([0., 1., 0., 1.], device=device)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    for step in range(4):
        model.train(); optimizer.zero_grad(set_to_none=True)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2. if step % 2 else 1., device=device))
        with base.amp_context(device, True): loss = criterion(model(x).flatten(), y)
        if not torch.isfinite(loss): raise FloatingPointError('BCE sintética')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update()
    if {int(v['step'].item()) for v in optimizer.state.values()} != {4}:
        raise ValueError('AdamW no avanzó cuatro pasos')
    changed = sum(not torch.equal(v, before[k]) for k, v in model.named_parameters())
    if not changed: raise ValueError('No cambió ningún peso')
    plain = cnn.CNNMultiescalaCinetica(cfg).to(device)
    plain.load_state_dict(model.state_dict(), strict=True); plain.eval(); model.eval()
    with torch.inference_mode():
        error = float((plain(x)-model(x)).abs().max())
    if error != 0: raise ValueError('MixStyle debe ser identidad en inferencia')
    return {'synthetic_only': True, 'patient_images_loaded': 0,
            'parameters': image.numero_parametros(model), 'adamw_steps': 4,
            'changed_parameter_tensors': changed, 'base_inference_max_error': error,
            'bce_final': float(loss.detach()), 'finite': True}


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
    cfg['name'] = 'cnn_cinetica_mixstyle'
    cfg['feature_augmentation'] = style_config()
    cfg['hypothesis'] = 'MixStyle en los dos bloques tempranos puede reducir dependencia de apariencia de adquisición sin ampliar la red ni modificar etiquetas; hipótesis no demostrada para pCR'
    cfg['history_plot_title'] = 'CNN cinética + MixStyle: fit y selección por paciente'
    if cfg['training'] != control_protocol['config']['training'] or cfg['augmentation'] != control_protocol['config']['augmentation']:
        raise ValueError('No cambiar entrenamiento ni aumentos geométricos')
    parameters = image.numero_parametros(make_model(cfg))
    if parameters != 593521 or cfg['model'] != control_protocol['config']['model']:
        raise ValueError('La arquitectura base no puede cambiar')
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
    protocol = {'created_at': now(), 'authorized_by_user': 'Investiga en internet e intenta mejorar resultados (26/10/2026)',
        'config': cfg, 'parameters': parameters, 'runs': 20, 'seeds': list(SEEDS), 'losses': list(LOSSES),
        'control': {'path': str(CONTROL.relative_to(PROJECT_ROOT)), 'code_commit': 'db845626bfca69276217074256fc131c0bffbc01',
                    'mean_auc': .5562698827127401, 'control_fit_retrained': False,
                    'contract': 'Mismos roles, semillas, BCE por corte1/n, paciente media, selección por pérdida, parada, LR/WD, lote, aumentos y política'},
        'only_candidate': 'MixStyle p0.5 alpha0.1 eps1e-6 tras bloques0/1; mismos parámetros, vistas/escala emparejadas, sin mezclar etiquetas ni imágenes; identidad en eval',
        'research': [{'url': 'https://arxiv.org/html/2104.02008v1', 'scope': 'MixStyle original, clasificación natural/reidentificación/RL; no prueba de mejora de pCR'}, {'url': 'https://pubmed.ncbi.nlm.nih.gov/38978834/', 'scope': 'Extensión CSU en clasificación médica/Camelyon17; no se implementa CSU ni demuestra mejora en estas RM'}],
        'gate': {'mean_auc_min': .60, 'delta_control_min': .01, 'delta_control_ci95_lower_positive': True, 'folds_improved_min': 3, 'action': 'Solo motivaría réplica prefijada, no activación ni test automático'},
        'novelty': 'No repetir bias-field, mixup, SAM, FiLM, GroupNorm ni reducción de capacidad; se mantienen BCE, PNG/255 y media por paciente',
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
        'primary': 'Delta media AUC cruda de cinco folds candidato-control; ensemble de dos semillas y checkpoints por pérdida',
        'secondary': 'Cohorte; regla bestAUC misma trayectoria, BCE fijas, AUC agrupada/calibrada y por cohorte, AP y umbrales',
        'criteria': {'point_estimate': 'mediaAUC>0.54 y delta positivo frente a cohorte en los mismos folds',
                     'evidence_beyond_cohort': 'extremo inferiorIC95% pareado de delta media>0, condicionado a estos modelos',
                     'curves': 'Describir mejoría conjunta1–14 (pendiente negativa y media12–14 menor que1–3), épocas elegidas y brechas; no garantiza generalización'},
        'data_policy': 'Conservar todos los PNG originales; ISPY1_1139 EARLY=LATE y brillo ISPY1_1193 documentados, sin corrección/exclusión inferida',
        'limits': ['Desarrollo adaptativo, train ya observado; no se presenta como confirmación externa',
                   'Las estadísticas de características pueden contener señal biológica; perturbarlas también podría perjudicar pCR',
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
        result['delta_control_ci95'] = np.quantile(control_deltas, [.025, .975]).tolist()
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
            summaries[rule]['folds_improved_vs_control'] = int(sum(a>b for a,b in zip(fold_auc, control_auc)))
            summaries[rule]['replication_gate'] = bool(summaries[rule]['mean_auc'] >= .60 and summaries[rule]['delta_mean_control'] >= .01 and summaries[rule]['delta_control_ci95'][0] > 0 and summaries[rule]['folds_improved_vs_control'] >= 3)
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
    x = np.arange(5); axes[2].plot(x, results['loss']['fold_auc'], 'o-', label='CNN + MixStyle'); axes[2].plot(x, results['loss']['control_fold_auc'], 'o-', label='Control cinético'); axes[2].plot(x, results['loss']['cohort_fold_auc'], 'o-', label='Cohorte'); axes[2].axhline(.54, linestyle=':'); axes[2].set(xlabel='Fold', ylabel='AUC', title='Comparación en las mismas pacientes'); axes[2].legend()
    fig.tight_layout(); fig.savefig(output/'03_roc_pr_cohortes.png', dpi=130); plt.close(fig)
    pd.DataFrame(records).to_csv(output/'diagnostico_curvas_seleccionadas.csv', index=False)
    primary = results['loss']; gaps = [r['auc_gap'] for r in records]
    joined = sum(bool(r['joint_improvement']) for r in records)
    protocol = json.loads((output/'protocolo.json').read_text())
    summaries = [json.loads(p.read_text()) for p in output.glob('runs/*/*/*/summary.json')]
    text = f"""# MixStyle en la CNN cinética: comparación prefijada

CNN propia de **{protocol['parameters']:,} parámetros**, desde cero, PRE/EARLY/LATE y diferencias firmadas. Misma arquitectura multiescala V007; se regularizan estadísticas de características después de los primeros dos bloques con MixStyle, p0,5 y Beta(0,1;0,1). En inferencia es identidad. Las tres escalas de un corte comparten pareja y coeficiente; cada vista usa estadísticas de su misma escala. No se mezclan fases, píxeles ni etiquetas de pacientes. No se usan clínica, máscaras, preentrenados ni cohortes como entrada. Se permite que una permutación conserve algún corte o empareje cortes de la misma paciente, igual que el barajado aleatorio; no se fuerza otra cohorte.

20 entrenamientos, cinco folds, dos semillas, BCE normal/ponderada. Roles, batch32, LR0,0003, WD0,01, dropout0,35/0,1, BN/ReLU, aumentos geométricos, máximo40/mínimo14/paciencia8, contribución1/n_cortes y normalizador fijo, selección por pérdida, elección de BCE por AUC interna y políticas coinciden con el control cerrado. Se reutiliza el control sin reentrenarlo. Las inicializaciones son iguales y se documenta la variabilidad estocástica; un único par de semillas no cubre toda la variabilidad. Datos originales conservados, incluidas incidencias auditadas. Test y validación privada cerrados; desarrollo con antecedentes adaptativos.

[Zhou et al., ICLR2021](https://arxiv.org/html/2104.02008v1) sustentan la idea de mezclar estadísticas tempranas. Sus benchmarks no son pCR ni estas RM. [CSU,2024](https://pubmed.ncbi.nlm.nih.gov/38978834/) evalúa una extensión distinta en clasificación médica; aquí no se implementa CSU. Esta adaptación multiescala es una hipótesis, sin mejora garantizada. Estadísticas enFP32, varianza muestral con epsilon1e-6 y gradiente detenido; entrenamiento desde cero y entrada originalPNG/255. La implementación es propia de las ecuaciones, sin descargar redes o pesos.

La referencia de solo cohorte aprende prevalencias exclusivamente en fit: mediaAUC **{primary['cohort_mean_auc']:.6f}**. El valor0,54 es orientativo; aquí se compara contra la referencia realmente calculada.

La CNN con checkpoint por BCE interna tiene mediaAUC **{primary['mean_auc']:.6f} ± {primary['sd_auc']:.6f}**; IC95% condicionado {primary['mean_auc_ci95']}. Diferencia respecto a cohorte **{primary['delta_mean_cohort']:+.6f}**, IC95% pareado {primary['delta_cohort_ci95']}. Criterio puntual superado: **{primary['point_criterion_passed']}**; evidencia condicionada de mejora sobre cohorte: **{primary['evidence_beyond_cohort']}**. AUC agrupada cruda secundaria {primary['pooled_raw_auc']:.6f}; no sustituye la media por fold.

Control cinético cerrado: media **{primary['control_mean_auc']:.6f}**. Delta MixStyle−control **{primary['delta_mean_control']:+.6f}**, IC95% pareado {primary['delta_control_ci95']}. Se compara sobre las mismas pacientes, sin reentrenar ni escoger otra receta del control. El contraste principal es candidato−control. Cohorte es un comparador secundario. Gate prefijado de réplica: media≥0,60, delta≥0,01, límite inferiorIC95%>0 y mejoría en3/5 folds. Cumplido: **{primary['replication_gate']}**. No confirma generalización ni habilita test automáticamente.

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
