"""CNN local con reducciones suavizadas: media espacial frente a atención por corte.

Solo PNG de train. Las dos recetas y cuarenta entrenamientos se fijan antes de
usar exterior. Una paciente conserva la media de probabilidades de sus cortes.
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

OUTPUT = RESULTS_DIR/'31_preservacion_espacial_20261026'
CONTROL = RESULTS_DIR/'27_criterio_docente_20261025'
ROLES = audit.OUTPUT/'protocolo_cinco_folds'
PROFILES = ('uniforme', 'atencion')
SEEDS, LOSSES, RULES = (42, 2026), ('normal', 'weighted'), ('loss', 'auc')
write, now = cache.write, cache.now


def model_config(profile='uniforme'):
    if profile not in PROFILES: raise ValueError('Perfil espacial desconocido')
    return {'architecture': 'native256_kinetic_smooth_local_cnn_v1',
            'input_phases': ['PRE', 'EARLY', 'LATE'],
            'temporal_channels': ['PRE', 'EARLY', 'LATE', 'EARLY-PRE', 'LATE-PRE', 'EARLY-LATE'],
            'input_size': 256, 'backbone_channels': [16, 32, 64, 96],
            'conv_per_block': 2, 'conv_stride': 1, 'normalization': 'batch',
            'downsampling': 'max2_stride1_then_fixed_binomial3_stride2_zero_padding1',
            'feature_grid': 16, 'hidden': 32, 'dropout': .35, 'spatial_dropout': .10,
            'pooling': profile, 'attention_mixture': .5, 'attention_temperature': 2.,
            'gate_initialization': 'zero_uniform', 'clinical': None,
            'slice_output': 'one_logit', 'patient_aggregation': 'mean_probabilities'}


class SmoothDownsample(nn.Module):
    """Máximo denso, filtro binomial fijo y submuestreo; inspirado en Zhang2019."""
    def __init__(self, channels):
        super().__init__()
        v=torch.tensor([1., 2., 1.]); kernel=torch.outer(v,v)/16.
        self.register_buffer('filter',kernel[None,None].repeat(channels,1,1,1))
        self.channels=channels
    def forward(self,x):
        x=F.max_pool2d(x,2,stride=1)
        return F.conv2d(x,self.filter,stride=2,padding=1,groups=self.channels)


class CNNLocalCinetica(nn.Module):
    """CNN propia, sin backbone externo; atención solo dentro de un corte 2D."""
    def __init__(self,config=None):
        super().__init__()
        self.config=deepcopy(config or model_config())
        self.config=self.config.get('model',self.config)
        profile=self.config.get('pooling')
        if self.config!=model_config(profile): raise ValueError('Configuración fuera del protocolo local')
        self.blocks=nn.ModuleList()
        for cin,cout in zip([6,16,32,64],[16,32,64,96]):
            self.blocks.append(nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),
                nn.BatchNorm2d(cout),nn.ReLU(inplace=True),
                nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),SmoothDownsample(cout)))
        self.spatial_dropout=nn.Dropout2d(.1)
        self.classifier=nn.Sequential(nn.Dropout(.35),nn.Linear(96,32),nn.ReLU(inplace=True),
                                      nn.Dropout(.35),nn.Linear(32,1))
        self.gate=nn.Conv2d(96,1,1,bias=False)
        self.apply(image.CNNImagenFases._initialize)
        nn.init.zeros_(self.gate.weight)
        self.gate.weight.requires_grad_(profile=='atencion')

    def encode(self,x):
        if x.ndim!=4 or tuple(x.shape[1:])!=(3,256,256):
            raise ValueError('Un corte por entrada [N,3,256,256]')
        x=cnn.CNNMultiescalaCinetica.temporal_maps(x)
        for block in self.blocks: x=block(x)
        if x.shape[-2:]!=(16,16): raise ValueError('Cuadrícula inesperada')
        return self.spatial_dropout(x)

    def spatial_weights(self,z):
        scores=self.gate(z).flatten(2).float()/self.config['attention_temperature']
        learned=scores.softmax(-1)
        return .5/learned.shape[-1]+.5*learned

    def forward(self,x):
        z=self.encode(x)
        weights=self.spatial_weights(z)
        pooled=(z.flatten(2).float()*weights).sum(-1)
        return self.classifier(pooled.to(z.dtype))


def make_model(config): return CNNLocalCinetica(config)


def synthetic_learning_check():
    device=torch.device('cuda');torch.set_num_threads(4)
    yy,xx=torch.meshgrid(torch.arange(256,device=device),torch.arange(256,device=device),indexing='ij')
    stripes=((xx//8)%2).float();checks=(((xx//8)+(yy//8))%2).float()
    patterns=torch.stack([stripes,checks]*4)
    x=torch.stack([.15+patterns*.05,.2+patterns*.3,.2+patterns*.2],1)
    y=torch.tensor([0.,1.]*4,device=device)
    reports={}
    for profile in PROFILES:
        base.seed_everything(42,True)
        model=make_model(model_config(profile)).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
        losses=[];start=time.monotonic()
        for step in range(40):
            model.train();optimizer.zero_grad(set_to_none=True)
            criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1. if step%2==0 else 2.,device=device))
            with base.amp_context(device,True):loss=criterion(model(x).flatten(),y)
            if not torch.isfinite(loss):raise FloatingPointError('BCE sintética no finita')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
        if {int(v['step'].item()) for v in optimizer.state.values()}!={40}:raise ValueError('Pasos sintéticos omitidos')
        if not np.mean(losses[-8:])<np.mean(losses[:8]):raise ValueError('No aprende el patrón sintético')
        model.eval()
        with torch.inference_mode():prob=model(x).sigmoid().flatten().cpu().numpy()
        if not np.isfinite(prob).all():raise FloatingPointError('Inferencia sintética')
        reports[profile]={'parameters_trainable':image.numero_parametros(model),
            'parameters_total':sum(p.numel() for p in model.parameters()),'steps':40,
            'first8_loss':float(np.mean(losses[:8])),'last8_loss':float(np.mean(losses[-8:])),
            'synthetic_auc':float(roc_auc_score(y.cpu(),prob)),
            'seconds':time.monotonic()-start,'finite':True}
    return {'synthetic_only':True,'patient_images_loaded':0,'profiles':reports}


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

def prepare_one(output, profile):
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
    cfg['name'] = 'cnn_local_cinetica_'+profile
    cfg['model'] = model_config(profile)
    cfg['hypothesis'] = 'Preservar resolución nativa con reducciones suaves y una cuadrícula16x16; comparar media espacial frente a atención acotada dentro del corte'
    cfg['history_plot_title'] = 'CNN local '+profile+': fit y selección por paciente'
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
    protocol = {'created_at': now(), 'authorized_by_user': 'Sigue investigando y mejorando; toca lo que quieras para mejorar (26/10/2026)',
        'config': cfg, 'parameters': parameters, 'runs': 20, 'seeds': list(SEEDS), 'losses': list(LOSSES),
        'control': {'path': str(CONTROL.relative_to(PROJECT_ROOT)), 'code_commit': 'db845626bfca69276217074256fc131c0bffbc01',
                    'mean_auc': .5562698827127401, 'control_fit_retrained': False,
                    'contract': 'Mismos roles, semillas, BCE por corte1/n, paciente media, selección por pérdida, parada, LR/WD, lote, aumentos y política'},
        'profile': profile,
        'only_candidate': 'CNN propia de una vista nativa256, cuatro bloques16/32/64/96, reducción suavizada y mapa16x16; pooling '+profile,
        'architecture_table': {'input': [3,256,256], 'derived': [6,256,256], 'blocks': [[16,128,128],[32,64,64],[64,32,32],[96,16,16]], 'head': [96,32,1]},
        'historical_spatial_warning': 'La cuadrícula4x4 con vista128 del ensayo28 y las redes de MIL anteriores no son estas recetas; aquí no se aprende la agregación entre cortes',
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
        'secondary': 'Dos contrastes principales frente a control cerrado con CI97.5 Bonferroni2; atención-uniforme secundaria; regla AUC secundaria',
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
                   'Familia de40 ejecuciones, sin cambiar recetas; réplica solo según protocolo familiar congelado'],
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

def bootstrap(frame, n=4000):
    rng = np.random.default_rng(20261026)
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
        result['delta_control_ci97_5_bonferroni2'] = np.quantile(control_deltas, [.0125, .9875]).tolist()
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


def prepare(output=OUTPUT):
    output=Path(output)
    if output.exists(): raise ValueError('Familia existente: no sobrescribir')
    output.mkdir(parents=True)
    children={}
    for profile in PROFILES:
        child=output/profile
        result=prepare_one(child,profile)
        children[profile]={'protocol_sha256':base.sha256(child/'protocolo.json'),
                           'parameters_trainable':result['parameters']}
    family={'created_at':now(),'runs':40,'profiles':children,'test_images_loaded':0,
        'research':[
            {'url':'https://proceedings.mlr.press/v97/zhang19a.html','use':'Filtro fijo previo a submuestreo; evidencia natural, no pCR'},
            {'url':'https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.php','use':'Motiva atención espacial; no implementamos CBAM ni importamos su red'},
            {'url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC8804673/','use':'Estudio propio de pooling en otras tareas médicas; no evidencia directa en pCR'}],
        'primary':'Dos contrastes, media AUC cruda de cada receta menos control27, mismas pacientes y roles; CI97.5 Bonferroni2',
        'secondary':'Atención menos uniforme pareado; cohortes, AUC agrupada cruda/calibrada, BCE fijas, checkpoint AUC y políticas',
        'freeze':'Entrenar las dos recetas y congelar diez elecciones antes de inferir cualquier exterior de esta familia',
        'bootstrap':{'replicates':4000,'seed':20261026,'stratification':'patient paired within fold x class','family_hypotheses':2},
        'replication_gate':{'mean_auc':.60,'pooled_pair_calibrated_auc':.60,'delta_control':.01,'delta_ci97_5_lower_positive':True,'folds_improved':3,
            'interpretation':'Solo motiva réplica interna prefijada; no éxito clínico ni test automático'},
        'next_step_if_gate':'Una réplica registrada con nuevas semillas y roles internos prefijados, control reentrenado y mismas recetas. No ajustar mirando test',
        'next_step_if_no_gate':'Cerrar familia y documentar; no buscar variantes contra este mismo exterior ni activar un candidato',
        'limits':['Desarrollo adaptativo con antecedentes; no nueva validación independiente',
                  'La nueva arquitectura cambia capacidad, resolución y reducciones; no atribuir causalmente todo a un factor',
                  'Atención frente a uniforme comparte arquitectura/inicialización y aísla este componente dentro de la nueva receta',
                  'La atención no es una segmentación ni una explicación clínica validada',
                  'Selección comparte checkpoint/BCE/calibración/umbral; políticas de parejas, no despliegue ensemble10',
                  'No nuevos datos, preentrenados, clínica, test, privada, push ni ramas; versiones anteriores inmutables'],
        'other_source_hashes':{str(p.relative_to(PROJECT_ROOT)):base.sha256(p) for p in
            [PROJECT_ROOT/'tests/test_preservacion_espacial.py',PROJECT_ROOT/'docs/mejoras_cnn/PROTOCOLO_PRESERVACION_ESPACIAL.md']}}
    write(output/'protocolo_familia.json',family)
    (output/'protocolo_familia.json.sha256').write_text(base.sha256(output/'protocolo_familia.json')+'\n')
    write(output/'estado.json',{'status':'prepared','runs_completed':0,'planned_runs':40,'updated_at':now()})
    return family


def verify_family(output):
    output=Path(output)
    if base.sha256(output/'protocolo_familia.json')!=(output/'protocolo_familia.json.sha256').read_text().strip():
        raise ValueError('Protocolo familiar modificado')
    family=json.loads((output/'protocolo_familia.json').read_text())
    for name,digest in family['other_source_hashes'].items():
        if base.sha256(PROJECT_ROOT/name)!=digest:raise ValueError('Fuente auxiliar modificada')
    for profile,entry in family['profiles'].items():
        if base.sha256(output/profile/'protocolo.json')!=entry['protocol_sha256']:raise ValueError('Receta modificada')
        verify(output/profile)
    return family


def pair_comparison(left,right,n=4000):
    if not left.patient_id.equals(right.patient_id) or not left.label.equals(right.label) or not left.fold.equals(right.fold):
        raise ValueError('Recetas no comparables')
    rng=np.random.default_rng(20261026)
    groups=[g.index.to_numpy() for _,g in left.groupby(['fold','label'])]
    values=[]
    for _ in range(n):
        indices=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        a,b=left.iloc[indices],right.iloc[indices]
        delta=[]
        for k in range(5):
            ga,gb=a[a.fold.eq(k)],b[b.fold.eq(k)]
            delta.append(fast_auc(ga.label.to_numpy(),ga.probability_raw.to_numpy())-fast_auc(gb.label.to_numpy(),gb.probability_raw.to_numpy()))
        values.append(np.mean(delta))
    point=np.mean([roc_auc_score(g.label,g.probability_raw) for _,g in left.groupby('fold')])-np.mean([roc_auc_score(g.label,g.probability_raw) for _,g in right.groupby('fold')])
    return {'delta_mean':float(point),'ci95':np.quantile(values,[.025,.975]).tolist(),'secondary':True,'replicates':n}


def report_family(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);family=verify_family(output)
    comparison={};frames={};curve_records=[]
    for profile in PROFILES:
        child=output/profile;results=json.loads((child/'comparacion.json').read_text())
        main=results['loss'];frames[profile]=pd.read_csv(child/'oof_loss.csv').sort_values('patient_id').reset_index(drop=True)
        improved=sum(a>b for a,b in zip(main['fold_auc'],main['control_fold_auc']))
        lo=main['delta_control_ci97_5_bonferroni2'][0]
        eligible=bool(main['mean_auc']>=.60 and main['calibrated_probability_metrics_at_0_5']['roc_auc']>=.60 and main['delta_mean_control']>=.01 and lo>0 and improved>=3)
        choices=json.loads((child/'elecciones_congeladas.json').read_text())['folds']
        selected=[]
        fig,axes=plt.subplots(5,2,figsize=(13,19))
        fig_auc,axes_auc=plt.subplots(5,2,figsize=(13,19))
        for fold in range(5):
            loss=choices[fold]['rules']['loss']['selected_loss']
            for col,seed in enumerate(SEEDS):
                path=child/'runs'/f'fold_{fold}'/loss/f'seed_{seed}'
                summary=json.loads((path/'summary.json').read_text());chosen=summary['selected']['loss'];h=pd.read_csv(path/'history.csv')
                selected.append({**chosen,'last_gap':float(h.iloc[-1].auc_gap),'joint':summary['curve_diagnostic']['joint_improvement']})
                curve_records.append({'profile':profile,'fold':fold,'loss':loss,'seed':seed,**selected[-1]})
                ax=axes[fold,col];ax.plot(h.epoch,h.fit_bce,label='Fit, eval');ax.plot(h.epoch,h.selection_bce,label='Selección, eval')
                ax.axvspan(12,14,alpha=.1,color='grey');ax.axvline(chosen['epoch'],ls=':',color='black')
                ax.set(title=f'{profile} · fold{fold} · {loss} · semilla{seed} · época{chosen["epoch"]}',xlabel='Época',ylabel='BCE por paciente');ax.legend(fontsize=8)
                ax=axes_auc[fold,col];ax.plot(h.epoch,h.fit_auc,label='Fit');ax.plot(h.epoch,h.selection_auc,label='Selección');ax.axvline(chosen['epoch'],ls=':',color='black')
                ax.set(title=f'{profile} · fold{fold} · {loss} · semilla{seed}',xlabel='Época',ylabel='AUC por paciente',ylim=(.3,1));ax.legend(fontsize=8)
        fig.suptitle('Misma BCE fit/selección, sin aumentos/dropout; puntos sin suavizar')
        fig.tight_layout(rect=(0,0,1,.98));fig.savefig(child/'01_perdidas.png',dpi=130);plt.close(fig)
        fig_auc.tight_layout();fig_auc.savefig(child/'02_auc.png',dpi=130);plt.close(fig_auc)
        comparison[profile]={'primary':main,'eligible_for_replication':eligible,'folds_improved_vs_control':improved,
            'selected_gap_median':float(np.median([v['auc_gap'] for v in selected])),
            'final_gap_median':float(np.median([v['last_gap'] for v in selected])),
            'selected_epoch_median':float(np.median([v['epoch'] for v in selected])),
            'joint_curves_selected':sum(bool(v['joint']) for v in selected),
            'selected_epochs':[v['epoch'] for v in selected]}
    comparison['attention_minus_uniform']=pair_comparison(frames['atencion'],frames['uniforme'])
    write(output/'comparacion_familia.json',comparison)
    pd.DataFrame(curve_records).to_csv(output/'diagnostico_seleccionados.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for profile,frame in frames.items():
        for fold,g in frame.groupby('fold'):
            x,y,_=roc_curve(g.label,g.probability_raw);axes[0].plot(x,y,alpha=.45,label=f'{profile} f{fold}: {roc_auc_score(g.label,g.probability_raw):.3f}')
        pr,rec,_=precision_recall_curve(frame.label,frame.probability_raw)
        axes[1].step(rec,pr,where='post',label=profile)
        axes[2].plot(range(5),comparison[profile]['primary']['fold_auc'],'o-',label=profile)
    ref=comparison['uniforme']['primary']
    axes[0].plot([0,1],[0,1],':',color='grey');axes[0].set(xlabel='1−Especificidad',ylabel='Sensibilidad',title='ROC por fold, cruda');axes[0].legend(fontsize=7)
    axes[1].axhline(frames['uniforme'].label.mean(),ls=':',color='grey');axes[1].set(xlabel='Recall',ylabel='Precisión',title='PR agrupada, secundaria');axes[1].legend()
    axes[2].plot(range(5),ref['control_fold_auc'],'o-',label='Control27');axes[2].plot(range(5),ref['cohort_fold_auc'],'o-',label='Cohorte')
    axes[2].set(xlabel='Fold',ylabel='AUC',title='Mismas pacientes');axes[2].legend()
    fig.tight_layout();fig.savefig(output/'03_comparacion.png',dpi=140);plt.close(fig)
    text='# Preservación espacial: dos CNN propias desde cero\n\n'
    text+='Solo PNG PRE/EARLY/LATE de train, un logit por corte y media de probabilidades por paciente. Las dos recetas se congelaron antes de inferir exterior. Desarrollo adaptativo, no nueva validación externa.\n\n'
    text+='[Zhang,2019](https://proceedings.mlr.press/v97/zhang19a.html) motiva filtrar antes de submuestrear. [Woo et al.,2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.php) motiva atención espacial. No se importan sus redes ni sus pesos, ni se implementa CBAM. Estos trabajos no demuestran beneficio en pCR.\n\n'
    text+='Entrada nativa256, seis canales temporales explícitos, cuatro bloques de dos convoluciones16/32/64/96, stride1, BN/ReLU. Tras cada bloque: máximo2×2 stride1, filtro binomial3×3 fijo y stride2. Mapas128→64→32→16. Dropout espacial0,1 y cabeza96→32→1 con dropout0,35. Uniforme congela en cero una puerta1×1; atención la aprende y combina mitad media uniforme con mitad softmax de temperatura2. No se aprende la agregación entre cortes. El mapa de atención no es una segmentación tumoral validada.\n\n'
    text+='Control cerrado27, mismos roles, semillas, LR0,0003, WD0,01, batch32, máximo40, mínimo observado14, paciencia8 y selección por pérdida. BCE normal/ponderada, dos semillas y cinco folds por receta:40 runs. La pérdida se elige por AUC de pareja en selección, nunca exterior; empate normal. Checkpoint por AUC se conserva como secundario. Los cambios conjuntos frente al control no permiten atribuir el efecto a un solo componente. Atención−uniforme es una ablación secundaria más específica.\n\n'
    text+='## Resultados de desarrollo\n\n| Receta | MediaAUC ±DE | AUC agrupada cruda | Δmedio frente control | IC97,5% deΔ | Réplica habilitada |\n|---|---:|---:|---:|---|---|\n'
    for profile in PROFILES:
        d=comparison[profile];v=d['primary']
        text+=f"| {profile} | {v['mean_auc']:.6f} ± {v['sd_auc']:.6f} | {v['pooled_raw_auc']:.6f} | {v['delta_mean_control']:+.6f} | {v['delta_control_ci97_5_bonferroni2']} | {d['eligible_for_replication']} |\n"
    text+='\nBootstrap4000 por pacientes, pareado, estratificado por fold/clase; CI97,5% Bonferroni para dos contrastes. Condiciona a estos modelos; no incluye toda variabilidad de entrenamiento ni búsqueda histórica. Los folds comparten datos de fit. No es prueba externa.\n\n'
    text+='Gate prefijado: media≥0,60, AUC agrupada con calibración por pareja≥0,60, delta≥0,01, límite inferiorIC97,5%>0 y mejora en3/5 folds. Pasarlo solo habilita réplica interna prefijada, no éxito final ni test automático.\n\n'
    text+=f"Atención−uniforme, secundario: {comparison['attention_minus_uniform']}.\n\n"
    text+='![Comparación](03_comparacion.png)\n\n## Curvas y sobreajuste\n\n'
    for profile in PROFILES:
        d=comparison[profile]
        text+=f"### {profile}\n\nÉpoca elegida mediana{d['selected_epoch_median']}; brecha AUC fit−selección mediana{d['selected_gap_median']:.4f}, final{d['final_gap_median']:.4f}. Tendencia conjunta1–14 en{d['joint_curves_selected']}/10 elegidos. No se fuerza un checkpoint después de14 ni se suavizan datos. Ambas BCE por paciente en eval, misma ponderación; optimización por corte separada.\n\n![Pérdidas {profile}]({profile}/01_perdidas.png)\n![AUC {profile}]({profile}/02_auc.png)\n\n"
    text+='## Políticas, evidencias y límites\n\nPlattC1 no negativo y umbrales de especificidad95% y macroF1 se ajustan en selección de la misma pareja. Las políticas no garantizan especificidad externa, no se eligen con exterior y no validan un ensemble10. Positivo significa pCR, respuesta completa. FP/FN, AP, calibración, cohortes y reglas secundarias están en comparacion.json y las tablas OOF de cada receta.\n\n'
    text+='1.097 pacientes,322 pCR. La CV mezcla cohortes; no evalúa una cohorte nunca vista. Las métricas internas de selección son optimistas porque comparten elección de checkpoint, BCE, calibración y umbral. Las comparaciones con V007 histórica media0,576593 y test histórico0,608376 son descriptivas; no se reabre ese test. No se abre privada, no se activan modelos ni se publica. Versiones anteriores verificadas por hash.\n\n'
    text+='Si no se supera el gate se cierra esta familia sin ampliar la búsqueda contra el mismo exterior. Configuraciones, roles, fuentes/hashes, trayectorias y selecciones se conservan; pesos y checkpoints de trabajo solo locales en runs/.\n'
    (output/'INFORME_RESULTADOS.md').write_text(text)
    return comparison


def execute(output=OUTPUT):
    output=Path(output);family=verify_family(output)
    if (output/'comparacion_familia.json').exists(): raise ValueError('Familia terminada: no reevaluar')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA requerida')
    device=torch.device('cuda');torch.set_num_threads(4)
    samples,_=audit.train_metadata(DATA_DIR)
    state={'status':'training','pid':os.getpid(),'started_at':now(),'runs_completed':0,'planned_runs':40}
    write(output/'estado.json',state);write(output/'entorno.json',base.environment(device))
    all_choices={}
    # Reutiliza caché de píxeles para ambas recetas; jamás evalúa exterior aquí.
    for fold in range(5):
        parts=roles_for(samples,pd.read_csv(output/'uniforme'/'roles'/f'roles_fold_{fold}.csv'))
        fit_x,sel_x=(cache.load_pixels(parts[r],device) for r in ('fit','selection'))
        for profile in PROFILES:
            child=output/profile;protocol=verify(child)
            if base.sha256(child/'roles'/f'roles_fold_{fold}.csv')!=base.sha256(output/'uniforme'/'roles'/f'roles_fold_{fold}.csv'):
                raise ValueError('Roles diferentes entre recetas')
            for loss in LOSSES:
                for seed in SEEDS:
                    state.update(profile=profile,fold=fold,loss=loss,seed=seed,updated_at=now());write(output/'estado.json',state)
                    path=child/'runs'/f'fold_{fold}'/loss/f'seed_{seed}'
                    train_run(parts['fit'],parts['selection'],fit_x,sel_x,protocol['config'],loss,seed,path,device)
                    state['runs_completed']+=1;state['updated_at']=now();write(output/'estado.json',state)
            choice=freeze_fold(child,fold,parts)
            all_choices.setdefault(profile,[]).append(choice)
            write(child/f'eleccion_fold_{fold}.json',choice)
        del fit_x,sel_x
    for profile,choices in all_choices.items():
        write(output/profile/'elecciones_congeladas.json',{'frozen_at':now(),'folds':choices})
    write(output/'familia_congelada.json',{'frozen_at':now(),'profiles':{p:base.sha256(output/p/'elecciones_congeladas.json') for p in PROFILES}})
    verify_family(output);state.update(status='evaluating_frozen',updated_at=now());write(output/'estado.json',state)
    for profile in PROFILES:
        evaluate(output/profile,verify(output/profile),samples,all_choices[profile],device)
    state.update(status='reporting',updated_at=now());write(output/'estado.json',state)
    result=report_family(output);verify_family(output)
    write(output/'verificacion_ejecucion.json',{'checked_at':now(),'runs':40,'source_and_old_versions_unchanged':True,'all_choices_frozen_before_outer':True,
        'test_images_loaded':0,'private_images_loaded':0,'no_push':True,'no_activation':True,
        'artifacts':{str(f.relative_to(output)):base.sha256(f) for f in output.rglob('*') if f.is_file() and 'runs' not in f.relative_to(output).parts and f.suffix!='.log' and f.name not in ('estado.json','verificacion_ejecucion.json')}})
    state.update(status='complete',updated_at=now(),eligible=[p for p in PROFILES if result[p]['eligible_for_replication']]);write(output/'estado.json',state)
    print(json.dumps(state,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['comprobar','preparar','entrenar','informe'])
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    try:
        if args.action=='comprobar':print(json.dumps(synthetic_learning_check(),ensure_ascii=False),flush=True)
        elif args.action=='preparar':print(json.dumps(prepare(args.output),ensure_ascii=False),flush=True)
        elif args.action=='entrenar':execute(args.output)
        else:report_family(args.output)
    except Exception:
        if args.action=='entrenar':write(args.output/'error.json',{'at':now(),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
