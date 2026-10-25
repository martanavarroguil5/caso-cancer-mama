#!/usr/bin/env python3
"""Evaluación final de tres paquetes congelados; nunca entrena ni ajusta políticas."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'resultados/23_evaluacion_test_imagen_20261025'
PACKAGES = {
    'v001': ROOT / 'modelos/versiones/v001_cnn_historica_20260930',
    'hibrida': ROOT / 'modelos/imagen_sola_20261025',
    'v007': ROOT / 'modelos/versiones/v007_cnn_cinetica_multiescala_20261021',
}
HISTORICAL = PACKAGES['v001'] / 'evidencia/repo/resultados/04_entrenamiento/historico/test_pacientes.csv'
SAMPLES = ROOT / 'breastdcedl/metadata/samples.csv'
PATIENTS = ROOT / 'breastdcedl/metadata/patients.csv'
PHASES = ('pre', 'early', 'late')
SEED, REPLICATES = 20261025, 5000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def now():
    return datetime.now(ZoneInfo('Europe/Madrid')).isoformat()


def inventory():
    paths = [Path(__file__).resolve(), SAMPLES, PATIENTS, HISTORICAL]
    paths += list((ROOT/'src/cancer_mama').glob('*.py'))
    for package in PACKAGES.values():
        paths += [p for p in package.rglob('*') if p.is_file() and
                  '__pycache__' not in p.parts]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def prepare(output):
    test_samples()  # Preflight del índice; no calcula predicciones ni métricas nuevas.
    if output.exists():
        raise ValueError('La salida ya existe: no se sustituye una evaluación anterior')
    output.mkdir(parents=True)
    protocol = {
        'created_at': now(),
        'git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'models': list(PACKAGES), 'expected_patients': 176, 'expected_positives': 53,
        'unit': 'patient',
        'population': 'Todas las pacientes del test docente; mismos cortes en los tres paquetes',
        'primary': 'ROC-AUC de la probabilidad final del paquete congelado',
        'secondary': ['AUC cruda','AP','logloss','Brier','matriz a 0.5 crudo',
                      'matriz a 0.5 final','política operativa histórica si existe'],
        'bootstrap': {'replicates':REPLICATES,'seed':SEED,'strata':['cohort','label'],
                      'paired':True,'interval':'percentile 95%; diferencias exploratorias'},
        'v001': 'Reutilizar CSV histórico, sin repetir inferencia ni elegir otros pesos',
        'v007': 'Media de probabilidades de diez modelos y todos los cortes; Platt y umbral archivados',
        'hibrida': 'Media de cinco grupos calibrados internamente, tres semillas por grupo; todos los cortes',
        'fairness': 'Misma población, PNG, unidad estadística e intervalos. Se conserva el pipeline de cada paquete; no iguala presupuesto ni historial de selección.',
        'test_history': 'Test consultado en el proyecto desde 30/09/2026; comparación retrospectiva, no validación ciega nueva',
        'teacher_validation': 'Cerrada: no cargar imágenes ni etiquetas',
        'constraints': ['sin entrenamiento','sin recalibración','sin ajuste de umbrales',
                        'sin selección por test','sin activar modelos','sin push'],
        'files_sha256': inventory(),
    }
    write_json(output/'protocolo_congelado.json',protocol)
    (output/'protocolo_congelado.json.sha256').write_text(sha(output/'protocolo_congelado.json')+'\n')
    print(json.dumps({'protocol_locked':True,'files':len(protocol['files_sha256'])}),flush=True)


def verify_protocol(output):
    path = output/'protocolo_congelado.json'
    if sha(path) != path.with_suffix('.json.sha256').read_text().strip():
        raise ValueError('Protocolo modificado')
    protocol = json.loads(path.read_text())
    for name, digest in protocol['files_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Fuente o peso modificado: {name}')
    return protocol


def test_samples():
    frame = pd.read_csv(SAMPLES)
    cohorts = pd.read_csv(PATIENTS,usecols=['pid','dataset']).rename(columns={'pid':'patient_id'})
    frame = frame.merge(cohorts,on='patient_id',how='left',validate='many_to_one')
    selected = frame.loc[frame['split'].eq('test')].copy()
    train_ids = set(frame.loc[frame['split'].eq('train'),'patient_id'])
    if set(selected.patient_id) & train_ids:
        raise ValueError('Hay pacientes compartidas entre train y test')
    if selected.patient_id.nunique()!=176 or selected.sample_id.duplicated().any():
        raise ValueError('Cohorte de test incompatible')
    historic = pd.read_csv(HISTORICAL).sort_values('patient_id')
    if historic.patient_id.duplicated().any() or set(historic.patient_id)!=set(selected.patient_id):
        raise ValueError('Las pacientes no coinciden con el test histórico')
    if historic.label.sum()!=53 or not set(historic.label).issubset({0,1}):
        raise ValueError('Etiquetas inesperadas')
    for row in historic.itertuples():
        patient = selected.loc[selected.patient_id.eq(row.patient_id)]
        if len(patient)!=row.n_slices or not patient.pCR.eq(row.label).all():
            raise ValueError('Cortes o etiquetas distintas de la evaluación histórica')
        if not patient.dataset.eq(row.cohort).all():
            raise ValueError('La cohorte no coincide')
    if not 1<=selected.groupby('patient_id').size().min()<=selected.groupby('patient_id').size().max()<=10:
        raise ValueError('Número de cortes incompatible')
    return selected.sort_values(['patient_id','slice_index']), historic


def load_patient(frame, paths=False):
    full, small, triples = [], [], []
    for row in frame.itertuples():
        a, b, triple = [], [], []
        for phase in PHASES:
            path = ROOT/'breastdcedl'/getattr(row,'path_'+phase)
            if path.suffix.lower()!='.png' or not path.resolve().is_relative_to((ROOT/'breastdcedl/dataset/test').resolve()):
                raise ValueError('Solo se permiten PNG de test')
            with Image.open(path) as im:
                if im.size!=(256,256) or im.mode!='L' or im.format!='PNG':
                    raise ValueError('PNG inválido')
                a.append(np.asarray(im).copy())
                b.append(np.asarray(im.resize((128,128),Image.Resampling.BOX)).copy())
            triple.append(path)
        full.append(np.stack(a));small.append(np.stack(b));triples.append(triple)
    result = (np.stack(full),np.stack(small))
    return (*result,triples) if paths else result


def audit_inputs(output):
    samples,_=test_samples()
    digests={}
    for row in samples.itertuples():
        for phase in PHASES:
            p=ROOT/'breastdcedl'/getattr(row,'path_'+phase)
            if not p.resolve().is_relative_to((ROOT/'breastdcedl/dataset/test').resolve()):
                raise ValueError('Ruta ajena al test')
            digests[str(p.relative_to(ROOT))]=sha(p)
    path=output/'entradas_sha256.json'
    if path.exists():
        if digests!=json.loads(path.read_text()):raise ValueError('Imágenes modificadas durante la evaluación')
    else:write_json(path,digests)
    return len(digests)


def load_models(kind, device):
    import torch
    package = PACKAGES[kind]
    sys.path.insert(0,str(package/'codigo/src'))
    models = []
    if kind=='v007':
        from cancer_mama import inferencia_imagen as runtime
        manifest = runtime.cargar_manifest(package/'manifiesto.json')
        for entry in manifest['models']:
            models.append(runtime.cargar_modelo(package/'manifiesto.json',entry).to(device))
    else:
        from cancer_mama import inferencia_mil as runtime
        manifest = json.loads((package/'manifest.json').read_text())
        for entry in manifest['models']:
            path = package/entry['file']
            if sha(path)!=entry['sha256']: raise ValueError('Peso modificado')
            ckpt = torch.load(path,map_location='cpu',weights_only=True)
            model = runtime.PatientNet(ckpt['config'],ckpt['n_features'])
            model.load_state_dict(ckpt['state_dict'],strict=True)
            models.append(model.eval().to(device))
    for model in models:
        model.requires_grad_(False)
        if model.training: raise ValueError('Modelo fuera de eval')
    return models,manifest,runtime


def predict(kind, models, manifest, runtime, full, small, device):
    import torch
    with torch.inference_mode():
        x = torch.tensor(full if kind=='v007' else small,device=device,dtype=torch.float32)/255
        if kind=='v007':
            probabilities = [float(torch.cat([model(c).sigmoid().flatten() for c in x.split(8)]).mean()) for model in models]
            raw = float(np.mean(probabilities))
            final = float(runtime.base.apply_calibration(np.array([raw]),manifest['calibration'])[0])
        else:
            features = {}
            mask = torch.ones((1,len(x)),device=device,dtype=torch.bool)
            probabilities = []
            for model in models:
                key = (model.config.get('feature_file'),model.config.get('roi_features'))
                if key not in features:
                    features[key] = runtime.features_from_images(x,model.config)[None]
                probabilities.append(float(model(x[None],mask,features[key]).sigmoid()[0]))
            group_probs = []
            for group in manifest['groups']:
                p = float(np.mean([probabilities[i] for i in group['model_indices']]))
                c=group['calibration'];p=np.clip(p,1e-6,1-1e-6)
                z=c['intercept']+c['coef']*(np.log(p/(1-p))-c['mean'])/c['std']
                group_probs.append(float(1/(1+np.exp(-np.clip(z,-30,30)))))
            raw,final=float(np.mean(probabilities)),float(np.mean(group_probs))
    if not np.isfinite([*probabilities,raw,final]).all(): raise ValueError('Probabilidad no finita')
    return {'probability_raw':raw,'probability_final':final}


def worker(kind,output,device,smoke=False):
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    models,manifest,runtime=load_models(kind,device)
    if smoke:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='cnn_equivalencia_') as tmp:
            tmp=Path(tmp);rng=np.random.default_rng(SEED)
            pre=rng.integers(0,140,(256,256),dtype=np.uint8)
            arrays=[pre,pre+35,pre+20];triple=[]
            for phase,arr in zip(PHASES,arrays):
                path=tmp/f'SYNTHETIC_z000_{phase.upper()}.png'
                Image.fromarray(arr).save(path);triple.append(path)
            full=np.stack(arrays)[None]
            small=np.stack([np.asarray(Image.fromarray(a).resize((128,128),Image.Resampling.BOX)) for a in arrays])[None]
            fast=predict(kind,models,manifest,runtime,full,small,device)
            if kind=='v007':
                reference=runtime.predecir_cortes(PACKAGES[kind]/'manifiesto.json',[triple])['probability_calibrated']
            else:
                reference=runtime.predict_package(PACKAGES[kind],[triple[0]],[triple[1]],[triple[2]])['probability_pcr']
            if abs(reference-fast['probability_final'])>1e-6:raise ValueError('Inferencia no equivalente a la CLI congelada')
            print(json.dumps({'kind':kind,'equivalent_to_archived_cli':True,'error':abs(reference-fast['probability_final'])}))
        return
    verify_protocol(output)
    destination=output/f'predicciones_{kind}.csv'
    if destination.exists():raise ValueError('Ya existen predicciones: no repetir inferencia')
    samples,_=test_samples();rows=[];start=time.monotonic()
    for i,(pid,frame) in enumerate(samples.groupby('patient_id',sort=True),1):
        full,small=load_patient(frame)
        pred=predict(kind,models,manifest,runtime,full,small,device)
        rows.append({'patient_id':pid,'n_slices':len(frame),**pred})
        if i%20==0 or i==176:
            print(json.dumps({'model':kind,'patients':i,'total':176,'seconds':round(time.monotonic()-start,1)}),flush=True)
    pd.DataFrame(rows).to_csv(destination,index=False)
    write_json(output/f'inferencia_{kind}.json',{
        'completed_at':now(),'seconds':time.monotonic()-start,'device':device,
        'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
        'max_memory_allocated':torch.cuda.max_memory_allocated() if device=='cuda' else 0,
        'source':str(PACKAGES[kind].relative_to(ROOT)),'patients':len(rows),
        'grad_enabled':False,'training':False,'amp':False,'validation_private_loaded':False,
        'predictions_sha256':sha(destination)})


def decisions(y,p,threshold):
    tn,fp,fn,tp=confusion_matrix(y,np.asarray(p)>=threshold,labels=[0,1]).ravel()
    div=lambda n,d:float(n/d) if d else 0.0
    return {'threshold':float(threshold),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
            'sensitivity':div(tp,tp+fn),'specificity':div(tn,tn+fp),'precision':div(tp,tp+fp),
            'f1':div(2*tp,2*tp+fp+fn),'accuracy':div(tp+tn,len(y))}


def metrics(y,p):
    return {'auc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p)),'logloss':float(log_loss(y,p,labels=[0,1])),
            'at_0_5':decisions(y,p,.5)}


def bootstrap(y, predictions, cohorts, repetitions=REPLICATES, seed=SEED):
    rng=np.random.default_rng(seed)
    strata=[np.flatnonzero((np.asarray(cohorts)==c)&(y==label)) for c in np.unique(cohorts) for label in (0,1)]
    strata=[s for s in strata if len(s)]
    names=list(predictions);scores=np.zeros((repetitions,len(names)))
    for b in range(repetitions):
        idx=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
        for j,name in enumerate(names):scores[b,j]=roc_auc_score(y[idx],predictions[name][idx])
    interval=lambda a:[float(v) for v in np.quantile(a,[.025,.975])]
    return {'auc_ci95':{n:interval(scores[:,j]) for j,n in enumerate(names)},
            'paired_differences':[{ 'a':a,'b':b,'delta_auc':float(roc_auc_score(y,predictions[a])-roc_auc_score(y,predictions[b])),
                'ci95':interval(scores[:,i]-scores[:,j])}
                for i,a in enumerate(names) for j,b in enumerate(names) if i<j],
            'replicates':repetitions,'seed':seed,'strata':['cohort','label'],
            'interpretation':'Condicional a pesos fijos y esta mezcla de cohortes; comparación exploratoria, sin corregir toda la selección histórica'}


def analyze(output):
    verify_protocol(output)
    audit_inputs(output)
    _,historical=test_samples()
    table=historical[['patient_id','label','cohort','n_slices']].copy()
    table['v001_raw']=historical.probability
    table['v001_final']=historical.probability_calibrated
    for kind in ('hibrida','v007'):
        predictions=pd.read_csv(output/f'predicciones_{kind}.csv')
        if predictions.patient_id.duplicated().any() or set(predictions.patient_id)!=set(table.patient_id):raise ValueError('Comparación no pareada')
        joined=table.merge(predictions,on='patient_id',validate='one_to_one',suffixes=('','_new'))
        if not joined.n_slices.eq(joined.n_slices_new).all():raise ValueError('Distintos cortes')
        table=joined.drop(columns=['n_slices_new']).rename(columns={'probability_raw':kind+'_raw','probability_final':kind+'_final'})
    table=table.sort_values('patient_id').reset_index(drop=True)
    y=table.label.to_numpy(dtype=int)
    probabilities={kind:table[kind+'_final'].to_numpy() for kind in PACKAGES}
    results={k:{'final':metrics(y,probabilities[k]),'raw':metrics(y,table[k+'_raw'])} for k in PACKAGES}
    assert abs(results['v001']['raw']['auc']-.5462494247583985)<1e-12
    for kind in ('v001','v007'):
        manifest=json.loads((PACKAGES[kind]/'manifiesto.json').read_text())
        results[kind]['frozen_operating_policy']=decisions(y,probabilities[kind],manifest['threshold'])
    intervals=bootstrap(y,probabilities,table.cohort.to_numpy())
    cohorts=[]
    for name,frame in table.groupby('cohort'):
        cohorts.append({'cohort':name,'n':len(frame),'positive':int(frame.label.sum()),
                        'auc':{kind:float(roc_auc_score(frame.label,frame[kind+'_final'])) for kind in PACKAGES}})
    table.to_csv(output/'comparacion_pacientes.csv',index=False)
    result={'completed_at':now(),'n_patients':len(table),'positives':int(y.sum()),
            'n_slices':int(table.n_slices.sum()),'unit':'patient','models':results,
            'intervals':intervals,'cohorts':cohorts,
            'limitations':['Test histórico conocido; no validación externa ciega nueva',
                          'No iguala entrenamiento ni selección histórica entre arquitecturas',
                          'Comparación por paciente con todos sus cortes; no estima rendimiento de un único corte en la app',
                          'No modificar ni escoger una nueva receta usando estos resultados']}
    write_json(output/'metricas.json',result)
    render(output,table,result)
    verify_protocol(output)
    write_json(output/'verificacion_final.json',{'completed_at':now(),'protocol_and_frozen_packages_unchanged':True,
        'same_176_patients':True,'same_slices':True,'historical_v001_reused':True,'private_validation_opened':False,
        'training_executed':False,'policies_fitted':False,
        'artifacts_sha256':{p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='ejecucion.log'}})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def render(output,table,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve,precision_recall_curve
    from sklearn.calibration import calibration_curve
    names={'v001':'V001 histórica','hibrida':'CNN híbrida','v007':'V007 cinética'}
    plt.rcParams.update({'font.size':11,'figure.dpi':150})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for kind in PACKAGES:
        y=table.label.to_numpy();p=table[kind+'_final'].to_numpy()
        fpr,tpr,_=roc_curve(y,p);precision,recall,_=precision_recall_curve(y,p)
        axes[0].plot(fpr,tpr,label=f"{names[kind]}: {result['models'][kind]['final']['auc']:.4f}")
        axes[1].plot(recall,precision,label=f"{names[kind]}: {result['models'][kind]['final']['ap']:.4f}")
    axes[0].plot([0,1],[0,1],'k--',alpha=.5)
    axes[1].axhline(table.label.mean(),color='k',linestyle='--',alpha=.5,label='Prevalencia')
    axes[0].set(xlabel='Tasa de falsos positivos',ylabel='Sensibilidad',title='ROC · test histórico · AUC')
    axes[1].set(xlabel='Sensibilidad',ylabel='Precisión',title='Precision–recall · AP')
    for ax in axes:ax.legend(fontsize=9);ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'01_roc_pr.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.8))
    for ax,kind in zip(axes,PACKAGES):
        d=result['models'][kind]['final']['at_0_5'];matrix=np.array([[d['tn'],d['fp']],[d['fn'],d['tp']]])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=123)
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>65 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['no pCR','pCR'],yticklabels=['no pCR','pCR'],xlabel='Predicción',ylabel='Real',title=names[kind])
    fig.suptitle('Umbral común 0,50 sobre probabilidad final; sin reajuste')
    fig.tight_layout();fig.savefig(output/'02_matrices_umbral_050.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.6,5))
    for kind in PACKAGES:
        observed,predicted=calibration_curve(table.label,table[kind+'_final'],n_bins=5,strategy='quantile')
        ax.plot(predicted,observed,'o-',label=names[kind])
    ax.plot([0,1],[0,1],'k--',alpha=.5)
    ax.set(xlabel='Probabilidad media predicha',ylabel='Fracción observada de pCR',title='Calibración descriptiva · 5 grupos de igual tamaño',xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=9);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'03_calibracion.png');plt.close(fig)
    lines=['# Evaluación final pareada de modelos de solo imagen','',
           '25/10/2026. Comparación de paquetes congelados, sin entrenamiento ni ajuste con test.',
           '',f"**{result['n_patients']} pacientes, {result['positives']} pCR y {result['n_slices']} cortes.** Una predicción por paciente; mismos PNG en los tres paquetes.",
           '', '## Resultado sobre test', '',
           '| Modelo | AUC final | IC95 % | AUC cruda | AP final | Logloss final | Brier final |',
           '|---|---:|---|---:|---:|---:|---:|']
    for kind in PACKAGES:
        m=result['models'][kind];lo,hi=result['intervals']['auc_ci95'][kind];p=m['final']
        lines.append(f"| {names[kind]} | **{p['auc']:.4f}** | [{lo:.4f}, {hi:.4f}] | {m['raw']['auc']:.4f} | {p['ap']:.4f} | {p['logloss']:.4f} | {p['brier']:.4f} |")
    lines += ['', 'La métrica principal es la AUC de la probabilidad final de cada paquete. No se escoge entre AUC cruda, calibrada o media por fold después de observar test. V001 y V007 aplican una calibración global monótona; en la híbrida se calibran los grupos antes de agregarlos, tal como estaba archivado.',
              '', '![Curvas ROC y PR](01_roc_pr.png)', '', '## Diferencias pareadas de AUC', '',
              '| Comparación A − B | Diferencia | IC95 % |','|---|---:|---|']
    for d in result['intervals']['paired_differences']:
        lines.append(f"| {names[d['a']]} − {names[d['b']]} | {d['delta_auc']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    lines += ['',f'Intervalos mediante {REPLICATES:,} remuestreos por paciente, pareados y estratificados por cohorte y clase, semilla {SEED}. Comparaciones exploratorias; no corrigen las decisiones históricas ni garantizan el rendimiento de otro entrenamiento.',
              '', '## Errores con el mismo umbral', '',
              'Umbral 0,50 sobre la probabilidad final. No se fuerza la misma sensibilidad o especificidad ajustando el test.', '',
              '| Modelo | TN | FP | FN | TP | Sensibilidad | Especificidad | Precisión | F1 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for kind in PACKAGES:
        d=result['models'][kind]['final']['at_0_5']
        lines.append(f"| {names[kind]} | {d['tn']} | {d['fp']} | {d['fn']} | {d['tp']} | {d['sensitivity']:.3f} | {d['specificity']:.3f} | {d['precision']:.3f} | {d['f1']:.3f} |")
    lines += ['', '![Matrices](02_matrices_umbral_050.png)', '',
              'Las matrices a 0,50 crudo y las políticas operativas ya archivadas de V001/V007 se conservan en `metricas.json`. Esas políticas tienen umbrales diferentes y no son una comparación al mismo punto operativo. La híbrida no tenía otro umbral clínico congelado; no se inventa uno usando test.',
              '', '![Calibración descriptiva](03_calibracion.png)', '', '## Alcance de la evaluación', '',
              '- Ninguna de estas pacientes aparece en el train según el índice original. Se verifican las mismas etiquetas y cortes de la evaluación histórica de V001.',
              '- V001 reutiliza exactamente sus predicciones archivadas; no se repite su inferencia. Las otras dos usan el código y pesos congelados, sin aumentado y en modo evaluación.',
              '- Equidad en población, unidad estadística y cálculo de métricas. No se igualan el presupuesto, las semillas, la capacidad ni la selección histórica; no demuestra causalmente que una arquitectura sea superior.',
              '- Este test se conoce en el proyecto desde septiembre. Es una evaluación retrospectiva sobre pacientes fuera del entrenamiento, no una validación externa o completamente ciega nueva.',
              '- La validación privada del profesorado permanece cerrada. No se puede prometer una AUC poblacional exacta ni deducir que supera 0,60 solo por una estimación puntual.',
              '- La evaluación usa todos los cortes por paciente. No mide el rendimiento de la aplicación cuando solo se carga una tripleta de un corte.',
              '- No se entrena, recalibra, ajusta el umbral, activa un modelo ni inicia otra búsqueda tras estos resultados. La comparación BCE normal/ponderada de la receta híbrida sigue pendiente para completar la entrega.',
              '', '## Reproducción y evidencia', '',
              '- `protocolo_congelado.json` y su SHA-256: fuentes, pesos, políticas y análisis fijados antes de calcular nuevas predicciones.',
              '- `comparacion_pacientes.csv`: predicciones alineadas por paciente; `metricas.json`: métricas e intervalos.',
              '- `predicciones_hibrida.csv`, `predicciones_v007.csv` e `inferencia_*.json`: inferencia y procedencia.',
              '- `verificacion_final.json`: integridad de paquetes y artefactos; sin imágenes, datasets ni checkpoints añadidos.',
              '', '```bash',
              '.venv/bin/python -B scripts/evaluar_test_imagen.py preparar',
              '.venv/bin/python -B scripts/evaluar_test_imagen.py ejecutar --device cuda',
              '```',
              '', 'La ejecución se bloquea para no repetir inferencia; `analizar` reutiliza las predicciones guardadas. Para reproducir en otra copia se debe usar una salida nueva sin sobrescribir este ensayo.']
    (output/'INFORME_RESULTADOS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preparar','ejecutar','analizar','worker','verificar'])
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--kind',choices=['hibrida','v007'])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args();output=args.output.resolve()
    if args.action=='preparar':prepare(output)
    elif args.action=='worker':worker(args.kind,output,args.device)
    elif args.action=='verificar':worker(args.kind,output,'cpu',smoke=True)
    elif args.action=='analizar':analyze(output)
    else:
        verify_protocol(output)
        with (output/'evaluacion.lock').open('x') as stream:stream.write(now()+'\n')
        count=audit_inputs(output)
        print(json.dumps({'test_pngs_verified':count,'training':False,'private_validation_opened':False}),flush=True)
        for kind in ['v007','hibrida']:
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'worker','--kind',kind,'--output',str(output),'--device',args.device]
            subprocess.run(command,check=True,env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8'})
        analyze(output)


if __name__=='__main__':main()
