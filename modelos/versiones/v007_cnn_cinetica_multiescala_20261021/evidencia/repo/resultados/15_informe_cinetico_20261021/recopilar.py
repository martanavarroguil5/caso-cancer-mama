"""Recopila el ensayo completo; no entrena ni lee imágenes de test/outer.

Se ejecuta únicamente después de terminar la evaluación exterior congelada.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
import torch

from cancer_mama import ajuste_cinetico as tune
from cancer_mama import entrenamiento as base
from cancer_mama import entrenamiento_imagen as image
from cancer_mama.paths import PROJECT_ROOT, RESULTS_DIR


def aligned(path, probability, reference):
    frame = pd.read_csv(path).sort_values('patient_id').reset_index(drop=True)
    for column in ('patient_id', 'label', 'fold', 'cohort'):
        if not np.array_equal(reference[column].to_numpy(), frame[column].to_numpy()):
            raise ValueError(f'Pacientes desalineadas en {column}: {path}')
    return frame[probability].to_numpy(dtype=float)


def intervals(frame, probabilities, n_boot):
    y = frame.label.to_numpy(dtype=int)
    strata = list(frame.groupby(['fold','cohort','label']).indices.values())
    rng = np.random.default_rng(20261021)
    samples = {name: np.empty(n_boot) for name in probabilities}
    for iteration in range(n_boot):
        ids = np.concatenate([rng.choice(index, len(index), replace=True) for index in strata])
        for name, probability in probabilities.items():
            samples[name][iteration] = roc_auc_score(y[ids], probability[ids])
    result = {'samples': n_boot, 'seed': 20261021,
              'method': 'paired patient bootstrap stratified by fold, cohort and label',
              'warning': 'Conditional on fitted models; does not repeat training or correct adaptive history',
              'auc': {}, 'delta_selected_minus': {}}
    for name, values in samples.items():
        result['auc'][name] = list(map(float, np.quantile(values,[.025,.975])))
        if name != 'selected':
            result['delta_selected_minus'][name] = {
                'estimate': float(roc_auc_score(y, probabilities['selected'])-
                                  roc_auc_score(y, probabilities[name])),
                'ci95': list(map(float, np.quantile(samples['selected']-values,[.025,.975])))}
    return result


def collect(run, destination, n_boot=2000):
    torch.set_num_threads(2)
    run, destination = Path(run), Path(destination)
    state = json.loads((run/'estado.json').read_text())
    if state['status'] != 'complete' or state['completed_runs'] != 60:
        raise ValueError('Se requieren 60 entrenamientos y la evaluación exterior terminada')
    frozen = json.loads((run/'seleccion_congelada.json').read_text())
    original_report = json.loads((run/'informe.json').read_text())
    protocol = json.loads((run/'protocolo.json').read_text())
    for name, checksum in protocol['source_sha256'].items():
        if base.sha256(PROJECT_ROOT/f'src/cancer_mama/{name}.py') != checksum:
            raise ValueError(f'Código de entrenamiento modificado: {name}')
    selected = pd.read_csv(run/'oof_pacientes_selected.csv').sort_values('patient_id').reset_index(drop=True)
    reference = pd.read_csv(run/'oof_pacientes_reference.csv').sort_values('patient_id').reset_index(drop=True)
    if len(selected) != 1097 or selected.patient_id.duplicated().any():
        raise ValueError('OOF de pacientes incompleto o duplicado')
    probabilities = {'selected': selected.probability_raw.to_numpy(),
                     'reference': aligned(run/'oof_pacientes_reference.csv', 'probability_raw', selected),
                     'phase_cnn_v006': aligned(RESULTS_DIR/'14_informe_cnn_imagen_20261014/oof_pacientes_weighted.csv',
                                               'probability_raw', selected),
                     'historical_image_v001': aligned(RESULTS_DIR/'04_entrenamiento/historico/oof_pacientes.csv',
                                                     'probability', selected),
                     'multimodal_v002': aligned(RESULTS_DIR/'06_informe_multimodal_20261007/comparacion_pareada.csv',
                                                'selected', selected)}
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('estado.json','lanzamiento.json','protocolo.json','seleccion_congelada.json','informe.json',
                 'oof_pacientes_selected.csv','oof_pacientes_reference.csv'):
        shutil.copy2(run/name,destination/name)
    executions, histories, selections, checks, diagnostics = [], [], [], [], []
    chosen_paths = {entry['path'] for fold in frozen['folds'] for entry in fold['selected']['models']}
    for path in sorted(run.rglob('summary.json')):
        if 'pruebas' in path.parts: continue
        folder = path.parent
        check = tune.check_run(folder)
        summary = json.loads(path.read_text())
        config = json.loads((folder/'config.json').read_text())
        environment = json.loads((folder/'environment.json').read_text())
        history = pd.read_csv(folder/'history.csv')
        patients = pd.read_csv(folder/'selection_patients.csv')
        relative = str(folder.relative_to(run))
        metadata = {'trial':config['name'].replace(tune.cnn.EXPERIMENT_NAME+'_',''),
                    'loss':config['loss'],'seed':config['seed'],'outer_fold':config['outer_fold'],
                    'selected':relative in chosen_paths}
        executions.append({'run':relative, **metadata,'config':config,'summary':summary,'environment':environment})
        histories.append(history.assign(**metadata)); selections.append(patients.assign(**metadata))
        checks.append({'run':relative, **check})
        best=history.iloc[summary['best_epoch']-1]; last=history.iloc[-1]
        diagnostics.append({**metadata,'epochs':len(history),'best_epoch':summary['best_epoch'],
                            'best_fit_auc':float(best.fit_patient_auc),
                            'best_inner_auc':float(best.selection_patient_auc),'best_gap':float(best.auc_gap),
                            'last_fit_auc':float(last.fit_patient_auc),
                            'last_inner_auc':float(last.selection_patient_auc),'last_gap':float(last.auc_gap),
                            'best_inner_loss':float(best.selection_loss),'last_inner_loss':float(last.selection_loss)})
    if len(executions)!=60 or len(chosen_paths)!=10:
        raise ValueError('Inventario de ejecuciones o pesos seleccionados incorrecto')
    pd.concat(histories,ignore_index=True).to_csv(destination/'historias.csv',index=False)
    pd.concat(selections,ignore_index=True).to_csv(destination/'predicciones_seleccion.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(destination/'diagnostico_ejecuciones.csv',index=False)
    base.json_write(destination/'ejecuciones.json',executions)
    base.json_write(destination/'verificacion_entrenamiento.json',checks)
    computed={'selected':tune.metrics(selected),'reference':tune.metrics(reference)}
    for arm,frame in (('selected',selected),('reference',reference)):
        computed[arm]['crossfit_calibrated_auc']=float(roc_auc_score(frame.label,frame.probability_calibrated))
        decision=computed[arm]['decision']
        decision['accuracy']=(decision['tn']+decision['tp'])/len(frame)
        decision['precision_pcr']=decision['tp']/(decision['tp']+decision['fp']) if decision['tp']+decision['fp'] else 0.
        decision['f1_pcr']=2*decision['tp']/(2*decision['tp']+decision['fp']+decision['fn'])
    for arm in computed:
        for key in ('pooled_raw_auc','mean_fold_auc','average_precision',
                    'crossfit_calibrated_log_loss','crossfit_calibrated_brier'):
            if not np.isclose(computed[arm][key],original_report['results'][arm][key],atol=1e-10,rtol=0):
                raise ValueError(f'Métrica no reproducida: {arm} {key}')
    bootstrap=intervals(selected,probabilities,n_boot)
    comparisons={name:{'auc':float(roc_auc_score(selected.label,probability)),
                       'average_precision':float(average_precision_score(selected.label,probability))}
                 for name,probability in probabilities.items()}
    calibration=base.fit_platt(selected.label.to_numpy(),probabilities['selected'])
    calibration['fit_data']='all_development_outer_oof_for_future_independent_test_only'
    calibrated=base.apply_calibration(probabilities['selected'],calibration)
    threshold=image.threshold_min_specificity(selected.label,calibrated,.90)
    evaluation={'generated_at':datetime.now(ZoneInfo('Europe/Madrid')).isoformat(),
                'patients':len(selected),'pcr':int(selected.label.sum()),
                'no_pcr':int((selected.label==0).sum()),'results':computed,
                'comparisons':comparisons,'bootstrap':bootstrap,
                'future_policy':{'calibration':calibration,'threshold':threshold,
                                 'fit_data':'development OOF; apparent metrics are not test metrics'},
                'warning':'Historical comparisons differ in architecture, inputs or selection; development only',
                'test_images_loaded':0}
    base.json_write(destination/'evaluacion.json',evaluation)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axis=plt.subplots(figsize=(8,7))
    labels={'selected':'Cinética ajustada','reference':'Cinética referencia',
            'phase_cnn_v006':'CNN por fases v006','historical_image_v001':'Imagen histórica v001',
            'multimodal_v002':'Multimodal v002 (incluye clínica)'}
    for name,probability in probabilities.items():
        fpr,tpr,_=roc_curve(selected.label,probability)
        axis.plot(fpr,tpr,lw=2 if name=='selected' else 1.3,
                  label=f'{labels[name]} · AUC {comparisons[name]["auc"]:.3f}')
    axis.plot([0,1],[0,1],color='.65',ls='--',lw=1)
    axis.set(xlabel='Tasa de falsos positivos',ylabel='Sensibilidad pCR',
             title='ROC OOF por paciente · desarrollo, sin nuevo test',xlim=(0,1),ylim=(0,1))
    axis.grid(alpha=.2); axis.legend(loc='lower right',fontsize=9)
    fig.tight_layout(); fig.savefig(destination/'01_roc_oof.png',dpi=170); plt.close(fig)
    fig,axis=plt.subplots(figsize=(8,4.8)); x=np.arange(5)
    for index,arm in enumerate(('reference','selected')):
        axis.bar(x+(index-.5)*.34,computed[arm]['fold_auc'],width=.34,label=labels[arm])
    axis.axhline(.6,color='.45',ls='--',label='Mínimo 0,60'); axis.axhline(.7,color='.65',ls=':',label='Objetivo 0,70')
    axis.set(xticks=x,xticklabels=np.arange(1,6),xlabel='Fold exterior',ylabel='AUC cruda',ylim=(0,1),
             title='Evaluación exterior después de congelar las elecciones')
    axis.legend(fontsize=9); axis.grid(axis='y',alpha=.2)
    fig.tight_layout(); fig.savefig(destination/'02_auc_por_fold.png',dpi=170); plt.close(fig)
    for metric,filename,ylabel in [('patient_auc','03_auc_modelos_seleccionados.png','AUC paciente'),
                                   ('loss','04_perdida_modelos_seleccionados.png','Pérdida de su propio brazo')]:
        fig,axes=plt.subplots(5,2,figsize=(12,15))
        for fold in frozen['folds']:
            for column,entry in enumerate(fold['selected']['models']):
                folder=run/entry['path']; history=pd.read_csv(folder/'history.csv')
                summary=json.loads((folder/'summary.json').read_text())
                axis=axes[fold['fold'],column]
                fit_column='fit_patient_auc' if metric=='patient_auc' else 'fit_eval_loss'
                inner_column='selection_patient_auc' if metric=='patient_auc' else 'selection_loss'
                axis.plot(history.epoch,history[fit_column],label='Fit')
                axis.plot(history.epoch,history[inner_column],label='Selección interna')
                axis.axvline(summary['best_epoch'],color='.55',ls='--',lw=1,label='Checkpoint elegido')
                axis.set(title=f"Fold {fold['fold']+1} · semilla {entry['seed']} · {fold['selected']['loss']}",
                         xlabel='Época',ylabel=ylabel)
                if metric=='patient_auc': axis.set_ylim(0,1)
                axis.grid(alpha=.2); axis.legend(fontsize=7)
        fig.suptitle('Diez modelos seleccionados · curvas completas de cada ejecución')
        fig.tight_layout(rect=(0,0,1,.98)); fig.savefig(destination/filename,dpi=160); plt.close(fig)
    entries=[]
    for fold in frozen['folds']:
        for entry in fold['selected']['models']:
            weight=run/entry['path']/'inference.pt'
            artifact=torch.load(weight,map_location='cpu',weights_only=True)
            entries.append({'path':os.path.relpath(weight,destination),'sha256':base.sha256(weight),
                            'seed':entry['seed'],'fold':fold['fold'],'outer_fold':fold['fold'],
                            'model_config':artifact['model_config'],'training_trial':fold['selected']['trial'],
                            'training_loss':fold['selected']['loss']})
    model_status=('experimental_conservado_sin_promover' if computed['selected']['pooled_raw_auc']>=.60
                  else 'ensayo_insuficiente_conservado')
    manifest={'schema_version':1,'status':model_status,
              'recommended_for_prediction':False,'model_class':'CNNMultiescalaCinetica',
              'models':entries,'slice_aggregation':'mean','ensemble':'mean probabilities across ten models',
              'calibration':calibration,'threshold':threshold,'threshold_policy':evaluation['future_policy'],
              'selection':{'metrics_raw_oof':image.metricas_binarias(selected.label,probabilities['selected'],.5),
                           'crossfit_evaluation':computed['selected'],'method':protocol['selection'],
                           'minimum_auc':.60,'objective_auc':.70},
              'preprocessing':{'phases':['PRE','EARLY','LATE'],'shape':[3,256,256],
                               'clinical_required':False,'phase_scaling':'uint8/255',
                               'center_crop_sizes':[256,128,64],'encoder_input_size':[128,128],
                               'scale_weights_shared':True,
                               'kinetic_maps':['EARLY-PRE','LATE-PRE','EARLY-LATE']},
              'test_evaluated_once':False,
              'provenance_files':[{'path':name,'sha256':base.sha256(destination/name)} for name in
                                  ('evaluacion.json','seleccion_congelada.json','protocolo.json',
                                   'oof_pacientes_selected.csv','oof_pacientes_reference.csv',
                                   'ejecuciones.json','diagnostico_ejecuciones.csv','verificacion_entrenamiento.json')]}
    manifest_path=destination/'manifiesto_candidato.json'
    base.json_write(manifest_path,manifest)
    manifest_path.with_suffix('.json.sha256').write_text(base.sha256(manifest_path)+'\n')
    print(json.dumps(evaluation,indent=2,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ejecucion',type=Path,default=tune.OUTPUT)
    parser.add_argument('--salida',type=Path,default=RESULTS_DIR/'15_informe_cinetico_20261021')
    parser.add_argument('--bootstrap',type=int,default=2000)
    args=parser.parse_args()
    if args.bootstrap<1000: parser.error('Usa al menos 1000 remuestreos')
    collect(args.ejecucion,args.salida,args.bootstrap)
