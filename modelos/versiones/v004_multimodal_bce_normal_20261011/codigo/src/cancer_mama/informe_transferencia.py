"""Consolida resultados completos y negativos de las cuatro tandas documentadas."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import entrenamiento as base
from . import experimento_transfer as transfer
from . import experimento_local as local
from . import experimento_finetune as finetune
from . import experimento_textura as textura
from . import informe_loss as diagnosis
from .paths import PROJECT_ROOT, RESULTS_DIR

OUTPUT = RESULTS_DIR / '09_informe_transferencia_20261010'
NAMES = {'control_auc':'Control: CNN original', 'control_logloss':'Candidata v003',
    'lineal':'ResNet + logística', 'no_lineal':'ResNet + árboles', 'fusion_tardia':'ResNet + mezcla',
    'alexnet_svm':'AlexNet local + SVM', 'resnet_svm':'ResNet + SVM',
    'alexnet_lineal':'AlexNet local + logística', 'finetune':'ResNet18 ajustada + clínica fija',
    'textura_lineal':'Textura + logística', 'textura_svm':'Textura + SVM', 'textura_arboles':'Textura + árboles'}


def main():
    sources = {'tanda1':transfer.DEFAULT_OUTPUT,'tanda2':local.DEFAULT_OUTPUT,'tanda3':finetune.DEFAULT_OUTPUT,
               'tanda4':textura.DEFAULT_OUTPUT}
    evidence={};frames={};decisions={};epochs=[];curves=[]
    for name,source in sources.items():
        state=json.loads((source/'estado.json').read_text())
        if state['status']!='complete' or state.get('protected_unchanged') is not True:
            raise ValueError(f'{name} aún no está completa/verificada')
        dest=OUTPUT/name;dest.mkdir(parents=True,exist_ok=True)
        for summary_path in source.glob('**/summary.json'):
            if 'pruebas' in summary_path.parts:continue
            summary=json.loads(summary_path.read_text())
            for file,digest in summary['files'].items():
                path=summary_path.parent/file
                if base.sha256(path)!=digest:raise ValueError(f'Artefacto cambiado: {path}')
                evidence[str(path.relative_to(PROJECT_ROOT))]=digest
            if name!='tanda3':
                target=dest/summary_path.parent.name;target.mkdir(exist_ok=True)
                for pattern in ('*.csv','*.json'):
                    for path in summary_path.parent.glob(pattern):shutil.copy2(path,target/path.name)
        for sub in ('comparacion',''):
            origin=source/sub
            for pattern in (('*.csv','*.json') if sub else ('protocolo.json','estado.json','features.json')):
                for path in origin.glob(pattern):shutil.copy2(path,dest/path.name)
        result=json.loads((source/'comparacion/resumen.json').read_text())
        if name!='tanda3':
            for family,comparison in result['comparisons'].items():
                frames[family]=pd.read_csv(source/f'comparacion/oof_calibrated_{family}.csv')
                decisions[family]=comparison
        else:
            frames['finetune']=pd.read_csv(source/'comparacion/oof_calibrated.csv')
            decisions['finetune']=result
            for folder in sorted(source.glob('seed_*/fold_*')):
                config=json.loads((folder/'config.json').read_text())
                summary=json.loads((folder/'summary.json').read_text())
                history=pd.read_csv(folder/'history.csv')
                curves.append((config['seed'],config['fold'],history,summary['tracker']['best_epoch']))
                epochs.append({'seed':config['seed'],'fold':config['fold'],**summary['tracker']})
                target=dest/f"seed_{config['seed']}_fold_{config['fold']}";target.mkdir(exist_ok=True)
                for file in ('config.json','summary.json','history.csv','inner.csv','oof.csv'):shutil.copy2(folder/file,target/file)
    controls={name:pd.read_csv(transfer.REFERENCE/f'oof_calibrated_inner_{selector}.csv')
              for name,selector in (('control_auc','auc'),('control_logloss','logloss'))}
    frames={**controls,**frames}
    keys=['patient_id','label','fold','cohort']
    for name,frame in frames.items():
        if not frame[keys].equals(controls['control_auc'][keys]):raise ValueError('Cobertura distinta')
        frame.to_csv(OUTPUT/f'oof_{name}.csv',index=False)
    rows=[{'candidate':name,**diagnosis.metrics(frame)} for name,frame in frames.items()]
    metrics=pd.DataFrame(rows);metrics.drop(columns='fold_auc').to_csv(OUTPUT/'metricas.csv',index=False)
    pd.DataFrame(epochs).to_csv(OUTPUT/'epocas_finetune.csv',index=False)
    base.json_write(OUTPUT/'entradas_sha256.json',evidence)
    base.json_write(OUTPUT/'resumen.json',{'metrics':rows,'comparisons':decisions,'test_images_loaded':0,
        'active_version':'v002_multimodal_clinico_20261007','adaptive_search':True,
        'assignment_eligible':False,'reason':'Assignment page 6 requires a 2D CNN from scratch; transfer learning is forbidden; texture classifiers are auxiliary only.'})

    os.environ.setdefault('MPLCONFIGDIR',str(PROJECT_ROOT/'.codex_tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,8),constrained_layout=True)
    labels=[NAMES[x] for x in metrics.candidate];ys=np.arange(len(metrics))
    colors=['#52616b','#8a949b',*['#338c99']*6,'#b75c37',*['#678342']*3]
    for ax,key,title in zip(axes,('log_loss','mean_fold_auc'),('Cambio de log-loss · menor es mejor','Cambio de AUC media · mayor es mejor')):
        reference_value=metrics.iloc[0][key]
        for y,(name,value),color in zip(ys,metrics[['candidate',key]].itertuples(index=False,name=None),colors):
            delta=value-reference_value
            if name in decisions:
                ci=next(v for k,v in decisions[name].items() if k.startswith('intervals') and 'v003' not in k)[key]
                ax.plot([ci['lower'],ci['upper']],[y,y],color=color,linewidth=2)
            ax.scatter([delta],[y],color=color,s=45,zorder=3)
        ax.set_yticks(ys,labels);ax.invert_yaxis();ax.set_title(title)
        ax.axvline(0,color='#303030',linestyle='--',linewidth=1)
        ax.set_xlabel(f'Diferencia frente al control ({reference_value:.4f})')
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Comparación interna: mismas 1.097 pacientes y separación de selección\nIntervalos ajustados por tanda; v003 se muestra solo como referencia puntual',fontsize=13)
    fig.savefig(OUTPUT/'01_comparacion.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(18,7),sharey=True,constrained_layout=True)
    for seed,fold,history,best in curves:
        ax=axes[0 if seed==42 else 1,fold]
        ax.plot(history.epoch,history.fit_log_loss,label='Fit',color='#267b89')
        ax.plot(history.epoch,history.selection_log_loss,label='Selección interna',color='#c65a3e')
        ax.axvline(best,color='#666',linestyle=':',label='Checkpoint elegido')
        ax.set_title(f'Semilla {seed} · fold {fold}');ax.set_xlabel('Época');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[0,0].set_ylabel('Log-loss por paciente');axes[1,0].set_ylabel('Log-loss por paciente')
    fig.suptitle('Ajuste parcial: evaluación sin dropout, todos los cortes y época 0 incluida')
    fig.savefig(OUTPUT/'02_curvas_finetune.png',dpi=150);plt.close(fig)
    favorable=[name for name,value in decisions.items() if value['promising_internal_result']]
    lines=['# Investigación y entrenamiento de mejoras del multimodal','',
        'Se buscaron estrategias documentadas, se fijaron cuatro protocolos y se entrenaron diez familias. '
        'Todas usan exclusivamente desarrollo y selección interna separada del fold exterior. '
        'La versión activa continúa siendo v002.','',
        '## Compatibilidad con el enunciado','',
        'Estas cuatro tandas se conservan como investigación descartada, no como candidatas para la entrega. '
        'La página 6, sección 7, exige CNN 2D desde cero y prohíbe expresamente modelos preentrenados y transfer learning. '
        'ResNet/AlexNet incumplen esa condición; los clasificadores de textura tampoco sustituyen la CNN obligatoria. '
        'La aplicación y v002 no fueron reemplazadas. Las siguientes pruebas se restringen al contrato docente.', '',
        '## Resultado','',
        ('Cumplen el criterio interno fijado: '+', '.join(NAMES[x] for x in favorable)+'.') if favorable else
        'Ninguna de las diez familias cumple el criterio de mejora interna fijado. No se ha demostrado una mejora del multimodal original.',
        '', '| Modelo | AUC media folds | AUC OOF calibrada | AP | Log-loss calibrada | Brier |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {NAMES[row['candidate']]} | {row['mean_fold_auc']:.4f} | {row['roc_auc']:.4f} | {row['average_precision']:.4f} | {row['log_loss']:.4f} | {row['brier']:.4f} |")
    lines += ['', '![Comparación](01_comparacion.png)', '',
        'El control es la arquitectura original reentrenada con selección interna y LR 0,0008. '
        'La AUC histórica de v002 (0,724568) utilizaba otro protocolo y no es directamente comparable. '
        'La AUC agrupada puede variar por la calibración distinta entre folds; la AUC media de folds también se informa.', '',
        '## Qué se probó y por qué','',
        '1. ResNet50 ImageNet congelada, dos escalas, PCA y fusión lineal, árboles o mezcla de probabilidades. '
        'La representación preentrenada reduce parámetros que deben aprenderse con estas pacientes.',
        '2. AlexNet pool2 con posiciones espaciales y selección supervisada dentro de fit; SVM/logística. '
        'También se probó selección supervisada y SVM sobre ResNet50.',
        '3. ResNet18 parcialmente ajustada, L2-SP, entrenamiento por paciente y corrección visual limitada de una clínica fija. '
        'Dos semillas y cinco folds; selección por log-loss interna.',
        '4. Descriptores de textura y diferencias entre fases a tres escalas, media/desviación entre cortes, selección supervisada y logística/SVM/árboles. '
        'Se usan las mismas cuatro variables clínicas. Los descriptores de PNG no son medidas físicas de perfusión.', '',
        '![Curvas](02_curvas_finetune.png)', '',
        '## Evidencia que orientó los ensayos','',
        '- [Scientific Reports, I-SPY1, 2021](https://www.nature.com/articles/s41598-021-93592-z): características locales de AlexNet y SVM. '
        'Sus mejores resultados combinan exploraciones previas y durante tratamiento; aquí solo hay exploraciones previas.',
        '- [AACR/Clinical Cancer Research, 2026](https://aacrjournals.org/clincancerres/article/32/4_Supplement/PS2-09-01/773092/Abstract-PS2-09-01-Optimizing-Transfer-Learning): '
        'transferencia parcial con ResNet en RM previa al tratamiento y combinación clínica. Es un resumen de congreso, no garantía de replicación.',
        '- [Cancers, 2025](https://www.mdpi.com/2072-6694/17/20/3394): extracción de características preentrenadas y clínica con clasificadores regularizados; incluye RM y TC.',
        '- [ICML 2018, L2-SP](https://proceedings.mlr.press/v80/li18a.html): regularización hacia los pesos preentrenados. Evidencia metodológica general, no validación específica de nuestro modelo.', '',
        '- [Radiómica de RM previa, PMID 35636571](https://pubmed.ncbi.nlm.nih.gov/35636571/): texturas con clínica y hallazgos radiológicos; orientó la cuarta tanda, sin reproducir sus segmentaciones ni sus medidas físicas.', '',
        '## Límites y reproducción','',
        'Los intervalos pareados están en resumen.json: primera tanda 98,33 %, segunda 99,17 %, tercera 99,29 % y cuarta 99,5 %. '
        'Se ajusta por las familias de cada tanda acumulada, pero no por toda la investigación histórica. '
        'Los intervalos condicionan a predicciones fijas y no representan validación externa.',
        'Los datos exteriores ya han sido observados durante investigación adaptativa. Probar más modelos hasta obtener un valor favorable '
        'no convierte ese valor en una confirmación. No se han leído imágenes del test ni de validación privada.',
        'Las OOF de SVM denominadas raw son sigmoid(margen), no probabilidades calibradas. La tabla principal usa Platt ajustado solo con selección interna.',
        'Las cuatro tandas tienen módulos, protocolos y artefactos locales separados. Los pesos de trabajo y las características son locales; '
        'este informe conserva métricas, predicciones, hashes, curvas y selección de checkpoints. '
        'Se regenera con `.venv/bin/python -B -m cancer_mama.informe_transferencia` cuando se dispone de los runs completos.']
    (OUTPUT/'INFORME_RESULTADOS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(OUTPUT),'promising':favorable},ensure_ascii=False))


if __name__=='__main__':main()
