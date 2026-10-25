"""Auditoría de todos los PNG de train y preparación de roles, sin entrenar.

Nunca abre imágenes de test/validation. Los avisos de intensidad y desplazamiento
son heurísticos para revisión, no diagnósticos ni reglas de exclusión.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import StratifiedShuffleSplit

from . import entrenamiento as base
from . import entrenamiento_cinetico as cnn
from . import entrenamiento_imagen as image
from .paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

OUTPUT=RESULTS_DIR/'26_auditoria_train_protocolo_20261025'
PHASES=('PRE','EARLY','LATE')
PATHS=('path_pre','path_early','path_late')


def now():return datetime.now(ZoneInfo('Europe/Madrid')).isoformat()
def write(path,value):base.json_write(Path(path),value)


def train_metadata(root):
    """Las etiquetas se convierten/comparan únicamente dentro de train."""
    root=Path(root);samples=base.cargar_train(root,include_clinical=False)
    with (root/'metadata/patients.csv').open(newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['split']=='train']
    patients=pd.DataFrame(rows)
    patients['pCR']=pd.to_numeric(patients.pCR,errors='raise').astype(int)
    if patients.pid.duplicated().any() or set(patients.pid)!=set(samples.patient_id):
        raise ValueError('IDs train no coinciden entre metadatos')
    joined=samples[['patient_id','pCR','dataset']].drop_duplicates().merge(
        patients[['pid','pCR','dataset']],left_on='patient_id',right_on='pid',suffixes=('_sample','_patient'),validate='one_to_one')
    if not (joined.pCR_sample.eq(joined.pCR_patient)&joined.dataset_sample.eq(joined.dataset_patient)).all():
        raise ValueError('Etiqueta/cohorte incoherente entre CSV')
    if samples.groupby('patient_id').dataset.nunique().ne(1).any():raise ValueError('Cohorte inconsistente')
    if samples.duplicated(['patient_id','slice_index']).any():raise ValueError('Índice axial duplicado por paciente')
    return samples.sort_values(['patient_id','sample_id']).reset_index(drop=True),patients


def checked_path(root,row,column,phase):
    root=Path(root);relative=Path(getattr(row,column))
    expected=Path('dataset/train')/str(row.patient_id)/(str(row.sample_id)+'_'+phase+'.png')
    if relative!=expected:raise ValueError('Ruta/nombre no corresponde a paciente, corte y fase de train')
    path=root/relative
    if not path.resolve().is_relative_to((root/'dataset/train').resolve()):raise ValueError('Ruta fuera de train')
    match=re.search(r'_z(\d+)$',str(row.sample_id))
    if match is None or int(match.group(1))!=int(row.slice_index):raise ValueError('z del nombre e índice axial distintos')
    return path


def gradient_view(array,factor=4):
    array=np.asarray(array,dtype=np.float32)
    if array.ndim!=2 or any(n%factor for n in array.shape):raise ValueError('Imagen 2D divisible por factor requerida')
    h,w=array.shape
    small=array.reshape(h//factor,factor,w//factor,factor).mean((1,3))
    dy,dx=np.gradient(gaussian_filter(small,.7));g=np.hypot(dy,dx)
    g=(g-g.mean())*np.outer(np.hanning(len(g)),np.hanning(g.shape[1]))
    return g


def translation_diagnostic(reference,moving,limit=6,factor=4):
    """NCC acotada de gradientes; desplazamiento que alinearía moving con reference.

    No aplica correcciones. La resolución es4px; cambios de contraste/anatomía
    pueden producir máximos falsos y requieren inspección humana.
    """
    a,b=gradient_view(reference,factor),gradient_view(moving,factor)
    norm=float(np.linalg.norm(a)*np.linalg.norm(b))
    if norm<1e-10:return {'estimable':False,'dy':0,'dx':0,'pixels':0.,'ncc_zero':0.,'ncc_best':0.,'gain':0.,'at_boundary':False,'flag':False}
    correlation=np.fft.ifft2(np.fft.fft2(a)*np.conj(np.fft.fft2(b))).real/norm
    shifts=np.arange(-limit,limit+1)
    region=correlation[np.ix_(shifts%a.shape[0],shifts%a.shape[1])]
    iy,ix=np.unravel_index(np.argmax(region),region.shape)
    dy,dx=int(shifts[iy]*factor),int(shifts[ix]*factor)
    zero,best=float(correlation[0,0]),float(region[iy,ix]);distance=float(np.hypot(dy,dx))
    boundary=bool(abs(shifts[iy])==limit or abs(shifts[ix])==limit)
    return {'estimable':True,'dy':dy,'dx':dx,'pixels':distance,'ncc_zero':zero,'ncc_best':best,'gain':best-zero,
        'at_boundary':boundary,'flag':bool(distance>=8 and best-zero>=.10 and best>=.50 and not boundary)}


def inspect_triplet(root,row):
    record={'sample_id':row.sample_id,'patient_id':row.patient_id,'cohort':row.dataset,'label':int(row.pCR),'fold':int(row.fold),'slice_index':int(row.slice_index)}
    arrays=[];files=[];histograms=[]
    try:
        for phase,column in zip(PHASES,PATHS):
            path=checked_path(root,row,column,phase);before=path.stat();data=path.read_bytes()
            with Image.open(BytesIO(data)) as im:
                if im.format!='PNG' or im.mode!='L' or im.size!=(256,256):raise ValueError('Se exige PNG L 256×256')
                array=np.asarray(im).copy()
            if array.dtype!=np.uint8:raise ValueError('Se exige uint8')
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Archivo cambió mientras se leía')
            hist=np.bincount(array.ravel(),minlength=256);histograms.append(hist)
            pixel_hash=hashlib.sha256(array.tobytes()).hexdigest()
            files.append({'path':str(path.relative_to(root)),'patient_id':row.patient_id,'sample_id':row.sample_id,'phase':phase,
                'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'file_sha256':hashlib.sha256(data).hexdigest(),'pixel_sha256':pixel_hash})
            key=phase.lower();cdf=np.cumsum(hist)
            record.update({key+'_min':int(array.min()),key+'_max':int(array.max()),key+'_zero_fraction':float(hist[0]/array.size),
                key+'_saturation_fraction':float(hist[-1]/array.size),key+'_levels':int((hist>0).sum()),key+'_p99':int(np.searchsorted(cdf,array.size*.99))})
            arrays.append(array)
        x=np.stack(arrays).astype(np.float32)/255;mask=x.max(0)>.05
        if mask.sum()==0:raise ValueError('Tripleta sin foreground >0,05')
        means=x[:,mask].mean(1,dtype=np.float64)
        for phase,value in zip(PHASES,means):record[phase.lower()+'_mean_fg']=float(value)
        record['foreground_fraction']=float(mask.mean());record['early_minus_pre']=float(means[1]-means[0]);record['late_minus_early']=float(means[2]-means[1])
        positive=np.maximum(x[1]-x[0],0)*mask;energy=float(positive.sum(dtype=np.float64));yy,xx=np.indices((256,256))
        record['enhancement_centroid_y']=float((positive*yy).sum()/energy) if energy>0 else 127.5
        record['enhancement_centroid_x']=float((positive*xx).sum()/energy) if energy>0 else 127.5
        for size in (64,128):
            start=(256-size)//2;record[f'enhancement_fraction_center_{size}']=float(positive[start:start+size,start:start+size].sum()/energy) if energy>0 else 0.
        flags=[]
        if record['early_minus_pre']<-.01:flags.append('realce_medio_negativo')
        if means[2]>.75 and means[2]/max(means[0],.001)>5:flags.append('late_extremadamente_brillante')
        if any(record[p.lower()+'_levels']<16 for p in PHASES):flags.append('pocos_niveles')
        if any(record[p.lower()+'_saturation_fraction']>.25 for p in PHASES):flags.append('saturacion_alta')
        if len({f['pixel_sha256'] for f in files})<3:flags.append('fases_identicas')
        for name,i,j in [('pre_early',0,1),('early_late',1,2),('pre_late',0,2)]:
            diagnostic=translation_diagnostic(arrays[i],arrays[j])
            for key,value in diagnostic.items():record[name+'_'+key]=value
            if diagnostic['flag']:flags.append('posible_movimiento_'+name)
        record['triplet_sha256']=hashlib.sha256(b''.join(a.tobytes() for a in arrays)).hexdigest()
        record['flags']=';'.join(flags);record['error']=''
        return record,files,histograms
    except Exception as exc:
        record['error']=str(exc);record['flags']='error_estructural'
        return record,files,[]


def historical_inventory():
    return {str(p.relative_to(PROJECT_ROOT)):base.sha256(p) for p in (PROJECT_ROOT/'modelos/versiones').rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def source_metadata_checks(patients,samples):
    columns=['pid','n_z','n_times','pre','post_early','post_late','n_xy','sraw','eraw','scol','ecol']
    if not set(columns).issubset(patients.columns):return {'available':False}
    frame=patients[columns].copy()
    for column in columns[1:]:frame[column]=pd.to_numeric(frame[column],errors='coerce')
    temporal=(frame.pre.ge(0)&frame.pre.lt(frame.post_early)&frame.post_early.lt(frame.post_late)&frame.post_late.lt(frame.n_times))
    boxes=(frame.sraw.ge(0)&frame.sraw.lt(frame.eraw)&frame.eraw.le(frame.n_xy)&frame.scol.ge(0)&frame.scol.lt(frame.ecol)&frame.ecol.le(frame.n_xy))
    paired=samples[['patient_id','slice_index']].merge(frame[['pid','n_z']],left_on='patient_id',right_on='pid',validate='many_to_one')
    invalid_z=paired[~(paired.slice_index.ge(0)&paired.slice_index.lt(paired.n_z))]
    return {'available':True,'phase_index_order_invalid':frame.loc[~temporal,'pid'].tolist(),'source_bbox_bounds_invalid':frame.loc[~boxes,'pid'].tolist(),
        'axial_indices_out_of_source_volume':invalid_z.to_dict('records'),
        'bbox_mapping_to_png':'No verificado: los límites del CSV son del volumen fuente; no se asumen coordenadas del PNG MinCrop',
        'phase_source_limit':'Los PNG no permiten confirmar el instante DCE real ni reconstruir el volumen fuente; solo coherencia de índices y nombres'}


def audit(root=DATA_DIR,output=OUTPUT,workers=8):
    root=Path(root);output=Path(output)
    if (output/'auditoria.json').exists():raise ValueError('Auditoría existente: no sobrescribir evidencia')
    output.mkdir(parents=True,exist_ok=True);start=time.monotonic();samples,patients=train_metadata(root)
    inventory=historical_inventory();write(output/'versiones_antes.json',inventory)
    sources={str(p.relative_to(PROJECT_ROOT)):base.sha256(p) for p in [Path(__file__),PROJECT_ROOT/'src/cancer_mama/entrenamiento.py',PROJECT_ROOT/'src/cancer_mama/datos.py']}
    metadata_hashes={f:base.sha256(root/'metadata'/f) for f in ('samples.csv','patients.csv')}
    expected={str(checked_path(root,row,col,phase).relative_to(root)) for row in samples.itertuples() for phase,col in zip(PHASES,PATHS)}
    actual={str(p.relative_to(root)) for p in (root/'dataset/train').rglob('*.png')}
    records=[];files=[];hist=np.zeros((3,256),dtype=np.int64)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,(record,entries,counts) in enumerate(pool.map(lambda row:inspect_triplet(root,row),samples.itertuples()),1):
            records.append(record);files.extend(entries)
            if counts:hist+=np.stack(counts)
            if i%250==0 or i==len(samples):
                write(output/'estado.json',{'status':'auditing_train_only','cuts_done':i,'cuts_total':len(samples),'updated_at':now()})
                print(f'AUDITORÍA {i}/{len(samples)} cortes',flush=True)
    frame=pd.DataFrame(records);file_frame=pd.DataFrame(files)
    frame.to_csv(output/'cortes_auditados.csv',index=False);file_frame.to_csv(output/'inventario_png.csv',index=False)
    good=frame[frame.error.eq('')].copy()
    if good.empty:raise ValueError('Ninguna tripleta válida')
    numeric=['pre_mean_fg','early_mean_fg','late_mean_fg','early_minus_pre','late_minus_early','foreground_fraction',
        'enhancement_fraction_center_64','enhancement_fraction_center_128','pre_early_pixels','early_late_pixels','pre_late_pixels']
    means=good.groupby('patient_id')[numeric].mean().reset_index()
    counts=good.groupby('patient_id').agg(cuts=('sample_id','size'),flagged_cuts=('flags',lambda s:int(s.ne('').sum())))
    patient_table=samples[['patient_id','dataset','pCR','fold']].drop_duplicates().merge(means,on='patient_id',how='left').merge(counts,on='patient_id',how='left')
    patient_table.to_csv(output/'pacientes_auditados.csv',index=False)
    cross=[]
    for digest,g in file_frame.groupby('pixel_sha256'):
        if g.patient_id.nunique()>1:cross.append({'pixel_sha256':digest,'patients':sorted(g.patient_id.unique().tolist()),'files':g.path.tolist()})
    triplet_cross=[];adjacent_duplicates=[]
    for digest,g in good.groupby('triplet_sha256'):
        if len(g)>1:
            item={'sha256':digest,'patients':sorted(g.patient_id.unique().tolist()),'samples':g.sample_id.tolist()}
            (triplet_cross if g.patient_id.nunique()>1 else adjacent_duplicates).append(item)
    write(output/'duplicados.json',{'cross_patient_phase_pixels':cross,'cross_patient_triplets':triplet_cross,'within_patient_repeated_triplets':adjacent_duplicates})
    flagged=good[good['flags'].ne('')].copy();flagged.to_csv(output/'avisos_para_revision.csv',index=False)
    flag_counts={flag:int(good['flags'].str.split(';').map(lambda x:flag in x).sum()) for flag in sorted({v for s in good['flags'] for v in s.split(';') if v})}
    cohorts=[]
    for name,g in patient_table.groupby('dataset'):
        cohorts.append({'cohort':name,'patients':len(g),'pcr':int(g.pCR.sum()),'prevalence':float(g.pCR.mean()),
            'negative_mean_enhancement_patients':int(g.early_minus_pre.lt(0).sum()),
            'median_enhancement':float(g.early_minus_pre.median()),'median_energy_center_128':float(g.enhancement_fraction_center_128.median())})
    for name,digest in metadata_hashes.items():
        if base.sha256(root/'metadata'/name)!=digest:raise ValueError('Metadatos cambiaron durante auditoría')
    for name,digest in inventory.items():
        if base.sha256(PROJECT_ROOT/name)!=digest:raise ValueError('Versión histórica modificada')
    for row in file_frame.itertuples():
        stat=(root/row.path).stat()
        if stat.st_size!=row.bytes or stat.st_mtime_ns!=row.mtime_ns:raise ValueError('PNG cambió durante auditoría')
    fingerprint=hashlib.sha256()
    for row in file_frame.sort_values('path').itertuples():fingerprint.update((row.path+':'+row.file_sha256+'\n').encode())
    summary={'completed_at':now(),'scope':'Todos los PNG referidos por train; identidad global de splits por metadata sin analizar etiquetas fuera de train',
        'patients':int(samples.patient_id.nunique()),'cuts':len(samples),'expected_png':len(expected),'opened_png':len(file_frame),
        'structural_errors':frame.loc[frame.error.ne(''),['sample_id','patient_id','error']].to_dict('records'),
        'missing_files':sorted(expected-actual),'unindexed_png_train':sorted(actual-expected),'dataframe_patient_overlap':False,
        'cross_patient_duplicate_phase_groups':len(cross),'cross_patient_duplicate_triplets':len(triplet_cross),
        'within_patient_duplicate_triplets':len(adjacent_duplicates),'flag_counts':flag_counts,'flagged_cuts':len(flagged),
        'flagged_patients':int(flagged.patient_id.nunique()),'cohorts':cohorts,
        'patient_mean_early_gt_pre_fraction':float(patient_table.early_minus_pre.gt(0).mean()),
        'histograms':{phase:hist[i].tolist() for i,phase in enumerate(PHASES)},
        'data_tree_sha256':fingerprint.hexdigest(),'metadata_sha256':metadata_hashes,'source_sha256':sources,'source_metadata_checks':source_metadata_checks(patients,samples),
        'seconds':time.monotonic()-start,'test_images_opened':0,'private_images_opened':0,'data_modified':False,'versions_unchanged':True,
        'heuristics':{'negative_enhancement':'media foreground EARLY−PRE <−0,01','late_extreme':'media foreground LATE>0,75 y LATE/PRE>5',
            'translation':'NCC de gradientes, rejilla4px, radio24px; distancia>=8, ganancia>=0,10, NCC>=0,50, máximo interior',
            'intensity':'foreground=max(PRE,EARLY,LATE)>0,05; no segmentación tumoral'},
        'limits':['Los avisos no prueban datos corruptos ni autorizan exclusiones','No se puede verificar el orden fisiológico individual solo por la media de una imagen',
            'Sin máscaras ni transformación documentada del recorte no se localiza el tumor con certeza','Los duplicados entre train y otros splits no se han contrastado por píxeles porque test/validation no se abren',
            'El estimador de movimiento no distingue siempre realce, anatomía y desplazamiento; requiere revisión visual']}
    write(output/'auditoria.json',summary);write(output/'estado.json',{'status':'audit_complete','cuts_done':len(samples),'updated_at':now()})
    figures(root,output,samples,good)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['histograms','source_sha256']},ensure_ascii=False),flush=True)
    return summary


def draw_case(axes,row,root,title):
    arrays=[]
    for phase,col in zip(PHASES,PATHS):
        with Image.open(checked_path(root,row,col,phase)) as im:arrays.append(np.asarray(im,dtype=float)/255)
    for ax,array,name in zip(axes,[*arrays,arrays[1]-arrays[0]],[*PHASES,'EARLY−PRE']):
        ax.imshow(array,cmap='gray' if name!='EARLY−PRE' else 'coolwarm',vmin=0 if name!='EARLY−PRE' else -.5,vmax=1 if name!='EARLY−PRE' else .5)
        ax.set_title(name,fontsize=9);ax.axis('off')
    axes[0].text(-.04,.5,title,transform=axes[0].transAxes,ha='right',va='center',fontsize=8)


def figures(root,output,samples,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Selección de ejemplos uniforme y prefijada por cohorte/clase, sin usar AUC.
    rng=np.random.default_rng(20261025);ids=[]
    patients=samples[['patient_id','dataset','pCR']].drop_duplicates().sort_values('patient_id')
    for _,g in patients.groupby(['dataset','pCR']):ids.append(str(rng.choice(g.patient_id.to_numpy())))
    chosen=[]
    for pid in ids:
        g=samples[samples.patient_id.eq(pid)].sort_values('slice_index');chosen.append(g.iloc[len(g)//2])
    fig,axes=plt.subplots(len(chosen),4,figsize=(11,2.6*len(chosen)),squeeze=False)
    for i,row in enumerate(chosen):draw_case(axes[i],row,root,f'{row.dataset} · pCR={row.pCR}\n{row.patient_id}')
    fig.suptitle('Ejemplos de train por cohorte y clase; no son predicciones ni diagnósticos')
    fig.text(.5,.008,'BreastDCEDL MinCrop · CC BY-NC 4.0 · Uso educativo',ha='center',fontsize=8)
    fig.tight_layout(rect=(.13,.02,1,.96));fig.savefig(output/'01_ejemplos_cohortes.png',dpi=130);plt.close(fig)
    flagged=metrics[metrics['flags'].ne('')].copy();selected=[]
    # Un representante por tipo, evitando repetir paciente; prevalece orden prefijado.
    for flag in ('late_extremadamente_brillante','realce_medio_negativo','posible_movimiento_pre_early','posible_movimiento_early_late','posible_movimiento_pre_late','fases_identicas'):
        g=flagged[flagged['flags'].str.contains(flag,regex=False)&~flagged.patient_id.isin([x.patient_id for x in selected])].sort_values('sample_id')
        if len(g):selected.append(g.iloc[0])
    if selected:
        fig,axes=plt.subplots(len(selected),4,figsize=(11,2.7*len(selected)),squeeze=False)
        for i,metric in enumerate(selected):
            row=samples.loc[samples.sample_id.eq(metric.sample_id)].iloc[0]
            draw_case(axes[i],row,root,f'{row.patient_id}\n{row.sample_id}\n'+str(metric['flags']).replace(';','\n'))
        fig.suptitle('Avisos heurísticos para inspección; no se corrigen ni excluyen estos casos')
        fig.text(.5,.008,'BreastDCEDL MinCrop · CC BY-NC 4.0 · Solo train',ha='center',fontsize=8)
        fig.tight_layout(rect=(.20,.02,1,.96));fig.savefig(output/'02_avisos_revision.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    colors={'duke':'#397392','spy1':'#b46b29','spy2':'#348065'}
    for cohort,g in metrics.groupby('cohort'):
        axes[0].hist(g.early_minus_pre,bins=40,histtype='step',density=True,label=cohort,color=colors[cohort])
        axes[1].hist(g.pre_early_pixels,bins=np.arange(0,37,4),histtype='step',density=True,label=cohort,color=colors[cohort])
        axes[2].hist(g.enhancement_fraction_center_128,bins=20,histtype='step',density=True,label=cohort,color=colors[cohort])
    axes[0].set(title='Realce medio de foreground',xlabel='EARLY−PRE')
    axes[1].set(title='Traslación sugerida PRE/EARLY',xlabel='Píxeles; resolución4px')
    axes[2].set(title='Fracción de realce positivo en centro128',xlabel='No equivale a cobertura tumoral')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Descripciones por corte; las métricas del modelo deben seguir siendo por paciente')
    fig.tight_layout();fig.savefig(output/'03_distribuciones.png',dpi=140);plt.close(fig)


def prepare_protocol(root=DATA_DIR,output=OUTPUT):
    root=Path(root);output=Path(output);dest=output/'protocolo_cinco_folds'
    if dest.exists():raise ValueError('Protocolo existente: no sobrescribir')
    samples,_=train_metadata(root);dest.mkdir(parents=True)
    cfg=cnn.configuracion();cfg['training'].update(lr=.0003,weight_decay=.01)
    cfg['evaluation'].update(min_specificity=.95)
    records=[]
    for fold in range(5):
        fit,selection,outer=image.particion_anidada(samples,fold,cfg)
        rows=[]
        for role,frame in [('fit',fit),('selection',selection),('outer',outer)]:
            p=frame[['patient_id','dataset','pCR','fold']].drop_duplicates().copy();p['role']=role;rows.append(p)
        roles=pd.concat(rows).sort_values('patient_id').reset_index(drop=True)
        if roles.patient_id.duplicated().any() or len(roles)!=samples.patient_id.nunique():raise ValueError('Solapamiento de roles')
        path=dest/f'roles_fold_{fold}.csv';roles.to_csv(path,index=False)
        patients_fit=fit.drop_duplicates('patient_id')
        records.append({'fold':fold,'counts':roles.groupby('role').size().to_dict(),'sha256':base.sha256(path),
            'strata':roles.groupby(['role','dataset','pCR']).size().rename('n').reset_index().to_dict('records'),
            'pos_weight':float(patients_fit.pCR.eq(0).sum()/patients_fit.pCR.eq(1).sum())})
    # Despliegue futuro de una pareja, evitando calibrar diez redes con puntuaciones de dos.
    patients=samples[['patient_id','dataset','pCR','fold']].drop_duplicates().sort_values('patient_id').reset_index(drop=True)
    strata=patients.dataset.astype(str)+'|'+patients.pCR.astype(str)
    fit_idx,val_idx=next(StratifiedShuffleSplit(n_splits=1,test_size=round(len(patients)*.15),random_state=20261026).split(patients.patient_id,strata))
    final=patients.copy();final['role']='fit';final.loc[val_idx,'role']='selection_and_policy'
    final.to_csv(dest/'roles_modelo_final.csv',index=False)
    selected_cfg=cfg['model'];manifest=json.loads((PROJECT_ROOT/'modelos/versiones/v007_cnn_cinetica_multiescala_20261021/manifiesto.json').read_text())
    if any(m['model_config']!=selected_cfg for m in manifest['models']):raise ValueError('Arquitectura distinta de referencia')
    protocol={'prepared_at':now(),'status':'prepared_no_training_authorized_or_executed','source_sha256':base.sha256(Path(__file__)),
        'config':cfg,'parameters':593521,'folds':records,'seeds':[42,2026],'losses':['normal','weighted'],'reference_cv_runs':20,
        'only_change_now':'Plan de tres roles, sin bloque extra de calibración; ninguna modificación de datos o entrenamiento activo',
        'selection':'Cada semilla elige época en selection por AUC paciente; la pareja elige BCE por AUC ensemble, empate normal',
        'aggregation':'Media de probabilidades de todos los cortes y dos semillas; no seleccionar cortes ni usar clínica',
        'objective':'BCEWithLogitsLoss por corte con contribución1/n_cortes; normalizador fijo; pos_weight por pacientes de fit',
        'policy':{'calibration':'Platt C1, pendiente no negativa; solo selection, exactamente misma pareja y media por paciente',
            'primary_threshold':'Máxima sensibilidad con especificidad>=.95 en selection; empates precisión, especificidad, umbral mayor',
            'secondary_threshold':'MacroF1 máximo en selection; descriptivo, sin escoger política mirando outer',
            'reused_selection_warning':'Selection comparte elección de época/pérdida y política; sus métricas son optimistas y nunca comprobación independiente',
            'outer_guard':'Cada fold exterior evalúa el procedimiento completo tras congelar épocas, pérdida, calibración y umbrales'},
        'oof':'Solo la pareja de su fold exterior predice a cada paciente; nunca ensemble de todos los folds sobre train',
        'primary':'Media±DE de AUC cruda por paciente de cinco folds; pooled cruda/calibrada secundarias y claramente separadas',
        'comparison':'Cualquier candidato requerirá protocolo nuevo prefijado, mismos roles/semillas/presupuesto; esta preparación no elige una modificación',
        'final_model':{'status':'Solo diseño; no se entrena ni activa ahora','fit_patients':len(fit_idx),'selection_and_policy_patients':len(val_idx),
            'split_sha256':base.sha256(dest/'roles_modelo_final.csv'),'seed':20261026,'runs_if_later_authorized':4,
            'deployment':'Una pareja de semillas con BCE elegida en selection_and_policy; calibración y umbral de esa misma pareja, sin mezcla de cinco parejas',
            'limitation':'La CV evalúa el procedimiento con fit~746, no los futuros pesos ajustados con932; val final es selección, no evaluación independiente'},
        'gates_before_training':['Resolver errores estructurales/identidad si los hay','Revisar avisos sin excluir por etiqueta ni rendimiento',
            'Congelar hipótesis de un solo cambio y protocolo','Autorización de entrenamiento posterior: este paso no lo inicia'],
        'limits':['Train completo tiene antecedentes adaptativos; cinco folds no restablecen validación ciega',
            'No garantiza alcanzar0.60 ni corregir sobreajuste','Test histórico se conserva; no se vuelve a usar para ajustes',
            'Confirmación nueva solo con evaluación privada por el profesor u otra evaluación permitida',
            'Datos, modelos archivados, aplicación y remoto no se modifican'],
        'test_images_loaded':0,'private_images_loaded':0,'training_started':False}
    write(dest/'protocolo.json',protocol);(dest/'protocolo.json.sha256').write_text(base.sha256(dest/'protocolo.json')+'\n')
    print(json.dumps({'status':protocol['status'],'folds':[{'fold':f['fold'],**f['counts']} for f in records],
        'final':protocol['final_model'],'training_started':False},ensure_ascii=False),flush=True)
    return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['auditar','preparar-protocolo']);parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    if args.workers<1 or args.workers>16:parser.error('workers entre1 y16')
    (audit(output=args.output,workers=args.workers) if args.action=='auditar' else prepare_protocol(output=args.output))


if __name__=='__main__':main()
