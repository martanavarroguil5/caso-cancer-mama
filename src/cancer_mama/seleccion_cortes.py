"""Piloto 10/3/1 cortes de fit; selección interna y exterior fijas, sin test."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import hashlib
import itertools

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch

from . import entrenamiento as base
from . import entrenamiento_imagen as imagen
from . import entrenamiento_antioverfit as anti
from .paths import DATA_DIR, RESULTS_DIR

OUTPUT = RESULTS_DIR / '18_piloto_seleccion_cortes_20261021'
RECIPE = 'medoid_triplet32_joint_center_scale_then_farthest_spatial_rank_v1'
KS = (10, 3, 1)


def descriptors(paths: list[list[Path]]) -> np.ndarray:
    """Representación por corte; misma escala para las tres fases, sin etiquetas."""
    vectors = []
    for triplet in paths:
        planes = []
        for path in triplet:
            with Image.open(path) as im:
                planes.append(np.asarray(im.convert('L').resize((32, 32), Image.Resampling.BOX), dtype=float)/255)
        x = np.stack(planes)
        # Conserva diferencias entre fases; evita que solo domine el brillo global.
        vectors.append(((x-x.mean())/max(float(x.std()), .02)).ravel())
    return np.stack(vectors)


def select_indices(features: np.ndarray, count: int) -> list[int]:
    """Medoide y diversidad con separación de posiciones en el listado axial.

    No supone que slice_index y límites tumorales compartan coordenadas.
    Los empates se resuelven por orden axial estable.
    """
    n = len(features)
    if count not in KS or n < 1:
        raise ValueError('Se requieren cortes y una cantidad 1, 3 o 10')
    if n <= count:
        return list(range(n))
    distances = ((features[:, None]-features[None, :])**2).mean(axis=2)
    chosen = [int(np.argmin(distances.sum(axis=1)))]
    minimum_rank_gap = max(1, n//count) if count > 1 else 1
    while len(chosen) < count:
        candidates = [i for i in range(n) if i not in chosen and
                      min(abs(i-j) for j in chosen) >= minimum_rank_gap]
        if not candidates:
            minimum_rank_gap -= 1
            continue
        chosen.append(max(candidates, key=lambda i: (float(distances[i, chosen].min()), -i)))
    return sorted(chosen)


def mean_pair_correlation(features: np.ndarray, indices: list[int]) -> float | None:
    if len(indices) < 2:
        return None  # Una sola imagen no tiene correlación por pares.
    values = []
    for i, j in itertools.combinations(indices, 2):
        a, b = features[i]-features[i].mean(), features[j]-features[j].mean()
        denom = np.linalg.norm(a)*np.linalg.norm(b)
        if denom > 1e-12:
            values.append(float(a@b/denom))
    return float(np.mean(values)) if values else None


def prepare(root: Path, output: Path) -> dict:
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train = base.cargar_train(root, include_clinical=False)
    records, redundancy = [], []
    for number, (pid, group) in enumerate(train.groupby('patient_id', sort=True), 1):
        group = group.sort_values(['slice_index', 'sample_id']).reset_index(drop=True)
        features = descriptors([[root/getattr(row, f'path_{phase}') for phase in ('pre','early','late')]
                                for row in group.itertuples()])
        for k in KS:
            indices = select_indices(features, k)
            for i in indices:
                records.append({'patient_id':pid, 'sample_id':group.iloc[i].sample_id,
                                'slice_index':int(group.iloc[i].slice_index), 'k':k})
            redundancy.append({'patient_id':pid, 'k':k, 'n_selected':len(indices),
                               'mean_pair_correlation':mean_pair_correlation(features, indices)})
        if number % 100 == 0:
            print(f'Selección: {number}/{train.patient_id.nunique()} pacientes', flush=True)
    pd.DataFrame(records).to_csv(output/'seleccion.csv', index=False)
    red = pd.DataFrame(redundancy)
    red.to_csv(output/'redundancia.csv', index=False)
    report = {'recipe':RECIPE, 'test_images_loaded':0, 'patients':int(train.patient_id.nunique()),
              'selection_sha256':base.sha256(output/'seleccion.csv'),
              'data_signature':hashlib.sha256(train.to_csv(index=False).encode()).hexdigest(),
              'counts':{str(k):int(sum(row['k']==k for row in records)) for k in KS},
              'median_patient_pair_correlation':{str(k):(None if k==1 else float(red[red.k.eq(k)].mean_pair_correlation.median())) for k in KS},
              'warning':'Descriptores de píxeles; no representan área tumoral ni prueban información predictiva.'}
    base.json_write(output/'preparacion.json', report)
    return report


def load_selection(root: Path, output: Path) -> tuple[pd.DataFrame, dict]:
    report = json.loads((output/'preparacion.json').read_text())
    train = base.cargar_train(root, include_clinical=False)
    if (report['recipe'] != RECIPE or report['selection_sha256'] != base.sha256(output/'seleccion.csv') or
            report['data_signature'] != hashlib.sha256(train.to_csv(index=False).encode()).hexdigest()):
        raise ValueError('Selección o metadatos modificados: usa una salida nueva y prepara de nuevo')
    selected = pd.read_csv(output/'seleccion.csv')
    for k in KS:
        frame = selected[selected.k.eq(k)]
        counts = frame.groupby('patient_id').size()
        if (set(counts.index) != set(train.patient_id) or frame.sample_id.duplicated().any() or
                not set(frame.sample_id).issubset(set(train.sample_id)) or
                (counts > k).any()):
            raise ValueError('Manifiesto incompleto o inválido')
    return selected, report


def configuration(k: int, manifest_hash: str) -> dict:
    cfg = anti.configuracion('antioverfit')
    cfg['name'] = f'cnn_seleccion_cortes_{k}'
    cfg['training']['loss_weight_unit'] = 'patient'
    cfg['slice_selection'] = {'recipe':RECIPE, 'fit_only':True, 'k':k,
                              'manifest_sha256':manifest_hash,
                              'selection_and_outer':'all_available_slices'}
    cfg['history_plot_title'] = f'{k} cortes por paciente en fit; evaluación con todos'
    return cfg


def run(root: Path, output: Path, device: torch.device, workers: int, smoke=False,
        ks=KS, seeds=(42,2026), folds=(0,3), cpu_pilot=False):
    selected, report = load_selection(root, output)
    patient_ids = None
    if cpu_pilot:
        patients = base.cargar_train(root, include_clinical=False).drop_duplicates('patient_id')
        chosen = []
        for (fold, cohort, label), group in patients.groupby(['fold','dataset','pCR']):
            chosen.extend(group.sort_values('patient_id').sample(
                n=min(8 if fold==0 else 2,len(group)),random_state=20261021).patient_id.tolist())
        patient_ids = sorted(chosen)
        base.json_write(output/'cpu_patient_ids.json',patient_ids)
    summaries = []
    for k in ks:
        allowed = set(selected[selected.k.eq(k)].sample_id)
        cfg = configuration(k, report['selection_sha256'])
        cfg['training']['num_workers'] = workers
        if cpu_pilot:
            cfg['name'] += '_cpu'
            cfg['model'].update(backbone_channels=[8,16,24,32],hidden=16)
            cfg['training'].update(epochs=6,min_epochs=6,patience=6,num_workers=0,cpu_threads=4)
            cfg['evaluation']['inner_selection_fraction']=.375
            cfg['development_patient_ids']=patient_ids
            cfg['cpu_pilot_warning']='Small stratified development subset, one seed, one outer fold, lighter CNN; exploratory only'
        if smoke:
            cfg['training'].update(epochs=1, min_epochs=1, patience=1, num_workers=0)
        for seed in seeds:
            for fold in folds:
                summaries.append(imagen.run_training(
                    cfg, 'weighted', seed, fold, root, output/'pruebas' if smoke else output,
                    device, smoke=smoke, model_type=anti.CNNTemporalSimple,
                    experiment_name=cfg['name'],
                    extra_source_files={'seleccion_cortes.py':Path(__file__),
                                        'entrenamiento_antioverfit.py':Path(anti.__file__)},
                    fit_samples_transform=lambda frame: frame[frame.sample_id.isin(allowed)].copy()))
    base.json_write(output/('smoke_summary.json' if smoke else 'cpu_training_summary.json' if cpu_pilot else 'training_summary.json'), summaries)


def compare(output: Path, root: Path = DATA_DIR, cpu_pilot=False) -> dict:
    selected, preparation = load_selection(root, output)
    results, histories = {}, []
    for k in KS:
        fold_frames, gaps, epoch_counts = [], [], []
        for fold in ((0,) if cpu_pilot else (0,3)):
            frames = []
            for seed in ((42,) if cpu_pilot else (42,2026)):
                run_dir=output/(f'cnn_seleccion_cortes_{k}'+('_cpu' if cpu_pilot else ''))/'weighted'/f'seed_{seed}'/f'fold_{fold}'
                summary=json.loads((run_dir/'summary.json').read_text())
                cfg=json.loads((run_dir/'config.json').read_text())
                if (summary['status']!='complete' or summary['smoke'] or not summary['outer_evaluated'] or
                        cfg['slice_selection']['manifest_sha256'] != preparation['selection_sha256'] or
                        summary['config_hash'] != base.config_hash(cfg) or
                        summary['outer_sha256'] != base.sha256(run_dir/'outer_patients.csv')):
                    raise ValueError(f'Run incompleto o modificado: {run_dir}')
                frames.append(pd.read_csv(run_dir/'outer_patients.csv'))
                h=pd.read_csv(run_dir/'history.csv'); h['k']=k;h['seed']=seed;h['fold']=fold
                gaps.append(float(h.loc[h.epoch.eq(summary['best_epoch']),'auc_gap'].iloc[0]))
                epoch_counts.append(int(summary['epochs_completed']));histories.append(h)
            fold_frames.append(imagen._aligned_average(frames))
        oof=pd.concat(fold_frames,ignore_index=True)
        oof.to_csv(output/f'oof_{k}{"_cpu" if cpu_pilot else ""}.csv',index=False)
        aucs=[float(roc_auc_score(f.label,f.probability_raw)) for f in fold_frames]
        results[str(k)]={'fold_auc':aucs,'mean_fold_auc':float(np.mean(aucs)),
                         'selected_gap_median':float(np.median(gaps)),
                         'epochs_completed':epoch_counts,
                         'metrics':imagen.metricas_binarias(oof.label,oof.probability_raw)}
    for k in (3,1):
        delta=np.array(results[str(k)]['fold_auc'])-results['10']['fold_auc']
        results[str(k)]['delta_mean_auc_vs_10']=float(delta.mean())
        results[str(k)]['exploratory_gate']=bool(delta.mean()>=.01 and delta.min()>=-.02 and
            results[str(k)]['selected_gap_median']<=results['10']['selected_gap_median'])
    suffix='_cpu' if cpu_pilot else ''
    pd.concat(histories,ignore_index=True).to_csv(output/f'historias{suffix}.csv',index=False)
    report={'results':results,'test_images_loaded':0,'warning':('Prueba CPU: muestra pequeña, una semilla y un fold exterior, red ligera. No es confirmatoria.' if cpu_pilot else 'Piloto exploratorio de dos folds; no demuestra generalización independiente.')}
    if cpu_pilot:
        for k in (3,1):results[str(k)].pop('exploratory_gate',None)
    base.json_write(output/f'comparacion{suffix}.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for k in KS:
        f=pd.read_csv(output/f'oof_{k}{suffix}.csv')
        RocCurveDisplay.from_predictions(f.label,f.probability_raw,name=f'{k} cortes fit',ax=axes[0])
        axes[1].plot([0] if cpu_pilot else [0,3],results[str(k)]['fold_auc'],'o-',label=f'{k} cortes fit')
    axes[1].set(xlabel='Fold exterior',ylabel='AUC',xticks=[0] if cpu_pilot else [0,3],ylim=(0,1));axes[1].legend()
    fig.tight_layout();fig.savefig(output/f'roc_folds{suffix}.png',dpi=160);plt.close(fig)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('accion',choices=['preparar','entrenar','comparar','cpu'])
    parser.add_argument('--datos',type=Path,default=DATA_DIR)
    parser.add_argument('--salida',type=Path,default=OUTPUT)
    parser.add_argument('--dispositivo',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--prueba',action='store_true')
    args=parser.parse_args()
    if args.workers<0:parser.error('workers debe ser no negativo')
    if args.accion=='preparar':print(json.dumps(prepare(args.datos,args.salida),indent=2));return
    if args.accion=='comparar':print(json.dumps(compare(args.salida,args.datos),indent=2));return
    if args.accion=='cpu':
        run(args.datos,args.salida,torch.device('cpu'),0,seeds=(42,),folds=(0,),cpu_pilot=True)
        print(json.dumps(compare(args.salida,args.datos,cpu_pilot=True),indent=2));return
    if args.dispositivo=='cuda' and not torch.cuda.is_available():parser.error('CUDA no está disponible')
    run(args.datos,args.salida,torch.device(args.dispositivo),args.workers,args.prueba,
        seeds=(42,) if args.prueba else (42,2026),folds=(0,) if args.prueba else (0,3))


if __name__=='__main__':main()
