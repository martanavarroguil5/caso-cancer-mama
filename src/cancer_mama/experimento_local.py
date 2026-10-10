"""Segunda tanda: características locales congeladas y selección supervisada."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
import torch

from . import entrenamiento as base
from . import experimento_transfer as transfer
from . import experimento_loss as previous
from . import informe_loss as diagnosis
from .paths import PROJECT_ROOT, RESULTS_DIR

DEFAULT_OUTPUT = RESULTS_DIR / "04_entrenamiento/local_multimodal_20261010"
FAMILIES = ("alexnet_svm", "resnet_svm", "alexnet_lineal")


def grid(family, smoke=False):
    if family not in FAMILIES:
        raise ValueError("Familia desconocida")
    result = []
    for k in (32, 128):
        for c in (.01, .1, 1., 10.):
            for gamma in ((.1, 1.) if family.endswith("svm") else (None,)):
                result.append({"family": family, "k": k, "C": c, "gamma": gamma})
    return result[:1] if smoke else result


def fit_preprocessing(features, clinical_frame, labels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        scores, _ = f_classif(features, labels)
    scores = np.nan_to_num(scores, nan=0., posinf=np.finfo(float).max, neginf=0.)
    order = np.argsort(-scores, kind="stable")[:128]
    return {"indices": order, "scores": scores[order],
        "visual_scaler": StandardScaler().fit(features[:, order]),
        "clinical": base.fit_clinical_preprocessing(clinical_frame)}


def transform(features, clinical_frame, state, k):
    selected = state["indices"]
    z = state["visual_scaler"].transform(features[:, selected])[:, :k]
    z = z / np.sqrt(z.shape[1])
    clinical = base.clinical_matrix(clinical_frame)
    missing = np.isnan(clinical)
    prep = state["clinical"]
    c = (np.where(missing, prep["medians"], clinical)-prep["means"])/prep["stds"]
    c = np.column_stack((c, missing.astype(float)))
    return np.column_stack((z, c))


def probability(model, x, family):
    if family.endswith("svm"):
        return expit(model.decision_function(x))
    return model.predict_proba(x)[:, 1]


def fit_selected(family, features, patients, fit_idx, select_idx, smoke=False):
    # Esta función no recibe etiquetas exteriores separadas ni las consulta.
    fit_labels = patients.iloc[fit_idx].pCR.to_numpy(dtype=int)
    sel_labels = patients.iloc[select_idx].pCR.to_numpy(dtype=int)
    prep = fit_preprocessing(features[fit_idx], patients.iloc[fit_idx], fit_labels)
    records, best, best_key = [], None, None
    for config in grid(family, smoke):
        xf = transform(features[fit_idx], patients.iloc[fit_idx], prep, config["k"])
        xs = transform(features[select_idx], patients.iloc[select_idx], prep, config["k"])
        if family.endswith("svm"):
            model = SVC(C=config["C"], gamma=config["gamma"], class_weight="balanced",
                        probability=False, cache_size=512, random_state=transfer.SEED)
        else:
            model = LogisticRegression(C=config["C"], solver="lbfgs", max_iter=4000,
                                       random_state=transfer.SEED)
        model.fit(xf, fit_labels)
        p = probability(model, xs, family)
        loss = float(base.log_loss(sel_labels, p, labels=[0,1]))
        auc = float(base.roc_auc_score(sel_labels, p))
        key = (-auc, loss) if family.endswith("svm") else (loss,)
        records.append(dict(config, selection_loss=loss, selection_auc=auc))
        if best_key is None or key < best_key:
            best_key = key
            best = {"config": config, "model": model, "preprocessing": prep}
    return best, records


@torch.inference_mode()
def extract_local(samples, root, output, device, smoke, progress):
    from torchvision.models import alexnet, AlexNet_Weights
    dest, meta_path = output / "features.npz", output / "features.json"
    weight_path = output / "alexnet_pool2.pth"
    signature = previous.signature(samples)
    if dest.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta["sample_signature"] != signature or meta["smoke"] != smoke
                or meta["sha256"] != base.sha256(dest) or meta["weights_sha256"] != base.sha256(weight_path)):
            raise ValueError("Extracción local modificada o incompatible")
        with np.load(dest) as data:
            return data["patient_id"].tolist(), data["features"].copy()
    weights = AlexNet_Weights.IMAGENET1K_V1
    model = alexnet(weights=weights).features[:6].eval().requires_grad_(False).to(device)
    previous.save_torch(weight_path, {k:v.cpu() for k,v in model.state_dict().items()})
    transform_image = weights.transforms()
    chosen = samples
    if smoke:
        ids = samples.drop_duplicates("patient_id").groupby(["fold", "pCR"]).head(8).patient_id
        chosen = samples[samples.patient_id.isin(ids)].groupby("patient_id", group_keys=False).head(2).reset_index(drop=True)
    ids = sorted(chosen.patient_id.unique())
    indices = {pid:i for i,pid in enumerate(ids)}
    accum, counts = None, np.zeros(len(ids), dtype=int)
    cache = base.CacheTrain(samples, root, previous.DEFAULT_OUTPUT / ".cache", workers=4)
    loader = base.make_loader(chosen, root, 64, 0 if smoke else 4, torch.Generator().manual_seed(transfer.SEED), cache=cache)
    for step, batch in enumerate(loader):
        x, _ = base.batch_device(batch, device)
        z = model(transform_image(x)).flatten(1).cpu().numpy()
        if accum is None:
            accum = np.zeros((len(ids),z.shape[1]), dtype=np.float64)
        for pid, row in zip(batch["patient_id"], z):
            accum[indices[pid]] += row
            counts[indices[pid]] += 1
        if step % 30 == 0 or step+1 == len(loader):
            print(f"AlexNet pool2 {step+1}/{len(loader)}", flush=True)
            progress({"stage":"extracting", "batch":step+1, "batches":len(loader)})
    features = (accum/counts[:,None]).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Características locales no finitas")
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, patient_id=np.asarray(ids,dtype=str),features=features)
    tmp.replace(dest)
    base.json_write(meta_path, {"sample_signature":signature,"smoke":smoke,"sha256":base.sha256(dest),
        "weights_sha256":base.sha256(weight_path),"shape":list(features.shape),"representation":"AlexNet_ImageNet_pool2_spatial_mean_patient"})
    return ids, features


def fit_fold(samples, patients, features, fold, output, smoke=False):
    fit, selection, outer = previous.split_patients(samples, fold)
    sets = [set(x.patient_id) for x in (fit,selection,outer)]
    fi, si, oi = [np.flatnonzero(patients.patient_id.isin(ids)) for ids in sets]
    folder = output / f"fold_{fold}"
    folder.mkdir(exist_ok=True)
    expected = {"fold":fold,"smoke":smoke,"partitions":{role:previous.signature(x)
        for role,x in (("fit",fit),("selection",selection),("outer",outer))}}
    summary_path = folder / "summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        if any(data[k] != v for k,v in expected.items()):
            raise ValueError("Particiones cambiadas")
        for name,sha in data["files"].items():
            if base.sha256(folder/name) != sha:
                raise ValueError("Resultado previo cambiado")
        return data
    summaries = {}
    for family in FAMILIES:
        visual = features["resnet" if family.startswith("resnet") else "alexnet"]
        chosen, records = fit_selected(family, visual, patients, fi, si, smoke)
        xx = transform(visual, patients, chosen["preprocessing"], chosen["config"]["k"])
        probabilities = probability(chosen["model"],xx,family)
        inner = transfer.prediction_frame(patients,si,probabilities[si])
        outer_frame = transfer.prediction_frame(patients,oi,probabilities[oi])
        calibrated, calibration, threshold = diagnosis.calibrate_inner(inner,outer_frame)
        inner.to_csv(folder/f"inner_{family}.csv",index=False)
        outer_frame.to_csv(folder/f"oof_raw_{family}.csv",index=False)
        calibrated.to_csv(folder/f"oof_calibrated_{family}.csv",index=False)
        chosen.update(calibration=calibration,threshold=threshold,fit_ids=sorted(sets[0]),
                      selection_ids=sorted(sets[1]),outer_ids=sorted(sets[2]))
        joblib.dump(chosen,folder/f"model_{family}.joblib",compress=3)
        base.json_write(folder/f"seleccion_{family}.json",records)
        summaries[family] = {"config":chosen["config"],
            "fit":base.binary_metrics(patients.iloc[fi].pCR,probabilities[fi]),
            "selection":base.binary_metrics(patients.iloc[si].pCR,probabilities[si]),
            "outer_raw":diagnosis.metrics(outer_frame),"outer_calibrated":diagnosis.metrics(calibrated)}
        print(f"Fold {fold} {family}: {chosen['config']}",flush=True)
    result = dict(expected,models=summaries,files={p.name:base.sha256(p) for p in folder.iterdir() if p.is_file()})
    base.json_write(summary_path,result)
    return result


def intervals(reference,candidate,repetitions=2000,alpha=.05/6):
    keys = ["patient_id","label","fold","cohort"]
    if not reference[keys].equals(candidate[keys]):
        raise ValueError("Pacientes distintas")
    rng = np.random.default_rng(transfer.SEED)
    y=reference.label.to_numpy(); a=reference.probability.to_numpy(); b=candidate.probability.to_numpy()
    folds=reference.fold.to_numpy()
    strata=list(reference.groupby(["fold","cohort","label"]).indices.values())
    values={k:[] for k in ("log_loss","brier","roc_auc","mean_fold_auc")}
    # Precalcula contribuciones aditivas; AUC sigue siendo pareada por paciente.
    a=np.clip(a,base.EPS,1-base.EPS); b=np.clip(b,base.EPS,1-base.EPS)
    dl=-(y*np.log(b/a)+(1-y)*np.log((1-b)/(1-a)))
    db=(y-b)**2-(y-a)**2
    for _ in range(repetitions):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in strata])
        yy,aa,bb,ff=y[idx],a[idx],b[idx],folds[idx]
        values['log_loss'].append(float(dl[idx].mean()))
        values['brier'].append(float(db[idx].mean()))
        values['roc_auc'].append(base.roc_auc_score(yy,bb)-base.roc_auc_score(yy,aa))
        values['mean_fold_auc'].append(float(np.mean([base.roc_auc_score(yy[ff==f],bb[ff==f])-base.roc_auc_score(yy[ff==f],aa[ff==f]) for f in sorted(set(ff))])))
    return {k:{"lower":float(np.quantile(v,alpha/2)),"upper":float(np.quantile(v,1-alpha/2))} for k,v in values.items()}


def compare(samples,output):
    reference=pd.read_csv(transfer.REFERENCE/'oof_calibrated_inner_auc.csv')
    roster=samples.drop_duplicates('patient_id').sort_values('patient_id').rename(columns={'pCR':'label','dataset':'cohort'}).reset_index(drop=True)
    keys=['patient_id','label','fold','cohort']
    if not reference[keys].equals(roster[keys]): raise ValueError('Control desalineado')
    dest=output/'comparacion'; dest.mkdir(exist_ok=True)
    metrics=[{'candidate':'control_auc_calibrated',**diagnosis.metrics(reference)}]; comparisons={}
    for family in FAMILIES:
        for scale in ('raw','calibrated'):
            pieces=[]
            for fold in range(5):
                folder=output/f'fold_{fold}'
                summary=json.loads((folder/'summary.json').read_text())
                path=folder/f'oof_{scale}_{family}.csv'
                if summary['smoke'] or base.sha256(path)!=summary['files'][path.name]: raise ValueError('OOF inválida')
                pieces.append(pd.read_csv(path))
            frame=pd.concat(pieces).sort_values('patient_id').reset_index(drop=True)
            if not frame[keys].equals(reference[keys]): raise ValueError('Cobertura exterior incorrecta')
            frame.to_csv(dest/f'oof_{scale}_{family}.csv',index=False)
            values=diagnosis.metrics(frame); metrics.append({'candidate':f'{family}_{scale}',**values})
            if scale=='calibrated':
                ci=intervals(reference,frame)
                delta={k:values[k]-diagnosis.metrics(reference)[k] for k in ('log_loss','brier','roc_auc','mean_fold_auc')}
                comparisons[family]={'delta_vs_control':delta,'intervals_99_17':ci,
                    'promising_internal_result':ci['log_loss']['upper']<0 and ci['mean_fold_auc']['lower']>=-.01 and delta['brier']<=0}
                print(f"Comparado {family}: {comparisons[family]['promising_internal_result']}",flush=True)
    report={'metrics':metrics,'comparisons':comparisons,'alpha':.05/6,'bootstrap_repetitions':2000,
            'test_images_loaded':0,'automatic_promotion':False,'adaptive_round':2}
    base.json_write(dest/'resumen.json',report)
    pd.DataFrame(metrics).drop(columns='fold_auc').to_csv(dest/'metricas.csv',index=False)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--prueba',action='store_true')
    args=parser.parse_args(); output=args.salida/'pruebas' if args.prueba else args.salida
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (output/'runner.lock').open('a') as lock,threadpool_limits(limits=4):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'status':'running','started_at':previous.now(),'pid':os.getpid(),'smoke':args.prueba}
        def progress(data):
            state.update(data,updated_at=previous.now()); base.json_write(output/'estado.json',state)
        progress({'stage':'verifying'})
        try:
            torch.set_num_threads(8); base.seed_everything(transfer.SEED)
            samples,old=diagnosis.verify_inputs(base.pdatos.DATOS,previous.DEFAULT_OUTPUT)
            protected=previous.protected_snapshot()
            sources={Path(p).name:base.sha256(p) for p in (__file__,transfer.__file__,previous.__file__,diagnosis.__file__,base.__file__)}
            env=base.environment(torch.device('cuda'));env.pop('git_commit',None)
            params={'sources':sources,'protected':protected,'data':previous.signature(samples),
                'images':old['parameters']['train_png_sha256'],'grid':{f:grid(f,args.prueba) for f in FAMILIES},
                'smoke':args.prueba,'environment':env,'reference':base.sha256(transfer.REFERENCE/'oof_calibrated_inner_auc.csv'),
                'resnet_features':base.sha256(transfer.DEFAULT_OUTPUT/'features.npz')}
            fingerprint=base.config_hash(params);protocol=output/'protocolo.json'
            if protocol.exists() and json.loads(protocol.read_text())['fingerprint']!=fingerprint:raise ValueError('Protocolo distinto')
            if not protocol.exists():base.json_write(protocol,{'fingerprint':fingerprint,'parameters':params,'created_at':previous.now()})
            ids,alex=extract_local(samples,base.pdatos.DATOS,output,torch.device('cuda'),args.prueba,progress)
            res_ids,resnet=transfer.extract(samples,base.pdatos.DATOS,transfer.DEFAULT_OUTPUT,torch.device('cuda'),False,progress)
            res_positions={pid:i for i,pid in enumerate(res_ids)}
            resnet=resnet[[res_positions[pid] for pid in ids]]
            patients=samples.drop_duplicates('patient_id').set_index('patient_id').loc[ids].reset_index()
            for fold in ([0] if args.prueba else range(5)):
                progress({'stage':'training','fold':fold})
                fit_fold(samples,patients,{'alexnet':alex,'resnet':resnet},fold,output,args.prueba)
            if not args.prueba:
                progress({'stage':'comparison'});report=compare(samples,output)
                print(json.dumps(report['metrics'],indent=2),flush=True)
            if previous.protected_snapshot()!=protected:raise RuntimeError('Archivos protegidos cambiados')
            progress({'status':'complete','stage':'complete','finished_at':previous.now(),'protected_unchanged':True})
        except BaseException as exc:
            progress({'status':'failed','error':str(exc)});traceback.print_exc();raise


if __name__=='__main__':main()
