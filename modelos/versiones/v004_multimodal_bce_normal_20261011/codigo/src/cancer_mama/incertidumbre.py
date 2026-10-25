"""Intervalos bootstrap pareados por paciente, cohorte, fold y clase."""
import numpy as np
from . import entrenamiento as base


def intervals(reference,candidate,repetitions=2000,alpha=.05/6):
    keys = ["patient_id","label","fold","cohort"]
    if not reference[keys].equals(candidate[keys]):
        raise ValueError("Pacientes distintas")
    rng = np.random.default_rng(20261010)
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
