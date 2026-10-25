"""Confirmación interna de tres candidatos antes de consultar folds exteriores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss
from cancer_mama import imagen_mil as m

FAMILIES=['hybrid_v1','hybrid_rank_v2','linear_texture_v3']


def main():
    data={}
    for name in FAMILIES:
        cfg=json.loads((m.OUTPUT/'runs'/name/'fold_0/seed_42/config.json').read_text())
        key=cfg.get('feature_file') or ('features_roi_v2.npy' if cfg.get('roi_features') else 'features.npy')
        if key not in data:
            data[key]=m.load_cache(feature_file=key)
        for fold,seed in [(0,2026),(0,3407),(3,42),(3,2026),(3,3407)]:
            m.train(data[key],cfg,name,fold,seed)
    records=[];frames={}
    for name in FAMILIES:
        for fold in [0,3]:
            ff=[];gaps=[]
            for seed in [42,2026,3407]:
                run=m.OUTPUT/'runs'/name/f'fold_{fold}'/f'seed_{seed}'
                f=pd.read_csv(run/'selection_predictions.csv').sort_values('patient_id').reset_index(drop=True)
                ff.append(f)
                s=json.loads((run/'summary.json').read_text())
                gaps.append(s['splits']['fit']['auc']-s['splits']['selection']['auc'])
            assert all(f.patient_id.equals(ff[0].patient_id) for f in ff)
            f=ff[0].copy();f['probability']=np.mean([f.probability for f in ff],axis=0)
            frames[name,fold]=f
            f.to_csv(m.OUTPUT/f'internal_{name}_fold{fold}.csv',index=False)
            records.append(dict(candidate=name,fold=fold,auc=roc_auc_score(f.pCR,f.probability),bce=log_loss(f.pCR,f.probability),median_gap=np.median(gaps)))
    for fold in [0,3]:
        a=frames['hybrid_rank_v2',fold].copy();b=frames['linear_texture_v3',fold]
        assert a.patient_id.equals(b.patient_id)
        a['probability']=.5*a.probability+.5*b.probability
        a.to_csv(m.OUTPUT/f'internal_ensemble_rank_texture_fold{fold}.csv',index=False)
        records.append(dict(candidate='ensemble_rank_texture',fold=fold,auc=roc_auc_score(a.pCR,a.probability),bce=log_loss(a.pCR,a.probability),median_gap=None))
    pd.DataFrame(records).to_csv(m.OUTPUT/'internal_confirmation.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False))
    scores=pd.DataFrame(records).groupby('candidate').agg(auc=('auc','mean'),bce=('bce','mean')).sort_values('auc',ascending=False)
    scores.to_csv(m.OUTPUT/'internal_ranking.csv')
    print(scores)

if __name__=='__main__':
    main()
