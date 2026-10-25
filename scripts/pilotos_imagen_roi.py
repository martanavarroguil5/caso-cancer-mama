"""Segunda tanda exploratoria de imagen; solo selección interna, sin outer/test."""
from pathlib import Path
from cancer_mama import imagen_mil as m


def main():
    m.prepare_roi_features()
    full=m.load_cache()
    roi=m.load_cache(roi_features=True)
    plans=[
        ('hybrid_fast_v2',m.configuration('hybrid',lr=1e-3,weight_decay=.02,epochs=100,patience=18,dropout=.45,slice_pool='attention_mean')),
        ('features_roi_v2',m.configuration('features',lr=1e-3,weight_decay=.1,epochs=120,patience=18,batch_size=64,roi_features=True)),
        ('hybrid_roi_v2',m.configuration('hybrid',lr=1e-3,weight_decay=.02,epochs=100,patience=18,dropout=.45,slice_pool='attention_mean',view='roi',roi_features=True)),
        ('hybrid_roi_bn_v2',m.configuration('hybrid',lr=8e-4,weight_decay=.01,epochs=100,patience=18,dropout=.5,widths=[16,32,64,96],norm='batch',slice_pool='attention_mean',view='roi',roi_features=True)),
        ('cnn_roi_bn_v2',m.configuration('cnn',lr=1e-3,weight_decay=.01,epochs=100,patience=18,dropout=.4,widths=[16,32,64,96],norm='batch',slice_pool='attention_mean',view='roi')),
        ('hybrid_rank_v2',m.configuration('hybrid',lr=8e-4,weight_decay=.03,epochs=90,patience=18,dropout=.5,slice_pool='attention_mean',view='roi',roi_features=True,rank_weight=.5,mixup=0.)),
    ]
    summaries=[]
    for name,cfg in plans:
        s=m.train(roi if cfg.get('roi_features') else full,cfg,name)
        summaries.append(s)
        m.write_json(m.OUTPUT/'pilot_roi_summaries.json',summaries)
        with Path('docs/mejoras_cnn/EXPERIMENTOS_IMAGEN_20261025.md').open('a') as f:
            f.write(f'\n### Piloto {name}\n\nSemilla 42, fold exterior 0 sin evaluar. '
                    f'Época por pérdida interna: {s["best_epoch"]}/{s["epochs"]}; '
                    f'AUC fit {s["splits"]["fit"]["auc"]:.4f}, selección '
                    f'{s["splits"]["selection"]["auc"]:.4f}; BCE selección '
                    f'{s["splits"]["selection"]["bce"]:.4f}; '
                    f'{s["seconds"]/60:.2f} min, {s["parameters"]} parámetros.\n')

if __name__=='__main__':
    main()
