"""Estabilidad temporal de pesos: evita seleccionar oscilaciones del optimizador."""
from pathlib import Path
from cancer_mama import imagen_mil as m


def main():
    cache=m.load_cache(feature_file='features_texture_v3.npy')
    common=dict(feature_file='features_texture_v3.npy',epochs=160,patience=28,min_epochs=35,
                feature_std_min=.001,selection_metric='inner_auc',max_auc_gap=.10,
                ema_decay=.995,dropout=.55,lr=3e-4,weight_decay=.03,slice_pool='attention_mean',
                view='roi',prior_strength=6.,mixup=.2)
    plans=[
        ('linear_stable_v4',m.configuration('features',**(common|dict(feature_head='linear',feature_l2=2.,batch_size=64,lr=1e-4,ema_decay=.99,epochs=240,patience=40)))),
        ('hybrid_linear_stable_v4',m.configuration('hybrid',**(common|dict(feature_head='linear',feature_l2=2.)))),
        ('hybrid_mlp_stable_v4',m.configuration('hybrid',**(common|dict(feature_hidden=8,feature_std_min=.02,lr=3e-4,mixup=.4)))),
    ]
    summaries=[]
    for name,cfg in plans:
        for fold in [0,3]:
            s=m.train(cache,cfg,name,fold,42)
            summaries.append(s)
            m.write_json(m.OUTPUT/'pilot_stable_summaries.json',summaries)
            with Path('docs/mejoras_cnn/EXPERIMENTOS_IMAGEN_20261025.md').open('a') as f:
                f.write(f'\n### Piloto {name}, contexto {fold}\n\nPromedio exponencial de pesos; '
                    f'época interna {s["best_epoch"]}/{s["epochs"]}; '
                    f'AUC fit {s["splits"]["fit"]["auc"]:.4f}, selección '
                    f'{s["splits"]["selection"]["auc"]:.4f}; BCE selección '
                    f'{s["splits"]["selection"]["bce"]:.4f}; {s["seconds"]/60:.2f} min.\n')

if __name__=='__main__':
    main()
