"""CNN 2D compartida y agregación axial Conv1d, sin redes preentrenadas."""
from pathlib import Path
from cancer_mama import imagen_mil as m


def main():
    full=m.load_cache()
    texture=m.load_cache(feature_file='features_texture_v3.npy')
    common=dict(epochs=110,patience=22,min_epochs=25,lr=6e-4,weight_decay=.01,
                dropout=.45,slice_pool='sequence',train_slices=10,batch_size=16,
                mixup=.3,selection_metric='inner_auc',max_auc_gap=.12,ema_decay=.99,
                widths=[12,24,48,64],norm='group')
    plans=[
        ('cnn_axial_v5',m.configuration('cnn',**common)),
        ('cnn_axial_weighted_v5',m.configuration('cnn',**(common|dict(weighted=True)))),
        ('hybrid_axial_v5',m.configuration('hybrid',**(common|dict(feature_file='features_texture_v3.npy',feature_hidden=8,dropout=.6,lr=3e-4,mixup=.4)))),
    ]
    summaries=[]
    for name,cfg in plans:
        cache=texture if cfg.get('feature_file') else full
        for fold in [0,3]:
            s=m.train(cache,cfg,name,fold,42)
            summaries.append(s)
            m.write_json(m.OUTPUT/'pilot_axial_summaries.json',summaries)
            with Path('docs/mejoras_cnn/EXPERIMENTOS_IMAGEN_20261025.md').open('a') as f:
                f.write(f'\n### Piloto {name}, contexto {fold}\n\nCNN 2D y Conv1d axial, '
                    f'época interna {s["best_epoch"]}/{s["epochs"]}; '
                    f'AUC fit {s["splits"]["fit"]["auc"]:.4f}, selección '
                    f'{s["splits"]["selection"]["auc"]:.4f}; BCE selección '
                    f'{s["splits"]["selection"]["bce"]:.4f}; {s["seconds"]/60:.2f} min.\n')

if __name__=='__main__':
    main()
