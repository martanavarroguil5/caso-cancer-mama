"""Tercera tanda: reducir capacidad de la rama estadística y textura fija."""
from pathlib import Path
from cancer_mama import imagen_mil as m
from cancer_mama import imagen_textura as t


def main():
    t.prepare()
    full=m.load_cache()
    roi=m.load_cache(roi_features=True)
    texture=m.load_cache(feature_file='features_texture_v3.npy')
    common=dict(lr=1e-3,weight_decay=.02,epochs=140,patience=24,dropout=.5,
                slice_pool='attention_mean',feature_head='linear',feature_l2=5.,
                feature_std_min=.001,selection_metric='inner_auc',max_auc_gap=.10,
                mixup=.2,label_smoothing=.02)
    plans=[
        ('linear_roi_v3',m.configuration('features',**(common|dict(roi_features=True,epochs=180,batch_size=64,lr=.003)))),
        ('hybrid_linear_v3',m.configuration('hybrid',**common)),
        ('hybrid_roi_linear_v3',m.configuration('hybrid',**(common|dict(view='roi',prior_strength=6.,roi_features=True)))),
        ('hybrid_texture_v3',m.configuration('hybrid',**(common|dict(view='roi',prior_strength=6.,feature_file='features_texture_v3.npy',feature_l2=10.)))),
        ('linear_texture_v3',m.configuration('features',**(common|dict(feature_file='features_texture_v3.npy',feature_l2=10.,epochs=180,batch_size=64,lr=.003)))),
        ('cnn_focus_v3',m.configuration('cnn',lr=1e-3,weight_decay=.01,epochs=120,patience=24,dropout=.5,
             slice_pool='attention_mean',view='roi',prior_strength=6.,widths=[16,32,64,96],norm='batch',
             selection_metric='inner_auc',max_auc_gap=.10,mixup=.4)),
    ]
    summaries=[]
    for name,cfg in plans:
        data=texture if cfg.get('feature_file') else roi if cfg.get('roi_features') else full
        s=m.train(data,cfg,name)
        summaries.append(s)
        m.write_json(m.OUTPUT/'pilot_regularized_summaries.json',summaries)
        with Path('docs/mejoras_cnn/EXPERIMENTOS_IMAGEN_20261025.md').open('a') as f:
            f.write(f'\n### Piloto {name}\n\nSemilla 42, fold exterior 0 sin evaluar. '
                    f'Época por AUC interna con brecha máxima 0,10: {s["best_epoch"]}/{s["epochs"]}; '
                    f'AUC fit {s["splits"]["fit"]["auc"]:.4f}, selección '
                    f'{s["splits"]["selection"]["auc"]:.4f}; BCE selección '
                    f'{s["splits"]["selection"]["bce"]:.4f}; '
                    f'{s["seconds"]/60:.2f} min, {s["parameters"]} parámetros.\n')

if __name__=='__main__':
    main()
