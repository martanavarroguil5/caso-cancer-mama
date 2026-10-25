"""Pilotos secuenciales; ningún fold exterior interviene en la selección."""
from pathlib import Path
from cancer_mama import imagen_mil as m


def main():
    m.prepare()
    cache=m.load_cache()
    plans=[
        ('features_v1',m.configuration('features',lr=1e-3,weight_decay=.1,epochs=80,patience=14,batch_size=64)),
        ('features_linearish',m.configuration('features',lr=3e-4,weight_decay=.3,epochs=80,patience=14,batch_size=64,dropout=.65,mixup=.4)),
        ('cnn_mil_v1',m.configuration('cnn',epochs=40)),
        ('hybrid_v1',m.configuration('hybrid',epochs=45)),
        ('hybrid_weighted_v1',m.configuration('hybrid',weighted=True,epochs=45)),
    ]
    summaries=[]
    for name,cfg in plans:
        summaries.append(m.train(cache,cfg,name))
        m.write_json(m.OUTPUT/'pilot_summaries.json',summaries)
        with (Path('docs/mejoras_cnn/EXPERIMENTOS_IMAGEN_20261025.md')).open('a') as f:
            s=summaries[-1]
            f.write(f'\n### Piloto {name}\n\nSemilla 42, fold exterior 0 sin evaluar. '
                    f'Época por pérdida interna: {s["best_epoch"]}/{s["epochs"]}; '
                    f'AUC fit {s["splits"]["fit"]["auc"]:.4f}, selección '
                    f'{s["splits"]["selection"]["auc"]:.4f}; BCE selección '
                    f'{s["splits"]["selection"]["bce"]:.4f}; '
                    f'{s["seconds"]/60:.2f} min, {s["parameters"]} parámetros.\n')

if __name__=='__main__':
    main()
