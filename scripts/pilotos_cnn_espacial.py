"""CNN propia con mapa espacial 4x4 y dos vistas nativas, piloto sin outer."""
from pathlib import Path
import os,time
from cancer_mama import imagen_mil as m
from cancer_mama import imagen_alta_resolucion as native
from cancer_mama.paths import RESULTS_DIR

OUTPUT=RESULTS_DIR/'22_cnn_espacial_nativa_20261025'


def main():
    while not (native.OUTPUT/'cache/native_images_ready.json').exists():time.sleep(2)
    cache_dir=OUTPUT/'cache';cache_dir.mkdir(parents=True,exist_ok=True)
    for name in ['images.npy','images_native.npy','mask.npy','patients.csv','features.npy','ready.json','native_images_ready.json']:
        if not (cache_dir/name).exists():os.link(native.OUTPUT/'cache'/name,cache_dir/name)
    full=m.load_cache(OUTPUT);natives=m.load_cache(OUTPUT,image_file='images_native.npy')
    common=dict(widths=[16,32,64],spatial_pool='grid4',norm='batch',dropout=.25,
        lr=1e-3,weight_decay=.001,epochs=100,min_epochs=35,patience=25,
        batch_size=24,train_slices=4,mixup=0.,label_smoothing=.02,
        selection_metric='inner_bce_loss',max_auc_gap=.15,ema_decay=.995)
    plans=[('spatial128_normal',m.configuration('cnn',**common)),
           ('spatial256_normal',m.configuration('cnn',**(common|dict(view='multires_native',image_file='images_native.npy')))),
           ('spatial256_weighted',m.configuration('cnn',**(common|dict(view='multires_native',image_file='images_native.npy',weighted=True))))]
    summaries=[]
    for name,cfg in plans:
        for fold in [0,3]:
            s=m.train(natives if cfg.get('image_file') else full,cfg,name,fold,42,OUTPUT)
            summaries.append(s);m.write_json(OUTPUT/'pilot_summaries.json',summaries)
            with Path('docs/mejoras_cnn/EXPERIMENTOS_IMAGEN_20261025.md').open('a') as f:
                f.write(f'\n### CNN espacial {name}, contexto {fold}\n\nÉpoca {s["best_epoch"]}/{s["epochs"]}, '
                f'AUC fit {s["splits"]["fit"]["auc"]:.4f}, selección {s["splits"]["selection"]["auc"]:.4f}; '
                f'BCE {s["splits"]["selection"]["bce"]:.4f}. Solo selección interna.\n')

if __name__=='__main__':main()
