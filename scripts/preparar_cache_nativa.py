"""Copia solo PNG de train a una caché uint8 nativa para pilotos rápidos."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np,pandas as pd
from PIL import Image
from cancer_mama import entrenamiento as base
from cancer_mama import imagen_alta_resolucion as native
from cancer_mama import imagen_mil as mil
from cancer_mama.paths import DATA_DIR


def main():
    cache=native.OUTPUT/'cache';destination=cache/'images_native.npy'
    if (cache/'native_images_ready.json').exists():return
    rows=base.cargar_train(DATA_DIR,include_clinical=False)
    patients=pd.read_csv(cache/'patients.csv')
    arr=np.lib.format.open_memmap(destination,mode='w+',dtype='uint8',shape=(len(patients),10,3,256,256));arr[:]=0
    def load(pid):
        group=rows[rows.patient_id.eq(pid)].sort_values('slice_index'); images=[]
        for row in group.itertuples():
            planes=[]
            for col in ['path_pre','path_early','path_late']:
                path=DATA_DIR/getattr(row,col)
                if 'test' in path.parts:raise ValueError('Test prohibido')
                with Image.open(path) as im:planes.append(np.asarray(im))
            images.append(np.stack(planes))
        return np.stack(images)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for start in range(0,len(patients),48):
            for offset,images in enumerate(pool.map(load,patients.patient_id.iloc[start:start+48])):
                arr[start+offset,:len(images)]=images
            print(f'Nativa: {min(start+48,len(patients))}/{len(patients)}',flush=True)
    arr.flush();mil.write_json(cache/'native_images_ready.json',dict(patients=len(patients),test_images_loaded=0,sha256=base.sha256(destination)))

if __name__=='__main__':main()
