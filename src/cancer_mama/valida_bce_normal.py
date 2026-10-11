"""Auditoría de particiones, inferencia guardada y formato de un corte (solo train)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from cancer_mama import entrenamiento as base, experimento_bce_mixup as ex
from cancer_mama import experimento_loss as old, informe_loss as diagnosis

def main():
    torch.set_num_threads(2)
    output = base.RESULTS_DIR/'10_informe_bce_mixup_20261011'
    samples = base.cargar_train(base.pdatos.DATOS, include_clinical=True)
    errors, hashes = [], {}
    for seed in ex.SEEDS:
        for fold in range(5):
            folder = ex.DEFAULT_OUTPUT/f'bce_normal/seed_{seed}/fold_{fold}'
            config = json.loads((folder/'config.json').read_text())
            control_config = json.loads((old.DEFAULT_OUTPUT/f'lr_actual/seed_{seed}/fold_{fold}/config.json').read_text())
            assert config['partitions'] == control_config['partition_signatures']
            fit, selection, outer = old.split_patients(samples, fold)
            assert base.fit_clinical_preprocessing(fit) == config['model']['clinical']['preprocessing']
            for name, frame in (('fit', fit), ('selection', selection), ('outer', outer)):
                assert old.signature(frame) == config['partitions'][name]
            row = outer.sort_values('sample_id').iloc[(seed+fold) % len(outer)]
            images = []
            for column in base.pdatos.COLUMNAS_RUTA:
                path = base.pdatos.DATOS/row[column]
                assert '/train/' in str(path)
                with Image.open(path) as image:
                    images.append(np.array(image, dtype=np.float32)/255.)
            x = torch.from_numpy(np.stack(images)).unsqueeze(0)
            clinical = torch.from_numpy(base.clinical_matrix(row.to_frame().T).astype(np.float32))
            payload = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
            hashes[str(folder/'best.pt')] = base.sha256(folder/'best.pt')
            models = [cls(payload['model_config']).eval() for cls in (base.CNN, ex.MixupCNN)]
            for model in models:
                model.load_state_dict(payload['state_dict'], strict=True)
            with torch.inference_mode():
                logits = [model(x, clinical) for model in models]
                torch.testing.assert_close(logits[0], logits[1], rtol=0, atol=0)
                probability = float(logits[0].sigmoid().item())
            recorded = pd.read_csv(folder/'oof.csv').set_index('sample_id').loc[row.sample_id, 'probability']
            difference = abs(probability-recorded)
            if difference > 5e-5:
                raise ValueError('La inferencia CPU no reproduce la OOF guardada')
            errors.append({'seed': seed, 'fold': fold, 'sample_id': row.sample_id, 'absolute_error': difference})

    keys = ['sample_id', 'patient_id', 'pCR', 'fold', 'dataset']
    def ensemble(source, relative, filename):
        frames = [pd.concat([pd.read_csv(source/f'{relative}/seed_{seed}/fold_{fold}/{filename}')
                             for fold in range(5)], ignore_index=True).sort_values('sample_id').reset_index(drop=True)
                  for seed in ex.SEEDS]
        assert frames[0][keys].equals(frames[1][keys])
        result = frames[0].copy()
        result.probability = (frames[0].probability+frames[1].probability)/2
        return result

    normal = ensemble(ex.DEFAULT_OUTPUT, 'bce_normal', 'oof.csv')
    weighted = ensemble(old.DEFAULT_OUTPUT, 'lr_actual', 'oof_auc.csv')
    assert normal[keys].equals(weighted[keys])
    calibrators = json.loads((diagnosis.DEFAULT_REPORT/'calibradores_internos.json').read_text())
    for record in calibrators:
        if record['criterion'] == 'auc':
            mask = weighted.fold.eq(record['fold'])
            weighted.loc[mask, 'probability'] = base.apply_calibration(weighted.loc[mask, 'probability'].to_numpy(), record['calibration'])
    positions = []
    for patient_id, group in normal.groupby('patient_id', sort=True):
        digest = hashlib.sha256(('single-cut-audit-20261011:'+patient_id).encode()).digest()
        positions.append(group.index[int.from_bytes(digest[:8], 'big') % len(group)])
    frames = {}
    for name, frame in (('normal', normal), ('weighted_control', weighted)):
        selected = frame.loc[positions].rename(columns={'pCR':'label', 'dataset':'cohort'}).reset_index(drop=True)
        frames[name] = selected
        selected.to_csv(output/f'oof_single_cut_{name}_diagnostico.csv', index=False)
    metrics = {name: diagnosis.metrics(frame) for name, frame in frames.items()}
    result = {'purpose': 'Posthoc technical audit; no model or hyperparameter selection',
        'audit_source_sha256': base.sha256(__file__),
        'test_images_loaded': 0, 'partition_checks': 10, 'checkpoint_checks': 10,
        'CPU_saved_OOF_checks': errors, 'maximum_absolute_error': max(item['absolute_error'] for item in errors),
        'weights_sha256': hashes, 'single_cut_rule': 'SHA256(single-cut-audit-20261011:patient_id) modulo available sorted slices',
        'single_cut_metrics': metrics,
        'single_cut_delta': {key: metrics['normal'][key]-metrics['weighted_control'][key]
                             for key in ('log_loss', 'brier', 'roc_auc', 'mean_fold_auc')},
        'warning': 'One deterministic slice per patient, same development patients; not external validation or proof for arbitrary slices.'}
    base.json_write(output/'auditoria_reproducibilidad_y_un_corte.json', result)
    print(json.dumps({key: result[key] for key in ('partition_checks','checkpoint_checks','maximum_absolute_error','single_cut_metrics','single_cut_delta')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
