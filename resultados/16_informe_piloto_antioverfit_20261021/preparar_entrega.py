"""Recalcula y documenta el piloto sin entrenar ni publicar cambios."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import torch

from cancer_mama import entrenamiento as base
from cancer_mama import entrenamiento_antioverfit as pilot
from cancer_mama import entrenamiento_imagen as image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = pilot.DEFAULT_OUTPUT
COLORS = {'control_normal': '#2166ac', 'antioverfit_normal': '#d6604d',
          'control_weighted': '#1b9e77', 'antioverfit_weighted': '#984ea3'}


def write(path, record):
    base.json_write(path, record)


def main():
    state = json.loads((RUN / 'estado_piloto.json').read_text())
    assert state['status'] == 'complete' and state['verified_runs'] == 16
    report = json.loads((RUN / 'comparacion_piloto.json').read_text())
    assert not (RUN / 'error.json').exists()
    for name in ('lanzamiento.json', 'estado_piloto.json', 'protocolo_ejecutado.json',
                 'comparacion_piloto.json', 'verificacion_final.json'):
        shutil.copy2(RUN / name, OUT / name)
    frames, metrics, histories, configs, selected_rows, runs = {}, {}, [], [], {}, []
    for arm in pilot.ARMS:
        for loss in image.LOSSES:
            key = arm + '_' + loss
            source = RUN / f'oof_piloto_{key}.csv'
            shutil.copy2(source, OUT / source.name)
            frame = pd.read_csv(source).sort_values('patient_id').reset_index(drop=True)
            frames[key] = frame
            result = image.metricas_binarias(frame.label, frame.probability_raw, .5)
            result['sensitivity'] = result['recall_pcr']
            result['specificity'] = result['recall_no_pcr']
            result['fold_auc'] = {str(fold): float(roc_auc_score(g.label, g.probability_raw))
                                  for fold, g in frame.groupby('fold')}
            result['mean_fold_auc'] = float(np.mean(list(result['fold_auc'].values())))
            assert np.isclose(result['mean_fold_auc'], report['losses'][loss][arm]['mean_fold_auc'])
            selected_rows[key] = []
            for seed in image.SEEDS:
                for fold in pilot.PILOT_FOLDS:
                    path = pilot.run_path(RUN, arm, loss, seed, fold)
                    summary = json.loads((path / 'summary.json').read_text())
                    config = json.loads((path / 'config.json').read_text())
                    history = pd.read_csv(path / 'history.csv')
                    assert summary['status'] == 'complete' and not summary['smoke']
                    assert base.sha256(path / 'inference.pt') == summary['inference_sha256']
                    best = history.loc[history.epoch.eq(summary['best_epoch'])].iloc[0]
                    selected_rows[key].append(best.to_dict())
                    history = history.assign(arm=arm, loss=loss, seed=seed, fold=fold,
                                             selected_epoch=summary['best_epoch'])
                    histories.append(history)
                    configs.append({'arm': arm, 'loss': loss, 'seed': seed, 'fold': fold,
                                    'config': config, 'summary': summary,
                                    'environment': json.loads((path / 'environment.json').read_text())})
                    runs.append((arm, loss, seed, fold, path, config, summary))
            result['selected_gap_median'] = float(np.median([r['auc_gap'] for r in selected_rows[key]]))
            result['selected_fit_auc_median'] = float(np.median([r['fit_patient_auc'] for r in selected_rows[key]]))
            result['selected_selection_auc_median'] = float(np.median([r['selection_patient_auc'] for r in selected_rows[key]]))
            result['final_gap_median'] = report['losses'][loss][arm]['gaps']['final_gap_median']
            metrics[key] = result
    first = frames['control_normal']
    for frame in frames.values():
        assert frame[['patient_id', 'label', 'fold', 'cohort']].equals(first[['patient_id', 'label', 'fold', 'cohort']])
    groups = list(first.groupby(['fold', 'label']).indices.values())
    rng = np.random.default_rng(20261021)
    auc_boot = {key: {'pooled': [], 'mean_fold': []} for key in frames}
    for _ in range(2000):
        sample = np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in groups])
        y = first.iloc[sample].label.to_numpy()
        fold_ids = first.iloc[sample].fold.to_numpy()
        for key, frame in frames.items():
            p = frame.iloc[sample].probability_raw.to_numpy()
            auc_boot[key]['pooled'].append(roc_auc_score(y, p))
            auc_boot[key]['mean_fold'].append(np.mean([roc_auc_score(y[fold_ids == f], p[fold_ids == f])
                                                     for f in pilot.PILOT_FOLDS]))
    comparisons = {}
    for key in frames:
        metrics[key]['pooled_auc_ci95'] = np.quantile(auc_boot[key]['pooled'], [.025, .975]).tolist()
        metrics[key]['mean_fold_auc_ci95'] = np.quantile(auc_boot[key]['mean_fold'], [.025, .975]).tolist()
    for loss in image.LOSSES:
        candidate, control = 'antioverfit_' + loss, 'control_' + loss
        comparisons[loss] = dict(report['losses'][loss]['comparison'])
        delta = np.array(auc_boot[candidate]['mean_fold']) - np.array(auc_boot[control]['mean_fold'])
        comparisons[loss]['delta_mean_fold_auc_ci95'] = np.quantile(delta, [.025, .975]).tolist()
    elapsed = (datetime.fromisoformat(state['finished_at'])-datetime.fromisoformat(state['started_at'])).total_seconds()
    evaluation = {'metric_scope': 'piloto adaptativo de desarrollo, solo folds 0 y 3',
                  'n_patients': len(first), 'n_positive': int(first.label.sum()),
                  'metrics': metrics, 'comparisons': comparisons, 'threshold': .5,
                  'probability_scale': 'raw, no post-hoc calibration', 'bootstrap_samples': 2000,
                  'bootstrap_seed': 20261021, 'bootstrap_strata': ['fold', 'label'],
                  'runtime_seconds': elapsed, 'physical_epochs': sum(len(h) for h in histories),
                  'test_images_loaded': 0, 'active_model': 'v002_multimodal_clinico_20261007'}
    write(OUT / 'evaluacion_recalculada.json', evaluation)
    write(OUT / 'configuraciones_y_resumenes.json', configs)
    pd.concat(histories, ignore_index=True).to_csv(OUT / 'historias.csv', index=False)
    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(figsize=(9.5, 7))
    for key, frame in frames.items():
        x, y, _ = roc_curve(frame.label, frame.probability_raw)
        ax.plot(x, y, color=COLORS[key], label=f"{key}: AUC {metrics[key]['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='Tasa de falsos positivos', ylabel='Sensibilidad pCR', xlim=(0, 1), ylim=(0, 1),
           title=f'ROC exterior del piloto: {len(first)} pacientes, folds 0 y 3')
    ax.legend(loc='lower right'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / '01_roc_piloto.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, loss in zip(axes, image.LOSSES):
        for j, arm in enumerate(pilot.ARMS):
            key = arm + '_' + loss
            ax.bar(np.arange(2)+(j-.5)*.30, [metrics[key]['fold_auc'][str(f)] for f in (0, 3)],
                   width=.30, label=arm, color=COLORS[key])
        ax.axhline(.60, color='gray', linestyle='--', label='Mínimo 0,60')
        ax.axhline(.70, color='gray', linestyle=':')
        ax.set(xticks=[0, 1], xticklabels=['Fold 0', 'Fold 3'], ylim=(0, 1), title='BCE ' + loss,
               ylabel='AUC cruda por paciente')
        ax.legend(loc='upper right'); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Comparación exterior del ensemble de dos semillas')
    fig.tight_layout(); fig.savefig(OUT / '02_auc_por_fold.png', dpi=160); plt.close(fig)
    for kind in ('auc', 'perdida'):
        fig, axes = plt.subplots(4, 4, figsize=(19, 13.5))
        for i, (arm, loss) in enumerate((a, l) for a in pilot.ARMS for l in image.LOSSES):
            for j, (seed, fold) in enumerate((s, f) for s in image.SEEDS for f in pilot.PILOT_FOLDS):
                ax = axes[i, j]
                h = next(h for h in histories if h.arm.iloc[0] == arm and h.loss.iloc[0] == loss
                         and h.seed.iloc[0] == seed and h.fold.iloc[0] == fold)
                fit_key, selection_key = ('fit_patient_auc', 'selection_patient_auc') if kind == 'auc' else ('fit_eval_loss', 'selection_loss')
                ax.plot(h.epoch, h[fit_key], label='Fit')
                ax.plot(h.epoch, h[selection_key], label='Selección interna')
                ax.axvline(h.selected_epoch.iloc[0], color='gray', linestyle='--', label='Checkpoint elegido')
                ax.set(title=f'{arm} · {loss}\nsemilla {seed} · fold {fold}', xlabel='Época')
                if kind == 'auc': ax.set_ylim(0, 1)
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle('AUC por paciente: fit y selección' if kind == 'auc' else
                     'Pérdida por corte: cada curva usa la BCE de su propio brazo')
        fig.tight_layout(rect=(0, 0, 1, .97))
        filename = '03_auc_entrenamiento.png' if kind == 'auc' else '04_perdidas_entrenamiento.png'
        fig.savefig(OUT / filename, dpi=160); plt.close(fig)
    lines = ['# Piloto anti-sobreajuste: resultados y preparación de entrega', '',
             '21/10/2026. Entrega local preparada; no se realiza commit ni push.', '',
             f"Se completaron 16 entrenamientos y {evaluation['physical_epochs']} épocas en {elapsed/60:.2f} minutos. "
             f"La evaluación exterior del piloto reúne {len(first)} pacientes ({int(first.label.sum())} pCR), "
             'de los folds 0 y 3. Cada paciente se evalúa con las dos semillas de su fold, sin entrar '
             'en el entrenamiento ni en la selección de esos modelos. Test y validación privada siguen cerrados.', '',
             '**Conclusión:** el paquete reduce la brecha fit−selección, pero no alcanza AUC media 0,60. '
             'La BCE ponderada cumple el filtro exploratorio (+0,0232 de AUC media); su intervalo de diferencia '
             'incluye cero. La BCE normal no mejora. El candidato ponderado tampoco supera la estimación '
             'puntual del control normal (0,5571 frente a 0,5593). V002 permanece activo; no se inicia otra matriz.', '',
             '## Resultado principal', '',
             '| BCE | Control: AUC media | Anti-overfit: AUC media | Diferencia | IC95 % diferencia media | Gate prefijado |',
             '|---|---:|---:|---:|---|---|']
    for loss in image.LOSSES:
        c, a, comparison = metrics['control_' + loss], metrics['antioverfit_' + loss], comparisons[loss]
        lo, hi = comparison['delta_mean_fold_auc_ci95']
        lines.append(f"| {loss} | {c['mean_fold_auc']:.6f} | {a['mean_fold_auc']:.6f} | "
                     f"{comparison['delta_mean_fold_auc']:+.6f} | [{lo:+.6f}; {hi:+.6f}] | {comparison['promising_gate']} |")
    lines += ['', 'El gate exige +0,01 de AUC media, ningún fold peor en más de 0,02 y mediana '
              'de brecha fit−selección del checkpoint no mayor. Es un filtro exploratorio: incluso '
              'si se supera, solo justificaría completar cinco folds con protocolo nuevo. '
              'No equivale a generalización independiente ni a autorización para abrir test.', '',
              '## Discriminación, pérdidas y errores', '',
              'Probabilidades crudas, sin calibración posterior. Todos los recuentos siguientes '
              'usan umbral fijo 0,5; no se optimiza ese umbral con estas pacientes. Positivo significa pCR, '
              'no presencia de cáncer. La AUC agrupada mezcla salidas de modelos de folds diferentes; '
              'la métrica principal prefijada es la media de AUC por fold.', '',
              '| Brazo / BCE | AUC agrupada | IC95 % | AP | Log-loss | Brier | FP | FN | Sensibilidad | Especificidad |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, m in metrics.items():
        lo, hi = m['pooled_auc_ci95']
        lines.append(f"| {key} | {m['roc_auc']:.6f} | [{lo:.6f}; {hi:.6f}] | {m['average_precision']:.6f} | "
                     f"{m['log_loss']:.6f} | {m['brier']:.6f} | {m['fp']} | {m['fn']} | {m['sensitivity']:.2%} | {m['specificity']:.2%} |")
    lines += ['', 'A umbral 0,5, ambas BCE normales predicen todas las pacientes como no pCR. '
              'En la ponderada, anti-overfit reduce FP de 232 a 52, pero aumenta FN de 36 a 110 '
              'y baja TP de 103 a 29. Es un cambio de compromiso entre sensibilidad y especificidad, '
              'no una mejora simultánea de ambos errores. No se reajusta el umbral con estas OOF.']
    lines += ['', '## Sobreajuste', '',
              '| Brazo / BCE | Brecha mediana en checkpoint | Brecha mediana al final | Mediana AUC fit elegida | Mediana AUC selección elegida |',
              '|---|---:|---:|---:|---:|']
    for key, m in metrics.items():
        lines.append(f"| {key} | {m['selected_gap_median']:.6f} | {m['final_gap_median']:.6f} | "
                     f"{m['selected_fit_auc_median']:.6f} | {m['selected_selection_auc_median']:.6f} |")
    lines += ['', 'Una brecha menor también puede reflejar menor aprendizaje en fit. Se requiere '
              'mejora exterior para considerar útil la regularización; no basta con acercar curvas.', '',
              '## Reproducibilidad y límites', '',
              '- CNN temporal simple desde cero, una sola escala, 551.913 parámetros; PRE/EARLY/LATE, sin clínica.',
              '- Control frente a un paquete conjunto: lotes de seis pacientes, igual peso por paciente, '
              'Dropout2d 0,10 y aumentos suaves comunes a las fases. No identifica cuál de sus componentes produce el efecto.',
              '- AdamW, LR 0,0008, WD 0,001, máximo 15 épocas, mínimo 4 y paciencia 4. Semillas 42/2026.',
              '- El arreglo local del informe corrige un desfase de una época: `best_epoch` ya está numerado desde uno. '
              'No altera pesos, arquitectura, selección ni entrenamiento. Prueba de regresión incluida.',
              '- 62 pruebas y cuatro pruebas cortas GPU pasadas; estas últimas no entran en selección ni métricas.',
              '- Bootstrap pareado de 2.000 remuestreos por fold y clase, semilla 20261021. '
              'Los intervalos condicionan a estos modelos y no recogen la variación de nuevos entrenamientos ni toda la búsqueda histórica.',
              '- Solo dos folds y antecedentes de exploración en train: desarrollo adaptativo, no evaluación independiente.',
              '- No se modifican V002 ni las siete versiones publicadas. No se inicia otro entrenamiento.', '',
              '## Entrega local', '',
              'Los dieciséis pesos de inferencia se conservan en un paquete específico de piloto bajo '
              '`modelos/pilotos/`, con código congelado y manifiesto. Contiene cuatro modelos por brazo/BCE '
              '(dos semillas × dos folds). No es una versión final de diez modelos ni se integra en la aplicación. '
              'Su inferencia conjunta usa cuatro modelos para entradas futuras; las métricas anteriores usan '
              'solo las dos semillas del fold que excluye a cada paciente.', '',
              'Se excluyen datasets, checkpoints con optimizador/RNG y pruebas cortas. '
              'Los artefactos quedan preparados para revisar y publicar únicamente cuando el usuario lo indique.', '',
              '- [Métricas completas](evaluacion_recalculada.json).',
              '- [Protocolo real y hashes de código](protocolo_ejecutado.json).',
              '- [Configuraciones, resúmenes y entornos](configuraciones_y_resumenes.json).',
              '- [Historias completas](historias.csv).',
              '- [Auditoría final de los 16 entrenamientos](verificacion_final.json).', '',
              '- [Verificación de carga e inferencia del paquete aislado](verificacion_paquete.json).', '',
              '![ROC](01_roc_piloto.png)', '', '![AUC por fold](02_auc_por_fold.png)', '',
              '![AUC de entrenamiento y selección](03_auc_entrenamiento.png)', '',
              '![Pérdidas de entrenamiento y selección](04_perdidas_entrenamiento.png)', '']
    (OUT / 'INFORME_RESULTADOS.md').write_text('\n'.join(lines))
    print(json.dumps({'metrics': metrics, 'comparisons': comparisons, 'elapsed_minutes': elapsed/60,
                      'physical_epochs': evaluation['physical_epochs']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
