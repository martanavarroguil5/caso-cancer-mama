import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import median

OUT = Path('/home/jorgesrab/Escritorio/caso-cancer-mama/resultados/04_entrenamiento/realce_temporal_20261009')
MONITOR = OUT / 'seguimiento'
state = json.loads((MONITOR / 'estado.json').read_text())
def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False

variants = ['enhancement_clinical', 'enhancement_regularized_clinical']
runs = []
times = {name: [] for name in variants}
finished_epochs = {name: [] for name in variants}
for name in variants:
    for seed in (42, 2026):
        for fold in range(5):
            folder = OUT / name / 'weighted' / f'seed_{seed}' / f'fold_{fold}'
            history = folder / 'history.csv'
            rows = list(csv.DictReader(history.open())) if history.is_file() else []
            summary = json.loads((folder / 'summary.json').read_text()) if (folder / 'summary.json').is_file() else {}
            completed = summary.get('status') == 'complete' and not summary.get('smoke', False)
            samples = rows[1:] if len(rows) > 1 else rows
            times[name] += [float(row['epoch_seconds']) for row in samples]
            epoch = int(rows[-1]['epoch']) if rows else 0
            if completed:
                finished_epochs[name].append(int(summary['epochs_completed']))
            runs.append({'variant': name, 'seed': seed, 'fold': fold, 'epoch': epoch,
                         'complete': completed, 'started': (folder / 'config.json').exists(),
                         'last_update_at': datetime.fromtimestamp(history.stat().st_mtime, timezone.utc).isoformat() if rows else None,
                         'best_auc': summary.get('best_auc') if completed else max((float(row['patient_auc']) for row in rows), default=None),
                         'last_train_loss': float(rows[-1]['train_loss']) if rows else None,
                         'last_val_loss': float(rows[-1]['val_loss']) if rows else None})

all_times = [t for group in times.values() for t in group]
all_finished = [e for group in finished_epochs.values() for e in group]
upper = typical = 0.0
for run in runs:
    if run['complete']:
        continue
    name = run['variant']
    duration = median(times[name] or all_times) if all_times else None
    if duration is None:
        continue
    target = median(finished_epochs[name] or all_finished) if all_finished else 46
    target = max(run['epoch'] + 1, target)
    upper += max(0, 46 - run['epoch']) * duration * 1.10
    typical += max(0, target - run['epoch']) * duration * 1.10

elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(state['started_at'])).total_seconds()
result = {'checked_at': datetime.now(timezone.utc).isoformat(), 'state': state,
          'runner_alive': alive(state.get('runner_pid')), 'child_alive': alive(state.get('child_pid')),
          'completed_runs': sum(run['complete'] for run in runs), 'total_runs': 20,
          'total_epochs_done': sum(run['epoch'] for run in runs), 'elapsed_seconds': elapsed,
          'active_runs': [run for run in runs if run['started'] and not run['complete']],
          'seconds_per_epoch': {name: median(values) if values else None for name, values in times.items()},
          'remaining_estimate_seconds': typical if all_times else None,
          'remaining_at_full_budget_seconds': upper if all_times else None,
          'estimate_basis': 'mediana de épocas completadas' if all_finished else 'máximo de 46 épocas por entrenamiento',
          'estimate_note': 'Incluye margen de 10%; la parada temprana y la comparación final pueden cambiar el tiempo.',
          'finished_runs': [run for run in runs if run['complete']]}
print(json.dumps(result, ensure_ascii=False, indent=2))
