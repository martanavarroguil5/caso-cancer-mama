import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

REPO = Path('/home/jorgesrab/Escritorio/caso-cancer-mama')
OUT = REPO / 'resultados/04_entrenamiento/realce_temporal_20261009'
MONITOR = OUT / 'seguimiento'
MONITOR.mkdir(parents=True, exist_ok=True)
lock = (MONITOR / 'runner.lock').open('a')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit('Ya existe un lanzador activo para este experimento.')

def now():
    return datetime.now(timezone.utc).isoformat()

def snapshot():
    base = REPO / 'resultados/04_entrenamiento'
    paths = list((base / 'modelos').glob('*.pt'))
    paths += [base / 'modelo_final.json', base / 'modelo_final.json.sha256']
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}

state = {'status': 'running', 'stage': 'training', 'started_at': now(),
         'runner_pid': os.getpid(), 'total_runs': 20, 'epoch_budget': 46,
         'tests_passed': 39, 'gpu_smoke_passed': 2,
         'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()}
baseline = snapshot()
(MONITOR / 'pesos_anteriores.sha256.json').write_text(json.dumps(baseline, indent=2))

def save():
    state['updated_at'] = now()
    temp = MONITOR / 'estado.tmp'
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    temp.replace(MONITOR / 'estado.json')

python = str(REPO / '.venv/bin/python')
common = [python, '-u', '-B', '-m', 'cancer_mama.entrenamiento']
commands = [
    ('training', common + ['entrenar', '--configuraciones', 'enhancement_clinical',
       'enhancement_regularized_clinical', '--perdidas', 'ponderada', '--semillas', '42', '2026',
       '--folds', '0', '1', '2', '3', '4', '--epocas', '46', '--lote', '64', '--workers', '4',
       '--dispositivo', 'cuda', '--datos', 'breastdcedl', '--salida', str(OUT)]),
    ('comparison', common + ['comparar', '--datos', 'breastdcedl', '--salida', str(OUT)])
]
save()
try:
    with (MONITOR / 'entrenamiento.log').open('a', buffering=1) as log:
        for stage, command in commands:
            state.update(stage=stage, command=command)
            log.write('\n' + now() + ' ' + stage + '\n')
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = process.pid
            save()
            rc = process.wait()
            state['last_exit_code'] = rc
            if rc:
                raise RuntimeError(f'{stage} terminó con código {rc}')
        state.update(status='complete', stage='complete', finished_at=now())
except Exception as exc:
    state.update(status='failed', error=str(exc), finished_at=now())
finally:
    state['existing_final_unchanged'] = snapshot() == baseline
    save()
sys.exit(0 if state['status'] == 'complete' else 1)
