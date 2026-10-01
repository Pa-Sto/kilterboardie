"""Run the three full training jobs sequentially on a shared held-out split."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    output = root / 'runs/model_comparison_20260930'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'status.json').exists():
        raise RuntimeError('Comparison already exists; refusing to overwrite its run state.')
    if not (output / 'split.json').exists():
        raise RuntimeError('Create and verify the shared split manifest before training.')
    state = {'started': time.time(), 'epochs': 30, 'jobs': {}}
    def save():
        temporary = output / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(output / 'status.json')
    for name, script, extra in (
        ('cvae', 'cvae_train.py', []),
        ('diffusion', 'diffusion_train.py', []),
        ('graph', 'graph_transformer_train.py', ['--max-sequence-length', '64']),
    ):
        command = [sys.executable, '-u', script, '--data-dir',
                   'ImageData/50Degree/ExportBoardsesh', '--split-manifest',
                   str(output / 'split.json'), '--epochs', '30', '--batch-size', '32',
                   '--device', 'mps', '--out-dir', str(output / name), *extra]
        job = state['jobs'][name] = dict(command=command, started=time.time(), status='running')
        save()
        with (output / f'{name}.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'OMP_NUM_THREADS': '4'})
        job.update(status='completed' if result.returncode == 0 else 'failed',
                   returncode=result.returncode, seconds=time.time()-job['started'])
        save()
    state['finished'] = time.time()
    save()
    if all(j['status'] == 'completed' for j in state['jobs'].values()):
        with (output / 'evaluation.log').open('w') as log:
            result = subprocess.run([sys.executable, '-u', 'evaluate_comparison.py',
                                     '--root', str(output)], stdout=log, stderr=subprocess.STDOUT)
        state['evaluation_returncode'] = result.returncode
        save()


if __name__ == '__main__':
    main()
