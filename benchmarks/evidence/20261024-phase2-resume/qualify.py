"""Five fresh-process comparisons per cell; matched GPU, frozen runner/binary."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time

P = Path('/mnt/efs/nlp/psyche/riley-phase2-20261023')
W = Path('/mnt/efs/nlp/psyche/riley-w0-a100-gpu67-20261019')
OUT = Path('/mnt/efs/nlp/psyche/riley-phase2-resume-20261024')
manifest = json.loads((OUT / 'qualification-preregistration.json').read_text())
assert hashlib.sha256(Path(manifest['candidate_binary']).read_bytes()).hexdigest() == manifest['candidate_sha256']
assert hashlib.sha256((P / 'measure-phase2-resume.py').read_bytes()).hexdigest() == manifest['runner_sha256']
assert json.loads((OUT / 'screen-no-fma/complete.json').read_text())['native_invariance_passed']
plan = json.loads((P / 'runs/screen-fi2-natural/plan.json').read_text())


def run(label, index, workload, concurrency):
    command = ['python3', str(W / 'control.py'), '--lease-dir', str(W / 'leases-v2'),
               '--index', str(index), 'run', '--', 'python3', str(P / 'measure-phase2-resume.py')]
    gpu = {6: 'GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7',
           7: 'GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0'}[index]
    values = {
        'out': str(OUT / label), 'binary': plan['binary'],
        'candidate-binary': plan['candidate_binary'], 'native-binary': plan['native_binary'],
        'native-reference': plan['native_reference'], 'native-extra': json.dumps(plan['native_extra_args']),
        'native-after-binary': manifest['candidate_binary'],
        'native-after-env': json.dumps({'RILEY_NATIVE_D128_ATTENTION_FLASHINFER': '1'}),
        'model': plan['model'], 'reference': plan['reference'], 'gpu': gpu,
        'workload': workload, 'concurrency': concurrency, 'repeats': '5',
        'keep': '192', 'warm': '64', 'modes': 'native,native-after,vllm',
    }
    for key, value in values.items():
        command.extend(['--' + key, value])
    command.extend(['--per-mode-canonical', '--require-native-invariance'])
    (OUT / (label + '.argv.json')).write_text(json.dumps(command, indent=2))
    with (OUT / (label + '.log')).open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    (OUT / (label + '.exit')).write_text(str(result.returncode))
    return result.returncode


def gpu7():
    result = run('qual-fixed-c32', 7, 'fixed', '32')
    if result:
        return result
    return run('qual-natural', 7, 'natural', '8,32')


(OUT / 'qualification.started.json').write_text(json.dumps({'started_ns': time.time_ns()}))
with concurrent.futures.ThreadPoolExecutor(2) as pool:
    results = [pool.submit(run, 'qual-fixed-c1-c8-c16', 6, 'fixed', '1,8,16'),
               pool.submit(gpu7)]
    exits = [task.result() for task in results]
(OUT / 'qualification.execution.json').write_text(json.dumps({'exit_codes': exits,
                                                             'finished_ns': time.time_ns()}, indent=2))
raise SystemExit(max(exits))
