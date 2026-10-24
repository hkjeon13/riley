"""Exploratory G2 screen; never treats a completed but variant arm as passing."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

P = Path('/mnt/efs/nlp/psyche/riley-phase2-20261023')
W = Path('/mnt/efs/nlp/psyche/riley-w0-a100-gpu67-20261019')
OUT = Path('/mnt/efs/nlp/psyche/riley-phase2-resume-20261024')
LABEL = 'p2g-no-fma'
while not (P / 'build' / (LABEL + '.finished')).exists():
    if (P / 'build' / (LABEL + '.failed')).exists():
        raise SystemExit('candidate build failed')
    time.sleep(5)
for test in ['fi', 'pd', 'fu', 'bench']:
    if (P / 'build' / ('gpu-' + test + '-' + LABEL + '.exit')).read_text().strip() != '0':
        raise SystemExit('candidate GPU test failed: ' + test)
plan = json.loads((P / 'runs/screen-fi2-natural/plan.json').read_text())
binary = P / 'build' / ('riley-' + LABEL)
command = ['python3', str(W / 'control.py'), '--lease-dir', str(W / 'leases-v2'),
           '--index', '7', 'run', '--', 'python3', str(P / 'measure-phase2-resume.py')]
values = {
    'out': str(OUT / 'screen-no-fma'), 'binary': plan['binary'],
    'candidate-binary': plan['candidate_binary'], 'native-binary': plan['native_binary'],
    'native-reference': plan['native_reference'], 'native-extra': json.dumps(plan['native_extra_args']),
    'native-after-binary': str(binary),
    'native-after-env': json.dumps({'RILEY_NATIVE_D128_ATTENTION_FLASHINFER': '1'}),
    'model': plan['model'], 'reference': plan['reference'], 'gpu': plan['gpu'],
    'workload': 'natural', 'concurrency': '8,32', 'repeats': '1',
    'keep': '192', 'warm': '64', 'modes': 'native-after',
}
for key, value in values.items():
    command.extend(['--' + key, value])
command.extend(['--per-mode-canonical', '--require-native-invariance'])
(OUT / 'screen-no-fma.argv.json').write_text(json.dumps(command, indent=2))
(OUT / 'screen-no-fma.binary.json').write_text(json.dumps({
    'path': str(binary), 'sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
}, indent=2))
with (OUT / 'screen-no-fma.log').open('x') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
(OUT / 'screen-no-fma.exit').write_text(str(result.returncode))
raise SystemExit(result.returncode)
