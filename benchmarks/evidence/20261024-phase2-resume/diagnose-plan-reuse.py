"""Re-run the interrupted plan-reuse isolation on the same frozen p2e binary."""
import hashlib
import json
from pathlib import Path
import subprocess

P = Path('/mnt/efs/nlp/psyche/riley-phase2-20261023')
W = Path('/mnt/efs/nlp/psyche/riley-w0-a100-gpu67-20261019')
OUT = Path('/mnt/efs/nlp/psyche/riley-phase2-resume-20261024')
OUT.mkdir(exist_ok=True)
source_plan = json.loads((P / 'runs/screen-fi2-natural/plan.json').read_text())
binary = P / 'build/riley-p2e'
(OUT / 'diagnostic-binary.json').write_text(json.dumps({
    'binary': str(binary), 'sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
    'purpose': 'plan reuse isolation, exploratory; no adoption decision',
}, indent=2))
for label, always in [('always-plan', True), ('reuse-plan', False)]:
    env = {'RILEY_NATIVE_D128_ATTENTION_FLASHINFER': '1'}
    if always:
        env['RILEY_NATIVE_D128_FLASHINFER_ALWAYS_PLAN'] = '1'
    command = ['python3', str(W / 'control.py'), '--lease-dir', str(W / 'leases-v2'),
               '--index', '7', 'run', '--', 'python3', str(P / 'measure-phase2.py')]
    values = {
        'out': str(OUT / label), 'binary': source_plan['binary'],
        'candidate-binary': source_plan['candidate_binary'],
        'native-binary': source_plan['native_binary'],
        'native-reference': source_plan['native_reference'],
        'native-extra': json.dumps(source_plan['native_extra_args']),
        'native-after-binary': str(binary), 'native-after-env': json.dumps(env),
        'model': source_plan['model'], 'reference': source_plan['reference'],
        'gpu': source_plan['gpu'], 'workload': 'natural', 'concurrency': '8,32',
        'repeats': '1', 'keep': '192', 'warm': '64', 'modes': 'native-after',
    }
    for key, value in values.items():
        command.extend(['--' + key, value])
    command.append('--per-mode-canonical')
    (OUT / (label + '.argv.json')).write_text(json.dumps(command, indent=2))
    with (OUT / (label + '.log')).open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    (OUT / (label + '.exit')).write_text(str(result.returncode))
    if result.returncode:
        raise SystemExit(result.returncode)
(OUT / 'diagnostic.finished').touch()
