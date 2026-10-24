"""Check the named profile against the qualified candidate's canonical output."""
import json
from pathlib import Path
import subprocess

P = Path('/mnt/efs/nlp/psyche/riley-phase2-20261023')
W = Path('/mnt/efs/nlp/psyche/riley-w0-a100-gpu67-20261019')
OUT = Path('/mnt/efs/nlp/psyche/riley-phase2-resume-20261024')
assert (OUT / 'build/profile.finished').exists()
analysis = json.loads((OUT / 'qualification-analysis.json').read_text())
assert analysis['accepted'], 'do not promote a rejected candidate through the new profile'
profile = OUT / 'profile-extra.json'
profile.write_text(json.dumps(['--inference-profile', 'native-bf16-varlen-v1']))
plan = json.loads((P / 'runs/screen-fi2-natural/plan.json').read_text())
command = ['python3', str(W / 'control.py'), '--lease-dir', str(W / 'leases-v2'),
           '--index', '7', 'run', '--', 'python3', str(P / 'measure-phase2-resume.py')]
values = {
    'out': str(OUT / 'profile-smoke'), 'binary': plan['binary'],
    'candidate-binary': plan['candidate_binary'], 'native-binary': plan['native_binary'],
    'native-reference': plan['native_reference'], 'native-extra': json.dumps(plan['native_extra_args']),
    'native-after-binary': str(P / 'build/riley-p2h-profile'),
    'native-after-extra': str(profile), 'native-after-env': '{}',
    'model': plan['model'], 'reference': plan['reference'], 'gpu': plan['gpu'],
    'workload': 'natural', 'concurrency': '1,32', 'repeats': '1',
    'keep': '12', 'warm': '3', 'modes': 'native-after',
}
for key, value in values.items():
    command.extend(['--' + key, value])
command.extend(['--per-mode-canonical', '--require-native-invariance'])
(OUT / 'profile-smoke.argv.json').write_text(json.dumps(command, indent=2))
with (OUT / 'profile-smoke.log').open('x') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
(OUT / 'profile-smoke.exit').write_text(str(result.returncode))
if result.returncode:
    raise SystemExit(result.returncode)
canonical = {}
for row in json.loads((OUT / 'qual-natural/c32-r0-native-after/correctness.json').read_text()):
    if row['id'] in canonical:
        assert canonical[row['id']] == row['value']
    canonical[row['id']] = row['value']
checked = 0
for conc in [1, 32]:
    arm = OUT / 'profile-smoke' / f'c{conc}-r0-native-after'
    assert 'RILEY_NATIVE_ATTENTION mode=fixed-tile-unsplit-v1' in (arm / 'server.log').read_text()
    for filename in ['correctness.json', 'w0/requests.json']:
        for row in json.loads((arm / filename).read_text()):
            assert row['value'] == canonical[row['id']], (conc, row['id'], filename)
            checked += 1
(OUT / 'profile-smoke-verdict.json').write_text(json.dumps({
    'canonical_checks': checked, 'all_equal': True,
    'selection': 'native-bf16-varlen-v1 -> fixed-tile-unsplit-v1',
    'scope': 'CLI/profile equivalence and basic HTTP streaming/recovery, not a new timing campaign',
}, indent=2))
print('Profile canonical checks passed:', checked)
