"""Run qualified batch3/batch4 campaigns serially, then restore Blender.

All binaries and plans must already exist. This script never builds sources,
never rewrites qualification evidence, and creates fresh output directories.
"""
import json
from pathlib import Path
import subprocess
import time

import run_engine_optimization as engine
import run_serving_optimization as http

ROOT = Path('/tmp/riley-opt-260912')
REQUEST = Path('/tmp/riley-g04-vllm-profile-260911/request.json')
SESSION = ROOT / 'remote_session_round3.py'


def run(argv):
    subprocess.run(argv, check=True)


def validate_all():
    request = json.loads(REQUEST.read_text())
    for batch in ('batch3', 'batch4'):
        binding_path = ROOT / (batch + '-binding.json')
        binding = json.loads(binding_path.read_text())
        qualification = json.loads((ROOT / (batch + '-qualification.json')).read_text())
        assert qualification['passed'] is True
        for mode, validator in [('http', http.validate_artifacts), ('engine', engine.validate_plan)]:
            plan = json.loads((ROOT / (batch + '-' + mode + '-plan.json')).read_text())
            assert not (ROOT / (batch + '-' + mode)).exists()
            validator(plan, REQUEST, binding_path, request, binding)
    diagnostic = json.loads((ROOT / 'diagnostic-batch4-build.json').read_text())
    build = json.loads((ROOT / 'batch4-build.json').read_text())
    assert diagnostic['base_source_commit'] == build['source_commit']
    assert diagnostic['binary_sha256'] == http.digest(ROOT / 'diagnostic-batch4-target/release/riley')
    assert diagnostic['instrumented'] and not diagnostic['performance_claim_eligible']
    assert not (ROOT / 'diagnostic-batch4-trial.log').exists()


validate_all()
run(['python3', str(SESSION), 'check'])
try:
    run(['python3', str(SESSION), 'stop'])
    stopped = json.loads((ROOT / 'blender-round3/stopped.json').read_text())['pids']
    binding = json.loads((ROOT / 'batch3-binding.json').read_text())
    deadline = time.monotonic() + 30
    while True:
        sample = http.gpu_snapshot(binding)
        assert set(sample['compute_pids']) <= {str(pid) for pid in stopped}, 'unexpected foreign GPU process'
        if not sample['compute_pids'] and sample['memory_used_mib'] <= http.CONDITION['idle_memory_limit_mib']:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('terminated Blender GPU contexts did not drain')
        time.sleep(.2)
    for batch in ('batch3', 'batch4'):
        for mode, runner in [('http', 'run_serving_optimization.py'), ('engine', 'run_engine_optimization.py')]:
            run(['python3', str(SESSION), 'extend'])
            print(json.dumps({'starting': batch + '-' + mode}), flush=True)
            run(['python3', str(ROOT / runner), '--plan', str(ROOT / (batch + '-' + mode + '-plan.json')),
                 '--request', str(REQUEST), '--binding', str(ROOT / (batch + '-binding.json')),
                 '--output', str(ROOT / (batch + '-' + mode)), '--measure'])
            assert json.loads((ROOT / (batch + '-' + mode) / 'summary.json').read_text())['completed']
    run(['python3', str(SESSION), 'extend'])
    plan = json.loads((ROOT / 'batch4-http-plan.json').read_text())
    binding = json.loads((ROOT / 'batch4-binding.json').read_text())
    preflight = ROOT / 'diagnostic-batch4-preflight'
    preflight.mkdir()
    http.preflight(plan, binding, preflight)
    run(['python3', str(ROOT / 'profile_batch4_trial.py')])
    with (ROOT / 'diagnostic-batch4-summary.json').open('x') as stream:
        subprocess.run(['python3', str(ROOT / 'profile_owned_graph_batch2.py'), 'summarize',
                        '--log', str(ROOT / 'diagnostic-batch4-trial.log')], stdout=stream, check=True)
    print(json.dumps({'completed': True, 'campaigns': 4, 'diagnostic_complete': True}), flush=True)
finally:
    if (ROOT / 'blender-round3/session.json').exists():
        run(['python3', str(SESSION), 'restore'])
