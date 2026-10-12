"""Freeze the Batch13 stability plan and run its lanes sequentially. Stops at the first failed lane; nothing is retried or excluded."""
import hashlib, json, subprocess, sys, time
from pathlib import Path

R = Path('/data/riley-serving-261007'); C = Path(__file__).parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
attempt = int(sys.argv[1])
matched = 'kernel-batch13-quiet-attempt01'
terminal = json.loads((R / matched / 'completion.json').read_text())
assert terminal['failure'] is None and terminal['all_lanes_complete'], 'matched serving attempt incomplete'
analysis = json.loads((R / 'kernel-batch13-independent-analysis-attempt01/analysis.json').read_text())
assert analysis['matrix_complete'] and not analysis['problems'], 'independent analysis incomplete'
serving_plan = json.loads((R / matched / 'plan-snapshot.json').read_text())
prep = json.loads((R / matched / 'preparation.json').read_text())
plan_path = C / 'stability-plan.json'
if not plan_path.exists():
    cases = [f'c{c}-{w}' for c in serving_plan['concurrency'] for w in serving_plan['workloads']]
    plan = {'status': 'FROZEN_FOR_EXECUTION', 'schema': 'riley.kernel-batch13.stability.v1', 'cases': cases, 'engines': ['candidate', 'vllm'],
            'lanes': [[c, 'candidate'] for c in cases] + [['c32-natural', 'vllm'], ['c32-fixed', 'vllm']],
            'lane_scope': 'candidate on all eight cells; vLLM on both C32 cells as an error/memory-trend reference only',
            'matched_attempt': matched, 'output_prefix': 'kernel-batch13-stability',
            'binaries': {'candidate': serving_plan['candidate_binary_sha256'], 'vllm': prep['pins'][str(R / 'vllm0271-venv/bin/vllm')]},
            'host_start_gate': serving_plan['host_start_gate'], 'sustained_seconds': 300, 'cancellation_trials_each_mode': 20,
            'loaded_cancellations': 40, 'loaded_cancel_interval_seconds': 0.25, 'loaded_round_requests': 96,
            'correctness_policy': 'Riley rows exact against the frozen token/text/finish reference in every phase including during and after cancellation; all engines transport/protocol valid',
            'pass_criteria': ['zero request errors', 'exact references (Riley)', 'KV blocks/active/waiting drain to zero after sustained load, after every idle cancellation, after re-requests and after cancellation under load',
                              'exit code 0, no GPU compute process, port closed, native device/pinned live allocations zero at close (Riley)'],
            'limits': ['early cancellation closes right after the request write; the server-side phase (queued/prefill) is not observable from the client',
                       'memory is sampled once per second; peaks between samples are not proven', 'no performance qualification'],
            'no_exclusions': True, 'pins': {str(C / n): sha(C / n) for n in ['stability13.py', 'serving_token_client_v2.py', 'mixed_phase_v34.py', 'fixtures.json']},
            'predeclared_time_ns': time.time_ns()}
    plan_path.write_text(json.dumps(plan, indent=1) + '\n')
plan = json.loads(plan_path.read_text())
out = R / f'kernel-batch13-stability-run-attempt{attempt:02d}'; out.mkdir()
records = []; failure = None
try:
    for case, engine in plan['lanes']:
        with (out / f'{case}-{engine}.log').open('x') as log:
            p = subprocess.run([str(R / 'vllm0271-venv/bin/python'), str(C / 'stability13.py'), '--case', case, '--engine', engine, '--attempt', str(attempt)], cwd=C, stdout=log, stderr=subprocess.STDOUT)
        records.append({'case': case, 'engine': engine, 'exit': p.returncode}); print(case, engine, p.returncode, flush=True)
        assert p.returncode == 0, f'{case} {engine} failed; evidence retained'
except BaseException as e:
    failure = {'type': type(e).__name__, 'message': str(e)}; raise
finally:
    (out / 'completion.json').write_text(json.dumps({'failure': failure, 'records': records, 'plan_sha256': sha(plan_path), 'all_lanes_complete': failure is None and len(records) == len(plan['lanes'])}, indent=1) + '\n')
