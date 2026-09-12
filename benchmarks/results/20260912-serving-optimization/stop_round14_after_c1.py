"""Stop only the owned Round14 controller after all 15 prescribed C1 pairs.

This does not modify the frozen 30-pair plan or its results. SIGINT invokes the
controller's existing cleanup/finally restoration. The campaign stays partial.
No GPU work, Blender signals, or server-process signals are performed here.
"""
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import time

ROOT = Path('/tmp/riley-opt-260912')
CAMPAIGN = ROOT/'token-serving-round14'
PID = 2289182
START = '83585224'
PLAN_SHA = '199d70ebf61e03d3db00ed557b454632daf2aad934ae56d6c4e027b37de7ed7b'
ARGV = ['/data/riley-vllm-interim.CfrT9T/venv/bin/python',
        str(ROOT/'run_serving_token_optimization_v4.py'), '--plan',
        str(ROOT/'token-serving-round14-plan.json'), '--output', str(CAMPAIGN), '--measure']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')


def main():
    plan_path = ROOT/'token-serving-round14-plan.json'
    if sha(plan_path) != PLAN_SHA:
        raise RuntimeError('frozen plan changed')
    plan = json.loads(plan_path.read_text())
    if plan['settings'][0]['id'] != 'c1-b128' or plan['workload']['pairs'] != 5 or plan['workload']['retained_requests_per_process'] != 1000:
        raise RuntimeError('predeclared C1 scope changed')
    expected = [(CAMPAIGN/f"c1-b128-{c['id']}-pair{i:02d}"/'pair.json', c, i)
                for c in plan['comparisons'] for i in range(1, 6)]
    if len(expected) != 15:
        raise RuntimeError('three complete C1 comparisons required')
    # V4 writes pair.json before appending it to its in-memory finalization
    # inventory. Entry into the next setting proves that append has completed.
    next_pair = CAMPAIGN/f"{plan['settings'][1]['id']}-{plan['comparisons'][0]['id']}-pair01"
    if not plan['settings'][1]['id'].startswith('c2-'):
        raise RuntimeError('expected next C2 setting changed')
    proc = Path('/proc')/str(PID)
    fd = os.pidfd_open(PID)
    try:
        if proc.joinpath('stat').read_text().rsplit(') ', 1)[1].split()[19] != START or [x.decode() for x in proc.joinpath('cmdline').read_bytes().split(b'\0')[:-1]] != ARGV:
            raise RuntimeError('owned controller identity changed')
        prior = []
        for path, comparison, index in expected[5:10]:
            row = json.loads(path.read_text())
            ratio = row['ratios']
            if comparison['id'] != 'candidate-baseline' or ratio['throughput_right_over_left'] >= 1 or ratio['token_tpot_ms_right_over_left']['median'] <= 1:
                raise RuntimeError('recorded reason for stopping is not supported')
            prior.append({'path': str(path), 'sha256': sha(path), 'ratios': ratio})
        write(ROOT/'token-round14-c1-stop-decision.json', {
            'created_unix_seconds': time.time(), 'controller_pid': PID, 'controller_start': START,
            'controller_argv': ARGV, 'supervisor_sha256': sha(Path(__file__)),
            'original_plan_sha256': PLAN_SHA, 'original_plan_modified': False,
            'reason': 'All five C1 candidate/baseline pairs regress in throughput and median TPOT. Finish all five C1 direct-vLLM pairs, then prioritize diagnosis of rejected candidate.',
            'observed_candidate_baseline_pairs': prior,
            'stop_after': [str(path) for path, _, _ in expected],
            'signal': 'SIGINT to exact owned controller via pidfd; controller performs cleanup and Blender restoration',
            'overall_campaign_will_remain_incomplete': True, 'c2_performance_claim': False})
        deadline = time.monotonic() + 2400
        outcome = None
        while time.monotonic() < deadline:
            if (CAMPAIGN/'finalization.json').exists() or select.select([fd], [], [], 0)[0]:
                outcome = {'signal_sent': False, 'reason': 'controller finalized or exited before C1 stopping point'}
                break
            if next_pair.is_dir() and all(path.exists() for path, _, _ in expected):
                try:
                    rows = [json.loads(path.read_text()) for path, _, _ in expected]
                except json.JSONDecodeError:
                    time.sleep(2)
                    continue
                refs = []
                for (path, comparison, index), row in zip(expected, rows):
                    if row['completed'] is not True or row['comparison'] != comparison or row['index'] != index:
                        raise RuntimeError('completed C1 pair identity differs')
                    for lane in row['processes'].values():
                        summary = lane['summary']
                        if lane['completed'] is not True or summary['completed'] is not True or summary['succeeded'] != 1000 or summary['failed'] != 0 or summary['reference_matches'] != 1000:
                            raise RuntimeError('C1 stopping point contains an incomplete lane')
                    refs.append({'path': str(path), 'sha256': sha(path)})
                if sha(plan_path) != PLAN_SHA:
                    raise RuntimeError('plan changed while waiting')
                c2_launches = sorted(str(path) for path in CAMPAIGN.glob('c2-*/*/launch.json'))
                signal.pidfd_send_signal(fd, signal.SIGINT)
                outcome = {'signal_sent': True, 'signal_unix_seconds': time.time(),
                           'completed_c1_pair_receipts': refs, 'c2_launches_at_signal': c2_launches,
                           'reason': 'all 15 prescribed C1 pairs completed; original 30-pair campaign intentionally incomplete'}
                break
            time.sleep(2)
        if outcome is None:
            outcome = {'signal_sent': False, 'reason': 'supervisor deadline; controller left untouched'}
        write(ROOT/'token-round14-c1-stop-outcome.json', outcome)
        print(json.dumps(outcome), flush=True)
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
