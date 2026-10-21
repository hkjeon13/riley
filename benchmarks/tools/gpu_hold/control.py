"""Run one benchmark command while keeping its GPU holder context alive."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from holder import GPUS, admitted_process, atomic_json, gpu_processes, identity


def status(lease, index):
    value = json.loads((lease / f'gpu{index}.status.json').read_text())
    assert value['status'] == 'running', value
    assert value['gpu'] == GPUS[index], value
    assert value['start_ticks'] == identity(value['pid']), value
    assert Path(f"/proc/{value['pid']}").stat().st_uid == os.getuid()
    assert time.time_ns() - value['updated_ns'] < 10**10, ('stale holder', value)
    return value


def phase(lease, index, target, owner=None):
    before = status(lease, index)
    command = {'phase': target}
    if owner:
        command['owner'] = owner
    atomic_json(lease / f'gpu{index}.desired.json', command)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        current = status(lease, index)
        assert (current['pid'], current['start_ticks']) == (before['pid'], before['start_ticks'])
        if current['phase'] == target:
            return current
        time.sleep(.25)
    raise RuntimeError(('holder transition timeout', target, current))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lease-dir', type=Path, required=True)
    parser.add_argument('--index', type=int, choices=GPUS, required=True)
    parser.add_argument('action', choices=('status', 'idle', 'run'))
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    lease = args.lease_dir.resolve()
    if args.action == 'status':
        print(json.dumps(status(lease, args.index), indent=2))
        return
    with (lease / f'gpu{args.index}.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'idle':
            print(json.dumps(phase(lease, args.index, 'idle'), indent=2))
            return
        command = args.command
        if command[:1] == ['--']:
            command = command[1:]
        assert command, 'run requires a command'
        holder = status(lease, args.index)
        for row in gpu_processes(GPUS[args.index]):
            if row['pid'] != holder['pid']:
                assert admitted_process(lease, args.index, row['pid'], row['memory_mib']), ('unadmitted process', row)
        owner = {'pid': os.getpid(), 'start_ticks': identity(os.getpid())}
        child = None
        def stop(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            phase(lease, args.index, 'active', owner)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[args.index],
                       RILEY_GPU_HOLDER_PID=str(holder['pid']),
                       RILEY_GPU_HOLDER_START_TICKS=holder['start_ticks'],
                       RILEY_GPU_HOLDER_LEASE_DIR=str(lease))
            child = subprocess.Popen(command, env=env, start_new_session=True)
            result = child.wait()
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    # This is the command's own process group, never another GPU job.
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            phase(lease, args.index, 'idle')
        raise SystemExit(result)


if __name__ == '__main__':
    main()
