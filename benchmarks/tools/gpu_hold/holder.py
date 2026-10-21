"""Keep a CUDA context between benchmark arms without running compute kernels."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

GPUS = {
    6: 'GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7',
    7: 'GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0',
}


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def identity(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def gpu_processes(gpu):
    raw = subprocess.check_output([
        'nvidia-smi', '-i', gpu, '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits',
    ], text=True, timeout=15)
    return [{'pid': int(row.split(',')[0]), 'memory_mib': float(row.split(',')[1])}
            for row in raw.splitlines() if row.strip()]


def admitted_process(lease, index, pid, memory_mib=None):
    """Recognize only explicitly authorized identities, including a pinned job tree."""
    policy = json.loads((lease / f'gpu{index}.policy.json').read_text())
    try:
        proc = Path(f'/proc/{pid}')
        if proc.stat().st_uid != os.getuid():
            return False
        tick = identity(pid)
        if any(int(row['pid']) == pid and str(row['start_ticks']) == tick
               for row in policy.get('admitted_processes', [])):
            return True
        for job in policy.get('admitted_jobs', []):
            if memory_mib is not None and memory_mib > job['max_memory_mib']:
                continue
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
            if len(argv) < 2:
                continue
            script = Path(argv[1].decode())
            exact_script = job.get('script') == str(script)
            prediction_script = (job.get('script_directory') is not None
                                 and script.resolve().parent == Path(job['script_directory']).resolve()
                                 and script.name.endswith('_predict.py'))
            if not (job.get('allow_descendants') or exact_script or prediction_script):
                continue
            current = pid
            for _ in range(64):
                if current == job['root_pid']:
                    return identity(current) == str(job['root_start_ticks'])
                if current <= 1:
                    break
                lines = Path(f'/proc/{current}/status').read_text().splitlines()
                current = int(next(row.split()[1] for row in lines if row.startswith('PPid:')))
    except (FileNotFoundError, ProcessLookupError):
        return False
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, choices=GPUS, required=True)
    parser.add_argument('--lease-dir', type=Path, required=True)
    args = parser.parse_args()
    gpu = GPUS[args.index]
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == gpu
    lease = args.lease_dir.resolve()
    lease.mkdir(parents=True, exist_ok=True)
    prefix = lease / f'gpu{args.index}'
    status = prefix.with_suffix('.status.json')
    desired = prefix.with_suffix('.desired.json')
    for row in gpu_processes(gpu):
        assert admitted_process(lease, args.index, row['pid'], row['memory_mib']), ('GPU occupied', row)
    import torch
    assert torch.cuda.get_device_capability() == (8, 0)
    pid = os.getpid()
    start_ticks = identity(pid)
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    allocation = None
    phase = None
    try:
        while not stopping:
            command = json.loads(desired.read_text()) if desired.exists() else {'phase': 'idle'}
            target = command['phase']
            assert target in ('idle', 'active', 'shutdown')
            if target == 'shutdown':
                break
            # A crashed controller must not leave the idle reservation reduced forever.
            owner = command.get('owner')
            if target == 'active' and owner:
                try:
                    alive = identity(owner['pid']) == str(owner['start_ticks'])
                except (FileNotFoundError, ProcessLookupError):
                    alive = False
                if not alive:
                    target = 'idle'
            others = [row for row in gpu_processes(gpu) if row['pid'] != pid]
            if target != phase:
                allocation = None
                torch.cuda.empty_cache()
                size = 20 * 1024**3 if target == 'idle' else 64 * 1024**2
                try:
                    allocation = torch.empty(size, dtype=torch.uint8, device='cuda')
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    allocation = torch.empty(64 * 1024**2, dtype=torch.uint8, device='cuda')
                    target = 'active'
                phase = target
            atomic_json(status, {
                'status': 'running', 'phase': phase, 'pid': pid,
                'start_ticks': start_ticks, 'gpu': gpu, 'physical_index': args.index,
                'allocation_bytes': allocation.numel(),
                'torch_reserved_bytes': torch.cuda.memory_reserved(),
                'updated_ns': time.time_ns(), 'other_processes': others,
                'policy': 'idle20GiB/active64MiB; persistent CUDA context; no compute spin',
            })
            # Retry an unsuccessful idle allocation; never kill another process.
            if phase == 'active' and command['phase'] == 'idle':
                phase = None
            time.sleep(.5)
    finally:
        allocation = None
        torch.cuda.empty_cache()
        atomic_json(status, {'status': 'stopped', 'pid': pid, 'start_ticks': start_ticks,
                            'gpu': gpu, 'physical_index': args.index, 'updated_ns': time.time_ns()})


if __name__ == '__main__':
    main()
