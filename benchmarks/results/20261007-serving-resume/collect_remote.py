"""Read-only recovery inventory; emits hashes, never environment credentials."""
import datetime
import hashlib
import json
import pathlib
import subprocess

root = pathlib.Path('/data/riley-serving-260913-recovery')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def command(argv):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=45)
    return {'exit': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

def record(p):
    return {'path': str(p), 'exists': p.exists(),
            **({'bytes': p.stat().st_size, 'sha256': sha(p)} if p.is_file() else {})}

result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
result['host'] = {k: command(v) for k, v in {
    'gpu': ['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu', '--format=csv'],
    'gpu_processes': ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'],
    'uptime': ['uptime'],
    'disk': ['df', '-h', '/data'],
}.items()}
result['pressure'] = {k: pathlib.Path('/proc/pressure', k).read_text() for k in ['cpu', 'io', 'memory']}
result['qwen_checkout'] = {k: command(['git', '-C', '/data/riley-serving-260915-n01', *args])
                           for k, args in {'revision': ['rev-parse', 'HEAD'], 'status': ['status', '--porcelain=v1']}.items()}
result['recovery_source'] = {str(p.relative_to(root / 'source')): record(p)
                           for p in (root / 'source').rglob('*')
                           if p.is_file() and not any(x.startswith('._') or x in ['target', '__pycache__'] for x in p.relative_to(root / 'source').parts)}
result['binaries'] = [record(p) for p in [root/'target/release/riley', *[root/f'variable-candidate-v{v}/riley' for v in [51,52,56]]]]
result['model'] = [record(p) for p in (root/'runtime-assets-20260915/model').iterdir() if p.is_file()]
result['model_manifest'] = json.loads((root/'runtime-assets-20260915/model/riley-checkpoint.json').read_text())
result['native_libraries'] = [record(root/'toolchain130/nvidia/cu13/lib'/n)
                            for n in ['libcublasLt.so.13','libcublas.so.13','libcudart.so.13']]
result['vllm029'] = command([str(root/'vllm029-venv/bin/python'), '-c',
    "import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ['vllm','torch','transformers','flashinfer-python']}))"])
result['legacy_paths'] = {p: pathlib.Path(p).exists() for p in [
    '/tmp/riley-opt-260912', '/data/riley-vllm-interim.CfrT9T/venv/bin/vllm',
    '/data/riley-benchmark/20260827T051948Z-d7ad713a/model']}
result['completion_files'] = [str(p) for p in root.glob('*/completion.json')]
print(json.dumps(result, indent=2))
