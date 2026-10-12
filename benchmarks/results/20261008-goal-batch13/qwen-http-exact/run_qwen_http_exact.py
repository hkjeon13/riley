"""Run the Qwen HTTP HF-exact gate against the pinned references. Correctness only."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
R = Path('/data/riley-serving-261007'); attempt = int(sys.argv[1]); O = R / f'qwen-http-exact-validation-attempt{attempt:02d}'; O.mkdir()
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
gpu = lambda: subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
S = R / 'qwen-http-exact-source-attempt01'; T = R / 'qwen-http-exact-target-attempt01'
base = Path('/data/riley-benchmarks/20260915T134348Z-n06a-shared-host'); teacher = base / 'qwen3b-hf-eager-teacher-forced-r1-20260916T002940Z'
model = Path('/data/riley-models/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1')
failure = None
try:
    assert not gpu(), 'GPU occupied'
    files = {str(p.relative_to(S)): sha(p) for p in sorted((S / 'crates').rglob('*')) if p.is_file()} | {str(p.relative_to(S)): sha(p) for p in sorted((S / 'kernels').rglob('*')) if p.is_file()}
    result = O / 'qwen3b-http-hf-exact-c1-result.json'
    env = {'HOME': '/home/psyche', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PATH': '/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin',
           'CUDA_HOME': '/data/cuda-12.8.1', 'CUDAToolkit_ROOT': '/data/cuda-12.8.1', 'CUDACXX': '/data/cuda-12.8.1/bin/nvcc', 'CMAKE': '/data/cmake-3.31.12/bin/cmake',
           'CARGO_BUILD_JOBS': '3', 'CMAKE_BUILD_PARALLEL_LEVEL': '3', 'CARGO_TARGET_DIR': str(T), 'RUSTUP_TOOLCHAIN': '1.85.0',
           'LD_LIBRARY_PATH': '/data/cuda-12.8.1/lib64', 'CUDA_VISIBLE_DEVICES': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
           'RILEY_QWEN_SERVING_WORKLOAD': str(base / 'inputs/qwen3b-c8-p2048-o128.json'),
           'RILEY_QWEN_HF_TEACHER_FORCED_ORACLE': str(teacher / 'teacher-forced-oracle.json'),
           'RILEY_QWEN_HF_CACHE_ON_SIDECAR': str(teacher / 'cache-on-logits.safetensors'),
           'RILEY_QWEN3B_CHECKPOINT': str(model), 'RILEY_QWEN_HTTP_HF_EXACT_OUTPUT': str(result)}
    argv = ['/home/psyche/.cargo/bin/cargo', 'test', '--locked', '-p', 'riley-server', '--features', 'server,hf-exact-probe', '--test', 'qwen3b_http_hf_exact_c1_gpu', '--', '--ignored', '--nocapture', '--test-threads=1']
    (O / 'launch.json').write_text(json.dumps({'argv': argv, 'env': env, 'source_commit': (R / 'qwen-http-exact-source.commit').read_text().strip(), 'source_files': len(files),
        'reference_pins': {str(p): sha(p) for p in [base / 'inputs/qwen3b-c8-p2048-o128.json', teacher / 'teacher-forced-oracle.json', teacher / 'cache-on-logits.safetensors', model / 'config.json']},
        'scope': 'HTTP token and logit exactness plus lifecycle; no serving performance'}, indent=1))
    (O / 'source-files.json').write_text(json.dumps(files, indent=1))
    with (O / 'test.log').open('x') as log: p = subprocess.run(argv, cwd=S, env=env, stdout=log, stderr=subprocess.STDOUT)
    after = {str(q.relative_to(S)): sha(q) for q in sorted((S / 'crates').rglob('*')) if q.is_file()} | {str(q.relative_to(S)): sha(q) for q in sorted((S / 'kernels').rglob('*')) if q.is_file()}
    (O / 'process.json').write_text(json.dumps({'exit': p.returncode, 'log_sha256': sha(O / 'test.log'), 'gpu_compute_after': gpu(), 'source_unchanged': after == files, 'result_sha256': sha(result) if result.exists() else None}, indent=1))
    assert p.returncode == 0 and after == files and not gpu(), 'HTTP HF-exact gate failed; evidence retained'
except BaseException as e:
    failure = {'type': type(e).__name__, 'message': str(e)}; raise
finally:
    (O / 'completion.json').write_text(json.dumps({'failure': failure, 'serving_performance': '미실행'}, indent=1) + '\n')
