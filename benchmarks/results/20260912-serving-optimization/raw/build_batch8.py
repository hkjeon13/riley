"""Build an immutable RoPE/KV/attention candidate over frozen optional-token batch8 API."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/tmp/riley-opt-260912')
SOURCE = ROOT / 'batch8-source'
TARGET = ROOT / 'batch8-target'
FILES = {
    'crates/riley-runtime/src/llama/graph_decode_full.rs',
    'kernels/src/graph_resources.cu',
    'kernels/src/graph_numerics.cu',
    'kernels/src/ffi_internal.hpp',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


build = json.loads((ROOT / 'http-token-build.json').read_text())
assert build['source_commit'] == 'a179617070526068b66ba5627ba82a7151da8c64'
assert sha(ROOT / 'batch8-source-overlay.json') == 'e542bd579525a37576905d61e0361ce50fb2952bc688724a14df8872f1570e1f'
base = ROOT / 'http-token-source'
assert git(base, 'rev-parse', 'HEAD') == build['source_commit']
assert not git(base, 'status', '--porcelain', '--untracked-files=all')
assert all(sha(base / name) == digest for name, digest in build['source_files'].items())
for name, digest in build['binaries'].items():
    assert sha(Path(name)) == digest
overlay = json.loads((ROOT / 'batch8-source-overlay.json').read_text())
assert set(overlay) == FILES
assert not SOURCE.exists() and not TARGET.exists()
assert not (ROOT / 'batch8-build.log').exists() and not (ROOT / 'batch8-build.json').exists()
run(['git', 'clone', '--no-hardlinks', '--quiet', str(base), str(SOURCE)])
run(['tar', '-xzf', str(ROOT / 'batch8-source-overlay.tar.gz'), '-C', str(SOURCE)])
assert all(sha(SOURCE / name) == digest for name, digest in overlay.items())
for name, digest in build['source_files'].items():
    if name not in FILES:
        assert sha(SOURCE / name) == digest
run(['git', 'add', '--', *sorted(FILES)], cwd=SOURCE)
assert set(git(SOURCE, 'diff', '--cached', '--name-only').splitlines()) == FILES
run(['git', '-c', 'user.name=Codex', '-c', 'user.email=codex@local', 'commit', '--quiet',
     '-m', 'snapshot: fuse exact RoPE KV and two-warp decode attention'], cwd=SOURCE)
assert not git(SOURCE, 'status', '--porcelain', '--untracked-files=all')
(TARGET / 'release').mkdir(parents=True)
run(['rsync', '-a', '--exclude=riley-cuda-*', '--exclude=libriley_cuda-*',
     str(ROOT / 'http-token-target/release') + '/', str(TARGET / 'release') + '/'])
env = dict(os.environ)
fixed = {'CUDA_HOME': '/data/riley-g04-cuda13', 'CUDAToolkit_ROOT': '/data/riley-g04-cuda13',
         'CMAKE': '/data/cmake-3.31.12/bin/cmake', 'CMAKE_BUILD_PARALLEL_LEVEL': '4',
         'CARGO_BUILD_JOBS': '4', 'CARGO_TARGET_DIR': str(TARGET),
         'LD_LIBRARY_PATH': '/data/riley-g04-cuda13/lib'}
env.update(fixed)
env['PATH'] = '/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:' + env['PATH']
commands = [
    ['cargo', 'test', '--release', '-p', 'riley-runtime', '--features', 'cuda', '--lib', '--no-run'],
    ['cargo', 'build', '--release', '-p', 'riley-server', '--features', 'server,bench,cuda',
     '--bin', 'riley', '--bin', 'riley-profile'],
]
with (ROOT / 'batch8-build.log').open('x') as log:
    for command in commands:
        run(command, cwd=SOURCE, env=env, stdout=log, stderr=log)
assert all(sha(SOURCE / name) == digest for name, digest in overlay.items())
assert not git(SOURCE, 'status', '--porcelain', '--untracked-files=all')
receipt = {
    'source_commit': git(SOURCE, 'rev-parse', 'HEAD'), 'source_root': str(SOURCE),
    'parent_source_commit': build['source_commit'], 'parent_build_sha256': sha(ROOT / 'http-token-build.json'),
    'source_files': {**build['source_files'], **overlay}, 'changed_files': sorted(FILES),
    'binaries': {str(TARGET / 'release' / name): sha(TARGET / 'release' / name)
                 for name in ['riley', 'riley-profile']},
    'build_argv': commands, 'build_environment': fixed, 'gpu_tests_executed': False,
    'build_log_sha256': sha(ROOT / 'batch8-build.log'), 'performance_measured': False,
}
with (ROOT / 'batch8-build.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps(receipt), flush=True)
