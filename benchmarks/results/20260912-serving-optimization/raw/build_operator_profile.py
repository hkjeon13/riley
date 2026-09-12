"""Build operation diagnostics only after retained batch4 measurements finish."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/tmp/riley-opt-260912')
SOURCE = ROOT / 'operator-profile-source'
TARGET = ROOT / 'operator-profile-target'
TOOL = ROOT / 'operator-tools/profile_decode_operators.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


for label in ['batch3-http', 'batch3-engine', 'batch4-http', 'batch4-engine']:
    assert json.loads((ROOT / label / 'summary.json').read_text())['completed']
assert json.loads((ROOT / 'blender-round3/verified.json').read_text())['alive_and_listening']
build = json.loads((ROOT / 'batch4-build.json').read_text())
base = ROOT / 'batch4-source'
assert subprocess.check_output(['git', '-C', str(base), 'rev-parse', 'HEAD'], text=True).strip() == build['source_commit']
assert not subprocess.check_output(['git', '-C', str(base), 'status', '--porcelain'], text=True)
assert all(sha(base / name) == digest for name, digest in build['source_files'].items())
assert all(sha(Path(name)) == digest for name, digest in build['binaries'].items())
run(['git', 'clone', '--no-hardlinks', '--quiet', str(base), str(SOURCE)])
(TARGET / 'release').mkdir(parents=True)
run(['rsync', '-a', '--exclude=riley-cuda-*', '--exclude=libriley_cuda-*',
     str(ROOT / 'batch4-target/release') + '/', str(TARGET / 'release') + '/'])
with (ROOT / 'operator-profile-instrumentation.json').open('x') as log:
    run(['python3', str(TOOL), 'instrument', '--source-root', str(SOURCE), '--projection-events', '--apply'], stdout=log)
env = dict(os.environ, **build['build_environment'])
env['CARGO_TARGET_DIR'] = str(TARGET)
env['PATH'] = '/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:' + env['PATH']
with (ROOT / 'operator-profile-build.log').open('x') as log:
    run(['cargo', 'build', '--release', '-p', 'riley-server', '--features', 'server,bench,cuda',
         '--bin', 'riley'], cwd=SOURCE, env=env, stdout=log, stderr=log)
instrument = json.loads((ROOT / 'operator-profile-instrumentation.json').read_text())
receipt = {
    'base_source_commit': build['source_commit'], 'base_build_sha256': sha(ROOT / 'batch4-build.json'),
    'source_root': str(SOURCE), 'source_clean': False,
    'instrumented_source_sha256': sha(SOURCE / 'kernels/src/graph_resources.cu'),
    'tool_sha256': sha(TOOL), 'historical_tool_sha256': sha(TOOL.with_name('profile_owned_graph.py')),
    'instrumentation': instrument, 'binary': str(TARGET / 'release/riley'),
    'binary_sha256': sha(TARGET / 'release/riley'), 'build_log_sha256': sha(ROOT / 'operator-profile-build.log'),
    'performance_claim_eligible': False, 'execution_started': False,
}
assert receipt['instrumented_source_sha256'] == instrument['instrumented_sha256']
with (ROOT / 'operator-profile-build.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps(receipt), flush=True)
