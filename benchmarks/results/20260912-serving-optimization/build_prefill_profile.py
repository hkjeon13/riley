"""Build operation diagnostics only after retained batch5 measurements finish."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/tmp/riley-opt-260912')
SOURCE = ROOT / 'prefill-profile-source'
TARGET = ROOT / 'prefill-profile-target'
TOOL = ROOT / 'prefill-operator-tools/profile_prefill_operators.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


for label in ['batch5-http', 'batch5-engine']:
    assert json.loads((ROOT / label / 'summary.json').read_text())['completed']
assert json.loads((ROOT / 'blender-round6/verified.json').read_text())['alive_and_listening']
build = json.loads((ROOT / 'batch5-build.json').read_text())
base = ROOT / 'batch5-source'
assert subprocess.check_output(['git', '-C', str(base), 'rev-parse', 'HEAD'], text=True).strip() == build['source_commit']
assert not subprocess.check_output(['git', '-C', str(base), 'status', '--porcelain'], text=True)
assert all(sha(base / name) == digest for name, digest in build['source_files'].items())
assert all(sha(Path(name)) == digest for name, digest in build['binaries'].items())
run(['git', 'clone', '--no-hardlinks', '--quiet', str(base), str(SOURCE)])
(TARGET / 'release').mkdir(parents=True)
run(['rsync', '-a', '--exclude=riley-cuda-*', '--exclude=libriley_cuda-*',
     str(ROOT / 'batch5-target/release') + '/', str(TARGET / 'release') + '/'])
with (ROOT / 'prefill-profile-instrumentation.json').open('x') as log:
    run(['python3', str(TOOL), 'instrument', '--source-root', str(SOURCE), '--apply'], stdout=log)
env = dict(os.environ, **build['build_environment'])
env['CARGO_TARGET_DIR'] = str(TARGET)
env['PATH'] = '/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:' + env['PATH']
with (ROOT / 'prefill-profile-build.log').open('x') as log:
    run(['cargo', 'build', '--release', '-p', 'riley-server', '--features', 'server,bench,cuda',
         '--bin', 'riley'], cwd=SOURCE, env=env, stdout=log, stderr=log)
instrument = json.loads((ROOT / 'prefill-profile-instrumentation.json').read_text())
receipt = {
    'base_source_commit': build['source_commit'], 'base_build_sha256': sha(ROOT / 'batch5-build.json'),
    'source_root': str(SOURCE), 'source_clean': False,
    'instrumented_source_sha256': sha(SOURCE / 'kernels/src/graph_resources.cu'),
    'tool_sha256': sha(TOOL), 'historical_tool_sha256': sha(TOOL.with_name('profile_owned_graph.py')),
    'instrumentation': instrument, 'binary': str(TARGET / 'release/riley'),
    'binary_sha256': sha(TARGET / 'release/riley'), 'build_log_sha256': sha(ROOT / 'prefill-profile-build.log'),
    'performance_claim_eligible': False, 'execution_started': False,
}
assert receipt['instrumented_source_sha256'] == instrument['instrumented_sha256']
with (ROOT / 'prefill-profile-build.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps(receipt), flush=True)
