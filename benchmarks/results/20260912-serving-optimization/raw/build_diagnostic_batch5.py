"""Build an isolated instrumented binary; do not execute or benchmark it."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path('/tmp/riley-opt-260912')
source = root / 'diagnostic-batch5-source'
target = root / 'diagnostic-batch5-target'
build = json.loads((root / 'batch5-build.json').read_text())
assert subprocess.check_output(['git', '-C', str(root / 'batch5-source'), 'rev-parse', 'HEAD'], text=True).strip() == build['source_commit']
assert not subprocess.check_output(['git', '-C', str(root / 'batch5-source'), 'status', '--porcelain'], text=True)
subprocess.run(['git', 'clone', '--no-hardlinks', '--quiet', str(root / 'batch5-source'), str(source)], check=True)
(target / 'release').mkdir(parents=True)
subprocess.run(['rsync', '-a', '--exclude=riley-cuda-*', '--exclude=libriley_cuda-*',
                str(root / 'batch5-target/release') + '/', str(target / 'release') + '/'], check=True)
with (root / 'diagnostic-batch5-instrumentation.json').open('x') as log:
    subprocess.run(['python3', str(root / 'profile_owned_graph_batch2.py'), 'instrument',
                    '--source-root', str(source), '--apply'], stdout=log, check=True)
env = dict(os.environ, **build['build_environment'])
env['CARGO_TARGET_DIR'] = str(target)
env['PATH'] = '/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:' + env['PATH']
with (root / 'diagnostic-batch5-build.log').open('x') as log:
    subprocess.run(['cargo', 'build', '--release', '-p', 'riley-server', '--features', 'server,bench,cuda',
                    '--bin', 'riley'], cwd=source, env=env, stdout=log, stderr=log, check=True)
result = {'instrumented': True, 'base_source_commit': build['source_commit'],
          'source_root': str(source), 'source_clean': False,
          'binary_sha256': hashlib.sha256((target / 'release/riley').read_bytes()).hexdigest(),
          'performance_claim_eligible': False, 'execution_started': False}
with (root / 'diagnostic-batch5-build.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(result))
