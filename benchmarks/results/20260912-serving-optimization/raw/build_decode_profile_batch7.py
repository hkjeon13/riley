"""Build isolated batch7 decode diagnostics after both campaigns and restoration.

Running this script builds a new binary; it never starts GPU profiling or pauses
Blender. Both instrumentation modes and paths are pinned below.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/tmp/riley-opt-260912')
SOURCE = ROOT / 'decode7-profile-source'
TARGET = ROOT / 'decode7-profile-target'
TOOL = ROOT / 'decode7-operator-tools/profile_decode_operators_batch7.py'
BASE_COMMIT = '1a2be0df01fe49daa4d4db155ad5c44f34ead6df'
BASE_BUILD_SHA = '4d3f65c70aa56beb487754f01ed04e04d36f4c6db64fd3ee17b665561fb25395'
TOOL_SHA = '6cb953a66df57150834e9df38d1a65fa82de976d2af1c583ffa2db3c912509d0'
HISTORICAL_SHA = '9f0f3a526ed1a47c3e6a37e33111a21356ac5b2d58294ff797ec0a40a5091b78'
GRAPH = 'kernels/src/graph_resources.cu'
GRAPH_SHA = '8586726646729333e9bcea17c419b530aeb30e47cc1727189b701d9468d5dad5'
INSTRUMENTED_SHA = '8d616c3cf2649c5bafd3f487af84b19ee870cdd0a7aea02150fd6e009caeccf5'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


def campaign_gate():
    for label in ('batch7-http', 'batch7-engine'):
        summary = json.loads((ROOT / label / 'summary.json').read_text())
        completion = json.loads((ROOT / label / 'completion.json').read_text())
        assert summary['completed'] is True and summary['source_commit'] == BASE_COMMIT
        assert summary['pairs'] == 5 and len(summary['pair_results']) == 5
        assert completion == {'completed': True, 'processes': 10}
    restored = json.loads((ROOT / 'blender-round10/verified.json').read_text())
    assert restored['alive_and_listening'] is True
    assert restored['commands_and_gui_environment_match'] is True
    assert len(restored['processes']) == 3


def verify_base(build):
    base = ROOT / 'batch7-source'
    assert sha(ROOT / 'batch7-build.json') == BASE_BUILD_SHA
    assert build['source_root'] == str(base) and build['source_commit'] == BASE_COMMIT
    assert subprocess.check_output(['git', '-C', str(base), 'rev-parse', 'HEAD'], text=True).strip() == BASE_COMMIT
    assert not subprocess.check_output(['git', '-C', str(base), 'status', '--porcelain', '--untracked-files=all'], text=True)
    assert build['source_files'][GRAPH] == GRAPH_SHA
    assert all(sha(base / name) == digest for name, digest in build['source_files'].items())
    assert len(build['binaries']) == 2
    assert all(sha(name) == digest for name, digest in build['binaries'].items())
    assert sha(TOOL) == TOOL_SHA and sha(TOOL.with_name('profile_owned_graph.py')) == HISTORICAL_SHA


def verify_instrumented(build):
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == BASE_COMMIT
    dirty = subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain', '--untracked-files=all'], text=True)
    assert dirty == ' M ' + GRAPH + '\n', 'only graph_resources.cu may be instrumented'
    for name, digest in build['source_files'].items():
        assert sha(SOURCE / name) == (INSTRUMENTED_SHA if name == GRAPH else digest)


def main():
    campaign_gate()
    build = json.loads((ROOT / 'batch7-build.json').read_text())
    verify_base(build)
    outputs = [SOURCE, TARGET, ROOT / 'decode7-profile-instrumentation.json',
               ROOT / 'decode7-profile-build.log', ROOT / 'decode7-profile-build.json']
    assert not any(path.exists() for path in outputs), 'diagnostic outputs must be fresh'
    run(['git', 'clone', '--no-hardlinks', '--quiet', str(ROOT / 'batch7-source'), str(SOURCE)])
    (TARGET / 'release').mkdir(parents=True)
    run(['rsync', '-a', '--exclude=riley-cuda-*', '--exclude=libriley_cuda-*',
         str(ROOT / 'batch7-target/release') + '/', str(TARGET / 'release') + '/'])
    with (ROOT / 'decode7-profile-instrumentation.json').open('x') as log:
        run(['python3', str(TOOL), 'instrument', '--source-root', str(SOURCE), '--apply'], stdout=log)
    instrument = json.loads((ROOT / 'decode7-profile-instrumentation.json').read_text())
    assert instrument['original_sha256'] == GRAPH_SHA and instrument['instrumented_sha256'] == INSTRUMENTED_SHA
    assert instrument['tool_sha256'] == TOOL_SHA and instrument['historical_tool_sha256'] == HISTORICAL_SHA
    assert instrument['applied'] is True and instrument['projection_events'] is False
    assert instrument['synchronizations_added'] == 0 and instrument['performance_claim_eligible'] is False
    verify_instrumented(build)
    env = dict(os.environ, **build['build_environment'])
    env['CARGO_TARGET_DIR'] = str(TARGET)
    env['PATH'] = '/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:' + env['PATH']
    argv = ['cargo', 'build', '--release', '-p', 'riley-server', '--features', 'server,bench,cuda', '--bin', 'riley']
    with (ROOT / 'decode7-profile-build.log').open('x') as log:
        run(argv, cwd=SOURCE, env=env, stdout=log, stderr=log)
    verify_base(build)
    verify_instrumented(build)
    campaign_gate()
    receipt = {
        'base_source_commit': BASE_COMMIT, 'base_build_sha256': BASE_BUILD_SHA,
        'source_root': str(SOURCE), 'source_clean': False,
        'instrumented_source_sha256': sha(SOURCE / GRAPH),
        'tool_sha256': TOOL_SHA, 'historical_tool_sha256': HISTORICAL_SHA,
        'instrumentation': instrument, 'binary': str(TARGET / 'release/riley'),
        'binary_sha256': sha(TARGET / 'release/riley'),
        'build_log_sha256': sha(ROOT / 'decode7-profile-build.log'),
        'builder_sha256': sha(__file__), 'build_argv': argv,
        'build_environment': dict(build['build_environment'], CARGO_TARGET_DIR=str(TARGET)),
        'performance_claim_eligible': False, 'execution_started': False,
    }
    with (ROOT / 'decode7-profile-build.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
