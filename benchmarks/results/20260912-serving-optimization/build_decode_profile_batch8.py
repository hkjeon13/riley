"""Build isolated batch8 decode diagnostics only after the Round14 token campaign stops and restores Blender.

Running this script builds a new binary; it never starts GPU profiling or pauses
Blender. Decode-only instrumentation and paths are pinned below; M16 prefill projection events are unsupported.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/tmp/riley-opt-260912')
SOURCE = ROOT / 'decode8-profile-source'
TARGET = ROOT / 'decode8-profile-target'
TOOL = ROOT / 'decode8-operator-tools/profile_decode_operators_batch8.py'
BASE_COMMIT = '8329c1aeec6e013f581128888c536e15f8bf7300'
BASE_BUILD_SHA = '629418e19f129436d9ee2d2a753a61317c5b8527e3f4fe438b9f9aa59a8392c6'
TOOL_SHA = '139577ccb86d14611b91b005bc4216389021fcf48c1ce70508301267cfb674d6'
HISTORICAL_SHA = '9f0f3a526ed1a47c3e6a37e33111a21356ac5b2d58294ff797ec0a40a5091b78'
GRAPH = 'kernels/src/graph_resources.cu'
GRAPH_SHA = 'bb978cc6557be27b844d356ba38cf462b37934dc283e8a89f2df8a3f3a0b132f'
INSTRUMENTED_SHA = '604b99c668729667cd39ef3d42f3fcd5c0dfd8088cc8f3277ae4d6f85b27f372'
CAMPAIGN_PLAN_SHA = '199d70ebf61e03d3db00ed557b454632daf2aad934ae56d6c4e027b37de7ed7b'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


def campaign_gate():
    # A failed campaign may still need diagnosis. This gate establishes stopped
    # serving and completed restoration, not performance or candidate acceptance.
    plan = ROOT / 'token-serving-round14-plan.json'
    assert sha(plan) == CAMPAIGN_PLAN_SHA
    directory = ROOT / 'token-serving-round14'
    preparation = json.loads((directory / 'preparation.json').read_text())
    assert preparation['plan'] == {'path': str(plan), 'sha256': CAMPAIGN_PLAN_SHA}
    final = json.loads((directory / 'finalization.json').read_text())
    ref = final['restoration']
    assert ref and ref['path'] == str(ROOT / 'blender-round14/verified.json')
    assert sha(Path(ref['path'])) == ref['sha256']
    restored = json.loads(Path(ref['path']).read_text())
    assert restored['alive_and_listening'] and restored['commands_and_gui_environment_match']
    assert restored['all_relaunched_processes_have_pinned_vendor_maps']
    assert not restored['host_or_global_configuration_modified'] and len(restored['processes']) == 3
    launched = {path.parent for pattern in ('*/*/launch.json', '*/*/process.json')
                for path in directory.glob(pattern)}
    exits = {path.parent for path in directory.glob('*/*/process-exit.json')}
    assert launched, 'no owned measurement process launch receipts'
    assert launched == exits, 'each launched lane must have a process exit receipt'
    cleanup_receipts = {}
    for lane in sorted(launched):
        path = lane / 'process-exit.json'
        cleanup = json.loads(path.read_text())['cleanup']
        assert cleanup and cleanup['cleanup_verified'] and not cleanup['remaining_owned_pids']
        cleanup_receipts[str(path)] = sha(path)
    return {'preparation_sha256': sha(directory/'preparation.json'),
            'finalization_sha256': sha(directory/'finalization.json'), 'restoration': ref,
            'process_exit_receipts': cleanup_receipts,
            'performance_or_acceptance_inferred': False}


def verify_base(build):
    base = ROOT / 'batch8-source'
    assert sha(ROOT / 'batch8-build.json') == BASE_BUILD_SHA
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
    gate = campaign_gate()
    build = json.loads((ROOT / 'batch8-build.json').read_text())
    verify_base(build)
    outputs = [SOURCE, TARGET, ROOT / 'decode8-profile-instrumentation.json',
               ROOT / 'decode8-profile-build.log', ROOT / 'decode8-profile-build.json']
    assert not any(path.exists() for path in outputs), 'diagnostic outputs must be fresh'
    run(['git', 'clone', '--no-hardlinks', '--quiet', str(ROOT / 'batch8-source'), str(SOURCE)])
    (TARGET / 'release').mkdir(parents=True)
    run(['rsync', '-a', '--exclude=riley-cuda-*', '--exclude=libriley_cuda-*',
         str(ROOT / 'batch8-target/release') + '/', str(TARGET / 'release') + '/'])
    with (ROOT / 'decode8-profile-instrumentation.json').open('x') as log:
        run(['python3', str(TOOL), 'instrument', '--source-root', str(SOURCE), '--apply'], stdout=log)
    instrument = json.loads((ROOT / 'decode8-profile-instrumentation.json').read_text())
    assert instrument['original_sha256'] == GRAPH_SHA and instrument['instrumented_sha256'] == INSTRUMENTED_SHA
    assert instrument['tool_sha256'] == TOOL_SHA and instrument['historical_tool_sha256'] == HISTORICAL_SHA
    assert instrument['applied'] is True and instrument['projection_events'] is False
    assert instrument['synchronizations_added'] == 0 and instrument['performance_claim_eligible'] is False
    verify_instrumented(build)
    env = dict(os.environ, **build['build_environment'])
    env['CARGO_TARGET_DIR'] = str(TARGET)
    env['PATH'] = '/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:' + env['PATH']
    argv = ['cargo', 'build', '--release', '-p', 'riley-server', '--features', 'server,bench,cuda', '--bin', 'riley']
    with (ROOT / 'decode8-profile-build.log').open('x') as log:
        run(argv, cwd=SOURCE, env=env, stdout=log, stderr=log)
    verify_base(build)
    verify_instrumented(build)
    assert campaign_gate() == gate, 'measurement finalization changed during build'
    receipt = {
        'measurement_finalization': gate,
        'base_source_commit': BASE_COMMIT, 'base_build_sha256': BASE_BUILD_SHA,
        'source_root': str(SOURCE), 'source_clean': False,
        'instrumented_source_sha256': sha(SOURCE / GRAPH),
        'tool_sha256': TOOL_SHA, 'historical_tool_sha256': HISTORICAL_SHA,
        'instrumentation': instrument, 'binary': str(TARGET / 'release/riley'),
        'binary_sha256': sha(TARGET / 'release/riley'),
        'build_log_sha256': sha(ROOT / 'decode8-profile-build.log'),
        'builder_sha256': sha(__file__), 'build_argv': argv,
        'build_environment': dict(build['build_environment'], CARGO_TARGET_DIR=str(TARGET), PATH=env['PATH']),
        'performance_claim_eligible': False, 'execution_started': False,
    }
    with (ROOT / 'decode8-profile-build.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
