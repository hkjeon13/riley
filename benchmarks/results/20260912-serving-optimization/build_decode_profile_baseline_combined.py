"""Build isolated baseline combined RoPE/attention diagnostics only after the Round14 token campaign stops and restores Blender.

Running this script builds a new binary; it never starts GPU profiling or pauses
Blender. Decode-only instrumentation and paths are pinned below; M16 prefill projection events are unsupported.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/tmp/riley-opt-260912')
SOURCE = ROOT / 'decode7-combined-profile-source'
TARGET = ROOT / 'decode7-combined-profile-target'
TOOL = ROOT / 'decode7-combined-operator-tools/profile_decode_rope_attention_baseline.py'
BASE_COMMIT = 'a179617070526068b66ba5627ba82a7151da8c64'
BASE_BUILD_SHA = '5576db88790d5a87e976c082df77f985595129b122fde55b2b475ce06e17a4f5'
TOOL_SHA = '893e292d7c3cb911a2e64c8a93b210df087f52aa102948e6f02b521e1e866ffa'
HISTORICAL_SHA = '9f0f3a526ed1a47c3e6a37e33111a21356ac5b2d58294ff797ec0a40a5091b78'
GRAPH = 'kernels/src/graph_resources.cu'
GRAPH_SHA = '8586726646729333e9bcea17c419b530aeb30e47cc1727189b701d9468d5dad5'
INSTRUMENTED_SHA = '91d473d01047dc8ad6dad475a567b92d3a290e33bb0a7ed20359d719642cd485'
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
    base = ROOT / 'http-token-source'
    assert sha(ROOT / 'http-token-build.json') == BASE_BUILD_SHA
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
    build = json.loads((ROOT / 'http-token-build.json').read_text())
    verify_base(build)
    outputs = [SOURCE, TARGET, ROOT / 'decode7-combined-profile-instrumentation.json',
               ROOT / 'decode7-combined-profile-build.log', ROOT / 'decode7-combined-profile-build.json']
    assert not any(path.exists() for path in outputs), 'diagnostic outputs must be fresh'
    run(['git', 'clone', '--no-hardlinks', '--quiet', str(ROOT / 'http-token-source'), str(SOURCE)])
    (TARGET / 'release').mkdir(parents=True)
    run(['rsync', '-a', '--exclude=riley-cuda-*', '--exclude=libriley_cuda-*',
         str(ROOT / 'http-token-target/release') + '/', str(TARGET / 'release') + '/'])
    with (ROOT / 'decode7-combined-profile-instrumentation.json').open('x') as log:
        run(['python3', str(TOOL), 'instrument', '--source-root', str(SOURCE), '--baseline-build', str(ROOT / 'http-token-build.json'), '--apply'], stdout=log)
    instrument = json.loads((ROOT / 'decode7-combined-profile-instrumentation.json').read_text())
    assert instrument['original_sha256'] == GRAPH_SHA and instrument['instrumented_sha256'] == INSTRUMENTED_SHA
    assert instrument['tool_sha256'] == TOOL_SHA and instrument['historical_tool_sha256'] == HISTORICAL_SHA
    assert instrument['baseline_build'] == {'path': str(ROOT / 'http-token-build.json'), 'sha256': BASE_BUILD_SHA}
    assert instrument['baseline_source_commit'] == BASE_COMMIT
    assert instrument['applied'] is True and instrument['projection_events'] is False
    assert instrument['synchronizations_added'] == 0 and instrument['performance_claim_eligible'] is False
    verify_instrumented(build)
    env = dict(os.environ, **build['build_environment'])
    env['CARGO_TARGET_DIR'] = str(TARGET)
    env['PATH'] = '/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:' + env['PATH']
    argv = ['cargo', 'build', '--release', '-p', 'riley-server', '--features', 'server,bench,cuda', '--bin', 'riley']
    with (ROOT / 'decode7-combined-profile-build.log').open('x') as log:
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
        'build_log_sha256': sha(ROOT / 'decode7-combined-profile-build.log'),
        'builder_sha256': sha(__file__), 'build_argv': argv,
        'build_environment': dict(build['build_environment'], CARGO_TARGET_DIR=str(TARGET), PATH=env['PATH']),
        'performance_claim_eligible': False, 'execution_started': False,
    }
    with (ROOT / 'decode7-combined-profile-build.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
