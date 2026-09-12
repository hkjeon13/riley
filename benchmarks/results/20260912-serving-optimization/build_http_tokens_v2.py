"""Build the isolated optional-token HTTP source over accepted batch7; no GPU execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path('/tmp/riley-opt-260912')
SOURCE = ROOT / 'http-token-source'
TARGET = ROOT / 'http-token-target'
BASE = ROOT / 'batch7-source'
PIN = 'd8a03a06c3bf7f930d044cf082574fb9443ed8a3023b8df7dce495a525610f53'
PARENT_PIN = '4d3f65c70aa56beb487754f01ed04e04d36f4c6db64fd3ee17b665561fb25395'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


def main():
    assert sha(ROOT / 'http-token-source-after-v2.json') == PIN
    assert sha(ROOT / 'batch7-build.json') == PARENT_PIN
    overlay = json.loads((ROOT / 'http-token-source-after-v2.json').read_text())
    parent = json.loads((ROOT / 'batch7-build.json').read_text())
    assert git(BASE, 'rev-parse', 'HEAD') == parent['source_commit']
    assert not git(BASE, 'status', '--porcelain', '--untracked-files=all')
    assert all(sha(BASE / name) == digest for name, digest in parent['source_files'].items())
    assert all(sha(name) == digest for name, digest in parent['binaries'].items())
    files = {row['path']: row for row in overlay['files']}
    assert set(files) == {f'crates/riley-server/src/{name}.rs' for name in ('domain', 'openai', 'engine', 'service')}
    assert all(sha(BASE / name) == row['before_sha256'] for name, row in files.items())
    assert not SOURCE.exists() and not TARGET.exists()
    assert not (ROOT / 'http-token-build.json').exists()
    run(['git', 'clone', '--no-hardlinks', '--quiet', str(BASE), str(SOURCE)])
    archive_path = ROOT / 'http-token-source-overlay-v2.tar.gz'
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        assert {member.name for member in members} == set(files)
        assert len(members) == len(files) and all(member.isfile() for member in members)
        for member in members:
            payload = archive.extractfile(member).read()
            assert hashlib.sha256(payload).hexdigest() == files[member.name]['after_sha256']
            (SOURCE / member.name).write_bytes(payload)
    run(['git', 'add', '--', *sorted(files)], cwd=SOURCE)
    assert set(git(SOURCE, 'diff', '--cached', '--name-only').splitlines()) == set(files)
    run(['git', '-c', 'user.name=Codex', '-c', 'user.email=codex@local', 'commit', '--quiet',
         '-m', 'snapshot: optional committed token observations on HTTP'], cwd=SOURCE)
    assert not git(SOURCE, 'status', '--porcelain', '--untracked-files=all')
    (TARGET / 'release').mkdir(parents=True)
    run(['rsync', '-a', str(ROOT / 'batch7-target/release') + '/', str(TARGET / 'release') + '/'])
    fixed = {'CUDA_HOME': '/data/riley-g04-cuda13', 'CUDAToolkit_ROOT': '/data/riley-g04-cuda13',
             'CMAKE': '/data/cmake-3.31.12/bin/cmake', 'CMAKE_BUILD_PARALLEL_LEVEL': '4',
             'CARGO_BUILD_JOBS': '4', 'CARGO_TARGET_DIR': str(TARGET),
             'LD_LIBRARY_PATH': '/data/riley-g04-cuda13/lib'}
    env = dict(os.environ, **fixed)
    env['PATH'] = '/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:' + env['PATH']
    commands = [
        ['cargo', 'test', '--release', '-p', 'riley-runtime', '--features', 'cuda', '--lib', '--no-run'],
        ['cargo', 'test', '--release', '-p', 'riley-server', '--features', 'server,bench,cuda', '--lib', '--no-run'],
        ['cargo', 'build', '--release', '-p', 'riley-server', '--features', 'server,bench,cuda',
         '--bin', 'riley', '--bin', 'riley-profile'],
    ]
    with (ROOT / 'http-token-build.log').open('x') as log:
        for command in commands:
            run(command, cwd=SOURCE, env=env, stdout=log, stderr=log)
    assert not git(SOURCE, 'status', '--porcelain', '--untracked-files=all')
    assert all(sha(SOURCE / name) == row['after_sha256'] for name, row in files.items())
    assert all(name in files or sha(SOURCE / name) == digest for name, digest in parent['source_files'].items())
    receipt = {'schema': 'riley.http-token-build.v1', 'source_commit': git(SOURCE, 'rev-parse', 'HEAD'),
               'source_root': str(SOURCE), 'parent_source_commit': parent['source_commit'],
               'parent_build_sha256': PARENT_PIN, 'local_source_receipt_sha256': PIN,
               'source_files': {**parent['source_files'], **{name: row['after_sha256'] for name, row in files.items()}},
               'changed_files': sorted(files), 'overlay_sha256': sha(archive_path),
               'binaries': {str(TARGET / 'release' / name): sha(TARGET / 'release' / name)
                            for name in ('riley', 'riley-profile')},
               'build_argv': commands, 'build_environment': fixed,
               'build_log_sha256': sha(ROOT / 'http-token-build.log'),
               'build_helper_sha256': sha(__file__), 'gpu_tests_executed': False,
               'performance_measured': False, 'new_binary_qualified': False,
               'numerical_kernel_sources_unchanged': True}
    with (ROOT / 'http-token-build.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
