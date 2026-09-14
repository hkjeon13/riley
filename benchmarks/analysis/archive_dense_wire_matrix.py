"""Export completed matrix v2 only, after all timed lanes and restorations finish.

Run locally after confirming the orchestrator process has exited. Remote terminal
receipts are checked again before any compression. Never overwrites an export.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/data/riley-serving-260913-recovery'
SOURCES = [
    'crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs',
    'crates/riley-runtime/src/llama/multi_descriptor/future_token.rs',
    'crates/riley-runtime/src/llama/variable_session.rs',
    'benchmarks/analysis/dense_wire_serving_screen.py',
    'benchmarks/analysis/paired_decode_serving_screen.py',
    'benchmarks/analysis/serving_evidence_validation.py',
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def members(path):
    result = {}
    with tarfile.open(path) as archive:
        for entry in archive:
            if entry.isfile():
                assert entry.name not in result
                h = hashlib.sha256()
                stream = archive.extractfile(entry)
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
                result[entry.name] = h.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--concurrency', type=int, nargs='+', choices=[8,16,64], default=[8,16,64])
    parser.add_argument('--version', type=int, choices=[2,3], default=2)
    args = parser.parse_args()
    version = args.version
    destinations = {c: ROOT / f'benchmarks/results/20260914-dense-wire-matrix-c{c}' for c in args.concurrency}
    assert not any(path.exists() for path in destinations.values()), 'export already exists; inspect before retry'
    # Read-only preflight. No matrix or serving controller may remain live.
    preflight = '''from pathlib import Path
import json,hashlib
r=Path(ROOT)
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try: argv=proc.joinpath('cmdline').read_bytes().split(b'\\0')
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 assert not any(Path(a.decode(errors='replace')).name.startswith('dense_wire_matrix') or a.endswith(b'/dense_wire_serving_screen.py') for a in argv), 'matrix process still live'
for c in CONDITIONS:
 p=Path(f'/dev/shm/riley-dense-wire-matrix-c{c}-vVERSION')
 life=r/f'dense-wire-matrix-c{c}-lifecycle-vVERSION'
 assert json.loads((p/'complete.json').read_text())=={'lanes':12,'all_complete':True}
 assert json.loads((life/'execution.json').read_text())==[{'name':'serving','exit_code':0}]
 assert json.loads((life/'blender-restored.json').read_text())['restored']
 assert (life/'complete.txt').read_text()=='complete\\n'
print(json.dumps({s:hashlib.sha256((r/'source'/s).read_bytes()).hexdigest() for s in SOURCES}))
'''.replace('ROOT', repr(REMOTE)).replace('SOURCES', repr(SOURCES[:5])).replace('CONDITIONS', repr(args.concurrency)).replace('VERSION', str(version))
    hashes = json.loads(subprocess.check_output(['ssh', 'ai-assistant', 'python3', '-'], input=preflight, text=True))
    assert hashes == {s: digest(ROOT / s) for s in SOURCES[:5]}, 'source changed since remote deployment'
    packages = subprocess.check_output(['ssh', 'ai-assistant', REMOTE + '/vllm029-venv/bin/python', '-m', 'pip', 'freeze'])
    assert b'vllm==0.29.0' in packages.splitlines()
    for concurrency, directory in destinations.items():
        remote_archive = f'{REMOTE}/dense-wire-matrix-c{concurrency}-v{version}-evidence.tar.gz'
        code = '''from pathlib import Path
import tarfile
r=Path(ROOT)
paths={'serving':Path(SPOOL),'lifecycle':r/LIFECYCLE,'build':r/'dense-wire-serving-build-v2'}
with tarfile.open(ARCHIVE,'x:gz') as archive:
 for alias,path in paths.items():
  for f in sorted(path.rglob('*')):
   if f.is_file() and f.suffix in ('.json','.jsonl','.log','.txt','.py'):
    archive.add(f,arcname=alias+'/'+str(f.relative_to(path)))
'''.replace('ROOT', repr(REMOTE)).replace('SPOOL', repr(f'/dev/shm/riley-dense-wire-matrix-c{concurrency}-v{version}')).replace('LIFECYCLE', repr(f'dense-wire-matrix-c{concurrency}-lifecycle-v{version}')).replace('ARCHIVE', repr(remote_archive))
        subprocess.run(['ssh', 'ai-assistant', 'python3', '-'], input=code, text=True, check=True)
        directory.mkdir()
        subprocess.run(['scp', 'ai-assistant:' + remote_archive, str(directory / 'evidence.tar.gz')], check=True)
        (directory / 'vllm029-packages.txt').write_bytes(packages)
        with tarfile.open(directory / 'source-snapshot.tar.gz', 'x:gz') as archive:
            for source in SOURCES:
                archive.add(ROOT / source, arcname=source)
        manifest = {
            'archive_sha256': digest(directory / 'evidence.tar.gz'),
            'files': members(directory / 'evidence.tar.gz'),
            'source_snapshot_sha256': digest(directory / 'source-snapshot.tar.gz'),
            'sources': {s: digest(ROOT / s) for s in SOURCES},
            'remote_source_hashes': hashes,
            'vllm029_packages_sha256': digest(directory / 'vllm029-packages.txt'),
        }
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        subprocess.run(['python3', str(ROOT / 'benchmarks/analysis/verify_dense_wire_matrix.py'), str(directory), '--concurrency', str(concurrency), '--retained', '2048', '--require-cooldown'], check=True)


if __name__ == '__main__':
    main()
