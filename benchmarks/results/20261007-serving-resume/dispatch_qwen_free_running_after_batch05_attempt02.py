#!/usr/bin/env python3
"""Wait for actual Batch05 terminal replay, then run/collect separate Qwen gate."""
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'qwen-free-running-dispatch-attempt02'
OUT.mkdir()
REMOTE = '/data/riley-serving-261007'
SCOPE = 'qwen-free-running-validation-attempt01'
UPLOADS = ['qwen-free-running-source-attempt01.tar', 'qwen-free-running-commit-attempt01.pack',
           'qwen-free-running-source-receipt-attempt01.json', 'qwen-free-running-plan-attempt01.json',
           'qwen-free-running-preparation-receipt-attempt01.json', 'run_qwen_free_running_attempt01.py']
PIN_NAMES = UPLOADS + ['materialize_qwen_m2_source.py', 'verify_qwen_free_running_attempt01.py', 'verify_kernel_batch05_attempt02.py', 'audit_kernel_batch05_serving_contract_attempt02.py', 'reconcile_kernel_batch05_source_metadata.py', 'kernel-batch05-source-metadata-reconciliation-attempt01.json', 'kernel-batch05-terminal-source-binary-reverification-attempt01.json']

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def remote_python(code, **kwargs):
    return subprocess.run(['ssh', 'ai-assistant', 'python3 -c ' + shlex.quote(code)], **kwargs)

def read_remote(code):
    return json.loads(remote_python(code, check=True, capture_output=True, text=True).stdout)

pins = {name: sha(ROOT / name) for name in PIN_NAMES}
write('preparation.json', {'created_ns': time.time_ns(), 'script_sha256': sha(pathlib.Path(__file__)),
                           'input_pins': pins, 'wait_for': 'Batch05 collection+replay terminal; actual controller absent; GPU idle',
                           'serving_performance': '미실행', 'goal_achieved': False})
failure = None
stage = 'waiting-for-Batch05-terminal'
try:
    deadline = time.monotonic() + 18000
    terminals = [ROOT / 'kernel-batch05-terminal-collection-attempt01/completion.json',
                 ROOT / 'kernel-batch05-independent-replay-attempt02/completion.json']
    while not all(path.exists() for path in terminals):
        write('wait-state.json', {'time_ns': time.time_ns(), 'terminals_present': [path.exists() for path in terminals],
                                  'remote_upload_executed': False, 'Qwen_GPU_execution': '미실행'})
        if time.monotonic() > deadline:
            raise RuntimeError('Batch05 terminal collection/replay absent at observation deadline; no restart inferred')
        time.sleep(30)
    collection, replay = [json.loads(path.read_text()) for path in terminals]
    assert collection['failure'] is None and replay['failure'] is None
    assert len(replay['records']) == 1 and replay['records'][0]['failure'] is None, 'full96 independent replay failed; inspect preserved replay logs'
    steps = replay['records'][0]['steps']
    assert [step['name'] for step in steps] == ['raw-replay', 'all8-summary', 'frozen-launch-audit']
    assert all(step['exit'] == 0 for step in steps)
    write('Batch05-terminal-prerequisites.json', {'collection': collection, 'replay': replay,
                                                'performance_adoption_inferred': False})
    assert all(sha(ROOT / name) == digest for name, digest in pins.items()), 'prepared inputs changed'
    stage = 'remote-preflight'
    state = read_remote("""import hashlib,json,subprocess
from pathlib import Path
r=Path('/data/riley-serving-261007')
c=json.loads((r/'kernel-batch05-quiet-attempt01/completion.json').read_text())
print(json.dumps({'completion':c,'controller_pid_exists':Path('/proc/2231583').exists(),
 'gpu_compute_pids':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),
 'materializer_sha256':hashlib.sha256((r/'materialize_qwen_m2_source.py').read_bytes()).hexdigest()}))
""")
    assert state['completion']['failure'] is None and state['completion']['all_lanes_complete']
    assert len(state['completion']['records']) == 96
    assert not state['controller_pid_exists'] and not state['gpu_compute_pids'], state
    assert state['materializer_sha256'] == pins['materialize_qwen_m2_source.py']
    write('remote-preflight.json', state)
    stage = 'create-only-upload'
    for name in UPLOADS:
        code = "import hashlib,sys;from pathlib import Path;p=Path(" + repr(REMOTE + '/' + name) + ");b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()==" + repr(pins[name]) + ";f=p.open('xb');f.write(b);f.flush();f.close()"
        with (ROOT / name).open('rb') as f:
            remote_python(code, stdin=f, check=True)
    write('upload-receipt.json', {'uploaded_sha256': {name: pins[name] for name in UPLOADS}, 'create_only': True})
    stage = 'actual-native-build-and-run'
    cmd = ['ssh', 'ai-assistant', '/usr/bin/python3', REMOTE + '/run_qwen_free_running_attempt01.py']
    write('launch.json', {'argv': cmd, 'time_ns': time.time_ns(), 'serving_performance': '미실행'})
    with (OUT / 'ssh-native.log').open('x') as log:
        process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    write('remote-process.json', {'exit': process.returncode, 'log_sha256': sha(OUT / 'ssh-native.log')})
    stage = 'terminal-evidence-collection'
    inventory_code = """import hashlib,json,subprocess
from pathlib import Path
r=Path('/data/riley-serving-261007/qwen-free-running-validation-attempt01')
assert (r/'completion.json').exists(),'actual native controller terminal absent'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied; archive IO deferred'
files={}
for p in sorted(r.rglob('*')):
 assert not p.is_symlink(),'unexpected symlink in evidence'
 if p.is_file():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
  files[str(p.relative_to(r))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
print(json.dumps(files))
"""
    before = read_remote(inventory_code)
    write('source-manifest-before.json', before)
    archive = OUT / (SCOPE + '.tar.gz')
    with archive.open('xb') as f:
        subprocess.run(['ssh', 'ai-assistant', 'tar -C ' + shlex.quote(REMOTE + '/' + SCOPE) + ' -cf - . | gzip -1'],
                       stdout=f, check=True)
    after = read_remote(inventory_code)
    write('source-manifest-after.json', after)
    assert before == after, 'terminal evidence changed during collection'
    dest = OUT / SCOPE
    dest.mkdir()
    observed = {}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = member.name.removeprefix('./')
            assert not pathlib.Path(name).is_absolute() and '..' not in pathlib.Path(name).parts
            if member.isdir():
                continue
            assert member.isfile() and name in before and name not in observed, name
            path = dest / name
            path.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with tar.extractfile(member) as src, path.open('xb') as out:
                for chunk in iter(lambda: src.read(1024 * 1024), b''):
                    h.update(chunk)
                    out.write(chunk)
            observed[name] = {'bytes': member.size, 'sha256': h.hexdigest()}
    assert observed == before
    write('collection-receipt.json', {'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
                                      'files_verified': len(observed), 'source_before_after_identical': True,
                                      'source_preserved': True, 'serving_performance': '미실행'})
    assert process.returncode == 0, 'actual native build/test failed; full raw evidence preserved'
    stage = 'independent-raw-replay'
    assert sha(ROOT / 'verify_qwen_free_running_attempt01.py') == pins['verify_qwen_free_running_attempt01.py']
    command = [sys.executable, str(ROOT / 'verify_qwen_free_running_attempt01.py'), str(dest),
               '--output', str(OUT / 'independent-proof.json')]
    with (OUT / 'independent-replay.log').open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    write('independent-replay-process.json', {'argv': command, 'exit': result.returncode,
                                             'log_sha256': sha(OUT / 'independent-replay.log')})
    assert result.returncode == 0, 'independent raw replay failed; all evidence preserved'
    stage = 'terminal'
except BaseException as error:
    failure = {'type': type(error).__name__, 'message': str(error), 'stage': stage}
    raise
finally:
    write('completion.json', {'failure': failure, 'stage': stage,
                              'serving_performance': '미실행', 'goal_achieved': False})
