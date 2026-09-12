"""Pause only the three verified round12 successors of the authorized first round.

State is private to round13. A lock serializes restoration and an incremental
journal lets the watchdog resume a partially completed restore. No SIGKILL.
"""
import argparse
import fcntl
import json
import hashlib
import stat
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid

from remote_session import PIDS, PORTS, identity, live

CAMPAIGN = Path('/tmp/riley-opt-260912')
PREVIOUS = CAMPAIGN / 'blender-round12'
ROOT = CAMPAIGN / 'blender-round13'
SNAPSHOT = ROOT / 'session.json'
TAG_KEY = 'RILEY_BLENDER_RESTORE_SESSION'
GUI_KEYS = {'DISPLAY', 'XAUTHORITY', 'XDG_RUNTIME_DIR',
            'DBUS_SESSION_BUS_ADDRESS', 'WAYLAND_DISPLAY'}


# Child-only runtime proven by fresh EGL/GLX contexts, not a host package change.
COMPUTE = CAMPAIGN / 'driver580173-runtime-20260901'
GUI_RUNTIME = CAMPAIGN / 'driver580173-gui-runtime-20260901'
GUI_PROOF = CAMPAIGN / 'driver-runtime-gui-probe-v2'
RUNTIME_SNAPSHOT = ROOT / 'runtime.json'
RUNTIME_PINS = {
    'driver580173-runtime-20260901/receipt.json': '36dba27991800124660e8f07274ffde21b79ebef14bb365d37a610f9a0802d70',
    'driver580173-gui-runtime-20260901/receipt.json': '091701fb0b91904024748a9baf59891735809ac2ba43e089165e619308f8f7ed',
    'driver-runtime-gui-probe-v2/completion.json': 'ded56de3396c5a60ebc1d6d42227b3330aff674eb85a1394606947960c1f4cfc',
    'driver-runtime-gui-probe-v2/preparation.json': '7f3f5580ddc30813edda371be1b300b652631a60c92bdc24f6fe243f2763d3db',
}
RUNTIME_KEYS = {'PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT',
                '__EGL_VENDOR_LIBRARY_FILENAMES', '__EGL_VENDOR_LIBRARY_DIRS',
                '__GLX_VENDOR_LIBRARY_NAME', '__GLX_FORCE_VENDOR_LIBRARY_0',
                '__EGL_EXTERNAL_PLATFORM_CONFIG_DIRS', '__EGL_EXTERNAL_PLATFORM_CONFIG_FILENAMES',
                '__NV_PRIME_RENDER_OFFLOAD', '__NV_PRIME_RENDER_OFFLOAD_PROVIDER',
                'VK_ICD_FILENAMES', 'VK_DRIVER_FILES'}
VENDOR_PREFIXES = ('libcuda.so', 'libcudadebugger.so', 'libnvidia-', 'libnvoptix.so',
                   'libEGL_nvidia', 'libGLX_nvidia', 'libGLESv1_CM_nvidia', 'libGLESv2_nvidia')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def evidence(path):
    return {'path': str(path), 'sha256': digest(path)}


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            ensure(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def private_root():
    info = ROOT.lstat()
    ensure(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
           and stat.S_IMODE(info.st_mode) == 0o700, 'round13 state directory must be private and owned')


def read_private(path):
    private_root()
    info = path.lstat()
    ensure(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
           and stat.S_IMODE(info.st_mode) == 0o600, 'round13 state file must be private and owned')
    return read(path)


def checked_member(root, relative):
    path = root / relative
    ensure(not Path(relative).is_absolute() and '..' not in Path(relative).parts
           and path.resolve(strict=True).is_relative_to(root.resolve(strict=True)),
           'runtime member escapes extracted tree')
    return path


def validate_runtime():
    """Read-only; hash every pinned dependency before stop/restore, no GPU call."""
    for name, expected in RUNTIME_PINS.items():
        ensure(digest(CAMPAIGN / name) == expected, 'validated runtime receipt changed: ' + name)
    compute, gui = read(COMPUTE / 'receipt.json'), read(GUI_RUNTIME / 'receipt.json')
    complete, prepared = read(GUI_PROOF / 'completion.json'), read(GUI_PROOF / 'preparation.json')
    ensure(compute['completed'] and gui['completed'] and compute['archive_signature_verified']
           and not compute['host_packages_modified'] and not gui['host_packages_modified']
           and not compute['host_restart_performed'] and not gui['host_restart_performed'],
           'private signed runtime preparation was not completed')
    ensure('580.173.02' in compute['kernel_version']
           and Path('/proc/driver/nvidia/version').read_text() == compute['kernel_version'],
           'running NVIDIA kernel differs from the validated private runtime')
    ensure(gui['signed_compute_receipt'] == evidence(COMPUTE / 'receipt.json'), 'GUI runtime compute dependency changed')
    files, links = {}, {}
    for root, receipt in ((COMPUTE, compute), (GUI_RUNTIME, gui)):
        extracted = root / 'extracted'
        ensure(receipt['library_path'] == str(extracted / 'usr/lib/x86_64-linux-gnu'), 'unexpected private library directory')
        ensure(receipt['files'] and receipt['symlinks'], 'runtime inventory is empty')
        for relative, expected in receipt['files'].items():
            path = checked_member(extracted, relative)
            ensure(path.is_file() and not path.is_symlink() and digest(path) == expected,
                   'runtime dependency changed: ' + str(path))
            files[str(path.resolve(strict=True))] = expected
        for relative, target in receipt['symlinks'].items():
            path = checked_member(extracted, relative)
            ensure(path.is_symlink() and str(path.readlink()) == target, 'runtime symlink changed: ' + str(path))
            ensure(str(path.resolve(strict=True)) in files, 'runtime symlink target is not a pinned file')
            links[str(path)] = target
    ensure(compute['nvidia_smi_path'] == str(COMPUTE / 'extracted/usr/bin/nvidia-smi'), 'unexpected private nvidia-smi path')
    vendor = GUI_RUNTIME / 'extracted/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    ensure(read(vendor)['ICD']['library_path'] == 'libEGL_nvidia.so.0', 'unexpected EGL vendor descriptor')
    overrides = complete['child_only_overrides']
    ensure(set(overrides) == {'LD_LIBRARY_PATH', 'PATH', '__EGL_VENDOR_LIBRARY_FILENAMES', '__GLX_VENDOR_LIBRARY_NAME'}
           and overrides['LD_LIBRARY_PATH'] == gui['library_path'] + ':' + compute['library_path']
           and overrides['PATH'].split(':')[0] == str(Path(compute['nvidia_smi_path']).parent)
           and all(part and Path(part).is_absolute() for part in overrides['PATH'].split(':'))
           and overrides['__EGL_VENDOR_LIBRARY_FILENAMES'] == str(vendor)
           and overrides['__GLX_VENDOR_LIBRARY_NAME'] == 'nvidia', 'proven child runtime override changed')
    ensure(complete['completed'] and complete['blender_sessions_unchanged'] and not complete['window_created']
           and not complete['performance_claim'] and prepared['child_only_overrides'] == overrides,
           'fresh EGL/GLX proof is incomplete')
    receipts = [evidence(COMPUTE / 'receipt.json'), evidence(GUI_RUNTIME / 'receipt.json')]
    ensure(complete['runtime_receipts'] == prepared['runtime_receipts'] == receipts, 'GUI probe runtime receipts changed')
    proofs = []
    for context in complete['contexts']:
        item = context['stdout']
        ensure(Path(item['path']).parent == GUI_PROOF and evidence(Path(item['path'])) == item, 'GUI context proof changed')
        observed = read(Path(item['path']))
        ensure(observed['completed'] and observed['cleanup_completed'] and not observed['window_created']
               and observed['gl'] == context['gl']
               and context['gl']['vendor'] == 'NVIDIA Corporation' and '580.173.02' in context['gl']['version'],
               'GUI context runtime proof failed')
        ensure(context['api'] in ('egl', 'glx'), 'unknown GUI context proof')
        # The archived EGL probe predates the richer GLX schema and has no
        # kernel_version field. The live kernel is checked above for both APIs.
        if context['api'] == 'egl':
            ensure(observed['egl_version'] == [1, 5], 'EGL context version differs from the validated proof')
        else:
            ensure(observed['schema_version'] == 'riley.glx-runtime-probe.v1'
                   and observed['glx_version'] == [1, 4]
                   and observed['kernel_version'] == compute['kernel_version'],
                   'GLX schema/version/kernel proof differs')
        proofs.append(item)
    sessions = complete['blender_after']
    ensure(len(sessions) == 3 and [row['port'] for row in sessions] == PORTS
           and prepared['blender_before'] == sessions, 'GUI proof process scope differs')
    for row in sessions:
        ensure({context['api'] for context in complete['contexts']
                if row['pid'] in context['blender_pids_using_this_gui_environment']} == {'egl', 'glx'},
               'a canonical GUI environment lacks both context proofs')
    for helper, expected in prepared['helpers'].items():
        ensure(digest(helper) == expected, 'GUI proof helper changed')
    return {'schema_version': 'riley.round13-private-runtime.v1', 'child_only_overrides': overrides,
            'pinned_receipts': {str(CAMPAIGN / name): expected for name, expected in RUNTIME_PINS.items()},
            'context_proofs': proofs, 'files': files, 'symlinks': links, 'proven_sessions': sessions,
            'kernel_version': compute['kernel_version'],
            'predecessor_inputs': {str(path): digest(path) for path in
                                   (CAMPAIGN / 'blender-session.json', PREVIOUS / 'session.json', PREVIOUS / 'verified.json')},
            'helpers': {str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
                        str(Path(__file__).resolve().with_name('remote_session.py')):
                        digest(Path(__file__).resolve().with_name('remote_session.py'))}}



def validate_snapshot(rows, runtime):
    canonical = read(CAMPAIGN / 'blender-session.json')
    ensure(len(rows) == len(canonical) == len(runtime['proven_sessions']) == 3
           and [row['pid'] for row in canonical] == PIDS, 'snapshot canonical authorization scope changed')
    for row, first, proven in zip(rows, canonical, runtime['proven_sessions']):
        ensure(same_command(row, first) and (row['pid'], row['start'], row['port'], row['env'])
               == (proven['pid'], proven['start'], proven['port'], proven['gui_env']),
               'snapshot process differs from canonical/proven successor')


def bound_runtime():
    expected = read_private(RUNTIME_SNAPSHOT)
    ensure(validate_runtime() == expected, 'round13 runtime dependency/contract changed; no new process launched')
    return expected


def runtime_environment(runtime):
    env = {key: value for key, value in os.environ.items()
           if key not in RUNTIME_KEYS and key != TAG_KEY}
    env.update(runtime['child_only_overrides'])
    return env


def process_runtime_environment(pid, runtime):
    observed = {}
    for raw in (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0'):
        if b'=' in raw:
            key, value = raw.split(b'=', 1)
            key = key.decode()
            if key in RUNTIME_KEYS:
                ensure(key not in observed, 'duplicate runtime key in successor environment')
                observed[key] = value.decode()
    ensure(observed == runtime['child_only_overrides'], 'successor private runtime environment differs')
    return observed


def vendor_lines(pid):
    result = []
    for line in (Path('/proc') / str(pid) / 'maps').read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and Path(fields[5].removesuffix(' (deleted)')).name.startswith(VENDOR_PREFIXES):
            result.append(line)
    return result


def verify_private_maps(pid, runtime):
    """Require actual GL vendor/core loading; validate every mapped vendor file.

    CUDA may be loaded lazily by Blender. Its presence is reported, not forced by
    a synthetic CUDA call. Any mapped compute library must be pinned too.
    """
    lines = vendor_lines(pid)
    files = {}
    for line in lines:
        fields = line.split(None, 5)
        path = Path(fields[5])
        ensure(not fields[5].endswith(' (deleted)') and str(path) in runtime['files'],
               'restored process mapped an unpinned/deleted NVIDIA library: ' + fields[5])
        info = path.stat()
        major, minor = (int(part, 16) for part in fields[3].split(':'))
        ensure(info.st_ino == int(fields[4]) and os.major(info.st_dev) == major and os.minor(info.st_dev) == minor,
               'mapped NVIDIA library inode differs from pinned path')
        if str(path) not in files:
            ensure(digest(path) == runtime['files'][str(path)], 'mapped NVIDIA library content changed')
            files[str(path)] = {'sha256': runtime['files'][str(path)], 'device': fields[3], 'inode': int(fields[4])}
    names = {Path(path).name for path in files}
    ready = (any(name.startswith(('libGLX_nvidia.so', 'libEGL_nvidia.so')) for name in names)
             and any(name.startswith(('libnvidia-glcore.so', 'libnvidia-eglcore.so')) for name in names))
    return {'ready': ready, 'private_vendor_files': files, 'driver_mappings': lines,
            'compute_loaded': any(name.startswith('libcuda.so') for name in names),
            'all_observed_vendor_mappings_pinned': True}



def require_pidfds():
    if (sys.platform != 'linux' or not callable(getattr(os, 'pidfd_open', None))
            or not callable(getattr(signal, 'pidfd_send_signal', None))):
        raise RuntimeError('Linux os.pidfd_open and signal.pidfd_send_signal are required')


def ensure(condition, message):
    if not condition:
        raise RuntimeError(message)


def private_append(path):
    private_root()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    if not (stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()):
        os.close(fd)
        raise RuntimeError('private journal descriptor identity differs')
    os.fchmod(fd, 0o600)
    return os.fdopen(fd, 'a')


def write(path, value):
    private_root()
    fd, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def process_start(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def process_tag(pid):
    raw = (Path('/proc') / str(pid) / 'environ').read_bytes()
    prefix = TAG_KEY.encode() + b'='
    values = [item[len(prefix):].decode() for item in raw.split(b'\0') if item.startswith(prefix)]
    ensure(len(values) <= 1, 'duplicate restore tag in process environment')
    return values[0] if values else None


def original_running(row):
    start = process_start(row['pid'])
    if start is None:
        return False
    ensure(start == row['start'], 'original PID was reused; no signal or replacement allowed')
    if not live(row['pid']):
        return False
    return True


def original_alive(row):
    if not original_running(row):
        return False
    try:
        current = identity(row['pid'])
        tag = process_tag(row['pid'])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        # /proc environment access can disappear during process exit. Suppress
        # the read error only after proving absence/zombie with the same birth.
        if not original_running(row):
            return False
        raise
    ensure(current['start'] == row['start'] and same_command(current, row)
           and tag == row.get('restore_tag'), 'original process identity changed')
    return True


def launch_environment(row, tag, runtime):
    env = {key: value for key, value in runtime_environment(runtime).items() if key not in GUI_KEYS}
    env.update(row['env'])
    env[TAG_KEY] = tag
    return env


def tagged_process(row, tag, runtime):
    matches = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            if entry.stat().st_uid != os.getuid() or not live(pid):
                continue
            argv = [value.decode() for value in (entry / 'cmdline').read_bytes().split(b'\0') if value]
            if argv != row['argv']:
                continue
            if str((entry / 'cwd').resolve(strict=True)) != row['cwd']:
                continue
            if process_tag(pid) != tag:
                continue
            current = identity(pid)
            ensure(same_command(current, row), 'tagged process command or GUI environment changed')
            process_runtime_environment(pid, runtime)
            ensure(current['start'] == process_start(pid) and live(pid),
                   'tagged process changed during discovery; retry required')
            matches.append(current)
        except (FileNotFoundError, ProcessLookupError):
            continue
        # Only exact command/cwd candidates need an environment read. Permission
        # errors for those candidates remain fatal so an unjournaled launch is
        # never missed and duplicated. Unrelated non-dumpable processes need no
        # environment access.
    ensure(len(matches) <= 1, 'multiple live processes have this restore tag; no new launch')
    return matches[0] if matches else None


def same_command(left, right):
    return all(left[key] == right[key] for key in ('argv', 'cwd', 'env'))


def listening(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def check(runtime=None):
    require_pidfds()
    runtime = validate_runtime() if runtime is None else runtime
    canonical = json.loads((CAMPAIGN / 'blender-session.json').read_text())
    previous = json.loads((PREVIOUS / 'session.json').read_text())
    receipt = json.loads((PREVIOUS / 'verified.json').read_text())
    ensure(receipt['alive_and_listening'] and receipt['commands_and_gui_environment_match'],
           'round12 restoration was not verified')
    restored = receipt['processes']
    ensure(len(canonical) == len(previous) == len(restored) == 3,
           'exactly three authorized processes required')
    ensure([row['pid'] for row in canonical] == PIDS, 'authorized first round scope changed')
    ensure(len({row['pid'] for row in previous}) == 3
           and len({row['new_pid'] for row in restored}) == 3, 'successor PIDs must be unique')
    ensure(TAG_KEY not in os.environ, 'restore tag environment key is already in use by this runner')
    rows = []
    for first, prior, resumed, port in zip(canonical, previous, restored, PORTS):
        ensure(same_command(first, prior), 'round12 snapshot differs from canonical process command')
        ensure(resumed['original_pid'] == prior['pid']
               and prior['port'] == resumed['port'] == port, 'authorized process/port mapping changed')
        expected_tag = (prior.get('restore_tag') if resumed['new_pid'] == prior['pid']
                        else resumed['tag'])
        if resumed['new_pid'] != prior['pid']:
            ensure(isinstance(expected_tag, str) and bool(expected_tag), 'round12 restore tag is missing')
        fd = os.pidfd_open(resumed['new_pid'])
        try:
            current = identity(resumed['new_pid'])
            ensure(current['start'] == resumed['start'], 'round12 successor PID was reused')
            ensure(same_command(current, first), 'restored process command or GUI environment changed')
            ensure(live(current['pid']) and listening(port), 'successor is not alive and listening')
            ensure('blender' in Path(current['argv'][0]).name, 'successor is not Blender')
            ensure(process_tag(current['pid']) == expected_tag, 'round12 successor restore tag changed')
            ensure(current == identity(current['pid']), 'successor changed during inspection')
        finally:
            os.close(fd)
        proof = runtime['proven_sessions'][len(rows)]
        ensure((current['pid'], current['start'], port, current['env'])
               == (proof['pid'], proof['start'], proof['port'], proof['gui_env']),
               'round12 successor differs from proven GUI runtime environment')
        rows.append(dict(current, port=port, restore_tag=expected_tag,
                         original_driver_mappings=vendor_lines(current['pid'])))
    return rows


def stop():
    runtime = validate_runtime()
    rows = check(runtime)
    ROOT.mkdir(mode=0o700)
    descriptors = []
    try:
        with private_append(ROOT / 'restore.lock') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for row in rows:
                descriptors.append(os.pidfd_open(row['pid']))
                ensure(original_alive(row), 'authorized successor exited before stop')
            write(SNAPSHOT, rows)
            write(RUNTIME_SNAPSHOT, runtime)
            write(ROOT / 'stop-intents.json', [])
            write(ROOT / 'deadline.json', time.time() + 1800)
            # The watchdog is running before any durable signal intent or signal.
            with private_append(ROOT / 'watchdog.log') as log:
                subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'watchdog'],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 start_new_session=True, env=runtime_environment(runtime))
            ensure(bound_runtime() == runtime, 'runtime changed before stop signals')
            intents = []
            for row, fd in zip(rows, descriptors):
                ensure(original_alive(row), 'authorized successor exited before signal')
                intents.append(row['pid'])
                write(ROOT / 'stop-intents.json', intents)
                signal.pidfd_send_signal(fd, signal.SIGTERM)
            deadline = time.monotonic() + 30
            while any(original_running(row) for row in rows):
                if time.monotonic() > deadline:
                    raise RuntimeError('Blender did not exit; no SIGKILL sent')
                time.sleep(.2)
            write(ROOT / 'stopped.json', {'pids': [row['pid'] for row in rows], 'time': time.time()})
    except BaseException:
        if SNAPSHOT.exists():
            write(ROOT / 'deadline.json', 0)
            try:
                restore()
            except Exception as error:
                print(f'Restore pending; watchdog will retry: {error}', file=sys.stderr, flush=True)
        raise
    finally:
        for fd in descriptors:
            os.close(fd)
    print(json.dumps({'stopped': [row['pid'] for row in rows], 'recovery_seconds': 1800}), flush=True)


def restore():
    require_pidfds()
    with private_append(ROOT / 'restore.lock') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (ROOT / 'verified.json').exists():
            return
        runtime = bound_runtime()
        rows = read_private(SNAPSHOT)
        validate_snapshot(rows, runtime)
        ensure(len(rows) == 3 and len({row['pid'] for row in rows}) == 3
               and [row['port'] for row in rows] == PORTS, 'invalid round13 process scope')
        intents = read_private(ROOT / 'stop-intents.json')
        ensure(len(set(intents)) == len(intents) and set(intents) <= {row['pid'] for row in rows},
               'invalid stop intent scope')
        # A durable intent may precede a crash before SIGTERM. Re-send only through
        # a pidfd bound to the exact original; never treat it as a restored process.
        deadline = time.monotonic() + 30
        for row in rows:
            if row['pid'] in intents and original_alive(row):
                fd = os.pidfd_open(row['pid'])
                try:
                    if original_alive(row):
                        signal.pidfd_send_signal(fd, signal.SIGTERM)
                finally:
                    os.close(fd)
        while any(original_running(row) for row in rows if row['pid'] in intents):
            if time.monotonic() > deadline:
                raise RuntimeError('original Blender is still terminating; watchdog must retry')
            time.sleep(.2)
        journal_path = ROOT / 'restoring.json'
        resumed = read_private(journal_path) if journal_path.exists() else []
        ensure(len({item['original_pid'] for item in resumed}) == len(resumed)
               and {item['original_pid'] for item in resumed} <= {row['pid'] for row in rows},
               'invalid restoration journal scope')
        ensure(len({item['tag'] for item in resumed}) == len(resumed), 'restore journal tags must be unique')
        for row in rows:
            existing = next((item for item in resumed if item['original_pid'] == row['pid']), None)
            if existing is None:
                existing = {'original_pid': row['pid'], 'port': row['port'],
                            'tag': uuid.uuid4().hex, 'launch_intent': False,
                            'runtime_manifest': evidence(RUNTIME_SNAPSHOT)}
                resumed.append(existing)
                write(journal_path, resumed)
            ensure(existing['port'] == row['port'] and bool(existing['tag'])
                   and existing['runtime_manifest'] == evidence(RUNTIME_SNAPSHOT), 'invalid restore entry/runtime binding')
            if 'new_pid' in existing:
                start = process_start(existing['new_pid'])
                ensure(start is None or start == existing['start'],
                       'journaled successor PID was reused; no replacement allowed')
                if start is not None and live(existing['new_pid']):
                    current = identity(existing['new_pid'])
                    ensure(current['start'] == existing['start'] and same_command(current, row),
                           'journaled successor identity changed')
                    if existing['new_pid'] == row['pid']:
                        ensure(row['pid'] not in intents, 'terminating original cannot be restored')
                        ensure(process_tag(current['pid']) == row.get('restore_tag'),
                               'preserved original restore tag changed')
                    else:
                        ensure(process_tag(current['pid']) == existing['tag'], 'successor tag changed')
                        process_runtime_environment(current['pid'], runtime)
                    continue
            current = tagged_process(row, existing['tag'], runtime)
            if current is None and original_alive(row):
                ensure(row['pid'] not in intents, 'terminating original cannot be restored')
                ensure(not existing['launch_intent'], 'original alive after a restore launch intent')
                current = identity(row['pid'])
            if current is None:
                ensure(not listening(row['port']), 'restore port occupied; no duplicate launched')
                ensure(bound_runtime() == runtime, 'runtime changed before restore launch')
                existing['launch_intent'] = True
                write(journal_path, resumed)
                with private_append(ROOT / f"resume-{row['pid']}.log") as log:
                    process = subprocess.Popen(row['argv'], cwd=row['cwd'],
                                               env=launch_environment(row, existing['tag'], runtime),
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                               start_new_session=True)
                # A failure here leaves a durable tag that the next retry adopts.
                current = identity(process.pid)
                ensure(live(process.pid) and same_command(current, row)
                       and process_tag(process.pid) == existing['tag'], 'launched successor identity mismatch')
                process_runtime_environment(process.pid, runtime)
            existing.update(new_pid=current['pid'], start=current['start'])
            write(journal_path, resumed)
        resumed = [next(item for item in resumed if item['original_pid'] == row['pid']) for row in rows]
        deadline = time.monotonic() + 60
        while True:
            runtime_ready = True
            for original, restored in zip(rows, resumed):
                ensure(live(restored['new_pid']), 'restored Blender exited')
                current = identity(restored['new_pid'])
                ensure(current['start'] == restored['start'] and same_command(current, original),
                       'restored process identity changed')
                if restored['new_pid'] == original['pid']:
                    ensure(original['pid'] not in intents, 'terminating original cannot be verified')
                    ensure(process_tag(current['pid']) == original.get('restore_tag'),
                           'preserved original restore tag changed')
                    restored['runtime_mode'] = 'preserved_original_not_relaunched'
                    restored['driver_runtime'] = {'driver_mappings': vendor_lines(current['pid']),
                                                  'private_runtime_launch_verified': False}
                else:
                    ensure(process_tag(restored['new_pid']) == restored['tag'], 'restored process tag changed')
                    process_runtime_environment(current['pid'], runtime)
                    restored['runtime_mode'] = 'validated_private_compute_and_gl_environment'
                    restored['driver_runtime'] = verify_private_maps(current['pid'], runtime)
                    runtime_ready = runtime_ready and restored['driver_runtime']['ready']
                ensure(current == identity(current['pid']) and live(current['pid']), 'successor changed during runtime verification')
            if runtime_ready and all(listening(row['port']) for row in resumed):
                break
            if time.monotonic() > deadline:
                raise RuntimeError('restore ports or pinned GL vendor/core mappings did not become ready')
            time.sleep(.5)
        ensure(bound_runtime() == runtime, 'runtime changed before restoration receipt')
        write(journal_path, resumed)
        receipt = {'runtime_manifest': evidence(RUNTIME_SNAPSHOT),
                   'all_relaunched_processes_have_pinned_vendor_maps': True,
                   'host_or_global_configuration_modified': False,
                   'alive_and_listening': True, 'commands_and_gui_environment_match': True,
                   'processes': resumed, 'time': time.time()}
        write(ROOT / 'verified.json', receipt)
        print(json.dumps(receipt), flush=True)


def watchdog():
    while not (ROOT / 'verified.json').exists():
        try:
            if time.time() > read_private(ROOT / 'deadline.json'):
                restore()
        except Exception as error:
            print(f'Restore pending; retrying in 5 seconds: {error}', file=sys.stderr, flush=True)
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['check', 'stop', 'restore', 'extend', 'watchdog'])
    action = parser.parse_args().action
    if action == 'check':
        rows = check()
        print(json.dumps({'verified_successor_pids': [row['pid'] for row in rows]}))
    elif action == 'extend':
        with private_append(ROOT / 'restore.lock') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ensure(not (ROOT / 'verified.json').exists(), 'already restored')
            write(ROOT / 'deadline.json', time.time() + 1800)
    else:
        globals()[action]()
