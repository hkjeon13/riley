"""Pause only the three verified round10 successors of the authorized first round.

State is private to round11. A lock serializes restoration and an incremental
journal lets the watchdog resume a partially completed restore. No SIGKILL.
"""
import argparse
import fcntl
import json
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
PREVIOUS = CAMPAIGN / 'blender-round10'
ROOT = CAMPAIGN / 'blender-round11'
SNAPSHOT = ROOT / 'session.json'
TAG_KEY = 'RILEY_BLENDER_RESTORE_SESSION'
GUI_KEYS = {'DISPLAY', 'XAUTHORITY', 'XDG_RUNTIME_DIR',
            'DBUS_SESSION_BUS_ADDRESS', 'WAYLAND_DISPLAY'}


def require_pidfds():
    if (sys.platform != 'linux' or not callable(getattr(os, 'pidfd_open', None))
            or not callable(getattr(signal, 'pidfd_send_signal', None))):
        raise RuntimeError('Linux os.pidfd_open and signal.pidfd_send_signal are required')


def ensure(condition, message):
    if not condition:
        raise RuntimeError(message)


def private_append(path):
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    os.fchmod(fd, 0o600)
    return os.fdopen(fd, 'a')


def write(path, value):
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


def launch_environment(row, tag):
    env = {key: value for key, value in os.environ.items()
           if key not in GUI_KEYS and key != TAG_KEY}
    env.update(row['env'])
    env[TAG_KEY] = tag
    return env


def tagged_process(row, tag):
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


def check():
    require_pidfds()
    canonical = json.loads((CAMPAIGN / 'blender-session.json').read_text())
    previous = json.loads((PREVIOUS / 'session.json').read_text())
    receipt = json.loads((PREVIOUS / 'verified.json').read_text())
    ensure(receipt['alive_and_listening'] and receipt['commands_and_gui_environment_match'],
           'round10 restoration was not verified')
    restored = receipt['processes']
    ensure(len(canonical) == len(previous) == len(restored) == 3,
           'exactly three authorized processes required')
    ensure([row['pid'] for row in canonical] == PIDS, 'authorized first round scope changed')
    ensure(len({row['pid'] for row in previous}) == 3
           and len({row['new_pid'] for row in restored}) == 3, 'successor PIDs must be unique')
    ensure(TAG_KEY not in os.environ, 'restore tag environment key is already in use by this runner')
    rows = []
    for first, prior, resumed, port in zip(canonical, previous, restored, PORTS):
        ensure(same_command(first, prior), 'round10 snapshot differs from canonical process command')
        ensure(resumed['original_pid'] == prior['pid']
               and prior['port'] == resumed['port'] == port, 'authorized process/port mapping changed')
        expected_tag = (prior.get('restore_tag') if resumed['new_pid'] == prior['pid']
                        else resumed['tag'])
        if resumed['new_pid'] != prior['pid']:
            ensure(isinstance(expected_tag, str) and bool(expected_tag), 'round10 restore tag is missing')
        fd = os.pidfd_open(resumed['new_pid'])
        try:
            current = identity(resumed['new_pid'])
            ensure(current['start'] == resumed['start'], 'round10 successor PID was reused')
            ensure(same_command(current, first), 'restored process command or GUI environment changed')
            ensure(live(current['pid']) and listening(port), 'successor is not alive and listening')
            ensure('blender' in Path(current['argv'][0]).name, 'successor is not Blender')
            ensure(process_tag(current['pid']) == expected_tag, 'round10 successor restore tag changed')
            ensure(current == identity(current['pid']), 'successor changed during inspection')
        finally:
            os.close(fd)
        rows.append(dict(current, port=port, restore_tag=expected_tag))
    return rows


def stop():
    rows = check()
    ROOT.mkdir(mode=0o700)
    descriptors = []
    try:
        with private_append(ROOT / 'restore.lock') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for row in rows:
                descriptors.append(os.pidfd_open(row['pid']))
                ensure(original_alive(row), 'authorized successor exited before stop')
            write(SNAPSHOT, rows)
            write(ROOT / 'stop-intents.json', [])
            write(ROOT / 'deadline.json', time.time() + 1800)
            # The watchdog is running before any durable signal intent or signal.
            with private_append(ROOT / 'watchdog.log') as log:
                subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'watchdog'],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 start_new_session=True)
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
        rows = json.loads(SNAPSHOT.read_text())
        ensure(len(rows) == 3 and len({row['pid'] for row in rows}) == 3
               and [row['port'] for row in rows] == PORTS, 'invalid round11 process scope')
        intents = json.loads((ROOT / 'stop-intents.json').read_text())
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
        resumed = json.loads(journal_path.read_text()) if journal_path.exists() else []
        ensure(len({item['original_pid'] for item in resumed}) == len(resumed)
               and {item['original_pid'] for item in resumed} <= {row['pid'] for row in rows},
               'invalid restoration journal scope')
        for row in rows:
            existing = next((item for item in resumed if item['original_pid'] == row['pid']), None)
            if existing is None:
                existing = {'original_pid': row['pid'], 'port': row['port'],
                            'tag': uuid.uuid4().hex, 'launch_intent': False}
                resumed.append(existing)
                write(journal_path, resumed)
            ensure(existing['port'] == row['port'] and bool(existing['tag']), 'invalid restore entry')
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
                    continue
            current = tagged_process(row, existing['tag'])
            if current is None and original_alive(row):
                ensure(row['pid'] not in intents, 'terminating original cannot be restored')
                ensure(not existing['launch_intent'], 'original alive after a restore launch intent')
                current = identity(row['pid'])
            if current is None:
                ensure(not listening(row['port']), 'restore port occupied; no duplicate launched')
                existing['launch_intent'] = True
                write(journal_path, resumed)
                with private_append(ROOT / f"resume-{row['pid']}.log") as log:
                    process = subprocess.Popen(row['argv'], cwd=row['cwd'],
                                               env=launch_environment(row, existing['tag']),
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                               start_new_session=True)
                # A failure here leaves a durable tag that the next retry adopts.
                current = identity(process.pid)
                ensure(live(process.pid) and same_command(current, row)
                       and process_tag(process.pid) == existing['tag'], 'launched successor identity mismatch')
            existing.update(new_pid=current['pid'], start=current['start'])
            write(journal_path, resumed)
        resumed = [next(item for item in resumed if item['original_pid'] == row['pid']) for row in rows]
        deadline = time.monotonic() + 60
        while True:
            for original, restored in zip(rows, resumed):
                ensure(live(restored['new_pid']), 'restored Blender exited')
                current = identity(restored['new_pid'])
                ensure(current['start'] == restored['start'] and same_command(current, original),
                       'restored process identity changed')
                if restored['new_pid'] == original['pid']:
                    ensure(original['pid'] not in intents, 'terminating original cannot be verified')
                    ensure(process_tag(current['pid']) == original.get('restore_tag'),
                           'preserved original restore tag changed')
                else:
                    ensure(process_tag(restored['new_pid']) == restored['tag'], 'restored process tag changed')
            if all(listening(row['port']) for row in resumed):
                break
            if time.monotonic() > deadline:
                raise RuntimeError('restore ports did not become ready')
            time.sleep(.5)
        receipt = {'alive_and_listening': True, 'commands_and_gui_environment_match': True,
                   'processes': resumed, 'time': time.time()}
        write(ROOT / 'verified.json', receipt)
        print(json.dumps(receipt), flush=True)


def watchdog():
    while not (ROOT / 'verified.json').exists():
        try:
            if time.time() > json.loads((ROOT / 'deadline.json').read_text()):
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
