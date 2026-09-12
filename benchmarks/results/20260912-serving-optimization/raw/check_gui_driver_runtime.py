"""Validate isolated NVIDIA runtime and offscreen contexts with all Blender sessions retained."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import remote_session_round12 as session

ROOT = Path('/tmp/riley-opt-260912')
OUTPUT = ROOT / 'driver-runtime-gui-probe-v2'
COMPUTE = ROOT / 'driver580173-runtime-20260901'
GUI = ROOT / 'driver580173-gui-runtime-20260901'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence(path):
    return {'path': str(path), 'sha256': digest(path)}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_runtime(root):
    receipt = read(root / 'receipt.json')
    assert receipt['completed'] and not receipt['host_packages_modified']
    extracted = root / 'extracted'
    for path, expected in receipt['files'].items():
        assert digest(extracted / path) == expected, path
    for path, expected in receipt['symlinks'].items():
        target = extracted / path
        assert target.is_symlink() and str(target.readlink()) == expected, path
        assert target.exists(), path
    return receipt


def check_sessions():
    previous = read(ROOT / 'blender-round12/session.json')
    restored = read(ROOT / 'blender-round12/verified.json')
    assert len(previous) == len(restored['processes']) == 3
    result = []
    for original, row in zip(previous, restored['processes']):
        current = session.identity(row['new_pid'])
        assert current['start'] == row['start'] and session.live(row['new_pid'])
        assert session.same_command(current, original) and session.process_tag(row['new_pid']) == row['tag']
        assert session.listening(row['port'])
        result.append({'pid': row['new_pid'], 'start': current['start'], 'port': row['port'],
                       'commands_and_gui_environment_match': True, 'gui_env': current['env']})
    return result


def main():
    assert not OUTPUT.exists()
    assert '580.173.02' in Path('/proc/driver/nvidia/version').read_text()
    assert not os.environ.get('LD_PRELOAD')
    compute, gui = check_runtime(COMPUTE), check_runtime(GUI)
    assert evidence(COMPUTE / 'receipt.json') == gui['signed_compute_receipt']
    before = check_sessions()
    OUTPUT.mkdir()
    library_path = gui['library_path'] + ':' + compute['library_path']
    vendor = GUI / 'extracted/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    assert read(vendor)['ICD']['library_path'] == 'libEGL_nvidia.so.0'
    overrides = {'LD_LIBRARY_PATH': library_path,
                 'PATH': str(Path(compute['nvidia_smi_path']).parent) + ':' + os.environ['PATH'],
                 '__EGL_VENDOR_LIBRARY_FILENAMES': str(vendor), '__GLX_VENDOR_LIBRARY_NAME': 'nvidia'}
    helpers = [ROOT / name for name in ('probe_egl_context.py', 'probe_glx_context.py',
                                       'remote_session.py', 'remote_session_round12.py')]
    pins = {str(path): digest(path) for path in (*helpers, Path(__file__))}
    write(OUTPUT / 'preparation.json', {'performance_claim': False, 'window_created': False,
          'runtime_receipts': [evidence(COMPUTE / 'receipt.json'), evidence(GUI / 'receipt.json')],
          'child_only_overrides': overrides, 'blender_before': before, 'helpers': pins})
    results = []
    environments = {}
    for row in before:
        key = json.dumps(row['gui_env'], sort_keys=True)
        environments.setdefault(key, []).append(row['pid'])
    try:
        for index, (serialized, pids) in enumerate(environments.items()):
            env = {k: v for k, v in os.environ.items() if k not in session.GUI_KEYS}
            env.update(json.loads(serialized))
            env.update(overrides)
            for api in ('egl', 'glx'):
                helper = ROOT / ('probe_' + api + '_context.py')
                result = subprocess.run(['python3', str(helper)], env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, timeout=60)
                prefix = OUTPUT / f'env-{index}-{api}'
                prefix.with_suffix('.stdout').write_text(result.stdout)
                prefix.with_suffix('.stderr').write_text(result.stderr)
                assert result.returncode == 0, (api, result.stderr)
                proof = json.loads(result.stdout)
                assert proof['completed'] and proof['cleanup_completed'] and not proof['window_created']
                assert all('580.178' not in line for line in proof['driver_mappings'])
                expected = 'libEGL_nvidia' if api == 'egl' else 'libGLX_nvidia'
                assert any(gui['library_path'] in line and expected in line for line in proof['driver_mappings'])
                results.append({'api': api, 'blender_pids_using_this_gui_environment': pids,
                                'stdout': evidence(prefix.with_suffix('.stdout')), 'gl': proof['gl']})
        after = check_sessions()
        assert before == after
        assert all(digest(path) == sha for path, sha in pins.items())
        write(OUTPUT / 'completion.json', {'completed': True, 'time': time.time(), 'performance_claim': False,
              'window_created': False, 'blender_sessions_unchanged': True, 'blender_after': after,
              'contexts': results, 'child_only_overrides': overrides,
              'runtime_receipts': [evidence(COMPUTE / 'receipt.json'), evidence(GUI / 'receipt.json')],
              'scope': 'Fresh EGL/GLX contexts; does not itself prove a future full Blender file/script restoration.'})
        print(json.dumps({'completed': True, 'contexts': len(results), 'blender_sessions_unchanged': True}))
    except BaseException as error:
        write(OUTPUT / 'failure.json', {'completed': False, 'error': str(error), 'type': type(error).__name__})
        raise


if __name__ == '__main__':
    main()
