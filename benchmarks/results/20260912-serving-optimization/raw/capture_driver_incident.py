"""Read-only host/runtime evidence after the unattended NVIDIA package update."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/tmp/riley-opt-260912')


def command(argv, env=None):
    result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=30, env=env)
    return {'argv': argv, 'exit_code': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}


def main():
    import check_gui_driver_runtime as check
    before = check.check_sessions()
    compute = check.check_runtime(check.COMPUTE)
    check.check_runtime(check.GUI)
    blocks = Path('/var/log/apt/history.log').read_text().split('\n\n')
    updates = []
    for block in blocks:
        if 'unattended-upgrade' not in block or '580.178.04' not in block:
            continue
        lines = []
        for line in block.splitlines():
            if line.startswith(('Start-Date:', 'End-Date:', 'Commandline:')):
                lines.append(line)
            elif line.startswith('Upgrade:'):
                import re
                packages = re.findall(r'((?:libnvidia|nvidia|xserver-xorg-video-nvidia)[^,]*:amd64 \([^)]*\))', line)
                lines.append('NVIDIA upgrade entries: ' + '; '.join(packages))
        updates.append(lines)
    assert updates, 'expected unattended NVIDIA upgrade evidence absent'
    defaults = command(['/usr/bin/nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader'])
    import os
    env = dict(os.environ, LD_LIBRARY_PATH=compute['library_path'])
    private = command([compute['nvidia_smi_path'], '--query-gpu=uuid,driver_version', '--format=csv,noheader'], env)
    assert defaults['exit_code'] != 0 and private['exit_code'] == 0
    assert '580.173.02' in private['stdout']
    maps = {}
    for row in before:
        maps[str(row['pid'])] = [line for line in Path(f"/proc/{row['pid']}/maps").read_text().splitlines()
                                 if any(name in line for name in ('libcuda.', 'libnvidia-', 'libGLX_nvidia', 'libEGL_nvidia'))]
    assert check.check_sessions() == before
    output = ROOT / 'driver-incident-host-audit.json'
    value = {'schema': 'riley.driver-incident-host-audit.v1', 'read_only': True,
             'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'kernel_driver': Path('/proc/driver/nvidia/version').read_text(),
             'host_kernel': command(['uname', '-r']), 'nvidia_updates': updates,
             'default_nvml': defaults, 'private_nvml': private,
             'installed_packages': command(['dpkg-query', '-W', '-f=${Package}\t${Version}\n',
                                            'libnvidia-compute-580', 'libnvidia-gl-580', 'nvidia-utils-580']),
             'host_libcuda': str(Path('/usr/lib/x86_64-linux-gnu/libcuda.so.1').resolve()),
             'host_libnvml': str(Path('/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1').resolve()),
             'blender_sessions_unchanged': True, 'blender': before, 'blender_driver_mappings': maps,
             'private_compute_receipt': check.evidence(check.COMPUTE / 'receipt.json'),
             'private_gui_receipt': check.evidence(check.GUI / 'receipt.json'),
             'fresh_context_completion': check.evidence(ROOT / 'driver-runtime-gui-probe-v2/completion.json'),
             'capture_helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'performance_claim': False, 'host_configuration_modified': False}
    with output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'complete': True, 'artifact': str(output), 'private_nvml_exit': private['exit_code'],
                      'default_nvml_exit': defaults['exit_code'], 'blender_sessions_unchanged': True}))


if __name__ == '__main__':
    main()
