"""Extract signed matching NVIDIA user libraries without installing host packages."""
import hashlib
import json
import lzma
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path('/tmp/riley-opt-260912/driver580173-runtime-20260901')
BASE = 'https://snapshot.ubuntu.com/ubuntu/20260901T000000Z/'
VERSION = '580.173.02-0ubuntu0.22.04.1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(relative, destination, limit):
    with urllib.request.urlopen(BASE + relative, timeout=60) as response, destination.open('xb') as out:
        total = 0
        while data := response.read(1024 * 1024):
            total += len(data)
            if total > limit:
                raise ValueError('download exceeds bounded size')
            out.write(data)


def main():
    ROOT.mkdir(exist_ok=False)
    binding = json.loads((ROOT.parent / 'batch7-binding.json').read_text())
    kernel = Path('/proc/driver/nvidia/version').read_text()
    assert '580.173.02' in kernel
    (ROOT / 'kernel-version.txt').write_text(kernel)
    sources = []
    # Signed archive metadata selects exact package bytes; no apt source mutation.
    release = ROOT / 'InRelease'
    fetch('dists/jammy-updates/InRelease', release, 1024 * 1024)
    verified = subprocess.run(['gpgv', '--keyring', '/usr/share/keyrings/ubuntu-archive-keyring.gpg', str(release)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (ROOT / 'signature.log').write_text(verified.stdout)
    assert verified.returncode == 0, 'Ubuntu archive signature failed'
    relative_index = 'restricted/binary-amd64/Packages.xz'
    in_sha = False
    expected = None
    for line in release.read_text().splitlines():
        if line == 'SHA256:':
            in_sha = True
            continue
        if in_sha and line and not line.startswith(' '):
            in_sha = False
        if in_sha:
            parts = line.split()
            if len(parts) == 3 and parts[2] == relative_index:
                expected = (parts[0], int(parts[1]))
    assert expected is not None
    index = ROOT / 'Packages.xz'
    fetch('dists/jammy-updates/' + relative_index, index, 32 * 1024 * 1024)
    assert (sha(index), index.stat().st_size) == expected
    selected = {}
    for paragraph in lzma.decompress(index.read_bytes()).decode().split('\n\n'):
        fields = {}
        for line in paragraph.splitlines():
            if line and not line[0].isspace() and ': ' in line:
                key, value = line.split(': ', 1)
                fields[key] = value
        if (fields.get('Package') in ('libnvidia-compute-580', 'nvidia-utils-580')
                and fields.get('Version') == VERSION and fields.get('Architecture') == 'amd64'):
            assert fields['Package'] not in selected
            selected[fields['Package']] = fields
    assert len(selected) == 2, 'snapshot does not supply both exact matching packages'
    extracted = ROOT / 'extracted'
    for name, fields in sorted(selected.items()):
        package = ROOT / Path(fields['Filename']).name
        fetch(fields['Filename'], package, 128 * 1024 * 1024)
        assert sha(package) == fields['SHA256'] and package.stat().st_size == int(fields['Size'])
        metadata = subprocess.check_output(['dpkg-deb', '-f', str(package), 'Package', 'Version', 'Architecture'], text=True)
        assert name in metadata and VERSION in metadata and 'amd64' in metadata
        subprocess.run(['dpkg-deb', '-x', str(package), str(extracted)], check=True)
        sources.append({'package': name, 'version': VERSION, 'url': BASE + fields['Filename'],
                        'path': str(package), 'sha256': sha(package), 'size': package.stat().st_size})
    lib = extracted / 'usr/lib/x86_64-linux-gnu'
    binary = extracted / 'usr/bin/nvidia-smi'
    assert (lib / 'libcuda.so.580.173.02').is_file()
    assert (lib / 'libnvidia-ml.so.580.173.02').is_file()
    for stem in ('libcuda', 'libnvidia-ml'):
        soname = lib / (stem + '.so.1')
        if not soname.exists():
            soname.symlink_to(stem + '.so.580.173.02')
    # This environment applies only to the child process.
    result = subprocess.run(['env', 'LD_LIBRARY_PATH=' + str(lib), str(binary),
                             '--query-gpu=uuid,driver_version,name,memory.used', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    (ROOT / 'nvidia-smi.stdout').write_text(result.stdout)
    (ROOT / 'nvidia-smi.stderr').write_text(result.stderr)
    assert result.returncode == 0
    expected_gpu = binding['environment']['gpu']
    assert f"{expected_gpu['uuid']}, 580.173.02, {expected_gpu['model']}" in result.stdout
    receipt = {'completed': True, 'diagnostic_runtime_only': True, 'host_packages_modified': False,
               'host_restart_performed': False, 'snapshot': BASE, 'kernel_version': kernel,
               'archive_signature_verified': True, 'release_sha256': sha(release),
               'index_sha256': sha(index), 'packages': sources, 'library_path': str(lib),
               'nvidia_smi_path': str(binary), 'gpu_query': result.stdout.strip(),
               'files': {str(p.relative_to(extracted)): sha(p) for p in sorted(extracted.rglob('*'))
                         if p.is_file() and not p.is_symlink()},
               'symlinks': {str(p.relative_to(extracted)): str(p.readlink())
                            for p in sorted(extracted.rglob('*')) if p.is_symlink()}}
    (ROOT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'completed': True, 'host_packages_modified': False, 'gpu': result.stdout.strip()}))


if __name__ == '__main__':
    main()
