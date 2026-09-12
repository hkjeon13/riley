"""Extract signed old GL libraries into a private runtime; never install packages."""
import json
import lzma
from pathlib import Path
import subprocess

import prepare_driver_runtime as compute

ROOT = Path('/tmp/riley-opt-260912/driver580173-gui-runtime-20260901')
NAMES = ('libnvidia-gl-580', 'libnvidia-cfg1-580', 'libnvidia-extra-580')


def main():
    receipt = json.loads((compute.ROOT / 'receipt.json').read_text())
    assert receipt['completed'] and receipt['archive_signature_verified']
    assert compute.sha(compute.ROOT / 'InRelease') == receipt['release_sha256']
    assert compute.sha(compute.ROOT / 'Packages.xz') == receipt['index_sha256']
    assert '580.173.02' in Path('/proc/driver/nvidia/version').read_text()
    ROOT.mkdir(exist_ok=False)
    fields_by_name = {}
    for paragraph in lzma.decompress((compute.ROOT / 'Packages.xz').read_bytes()).decode().split('\n\n'):
        fields = {}
        for line in paragraph.splitlines():
            if line and not line[0].isspace() and ': ' in line:
                key, value = line.split(': ', 1)
                fields[key] = value
        if (fields.get('Package') in NAMES and fields.get('Version') == compute.VERSION
                and fields.get('Architecture') == 'amd64'):
            assert fields['Package'] not in fields_by_name
            fields_by_name[fields['Package']] = fields
    assert set(fields_by_name) == set(NAMES)
    extracted = ROOT / 'extracted'
    sources = []
    for name, fields in sorted(fields_by_name.items()):
        package = ROOT / Path(fields['Filename']).name
        compute.fetch(fields['Filename'], package, 256 * 1024 * 1024)
        assert compute.sha(package) == fields['SHA256'] and package.stat().st_size == int(fields['Size'])
        subprocess.run(['dpkg-deb', '-x', str(package), str(extracted)], check=True)
        sources.append({'package': name, 'version': compute.VERSION,
                        'url': compute.BASE + fields['Filename'], 'path': str(package),
                        'sha256': compute.sha(package), 'size': package.stat().st_size})
    result = {'completed': True, 'host_packages_modified': False, 'host_restart_performed': False,
              'runtime_only': True, 'snapshot': compute.BASE, 'packages': sources,
              'signed_compute_receipt': {'path': str(compute.ROOT / 'receipt.json'),
                                         'sha256': compute.sha(compute.ROOT / 'receipt.json')},
              'helpers': {str(p): compute.sha(p) for p in (Path(__file__), Path(compute.__file__))},
              'library_path': str(extracted / 'usr/lib/x86_64-linux-gnu'),
              'files': {str(p.relative_to(extracted)): compute.sha(p) for p in sorted(extracted.rglob('*'))
                        if p.is_file() and not p.is_symlink()},
              'symlinks': {str(p.relative_to(extracted)): str(p.readlink())
                           for p in sorted(extracted.rglob('*')) if p.is_symlink()}}
    (ROOT / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'completed': True, 'packages': len(sources), 'host_packages_modified': False}))


if __name__ == '__main__':
    main()
