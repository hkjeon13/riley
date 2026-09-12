"""Extract the two frozen diagnostic bundles after Round14 cleanup and restore.

Existing identical files are reused; differing files, links, and unsafe paths
are rejected before any member is written. No GPU or process action occurs.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path('/tmp/riley-opt-260912')
BUNDLES = {
    'post-round14-tools-v1': 'a1338f26259e37cd3a4a5f85c2ea848bd1e2e0e0c64d99ba61f253296a681436',
    'post-round14-precise-v1': 'c469fe5a6796b4d144fc8f789cef2154b1d5021a04c0c6acb79a0cbd01eacf57',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    final = json.loads((ROOT/'token-serving-round14/finalization.json').read_text())
    restored = final['restoration']
    assert restored['path'] == str(ROOT/'blender-round14/verified.json')
    assert digest(Path(restored['path']).read_bytes()) == restored['sha256']
    proof = json.loads(Path(restored['path']).read_text())
    assert proof['alive_and_listening'] and proof['commands_and_gui_environment_match']
    assert proof['all_relaunched_processes_have_pinned_vendor_maps'] and len(proof['processes']) == 3
    files = {}
    for name, expected in BUNDLES.items():
        archive = ROOT/(name+'.tar.gz')
        assert digest(archive.read_bytes()) == expected
        manifest = json.loads((ROOT/(name+'-manifest.json')).read_text())
        assert manifest['archive']['sha256'] == expected
        with tarfile.open(archive, 'r:gz') as source:
            members = source.getmembers()
            assert len(members) == len(manifest['files'])
            assert {m.name for m in members} == set(manifest['files'])
            for member in members:
                relative = PurePosixPath(member.name)
                assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
                assert str(relative) == member.name and member.name not in files
                target = ROOT/relative
                assert target.resolve().is_relative_to(ROOT.resolve())
                assert not any(p.is_symlink() for p in (target, *target.parents))
                data = source.extractfile(member).read()
                ref = manifest['files'][member.name]
                assert len(data) == ref['bytes'] and digest(data) == ref['sha256']
                if target.exists():
                    assert target.is_file() and target.read_bytes() == data, str(target)
                files[member.name] = (target, data)
    # Run the unchanged full cleanup gate from its verified archive bytes before
    # writing any source. The module's __main__ build entrypoint is not invoked.
    gate_path, gate_bytes = files['build_decode_profile_batch8.py']
    namespace = {'__name__': 'verified_post_round14_gate', '__file__': str(gate_path)}
    exec(compile(gate_bytes, str(gate_path), 'exec'), namespace)
    gate = namespace['campaign_gate']()
    created = []
    for name, (target, data) in files.items():
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with target.open('xb') as stream:
            stream.write(data)
        target.chmod(0o600)
        created.append(name)
    assert namespace['campaign_gate']() == gate
    for target, data in files.values():
        assert target.read_bytes() == data
    print(json.dumps({'files_verified': len(files), 'created': created,
                      'campaign_gate': gate, 'gpu_execution': False}))


if __name__ == '__main__':
    main()
