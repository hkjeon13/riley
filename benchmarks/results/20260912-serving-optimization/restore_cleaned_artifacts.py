"""List, verify, or restore the losslessly archived serving response evidence."""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import tarfile


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--list', action='store_true')
    actions.add_argument('--verify', action='store_true')
    actions.add_argument('--restore', metavar='GLOB', help='Match original paths relative to this campaign')
    parser.add_argument('--destination', type=Path, help='Alternate restoration root (default: campaign directory)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    inventory = json.loads((root / 'raw/storage-cleanup-20260916/inventory.json').read_text())
    records = {row['path']: row for row in inventory['archived_originals']}
    if args.list:
        for row in records.values():
            print(f"{row['bytes']:>10} {row['path']}")
        return
    selected = records if args.verify else {name: row for name, row in records.items()
                                            if fnmatch.fnmatchcase(name, args.restore)}
    if not selected:
        parser.error('No archived paths match the pattern')
    archive = root / inventory['archive']['path']
    if sha256(archive) != inventory['archive']['sha256']:
        raise ValueError('Archive SHA256 mismatch')
    destination = (args.destination or root).resolve()
    seen = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if member.name not in records or not member.isfile() or member.name in seen:
                raise ValueError(f'Unexpected archive entry: {member.name}')
            seen.add(member.name)
            if member.name not in selected:
                continue
            row = selected[member.name]
            target = destination / member.name
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f'Unsafe restoration path: {member.name}')
            if not args.verify and target.exists():
                if sha256(target) != row['sha256']:
                    raise FileExistsError(f'Refusing to overwrite different contents: {target}')
                continue
            tmp = target.with_name(target.name + '.restore-partial')
            output = None
            created_tmp = False
            try:
                if not args.verify:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    output = tmp.open('xb')
                    created_tmp = True
                h = hashlib.sha256()
                size = 0
                with tar.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(block)
                        size += len(block)
                        if output is not None:
                            output.write(block)
                if size != row['bytes'] or h.hexdigest() != row['sha256']:
                    raise ValueError(f'Original contents SHA256 mismatch: {member.name}')
                if output is not None:
                    output.close()
                    output = None
                    tmp.chmod(member.mode & 0o777)
                    # Exclusive creation also prevents replacing a file created during restoration.
                    os.link(tmp, target)
                    tmp.unlink()
            finally:
                if output is not None:
                    output.close()
                if created_tmp:
                    tmp.unlink(missing_ok=True)
    if seen != set(records):
        raise ValueError('Archive is missing inventory entries')
    print(f"{'Verified' if args.verify else 'Restored or already present'} {len(selected)} files")


if __name__ == '__main__':
    main()
