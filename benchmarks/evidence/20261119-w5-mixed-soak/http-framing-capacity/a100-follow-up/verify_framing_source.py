"""Bind the isolated HTTP build to the approved source and its one changed file."""

from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path, PurePosixPath
import tarfile

APPROVED_ARCHIVE = "cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5"
HTTP_SOURCE = "crates/riley-server/src/http.rs"
HTTP_HASHES = ("c0d1b93d7605ab8d6fd55ea20756fd70c9975cd080144cabbea72a6dc57c7c79",
               "617f26a7d9333e44f8265666868a032ea539f303cd54ccac9edb408c74e5aef6")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def inventory(path):
    entries = {}
    with tarfile.open(path, "r|") as archive:
        for item in archive:
            if item.isdir():
                continue
            name = PurePosixPath(item.name)
            if name.is_absolute() or ".." in name.parts or str(name) in entries:
                raise ValueError(f"invalid or duplicate archive member: {item.name}")
            if item.isfile():
                with archive.extractfile(item) as contents:
                    entries[str(name)] = ("file", file_digest(contents, "sha256").hexdigest(), bool(item.mode & 0o100))
            elif item.issym():
                entries[str(name)] = ("symlink", item.linkname)
            else:
                raise ValueError(f"unexpected archive member type: {item.name}")
    return entries


def verify(base_path, candidate_path, recorded_candidate_hash):
    base_hash = digest(base_path)
    candidate_hash = digest(candidate_path)
    if base_hash != APPROVED_ARCHIVE or candidate_hash != recorded_candidate_hash:
        raise ValueError("source tar hash differs from the approved base or the diagnostic build")
    base = inventory(base_path)
    candidate = inventory(candidate_path)
    if base.keys() != candidate.keys():
        raise ValueError("source file inventory changed")
    changed = sorted(name for name in base if base[name] != candidate[name])
    if changed != [HTTP_SOURCE] or tuple(archive[HTTP_SOURCE][1] for archive in (base, candidate)) != HTTP_HASHES:
        raise ValueError(f"source changed outside the registered HTTP patch: {changed}")
    return {
        "scope": "Files and executable bits in the staged Linux Docker build inputs.",
        "baseline_archive_sha256": base_hash, "diagnostic_archive_sha256": candidate_hash,
        "snapshot_members": len(base), "unchanged_members": len(base) - 1,
        "changed_member": HTTP_SOURCE, "base_http_sha256": HTTP_HASHES[0], "diagnostic_http_sha256": HTTP_HASHES[1],
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("candidate_sha256", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.base, args.candidate, args.candidate_sha256.read_text().strip()), indent=2))
