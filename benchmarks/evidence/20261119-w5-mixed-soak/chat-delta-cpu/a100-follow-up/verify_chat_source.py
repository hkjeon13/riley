"""Verify that the approved source archive differs only in the three chat SSE files."""

from argparse import ArgumentParser
import json
from pathlib import Path

from archive_inventory import APPROVED_ARCHIVE, digest, inventory


FILES = {f"crates/riley-server/src/{name}.rs" for name in ("chat", "openai", "service")}


def verify(base, candidate, expected):
    base_sha, candidate_sha = digest(base), digest(candidate)
    if base_sha != APPROVED_ARCHIVE or set(expected) != FILES:
        raise ValueError("baseline or designated source files differ")
    before, after = inventory(base), inventory(candidate)
    if before.keys() != after.keys():
        raise ValueError("build input inventory changed")
    changed = {name for name in before if before[name] != after[name]}
    if changed != FILES:
        raise ValueError(f"unexpected changed build inputs: {sorted(changed)}")
    for name, hashes in expected.items():
        old, new = before[name], after[name]
        if old != ("file", hashes["base"], False) or new != ("file", hashes["candidate"], False):
            raise ValueError(f"patch contents or permissions differ for {name}")
    return {"scope": "File bytes and executable permission in isolated Docker build inputs.",
            "base_archive_sha256": base_sha, "candidate_archive_sha256": candidate_sha,
            "regular_and_symlink_entries": len(before), "unchanged_entries": len(before) - len(FILES),
            "changed": expected}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("changes", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.base, args.candidate, json.loads(args.changes.read_text())),
                     indent=2, allow_nan=False))
