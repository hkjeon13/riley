"""Compare RSS from private mappings shaped like glibc 2.35 secondary heaps."""

from argparse import ArgumentParser
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from typing import NamedTuple

from allocator_probe import read_mappings


HEAP_ALIGNMENT = 64 * 1024**2


class Mapping(NamedTuple):
    end: int
    metadata: tuple[str, ...]
    category: str
    rss: int
    anonymous: int


def capture(file):
    raw = file.read_bytes()
    text = gzip.decompress(raw).decode()
    maps = {}
    for (start, end, *metadata), (category, values) in read_mappings(text.splitlines()).items():
        address = int(start, 16)
        if address in maps:
            raise ValueError("duplicate map start")
        maps[address] = Mapping(int(end, 16), tuple(metadata), category, values["Rss"], values["Anonymous"])
    return maps, sha256(raw).hexdigest()


def heap_shape(address, maps):
    segment = maps[address]
    guard = maps.get(segment.end)
    ceiling = address + HEAP_ALIGNMENT
    return (
        address % HEAP_ALIGNMENT == 0
        and address < segment.end < ceiling
        and segment.metadata == ("rw-p", "00000000", "00:00", "0", "")
        and guard is not None
        and guard.metadata == ("---p", "00000000", "00:00", "0", "")
        and guard.rss == 0
        and guard.anonymous == 0
        and guard.end >= ceiling
    )


def compare(folder):
    midpoint, start_sha = capture(folder / "smaps-maps-midpoint.txt.gz")
    drain, end_sha = capture(folder / "smaps-maps-drain.txt.gz")
    common = midpoint.keys() & drain.keys()
    shaped = {start for start in common if heap_shape(start, midpoint) and heap_shape(start, drain)}
    groups = {"matching_heap_shape": Counter(), "other_private_mapping": Counter()}
    addresses = {address for address in common
                 if midpoint[address].metadata == drain[address].metadata
                 and midpoint[address].category == drain[address].category == "anonymous-private"}
    for address in addresses:
        old, new = midpoint[address], drain[address]
        group = groups["matching_heap_shape" if address in shaped else "other_private_mapping"]
        change = new.anonymous - old.anonymous
        prefix = "same_extent" if old.end == new.end else "changed_extent"
        group[f"{prefix}_mappings"] += 1
        group[f"{prefix}_rss_change_kib"] += change
        group[f"{prefix}_increased_mappings"] += change > 0
        group[f"{prefix}_decreased_mappings"] += change < 0
    return {
        "scope": "64 MiB aligned anonymous writable segment with an adjacent inaccessible private reservation. "
                 "Mapping geometry is compared, without reading allocator metadata or retained addresses.",
        "heap_alignment_bytes": HEAP_ALIGNMENT,
        "midpoint_sha256": start_sha,
        "drain_sha256": end_sha,
        "matching_starts_at_both_samples": len(shaped),
        "matching_starts_with_exact_reservation_end_at_both_samples": sum(
            midpoint[midpoint[address].end].end == drain[drain[address].end].end == address + HEAP_ALIGNMENT
            for address in shaped
        ),
        "total_anonymous_rss_change_kib": sum(m.anonymous for m in drain.values())
        - sum(m.anonymous for m in midpoint.values()),
        "paired_groups": groups,
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.capture_dir), indent=2, allow_nan=False))
