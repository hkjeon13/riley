"""Compare anonymous RSS by mapping class without retaining addresses or filenames."""
import json
from pathlib import Path
import re
import sys
import time

HEADER = re.compile(r"^([0-9a-f]+)-[0-9a-f]+ ([rwxps-]{4}) [0-9a-f]+ [0-9a-f]+:[0-9a-f]+ +\d+\s*(.*?)\s*$")
FIELDS = ("Rss", "Anonymous", "Private_Dirty", "AnonHugePages")


def mapping_class(name, permissions):
    if name == "[heap]":
        return "heap"
    if name.startswith("[stack"):
        return "labeled-stack"
    if not name or name.startswith(("[anon:", "[anon_shmem:")):
        return "anonymous-private" if permissions.endswith("p") else "anonymous-shared"
    if name.startswith("["):
        return "other-kernel"
    return "file-or-device"


def read_mappings(pid):
    started = time.perf_counter()
    mappings = {}

    def append_mapping():
        if address is not None:
            if address in mappings or set(fields) != set(FIELDS):
                raise ValueError("incomplete or duplicate smaps mapping")
            mappings[address] = (label, fields)

    address, label, fields = None, None, {}
    with open(f"/proc/{pid}/smaps", encoding="utf-8", errors="replace") as smaps:
        for line in smaps:
            header = HEADER.fullmatch(line)
            if header:
                append_mapping()
                address, permissions, name = header.groups()
                label = mapping_class(name, permissions)
                fields = {}
                continue
            key, _, value = line.partition(":")
            if key in FIELDS:
                number, unit = value.split()
                if key in fields or unit != "kB":
                    raise ValueError(f"invalid smaps field {key}")
                fields[key] = int(number)
    append_mapping()
    if not mappings:
        raise ValueError("no smaps mappings read")
    return mappings, (time.perf_counter() - started) * 1000


def totals(mappings):
    grouped = {}
    for label, fields in mappings.values():
        category = grouped.setdefault(label, dict.fromkeys(FIELDS, 0))
        for field, value in fields.items():
            category[field] += value
    return grouped


def compare(before, after):
    start, end = totals(before), totals(after)
    groups = sorted(start.keys() | end.keys())
    by_group = {
        label: {field: end.get(label, {}).get(field, 0) - start.get(label, {}).get(field, 0)
                for field in FIELDS}
        for label in groups
    }
    retained_deltas = []
    for address in before.keys() & after.keys():
        start_label, start_fields = before[address]
        end_label, end_fields = after[address]
        if start_label == end_label:
            delta = end_fields["Anonymous"] - start_fields["Anonymous"]
            if delta:
                retained_deltas.append({"class": end_label, "anonymous_delta_kib": delta})
    retained_deltas.sort(key=lambda row: (-abs(row["anonymous_delta_kib"]), row["class"]))
    return {
        "baseline_by_class_kib": start,
        "drain_by_class_kib": end,
        "delta_by_class_kib": by_group,
        "baseline_mappings": len(before),
        "drain_mappings": len(after),
        "new_mapping_starts": len(after.keys() - before.keys()),
        "removed_mapping_starts": len(before.keys() - after.keys()),
        "retained_mappings_with_anonymous_change": len(retained_deltas),
        "largest_retained_anonymous_changes": retained_deltas[:12],
    }


def run():
    import upstream_soak

    original = upstream_soak.read_smaps_rollup
    snapshots = []

    def attributed_rollup(pid):
        precise = original(pid)
        maps, elapsed_ms = read_mappings(pid)
        snapshots.append((maps, elapsed_ms))
        if len(snapshots) == 2:
            report = compare(snapshots[0][0], snapshots[1][0])
            report["read_ms"] = [sample[1] for sample in snapshots]
            out = Path(sys.argv[sys.argv.index("--out") + 1]).with_name("anon-attribution.json")
            with out.open("x", encoding="utf-8") as file:
                json.dump(report, file, indent=2, allow_nan=False)
                file.write("\n")
        return precise

    upstream_soak.read_smaps_rollup = attributed_rollup
    return upstream_soak.main()


if __name__ == "__main__":
    raise SystemExit(run())
