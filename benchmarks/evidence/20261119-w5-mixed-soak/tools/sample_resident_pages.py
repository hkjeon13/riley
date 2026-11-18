"""Record Linux RSS components for the Riley server owned by one soak campaign."""

from argparse import ArgumentParser
import json
from pathlib import Path
import time


FIELDS = ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmHWM")


def identity(process):
    fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
    return fields[0], int(fields[19])


def descendants(process):
    queue = [process]
    while queue:
        current = queue.pop()
        try:
            children = (current / "task" / current.name / "children").read_text().split()
        except FileNotFoundError:
            continue
        queue.extend(Path("/proc") / child for child in children)
        if current != process:
            yield current


def server(campaign, binary):
    for process in descendants(campaign):
        try:
            if ((process / "exe").resolve(strict=True) == binary
                    and (process / "cmdline").read_bytes().split(b"\0")[1:2] == [b"serve"]):
                yield process
        except FileNotFoundError:
            continue


def sample(process):
    lines = (process / "status").read_text().splitlines()
    counts = {key: int(value.split()[0]) for line in lines
              for key, _, value in [line.partition(":")] if key in FIELDS}
    if counts.keys() != set(FIELDS):
        raise ValueError("Linux status is missing resident-page fields")
    return counts


def observe(run, binary, output):
    campaign = Path("/proc") / (run / "campaign.pid").read_text().strip()
    candidates = list(server(campaign, binary.resolve(strict=True)))
    if len(candidates) != 1:
        raise ValueError("expected exactly one dedicated server child of this campaign")
    process, = candidates
    state, start_ticks = identity(process)
    if state in ("Z", "X"):
        raise ValueError("server already exited")
    with output.open("x", encoding="utf-8", buffering=1) as handle:
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            try:
                state, current_start = identity(process)
                if state in ("Z", "X") or current_start != start_ticks:
                    break
                counts = sample(process)
                if identity(process)[1] != start_ticks:
                    break
            except FileNotFoundError:
                break
            handle.write(json.dumps({"unix_seconds": time.time(), "pid": int(process.name),
                                     "start_ticks": start_ticks, "kib": counts}) + "\n")
            time.sleep(15)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("run", "binary", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    observe(args.run, args.binary, args.output)
