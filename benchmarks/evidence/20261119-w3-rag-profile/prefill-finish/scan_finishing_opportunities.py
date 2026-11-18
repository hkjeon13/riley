"""Count potential finishing requests in recorded 14B prefill batches."""
from argparse import ArgumentParser
import collections
import gzip
import json
from itertools import groupby
from pathlib import Path

parser = ArgumentParser(description=__doc__)
parser.add_argument("trace", type=Path)
parser.add_argument("--aging-threshold-ns", type=int, required=True)
args = parser.parse_args()
root = args.trace
server = json.loads((root / "plan.json").read_bytes())["server"]
budget = int(server[server.index("--batch-token-budget") + 1])
chunk = int(server[server.index("--prefill-chunk-tokens") + 1])
events = []
prompt_sizes = {}
with gzip.open(root/"server.log.gz","rt") as stream:
    for line in stream:
        if not line.startswith("W3-PHASE "):
            continue
        _, ident, stage, stamp, *fields = line.split()
        values = tuple(map(int,fields))
        events.append((ident,stage,int(stamp),values))
        if stage == "prefill_start" and values[2]:
            assert ident not in prompt_sizes
            prompt_sizes[ident] = values[1]

scenarios = {}
for path in root.glob("*.json"):
    d=json.loads(path.read_bytes())
    if "measured" in d:
        scenarios.update({row["request_id"]:path.stem for row in d["measured"]})
analysis = collections.defaultdict(collections.Counter)
# The recorded admitted timestamp and GPU-return timestamp approximate ready_since_ns.
# Measure the conditions on the trace without replaying service time or policy.
ready = {}
for stage, group in groupby(events, lambda row:row[1]):
    batch = list(group)
    if stage == "admitted":
        for ident, _, stamp, _ in batch:
            assert ident in prompt_sizes
            ready[ident]=(stamp,prompt_sizes[ident])
    elif stage == "prefill_return":
        epoch=batch[-1][2]
        for ident, _, _, fields in batch:
            remaining=prompt_sizes[ident]-fields[0]
            if remaining:
                ready[ident]=(epoch,remaining)
            else:
                del ready[ident]
    elif stage == "prefill_start":
        wants = {scenarios[row[0]] for row in batch if row[0] in scenarios}
        if not wants: continue
        label,=wants
        first,_,stamp,fields=batch[0]
        decode=fields[4]
        analysis[label]["batches"]+=1
        neighbors=sorted((when,k,remaining) for k,(when,remaining) in ready.items())
        available = min(budget - decode, chunk)
        matches=neighbors[0][1]==first
        if not matches:
            analysis[label]["trace_front_or_tie_differs"]+=1
        if decode==0:
            analysis[label]["prefill_only_batches"]+=1
        if len(neighbors)>1:
            analysis[label]["prefill_batches_with_other_ready"]+=1
        if ready[first][1] > available and any(k!=first and n<=available for _,k,n in neighbors):
            analysis[label]["finishing_alternative"]+=1
            if decode==0:
                analysis[label]["finishing_alternative_no_decode"]+=1
            if stamp - neighbors[0][0] < args.aging_threshold_ns:
                analysis[label]["finishing_alternative_within_aging"]+=1
                if decode==0:
                    analysis[label]["finishing_alternative_no_decode_within_aging"]+=1

summary = {
    "interpretation": "Recorded admit and return times approximate scheduler readiness; this is eligibility screening",
    "budget": budget,
    "chunk_limit": chunk,
    "aging_threshold_ns": args.aging_threshold_ns,
    "cells": {label: {
        key: value.get(key, 0) for key in (
            "batches", "prefill_only_batches", "prefill_batches_with_other_ready",
            "trace_front_or_tie_differs", "finishing_alternative", "finishing_alternative_no_decode",
            "finishing_alternative_within_aging", "finishing_alternative_no_decode_within_aging"
        )
    } for label, value in analysis.items()},
}
print(json.dumps(summary, indent=2))
