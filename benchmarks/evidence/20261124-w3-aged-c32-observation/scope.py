"""Keep the original replay prefix through the C32 cell with invalid timing."""

from hashlib import sha256
import json
from pathlib import Path


path = Path("schedules.json")
raw = path.read_bytes()
if sha256(raw).hexdigest() != "7921e63069ddd4b58736558af62590b5f340083ee129d25aa54eeefbdf78c116":
    raise ValueError("original arrival schedules differ")
schedules = json.loads(raw)["schedules"]
prefix = ("rag-c8-s1-vllm-arrivals", "rag-c8-s2-vllm-arrivals", "c32-riley-arrivals")
scoped = {}
for gpu, cells in schedules.items():
    if tuple(cells)[:len(prefix)] != prefix:
        raise ValueError("requested C32 case would have a different warmup sequence")
    scoped[gpu] = {label: cells[label] for label in prefix}
path.write_text(json.dumps({"schedules": scoped}, indent=2) + "\n")
