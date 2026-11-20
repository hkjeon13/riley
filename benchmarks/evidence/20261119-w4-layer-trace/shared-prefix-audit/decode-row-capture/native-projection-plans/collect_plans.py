"""Verify captured rows are unchanged and pair plans with the pinned decode positions."""
from hashlib import sha256
import json
from pathlib import Path
import sys


stage, source = map(Path, sys.argv[1:])
trace = stage / "riley"
previous = source / "riley"
receipt = json.loads((trace / "client.log").read_bytes())
prior_receipt = json.loads((previous / "client.log").read_bytes())
assert receipt == prior_receipt
positions = {}
for case, current in receipt.items():
    for name in ("rows.bf16", "index.tsv"):
        assert (trace / "rows" / case / name).read_bytes() == (previous / "rows" / case / name).read_bytes()
    assert current["output_match"]
    positions[(current["position"], current["token_id"])] = case

specs = {"o_proj": (1, 3584, 3584), "down_proj": (1, 3584, 18944)}
cases = {case: {} for case in receipt}
for line in (trace / "server.log").read_text().splitlines():
    if not line.startswith("W4_NATIVE_PROJECTION "):
        continue
    _, name, *fields = line.split()
    plan = dict(item.split("=", 1) for item in fields)
    current = cases[positions[(int(plan["position"]), int(plan["token"]))]]
    assert name not in current and tuple(int(plan[dim]) for dim in ("m", "n", "k")) == specs[name]
    assert (plan["backend"], plan["sm"], plan["cuda"], plan["cublaslt"]) == ("1", "8.0", "12040", "120405")
    current[name] = {key: int(value) if value.isdigit() else value for key, value in plan.items()}
assert len(cases) == 2 and all(set(row) == set(specs) for row in cases.values())
for name in specs:
    values = [{key: val for key, val in case[name].items() if key not in {"position", "token"}}
              for case in cases.values()]
    assert values[0] == values[1]
print(json.dumps({"cases": cases, "native_rows_reproduced": receipt,
                  "baseline_result_sha256": sha256((source / "result.sha256").read_bytes()).hexdigest(),
                  "binary_sha256": (stage / "binary.sha256").read_text().split()[0]}, indent=2))
