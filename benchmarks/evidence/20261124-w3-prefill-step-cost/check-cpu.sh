#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$here")
repo=$(cd "$evidence/../.." && pwd -P)
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
mkdir "$scratch/src"
git -C "$repo" archive 0511b9280 Cargo.toml Cargo.lock rust-toolchain.toml crates kernels third_party |
    tar -xf - -C "$scratch/src"
gzip -dc "$evidence/20261122-w3-c32-admission-observations/export-observations.patch.gz" |
    (cd "$scratch/src" && git apply --whitespace=error)
python3 -B "$here/instrument.py" "$scratch/src" \
    "$evidence/20261124-w3-rag-prefill-order/instrument.py" \
    "$evidence/20261124-w3-wide-prefill-order/instrument.py" \
    "$evidence/20261124-w3-rag-prefill-steps/instrument.py"
python3 -B - "$scratch/src" "$evidence" <<'PY'
from pathlib import Path
import sys

root, evidence = map(Path, sys.argv[1:])
source = (evidence / "20261124-w3-wide-prefill-order/scheduler_test.rs").read_text()
for old, new in (
    ("use riley_runtime::paged_kv::KvLayout;", '''#[allow(dead_code)]
#[path = "../../riley-server/src/rag_steps.rs"]
mod rag_steps;

use riley_runtime::paged_kv::KvLayout;'''),
    ("    let next_turn = plan(&mut scheduler, 31_000_000);", '''    let trace_start = rag_steps::clock();
    let next_turn = plan(&mut scheduler, 31_000_000);'''),
    ("    settle(&mut scheduler, &next_turn, 32_000_000);", '''    settle(&mut scheduler, &next_turn, 32_000_000);
    if let (Some(start), Some(end)) = (trace_start, rag_steps::clock()) {
        rag_steps::step(&next_turn, start, end, 1);
    }'''),
    ("        2 * u64::from(fifo)\n    );\n}", '''        2 * u64::from(fifo)
    );
    Scheduler::export_mixed_prefill_choices();
    rag_steps::export();
}'''),
):
    assert source.count(old) == 1
    source = source.replace(old, new)
(root / "crates/riley-scheduler/tests/prefill_step_cost.rs").write_text(source)
PY
for trace in 0 1; do
    for order in 0 1; do
        log="$scratch/test-$trace-$order.log"
        (cd "$scratch/src" && RILEY_W3_RAG_STEPS="$trace" RILEY_W3_PREFILL_FIFO="$order" \
            cargo test --offline --quiet -p riley-scheduler --test prefill_step_cost -- --nocapture) >"$log" 2>&1 ||
            { cat "$log"; exit 1; }
        python3 -B - "$log" "$trace" "$order" <<'PY'
from pathlib import Path
import re
import sys

log = Path(sys.argv[1]).read_text()
trace, order = map(int, sys.argv[2:])
assert "1 passed; 0 failed;" in log
if trace:
    assert re.findall(r"^W3-MIXED-CHOICES (.*)$", log, re.M) == ["2 0"]
    choices = re.findall(r"^W3-MIXED-CHOICE (.*)$", log, re.M)
    assert choices == [f"3 {order} 3 3500 0 2 3500 1023 1023",
                       f"3 {order} 3 3500 0 2 3072 1023 1023"], choices
    assert re.findall(r"^W3-MIXED-DECODE (.*)$", log, re.M) == ["3 1:3"]
    assert re.findall(r"^W3-RAG-SNAPSHOT (.*)$", log, re.M) == ["1 0"]
else:
    assert "W3-MIXED-" not in log and "W3-RAG-" not in log
print(f"trace={trace} prefill_order={order} plan, output shape and trace markers passed")
PY
    done
done
