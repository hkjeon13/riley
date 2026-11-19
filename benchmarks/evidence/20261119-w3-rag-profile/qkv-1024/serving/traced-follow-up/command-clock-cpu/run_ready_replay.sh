#!/usr/bin/env bash
set -euo pipefail
evidence=$(cd -- "$(dirname -- "$0")" && pwd -P)
prior=$(dirname "$evidence")
repository=$(git -C "$evidence" rev-parse --show-toplevel)
work=$(mktemp -d "${TMPDIR:-/tmp}/riley-ready-replay.XXXXXXXX")
revision=d2f3e411791875c4dc3c673a2cf33d2bc52f2650
mkdir "$work/src"
git -C "$repository" archive "$revision" Cargo.toml Cargo.lock rust-toolchain.toml crates kernels |
    tar -xf - -C "$work/src"
cd "$work/src"
export CARGO_TARGET_DIR="$work/target"
stock="$work/src/crates/riley-scheduler/src/scheduler.rs"
cp "$stock" "$work/scheduler-stock.rs"
python3.12 -B - "$prior/wave-admission.json" "$stock" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

witness, source = map(Path, sys.argv[1:])
actual = {name: sha256(source.with_name(name).read_bytes()).hexdigest()
          for name in ("scheduler.rs", "config.rs")}
if actual != json.loads(witness.read_bytes())["scheduler_inputs_sha256"]:
    raise ValueError("stock scheduler differs from the GPU scheduler inputs")
PY
cargo build --locked --offline -p riley-scheduler --message-format=json >"$work/stock-build.jsonl"
python3.12 -B "$evidence/compile_replay.py" "$work/stock-build.jsonl" "$work/target" \
    "$prior/replay_wave.rs" "$work/replay_stock"
git apply --check "$evidence/command-timestamp-scheduler.patch"
git apply "$evidence/command-timestamp-scheduler.patch"
cargo build --locked --offline -p riley-scheduler --message-format=json >"$work/probe-build.jsonl"
python3.12 -B "$evidence/compile_replay.py" "$work/probe-build.jsonl" "$work/target" \
    "$evidence/replay_ready_timestamp.rs" "$work/replay_timestamp"
cargo test --locked --offline -p riley-scheduler --lib >&2
python3.12 -B "$evidence/compare_ready_replay.py" "$evidence/cpu-ready-inputs.json" \
    "$prior/phase-gpu/verified-comparison.json" "$work/scheduler-stock.rs" \
    "$evidence/command-timestamp-scheduler.patch" "$work/replay_stock" "$work/replay_timestamp"
