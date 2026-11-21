#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
    printf '%s\n' 'Pass paths to compile_replay.py, replay_wave.rs and cpu-ready-inputs.json.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
repository=$(git rev-parse --show-toplevel)
compile=$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")
stock_source=$(cd "$(dirname "$2")" && pwd -P)/$(basename "$2")
qwen_inputs=$(cd "$(dirname "$3")" && pwd -P)/$(basename "$3")
work=$(mktemp -d "${TMPDIR:-/tmp}/riley-prefill-dispatch.XXXXXXXX")
revision=3304b6f355fed2e75cc37a715d7f9d68f531ebaf
mkdir "$work/src"
git -C "$repository" archive "$revision" Cargo.toml Cargo.lock rust-toolchain.toml crates kernels |
    tar -xf - -C "$work/src"
cd "$work/src"
export CARGO_TARGET_DIR="$work/target"
cargo build --locked --offline -p riley-scheduler --message-format=json >"$work/stock-build.jsonl"
python3.12 -B "$compile" "$work/stock-build.jsonl" "$work/target" "$stock_source" "$work/replay_stock"
gzip -dc "$here/early-prefill-priority.patch.gz" >"$work/early-prefill-priority.patch"
git apply --check "$work/early-prefill-priority.patch"
git apply "$work/early-prefill-priority.patch"
cargo build --locked --offline -p riley-scheduler --message-format=json >"$work/diagnostic-build.jsonl"
python3.12 -B "$compile" "$work/diagnostic-build.jsonl" "$work/target" \
    "$here/replay_early_priority.rs" "$work/replay_diagnostic"
cargo test --locked --offline -p riley-scheduler --lib >&2
python3.12 -B "$here/check_early_priority.py" "$here/coder-phase-inputs.json" "$qwen_inputs" \
    "$here/confirmation-inputs.json" "$work/replay_stock" "$work/replay_diagnostic" "$work/early-prefill-priority.patch"
