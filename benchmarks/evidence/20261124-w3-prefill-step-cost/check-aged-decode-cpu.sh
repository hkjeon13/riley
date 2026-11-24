#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(cd "$here/../../.." && pwd -P)
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
git -C "$repo" archive 0511b9280 Cargo.toml Cargo.lock rust-toolchain.toml crates kernels third_party |
    tar -xf - -C "$scratch"
python3 -B "$here/reserve-aged-decode.py" "$scratch"
cp "$here/aged-decode-reserve.rs" "$scratch/crates/riley-scheduler/tests/aged_decode_reserve.rs"
cd "$scratch"
for reserve in 0 1; do
    printf 'reserve=%s\n' "$reserve"
    RILEY_W3_RESERVE_AGED_DECODE="$reserve" cargo test --offline --quiet -p riley-scheduler
done
