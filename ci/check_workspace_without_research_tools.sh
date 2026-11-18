#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
scratch_parent=${TMPDIR:-/tmp}
scratch=$(mktemp -d "$scratch_parent/riley-production-workspace.XXXXXX")

cleanup() {
    case "$scratch" in
        "$scratch_parent"/riley-production-workspace.*) rm -rf -- "$scratch" ;;
        *) echo "refusing to remove unexpected scratch path: $scratch" >&2 ;;
    esac
}
trap cleanup EXIT HUP INT TERM

(
    cd "$repo_root"
    # Copy the production build inputs, including its source-contract fixtures,
    # rather than copying local checkpoints, benchmark dumps and tool targets.
    # A new external include must be admitted here or the isolated build fails.
    tar \
        -cf "$scratch/source.tar" \
        Cargo.toml Cargo.lock rust-toolchain.toml rustfmt.toml LICENSE README.md \
        crates kernels third_party ci/release/release_common.py \
        benchmarks/release benchmarks/evidence/20261022-phase0/native-extra.json
)
tar -xf "$scratch/source.tar" -C "$scratch"
rm "$scratch/source.tar"

test ! -e "$scratch/tools/python"
test ! -e "$scratch/tools/native"
test ! -e "$scratch/experiments/triton"

(
    cd "$scratch"
    cargo metadata --locked --format-version 1 --no-deps >/dev/null
    cargo check --locked --workspace --all-targets --no-default-features
)

echo "workspace metadata and CPU build passed without research-tool directories"
