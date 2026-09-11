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
    tar \
        --exclude='./.git' \
        --exclude='./target' \
        --exclude='./benchmarks/results' \
        --exclude='./tools/python' \
        --exclude='./tools/native' \
        --exclude='./experiments/triton' \
        -cf - .
) | tar -xf - -C "$scratch"

test ! -e "$scratch/tools/python"
test ! -e "$scratch/tools/native"
test ! -e "$scratch/experiments/triton"

(
    cd "$scratch"
    cargo metadata --locked --format-version 1 --no-deps >/dev/null
    cargo check --locked --workspace --all-targets --no-default-features
)

echo "workspace metadata and CPU build passed without research-tool directories"
