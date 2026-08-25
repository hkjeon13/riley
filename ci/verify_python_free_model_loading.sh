#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
scratch_parent=$(CDPATH= cd -- "${TMPDIR:-/tmp}" && pwd)
scratch=$(mktemp -d "$scratch_parent/rustinfer-python-free-model.XXXXXX")
target_dir=$scratch/target
empty_path=$scratch/empty-path
mkdir -p "$empty_path"

cleanup() {
    case "$scratch" in
        "$scratch_parent"/rustinfer-python-free-model.*) rm -rf -- "$scratch" ;;
        *) echo "refusing to remove unexpected scratch path: $scratch" >&2 ;;
    esac
}
trap cleanup EXIT HUP INT TERM

if grep -ERn \
    'std::process|process::Command|Command[[:space:]]*::[[:space:]]*new' \
    "$repo_root/crates/rustinfer-model/src"; then
    echo "production model loader must not invoke subprocesses" >&2
    exit 1
fi

(
    cd "$repo_root"
    CARGO_TARGET_DIR=$target_dir cargo test \
        --locked \
        --no-default-features \
        -p rustinfer-model \
        --test python_free_loading \
        --no-run
)

test_binary=
for candidate in "$target_dir"/debug/deps/python_free_loading-*; do
    case "$candidate" in
        *.d) continue ;;
    esac
    if test -f "$candidate" && test -x "$candidate"; then
        if test -n "$test_binary"; then
            echo "multiple Python-free model-loading test binaries found" >&2
            exit 1
        fi
        test_binary=$candidate
    fi
done

if test -z "$test_binary"; then
    echo "Python-free model-loading test binary was not produced" >&2
    exit 1
fi

if command -v ldd >/dev/null 2>&1; then
    if ldd "$test_binary" | grep -Eiq 'libpython|libpytorch|libtorch|libtransformers|libtriton'; then
        echo "model-loading test binary has a forbidden runtime dependency" >&2
        exit 1
    fi
elif command -v otool >/dev/null 2>&1; then
    if otool -L "$test_binary" | grep -Eiq 'libpython|libpytorch|libtorch|libtransformers|libtriton'; then
        echo "model-loading test binary has a forbidden runtime dependency" >&2
        exit 1
    fi
else
    echo "neither ldd nor otool is available for runtime dependency inspection" >&2
    exit 1
fi

/usr/bin/env -i \
    PATH="$empty_path" \
    RUST_BACKTRACE=1 \
    RUSTINFER_REQUIRE_EMPTY_PATH=1 \
    "$test_binary" \
    --exact python_free_process_loads_complete_checkpoint \
    --nocapture

echo "Python-free synthetic model loading passed with an empty executable PATH"
