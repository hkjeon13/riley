#!/usr/bin/env bash
set -euo pipefail
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$run")
root=$(dirname "$evidence")
first="$evidence/w5-gpu-memory-faults-20261119.u0cxUF9V"
source="$first/src"
preceding="$evidence/w5-allocator-trace-20261119.Yow8N2MR"
hold="$evidence/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
cd "$run"
test "$(cat "$first/campaign.exit")" = 101
sha256sum -c dispatch.sha256
printf 'WAIT %s %s\n' "$preceding" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 36000))
until test -s "$preceding/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$preceding/campaign.pid")" | grep -Fq -- "$preceding/campaign.sh"; then
        test -s "$preceding/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 30
done
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4
export CUDAToolkit_ROOT=/usr/local/cuda-12.4 LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
export CARGO_TARGET_DIR="$run/target" RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4 PYTHONDONTWRITEBYTECODE=1
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
mkdir src
cp -a -- "$source/." src/
(
    cd src
    sha256sum -c "$first/source-inputs.sha256"
    gzip -dc "$run/header.patch.gz" | git apply --check -
    gzip -dc "$run/header.patch.gz" | git apply -
    sha256sum -c "$run/header.sha256"
) > source-verification.log
(
    cd src
    "$toolchain/cargo" test --locked --offline --release -p riley-cuda \
        --no-default-features --features cuda-test-fault-injection --test memory_fault_injection_gpu \
        --no-run --message-format=json
) > build.jsonl 2>build.log
binary=$(/usr/bin/python3 -B - build.jsonl <<'PYTHON'
import json
import sys
from pathlib import Path

rows = map(json.loads, Path(sys.argv[1]).read_text().splitlines())
executables = [row["executable"] for row in rows if row.get("reason") == "compiler-artifact"
               and row["target"]["name"] == "memory_fault_injection_gpu" and row["executable"]]
binary, = executables
print(binary)
PYTHON
)
test -x "$binary"
sha256sum -- "$binary" > binary.sha256
unset RILEY_CUDA_MEMORY_FAULT_CHILD
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 status > idle-before.json
printf 'START memory fault cases %s\n' "$(date -u +%FT%TZ)"
result=0
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 run -- \
    /usr/bin/timeout 120 "$binary" --ignored --exact memory_fault_cases_are_subprocess_isolated --nocapture \
    > tests.log 2>&1 || result=$?
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 status > idle-after.json
printf 'END memory fault cases %s exit=%d\n' "$(date -u +%FT%TZ)" "$result"
exit "$result"
