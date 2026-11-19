#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
long="$evidence/w4-long-20261119.qIEtmCwO"
holdout="$evidence/w4-holdout-20261119.cgmtjz4x"
previous="$evidence/w3-qkv-serving-20261119.MhydzNsX"
preceding="$evidence/w5-tcache-only-20261119.G3IuT0Wh"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$stage/campaign.exit.tmp"; mv -- "$stage/campaign.exit.tmp" "$stage/campaign.exit"' EXIT
cd "$stage"
sha256sum -c dispatch.sha256
printf 'WAIT %s %s\n' "$preceding" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 14400))
until test -s "$preceding/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$preceding/campaign.pid")" | grep -Fq -- "$preceding/campaign.sh"; then
        test -s "$preceding/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 30
done
printf 'PREDECESSOR_EXIT %s %s\n' "$(cat "$preceding/campaign.exit")" "$(date -u +%FT%TZ)"
test "$(cat "$base/campaign.exit")" = 0
holder_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}
for index in 6 7; do holder_status "$index" >"initial-reservation-$index.json"; done
/usr/bin/python3 -B - <<'PY'
import json
from pathlib import Path

for index, pid in ((6, 2431228), (7, 2431237)):
    state = json.loads(Path(f"initial-reservation-{index}.json").read_bytes())
    if (state["physical_index"] != index or state["pid"] != pid or state["phase"] != "idle"
            or state["other_processes"] or state["allocation_bytes"] != state["torch_reserved_bytes"]
            or state["allocation_bytes"] != 20 * 1024**3):
        raise ValueError(f"GPU {index} is not in its original 20 GiB idle reservation")
PY
(cd "$evidence" && sha256sum -c "$stage/inputs.sha256")
mkdir src tools fixtures
for suite in long holdout; do
    folder="$stage/fixtures/20261119-w4-$suite"
    if [ "$suite" = long ]; then origin=$long; else origin=$holdout; fi
    mkdir -p "$folder/riley" "$folder/hf"
    gzip -n -c "$origin/riley/capture.json" >"$folder/riley/capture.json.gz"
    cp -- "$origin/hf/reference.json" "$folder/hf/"
done
cp -- "$holdout/holdout.json.gz" fixtures/20261119-w4-holdout/
cp -- "$previous/holder.py" .
cp -- "$previous/tools/compare_runner.py" "$long/tools/run_accuracy.py" "$holdout/tools/long_accuracy.py" capture_decode_rows.py tools/
sha256sum -c helpers.sha256 runner-base.sha256
gzip -dc runner.patch.gz | git apply --check -
gzip -dc runner.patch.gz | git apply -
sha256sum -c runner-candidate.sha256
printf '%s  %s\n' 655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f \
    "$root/models/Qwen2.5-7B-Instruct/riley-checkpoint.json" | sha256sum -c
printf '%s  %s\n' cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5 \
    "$base/source.tar" | sha256sum -c
tar -xf "$base/source.tar" -C src --no-same-owner
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export CARGO_TARGET_DIR="$stage/target" RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4 PYTHONDONTWRITEBYTECODE=1 W3_GPU_INDEX=7
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
(
    cd src
    sha256sum -c "$stage/engine-base.sha256"
    gzip -dc "$stage/capture-decode.patch.gz" | git apply --check -
    gzip -dc "$stage/capture-decode.patch.gz" | git apply -
    sha256sum -c "$stage/engine-patched.sha256"
    "$toolchain/cargo" build --locked --offline --release -p riley-server --features server,hf-exact-probe
) >build.log 2>&1
sha256sum target/release/riley >binary.sha256
holder_status 7 >idle-before.json
printf 'START riley %s\n' "$(date -u +%FT%TZ)"
status=0
/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
    /usr/bin/python3 -B "$stage/tools/run_accuracy.py" riley || status=$?
holder_status 7 >idle-after.json
printf 'END riley %s exit=%d\n' "$(date -u +%FT%TZ)" "$status"
test "$status" = 0
/usr/bin/python3 -B - <<'PY'
from pathlib import Path
import json

read = lambda name: json.loads(Path(name).read_bytes())
a,b = read('idle-before.json'),read('idle-after.json')
if not all(a[key] == b[key] for key in a if key != 'updated_ns'):
    raise ValueError('the original 20 GiB holder was not restored')
if len(read('riley/client.log')['cases']) != 8 or len(list(Path('riley/rows').glob('*.bf16'))) != 8:
    raise ValueError('wrong output row count')
PY
sha256sum riley/{receipt.json,client.log,server.log} idle-{before,after}.json riley/rows/*.bf16 >result.sha256
