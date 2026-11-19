#!/usr/bin/env bash
set -euo pipefail
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$run")
root=$(dirname "$evidence")
archive="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l/source.tar"
profile="$evidence/w3-qwen4-profile-r2-20261119.Ydyk8B8n"
preceding="$evidence/w4-hf-bias-partition-20261119.hj2XRENK"
hold="$evidence/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
cd "$run"
test "$(cat "$profile/campaign.exit")" = 0
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
mkdir src tools
printf '%s  %s\n' cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5 "$archive" | sha256sum -c -
tar -xf "$archive" -C src
for file in run_profile_retry.py profile_workload.py load.py profile_summary.py profile_process.py compare_runner.py; do
    cp -- "$profile/tools/$file" "tools/$file"
done
cp -- "$profile/holder.py" .
sha256sum -c tools-base.sha256
gzip -dc tools.patch.gz | git apply --check -
gzip -dc tools.patch.gz | git apply -
sha256sum -c tools.sha256
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4
export CUDAToolkit_ROOT=/usr/local/cuda-12.4 LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
export CARGO_TARGET_DIR="$run/target" RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4 PYTHONDONTWRITEBYTECODE=1
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
(
    cd src
    sha256sum -c "$run/source-base.sha256"
    gzip -dc "$run/source.patch.gz" | git apply --check -
    gzip -dc "$run/source.patch.gz" | git apply -
    sha256sum -c "$run/source.sha256"
    "$toolchain/cargo" build --locked --offline --release -p riley-server --features server,hf-exact-probe
) > build.log 2>&1
sha256sum target/release/riley > binary.sha256
for arm in riley stage-trace; do
    printf 'START %s %s\n' "$arm" "$(date -u +%FT%TZ)"
    status=0
    /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 run -- \
        /usr/bin/python3 -B "$run/tools/run_profile_retry.py" qwen14 "$arm" || status=$?
    if test -d "qwen14-$arm"; then
        /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status > "qwen14-$arm/idle-after.json"
    fi
    printf 'END %s %s exit=%d\n' "$arm" "$(date -u +%FT%TZ)" "$status"
    if (( status != 0 )); then exit "$status"; fi
done
