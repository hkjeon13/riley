#!/usr/bin/env bash
set -euo pipefail
umask 077
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$run")
root=$(dirname "$evidence")
source="$evidence/w4-schema-build-20261119.QWnmhslP/src"
previous="$evidence/w3-prefill-continuation-20261119.QjxM6xaM"
hold="$evidence/w2-sm17-20261118.6UMfeHh2"
preceding="$evidence/w5-anon-mappings-20261119.sZulouQC"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
cd "$run"
test "$(cat "$previous/campaign.exit")" = 0
sha256sum -c dispatch.sha256
printf 'WAIT %s %s\n' "$preceding" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 14400))
until test -s "$preceding/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$preceding/campaign.pid")" | grep -Fq -- "$preceding/campaign.sh"; then
        test -s "$preceding/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 25
done
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status |
    /usr/bin/python3 -c 'import json,sys; lease=json.load(sys.stdin); assert lease["phase"]=="idle"'
mkdir src tools
cp -a -- "$source/." src/
for file in run_profile_retry.py profile_workload.py load.py profile_summary.py profile_process.py compare_runner.py; do
    cp -- "$previous/tools/$file" "tools/$file"
done
cp -- "$previous/holder.py" .
sha256sum -c "$previous/tools.sha256"
sha256sum -c tools-base.sha256
git apply --check tools.patch
git apply tools.patch
sha256sum -c tools.sha256
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4
export CUDAToolkit_ROOT=/usr/local/cuda-12.4
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
) >build.log 2>&1
sha256sum target/release/riley >binary.sha256
for arm in riley finishing; do
    /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 status >"idle-before-$arm.json"
    printf 'START %s %s\n' "$arm" "$(date -u +%FT%TZ)"
    status=0
    /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 run -- \
        /usr/bin/python3 -B "$run/tools/run_profile_retry.py" qwen14 "$arm" || status=$?
    if test -d "qwen14-$arm"; then
        /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 status >"qwen14-$arm/idle-after.json"
    fi
    printf 'END %s %s exit=%d\n' "$arm" "$(date -u +%FT%TZ)" "$status"
    if (( status != 0 )); then exit "$status"; fi
done
