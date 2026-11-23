#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
package="$evidence/w4-schema-ids-20261123.ZsUYnqkm"
cpu="$evidence/w5-parallel-shards-20261124.iMsJcmnF"
w5="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
python="$root/tools/vllm-venv/bin/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
export CARGO_TARGET_DIR="$stage/target" CUDA_VISIBLE_DEVICES= RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=2 CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 PYTHONDONTWRITEBYTECODE=1
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$cpu" && sha256sum --strict -c dispatch.sha256)
printf '%s  %s\n' 7dc88437ba28ea18ca1c0c11b9b3cd85324110da913ce1b10c521da60ce54dcb "$package/source.tar" | sha256sum --strict -c
(cd "$w5" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
cp -- "$w5/harness/benchmarks/live/"{load.py,api_check.py} .
printf '%s  %s\n' b1e4b523d58842b746044d5d4b9b6a91fe7a2470ab69d9842343228fd27428f8 load.py | sha256sum --strict -c
export PYTHONPATH="$stage:$w5/harness/ci:$package/src/ci"
"$python" -B -c 'from inspect import signature; from load import one; import check_distribution_gpu
assert "capture_token_ids" in signature(one).parameters'
sha256sum load.py api_check.py >runtime-tools.sha256
mkdir src gpu6 gpu7
tar -xf "$package/source.tar" -C src
/usr/bin/python3 -B - "$(cat "$cpu/affinity.txt")" <<'PY' >affinity.txt
from collections import defaultdict
import os, subprocess, sys

allowed = os.sched_getaffinity(0)
reserved = set(map(int, sys.argv[1].split(",")))
hosts = defaultdict(dict)
rows = subprocess.check_output(["lscpu", "--parse=CPU,CORE,SOCKET,ONLINE"], text=True).splitlines()
excluded = set()
for line in rows:
    if line.startswith("#"):
        continue
    cpu, core, socket, online = line.split(",")
    if int(cpu) in reserved:
        excluded.add((socket, core))
    if int(cpu) in allowed and online == "Y":
        hosts[socket].setdefault(core, int(cpu))
for socket, cores in sorted(hosts.items(), reverse=True):
    choices = [cpu for core, cpu in cores.items() if (socket, core) not in excluded]
    if len(choices) >= 4:
        print(f"{choices[0]},{choices[1]} {choices[2]},{choices[3]}")
        break
else:
    raise SystemExit("expected four distinct available cores")
PY
read -r cpus6 cpus7 <affinity.txt
build() {
    local engine=$1
    printf 'BUILD_START %s %s\n' "$engine" "$(date -u +%FT%TZ)"
    (cd src && taskset -c "$cpus6,$cpus7" cargo build --offline --locked --release -p riley-server --features server,hf-exact-probe) >"build-$engine.log" 2>&1
    install -m 700 target/release/riley "riley-$engine"
    ldd "riley-$engine" >"ldd-$engine.txt"
    ! grep -q 'not found' "ldd-$engine.txt"
    printf 'BUILD_END %s %s\n' "$engine" "$(date -u +%FT%TZ)"
}
(cd src && sha256sum --strict -c "$cpu/baseline-inputs.sha256")
build baseline
(cd src && git apply --check "$cpu/overlap.patch" && git apply "$cpu/overlap.patch" &&
    sha256sum --strict -c "$cpu/candidate-inputs.sha256")
build candidate
sha256sum riley-{baseline,candidate} >server-binaries.sha256
test "$(cut -d' ' -f1 server-binaries.sha256 | sort -u | wc -l)" -eq 2
deadline=$((SECONDS + 1800))
until test -s "$cpu/campaign.exit"; do
    if ! ps -p "$(cat "$cpu/campaign.pid")" -o args= | grep -Fq 'bash campaign.sh' ||
        (( SECONDS >= deadline )); then
        test -s "$cpu/campaign.exit" && break
        exit 75
    fi
    sleep 5
done
test "$(cat "$cpu/campaign.exit")" = 0
(cd "$cpu" && sha256sum --strict -c result.sha256 binary.sha256 >/dev/null)
reservations() {
    for index in 6 7; do
        "${control[@]}" --index "$index" status >"gpu$index/$1.json"
    done
}
arm() {
    local index=$1 engine=$2 affinity=$3 rc=0
    printf 'START gpu%s %s %s\n' "$index" "$engine" "$(date -u +%FT%TZ)"
    "${control[@]}" --index "$index" run -- \
        taskset -c "$affinity" "$python" -B serve.py "$index" "$engine" >"gpu$index/$engine.log" 2>&1 || rc=$?
    printf 'END gpu%s %s %s exit=%s\n' "$index" "$engine" "$(date -u +%FT%TZ)" "$rc"
    return "$rc"
}
round() {
    local left right rc=0
    arm 6 "$1" "$cpus6" &
    left=$!
    arm 7 "$2" "$cpus7" &
    right=$!
    wait "$left" || rc=1
    wait "$right" || rc=1
    return "$rc"
}
reservations before
rc=0
if round baseline candidate; then
    reservations between
    round candidate baseline || rc=1
else
    rc=1
fi
reservations after
find gpu6 gpu7 -type f -print0 | sort -z | xargs -0 sha256sum >result.sha256
sha256sum server-binaries.sha256 runtime-tools.sha256 affinity.txt >>result.sha256
exit "$rc"
