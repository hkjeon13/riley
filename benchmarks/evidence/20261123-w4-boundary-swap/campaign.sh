#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
build="${1:?diagnostic build required}"
[[ $build =~ ^"$evidence"/w4-boundary-swap-build-20261123\.[[:alnum:]]+$ ]]
prior="$evidence/w4-long-code-gpu-20261122.QAbtnxKV"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
model="$root/models/Qwen2.5-7B-Instruct"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$prior"
    printf '076cf1619f8011cabd6468f6ec92425b65bea075a8e14008fb57d860570ced29  result.sha256\n' | sha256sum --strict -c
    printf 'b2b03862c274281ede7ee70a8281d84063be79d31e3ae31256f1e3c82c612ee7  captures.sha256\n' | sha256sum --strict -c
    sha256sum --strict -c result.sha256
    grep '  run_case.py$' dispatch.sha256 | sha256sum --strict -c)
(cd "$evidence" && sha256sum --strict -c "$prior/captures.sha256")
while [ ! -e "$build/build.exit" ]; do
    pid=$(cat "$build/build.pid")
    test "$(readlink "/proc/$pid/cwd")" = "$build"
    ps -p "$pid" -o stat= | awk 'NR==1 && $1 !~ /^Z/ { live=1 } END { exit !live }'
    sleep 5
done
(cd "$build" && test "$(cat build.exit)" = 0 && sha256sum --strict -c build-inputs.sha256 build-result.sha256 patched-source.sha256)
printf '%s\n' "$build" >build-path.txt
prior_build=$(cat "$prior/build-path.txt")
export PYTHONPATH="$prior:$prior_build:$prior_build/tools" PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
python="$root/tools/vllm-venv/bin/python"
image=sha256:0a907408ec73fc8b9b00f440037305ce1bfb00e05bd8d3ef3096fcee59849ea1
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
docker run --rm --network none --env NVIDIA_VISIBLE_DEVICES=void --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$model,dst=/models,readonly" "$image" riley-checkpoint verify --checkpoint /models >checkpoint-verify.log
grep -qx 'checkpoint verify: verified' checkpoint-verify.log
control() {
    local index="$1"
    shift
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" "$@"
}
run_arm() {
    local comparison="$1" role="$2" gpu="$3" rc=0
    local out="$comparison/$role"
    control "$gpu" status >"$out/reservation-before.json"
    control "$gpu" run -- "$python" -B "$stage/trial.py" run "$stage" "$build" "$comparison" "$role" >"$out/driver.log" 2>&1 || rc=$?
    printf '%d\n' "$rc" >"$out/worker.exit"
    control "$gpu" status >"$out/reservation-after.json"
    return "$rc"
}
for index in 6 7; do control "$index" status >"gpu$index-before.json"; done
for comparison in attention31 fp32_down21; do
    mkdir -p "$comparison"/{baseline,candidate}
    printf 'START %s %s\n' "$comparison" "$(date -u +%FT%TZ)"
    run_arm "$comparison" baseline 6 & left=$!
    run_arm "$comparison" candidate 7 & right=$!
    left_rc=0
    right_rc=0
    wait "$left" || left_rc=$?
    wait "$right" || right_rc=$?
    printf 'END %s %s %s %s\n' "$comparison" "$(date -u +%FT%TZ)" "$left_rc" "$right_rc"
    test "$left_rc" = 0 && test "$right_rc" = 0 || exit 1
done
for index in 6 7; do control "$index" status >"gpu$index-after.json"; done
"$python" -B "$stage/trial.py" verify "$stage" "$build" >verified.json.tmp
mv -- verified.json.tmp verified.json
{
    printf '%s\n' build-path.txt checkpoint-verify.log verified.json gpu{6,7}-{before,after}.json
    find attention31 fp32_down21 -type f ! -name calibrated -print
} | LC_ALL=C sort | xargs sha256sum >result.sha256
