#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
build="${1:?diagnostic build required}"
[[ $build =~ ^"$evidence"/w4-layer1-down-build-20261123\.[[:alnum:]]+$ ]]
parent="$evidence/w4-kv-history-gpu-20261123.G8mePjYa"
oracle="$evidence/w4-projection-oracle-20261120.RB5VX8Md"
prior="$evidence/w4-long-code-gpu-20261122.QAbtnxKV"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
model="$root/models/Qwen2.5-7B-Instruct"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
cp -- "$parent/verification/trial.py" kv_trial.py
cp -- "$parent/verification/result.sha256" kv_result.sha256
cp -- "$oracle/projection_oracle.py" projection_oracle.py
sha256sum --strict -c parent-inputs.sha256
(cd "$parent" && sha256sum --strict -c "$stage/kv_result.sha256")
cp -- "$parent/boundary_trial.py" boundary_trial.py
python3 -B prepare-runner.py kv_trial.py runner.py
printf 'aa92404ce44f601a992229858d993310a8ae776deed971fd69e57b87591c82f3  runner.py\n' | sha256sum --strict -c
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
export PYTHONPATH="$stage:$prior:$prior_build:$prior_build/tools" PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
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
    local role="$1" gpu="$2" rc=0
    control "$gpu" status >"$role/reservation-before.json"
    control "$gpu" run -- "$python" -B "$stage/runner.py" "$stage" "$build" "$role" >"$role/driver.log" 2>&1 || rc=$?
    printf '%d\n' "$rc" >"$role/worker.exit"
    control "$gpu" status >"$role/reservation-after.json"
    return "$rc"
}
for index in 6 7; do control "$index" status >"gpu$index-before.json"; done
"$python" -B -c 'from pathlib import Path; from boundary_trial import reservation; [reservation(Path(f"gpu{i}-before.json"), i) for i in (6, 7)]'
mkdir baseline candidate
printf 'START %s\n' "$(date -u +%FT%TZ)"
run_arm baseline 6 & left=$!
run_arm candidate 7 & right=$!
left_rc=0
right_rc=0
wait "$left" || left_rc=$?
wait "$right" || right_rc=$?
printf 'END %s %s %s\n' "$(date -u +%FT%TZ)" "$left_rc" "$right_rc"
for index in 6 7; do control "$index" status >"gpu$index-after.json"; done
test "$left_rc" = 0 && test "$right_rc" = 0
CUDA_VISIBLE_DEVICES= "$python" -B "$stage/verify.py" "$stage" "$build" "$model" >verified.json.tmp
mv -- verified.json.tmp verified.json
{
    printf '%s\n' build-path.txt boundary_trial.py runner.py kv_trial.py kv_result.sha256 projection_oracle.py checkpoint-verify.log verified.json gpu{6,7}-{before,after}.json
    find baseline candidate -type f ! -name calibrated -print
} | LC_ALL=C sort | xargs sha256sum >result.sha256
