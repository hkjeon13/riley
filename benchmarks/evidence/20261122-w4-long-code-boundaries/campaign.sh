#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
build="${1:?diagnostic build required}"
[[ $build =~ ^"$evidence"/w4-long-code-build-20261122\.[[:alnum:]]+$ ]]
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
model="$root/models/Qwen2.5-7B-Instruct"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$evidence" && sha256sum --strict -c "$stage/captures.sha256")
while [ ! -e "$build/build.exit" ]; do
    pid=$(cat "$build/build.pid")
    test "$(readlink "/proc/$pid/cwd")" = "$build"
    ps -p "$pid" -o stat= | awk 'NR==1 && $1 !~ /^Z/ { live=1 } END { exit !live }'
    sleep 5
done
(cd "$build" && test "$(cat build.exit)" = 0 && sha256sum --strict -c build-inputs.sha256 build-result.sha256)
printf '%s\n' "$build" >build-path.txt
export PYTHONPATH="$build:$build/tools" PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
image=sha256:0a907408ec73fc8b9b00f440037305ce1bfb00e05bd8d3ef3096fcee59849ea1
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
docker run --rm --network none --env NVIDIA_VISIBLE_DEVICES=void --user "$(id -u):$(id -g)"     --mount "type=bind,src=$model,dst=/models,readonly" "$image" riley-checkpoint verify --checkpoint /models >checkpoint-verify.log
grep -qx 'checkpoint verify: verified' checkpoint-verify.log
reserve() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}
for index in 6 7; do reserve "$index" >"gpu$index-before.json"; done
for mode in baseline attention31 fp32_down21; do
    mkdir "$mode"
    reserve 7 >"$mode/reservation-before.json"
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    rc=0
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run --         "$root/tools/vllm-venv/bin/python" -B "$stage/run_case.py" "$stage" "$build" "$mode" >"$mode/driver.log" 2>&1 || rc=$?
    reserve 7 >"$mode/reservation-after.json"
    printf 'END %s %s rc=%s\n' "$mode" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
for index in 6 7; do reserve "$index" >"gpu$index-after.json"; done
sha256sum build-path.txt checkpoint-verify.log gpu{6,7}-{before,after}.json     {baseline,attention31,fp32_down21}/{server.log,receipt.json,shutdown.json,reservation-{before,after}.json,trace/{prompt-final,decode-7}/{index.tsv,rows.bf16}}     >result.sha256
