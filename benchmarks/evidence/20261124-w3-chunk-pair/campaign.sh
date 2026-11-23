#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
previous="$evidence/w3-package-coder14-20261124.GgTQVni3"
w5="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
package="$evidence/w4-schema-ids-20261123.ZsUYnqkm"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
runner="$root/tools/vllm-venv/bin/python"
first="$evidence/w3-chunk-pair-20261124.IMTJAGQS"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$previous"
    test "$(cat campaign.exit)" = 0
    printf 'fea2613b465f02faefb623f8e421cd384eecc8ab48cdf083b3d57b9ce6e0f2e4  dispatch.sha256\n' |
        sha256sum --strict -c
    sha256sum --strict -c dispatch.sha256 runtime.sha256 result.sha256 first-attempt.sha256 >/dev/null)
(cd "$root" && sha256sum --strict -c "$previous/pins.sha256")
(cd "$w5" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
test "$(cat "$package/campaign.exit")" = 0
(cd "$first"
    test "$(cat campaign.exit)" = 1
    printf '6608828eb6148037e5d86303704934667785c4451a4735d434c7d210ff6bc753  result.sha256\n' |
        sha256sum --strict -c
    sha256sum --strict -c result.sha256 >/dev/null)
sha256sum "$first"/{dispatch.sha256,campaign.exit,result.sha256,campaign.log} >first-attempt.sha256
mkdir tools gpu6 gpu7
cp -- "$previous"/{arm.py,holder.py,runtime.sha256,pins.sha256,checkpoint.log} .
cp -- "$previous"/tools/*.py tools/
sha256sum --strict -c runtime.sha256
cp -- "$w5/harness/benchmarks/live/load.py" tools/holdout_load.py
printf 'b1e4b523d58842b746044d5d4b9b6a91fe7a2470ab69d9842343228fd27428f8  tools/holdout_load.py\n' >holdout-tool.sha256
sha256sum --strict -c holdout-tool.sha256
git apply --check prefill-chunk.patch
git apply prefill-chunk.patch
sha256sum --strict -c arm.sha256
export PYTHONPATH="$stage/tools:$w5/harness/ci:$package/src/ci"
export PYTHONDONTWRITEBYTECODE=1
"$runner" -B - <<'PYTHON'
from inspect import signature
from holdout_load import one
from check_distribution_gpu import current_batch_shapes
assert "capture_token_ids" in signature(one).parameters
PYTHON
reservations() {
    for index in 6 7; do
        "${control[@]}" --index "$index" status >"gpu$index/$1.json"
    done
}
arm() {
    local index=$1 chunk=$2 rc=0
    printf 'START gpu%s chunk-%s %s\n' "$index" "$chunk" "$(date -u +%FT%TZ)"
    W3_GPU_INDEX="$index" "${control[@]}" --index "$index" run -- \
        "$runner" -B arm.py "$chunk" >"gpu$index/chunk-$chunk.log" 2>&1 || rc=$?
    printf 'END gpu%s chunk-%s %s exit=%s\n' "$index" "$chunk" "$(date -u +%FT%TZ)" "$rc"
    return "$rc"
}
round() {
    local left right rc=0
    arm 6 "$1" &
    left=$!
    arm 7 "$2" &
    right=$!
    wait "$left" || rc=1
    wait "$right" || rc=1
    return "$rc"
}
reservations before
rc=0
if round 512 1024; then
    reservations between
    round 1024 512 || rc=1
else
    rc=1
fi
reservations after
(cd "$root" && sha256sum --strict -c "$stage/pins.sha256")
find gpu6 gpu7 -type f \( -name '*.json' -o -name '*.log' \) -print0 |
    sort -z | xargs -0 sha256sum >result.sha256
sha256sum checkpoint.log runtime.sha256 arm.py pins.sha256 first-attempt.sha256 holdout-tool.sha256 >>result.sha256
exit "$rc"
