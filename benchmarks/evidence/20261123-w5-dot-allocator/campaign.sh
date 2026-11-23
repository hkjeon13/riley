#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
stock="$evidence/w4-dot-package-20261123.nKyEd9x9"
variant="$evidence/w5-prefix-package-allocator-20261121.SNyhv9qr"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
run="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
prior="$evidence/w5-current-package-20261121.f6SEbjvA"
rag="$evidence/w5-current-rag-20261121.hvh3VS67"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
runner="$(dirname "$evidence")/tools/vllm-venv/bin/python"
cpu="$stock/cpu-evidence/report.json"
deb_sha=144eaa64b9da0319760c407ebe330b46d9d03ef9952fc067d48221cc9832e45c

cleanup() {
    local code=$?
    if test -n "${parent_tag:-}"; then docker image rm "$parent_tag" >parent-untag.log 2>&1 || true; fi
    printf '%s\n' "$code" >campaign.exit.tmp
    mv -- campaign.exit.tmp campaign.exit
}
trap cleanup EXIT
sha256sum --strict -c dispatch.sha256
(cd "$stock"
    test "$(cat campaign.exit)" = 0
    printf 'd412c16ac878610b3470352b1e6f97a7da408f68aa2d0e13072e8da134e202dc  result.sha256\n' | sha256sum --strict -c
    sha256sum --strict -c result.sha256 >/dev/null)
(cd "$variant"
    test "$(cat campaign.exit)" = 0
    printf 'c9da29e9f6e085bab55563eb6c4b2e61335af422137f5979f3956884f9e9de6a  context.sha256\n8bd1d7c49a56060cd2f2cb45eaeb375b1311260fa367c5b9d23fb5847b132c21  allocator-dispatch.sha256\n112ff6856febb31402b136cd4f86780928810157dcfe664d4914fa68d972d25b  wrapper-source.sha256\n' |
        sha256sum --strict -c
    sha256sum --strict -c context.sha256 allocator-dispatch.sha256 wrapper-source.sha256 >/dev/null)
(cd "$run" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
cp -a -- "$variant/"{context,context.sha256,verify_allocator_image.py,allocator.py,verify.py} .
sha256sum --strict -c context.sha256
export PYTHONPATH="$run/harness/ci:$stock/src/ci:$rag:$prior/src/ci"
parent=$(cat "$stock/image.id")
test "$parent" = sha256:76767e33ce49f83a1b7bef20874a2aeb2b0a062ab556194d26b9d38c64df4ed8
test "$(docker image inspect --format '{{.Id}}' "$parent")" = "$parent"
tag="riley-local-dot-allocator-parent:${stage##*.}"
if docker image inspect "$tag" >/dev/null 2>&1; then exit 74; fi
docker image tag "$parent" "$tag"
parent_tag=$tag
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
DOCKER_BUILDKIT=0 docker build --pull=false --network=none --no-cache \
    --build-arg "RILEY_PARENT=$parent_tag" --build-arg "RILEY_PARENT_ID=$parent" \
    --build-arg "JEMALLOC_PACKAGE_SHA256=$deb_sha" --iidfile image.id context >docker-build.log 2>&1
"$runner" -B verify_allocator_image.py "$parent" "$(cat image.id)" "$stage/context" "$cpu" \
    "$base/driver/libcuda.so.1" "$stage/cpu-output" >verified-image.json
"$runner" -B allocator.py bind "$stage" "$cpu" >gpu-checker-input.json
printf 'IMAGE_VERIFIED %s\n' "$(date -u +%FT%TZ)"

reservations() {
    local phase=$1
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"idle-$phase.json"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status >"gpu7-$phase.json"
    "$runner" -B - "$phase" <<'PY'
from pathlib import Path
import sys
from verify import check_reservations
check_reservations(Path.cwd(), sys.argv[1])
PY
}

run_arm() {
    local label=$1 index=$2 rc=0
    printf 'START %s %s %s\n' "$label" "$index" "$(date -u +%FT%TZ)"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" run -- \
        "$runner" -B allocator.py run "$stage" "$cpu" "$base" "$label" >"$label.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" >"$label.exit"
    printf 'END %s %s %s exit=%d\n' "$label" "$index" "$(date -u +%FT%TZ)" "$rc"
    test -s "$label/report.json"
}

reservations before
run_arm cache-off 6 &
off=$!
run_arm cache-on 7 &
on=$!
rc=0
wait "$off" || rc=1
wait "$on" || rc=1
reservations after-soaks
test "$rc" = 0
run_arm accuracy 6
reservations after
test "$(cat accuracy.exit)" = 0
"$runner" -B allocator.py verify "$stage" "$cpu" >verified.json
sha256sum verified-image.json verified.json context.sha256 gpu-checker-input.json \
    {cache-off,cache-on,accuracy}.exit {cache-off,cache-on,accuracy}/{report,api,shutdown}.json \
    cache-{off,on}/soak.json {cache-off,cache-on,accuracy}-allocator-{before,after}.json \
    idle-{before,after-soaks,after}.json gpu7-{before,after-soaks,after}.json >result.sha256
