#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
stock="$evidence/w5-current-package-20261121.f6SEbjvA"
stock_soaks="$evidence/w5-current-soaks-20261121.OyGugR0K"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
candidate="$evidence/w5-jemalloc-20261120.hLPizqfG"
prior="$evidence/w4-dist-d3d8fe6e5.FE5PU6Y4"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
cpu="$stock/cpu-evidence/report.json"
stub="$base/driver/libcuda.so.1"
parent=sha256:b731f48cc6ec0820197ec484e98c673e2de592f12ab8448e3fa9f38ca7217c9b
package_sha=144eaa64b9da0319760c407ebe330b46d9d03ef9952fc067d48221cc9832e45c
library_sha=c769f7591006b37ac8893a69a82fc40d5ac831e8e109670878c4e4d44c8a6ba9
parent_tag=
trap 'rc=$?; if test -n "$parent_tag"; then docker image rm "$parent_tag" >parent-untag.log 2>&1 || true; fi
      printf "%d\n" "$rc" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT

sha256sum -c dispatch.sha256
test "$(cat "$stock/campaign.exit")" = 0
test "$(cat "$stock_soaks/campaign.exit")" = 0
(cd "$stock" && sha256sum -c runtime-tools.sha256 >/dev/null)
printf '%s  %s\n' f35ef0db484b8ea1b60b2c3d97af422280a3831660c0801a17233e014df0650e "$stock_soaks/verify_soaks.py" | sha256sum -c
printf '%s  %s\n' 9aba66838fb889e749e9a9235e0d633408d79917bc71a5924abb4cdc5396ddf3 "$cpu" | sha256sum -c
cp -- "$stock_soaks/verify_soaks.py" .
cp -- "$prior/harness/benchmarks/live/long_accuracy.py" .
printf '%s  long_accuracy.py\n' 679aec2b87b87957810ebd6984d48a57d8345802287adb6fe127c753f2a47063 | sha256sum -c

printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
deb="$candidate/libjemalloc2_5.2.1-4ubuntu1_amd64.deb"
printf '%s  %s\n' "$package_sha" "$deb" | sha256sum -c
dpkg-deb --extract "$deb" unpacked
mkdir context
install -m 0644 Dockerfile context/Dockerfile
install -m 0555 riley-with-allocator context/riley-with-allocator
install -m 0444 unpacked/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 context/libjemalloc.so.2
install -m 0444 unpacked/usr/share/doc/libjemalloc2/copyright context/jemalloc-copyright
printf '%s  context/libjemalloc.so.2\n' "$library_sha" | sha256sum -c
sha256sum context/* >context.sha256
tag="riley-local-allocator-parent:${stage##*.}"
if docker image inspect "$tag" >/dev/null 2>&1; then exit 74; fi
test "$(docker image inspect --format '{{.Id}}' "$parent")" = "$parent"
docker image tag "$parent" "$tag"
parent_tag=$tag
DOCKER_BUILDKIT=0 docker build --pull=false --network=none --no-cache \
  --build-arg "RILEY_PARENT=$tag" --build-arg "RILEY_PARENT_ID=$parent" \
  --build-arg "JEMALLOC_PACKAGE_SHA256=$package_sha" --iidfile image.id context >build.log 2>&1
"$python" -B verify_allocator_image.py "$parent" "$(cat image.id)" "$stage/context" "$cpu" "$stub" "$stage/cpu-output" >verified-image.json
export PYTHONPATH="$stock/harness/ci:$stock/src/ci"
"$python" -B qualify.py bind "$stage" "$cpu" >gpu-checker-input.json

reserve_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}

reservations() {
    for index in 6 7; do reserve_status "$index" >"gpu$index-$1.json"; done
    "$python" -B verify_soaks.py --reservations "$stage" "$1"
}

run_arm() {
    local scope=$1 index=$2 rc=0
    printf 'START %s %s %s\n' "$scope" "$index" "$(date -u +%FT%TZ)"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" run -- \
        "$python" -B "$stage/qualify.py" run "$stage" "$cpu" "$base" "$scope" >"$scope.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" >"$scope.exit"
    printf 'END %s %s %s exit=%d\n' "$scope" "$index" "$(date -u +%FT%TZ)" "$rc"
}

reservations before
run_arm cache-off 6 &
off=$!
run_arm cache-on 7 &
on=$!
wait "$off"
wait "$on"
reservations after
"$python" -B qualify.py verify "$stage" "$cpu" >verified.json
sha256sum verified.json verified-image.json context.sha256 gpu-checker-input.json \
    cache-{off,on}.exit cache-{off,on}/{api,soak,report,shutdown}.json \
    cache-{off,on}-allocator-{before,after}.json gpu{6,7}-{before,after}.json >result.sha256
if "$python" -B -c 'import json; raise SystemExit(json.load(open("verified.json"))["release_gate"]!="passed")'; then
    run_arm accuracy 6
    test "$(cat accuracy.exit)" = 0
    reservations after-accuracy
    "$python" -B qualify.py verify-accuracy "$stage" "$cpu" >verified-accuracy.json
    sha256sum verified-accuracy.json accuracy/{report,shutdown}.json accuracy-allocator-{before,after}.json \
        gpu{6,7}-after-accuracy.json >accuracy-result.sha256
fi
