#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
candidate="$evidence/w5-jemalloc-20261120.hLPizqfG"
accuracy="$evidence/w5-jemalloc-accuracy-20261120.CJrnRg9W"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
prior="$evidence/w4-dist-d3d8fe6e5.FE5PU6Y4"
python="$root/tools/vllm-venv/bin/python"
package_sha=144eaa64b9da0319760c407ebe330b46d9d03ef9952fc067d48221cc9832e45c
library_sha=c769f7591006b37ac8893a69a82fc40d5ac831e8e109670878c4e4d44c8a6ba9
parent=sha256:0a907408ec73fc8b9b00f440037305ce1bfb00e05bd8d3ef3096fcee59849ea1
parent_tag=
trap 'rc=$?; if test -n "$parent_tag"; then docker image rm "$parent_tag" >parent-untag.log 2>&1 || true; fi
      printf "%d\n" "$rc" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
printf 'WAIT %s %s\n' "$accuracy" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 10800))
until test -s "$accuracy/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$accuracy/campaign.pid")" | grep -Fq -- "$accuracy/campaign.sh"; then
        test -s "$accuracy/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 20
done
test "$(cat "$accuracy/campaign.exit")" = 0
(cd "$accuracy" && sha256sum -c dispatch.sha256 result.sha256)
(cd "$candidate" && sha256sum -c dispatch.sha256 gpu-dispatch.sha256 preflight.sha256 result.sha256)
(cd "$prior" && sha256sum -c "$accuracy/accuracy-harness.sha256")
export PYTHONPATH="$accuracy:$candidate:$prior/harness/ci:$base/src/ci"
"$python" -B "$accuracy/qualify_allocator.py" decision "$candidate" "$evidence" "$accuracy" >decision.json
cmp decision.json "$accuracy/decision.json"
if ! "$python" -B -c 'import json; raise SystemExit(not json.load(open("decision.json"))["run_accuracy"])'; then
    printf 'SKIP_STRICT_SOAK %s\n' "$(date -u +%FT%TZ)"
    sha256sum decision.json >result.sha256
    exit 0
fi
"$python" -B "$accuracy/qualify_allocator.py" verify "$candidate" "$evidence" "$accuracy" >accuracy-verified.json
cmp accuracy-verified.json "$accuracy/verified-accuracy.json"
printf 'START packaging %s\n' "$(date -u +%FT%TZ)"
deb="$candidate/libjemalloc2_5.2.1-4ubuntu1_amd64.deb"
printf '%s  %s\n' "$package_sha" "$deb" | sha256sum -c -
dpkg-deb --extract "$deb" unpacked
mkdir context
install -m 0644 Dockerfile context/Dockerfile
install -m 0555 riley-with-allocator context/riley-with-allocator
install -m 0444 unpacked/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 context/libjemalloc.so.2
install -m 0444 unpacked/usr/share/doc/libjemalloc2/copyright context/jemalloc-copyright
printf '%s  %s\n' "$library_sha" context/libjemalloc.so.2 | sha256sum -c -
sha256sum context/* >context.sha256

tag="riley-local-allocator-parent:${stage##*.}"
if docker image inspect "$tag" >/dev/null 2>&1; then exit 74; fi
test "$(docker image inspect --format '{{.Id}}' "$parent")" = "$parent"
docker image tag "$parent" "$tag"
parent_tag=$tag
DOCKER_BUILDKIT=0 docker build --pull=false --network=none --no-cache \
    --build-arg "RILEY_PARENT=$parent_tag" --build-arg "RILEY_PARENT_ID=$parent" \
    --build-arg "JEMALLOC_PACKAGE_SHA256=$package_sha" --iidfile image.id context >build.log 2>&1
"$python" -B verify_allocator_image.py "$parent" "$(cat image.id)" "$stage/context" \
    "$base/cpu-evidence/report.json" "$base/driver/libcuda.so.1" "$stage/cpu-output" >verified-image.json
printf 'END packaging %s\n' "$(date -u +%FT%TZ)"
sha256sum decision.json accuracy-verified.json context.sha256 image.id verified-image.json build.log >result.sha256
