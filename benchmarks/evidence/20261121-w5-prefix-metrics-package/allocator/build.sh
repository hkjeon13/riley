#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
stock="$evidence/w5-prefix-metrics-package-20261121.nEFyze5I"
trial="$evidence/w5-current-allocator-20261121.ZVgY3v8o"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
candidate="$evidence/w5-jemalloc-20261120.hLPizqfG"
runner="$(dirname "$evidence")/tools/vllm-venv/bin/python"
cpu="$stock/cpu-evidence/report.json"
deb="$candidate/libjemalloc2_5.2.1-4ubuntu1_amd64.deb"
deb_sha=144eaa64b9da0319760c407ebe330b46d9d03ef9952fc067d48221cc9832e45c
lib_sha=c769f7591006b37ac8893a69a82fc40d5ac831e8e109670878c4e4d44c8a6ba9
cleanup() {
    local code=$?
    if test -n "${parent_tag:-}"; then docker image rm "$parent_tag" >parent-untag.log 2>&1 || true; fi
    printf '%s\n' "$code" >build.exit.tmp
    mv -- build.exit.tmp build.exit
}
trap cleanup EXIT
sha256sum --strict -c wrapper-source.sha256
(cd "$stock" && test "$(cat build.exit)" = 0 && sha256sum --strict -c build-result.sha256 >/dev/null)
for name in Dockerfile riley-with-allocator verify_allocator_image.py; do
    cmp "$name" "$trial/$name"
done
printf '%s  %s\n' "$deb_sha" "$deb" | sha256sum --strict -c
dpkg-deb --extract "$deb" unpacked
mkdir context
install -m 0644 Dockerfile context/Dockerfile
install -m 0555 riley-with-allocator context/riley-with-allocator
install -m 0444 unpacked/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 context/libjemalloc.so.2
install -m 0444 unpacked/usr/share/doc/libjemalloc2/copyright context/jemalloc-copyright
printf '%s  context/libjemalloc.so.2\n' "$lib_sha" | sha256sum --strict -c
sha256sum context/* >context.sha256
parent=$(cat "$stock/image.id")
test "$(docker image inspect --format '{{.Id}}' "$parent")" = "$parent"
tag="riley-local-prefix-allocator-parent:${stage##*.}"
if docker image inspect "$tag" >/dev/null 2>&1; then exit 74; fi
docker image tag "$parent" "$tag"
parent_tag=$tag
DOCKER_BUILDKIT=0 docker build --pull=false --network=none --no-cache \
    --build-arg "RILEY_PARENT=$parent_tag" --build-arg "RILEY_PARENT_ID=$parent" \
    --build-arg "JEMALLOC_PACKAGE_SHA256=$deb_sha" --iidfile image.id context >docker-build.log 2>&1
"$runner" -B verify_allocator_image.py "$parent" "$(cat image.id)" "$stage/context" "$cpu" \
    "$base/driver/libcuda.so.1" "$stage/cpu-output" >verified-image.json
sha256sum image.id wrapper-source.sha256 context.sha256 verified-image.json >build-result.sha256
