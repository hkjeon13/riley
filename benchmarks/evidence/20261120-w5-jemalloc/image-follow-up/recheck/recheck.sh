#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
first="$evidence/w5-allocator-image-20261120.1sYG1O9J"
accuracy="$evidence/w5-jemalloc-accuracy-20261120.CJrnRg9W"
image=$(cat "$first/image.id")
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$first/campaign.exit")" = 1
[[ "$image" =~ ^sha256:[0-9a-f]{64}$ ]]
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
test "$(cat "$accuracy/campaign.exit")" = 0
(cd "$first" && sha256sum -c dispatch.sha256 context.sha256)
(cd "$accuracy" && sha256sum -c result.sha256)
cp -- "$first/verify_allocator_image.py" verify_allocator_image.py
gzip -dc loader-source-name.patch.gz | git apply --check -
gzip -dc loader-source-name.patch.gz | git apply -
printf '%s\n' "$image" >image.id
"$root/tools/vllm-venv/bin/python" -B verify_allocator_image.py \
    sha256:0a907408ec73fc8b9b00f440037305ce1bfb00e05bd8d3ef3096fcee59849ea1 "$image" "$first/context" \
    "$base/cpu-evidence/report.json" "$base/driver/libcuda.so.1" "$stage/cpu-output" >verified-image.json
sha256sum image.id verified-image.json verify_allocator_image.py >result.sha256
cat verified-image.json
