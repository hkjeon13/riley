#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
base=sha256:67efe6fbdaebcc8a088eaa9626f1adb43cf1af3e673ccae2d9af076fa99bdd24
binary=$(realpath ../named-prefix32/target/release/riley)
test "$(sha256sum "$binary" | cut -d' ' -f1)" = "$(cut -d' ' -f1 ../named-prefix32/binary.sha256)"
container=$(docker create --pull=never "$base")
trap 'docker rm "$container" >/dev/null' EXIT
docker cp "$binary" "$container:/opt/riley/bin/riley"
docker commit "$container" > candidate-image.id
docker image inspect --format '{{.Id}}' "$base" > base-image.id
docker image inspect --format '{{.Id}}' "$(cat candidate-image.id)" > inspected-candidate.id
cmp candidate-image.id inspected-candidate.id
sha256sum "$binary" > candidate-binary.sha256
test "$(cat base-image.id)" != "$(cat candidate-image.id)"
for ((i=0; i<900; i++)); do
    [[ -f ../coder-fleet-reference/released ]] && break
    sleep 2
done
test -f ../coder-fleet-reference/released
export OPERATIONAL_TRIAL=1 IMAGE_ROLLBACK_DIR="$PWD"
python3 -B process_qualification.py ../../w2-sm17-20261118.6UMfeHh2 retry --explicit-citations > driver.log 2>&1
touch released
