#!/usr/bin/env bash
# Run CUDA compilation, GPU tests, and model parity only on server-4096.

set -euo pipefail
umask 022

DESIGNATED_HOSTNAME=psyche-MS-7D91
DESIGNATED_GPU_UUID=GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0

usage() {
    cat >&2 <<'EOF'
usage: ci/run_remote_optimization_evidence.sh \
  --builder-image sha256:... \
  --expected-source-archive-sha256 HEX \
  --expected-model-tree-sha256 HEX \
  --expected-profile-binary-sha256 HEX \
  --model-dir PATH \
  --profile-binary PATH \
  --output-dir PATH \
  [--source-revision COMMIT]
EOF
}

builder_image=
expected_source_archive_sha256=
expected_model_tree_sha256=
expected_profile_binary_sha256=
model_dir=
profile_binary=
output_dir=
source_revision=HEAD
active_container=

cleanup_container() {
    if [[ -n ${active_container} ]]; then
        docker container rm --force --volumes "${active_container}" >/dev/null 2>&1 || true
    fi
}
trap cleanup_container EXIT

while (($#)); do
    case "$1" in
        --builder-image)
            (($# >= 2)) || { usage; exit 2; }
            builder_image=$2
            shift 2
            ;;
        --expected-source-archive-sha256)
            (($# >= 2)) || { usage; exit 2; }
            expected_source_archive_sha256=$2
            shift 2
            ;;
        --expected-model-tree-sha256)
            (($# >= 2)) || { usage; exit 2; }
            expected_model_tree_sha256=$2
            shift 2
            ;;
        --expected-profile-binary-sha256)
            (($# >= 2)) || { usage; exit 2; }
            expected_profile_binary_sha256=$2
            shift 2
            ;;
        --model-dir)
            (($# >= 2)) || { usage; exit 2; }
            model_dir=$2
            shift 2
            ;;
        --profile-binary)
            (($# >= 2)) || { usage; exit 2; }
            profile_binary=$2
            shift 2
            ;;
        --output-dir)
            (($# >= 2)) || { usage; exit 2; }
            output_dir=$2
            shift 2
            ;;
        --source-revision)
            (($# >= 2)) || { usage; exit 2; }
            source_revision=$2
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

test -n "${builder_image}" || { usage; exit 2; }
test -n "${expected_source_archive_sha256}" || { usage; exit 2; }
test -n "${expected_model_tree_sha256}" || { usage; exit 2; }
test -n "${expected_profile_binary_sha256}" || { usage; exit 2; }
test -n "${model_dir}" || { usage; exit 2; }
test -n "${profile_binary}" || { usage; exit 2; }
test -n "${output_dir}" || { usage; exit 2; }

sha_re='^[0-9a-f]{64}$'
image_re='^sha256:[0-9a-f]{64}$'
[[ ${builder_image} =~ ${image_re} ]] || { echo "builder image must be an immutable sha256 ID" >&2; exit 2; }
for digest in \
    "${expected_source_archive_sha256}" \
    "${expected_model_tree_sha256}" \
    "${expected_profile_binary_sha256}"
do
    [[ ${digest} =~ ${sha_re} ]] || { echo "trusted digests must be lowercase SHA-256 values" >&2; exit 2; }
done

actual_hostname=$(hostname)
test "${actual_hostname}" = "${DESIGNATED_HOSTNAME}" || {
    echo "optimizer evidence may run only on server-4096 (${DESIGNATED_HOSTNAME}), got ${actual_hostname}" >&2
    exit 1
}
nvidia-smi --id="${DESIGNATED_GPU_UUID}" \
    --query-gpu=name,uuid,compute_cap \
    --format=csv,noheader,nounits \
    | grep -F "NVIDIA GeForce RTX 4090, ${DESIGNATED_GPU_UUID}, 8.9" >/dev/null

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "${repository_root}"

run_release_python() {
    /usr/bin/env -i \
        PATH=/usr/bin:/bin \
        PYTHONHASHSEED=0 \
        PYTHONNOUSERSITE=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        LC_ALL=C \
        TZ=UTC \
        /usr/bin/python3 \
        "${repository_root}/ci/release/run_release_python.py" \
        "$@"
}

resolved_revision=$(git rev-parse --verify "${source_revision}^{commit}")
[[ ${resolved_revision} =~ ^[0-9a-f]{40}$ ]]
source_date_epoch=$(git show -s --format=%ct "${resolved_revision}")
[[ ${source_date_epoch} =~ ^[0-9]+$ ]]

require_exact_clean_checkout() {
    test "$(git rev-parse --verify 'HEAD^{commit}')" = "${resolved_revision}" || {
        echo "checkout HEAD differs from the selected source revision" >&2
        return 1
    }
    test -z "$(git status --porcelain=v1 --untracked-files=all)" || {
        echo "optimizer evidence checkout must be completely clean" >&2
        return 1
    }
}
require_exact_clean_checkout

resolved_image_id=$(docker image inspect --format '{{.Id}}' "${builder_image}")
test "${resolved_image_id}" = "${builder_image}" || {
    echo "builder image resolution differs from the trusted immutable ID" >&2
    exit 1
}
test "$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "${builder_image}")" = linux/amd64

model_dir=$(cd "${model_dir}" && pwd -P)
test -d "${model_dir}"
test ! -L "${model_dir}"
if find "${model_dir}" -mindepth 1 ! -type d ! -type f -print -quit | grep -q .; then
    echo "model tree contains a link or non-regular entry" >&2
    exit 1
fi
test "$(sha256sum "${model_dir}/model.safetensors" | awk '{print $1}')" = \
    80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1
test "$(sha256sum "${model_dir}/tokenizer.json" | awk '{print $1}')" = \
    9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c

profile_binary=$(cd "$(dirname "${profile_binary}")" && pwd -P)/$(basename "${profile_binary}")
test -f "${profile_binary}"
test ! -L "${profile_binary}"
test -x "${profile_binary}"
test "$(sha256sum "${profile_binary}" | awk '{print $1}')" = "${expected_profile_binary_sha256}"

if [[ -e ${output_dir} || -L ${output_dir} ]]; then
    echo "refusing to reuse optimizer evidence output: ${output_dir}" >&2
    exit 1
fi
mkdir -m 0700 -p "${output_dir}"
output_dir=$(cd "${output_dir}" && pwd -P)
case "${output_dir}" in
    "${repository_root}"|"${repository_root}/"*)
        echo "optimizer evidence output must be outside the source checkout" >&2
        exit 1
        ;;
esac
mkdir -m 0777 "${output_dir}/evidence"
mkdir -m 0777 "${output_dir}/runner-output"

source_archive="${output_dir}/source.tar"
git -c tar.umask=0002 archive \
    --format=tar \
    --output="${source_archive}" \
    "${resolved_revision}"
source_archive_sha256=$(sha256sum "${source_archive}" | awk '{print $1}')
test "${source_archive_sha256}" = "${expected_source_archive_sha256}" || {
    echo "generated source archive differs from the trusted digest" >&2
    exit 1
}
test "$(git get-tar-commit-id <"${source_archive}")" = "${resolved_revision}"

model_manifest="${output_dir}/model-SHA256SUMS"
: >"${model_manifest}"
model_count=0
while IFS= read -r -d '' model_file; do
    relative=${model_file#"${model_dir}"/}
    [[ ${relative} =~ ^[A-Za-z0-9._/+@=-]+$ ]] || {
        echo "model path uses an unsafe alphabet: ${relative}" >&2
        exit 1
    }
    printf '%s  %s\n' "$(sha256sum "${model_file}" | awk '{print $1}')" "${relative}" \
        >>"${model_manifest}"
    model_count=$((model_count + 1))
done < <(find "${model_dir}" -type f -print0 | sort -z)
test "${model_count}" -gt 0
test "$(sha256sum "${model_manifest}" | awk '{print $1}')" = "${expected_model_tree_sha256}" || {
    echo "model tree differs from the trusted manifest digest" >&2
    exit 1
}

docker image inspect "${builder_image}" >"${output_dir}/builder-image-inspect.json"
container_id=$(docker create \
    --restart no \
    --entrypoint /bin/bash \
    --user 0:0 \
    --workdir /workspace \
    --no-healthcheck \
    --network none \
    --gpus "device=${DESIGNATED_GPU_UUID}" \
    --read-only \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --pids-limit 8192 \
    --tmpfs /tmp:rw,nosuid,nodev,exec,size=2147483648 \
    --env "RUSTINFER_SOURCE_REVISION=${resolved_revision}" \
    --env "RUSTINFER_SOURCE_ARCHIVE_SHA256=${source_archive_sha256}" \
    --env "RUSTINFER_BUILD_IMAGE_ID=${builder_image}" \
    --env "RUSTINFER_MODEL_TREE_SHA256=${expected_model_tree_sha256}" \
    --env "SOURCE_DATE_EPOCH=${source_date_epoch}" \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --env ALL_PROXY= \
    --env FTP_PROXY= \
    --env HTTP_PROXY= \
    --env HTTPS_PROXY= \
    --env NO_PROXY= \
    --env all_proxy= \
    --env ftp_proxy= \
    --env http_proxy= \
    --env https_proxy= \
    --env no_proxy= \
    --mount "type=bind,source=${source_archive},destination=/input/source.tar,readonly" \
    --mount "type=bind,source=${model_dir},destination=/model,readonly" \
    --mount "type=bind,source=${output_dir}/evidence,destination=/evidence" \
    --mount "type=bind,source=${output_dir}/runner-output,destination=/runner-output" \
    --mount type=volume,destination=/workspace,volume-nocopy \
    "${builder_image}" \
    -ceu 'test -z "$(find /workspace -mindepth 1 -print -quit)"; tar --extract --file /input/source.tar --directory /workspace; cd /workspace; exec /bin/bash ci/release/run_optimization_evidence_once.sh')
[[ ${container_id} =~ ^[0-9a-f]{64}$ ]]
active_container=${container_id}
docker inspect "${container_id}" >"${output_dir}/container-inspect.json"
set +e
docker start --attach "${container_id}"
container_status=$?
set -e
docker inspect "${container_id}" >"${output_dir}/container-inspect-post.json"
docker container rm --volumes "${container_id}" >/dev/null
active_container=
if ((container_status != 0)); then
    echo "remote optimizer evidence container failed with status ${container_status}" >&2
    exit "${container_status}"
fi

test "$(cat "${output_dir}/runner-output/completed")" = rustinfer.optimizer-remote-run.completed.v3
cmp --silent "${profile_binary}" "${output_dir}/runner-output/rustinfer-profile" || {
    echo "optimizer-run profile binary differs from the reproducible candidate profile" >&2
    exit 1
}
test "$(sha256sum "${output_dir}/runner-output/rustinfer-profile" | awk '{print $1}')" = \
    "${expected_profile_binary_sha256}"
cmp --silent "${model_manifest}" "${output_dir}/runner-output/model-SHA256SUMS" || {
    echo "container model manifest differs from the host-verified model tree" >&2
    exit 1
}

require_exact_clean_checkout

report="${output_dir}/optimization-correctness-report.json"
receipt="${output_dir}/evidence/run-receipt.json"
recorded_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
run_release_python "${repository_root}/ci/release/write_optimization_execution_evidence.py" \
    --evidence-dir "${output_dir}/evidence" \
    --command-records "${output_dir}/runner-output/commands.v2" \
    --subject-records "${output_dir}/runner-output/subjects.v2" \
    --gpu-csv "${output_dir}/runner-output/gpu.csv" \
    --report "${report}" \
    --receipt "${receipt}" \
    --source-revision "${resolved_revision}" \
    --source-archive-sha256 "${source_archive_sha256}" \
    --build-image-id "${builder_image}" \
    --profile-binary "${profile_binary}" \
    --model-tree-sha256 "${expected_model_tree_sha256}" \
    --recorded-at-utc "${recorded_at_utc}" \
    >"${output_dir}/writer-result.json"

raw_evidence="${output_dir}/optimization-correctness-evidence.tar"
run_release_python "${repository_root}/ci/release/check_optimization_evidence.py" \
    --evidence-dir "${output_dir}/evidence" \
    --raw-evidence "${raw_evidence}" \
    --report "${report}" \
    --source-revision "${resolved_revision}" \
    --source-archive-sha256 "${source_archive_sha256}" \
    --build-image-id "${builder_image}" \
    --profile-binary "${profile_binary}" \
    >"${output_dir}/replay-result.json"

sha256sum \
    "${report}" \
    "${raw_evidence}" \
    "${profile_binary}" \
    "${output_dir}/writer-result.json" \
    "${output_dir}/replay-result.json" \
    >"${output_dir}/SHA256SUMS"
echo "remote optimizer evidence passed: ${output_dir}"
