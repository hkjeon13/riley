#!/usr/bin/env bash
# Run only on the designated remote build host; this script never initializes a GPU.

set -euo pipefail
umask 022

usage() {
    echo "usage: $0 --builder-image sha256:... --output-dir PATH [--source-revision COMMIT]" >&2
}

builder_image=""
output_dir=""
source_revision="HEAD"
active_container=""

cleanup_container() {
    if [[ -n "${active_container}" ]]; then
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
test -n "${output_dir}" || { usage; exit 2; }
[[ "${builder_image}" =~ ^sha256:[0-9a-f]{64}$ ]] || {
    echo "--builder-image must be an immutable local OCI image ID" >&2
    exit 2
}

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "${repository_root}"

resolved_revision=$(git rev-parse --verify "${source_revision}^{commit}")
[[ "${resolved_revision}" =~ ^[0-9a-f]{40}$ ]]
source_date_epoch=$(git show -s --format=%ct "${resolved_revision}")
[[ "${source_date_epoch}" =~ ^[0-9]+$ ]]

resolved_image_id=$(docker image inspect --format '{{.Id}}' "${builder_image}")
test "${resolved_image_id}" = "${builder_image}" || {
    echo "builder image resolution differs from the requested immutable ID" >&2
    exit 1
}
image_platform=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "${builder_image}")
test "${image_platform}" = linux/amd64 || {
    echo "builder image platform must be linux/amd64, got ${image_platform}" >&2
    exit 1
}

if [[ -e "${output_dir}" || -L "${output_dir}" ]]; then
    echo "refusing to reuse reproducibility output directory: ${output_dir}" >&2
    exit 1
fi
mkdir -p "${output_dir}"
output_dir=$(cd "${output_dir}" && pwd -P)
source_archive="${output_dir}/source.tar"
git archive --format=tar --output="${source_archive}" "${resolved_revision}"
source_archive_sha256=$(sha256sum "${source_archive}" | cut -d ' ' -f 1)
embedded_revision=$(git get-tar-commit-id < "${source_archive}")
test "${embedded_revision}" = "${resolved_revision}"

run_one() {
    local build_id=$1
    local lower_id=${build_id,,}
    local run_dir="${output_dir}/run-${lower_id}"
    local inspect_path="${output_dir}/container-inspect-${lower_id}.json"
    local container_id
    local run_status
    mkdir "${run_dir}"
    : > "${inspect_path}"

    container_id=$(docker create \
        --runtime runc \
        --network none \
        --cap-drop ALL \
        --security-opt no-new-privileges \
        --pids-limit 4096 \
        --env "RUSTINFER_REPRO_BUILD_ID=${build_id}" \
        --env "RUSTINFER_SOURCE_REVISION=${resolved_revision}" \
        --env "RUSTINFER_SOURCE_ARCHIVE_SHA256=${source_archive_sha256}" \
        --env "RUSTINFER_BUILD_IMAGE_ID=${builder_image}" \
        --env "SOURCE_DATE_EPOCH=${source_date_epoch}" \
        --mount "type=bind,source=${source_archive},destination=/input/source.tar,readonly" \
        --mount "type=bind,source=${inspect_path},destination=/input/container-inspect.json,readonly" \
        --mount "type=bind,source=${run_dir},destination=/evidence" \
        --mount type=volume,destination=/workspace,volume-nocopy \
        "${builder_image}" \
        /bin/bash -ceu '
            test -z "$(find /workspace -mindepth 1 -print -quit)"
            tar --extract --file /input/source.tar --directory /workspace
            cd /workspace
            exec /bin/bash ci/release/run_reproducible_build_once.sh
        ')
    [[ "${container_id}" =~ ^[0-9a-f]{64}$ ]]
    active_container=${container_id}
    docker inspect "${container_id}" > "${inspect_path}"
    set +e
    docker start --attach "${container_id}"
    run_status=$?
    set -e
    docker container rm --volumes "${container_id}" >/dev/null
    active_container=""
    if ((run_status != 0)); then
        echo "reproducibility build ${build_id} failed with status ${run_status}" >&2
        return "${run_status}"
    fi
}

run_one A
run_one B

mkdir "${output_dir}/final"
install -m 0755 "${output_dir}/run-a/artifacts/rustinfer" \
    "${output_dir}/final/rustinfer"
install -m 0644 "${output_dir}/run-a/artifacts/rustinfer.tar.gz" \
    "${output_dir}/final/rustinfer.tar.gz"
install -m 0644 "${output_dir}/run-a/artifacts/native-dependencies.txt" \
    "${output_dir}/final/native-dependencies.txt"

python3 ci/release/check_reproducible_build.py \
    --evidence-a "${output_dir}/run-a/repro-build-a.tar" \
    --evidence-b "${output_dir}/run-b/repro-build-b.tar" \
    --source-archive "${source_archive}" \
    --source-revision "${resolved_revision}" \
    --source-date-epoch "${source_date_epoch}" \
    --build-image-id "${builder_image}" \
    --final-binary "${output_dir}/final/rustinfer" \
    --final-bundle "${output_dir}/final/rustinfer.tar.gz" \
    --final-native-manifest "${output_dir}/final/native-dependencies.txt" \
    --output-report "${output_dir}/reproducibility-report-v1.json"

(
    cd "${output_dir}"
    sha256sum \
        source.tar \
        container-inspect-a.json \
        container-inspect-b.json \
        run-a/repro-build-a.tar \
        run-b/repro-build-b.tar \
        final/rustinfer \
        final/rustinfer.tar.gz \
        final/native-dependencies.txt \
        reproducibility-report-v1.json \
        > SHA256SUMS
)

echo "release reproducibility evidence: ${output_dir}"
