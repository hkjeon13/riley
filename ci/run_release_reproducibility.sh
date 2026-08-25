#!/usr/bin/env bash
# Run only on the designated remote build host; this script never initializes a GPU.

set -euo pipefail
umask 022

usage() {
    echo "usage: $0 --builder-image sha256:... --expected-source-archive-sha256 HEX --output-dir PATH [--source-revision COMMIT]" >&2
}

builder_image=""
expected_source_archive_sha256=""
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
        --expected-source-archive-sha256)
            (($# >= 2)) || { usage; exit 2; }
            expected_source_archive_sha256=$2
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
test -n "${output_dir}" || { usage; exit 2; }
[[ "${builder_image}" =~ ^sha256:[0-9a-f]{64}$ ]] || {
    echo "--builder-image must be an immutable local OCI image ID" >&2
    exit 2
}
[[ "${expected_source_archive_sha256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "--expected-source-archive-sha256 must be a trusted lowercase SHA-256" >&2
    exit 2
}

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
[[ "${resolved_revision}" =~ ^[0-9a-f]{40}$ ]]
require_exact_clean_checkout() {
    local checked_head
    checked_head=$(git rev-parse --verify 'HEAD^{commit}')
    test "${checked_head}" = "${resolved_revision}" || {
        echo "runner checkout HEAD differs from --source-revision" >&2
        return 1
    }
    test -z "$(git status --porcelain=v1 --untracked-files=all)" || {
        echo "runner checkout must be completely clean, including untracked files" >&2
        return 1
    }
}
require_exact_clean_checkout
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
case "${output_dir}" in
    "${repository_root}"|"${repository_root}/"*)
        echo "reproducibility output directory must be outside the source checkout" >&2
        exit 1
        ;;
esac
source_archive="${output_dir}/source.tar"
git -c tar.umask=0002 archive \
    --format=tar \
    --output="${source_archive}" \
    "${resolved_revision}"
source_archive_sha256=$(sha256sum "${source_archive}" | cut -d ' ' -f 1)
test "${source_archive_sha256}" = "${expected_source_archive_sha256}" || {
    echo "generated source archive differs from the trusted expected SHA-256" >&2
    exit 1
}
embedded_revision=$(git get-tar-commit-id < "${source_archive}")
test "${embedded_revision}" = "${resolved_revision}"
run_release_python -c \
    'import sys; from pathlib import Path; from check_reproducible_build import _validate_source_archive; _validate_source_archive(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]))' \
    "${source_archive}" "${resolved_revision}" "${source_date_epoch}"
builder_image_inspect="${output_dir}/builder-image-inspect.json"
docker image inspect "${builder_image}" > "${builder_image_inspect}"

run_one() {
    local build_id=$1
    local lower_id=${build_id,,}
    local run_dir="${output_dir}/run-${lower_id}"
    local inspect_path="${output_dir}/container-inspect-${lower_id}.json"
    local post_inspect_path="${output_dir}/container-inspect-${lower_id}-post.json"
    local container_id
    local run_status
    mkdir "${run_dir}"
    : > "${inspect_path}"

    container_id=$(docker create \
        --runtime runc \
        --cgroupns private \
        --restart no \
        --entrypoint /bin/bash \
        --user 0:0 \
        --workdir /workspace \
        --no-healthcheck \
        --network none \
        --cap-drop ALL \
        --security-opt no-new-privileges \
        --pids-limit 4096 \
        --env "RUSTINFER_REPRO_BUILD_ID=${build_id}" \
        --env "RUSTINFER_SOURCE_REVISION=${resolved_revision}" \
        --env "RUSTINFER_SOURCE_ARCHIVE_SHA256=${source_archive_sha256}" \
        --env "RUSTINFER_BUILD_IMAGE_ID=${builder_image}" \
        --env "SOURCE_DATE_EPOCH=${source_date_epoch}" \
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
        --mount "type=bind,source=${inspect_path},destination=/input/container-inspect.json,readonly" \
        --mount "type=bind,source=${builder_image_inspect},destination=/input/builder-image-inspect.json,readonly" \
        --mount "type=bind,source=${run_dir},destination=/evidence" \
        --mount type=volume,destination=/workspace,volume-nocopy \
        "${builder_image}" \
        -ceu 'test -z "$(find /workspace -mindepth 1 -print -quit)"; tar --extract --file /input/source.tar --directory /workspace; cd /workspace; exec /bin/bash ci/release/run_reproducible_build_once.sh')
    [[ "${container_id}" =~ ^[0-9a-f]{64}$ ]]
    active_container=${container_id}
    docker inspect "${container_id}" > "${inspect_path}"
    set +e
    docker start --attach "${container_id}"
    run_status=$?
    set -e
    docker inspect "${container_id}" > "${post_inspect_path}"
    docker container rm --volumes "${container_id}" >/dev/null
    active_container=""
    if ((run_status != 0)); then
        echo "reproducibility build ${build_id} failed with status ${run_status}" >&2
        return "${run_status}"
    fi
}

run_one A
run_one B

require_exact_clean_checkout

package_one() {
    local build_id=$1
    local lower_id=${build_id,,}
    local run_dir="${output_dir}/run-${lower_id}"
    run_release_python \
        "${repository_root}/ci/release/package_reproducible_build_evidence.py" \
        --build-id "${build_id}" \
        --source-archive "${source_archive}" \
        --source-revision "${resolved_revision}" \
        --source-date-epoch "${source_date_epoch}" \
        --build-image-id "${builder_image}" \
        --binary "${run_dir}/artifacts/rustinfer" \
        --profile-binary "${run_dir}/artifacts/rustinfer-profile" \
        --bundle "${run_dir}/artifacts/rustinfer.tar.gz" \
        --native-manifest "${run_dir}/artifacts/native-dependencies.txt" \
        --toolchain-log "${run_dir}/logs/toolchain.txt" \
        --builder-image-inspect "${builder_image_inspect}" \
        --container-inspect "${output_dir}/container-inspect-${lower_id}.json" \
        --post-container-inspect "${output_dir}/container-inspect-${lower_id}-post.json" \
        --completion-receipt "${run_dir}/logs/build-completion.json" \
        --preflight-log "${run_dir}/logs/preflight.log" \
        --cargo-build-log "${run_dir}/logs/cargo-build.log" \
        --profile-build-log "${run_dir}/logs/profile-build.log" \
        --bundle-build-log "${run_dir}/logs/bundle-build.log" \
        --bundle-verify-log "${run_dir}/logs/bundle-verify.log" \
        --output "${run_dir}/repro-build-${lower_id}.tar"
}

package_one A
package_one B

mkdir "${output_dir}/final"
install -m 0755 "${output_dir}/run-a/artifacts/rustinfer" \
    "${output_dir}/final/rustinfer"
install -m 0755 "${output_dir}/run-a/artifacts/rustinfer-profile" \
    "${output_dir}/final/rustinfer-profile"
install -m 0644 "${output_dir}/run-a/artifacts/rustinfer.tar.gz" \
    "${output_dir}/final/rustinfer.tar.gz"
install -m 0644 "${output_dir}/run-a/artifacts/native-dependencies.txt" \
    "${output_dir}/final/native-dependencies.txt"

require_exact_clean_checkout
run_release_python "${repository_root}/ci/release/check_reproducible_build.py" \
    --evidence-a "${output_dir}/run-a/repro-build-a.tar" \
    --evidence-b "${output_dir}/run-b/repro-build-b.tar" \
    --source-archive "${source_archive}" \
    --expected-source-archive-sha256 "${expected_source_archive_sha256}" \
    --source-revision "${resolved_revision}" \
    --source-date-epoch "${source_date_epoch}" \
    --build-image-id "${builder_image}" \
    --final-binary "${output_dir}/final/rustinfer" \
    --final-profile-binary "${output_dir}/final/rustinfer-profile" \
    --final-bundle "${output_dir}/final/rustinfer.tar.gz" \
    --final-native-manifest "${output_dir}/final/native-dependencies.txt" \
    --output-report "${output_dir}/reproducibility-report-v1.json"

(
    cd "${output_dir}"
    sha256sum \
        source.tar \
        builder-image-inspect.json \
        container-inspect-a.json \
        container-inspect-a-post.json \
        container-inspect-b.json \
        container-inspect-b-post.json \
        run-a/repro-build-a.tar \
        run-b/repro-build-b.tar \
        final/rustinfer \
        final/rustinfer-profile \
        final/rustinfer.tar.gz \
        final/native-dependencies.txt \
        reproducibility-report-v1.json \
        > SHA256SUMS
)

echo "release reproducibility evidence: ${output_dir}"
