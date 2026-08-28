#!/usr/bin/env bash
# Capture raw C02-P0 runtime-configuration evidence on a designated GPU host.
#
# This is deliberately a producer only. It neither creates a candidate freeze
# nor executes Gate E/C02 finalization; those operations remain outside this
# script's authority.

set -euo pipefail
set -o noclobber
umask 077
IFS=$' \t\n'

readonly SCRIPT_NAME='run_remote_c02_runtime_config_capture.sh'
readonly GPU_LOCK_PATH='/var/tmp/riley-server-4096-gpu-evidence.lock'
readonly DEFAULT_MAX_GPU_MEMORY_MIB=256
readonly DEFAULT_STARTUP_TIMEOUT_SECONDS=60
readonly DEFAULT_REQUEST_TIMEOUT_SECONDS=15
readonly DEFAULT_SHUTDOWN_TIMEOUT_SECONDS=30
readonly CONTAINER_MODEL_DIR='/riley-c02/model'
readonly CONTAINER_SERVER_OUTPUT_DIR='/riley-c02/server-output'
readonly CONTAINER_VISIBLE_GPU_INDEX=0

usage() {
    cat <<'EOF'
usage: bash ci/release/run_remote_c02_runtime_config_capture.sh \
  (--binary ABSOLUTE_RELEASE_BINARY | \
   --container-image DOCKER_IMAGE_REFERENCE \
   --container-binary ABSOLUTE_CONTAINER_RELEASE_BINARY) \
  --binary-sha256 LOWERCASE_SHA256 \
  --model-dir ABSOLUTE_MODEL_DIRECTORY \
  --model-tree-sha256 LOWERCASE_SHA256 \
  --candidate-id riley-X.Y.Z-rcN \
  --gpu-index ORDINAL \
  --gpu-uuid GPU-UUID \
  --evidence-dir NEW_ABSOLUTE_EXTERNAL_DIRECTORY \
  --stable-args-file ABSOLUTE_LINE_DELIMITED_ARGUMENT_FILE \
  --stable-env-file ABSOLUTE_KEY_VALUE_ENVIRONMENT_FILE \
  --stable-port LOOPBACK_PORT \
  --max-args-file ABSOLUTE_LINE_DELIMITED_ARGUMENT_FILE \
  --max-env-file ABSOLUTE_KEY_VALUE_ENVIRONMENT_FILE \
  --max-port LOOPBACK_PORT \
  [--max-gpu-memory-mib 256] \
  [--startup-timeout-seconds 60] \
  [--request-timeout-seconds 15] \
  [--shutdown-timeout-seconds 30]

Each args file is one additional riley serve argument per nonempty line. The
runner owns serve, --model, --bind, --device, and all three --c02-* arguments;
supplying any of those in an args file is rejected. Each environment file is
an explicit, nonempty KEY=VALUE map, one entry per line. The server is launched
through env -i using exactly that map.

Choose exactly one launch mode. --binary launches a host release binary. The
container mode resolves the supplied local image to its immutable image ID,
checks the in-image non-symlink executable against --binary-sha256, and records
that binding in evidence. It exposes only --gpu-index to the container (as
container device 0), mounts --model-dir read-only, mounts only a per-arm server
artifact directory writable, and uses host networking only with the server
bound to 127.0.0.1.

The evidence directory must not yet exist and must be outside the source tree.
The runner uses only 127.0.0.1, captures GET /v1/config body bytes without JSON
reserialization, and retains all evidence and logs when an arm fails. It never
uses sudo, stops GDM, creates a C02 freeze, writes Gate E, or makes a C02
pass/fail decision.
EOF
}

die() {
    printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
    exit 2
}

# Remove ambient launch state before parsing external input. In particular, do
# not inherit BASH_ENV, proxy settings, or an accidental CUDA selector.
if [[ ${RILEY_C02_CAPTURE_CLEAN_ENV:-} != 1 ]]; then
    exec /usr/bin/env -i \
        PATH=/usr/bin:/bin \
        LC_ALL=C \
        TZ=UTC \
        RILEY_C02_CAPTURE_CLEAN_ENV=1 \
        /usr/bin/bash "$0" "$@"
fi
unset BASH_ENV ENV CDPATH

if (($# == 0)); then
    usage >&2
    exit 2
fi
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi

binary=
container_image=
container_binary=
binary_sha256=
model_dir=
model_tree_sha256=
candidate_id=
gpu_index=
gpu_uuid=
evidence_dir=
stable_args_file=
stable_env_file=
stable_port=
max_args_file=
max_env_file=
max_port=
max_gpu_memory_mib=$DEFAULT_MAX_GPU_MEMORY_MIB
startup_timeout_seconds=$DEFAULT_STARTUP_TIMEOUT_SECONDS
request_timeout_seconds=$DEFAULT_REQUEST_TIMEOUT_SECONDS
shutdown_timeout_seconds=$DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
launch_mode=
container_image_id=
container_user=

need_value() {
    local flag=$1
    (($# >= 2)) || die "$flag requires a value"
    [[ -n $2 ]] || die "$flag must not be empty"
}

while (($# > 0)); do
    case $1 in
        --binary)
            need_value "$1" "${2:-}"
            binary=$2
            shift 2
            ;;
        --container-image)
            need_value "$1" "${2:-}"
            container_image=$2
            shift 2
            ;;
        --container-binary)
            need_value "$1" "${2:-}"
            container_binary=$2
            shift 2
            ;;
        --binary-sha256)
            need_value "$1" "${2:-}"
            binary_sha256=$2
            shift 2
            ;;
        --model-dir)
            need_value "$1" "${2:-}"
            model_dir=$2
            shift 2
            ;;
        --model-tree-sha256)
            need_value "$1" "${2:-}"
            model_tree_sha256=$2
            shift 2
            ;;
        --candidate-id)
            need_value "$1" "${2:-}"
            candidate_id=$2
            shift 2
            ;;
        --gpu-index)
            need_value "$1" "${2:-}"
            gpu_index=$2
            shift 2
            ;;
        --gpu-uuid)
            need_value "$1" "${2:-}"
            gpu_uuid=$2
            shift 2
            ;;
        --evidence-dir)
            need_value "$1" "${2:-}"
            evidence_dir=$2
            shift 2
            ;;
        --stable-args-file)
            need_value "$1" "${2:-}"
            stable_args_file=$2
            shift 2
            ;;
        --stable-env-file)
            need_value "$1" "${2:-}"
            stable_env_file=$2
            shift 2
            ;;
        --stable-port)
            need_value "$1" "${2:-}"
            stable_port=$2
            shift 2
            ;;
        --max-args-file)
            need_value "$1" "${2:-}"
            max_args_file=$2
            shift 2
            ;;
        --max-env-file)
            need_value "$1" "${2:-}"
            max_env_file=$2
            shift 2
            ;;
        --max-port)
            need_value "$1" "${2:-}"
            max_port=$2
            shift 2
            ;;
        --max-gpu-memory-mib)
            need_value "$1" "${2:-}"
            max_gpu_memory_mib=$2
            shift 2
            ;;
        --startup-timeout-seconds)
            need_value "$1" "${2:-}"
            startup_timeout_seconds=$2
            shift 2
            ;;
        --request-timeout-seconds)
            need_value "$1" "${2:-}"
            request_timeout_seconds=$2
            shift 2
            ;;
        --shutdown-timeout-seconds)
            need_value "$1" "${2:-}"
            shutdown_timeout_seconds=$2
            shift 2
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

if [[ -n $binary ]]; then
    [[ -z $container_image && -z $container_binary ]] || \
        die 'choose exactly one launch mode: --binary or --container-image with --container-binary'
    launch_mode=host-binary
else
    [[ -n $container_image && -n $container_binary ]] || \
        die 'choose exactly one launch mode: --binary or --container-image with --container-binary'
    launch_mode=container
fi

for required in \
    binary_sha256 model_dir model_tree_sha256 candidate_id gpu_index gpu_uuid evidence_dir \
    stable_args_file stable_env_file stable_port max_args_file max_env_file max_port
do
    [[ -n ${!required} ]] || die "missing --${required//_/-}"
done

for tool in bash basename cat curl dirname flock nvidia-smi python3 realpath sha256sum find sort awk grep tr wc sleep mktemp; do
    command -v "$tool" >/dev/null 2>&1 || die "required host tool is unavailable: $tool"
done
if [[ $launch_mode == container ]]; then
    command -v docker >/dev/null 2>&1 || die 'required host tool is unavailable: docker'
    command -v id >/dev/null 2>&1 || die 'required host tool is unavailable: id'
fi

readonly sha_re='^[0-9a-f]{64}$'
readonly env_key_re='^[A-Z_][A-Z0-9_]*$'
[[ $binary_sha256 =~ $sha_re ]] || die '--binary-sha256 must be lowercase SHA-256'
[[ $model_tree_sha256 =~ $sha_re ]] || die '--model-tree-sha256 must be lowercase SHA-256'
[[ $candidate_id =~ ^riley-(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)-rc([1-9][0-9]*)$ ]] || \
    die '--candidate-id must match riley-X.Y.Z-rcN with canonical decimal components'
[[ $gpu_index =~ ^[0-9]+$ ]] || die '--gpu-index must be a non-negative ordinal'
[[ $gpu_uuid =~ ^GPU-[0-9A-Fa-f-]+$ ]] || die '--gpu-uuid must be an NVIDIA GPU UUID'

for numeric_option in \
    max_gpu_memory_mib startup_timeout_seconds request_timeout_seconds shutdown_timeout_seconds
do
    [[ ${!numeric_option} =~ ^[1-9][0-9]*$ ]] || die "--${numeric_option//_/-} must be a positive integer"
done
for port_name in stable_port max_port; do
    port=${!port_name}
    [[ $port =~ ^[0-9]+$ ]] || die "--${port_name//_/-} must be a TCP port"
    ((port >= 1024 && port <= 65535)) || die "--${port_name//_/-} must be from 1024 through 65535"
done
[[ $stable_port != "$max_port" ]] || die '--stable-port and --max-port must differ'

require_absolute() {
    local label=$1
    local path=$2
    [[ $path == /* && $path != *$'\n'* && $path != *$'\r'* ]] || die "$label must be an absolute single-line path"
}

require_regular_file() {
    local label=$1
    local path=$2
    require_absolute "$label" "$path"
    [[ -f $path && ! -L $path ]] || die "$label must be a regular non-symlink file"
}

require_container_image() {
    local image=$1
    [[ $image =~ ^[A-Za-z0-9][A-Za-z0-9._/:@+-]*$ ]] || \
        die '--container-image must be a single Docker image reference without whitespace'
}

require_container_binary() {
    local path=$1
    [[ $path =~ ^/[A-Za-z0-9._/+@=-]+$ ]] || \
        die '--container-binary must be an absolute, single-line safe container path'
}

require_absolute '--model-dir' "$model_dir"
[[ -d $model_dir && ! -L $model_dir ]] || die '--model-dir must be a real non-symlink directory'
model_dir=$(/usr/bin/realpath -e "$model_dir") || die '--model-dir cannot be resolved'

require_regular_file '--stable-args-file' "$stable_args_file"
require_regular_file '--stable-env-file' "$stable_env_file"
require_regular_file '--max-args-file' "$max_args_file"
require_regular_file '--max-env-file' "$max_env_file"
stable_args_file=$(/usr/bin/realpath -e "$stable_args_file")
stable_env_file=$(/usr/bin/realpath -e "$stable_env_file")
max_args_file=$(/usr/bin/realpath -e "$max_args_file")
max_env_file=$(/usr/bin/realpath -e "$max_env_file")

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)
repo_root=$(/usr/bin/realpath -e "$repo_root") || die 'cannot resolve repository root'
require_absolute '--evidence-dir' "$evidence_dir"
evidence_parent=$(/usr/bin/realpath -e "$(dirname -- "$evidence_dir")") || \
    die '--evidence-dir parent must already exist and be resolvable'
evidence_leaf=$(basename -- "$evidence_dir")
[[ $evidence_leaf != . && $evidence_leaf != .. && $evidence_leaf != / ]] || die '--evidence-dir has an unsafe basename'
evidence_dir="$evidence_parent/$evidence_leaf"
case $evidence_dir in
    "$repo_root"|"$repo_root"/*) die '--evidence-dir must be outside the source tree' ;;
esac
[[ ! -e $evidence_dir && ! -L $evidence_dir ]] || die '--evidence-dir must be a new path'

sha_file() {
    /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'
}

verify_container_binary() {
    local checksum observed_checksum
    checksum=$(docker run --rm --pull=never --network none --entrypoint /usr/bin/env "$container_image_id" \
        -i LC_ALL=C /bin/sh -ec '
candidate=$1
if [ ! -f "$candidate" ] || [ -L "$candidate" ] || [ ! -x "$candidate" ]; then
    printf "%s\\n" "container binary is not an executable non-symlink regular file: $candidate" >&2
    exit 64
fi
exec /usr/bin/sha256sum -- "$candidate"
' riley-c02-container-binary-check "$container_binary") || \
        die 'could not verify the in-image --container-binary'
    observed_checksum=${checksum%%[[:space:]]*}
    [[ $observed_checksum =~ $sha_re && $checksum == "$observed_checksum  $container_binary" ]] || \
        die 'in-image --container-binary SHA-256 output is malformed'
    [[ $observed_checksum == "$binary_sha256" ]] || \
        die 'in-image --container-binary SHA-256 does not match --binary-sha256'
}

if [[ $launch_mode == host-binary ]]; then
    require_absolute '--binary' "$binary"
    [[ -x $binary && ! -L $binary ]] || die '--binary must be an executable non-symlink file'
    binary=$(/usr/bin/realpath -e "$binary") || die '--binary cannot be resolved'
    [[ $(sha_file "$binary") == "$binary_sha256" ]] || die '--binary SHA-256 does not match --binary-sha256'
else
    require_container_image "$container_image"
    require_container_binary "$container_binary"
    container_image_id=$(docker image inspect --format '{{.Id}}' "$container_image") || \
        die '--container-image is not available locally'
    [[ $container_image_id =~ ^sha256:[0-9a-f]{64}$ ]] || \
        die '--container-image did not resolve to one immutable Docker image ID'
    verify_container_binary
    container_user="$(id -u):$(id -g)"
    [[ $container_user =~ ^[0-9]+:[0-9]+$ ]] || die 'could not resolve the host capture user'
fi

scratch_dir=$(/usr/bin/mktemp -d /tmp/riley-c02-p0-capture.XXXXXX)
server_pid=

stop_server() {
    local attempts=0
    local pid=${server_pid:-}
    [[ -n $pid ]] || return 0
    if /bin/kill -0 "$pid" >/dev/null 2>&1; then
        /bin/kill -TERM "$pid" >/dev/null 2>&1 || true
        while /bin/kill -0 "$pid" >/dev/null 2>&1 && ((attempts < shutdown_timeout_seconds)); do
            /usr/bin/sleep 1
            attempts=$((attempts + 1))
        done
        if /bin/kill -0 "$pid" >/dev/null 2>&1; then
            /bin/kill -KILL "$pid" >/dev/null 2>&1 || true
        fi
    fi
    wait "$pid" 2>/dev/null || true
    server_pid=
}

cleanup() {
    local status=$?
    trap - EXIT
    stop_server
    [[ -n ${scratch_dir:-} && -d $scratch_dir ]] && /bin/rm -rf -- "$scratch_dir"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

model_manifest="$scratch_dir/model.SHA256SUMS"
if /usr/bin/find "$model_dir" -mindepth 1 ! -type d ! -type f -print -quit | /usr/bin/grep -q .; then
    die '--model-dir contains a symlink or non-regular entry'
fi
model_file_count=0
while IFS= read -r -d '' model_file; do
    relative_path=${model_file#"$model_dir"/}
    [[ $relative_path =~ ^[A-Za-z0-9._/+@=-]+$ ]] || die "--model-dir contains unsafe model path: $relative_path"
    printf '%s  %s\n' "$(sha_file "$model_file")" "$relative_path" >>"$model_manifest"
    model_file_count=$((model_file_count + 1))
done < <(/usr/bin/find "$model_dir" -type f -print0 | /usr/bin/sort -z)
((model_file_count > 0)) || die '--model-dir contains no regular files'
[[ $(sha_file "$model_manifest") == "$model_tree_sha256" ]] || die '--model-dir tree SHA-256 does not match --model-tree-sha256'

# Coordinate with the existing GPU-evidence runner. This lock is advisory,
# process-scoped, and no evidence can be created until it is held.
[[ ! -L $GPU_LOCK_PATH ]] || die "unsafe GPU lock path: $GPU_LOCK_PATH"
exec 9>>"$GPU_LOCK_PATH"
/usr/bin/flock -n 9 || die 'another GPU evidence capture holds the host lock'

preflight_gpu() {
    local gpu_probe observed_gpu_uuid observed_gpu_memory
    gpu_probe=$(/usr/bin/nvidia-smi -i "$gpu_index" --query-gpu=uuid,memory.used --format=csv,noheader,nounits) || \
        die 'nvidia-smi preflight failed'
    [[ $(printf '%s\n' "$gpu_probe" | /usr/bin/wc -l | /usr/bin/tr -d '[:space:]') == 1 ]] || \
        die 'nvidia-smi preflight returned an ambiguous GPU inventory'
    IFS=, read -r observed_gpu_uuid observed_gpu_memory <<<"$gpu_probe"
    observed_gpu_uuid=${observed_gpu_uuid//[[:space:]]/}
    observed_gpu_memory=${observed_gpu_memory//[[:space:]]/}
    [[ $observed_gpu_uuid == "$gpu_uuid" ]] || die "GPU UUID mismatch: expected $gpu_uuid, observed $observed_gpu_uuid"
    [[ $observed_gpu_memory =~ ^[0-9]+$ ]] || die 'nvidia-smi returned a non-numeric used-memory value'
    ((observed_gpu_memory <= max_gpu_memory_mib)) || \
        die "GPU preflight failed: ${observed_gpu_memory}MiB is above ${max_gpu_memory_mib}MiB"
}

preflight_gpu

[[ ! -e $evidence_dir && ! -L $evidence_dir ]] || die '--evidence-dir appeared during preflight'
/bin/mkdir -m 0700 "$evidence_dir" || die 'could not create new evidence directory'

declare -a loaded_arguments=()
declare -a loaded_environment=()

load_arguments() {
    local file=$1
    local argument
    loaded_arguments=()
    while IFS= read -r argument || [[ -n $argument ]]; do
        [[ -n $argument && $argument != *$'\r'* ]] || die "argument file contains an empty or CR-terminated argument: $file"
        case $argument in
            serve|--help|-h|--version|--model|--model=*|--bind|--bind=*|--device|--device=*|--c02-candidate-id|--c02-candidate-id=*|--c02-configuration-profile|--c02-configuration-profile=*|--c02-startup-artifact|--c02-startup-artifact=*)
                die "argument file attempts to override runner-owned argument: $argument"
                ;;
        esac
        loaded_arguments+=("$argument")
    done <"$file"
}

load_environment() {
    local file=$1
    local entry key value
    local -A seen=()
    loaded_environment=()
    while IFS= read -r entry || [[ -n $entry ]]; do
        [[ -n $entry && $entry != *$'\r'* && $entry == *=* ]] || \
            die "environment file contains an invalid KEY=VALUE entry: $file"
        key=${entry%%=*}
        value=${entry#*=}
        [[ $key =~ $env_key_re ]] || die "environment file has an invalid key: $key"
        [[ -z ${seen[$key]+x} ]] || die "environment file repeats a key: $key"
        seen[$key]=1
        case $key in
            RILEY_FREEZE_SHA|RILEY_GATE_E_REPORT_SHA|RILEY_CONFIGURATION_SHA|RILEY_BASE_RELEASE_CANDIDATE_REPORT_SHA)
                die "environment file contains forbidden self-referential C02 input: $key"
                ;;
        esac
        loaded_environment+=("$key=$value")
    done <"$file"
    ((${#loaded_environment[@]} > 0)) || die "environment file must contain at least one explicit KEY=VALUE entry: $file"
}

copy_create_only() {
    local source=$1
    local destination=$2
    /usr/bin/python3 -B -I -S - "$source" "$destination" <<'PY'
import os
import stat
import sys

source, destination = sys.argv[1:]
source_stat = os.lstat(source)
if not stat.S_ISREG(source_stat.st_mode):
    raise SystemExit(f"capture source is not a regular file: {source}")
parent_stat = os.lstat(os.path.dirname(destination))
if not stat.S_ISDIR(parent_stat.st_mode):
    raise SystemExit(f"capture destination parent is not a directory: {destination}")
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
fd = os.open(destination, flags, 0o600)
try:
    with open(source, "rb", buffering=0) as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            view = memoryview(block)
            while view:
                written = os.write(fd, view)
                view = view[written:]
    os.fsync(fd)
finally:
    os.close(fd)
PY
}

write_container_runtime_receipt() {
    local destination="$evidence_dir/container-runtime.json"
    /usr/bin/python3 -B -I -S - \
        "$destination" "$container_image" "$container_image_id" "$container_binary" \
        "$binary_sha256" "$model_tree_sha256" "$gpu_index" "$gpu_uuid" "$container_user" <<'PY'
import json
import os
import stat
import sys

(
    destination,
    image_reference,
    image_id,
    binary_path,
    binary_sha256,
    model_tree_sha256,
    host_gpu_index,
    host_gpu_uuid,
    container_user,
) = sys.argv[1:]
parent_stat = os.lstat(os.path.dirname(destination))
if not stat.S_ISDIR(parent_stat.st_mode):
    raise SystemExit(f"container receipt parent is not a directory: {destination}")
payload = {
    "binary_sha256": binary_sha256,
    "container_binary": binary_path,
    "container_gpu_index": 0,
    "container_image_id": image_id,
    "container_image_reference": image_reference,
    "container_user": container_user,
    "host_gpu_index": int(host_gpu_index),
    "host_gpu_uuid": host_gpu_uuid,
    "model_tree_sha256": model_tree_sha256,
    "schema_version": 1,
}
encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii") + b"\n"
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
fd = os.open(destination, flags, 0o600)
try:
    view = memoryview(encoded)
    while view:
        written = os.write(fd, view)
        view = view[written:]
    os.fsync(fd)
finally:
    os.close(fd)
PY
}

validate_capture() {
    local profile=$1
    local endpoint_path=$2
    local artifact_path=$3
    local server_artifact_path=$4
    /usr/bin/python3 -B -I -S \
        "$repo_root/ci/release/validate_raw_c02_runtime_config.py" \
        --profile "$profile" \
        --candidate-id "$candidate_id" \
        --endpoint "$endpoint_path" \
        --startup-artifact "$artifact_path" \
        --server-startup-artifact "$server_artifact_path"
}

wait_for_ready() {
    local port=$1
    local attempt status
    for ((attempt = 0; attempt < startup_timeout_seconds; attempt++)); do
        if status=$(/usr/bin/curl --noproxy '*' --http1.1 --silent --show-error \
            --connect-timeout 1 --max-time 2 --output /dev/null --write-out '%{http_code}' \
            "http://127.0.0.1:${port}/readyz" 2>/dev/null); then
            [[ $status == 200 ]] && return 0
        fi
        if ! /bin/kill -0 "$server_pid" >/dev/null 2>&1; then
            wait "$server_pid" 2>/dev/null || true
            server_pid=
            die 'server exited before readiness; inspect the retained arm log'
        fi
        /usr/bin/sleep 1
    done
    die "server did not become ready within ${startup_timeout_seconds}s"
}

captured_config_hash=

if [[ $launch_mode == container ]]; then
    write_container_runtime_receipt
fi

capture_arm() {
    local profile=$1
    local port=$2
    local args_file=$3
    local env_file=$4
    local arm_dir="$evidence_dir/$profile"
    local server_output_dir=
    local server_artifact=
    local endpoint_body="$arm_dir/endpoint-payload.json"
    local captured_artifact="$arm_dir/startup-artifact.json"
    local endpoint_headers="$arm_dir/endpoint-headers.txt"
    local endpoint_temporary="$arm_dir/endpoint-payload.pending"
    local server_log="$arm_dir/server.log"
    local status content_length content_length_count actual_length config_hash

    /bin/mkdir -m 0700 "$arm_dir" || die "could not create $profile evidence directory"
    if [[ $launch_mode == container ]]; then
        server_output_dir="$arm_dir/container-server-output"
        /bin/mkdir -m 0700 "$server_output_dir" || die "could not create $profile container output directory"
        server_artifact="$server_output_dir/server-startup-artifact.json"
    else
        server_artifact="$arm_dir/server-startup-artifact.json"
    fi
    for output_path in "$server_artifact" "$endpoint_body" "$captured_artifact" "$endpoint_headers" "$endpoint_temporary" "$server_log"; do
        [[ ! -e $output_path && ! -L $output_path ]] || die "evidence path already exists: $output_path"
    done

    preflight_gpu
    load_arguments "$args_file"
    load_environment "$env_file"
    if [[ $launch_mode == host-binary ]]; then
        (
            exec /usr/bin/env -i "${loaded_environment[@]}" \
                "$binary" serve \
                --model "$model_dir" \
                --bind "127.0.0.1:${port}" \
                --device "$gpu_index" \
                --c02-candidate-id "$candidate_id" \
                --c02-configuration-profile "$profile" \
                --c02-startup-artifact "$server_artifact" \
                "${loaded_arguments[@]}"
        ) </dev/null >"$server_log" 2>&1 &
    else
        # Docker exposes just the selected host GPU. Within that constrained
        # namespace it is deterministically CUDA device 0; never pass the host
        # ordinal through as the server's in-container --device value. Host
        # networking deliberately preserves the same loopback-only bind that
        # the host-binary mode records and probes.
        (
            exec docker run --rm --pull=never \
                --user "$container_user" \
                --gpus "device=${gpu_index}" \
                --network host \
                --mount "type=bind,src=$model_dir,dst=$CONTAINER_MODEL_DIR,readonly" \
                --mount "type=bind,src=$server_output_dir,dst=$CONTAINER_SERVER_OUTPUT_DIR" \
                --entrypoint /usr/bin/env \
                "$container_image_id" \
                -i "${loaded_environment[@]}" \
                "$container_binary" serve \
                --model "$CONTAINER_MODEL_DIR" \
                --bind "127.0.0.1:${port}" \
                --device "$CONTAINER_VISIBLE_GPU_INDEX" \
                --c02-candidate-id "$candidate_id" \
                --c02-configuration-profile "$profile" \
                --c02-startup-artifact "$CONTAINER_SERVER_OUTPUT_DIR/server-startup-artifact.json" \
                "${loaded_arguments[@]}"
        ) </dev/null >"$server_log" 2>&1 &
    fi
    server_pid=$!

    wait_for_ready "$port"
    if ! status=$(/usr/bin/curl --noproxy '*' --http1.1 --silent --show-error \
        --connect-timeout 2 --max-time "$request_timeout_seconds" \
        --dump-header "$endpoint_headers" --output "$endpoint_temporary" --write-out '%{http_code}' \
        --request GET "http://127.0.0.1:${port}/v1/config"); then
        die "GET /v1/config failed for $profile"
    fi
    [[ $status == 200 ]] || die "GET /v1/config returned HTTP $status for $profile"
    content_length_count=$(/usr/bin/grep -a -i '^content-length:' "$endpoint_headers" | /usr/bin/wc -l | /usr/bin/tr -d '[:space:]' || true)
    [[ $content_length_count == 1 ]] || die "GET /v1/config must return exactly one Content-Length header for $profile"
    content_length=$(/usr/bin/grep -a -i '^content-length:' "$endpoint_headers" | /usr/bin/tr -d '\r' | /usr/bin/awk '{print $2}')
    [[ $content_length =~ ^[0-9]+$ ]] || die "GET /v1/config returned an invalid Content-Length for $profile"
    actual_length=$(/usr/bin/wc -c <"$endpoint_temporary" | /usr/bin/tr -d '[:space:]')
    [[ $actual_length == "$content_length" ]] || die "GET /v1/config Content-Length/body mismatch for $profile"
    [[ -f $server_artifact && ! -L $server_artifact ]] || die "server did not create a regular C02 startup artifact for $profile"

    copy_create_only "$endpoint_temporary" "$endpoint_body"
    copy_create_only "$server_artifact" "$captured_artifact"
    config_hash=$(validate_capture "$profile" "$endpoint_body" "$captured_artifact" "$server_artifact") || \
        die "captured C02 raw evidence is invalid for $profile"
    [[ $config_hash =~ $sha_re ]] || \
        die "raw C02 validator returned an invalid effective-config SHA-256 for $profile"
    stop_server
    captured_config_hash=$config_hash
}

capture_arm stable-default "$stable_port" "$stable_args_file" "$stable_env_file"
stable_config_hash=$captured_config_hash
capture_arm max-performance-exact "$max_port" "$max_args_file" "$max_env_file"
max_config_hash=$captured_config_hash
[[ $stable_config_hash != "$max_config_hash" ]] || \
    die 'two C02-P0 arms resolved the same effective_config; refusing indistinguishable raw evidence'

printf 'C02-P0 raw capture complete (no C02 qualification was run): %s\n' "$evidence_dir"
printf '  stable endpoint: %s\n' "$evidence_dir/stable-default/endpoint-payload.json"
printf '  stable artifact: %s\n' "$evidence_dir/stable-default/startup-artifact.json"
printf '  max endpoint: %s\n' "$evidence_dir/max-performance-exact/endpoint-payload.json"
printf '  max artifact: %s\n' "$evidence_dir/max-performance-exact/startup-artifact.json"
