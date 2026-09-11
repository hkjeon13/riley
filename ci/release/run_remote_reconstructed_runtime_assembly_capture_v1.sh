#!/usr/bin/env bash
# Capture one source-free reconstructed runtime-image assembly without running it.
#
# This is a raw Docker host producer.  It deliberately creates no candidate
# freeze, OCI-input receipt, assembly-capture receipt, content bridge, semantic
# report, qualification decision, service, or GPU evidence.  A later 3.11+
# source-only replay consumes the raw output.  The runner itself never starts
# the created container.

set -euo pipefail
set -o noclobber
umask 077
IFS=$' \t\n'

readonly SCRIPT_NAME='run_remote_reconstructed_runtime_assembly_capture_v1.sh'
readonly DOCKER_LOCK_PATH='/var/tmp/riley-reconstructed-runtime-assembly.lock'
readonly SUPERVISOR_LOCK_FD=9
readonly RECIPE_NORMALIZED_INSTRUCTIONS_SHA256='d80d657db557f9af62734aebef3527fcf46a0227de1f9ac1cacbbf0c70751114'
readonly ZERO_SHA256='0000000000000000000000000000000000000000000000000000000000000000'
readonly PINNED_RUNTIME='nvidia/cuda:12.8.1-runtime-ubuntu22.04@sha256:fcbbd60a5ad3db3a1c7375bf14546b369b54064c513224310b2026df50c7a9bd'
readonly MAX_BUILD_LOG_BYTES=16777216
readonly MAX_JSON_BYTES=1048576
readonly MAX_RUNTIME_TREE_BYTES=2147483648
readonly MAX_IMAGE_EXPORT_ARCHIVE_BYTES=8657043455
readonly MAX_ID_BYTES=80
readonly MAX_DIAGNOSTIC_FILE_BLOCKS=2048

usage() {
    /bin/cat <<'EOF'
usage: bash ci/release/run_remote_reconstructed_runtime_assembly_capture_v1.sh \
  --reconstruction-id a|b \
  --source-revision LOWERCASE_40_HEX_GIT_REVISION \
  --expected-source-archive-sha256 LOWERCASE_SHA256 \
  --repro-build-inputs-sha256 LOWERCASE_SHA256 \
  --release-binary ABSOLUTE_RELEASE_BINARY \
  --release-binary-sha256 LOWERCASE_SHA256 \
  --release-bundle ABSOLUTE_RELEASE_BUNDLE \
  --release-bundle-sha256 LOWERCASE_SHA256 \
  --evidence-dir NEW_ABSOLUTE_EXTERNAL_DIRECTORY

Build exactly one source-free linux/amd64 runtime image from a canonical
three-member USTAR context, preserve Docker's raw image-save bytes, normalize
them to canonical OCI bytes, create (but never start) a network-none container,
and emit the fixed raw assembly-capture USTAR.  The evidence directory must be
new, mode 0700, and outside this source checkout (including mount aliases). It
is retained on failure. The pinned base must already be present locally:
``--network none`` constrains Docker build-step networking, not daemon/control-
plane egress, so this runner does not claim host-network isolation.

The runner owns the Docker build/create/export command. It accepts no Docker
tag, cache, pull, mount, secret, SSH, GPU, service, or caller-command option.
It does not claim that the Docker build/export/copy occurred in one invocation,
that the image came from the source or bundle, or that any runtime/GPU/service
qualification passed.  Use a later reviewed replay on Python 3.11+ before
consuming these raw leaves.
EOF
}

outer_die() {
    printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
    exit 2
}

# Reject malformed calls before opening a lock, checking a path, or touching
# Docker. The authenticated child parses the exact same closed option set.
preflight_invocation() {
    local option
    local seen='|'
    local -a required=(
        --reconstruction-id --source-revision --expected-source-archive-sha256
        --repro-build-inputs-sha256 --release-binary --release-binary-sha256
        --release-bundle --release-bundle-sha256 --evidence-dir
    )
    while (($# > 0)); do
        option=$1
        case $option in
            --reconstruction-id|--source-revision|--expected-source-archive-sha256|--repro-build-inputs-sha256|--release-binary|--release-binary-sha256|--release-bundle|--release-bundle-sha256|--evidence-dir)
                (($# >= 2)) || outer_die "$option requires a value"
                [[ -n $2 && $2 != --* ]] || outer_die "$option requires a non-option value"
                [[ $seen != *"|${option}|"* ]] || outer_die "$option may occur only once"
                seen+="${option}|"
                shift 2
                ;;
            *)
                outer_die "unknown option: $option"
                ;;
        esac
    done
    for option in "${required[@]}"; do
        [[ $seen == *"|${option}|"* ]] || outer_die "missing $option"
    done
}

if (($# == 0)); then
    usage >&2
    exit 2
fi
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi

# The first Bash process cannot supply or substitute the lock descriptor. A
# small clean Python parent opens the no-follow lock, authenticates its inode,
# holds the nonblocking exclusive flock for the full child lifetime, and passes
# exactly that one descriptor into the child.  An attacker-supplied internal
# sentinel has no matching parent/descriptor and is rejected below.
if [[ ${1:-} != --assembly-lock-supervised ]]; then
    preflight_invocation "$@"
    exec /usr/bin/env -i \
        PATH=/usr/bin:/bin \
        LC_ALL=C \
        TZ=UTC \
        HOME=/nonexistent \
        /usr/bin/python3 -I -S -E -c '
import ctypes
import fcntl
import os
import secrets
import signal
import stat
import sys

LOCK_PATH = "/var/tmp/riley-reconstructed-runtime-assembly.lock"
LOCK_FD = 9
PR_SET_PDEATHSIG = 1

flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
try:
    opened_fd = os.open(LOCK_PATH, flags, 0o600)
except OSError as error:
    raise SystemExit(f"runtime assembly supervisor: cannot open Docker lock safely: {error}")
try:
    if opened_fd != LOCK_FD:
        os.dup2(opened_fd, LOCK_FD, inheritable=False)
        os.close(opened_fd)
    lock_fd = LOCK_FD
    metadata = os.fstat(lock_fd)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600):
        raise SystemExit("runtime assembly supervisor: unsafe Docker lock inode")
    named = os.stat(LOCK_PATH, follow_symlinks=False)
    if (named.st_dev, named.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("runtime assembly supervisor: Docker lock path changed while opening")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("runtime assembly supervisor: another runtime assembly capture holds the Docker lock")
    named = os.stat(LOCK_PATH, follow_symlinks=False)
    if (named.st_dev, named.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("runtime assembly supervisor: Docker lock path changed while locking")

    supervisor_pid = os.getpid()
    token = secrets.token_hex(32)
    forwarded = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, forwarded)
    ready_read, ready_write = os.pipe2(os.O_CLOEXEC)
    child_pid = os.fork()
    if child_pid == 0:
        os.close(ready_read)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
            os._exit(125)
        if os.getppid() != supervisor_pid:
            os._exit(125)
        try:
            inherited = os.listdir("/proc/self/fd")
        except OSError:
            os._exit(125)
        for raw_fd in inherited:
            try:
                fd = int(raw_fd)
            except ValueError:
                os._exit(125)
            if fd > 2 and fd not in {lock_fd, ready_write}:
                try:
                    os.close(fd)
                except OSError:
                    pass
        os.setsid()
        os.set_inheritable(lock_fd, True)
        try:
            os.write(ready_write, b"1")
        except OSError:
            os._exit(125)
        os.close(ready_write)
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        environment = {
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "TZ": "UTC",
            "HOME": "/nonexistent",
            "RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_PID": str(supervisor_pid),
            "RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_EXE": "/usr/bin/python3",
            "RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_LOCK_FD": str(lock_fd),
            "RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_LOCK_ID": f"{metadata.st_dev}:{metadata.st_ino}",
            "RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_TOKEN": token,
        }
        script = os.path.abspath(sys.argv[1])
        os.execve("/usr/bin/bash", ["/usr/bin/bash", script, "--assembly-lock-supervised", *sys.argv[2:]], environment)

    os.close(ready_write)
    try:
        ready = os.read(ready_read, 1)
    finally:
        os.close(ready_read)
    if ready != b"1":
        raise SystemExit("runtime assembly supervisor: child did not establish its isolated process group")

    def forward_signal(signum, _frame):
        try:
            os.killpg(child_pid, signum)
        except ProcessLookupError:
            pass

    for forwarded_signal in forwarded:
        signal.signal(forwarded_signal, forward_signal)
    signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    while True:
        try:
            _pid, wait_status = os.waitpid(child_pid, 0)
            break
        except InterruptedError:
            continue
    if os.WIFEXITED(wait_status):
        raise SystemExit(os.WEXITSTATUS(wait_status))
    if os.WIFSIGNALED(wait_status):
        raise SystemExit(128 + os.WTERMSIG(wait_status))
    raise SystemExit(125)
finally:
    try:
        os.close(LOCK_FD)
    except OSError:
        pass
' "$0" "$@"
fi
shift

[[ ${RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_PID:-} =~ ^[1-9][0-9]*$ ]] || outer_die 'supervisor PID was not authenticated'
[[ ${PPID} == "${RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_PID}" ]] || outer_die 'supervisor is not the direct parent'
[[ ${RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_EXE:-} == /usr/bin/python3 ]] || outer_die 'supervisor executable identity is invalid'
[[ /proc/${PPID}/exe -ef /usr/bin/python3 ]] || outer_die 'supervisor executable differs from expected Python'
[[ ${RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_LOCK_FD:-} == "$SUPERVISOR_LOCK_FD" ]] || outer_die 'supervisor lock descriptor is invalid'
[[ ${RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_LOCK_ID:-} =~ ^[0-9]+:[0-9]+$ ]] || outer_die 'supervisor lock identity is invalid'
[[ ${RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_TOKEN:-} =~ ^[0-9a-f]{64}$ ]] || outer_die 'supervisor token is invalid'
[[ /proc/${PPID}/fd/${SUPERVISOR_LOCK_FD} -ef "$DOCKER_LOCK_PATH" ]] || outer_die 'supervisor does not hold the canonical Docker lock inode'
[[ /proc/$$/fd/${SUPERVISOR_LOCK_FD} -ef "$DOCKER_LOCK_PATH" ]] || outer_die 'authenticated Docker lock descriptor was not inherited'
if ! /usr/bin/grep -Eq "^lock:.*FLOCK[[:space:]]+ADVISORY[[:space:]]+WRITE[[:space:]]+${PPID}([[:space:]]|$)" "/proc/${PPID}/fdinfo/${SUPERVISOR_LOCK_FD}"; then
    outer_die 'supervisor does not own the kernel Docker flock'
fi
exec 9>&-
[[ ! -e /proc/$$/fd/${SUPERVISOR_LOCK_FD} ]] || outer_die 'Bash retained the supervisor lock descriptor'
unset \
    RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_PID \
    RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_EXE \
    RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_LOCK_FD \
    RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_LOCK_ID \
    RILEY_RUNTIME_ASSEMBLY_SUPERVISOR_TOKEN \
    BASH_ENV ENV CDPATH
export PATH=/usr/bin:/bin
export LC_ALL=C TZ=UTC
hash -r

die() {
    printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
    exit 2
}

for tool in bash basename cat cmp dirname docker grep mktemp python3 realpath sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || die "required host tool is unavailable: $tool"
done

reconstruction_id=
source_revision=
expected_source_archive_sha256=
repro_build_inputs_sha256=
release_binary=
release_binary_sha256=
release_bundle=
release_bundle_sha256=
evidence_dir=
seen_options='|'

need_value() {
    local flag=$1
    (($# >= 2)) || die "$flag requires a value"
    [[ -n $2 ]] || die "$flag must not be empty"
}

set_once() {
    local flag=$1
    local variable=$2
    local value=$3
    [[ $seen_options != *"|${flag}|"* ]] || die "$flag may occur only once"
    seen_options+="${flag}|"
    printf -v "$variable" '%s' "$value"
}

while (($# > 0)); do
    case $1 in
        --reconstruction-id|--source-revision|--expected-source-archive-sha256|--repro-build-inputs-sha256|--release-binary|--release-binary-sha256|--release-bundle|--release-bundle-sha256|--evidence-dir)
            need_value "$1" "${2:-}"
            case $1 in
                --reconstruction-id) set_once "$1" reconstruction_id "$2" ;;
                --source-revision) set_once "$1" source_revision "$2" ;;
                --expected-source-archive-sha256) set_once "$1" expected_source_archive_sha256 "$2" ;;
                --repro-build-inputs-sha256) set_once "$1" repro_build_inputs_sha256 "$2" ;;
                --release-binary) set_once "$1" release_binary "$2" ;;
                --release-binary-sha256) set_once "$1" release_binary_sha256 "$2" ;;
                --release-bundle) set_once "$1" release_bundle "$2" ;;
                --release-bundle-sha256) set_once "$1" release_bundle_sha256 "$2" ;;
                --evidence-dir) set_once "$1" evidence_dir "$2" ;;
            esac
            shift 2
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

for required in reconstruction_id source_revision expected_source_archive_sha256 repro_build_inputs_sha256 release_binary release_binary_sha256 release_bundle release_bundle_sha256 evidence_dir; do
    [[ -n ${!required} ]] || die "missing --${required//_/-}"
done

readonly sha_re='^[0-9a-f]{64}$'
readonly revision_re='^[0-9a-f]{40}$'
[[ $reconstruction_id == a || $reconstruction_id == b ]] || die '--reconstruction-id must be exactly a or b'
[[ $source_revision =~ $revision_re && $source_revision != "${ZERO_SHA256:0:40}" ]] || die '--source-revision must be a non-zero lowercase 40-hex Git revision'
for hash_name in expected_source_archive_sha256 repro_build_inputs_sha256 release_binary_sha256 release_bundle_sha256; do
    [[ ${!hash_name} =~ $sha_re && ${!hash_name} != "$ZERO_SHA256" ]] || die "--${hash_name//_/-} must be a non-zero lowercase SHA-256"
done

require_absolute() {
    local label=$1
    local path=$2
    [[ $path == /* && $path != *$'\n'* && $path != *$'\r'* && $path != *\\* ]] || die "$label must be an absolute single-line path"
}

require_regular_file() {
    local label=$1
    local path=$2
    require_absolute "$label" "$path"
    [[ -f $path && ! -L $path ]] || die "$label must be a regular non-symlink file"
}

require_regular_file '--release-binary' "$release_binary"
require_regular_file '--release-bundle' "$release_bundle"
release_binary=$(/usr/bin/realpath -e "$release_binary") || die '--release-binary cannot be resolved'
release_bundle=$(/usr/bin/realpath -e "$release_bundle") || die '--release-bundle cannot be resolved'

script_path=$(/usr/bin/realpath -e "${BASH_SOURCE[0]}") || die 'runner script cannot be resolved'
[[ -f $script_path && ! -L $script_path ]] || die 'runner script is not a regular non-symlink file'
script_dir=$(CDPATH= cd -- "$(dirname -- "$script_path")" && pwd -P)
repo_root=$(CDPATH= cd -- "${script_dir}/../.." && pwd -P)
dockerfile="$repo_root/ci/release/ReconstructedRuntimeAssembly.Dockerfile"
normalizer="$repo_root/ci/release/prepare_reconstructed_runtime_image_export_oci_normalization_v1.py"
composer="$repo_root/ci/release/compose_reconstructed_runtime_assembly_capture_v1.py"
evidence_initializer="$repo_root/ci/release/initialize_reconstructed_runtime_assembly_evidence_v1.py"
recipe_verifier="$repo_root/ci/release/verify_reconstructed_runtime_assembly_dockerfile.py"
for trusted in "$dockerfile" "$normalizer" "$composer" "$evidence_initializer" "$recipe_verifier"; do
    [[ -f $trusted && ! -L $trusted ]] || die "trusted runner input is missing or unsafe: $trusted"
done

require_absolute '--evidence-dir' "$evidence_dir"
case $evidence_dir in
    "$repo_root"|"$repo_root"/*) die '--evidence-dir must be outside the source checkout' ;;
esac

sha_file() {
    /usr/bin/sha256sum "$1" | /usr/bin/grep -Eo '^[0-9a-f]{64}'
}

verify_static_recipe() {
    run_python "$recipe_verifier" --print-source-sha256
}

verify_release_inputs() {
    local phase=$1
    require_regular_file "release binary ($phase)" "$release_binary"
    require_regular_file "release bundle ($phase)" "$release_bundle"
    [[ $(sha_file "$release_binary") == "$release_binary_sha256" ]] || die "release binary SHA-256 changed at $phase"
    [[ $(sha_file "$release_bundle") == "$release_bundle_sha256" ]] || die "release bundle SHA-256 changed at $phase"
}

scratch_dir=
container_id=

cleanup() {
    local runner_exit=$?
    trap - EXIT
    if [[ ${container_id:-} =~ ^[0-9a-f]{64}$ ]]; then
        # Do not force removal: if an external actor started or replaced this
        # object, ordinary rm fails instead of stopping a container.
        docker container rm "$container_id" >/dev/null 2>&1 || true
    fi
    case ${scratch_dir:-} in
        /tmp/riley-reconstructed-runtime-assembly.*) [[ -d $scratch_dir ]] && /bin/rm -rf -- "$scratch_dir" ;;
        '') ;;
        *) printf '%s: retained unexpected scratch path %s\n' "$SCRIPT_NAME" "$scratch_dir" >&2 ;;
    esac
    exit "$runner_exit"
}

terminate_direct_children() {
    local exit_code=$1
    local children=''
    local child_pid
    # A parent-death signal targets Bash itself.  Terminate every direct
    # foreground pipeline client before running EXIT cleanup, so a build/save/
    # copy client cannot outlive the authenticated child while the lock parent
    # has already gone away.  The supervisor uses killpg after its readiness
    # handshake for ordinary externally delivered signals.
    trap - INT TERM HUP
    if [[ -r /proc/$$/task/$$/children ]]; then
        IFS= read -r children < "/proc/$$/task/$$/children" || true
        for child_pid in $children; do
            [[ $child_pid =~ ^[1-9][0-9]*$ ]] || continue
            kill -TERM "$child_pid" >/dev/null 2>&1 || true
        done
        # Do not let the supervisor's lock release while a direct Docker or
        # isolated-helper client is still alive after its termination request.
        # An unresponsive client therefore fails closed by retaining the lock
        # instead of permitting an overlapping capture.
        for child_pid in $children; do
            [[ $child_pid =~ ^[1-9][0-9]*$ ]] || continue
            wait "$child_pid" >/dev/null 2>&1 || true
        done
    fi
    exit "$exit_code"
}

trap cleanup EXIT
trap 'terminate_direct_children 130' INT
trap 'terminate_direct_children 143' TERM HUP

# Use a minimal isolated import bridge. ``-I`` deliberately removes the
# script directory; restore only the checked sibling directory after lstat.
run_python() {
    local program=$1
    shift
    [[ $program == "$repo_root"/ci/release/*.py && -f $program && ! -L $program ]] || \
        die "isolated Python target is missing or unsafe: $program"
    /usr/bin/env -i \
        PATH=/usr/bin:/bin \
        LC_ALL=C \
        TZ=UTC \
        PYTHONDONTWRITEBYTECODE=1 \
        /usr/bin/python3 -B -I -S -E -c '
import os
import runpy
import stat
import sys

script = sys.argv[1]
metadata = os.lstat(script)
if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
    raise SystemExit("unsafe isolated Python target")
directory = os.path.dirname(script)
if not os.path.isabs(script) or not directory:
    raise SystemExit("isolated Python target must be absolute")
sys.path.insert(0, directory)
sys.argv = [script, *sys.argv[2:]]
runpy.run_path(script, run_name="__main__")
' "$program" "$@"
}

dockerfile_sha256=$(verify_static_recipe) || die 'reviewed Dockerfile static verification failed'
[[ $dockerfile_sha256 =~ $sha_re && $dockerfile_sha256 != "$ZERO_SHA256" ]] || \
    die 'reviewed Dockerfile verifier did not return one source SHA-256'
verify_release_inputs pre-build

scratch_dir=$(/usr/bin/mktemp -d /tmp/riley-reconstructed-runtime-assembly.XXXXXX)
/bin/chmod 0700 "$scratch_dir"
# The authenticated child deliberately has no usable HOME.  Docker still needs
# a writable CLI config directory even for local inspect/build/save operations,
# so give it one ephemeral owner-only directory below this invocation's
# verified scratch root.  It is never caller supplied or evidence input.
docker_config_dir="$scratch_dir/docker-config"
/bin/mkdir --mode=0700 -- "$docker_config_dir"
export DOCKER_CONFIG="$docker_config_dir"
run_python "$evidence_initializer" --evidence-dir "$evidence_dir" >"$scratch_dir/evidence-initialization.json"
raw_dir="$evidence_dir/raw"

# The exact pinned base must already be resident in the local daemon. This
# query cannot pull; it prevents a missing base from turning a later build into
# a daemon-side fetch despite the Dockerfile's per-build-step network setting.
docker image inspect "$PINNED_RUNTIME" >/dev/null 2>&1 || \
    die 'reviewed pinned runtime base is not present locally; refusing any daemon pull'

context_tar="$raw_dir/context.tar"
run_python "$composer" context \
    --output "$context_tar" \
    --dockerfile "$dockerfile" \
    --dockerfile-sha256 "$dockerfile_sha256" \
    --release-binary "$release_binary" \
    --release-binary-sha256 "$release_binary_sha256" \
    --release-bundle "$release_bundle" \
    --release-bundle-sha256 "$release_bundle_sha256" \
    >"$scratch_dir/context-report.json"

# No tag is assigned and no mutable caller image reference is accepted. The
# iidfile is hard-capped by the client subprocess, then parsed by the held-FD
# helper as exact Docker raw bytes before it becomes an image selector.
if ! (
    ulimit -f 1
    docker build \
        --file Dockerfile \
        --platform linux/amd64 \
        --network none \
        --pull=false \
        --no-cache \
        --iidfile "$raw_dir/build.iid" \
        --build-arg "RILEY_RECONSTRUCTION_ID=$reconstruction_id" \
        --build-arg "RILEY_SOURCE_REVISION=$source_revision" \
        --build-arg "RILEY_SOURCE_ARCHIVE_SHA256=$expected_source_archive_sha256" \
        --build-arg "RILEY_REPRO_BUILD_INPUTS_SHA256=$repro_build_inputs_sha256" \
        --build-arg "RILEY_RELEASE_BINARY_SHA256=$release_binary_sha256" \
        --build-arg "RILEY_RELEASE_BUNDLE_SHA256=$release_bundle_sha256" \
        --build-arg "RILEY_RUNTIME_ASSEMBLY_RECIPE_SHA256=$RECIPE_NORMALIZED_INSTRUCTIONS_SHA256" \
        - <"$context_tar"
) 2>&1 | run_python "$composer" stream \
    --output "$raw_dir/build.log" \
    --maximum-bytes "$MAX_BUILD_LOG_BYTES" \
    >"$scratch_dir/build-log-stream.json"; then
    die 'docker build failed or exceeded the raw build.log bound; retained evidence is incomplete'
fi

image_id=$(run_python "$composer" read-id --kind image --input "$raw_dir/build.iid") || \
    die 'docker build iidfile is not one exact immutable Docker image ID'
[[ $image_id =~ ^sha256:[0-9a-f]{64}$ && $image_id != "sha256:$ZERO_SHA256" ]] || \
    die 'validated docker build iidfile is not one immutable image ID'

# Keep the raw Docker-save bytes distinct from the canonical OCI archive. The
# normalizer records raw format/sidecars and creates its own private root.
if ! (
    ulimit -f "$MAX_DIAGNOSTIC_FILE_BLOCKS"
    docker image inspect "$image_id" 2>"$scratch_dir/image-inspect.stderr"
) | run_python "$composer" stream \
    --output "$raw_dir/image-inspect.json" \
    --maximum-bytes "$MAX_JSON_BYTES" \
    >"$scratch_dir/image-inspect-stream.json"; then
    die 'docker image inspect failed or exceeded the raw inspect bound'
fi
if ! (
    ulimit -f "$MAX_DIAGNOSTIC_FILE_BLOCKS"
    docker image save "$image_id" 2>"$scratch_dir/image-save.stderr"
) | run_python "$composer" stream \
    --output "$raw_dir/runtime-image-export.tar" \
    --maximum-bytes "$MAX_IMAGE_EXPORT_ARCHIVE_BYTES" \
    >"$scratch_dir/image-save-stream.json"; then
    die 'docker image save failed or exceeded the raw export bound'
fi
run_python "$normalizer" \
    --evidence-root "$evidence_dir/normalization" \
    --image-inspect "$raw_dir/image-inspect.json" \
    --image-export-archive "$raw_dir/runtime-image-export.tar" \
    --reconstruction-id "$reconstruction_id" \
    >"$scratch_dir/normalization-report.json"
canonical_oci="$evidence_dir/normalization/normalized-oci/oci-image-layout.tar"
[[ -f $canonical_oci && ! -L $canonical_oci ]] || die 'normalizer did not create canonical OCI bytes'

# This is deliberately ``create`` rather than run/start/exec. The v1 capture
# verifier accepts the Docker daemon's default private namespace representation
# only, so no extra cap/security/mount/device options are supplied here.
if ! (
    ulimit -f "$MAX_DIAGNOSTIC_FILE_BLOCKS"
    docker create --network none --restart no "$image_id" 2>"$scratch_dir/container-create.stderr"
) | run_python "$composer" stream \
    --output "$scratch_dir/container-id.raw" \
    --maximum-bytes "$MAX_ID_BYTES" \
    >"$scratch_dir/container-create-stream.json"; then
    die 'docker create failed or returned an oversized ID stream'
fi
container_id=$(run_python "$composer" read-id --kind container --input "$scratch_dir/container-id.raw") || \
    die 'docker create did not return one exact immutable container ID'
[[ $container_id =~ ^[0-9a-f]{64}$ ]] || die 'validated docker create ID is invalid'
if ! (
    ulimit -f "$MAX_DIAGNOSTIC_FILE_BLOCKS"
    docker container inspect "$container_id" 2>"$scratch_dir/container-inspect.stderr"
) | run_python "$composer" stream \
    --output "$raw_dir/container-inspect.json" \
    --maximum-bytes "$MAX_JSON_BYTES" \
    >"$scratch_dir/container-inspect-stream.json"; then
    die 'docker container inspect failed or exceeded the raw inspect bound'
fi
if ! (
    ulimit -f "$MAX_DIAGNOSTIC_FILE_BLOCKS"
    docker cp "$container_id:/opt/riley" - 2>"$scratch_dir/container-copy.stderr"
) | run_python "$composer" stream \
    --output "$raw_dir/container-opt-riley.docker-cp.tar" \
    --maximum-bytes "$MAX_RUNTIME_TREE_BYTES" \
    >"$scratch_dir/container-copy-stream.json"; then
    die 'docker cp failed or exceeded the raw runtime-tree bound'
fi
if ! (
    ulimit -f "$MAX_DIAGNOSTIC_FILE_BLOCKS"
    docker container inspect "$container_id" 2>"$scratch_dir/container-inspect-after.stderr"
) | run_python "$composer" stream \
    --output "$scratch_dir/container-inspect-after.json" \
    --maximum-bytes "$MAX_JSON_BYTES" \
    >"$scratch_dir/container-inspect-after-stream.json"; then
    die 'post-copy docker container inspect failed or exceeded the raw inspect bound'
fi
cmp --silent "$raw_dir/container-inspect.json" "$scratch_dir/container-inspect-after.json" || \
    die 'container state changed between pre-copy and post-copy inspection'
run_python "$composer" runtime-tree \
    --input "$raw_dir/container-opt-riley.docker-cp.tar" \
    --output "$raw_dir/container-opt-riley.tar" \
    >"$scratch_dir/runtime-tree-report.json"

run_python "$composer" capture \
    --output "$raw_dir/assembly-capture.tar" \
    --context "$context_tar" \
    --runtime-tree "$raw_dir/container-opt-riley.tar" \
    --build-iid "$raw_dir/build.iid" \
    --build-log "$raw_dir/build.log" \
    --image-inspect "$raw_dir/image-inspect.json" \
    --oci-archive "$canonical_oci" \
    --container-inspect "$raw_dir/container-inspect.json" \
    --reconstruction-id "$reconstruction_id" \
    --source-revision "$source_revision" \
    --expected-source-archive-sha256 "$expected_source_archive_sha256" \
    --repro-build-inputs-sha256 "$repro_build_inputs_sha256" \
    --release-binary-sha256 "$release_binary_sha256" \
    --release-bundle-sha256 "$release_bundle_sha256" \
    --recipe-normalized-instructions-sha256 "$RECIPE_NORMALIZED_INSTRUCTIONS_SHA256" \
    --image-id "$image_id" \
    --container-id "$container_id" \
    >"$scratch_dir/capture-report.json"

verify_release_inputs post-capture
verify_static_recipe >/dev/null

# A successful raw snapshot is still not a same-invocation attestation or a
# qualification result. Only remove the exact non-running container created by
# this invocation; retain the untagged image and all raw evidence for review.
docker container rm "$container_id" >/dev/null || die 'could not remove the never-started captured container'
container_id=
/bin/cat "$scratch_dir/capture-report.json"
