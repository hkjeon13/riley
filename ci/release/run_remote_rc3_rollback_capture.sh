#!/usr/bin/env bash
# Run one raw RC3-to-reconstructed-RC2 rollback capture under an authenticated
# GPU supervisor. It never declares a semantic rollback or qualification result.

set -euo pipefail
set -o noclobber
umask 077
IFS=$' \t\n'

readonly SCRIPT_NAME='run_remote_rc3_rollback_capture.sh'
readonly GPU_LOCK_PATH='/var/tmp/riley-server-4096-gpu-evidence.lock'
readonly SUPERVISOR_LOCK_FD=9
readonly DEFAULT_MAX_GPU_MEMORY_MIB=256
readonly DEFAULT_STARTUP_TIMEOUT_SECONDS=60
readonly DEFAULT_SHUTDOWN_TIMEOUT_SECONDS=30
readonly DEFAULT_AUDIT_WAIT_SECONDS=15
readonly CONFIG_CAPTURE_NAME='config-bridge'
readonly CANDIDATE_PHASE_CAPTURE_NAME='candidate-phase'
readonly SERIAL_CAPTURE_NAME='serial-capture'
readonly ROLLBACK_PHASE_CAPTURE_NAME='rollback-phase'
readonly SOURCE_AUDIT_DIRECTORY='source-audit'
readonly STABLE_DEFAULT_PROFILE='stable-default'
readonly ZERO_SHA256='0000000000000000000000000000000000000000000000000000000000000000'

usage() {
    /bin/cat <<'EOF'
usage: bash ci/release/run_remote_rc3_rollback_capture.sh \
  --evidence-root EXISTING_PRIVATE_RECONSTRUCTED_RC2_ROOT \
  --baseline-manifest-path ROOT_RELATIVE_BASELINE_MANIFEST \
  --candidate-id riley-X.Y.Z-rcN \
  --freeze-input ABSOLUTE_OPAQUE_FREEZE_INPUT \
  --base-release-candidate-report-input ABSOLUTE_OPAQUE_REPORT_INPUT \
  --stable-default-configuration-input ABSOLUTE_OPAQUE_CONFIGURATION_INPUT \
  --candidate-binary ABSOLUTE_CANDIDATE_BINARY \
  --candidate-binary-sha256 LOWERCASE_SHA256 \
  --candidate-bundle ABSOLUTE_CANDIDATE_BUNDLE \
  --candidate-image-inspect ABSOLUTE_CANDIDATE_IMAGE_INSPECT \
  --candidate-model-dir ABSOLUTE_CANDIDATE_MODEL_DIRECTORY \
  --candidate-model-tree-sha256 LOWERCASE_SHA256 \
  --candidate-args-file ABSOLUTE_LINE_DELIMITED_ARGUMENT_FILE \
  --candidate-env-file ABSOLUTE_KEY_VALUE_ENVIRONMENT_FILE \
  --candidate-port LOOPBACK_PORT \
  --candidate-scenario-contract ABSOLUTE_CANONICAL_JSON_FILE \
  --rollback-binary ABSOLUTE_RECONSTRUCTED_RC2_BINARY \
  --rollback-binary-sha256 LOWERCASE_SHA256 \
  --rollback-bundle ABSOLUTE_RECONSTRUCTED_RC2_BUNDLE \
  --rollback-image-inspect ABSOLUTE_RECONSTRUCTED_RC2_IMAGE_INSPECT \
  --rollback-model-dir ABSOLUTE_RECONSTRUCTED_RC2_MODEL_DIRECTORY \
  --rollback-model-tree-sha256 LOWERCASE_SHA256 \
  --rollback-args-file ABSOLUTE_LINE_DELIMITED_ARGUMENT_FILE \
  --rollback-env-file ABSOLUTE_KEY_VALUE_ENVIRONMENT_FILE \
  --rollback-port LOOPBACK_PORT \
  --rollback-generation-request ABSOLUTE_CANONICAL_NONSTREAM_REQUEST \
  --gpu-index ORDINAL \
  [--max-gpu-memory-mib 256] \
  [--startup-timeout-seconds 60] \
  [--shutdown-timeout-seconds 30] \
  [--audit-wait-seconds 15]

This is a raw RC3-to-reconstructed-RC2 rollback provenance capture. The
evidence root must already contain one complete reconstructed RC2 A/B
baseline. This runner owns every capture name, config path, source-audit path,
process identity, target tuple, manifest and receipt name. It accepts no
server command, configuration SHA, PID/start tick, listener/GPU UUID, target,
manifest, or receipt override.

Candidate arguments may not override serve, --model, --bind, --device, or any
--c02-* option. Rollback arguments may not inject a C02 option. Both servers
are launched through env -i and shut down only through the bound process
guard. Evidence is retained on failure. No candidate freeze, deployment-path
mutation, semantic check, Gate E action, or qualification verdict is run.
EOF
}

outer_die() {
    printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
    exit 2
}

# This outer parser deliberately performs no path, lock, or GPU operation.
preflight_invocation() {
    local option
    local seen='|'
    local -a required=(
        --evidence-root --baseline-manifest-path --candidate-id --freeze-input
        --base-release-candidate-report-input --stable-default-configuration-input
        --candidate-binary --candidate-binary-sha256 --candidate-bundle
        --candidate-image-inspect --candidate-model-dir --candidate-model-tree-sha256
        --candidate-args-file --candidate-env-file --candidate-port
        --candidate-scenario-contract --rollback-binary --rollback-binary-sha256
        --rollback-bundle --rollback-image-inspect --rollback-model-dir
        --rollback-model-tree-sha256 --rollback-args-file --rollback-env-file
        --rollback-port --rollback-generation-request --gpu-index
    )
    while (($# > 0)); do
        option=$1
        case $option in
            --evidence-root|--baseline-manifest-path|--candidate-id|--freeze-input|--base-release-candidate-report-input|--stable-default-configuration-input|--candidate-binary|--candidate-binary-sha256|--candidate-bundle|--candidate-image-inspect|--candidate-model-dir|--candidate-model-tree-sha256|--candidate-args-file|--candidate-env-file|--candidate-port|--candidate-scenario-contract|--rollback-binary|--rollback-binary-sha256|--rollback-bundle|--rollback-image-inspect|--rollback-model-dir|--rollback-model-tree-sha256|--rollback-args-file|--rollback-env-file|--rollback-port|--rollback-generation-request|--gpu-index|--max-gpu-memory-mib|--startup-timeout-seconds|--shutdown-timeout-seconds|--audit-wait-seconds)
                (($# >= 2)) || outer_die "$option requires a value"
                [[ -n $2 && $2 != --* ]] || outer_die "$option requires a non-option value"
                [[ $seen != *"|$option|"* ]] || outer_die "$option may occur only once"
                seen+="$option|"
                shift 2
                ;;
            *) outer_die "unknown option: $option" ;;
        esac
    done
    for option in "${required[@]}"; do
        [[ $seen == *"|$option|"* ]] || outer_die "missing $option"
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

# The parent holds the no-follow host GPU lock throughout the child lifetime.
if [[ ${1:-} != --gpu-lock-supervised ]]; then
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

LOCK_PATH = "/var/tmp/riley-server-4096-gpu-evidence.lock"
LOCK_FD = 9
PR_SET_PDEATHSIG = 1
flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
try:
    opened_fd = os.open(LOCK_PATH, flags, 0o600)
except OSError as error:
    raise SystemExit(f"RC3 rollback supervisor: cannot open GPU lock safely: {error}")
try:
    if opened_fd != LOCK_FD:
        os.dup2(opened_fd, LOCK_FD, inheritable=False)
        os.close(opened_fd)
    lock_fd = LOCK_FD
    metadata = os.fstat(lock_fd)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600):
        raise SystemExit("RC3 rollback supervisor: unsafe shared GPU lock inode")
    named = os.stat(LOCK_PATH, follow_symlinks=False)
    if (named.st_dev, named.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("RC3 rollback supervisor: GPU lock path changed while opening")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("RC3 rollback supervisor: another GPU evidence capture holds the host lock")
    named = os.stat(LOCK_PATH, follow_symlinks=False)
    if (named.st_dev, named.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("RC3 rollback supervisor: GPU lock path changed while locking")

    supervisor_pid = os.getpid()
    supervisor_token = secrets.token_hex(32)
    forwarded = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, forwarded)
    child_pid = os.fork()
    if child_pid == 0:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
            os._exit(125)
        if os.getppid() != supervisor_pid:
            os._exit(125)
        try:
            inherited_fds = os.listdir("/proc/self/fd")
        except OSError:
            os._exit(125)
        for raw_fd in inherited_fds:
            try:
                fd = int(raw_fd)
            except ValueError:
                os._exit(125)
            if fd > 2 and fd != lock_fd:
                try:
                    os.close(fd)
                except OSError:
                    pass
        os.setsid()
        os.set_inheritable(lock_fd, True)
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        environment = {
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "TZ": "UTC",
            "HOME": "/nonexistent",
            "RILEY_RC3_ROLLBACK_SUPERVISOR_PID": str(supervisor_pid),
            "RILEY_RC3_ROLLBACK_SUPERVISOR_EXE": "/usr/bin/python3",
            "RILEY_RC3_ROLLBACK_SUPERVISOR_LOCK_FD": str(lock_fd),
            "RILEY_RC3_ROLLBACK_SUPERVISOR_LOCK_ID": f"{metadata.st_dev}:{metadata.st_ino}",
            "RILEY_RC3_ROLLBACK_SUPERVISOR_TOKEN": supervisor_token,
        }
        script = os.path.abspath(sys.argv[1])
        os.execve("/usr/bin/bash", ["/usr/bin/bash", script, "--gpu-lock-supervised", *sys.argv[2:]], environment)

    def forward_signal(_signum, _frame):
        try:
            os.kill(child_pid, _signum)
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

[[ ${RILEY_RC3_ROLLBACK_SUPERVISOR_PID:-} =~ ^[1-9][0-9]*$ ]] || outer_die 'supervisor PID was not authenticated'
[[ $PPID == "${RILEY_RC3_ROLLBACK_SUPERVISOR_PID}" ]] || outer_die 'supervisor is not the direct parent'
[[ ${RILEY_RC3_ROLLBACK_SUPERVISOR_EXE:-} == /usr/bin/python3 ]] || outer_die 'supervisor executable identity is invalid'
[[ /proc/$PPID/exe -ef /usr/bin/python3 ]] || outer_die 'supervisor executable differs from expected Python'
[[ ${RILEY_RC3_ROLLBACK_SUPERVISOR_LOCK_FD:-} == "$SUPERVISOR_LOCK_FD" ]] || outer_die 'supervisor lock descriptor is invalid'
[[ ${RILEY_RC3_ROLLBACK_SUPERVISOR_LOCK_ID:-} =~ ^[0-9]+:[0-9]+$ ]] || outer_die 'supervisor lock identity is invalid'
[[ ${RILEY_RC3_ROLLBACK_SUPERVISOR_TOKEN:-} =~ ^[0-9a-f]{64}$ ]] || outer_die 'supervisor token is invalid'
[[ /proc/$PPID/fd/$SUPERVISOR_LOCK_FD -ef "$GPU_LOCK_PATH" ]] || outer_die 'supervisor does not hold canonical GPU lock inode'
[[ /proc/$$/fd/$SUPERVISOR_LOCK_FD -ef "$GPU_LOCK_PATH" ]] || outer_die 'authenticated lock descriptor was not inherited'
if ! /usr/bin/grep -Eq "^lock:.*FLOCK[[:space:]]+ADVISORY[[:space:]]+WRITE[[:space:]]+$PPID([[:space:]]|$)" "/proc/$PPID/fdinfo/$SUPERVISOR_LOCK_FD"; then
    outer_die 'supervisor does not own kernel GPU flock'
fi
exec 9>&-
[[ ! -e /proc/$$/fd/$SUPERVISOR_LOCK_FD ]] || outer_die 'Bash retained supervisor lock descriptor'
unset RILEY_RC3_ROLLBACK_SUPERVISOR_PID RILEY_RC3_ROLLBACK_SUPERVISOR_EXE \
    RILEY_RC3_ROLLBACK_SUPERVISOR_LOCK_FD RILEY_RC3_ROLLBACK_SUPERVISOR_LOCK_ID \
    RILEY_RC3_ROLLBACK_SUPERVISOR_TOKEN BASH_ENV ENV CDPATH
export PATH=/usr/bin:/bin
export LC_ALL=C TZ=UTC
hash -r

die() {
    printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
    exit 2
}

for tool in bash basename curl dirname grep mktemp nvidia-smi python3 realpath sleep tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "required host tool is unavailable: $tool"
done

evidence_root=
baseline_manifest_path=
candidate_id=
freeze_input=
base_release_candidate_report_input=
stable_default_configuration_input=
candidate_binary=
candidate_binary_sha256=
candidate_bundle=
candidate_image_inspect=
candidate_model_dir=
candidate_model_tree_sha256=
candidate_args_file=
candidate_env_file=
candidate_port=
candidate_scenario_contract=
rollback_binary=
rollback_binary_sha256=
rollback_bundle=
rollback_image_inspect=
rollback_model_dir=
rollback_model_tree_sha256=
rollback_args_file=
rollback_env_file=
rollback_port=
rollback_generation_request=
gpu_index=
max_gpu_memory_mib=$DEFAULT_MAX_GPU_MEMORY_MIB
startup_timeout_seconds=$DEFAULT_STARTUP_TIMEOUT_SECONDS
shutdown_timeout_seconds=$DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
audit_wait_seconds=$DEFAULT_AUDIT_WAIT_SECONDS

need_value() {
    local flag=$1
    (($# >= 2)) || die "$flag requires a value"
    [[ -n $2 ]] || die "$flag must not be empty"
}

set_once() {
    local flag=$1 variable=$2 value=$3
    [[ $seen_options != *"|$flag|"* ]] || die "$flag may occur only once"
    seen_options+="$flag|"
    printf -v "$variable" '%s' "$value"
}

seen_options='|'
while (($# > 0)); do
    case $1 in
        --evidence-root|--baseline-manifest-path|--candidate-id|--freeze-input|--base-release-candidate-report-input|--stable-default-configuration-input|--candidate-binary|--candidate-binary-sha256|--candidate-bundle|--candidate-image-inspect|--candidate-model-dir|--candidate-model-tree-sha256|--candidate-args-file|--candidate-env-file|--candidate-port|--candidate-scenario-contract|--rollback-binary|--rollback-binary-sha256|--rollback-bundle|--rollback-image-inspect|--rollback-model-dir|--rollback-model-tree-sha256|--rollback-args-file|--rollback-env-file|--rollback-port|--rollback-generation-request|--gpu-index|--max-gpu-memory-mib|--startup-timeout-seconds|--shutdown-timeout-seconds|--audit-wait-seconds)
            need_value "$1" "${2:-}"
            case $1 in
                --evidence-root) set_once "$1" evidence_root "$2" ;;
                --baseline-manifest-path) set_once "$1" baseline_manifest_path "$2" ;;
                --candidate-id) set_once "$1" candidate_id "$2" ;;
                --freeze-input) set_once "$1" freeze_input "$2" ;;
                --base-release-candidate-report-input) set_once "$1" base_release_candidate_report_input "$2" ;;
                --stable-default-configuration-input) set_once "$1" stable_default_configuration_input "$2" ;;
                --candidate-binary) set_once "$1" candidate_binary "$2" ;;
                --candidate-binary-sha256) set_once "$1" candidate_binary_sha256 "$2" ;;
                --candidate-bundle) set_once "$1" candidate_bundle "$2" ;;
                --candidate-image-inspect) set_once "$1" candidate_image_inspect "$2" ;;
                --candidate-model-dir) set_once "$1" candidate_model_dir "$2" ;;
                --candidate-model-tree-sha256) set_once "$1" candidate_model_tree_sha256 "$2" ;;
                --candidate-args-file) set_once "$1" candidate_args_file "$2" ;;
                --candidate-env-file) set_once "$1" candidate_env_file "$2" ;;
                --candidate-port) set_once "$1" candidate_port "$2" ;;
                --candidate-scenario-contract) set_once "$1" candidate_scenario_contract "$2" ;;
                --rollback-binary) set_once "$1" rollback_binary "$2" ;;
                --rollback-binary-sha256) set_once "$1" rollback_binary_sha256 "$2" ;;
                --rollback-bundle) set_once "$1" rollback_bundle "$2" ;;
                --rollback-image-inspect) set_once "$1" rollback_image_inspect "$2" ;;
                --rollback-model-dir) set_once "$1" rollback_model_dir "$2" ;;
                --rollback-model-tree-sha256) set_once "$1" rollback_model_tree_sha256 "$2" ;;
                --rollback-args-file) set_once "$1" rollback_args_file "$2" ;;
                --rollback-env-file) set_once "$1" rollback_env_file "$2" ;;
                --rollback-port) set_once "$1" rollback_port "$2" ;;
                --rollback-generation-request) set_once "$1" rollback_generation_request "$2" ;;
                --gpu-index) set_once "$1" gpu_index "$2" ;;
                --max-gpu-memory-mib) set_once "$1" max_gpu_memory_mib "$2" ;;
                --startup-timeout-seconds) set_once "$1" startup_timeout_seconds "$2" ;;
                --shutdown-timeout-seconds) set_once "$1" shutdown_timeout_seconds "$2" ;;
                --audit-wait-seconds) set_once "$1" audit_wait_seconds "$2" ;;
            esac
            shift 2 ;;
        *) die "unknown option: $1" ;;
    esac
done

readonly -a required_variables=(
    evidence_root baseline_manifest_path candidate_id freeze_input
    base_release_candidate_report_input stable_default_configuration_input
    candidate_binary candidate_binary_sha256 candidate_bundle candidate_image_inspect
    candidate_model_dir candidate_model_tree_sha256 candidate_args_file candidate_env_file
    candidate_port candidate_scenario_contract rollback_binary rollback_binary_sha256
    rollback_bundle rollback_image_inspect rollback_model_dir rollback_model_tree_sha256
    rollback_args_file rollback_env_file rollback_port rollback_generation_request gpu_index
)
for required in "${required_variables[@]}"; do
    [[ -n ${!required} ]] || die "missing --${required//_/-}"
done

readonly sha_re='^[0-9a-f]{64}$'
for sha_variable in candidate_binary_sha256 candidate_model_tree_sha256 rollback_binary_sha256 rollback_model_tree_sha256; do
    [[ ${!sha_variable} =~ $sha_re && ${!sha_variable} != "$ZERO_SHA256" ]] || die "--${sha_variable//_/-} must be a non-zero lowercase SHA-256"
done
[[ $candidate_id =~ ^riley-(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)-rc([1-9][0-9]*)$ ]] || die '--candidate-id must be canonical riley-X.Y.Z-rcN'
[[ $gpu_index =~ ^[0-9]+$ ]] || die '--gpu-index must be a non-negative ordinal'
for port_variable in candidate_port rollback_port; do
    [[ ${!port_variable} =~ ^[0-9]+$ ]] && (( ${!port_variable} >= 1024 && ${!port_variable} <= 65535 )) || die "--${port_variable//_/-} must be from 1024 through 65535"
done
[[ $candidate_port != "$rollback_port" ]] || die '--candidate-port and --rollback-port must differ'
for numeric in max_gpu_memory_mib startup_timeout_seconds shutdown_timeout_seconds audit_wait_seconds; do
    [[ ${!numeric} =~ ^[1-9][0-9]*$ ]] || die "--${numeric//_/-} must be a positive integer"
done

require_absolute() {
    local label=$1 path=$2
    [[ $path == /* && $path != *$'\n'* && $path != *$'\r'* && $path != *\\* ]] || die "$label must be an absolute single-line path"
}

require_regular_file() {
    local label=$1 path=$2
    require_absolute "$label" "$path"
    [[ -f $path && ! -L $path ]] || die "$label must be a regular non-symlink file"
}

require_absolute '--evidence-root' "$evidence_root"
require_absolute '--candidate-binary' "$candidate_binary"
require_absolute '--candidate-model-dir' "$candidate_model_dir"
require_absolute '--rollback-binary' "$rollback_binary"
require_absolute '--rollback-model-dir' "$rollback_model_dir"
for pair in \
    '--freeze-input:'"$freeze_input" \
    '--base-release-candidate-report-input:'"$base_release_candidate_report_input" \
    '--stable-default-configuration-input:'"$stable_default_configuration_input" \
    '--candidate-bundle:'"$candidate_bundle" \
    '--candidate-image-inspect:'"$candidate_image_inspect" \
    '--candidate-args-file:'"$candidate_args_file" \
    '--candidate-env-file:'"$candidate_env_file" \
    '--candidate-scenario-contract:'"$candidate_scenario_contract" \
    '--rollback-bundle:'"$rollback_bundle" \
    '--rollback-image-inspect:'"$rollback_image_inspect" \
    '--rollback-args-file:'"$rollback_args_file" \
    '--rollback-env-file:'"$rollback_env_file" \
    '--rollback-generation-request:'"$rollback_generation_request"; do
    label=${pair%%:*}
    path=${pair#*:}
    require_regular_file "$label" "$path"
done
case $baseline_manifest_path in
    ''|/*|*//*|.|..|./*|../*|*/.|*/./*|*/..|*/../*)
        die '--baseline-manifest-path must be a normalized root-relative path' ;;
esac

candidate_args_file=$(/usr/bin/realpath -e "$candidate_args_file") || die '--candidate-args-file cannot be resolved'
candidate_env_file=$(/usr/bin/realpath -e "$candidate_env_file") || die '--candidate-env-file cannot be resolved'
candidate_scenario_contract=$(/usr/bin/realpath -e "$candidate_scenario_contract") || die '--candidate-scenario-contract cannot be resolved'
rollback_args_file=$(/usr/bin/realpath -e "$rollback_args_file") || die '--rollback-args-file cannot be resolved'
rollback_env_file=$(/usr/bin/realpath -e "$rollback_env_file") || die '--rollback-env-file cannot be resolved'
rollback_generation_request=$(/usr/bin/realpath -e "$rollback_generation_request") || die '--rollback-generation-request cannot be resolved'

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd -P)
case $evidence_root in
    "$repo_root"|"$repo_root"/*) die '--evidence-root must be outside the source tree' ;;
esac
for external_path in "$freeze_input" "$base_release_candidate_report_input" "$stable_default_configuration_input" "$candidate_binary" "$candidate_bundle" "$candidate_image_inspect" "$candidate_model_dir" "$candidate_args_file" "$candidate_env_file" "$candidate_scenario_contract" "$rollback_binary" "$rollback_bundle" "$rollback_image_inspect" "$rollback_model_dir" "$rollback_args_file" "$rollback_env_file" "$rollback_generation_request"; do
    case $external_path in
        "$evidence_root"|"$evidence_root"/*) die 'host input must be outside --evidence-root' ;;
    esac
done

declare -a candidate_arguments=() rollback_arguments=() candidate_environment=() rollback_environment=()
readonly env_key_re='^[A-Z_][A-Z0-9_]*$'

load_candidate_arguments() {
    local line
    while IFS= read -r line || [[ -n $line ]]; do
        [[ -n $line && $line != *$'\r'* ]] || die '--candidate-args-file contains an empty or CR-terminated argument'
        case $line in
            serve|--help|-h|--version|--model|--model=*|--bind|--bind=*|--device|--device=*|--c02-*|--shutdown-on-stdin|--shutdown-on-stdin=*) die "--candidate-args-file attempts to override a runner-owned argument: $line" ;;
        esac
        candidate_arguments+=("$line")
    done <"$candidate_args_file"
}

load_rollback_arguments() {
    local line
    while IFS= read -r line || [[ -n $line ]]; do
        [[ -n $line && $line != *$'\r'* ]] || die '--rollback-args-file contains an empty or CR-terminated argument'
        case $line in
            serve|--help|-h|--version|--model|--model=*|--bind|--bind=*|--device|--device=*|--c02-*|--shutdown-on-stdin|--shutdown-on-stdin=*) die "--rollback-args-file attempts to override a runner-owned argument: $line" ;;
        esac
        rollback_arguments+=("$line")
    done <"$rollback_args_file"
}

load_environment_file() {
    local source=$1 label=$2 destination=$3
    local line key value environment_keys='|'
    local -a parsed=()
    while IFS= read -r line || [[ -n $line ]]; do
        [[ $line != *$'\r'* && $line == *=* ]] || die "$label contains an invalid KEY=VALUE entry"
        key=${line%%=*}
        value=${line#*=}
        [[ $key =~ $env_key_re ]] || die "$label has an invalid key: $key"
        [[ $environment_keys != *"|$key|"* ]] || die "$label repeats a key: $key"
        environment_keys+="$key|"
        case $key in
            PATH|HOME|SHELL|BASH_ENV|ENV|CDPATH|IFS|LD_PRELOAD|LD_AUDIT|LD_LIBRARY_PATH|PYTHON*|CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|RILEY_SHUTDOWN_METRICS_PATH|RILEY_C02_*|RILEY_FREEZE_SHA|RILEY_GATE_E_REPORT_SHA|RILEY_CONFIGURATION_SHA|RILEY_BASE_RELEASE_CANDIDATE_REPORT_SHA) die "$label contains a forbidden lifecycle control key: $key" ;;
            RUST_LOG|RUST_BACKTRACE|CUDA_MODULE_LOADING|CUDA_CACHE_DISABLE|CUDA_CACHE_PATH|CUDA_FORCE_PTX_JIT|CUBLAS_WORKSPACE_CONFIG|NCCL_DEBUG) ;;
            *) die "$label key is outside the frozen lifecycle allowlist: $key" ;;
        esac
        [[ $value != *$'\n'* && $value != *$'\r'* ]] || die "$label value is not single-line: $key"
        parsed+=("$key=$value")
    done <"$source"
    if [[ $destination == candidate ]]; then candidate_environment=("${parsed[@]}"); else rollback_environment=("${parsed[@]}"); fi
}

load_candidate_arguments
load_rollback_arguments
load_environment_file "$candidate_env_file" '--candidate-env-file' candidate
load_environment_file "$rollback_env_file" '--rollback-env-file' rollback

scratch_dir=$(/usr/bin/mktemp -d /tmp/riley-rc3-rollback.XXXXXX)
server_pid=
server_start_ticks=
server_phase=

run_python() {
    local program=$1
    shift
    [[ $program == "$repo_root"/ci/release/*.py && -f $program && ! -L $program ]] || die "isolated Python target is missing or unsafe: $program"
    /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 \
        /usr/bin/python3 -B -I -S -c '
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

run_process_guard() {
    run_python "$repo_root/ci/release/c02_lifecycle_process_guard_v1.py" "$@"
}

server_identity_state() {
    local pid=$1 expected_ticks=$2 observed_ticks status
    if observed_ticks=$(run_process_guard --pid "$pid" --read-start-ticks 2>/dev/null); then
        [[ $observed_ticks =~ ^[1-9][0-9]*$ ]] || return 2
        [[ $observed_ticks == "$expected_ticks" ]] && return 0
        return 3
    else
        status=$?
    fi
    ((status == 3)) && return 3
    return 2
}

signal_server_if_current() {
    run_process_guard --pid "$1" --expected-start-ticks "$2" --signal "$3"
}

stop_server_after_failure() {
    local pid=${server_pid:-} ticks=${server_start_ticks:-} attempt=0 state signal_status
    [[ -n $pid ]] || return 0
    if [[ ! $ticks =~ ^[1-9][0-9]*$ ]]; then
        printf '%s: refusing to signal %s server without a bound start tick\n' "$SCRIPT_NAME" "$server_phase" >&2
        server_pid=
        return 0
    fi
    if server_identity_state "$pid" "$ticks"; then state=0; else state=$?; fi
    case $state in
        0)
            if signal_server_if_current "$pid" "$ticks" TERM; then
                while ((attempt < shutdown_timeout_seconds)); do
                    if server_identity_state "$pid" "$ticks"; then state=0; else state=$?; fi
                    ((state == 0)) || break
                    /usr/bin/sleep 1
                    attempt=$((attempt + 1))
                done
                if ((state == 0)); then if server_identity_state "$pid" "$ticks"; then state=0; else state=$?; fi; fi
                if ((state == 0 && attempt >= shutdown_timeout_seconds)); then
                    if signal_server_if_current "$pid" "$ticks" KILL; then wait "$pid" 2>/dev/null || true; else printf '%s: refused unsafe forced shutdown of %s PID %s\n' "$SCRIPT_NAME" "$server_phase" "$pid" >&2; fi
                elif ((state == 3)); then
                    wait "$pid" 2>/dev/null || true
                else
                    printf '%s: could not authenticate %s PID %s during cleanup\n' "$SCRIPT_NAME" "$server_phase" "$pid" >&2
                fi
            else
                signal_status=$?
                if ((signal_status == 3)); then wait "$pid" 2>/dev/null || true; else printf '%s: refused unsafe cleanup signal for %s PID %s\n' "$SCRIPT_NAME" "$server_phase" "$pid" >&2; fi
            fi ;;
        3) wait "$pid" 2>/dev/null || true ;;
        *) printf '%s: could not authenticate %s PID %s during cleanup\n' "$SCRIPT_NAME" "$server_phase" "$pid" >&2 ;;
    esac
    server_pid=
    server_start_ticks=
    server_phase=
}

cleanup() {
    local status=$?
    trap - EXIT
    stop_server_after_failure
    case ${scratch_dir:-} in
        /tmp/riley-rc3-rollback.*)
            if ((status == 0)); then
                [[ -d $scratch_dir ]] && /bin/rm -rf -- "$scratch_dir" || true
            else
                printf '%s: retained diagnostic logs at %s\n' "$SCRIPT_NAME" "$scratch_dir" >&2
            fi
            ;;
        '') ;;
        *) printf '%s: retained unexpected scratch path %s\n' "$SCRIPT_NAME" "$scratch_dir" >&2 ;;
    esac
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

run_private_config_initializer() {
    local module_path="$repo_root/ci/release/materialize_rc3_rollback_candidate_config_v1.py"
    [[ -f $module_path && ! -L $module_path ]] || die 'private RC3 config initializer is missing or unsafe'
    /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 \
        /usr/bin/python3 -B -I -S -c '
import os
import stat
import sys
from pathlib import Path
module_path, evidence_root = sys.argv[1:]
metadata = os.lstat(module_path)
if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
    raise SystemExit("unsafe private RC3 config initializer")
directory = os.path.dirname(module_path)
if not os.path.isabs(module_path) or not directory:
    raise SystemExit("private RC3 config initializer must be absolute")
sys.path.insert(0, directory)
import materialize_rc3_rollback_candidate_config_v1 as materializer
try:
    materializer._initialize_candidate_config_directory(Path(evidence_root))
except (materializer.RollbackCandidateConfigMaterializationError, OSError) as error:
    print(f"RC3 rollback config initialization refused: {error}", file=sys.stderr)
    raise SystemExit(2)
' "$module_path" "$evidence_root"
}

run_private_config_materializer() {
    local module_path="$repo_root/ci/release/materialize_rc3_rollback_candidate_config_v1.py"
    [[ -f $module_path && ! -L $module_path ]] || die 'private RC3 config materializer is missing or unsafe'
    /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 \
        /usr/bin/python3 -B -I -S -c '
import os
import re
import stat
import sys
from pathlib import Path
module_path, evidence_root, candidate_id = sys.argv[1:]
metadata = os.lstat(module_path)
if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
    raise SystemExit("unsafe private RC3 config materializer")
directory = os.path.dirname(module_path)
if not os.path.isabs(module_path) or not directory:
    raise SystemExit("private RC3 config materializer must be absolute")
sys.path.insert(0, directory)
import materialize_rc3_rollback_candidate_config_v1 as materializer
try:
    replayed = materializer._materialize_candidate_config_bridge(Path(evidence_root), candidate_id=candidate_id, configuration_profile="stable-default")
except (materializer.RollbackCandidateConfigMaterializationError, OSError) as error:
    print(f"RC3 rollback config materialization refused: {error}", file=sys.stderr)
    raise SystemExit(2)
if re.fullmatch(r"[0-9a-f]{64}", replayed.configuration_sha256) is None:
    raise SystemExit("RC3 rollback config materializer returned an invalid configuration SHA")
print(replayed.configuration_sha256)
' "$module_path" "$evidence_root" "$candidate_id"
}

validate_rollback_generation_request() {
    local module_path="$repo_root/ci/release/capture_rc3_rollback_phase_v1.py"
    [[ -f $module_path && ! -L $module_path ]] || die 'rollback phase collector is missing or unsafe'
    /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 \
        /usr/bin/python3 -B -I -S -c '
import os
import stat
import sys
from pathlib import Path
module_path, request_path = sys.argv[1:]
metadata = os.lstat(module_path)
if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
    raise SystemExit("unsafe rollback phase collector")
directory = os.path.dirname(module_path)
if not os.path.isabs(module_path) or not directory:
    raise SystemExit("rollback phase collector must be absolute")
sys.path.insert(0, directory)
import capture_rc3_rollback_phase_v1 as phase
try:
    phase._read_generation_request(Path(request_path))
except (phase.RollbackPhaseCaptureError, OSError) as error:
    print(f"rollback generation request refused: {error}", file=sys.stderr)
    raise SystemExit(2)
' "$module_path" "$rollback_generation_request"
}

validate_candidate_scenario_contract() {
    local module_path="$repo_root/ci/release/capture_c02_raw_soak_scenarios_v1.py"
    [[ -f $module_path && ! -L $module_path ]] || die 'candidate scenario producer is missing or unsafe'
    /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 \
        /usr/bin/python3 -B -I -S -c '
import os
import stat
import sys
from pathlib import Path
module_path, contract_path, candidate_id = sys.argv[1:]
metadata = os.lstat(module_path)
if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
    raise SystemExit("unsafe candidate scenario producer")
directory = os.path.dirname(module_path)
if not os.path.isabs(module_path) or not directory:
    raise SystemExit("candidate scenario producer must be absolute")
sys.path.insert(0, directory)
import capture_c02_raw_soak_scenarios_v1 as scenarios
try:
    raw = scenarios._read_absolute_regular(
        Path(contract_path),
        "--candidate-scenario-contract",
        maximum=scenarios.MAX_CONTRACT_BYTES,
    )
    contract = scenarios.validate_contract(
        raw,
        candidate_id=candidate_id,
        configuration_profile="stable-default",
    )
except (scenarios.RawScenarioCaptureError, OSError) as error:
    print(f"candidate scenario contract refused: {error}", file=sys.stderr)
    raise SystemExit(2)
if contract.get("schema_version") != scenarios.CONTRACT_VERSION:
    raise SystemExit("candidate scenario contract must use the standard v1 source-capture grammar")
if not isinstance(contract.get("scenarios"), list) or len(contract["scenarios"]) != 1:
    raise SystemExit("candidate scenario contract must contain exactly one scenario")
' "$module_path" "$candidate_scenario_contract" "$candidate_id"
}

run_private_rollback_finalizer() {
    local module_path="$repo_root/ci/release/finalize_rc3_rollback_finalizer_receipt_v1.py"
    [[ -f $module_path && ! -L $module_path ]] || die 'private RC3 rollback finalizer is missing or unsafe'
    # This replaces the shell. After normal receipt publication, no report is
    # printed and no later shell operation can turn it into a failed result.
    exec /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 \
        /usr/bin/python3 -B -I -S -c '
import os
import stat
import sys
from pathlib import Path
(module_path, evidence_root, candidate_binary, candidate_bundle, candidate_image,
 rollback_binary, rollback_bundle, rollback_image) = sys.argv[1:]
metadata = os.lstat(module_path)
if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
    raise SystemExit("unsafe private RC3 rollback finalizer")
directory = os.path.dirname(module_path)
if not os.path.isabs(module_path) or not directory:
    raise SystemExit("private RC3 rollback finalizer must be absolute")
sys.path.insert(0, directory)
import finalize_rc3_rollback_finalizer_receipt_v1 as finalizer
import prepare_rc3_rollback_artifacts_v1 as prepare
try:
    request = prepare.PreparationRequest(
        evidence_root=Path(evidence_root),
        candidate_binary=Path(candidate_binary),
        candidate_bundle=Path(candidate_bundle),
        candidate_image_inspect=Path(candidate_image),
        rollback_binary=Path(rollback_binary),
        rollback_bundle=Path(rollback_bundle),
        rollback_image_inspect=Path(rollback_image),
    )
    finalizer._finalize_authenticated_rollback_raw_once(request)
except (finalizer.AuthenticatedRollbackFinalizationError, OSError) as error:
    print(f"authenticated RC3 rollback finalization refused: {error}", file=sys.stderr)
    raise SystemExit(2)
os._exit(0)
' "$module_path" "$evidence_root" "$candidate_binary" "$candidate_bundle" "$candidate_image_inspect" "$rollback_binary" "$rollback_bundle" "$rollback_image_inspect"
}

verify_candidate_launch_inputs() {
    run_python "$repo_root/ci/release/verify_c02_lifecycle_launch_inputs_v1.py" \
        --binary "$candidate_binary" --binary-sha256 "$candidate_binary_sha256" \
        --model-dir "$candidate_model_dir" --model-tree-sha256 "$candidate_model_tree_sha256" \
        --phase "$1"
}

verify_rollback_launch_inputs() {
    run_python "$repo_root/ci/release/verify_c02_lifecycle_launch_inputs_v1.py" \
        --binary "$rollback_binary" --binary-sha256 "$rollback_binary_sha256" \
        --model-dir "$rollback_model_dir" --model-tree-sha256 "$rollback_model_tree_sha256" \
        --phase "$1"
}

preflight_gpu() {
    local probe observed_uuid observed_memory
    probe=$(/usr/bin/nvidia-smi -i "$gpu_index" --query-gpu=uuid,memory.used --format=csv,noheader,nounits) || die 'nvidia-smi GPU preflight failed'
    [[ $(printf '%s\n' "$probe" | /usr/bin/wc -l | /usr/bin/tr -d '[:space:]') == 1 ]] || die 'nvidia-smi GPU preflight returned an ambiguous inventory'
    IFS=, read -r observed_uuid observed_memory <<<"$probe"
    observed_uuid=${observed_uuid//[[:space:]]/}
    observed_memory=${observed_memory//[[:space:]]/}
    [[ $observed_uuid =~ ^GPU-[0-9A-Fa-f-]+$ ]] || die 'nvidia-smi returned an invalid GPU UUID'
    [[ $observed_memory =~ ^[0-9]+$ ]] || die 'nvidia-smi returned non-numeric used memory'
    ((observed_memory <= max_gpu_memory_mib)) || die "GPU preflight failed: ${observed_memory}MiB is above ${max_gpu_memory_mib}MiB"
}

bind_started_server() {
    server_start_ticks=$(run_process_guard --pid "$server_pid" --read-start-ticks) || die "could not bind a start tick to launched $server_phase server"
    [[ $server_start_ticks =~ ^[1-9][0-9]*$ ]] || die "process guard returned invalid $server_phase start tick"
}

require_current_server() {
    local action=$1 identity_status
    if server_identity_state "$server_pid" "$server_start_ticks"; then
        return 0
    else
        identity_status=$?
    fi
    if ((identity_status == 3)); then
        wait "$server_pid" 2>/dev/null || true
        server_pid=
    fi
    die "$server_phase server exited, was reused, or could not be authenticated before $action"
}

wait_for_ready() {
    local attempt status identity_status port
    if [[ $server_phase == candidate ]]; then port=$candidate_port; else port=$rollback_port; fi
    for ((attempt = 0; attempt < startup_timeout_seconds; attempt++)); do
        if server_identity_state "$server_pid" "$server_start_ticks"; then :; else
            identity_status=$?
            if ((identity_status == 3)); then wait "$server_pid" 2>/dev/null || true; server_pid=; die "$server_phase server exited or PID was reused before readiness"; fi
            die "could not authenticate launched $server_phase server during readiness"
        fi
        if status=$(/usr/bin/curl --noproxy '*' --http1.1 --silent --show-error --connect-timeout 1 --max-time 2 --output /dev/null --write-out '%{http_code}' "http://127.0.0.1:$port/readyz" 2>/dev/null); then
            if [[ $status == 200 ]]; then
                if server_identity_state "$server_pid" "$server_start_ticks"; then return 0; else identity_status=$?; fi
                if ((identity_status == 3)); then wait "$server_pid" 2>/dev/null || true; server_pid=; die "$server_phase server exited or PID was reused while readiness was observed"; fi
                die "could not authenticate launched $server_phase server after readiness response"
            fi
        fi
        /usr/bin/sleep 1
    done
    die "$server_phase server did not become ready within ${startup_timeout_seconds}s"
}

shutdown_server_successfully() {
    local pid=$server_pid ticks=$server_start_ticks phase=$server_phase
    local attempt=0 identity_status signal_status server_status
    [[ -n $pid && $ticks =~ ^[1-9][0-9]*$ ]] || die "$phase server identity disappeared before graceful shutdown"
    if server_identity_state "$pid" "$ticks"; then identity_status=0; else identity_status=$?; fi
    if ((identity_status != 0)); then
        if ((identity_status == 3)); then wait "$pid" 2>/dev/null || true; fi
        server_pid=
        die "$phase server exited, was reused, or could not be authenticated before graceful shutdown"
    fi
    if signal_server_if_current "$pid" "$ticks" TERM; then signal_status=0; else signal_status=$?; fi
    if ((signal_status != 0)); then
        if ((signal_status == 3)); then wait "$pid" 2>/dev/null || true; fi
        server_pid=
        die "could not safely deliver SIGTERM to bridged $phase server"
    fi
    while ((attempt < shutdown_timeout_seconds)); do
        if server_identity_state "$pid" "$ticks"; then identity_status=0; else identity_status=$?; fi
        ((identity_status == 0)) || break
        /usr/bin/sleep 1
        attempt=$((attempt + 1))
    done
    if ((identity_status == 0)); then if server_identity_state "$pid" "$ticks"; then identity_status=0; else identity_status=$?; fi; fi
    if ((identity_status == 0)); then
        if signal_server_if_current "$pid" "$ticks" KILL; then wait "$pid" 2>/dev/null || true; fi
        server_pid=
        die "$phase server did not exit within ${shutdown_timeout_seconds}s; no terminal rollback receipt may be emitted"
    fi
    if ((identity_status != 3)); then
        server_pid=
        die "could not authenticate $phase server while waiting for graceful shutdown"
    fi
    if wait "$pid"; then server_status=0; else server_status=$?; fi
    server_pid=
    server_start_ticks=
    server_phase=
    ((server_status == 0)) || die "$phase server exited with status ${server_status}; no terminal rollback receipt may be emitted"
}

launch_candidate_server() {
    local log_path="$scratch_dir/candidate-server.log"
    [[ ! -e $log_path && ! -L $log_path ]] || die 'candidate server log path unexpectedly exists'
    server_phase=candidate
    (
        exec /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C TZ=UTC \
            "${candidate_environment[@]}" \
            "$candidate_binary" serve --model "$candidate_model_dir" --bind "127.0.0.1:$candidate_port" --device "$gpu_index" \
            --c02-candidate-id "$candidate_id" --c02-configuration-profile "$STABLE_DEFAULT_PROFILE" \
            --c02-startup-artifact "$evidence_root/config/startup.json" \
            --c02-audit-dir "$evidence_root/$SOURCE_AUDIT_DIRECTORY" \
            --c02-shutdown-artifact "$evidence_root/$SOURCE_AUDIT_DIRECTORY/shutdown.json" \
            "${candidate_arguments[@]}"
    ) </dev/null >"$log_path" 2>&1 &
    server_pid=$!
    bind_started_server
}

launch_rollback_server() {
    local log_path="$scratch_dir/rollback-server.log"
    [[ ! -e $log_path && ! -L $log_path ]] || die 'rollback server log path unexpectedly exists'
    server_phase=rollback
    (
        exec /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C TZ=UTC \
            "${rollback_environment[@]}" \
            "$rollback_binary" serve --model "$rollback_model_dir" --bind "127.0.0.1:$rollback_port" --device "$gpu_index" \
            "${rollback_arguments[@]}"
    ) </dev/null >"$log_path" 2>&1 &
    server_pid=$!
    bind_started_server
}

remove_scratch_for_terminal() {
    [[ -z $server_pid && -z $server_start_ticks && -z $server_phase ]] || die 'refusing terminal finalization while service identity remains'
    [[ $scratch_dir == /tmp/riley-rc3-rollback.* && -d $scratch_dir ]] || die 'refusing terminal finalization with unexpected scratch directory'
    /bin/rm -rf -- "$scratch_dir"
    scratch_dir=
    trap - EXIT
}

# No evidence root, service, or GPU action occurs before these pure host-input
# replays and the canonical rollback request check.
verify_candidate_launch_inputs pre-launch >"$scratch_dir/candidate-launch-inputs-pre.json"
verify_rollback_launch_inputs pre-launch >"$scratch_dir/rollback-launch-inputs-pre.json"
validate_rollback_generation_request
validate_candidate_scenario_contract

preflight_gpu

run_python "$repo_root/ci/release/prepare_rc3_rollback_evidence_v1.py" \
    --evidence-root "$evidence_root" \
    --baseline-manifest-path "$baseline_manifest_path" \
    --candidate-id "$candidate_id" \
    --freeze-input "$freeze_input" \
    --base-release-candidate-report-input "$base_release_candidate_report_input" \
    --stable-default-configuration-input "$stable_default_configuration_input" \
    >"$scratch_dir/static-preparation.json"

run_private_config_initializer
[[ ! -e $evidence_root/$SOURCE_AUDIT_DIRECTORY && ! -L $evidence_root/$SOURCE_AUDIT_DIRECTORY ]] || die 'source-audit path unexpectedly exists before candidate launch'

launch_candidate_server
wait_for_ready
require_current_server 'configuration observation'

"$repo_root/ci/release/run_remote_c02_config_endpoint_observation_v1.sh" \
    --endpoint "http://127.0.0.1:$candidate_port/v1/config" \
    --server-pid "$server_pid" \
    --gpu-index "$gpu_index" \
    --evidence-root "$evidence_root" \
    --capture-name "$CONFIG_CAPTURE_NAME" \
    >"$scratch_dir/config-bridge.json"

configuration_sha256=$(run_private_config_materializer)
[[ $configuration_sha256 =~ $sha_re && $configuration_sha256 != "$ZERO_SHA256" ]] || die 'private config materializer returned invalid configuration SHA'
require_current_server 'candidate phase capture'

run_python "$repo_root/ci/release/capture_rc3_rollback_phase_v1.py" \
    --endpoint "http://127.0.0.1:$candidate_port" \
    --server-pid "$server_pid" \
    --gpu-index "$gpu_index" \
    --evidence-root "$evidence_root" \
    --capture-name "$CANDIDATE_PHASE_CAPTURE_NAME" \
    >"$scratch_dir/candidate-phase.json"

require_current_server 'candidate serial capture'
"$repo_root/ci/release/run_remote_c02_raw_soak_scenarios_v1.sh" \
    --endpoint "http://127.0.0.1:$candidate_port/v1/completions" \
    --server-pid "$server_pid" \
    --candidate-id "$candidate_id" \
    --configuration-profile "$STABLE_DEFAULT_PROFILE" \
    --configuration-sha256 "$configuration_sha256" \
    --evidence-root "$evidence_root" \
    --capture-name "$SERIAL_CAPTURE_NAME" \
    --audit-dir-name "$SOURCE_AUDIT_DIRECTORY" \
    --scenario-contract "$candidate_scenario_contract" \
    --repository-root "$repo_root" \
    --audit-wait-seconds "$audit_wait_seconds" \
    >"$scratch_dir/serial-capture.json"

shutdown_server_successfully
verify_candidate_launch_inputs post-exit >"$scratch_dir/candidate-launch-inputs-post.json"

launch_rollback_server
wait_for_ready
require_current_server 'rollback phase capture'

run_python "$repo_root/ci/release/capture_rc3_rollback_phase_v1.py" \
    --endpoint "http://127.0.0.1:$rollback_port" \
    --server-pid "$server_pid" \
    --gpu-index "$gpu_index" \
    --evidence-root "$evidence_root" \
    --capture-name "$ROLLBACK_PHASE_CAPTURE_NAME" \
    --generation-request "$rollback_generation_request" \
    >"$scratch_dir/rollback-phase.json"

shutdown_server_successfully
verify_rollback_launch_inputs post-exit >"$scratch_dir/rollback-launch-inputs-post.json"

# The private finalizer is the terminal process. It owns the fixed
# preparation-to-atomic-to-v3/v4-to-receipt normal-return chain.
remove_scratch_for_terminal
run_private_rollback_finalizer
