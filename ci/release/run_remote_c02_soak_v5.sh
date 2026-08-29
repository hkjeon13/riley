#!/usr/bin/env bash
# Run one narrow, host-binary C02 lifecycle capture under an authenticated GPU
# supervisor.  This is raw provenance only: it cannot create a candidate
# freeze, invoke a semantic checker, or declare a candidate qualified.

set -euo pipefail
set -o noclobber
umask 077
IFS=$' \t\n'

readonly SCRIPT_NAME='run_remote_c02_soak_v5.sh'
readonly GPU_LOCK_PATH='/var/tmp/riley-server-4096-gpu-evidence.lock'
readonly SUPERVISOR_LOCK_FD=9
readonly DEFAULT_MAX_GPU_MEMORY_MIB=256
readonly DEFAULT_STARTUP_TIMEOUT_SECONDS=60
readonly DEFAULT_SHUTDOWN_TIMEOUT_SECONDS=30
readonly DEFAULT_AUDIT_WAIT_SECONDS=15
readonly CONFIG_CAPTURE_NAME='config-bridge'
readonly SOURCE_AUDIT_DIRECTORY='source-audit'
readonly FALLBACK_CAPTURE_NAME='fallback-capture'
readonly FALLBACK_OBSERVATION_NAME='fallback-observation'
readonly FROZEN_CONTRACT_NAME='fallback-lifecycle-scenario-contract.json'
readonly ZERO_SHA256='0000000000000000000000000000000000000000000000000000000000000000'

usage() {
    /bin/cat <<'EOF'
usage: bash ci/release/run_remote_c02_soak_v5.sh \
  --binary ABSOLUTE_RELEASE_BINARY \
  --binary-sha256 LOWERCASE_SHA256 \
  --model-dir ABSOLUTE_MODEL_DIRECTORY \
  --model-tree-sha256 LOWERCASE_SHA256 \
  --candidate-id riley-X.Y.Z-rcN \
  --configuration-profile max-performance-exact \
  --gpu-index ORDINAL \
  --port LOOPBACK_PORT \
  --evidence-dir NEW_ABSOLUTE_EXTERNAL_DIRECTORY \
  --args-file ABSOLUTE_LINE_DELIMITED_ARGUMENT_FILE \
  --env-file ABSOLUTE_KEY_VALUE_ENVIRONMENT_FILE \
  --scenario-contract ABSOLUTE_CANONICAL_JSON_FILE \
  --freeze-sha256 LOWERCASE_SHA256 \
  --base-release-candidate-report-sha256 LOWERCASE_SHA256 \
  [--max-gpu-memory-mib 256] \
  [--startup-timeout-seconds 60] \
  [--shutdown-timeout-seconds 30] \
  [--audit-wait-seconds 15]

This native-fallback lifecycle version is host-binary-only. It freezes exactly
one canonical fallback-v2 scenario into a new private evidence root, forces
the server's gpu-greedy sampling arm, performs one immediate C02 metrics
observation, and publishes at most one raw v5 manifest. It never publishes a
lifecycle receipt. It uses only 127.0.0.1 and does not accept a server command,
caller-supplied configuration SHA, PID/start tick, listener tuple, or GPU UUID.

The runner owns `serve`, --model, --bind, --device, --sampling-backend,
all --c02-* options, and the SIGTERM shutdown trigger. Each args-file line is
one additional server argument. Each env-file line is one approved KEY=VALUE
entry. The server is launched through env -i. Evidence is retained on every
failure; no C02 qualification, candidate freeze, Gate E result, Docker, SSH,
system service, privileged action, or lifecycle receipt is performed.
EOF
}

outer_die() {
    printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
    exit 2
}

# Before obtaining the shared GPU lock, reject malformed invocations without
# looking at a filesystem, a GPU, or the evidence root.  The authenticated
# child performs the full semantic/path validation again after handover.
preflight_invocation() {
    local option
    local seen='|'
    local -a required=(
        --binary --binary-sha256 --model-dir --model-tree-sha256 --candidate-id
        --configuration-profile --gpu-index --port --evidence-dir --args-file
        --env-file --scenario-contract --freeze-sha256
        --base-release-candidate-report-sha256
    )
    while (($# > 0)); do
        option=$1
        case $option in
            --binary|--binary-sha256|--model-dir|--model-tree-sha256|--candidate-id|--configuration-profile|--gpu-index|--port|--evidence-dir|--args-file|--env-file|--scenario-contract|--freeze-sha256|--base-release-candidate-report-sha256|--max-gpu-memory-mib|--startup-timeout-seconds|--shutdown-timeout-seconds|--audit-wait-seconds)
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

# The first caller may have entered through an ambient interactive shell.  It
# gets no opportunity to bypass the clean environment or replace the GPU lock:
# a small Python parent opens and holds the authenticated no-follow lock for
# the full lifetime of the Bash child.  A user-supplied internal sentinel has
# no matching parent/FD and is rejected below.
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
import subprocess
import sys

LOCK_PATH = "/var/tmp/riley-server-4096-gpu-evidence.lock"
LOCK_FD = 9
PR_SET_PDEATHSIG = 1

flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
try:
    opened_fd = os.open(LOCK_PATH, flags, 0o600)
except OSError as error:
    raise SystemExit(f"C02 lifecycle supervisor: cannot open GPU lock safely: {error}")
try:
    if opened_fd != LOCK_FD:
        os.dup2(opened_fd, LOCK_FD, inheritable=False)
        os.close(opened_fd)
    lock_fd = LOCK_FD
    metadata = os.fstat(lock_fd)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600):
        raise SystemExit("C02 lifecycle supervisor: unsafe shared GPU lock inode")
    named = os.stat(LOCK_PATH, follow_symlinks=False)
    if (named.st_dev, named.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("C02 lifecycle supervisor: GPU lock path changed while opening")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("C02 lifecycle supervisor: another GPU evidence capture holds the host lock")
    named = os.stat(LOCK_PATH, follow_symlinks=False)
    if (named.st_dev, named.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("C02 lifecycle supervisor: GPU lock path changed while locking")

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
        # env -i does not close caller-inherited descriptors. Keep only
        # stdin/stdout/stderr and the one authenticated lock descriptor so
        # neither the runner nor the release binary receives ambient files,
        # sockets, or agent-control pipes.
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
            "RILEY_C02_LIFECYCLE_SUPERVISOR_PID": str(supervisor_pid),
            "RILEY_C02_LIFECYCLE_SUPERVISOR_EXE": "/usr/bin/python3",
            "RILEY_C02_LIFECYCLE_SUPERVISOR_LOCK_FD": str(lock_fd),
            "RILEY_C02_LIFECYCLE_SUPERVISOR_LOCK_ID": f"{metadata.st_dev}:{metadata.st_ino}",
            "RILEY_C02_LIFECYCLE_SUPERVISOR_TOKEN": supervisor_token,
        }
        script = os.path.abspath(sys.argv[1])
        os.execve(
            "/usr/bin/bash",
            ["/usr/bin/bash", script, "--gpu-lock-supervised", *sys.argv[2:]],
            environment,
        )

    def forward_signal(_signum, _frame):
        try:
            # The child calls setsid() before exec, but a signal can race that
            # transition.  Signalling its PID is safe in both windows; Bash
            # owns cleanup of its service child through its EXIT trap.
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

[[ ${RILEY_C02_LIFECYCLE_SUPERVISOR_PID:-} =~ ^[1-9][0-9]*$ ]] || outer_die 'supervisor PID was not authenticated'
[[ ${PPID} == "${RILEY_C02_LIFECYCLE_SUPERVISOR_PID}" ]] || outer_die 'supervisor is not the direct parent'
[[ ${RILEY_C02_LIFECYCLE_SUPERVISOR_EXE:-} == /usr/bin/python3 ]] || outer_die 'supervisor executable identity is invalid'
[[ /proc/${PPID}/exe -ef /usr/bin/python3 ]] || outer_die 'supervisor executable differs from the expected Python'
[[ ${RILEY_C02_LIFECYCLE_SUPERVISOR_LOCK_FD:-} == "$SUPERVISOR_LOCK_FD" ]] || outer_die 'supervisor lock descriptor is invalid'
[[ ${RILEY_C02_LIFECYCLE_SUPERVISOR_LOCK_ID:-} =~ ^[0-9]+:[0-9]+$ ]] || outer_die 'supervisor lock identity is invalid'
[[ ${RILEY_C02_LIFECYCLE_SUPERVISOR_TOKEN:-} =~ ^[0-9a-f]{64}$ ]] || outer_die 'supervisor token is invalid'
[[ /proc/${PPID}/fd/${SUPERVISOR_LOCK_FD} -ef "$GPU_LOCK_PATH" ]] || outer_die 'supervisor does not hold the canonical GPU lock inode'
[[ /proc/$$/fd/${SUPERVISOR_LOCK_FD} -ef "$GPU_LOCK_PATH" ]] || outer_die 'authenticated lock descriptor was not inherited'
if ! /usr/bin/grep -Eq "^lock:.*FLOCK[[:space:]]+ADVISORY[[:space:]]+WRITE[[:space:]]+${PPID}([[:space:]]|$)" "/proc/${PPID}/fdinfo/${SUPERVISOR_LOCK_FD}"; then
    outer_die 'supervisor does not own the kernel GPU flock'
fi
exec 9>&-
[[ ! -e /proc/$$/fd/${SUPERVISOR_LOCK_FD} ]] || outer_die 'Bash retained the supervisor lock descriptor'
unset \
    RILEY_C02_LIFECYCLE_SUPERVISOR_PID \
    RILEY_C02_LIFECYCLE_SUPERVISOR_EXE \
    RILEY_C02_LIFECYCLE_SUPERVISOR_LOCK_FD \
    RILEY_C02_LIFECYCLE_SUPERVISOR_LOCK_ID \
    RILEY_C02_LIFECYCLE_SUPERVISOR_TOKEN \
    BASH_ENV ENV CDPATH
export PATH=/usr/bin:/bin
export LC_ALL=C TZ=UTC
hash -r

die() {
    printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
    exit 2
}

for tool in bash basename curl dirname grep mktemp nvidia-smi python3 realpath sha256sum sleep tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "required host tool is unavailable: $tool"
done

binary=
binary_sha256=
model_dir=
model_tree_sha256=
candidate_id=
configuration_profile=
gpu_index=
port=
evidence_dir=
args_file=
env_file=
scenario_contract=
freeze_sha256=
base_release_candidate_report_sha256=
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
    local flag=$1
    local variable=$2
    local value=$3
    [[ $seen_options != *"|${flag}|"* ]] || die "$flag may occur only once"
    seen_options+="${flag}|"
    printf -v "$variable" '%s' "$value"
}

seen_options='|'
while (($# > 0)); do
    case $1 in
        --binary|--binary-sha256|--model-dir|--model-tree-sha256|--candidate-id|--configuration-profile|--gpu-index|--port|--evidence-dir|--args-file|--env-file|--scenario-contract|--freeze-sha256|--base-release-candidate-report-sha256|--max-gpu-memory-mib|--startup-timeout-seconds|--shutdown-timeout-seconds|--audit-wait-seconds)
            need_value "$1" "${2:-}"
            case $1 in
                --binary) set_once "$1" binary "$2" ;;
                --binary-sha256) set_once "$1" binary_sha256 "$2" ;;
                --model-dir) set_once "$1" model_dir "$2" ;;
                --model-tree-sha256) set_once "$1" model_tree_sha256 "$2" ;;
                --candidate-id) set_once "$1" candidate_id "$2" ;;
                --configuration-profile) set_once "$1" configuration_profile "$2" ;;
                --gpu-index) set_once "$1" gpu_index "$2" ;;
                --port) set_once "$1" port "$2" ;;
                --evidence-dir) set_once "$1" evidence_dir "$2" ;;
                --args-file) set_once "$1" args_file "$2" ;;
                --env-file) set_once "$1" env_file "$2" ;;
                --scenario-contract) set_once "$1" scenario_contract "$2" ;;
                --freeze-sha256) set_once "$1" freeze_sha256 "$2" ;;
                --base-release-candidate-report-sha256) set_once "$1" base_release_candidate_report_sha256 "$2" ;;
                --max-gpu-memory-mib) set_once "$1" max_gpu_memory_mib "$2" ;;
                --startup-timeout-seconds) set_once "$1" startup_timeout_seconds "$2" ;;
                --shutdown-timeout-seconds) set_once "$1" shutdown_timeout_seconds "$2" ;;
                --audit-wait-seconds) set_once "$1" audit_wait_seconds "$2" ;;
            esac
            shift 2
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

for required in binary binary_sha256 model_dir model_tree_sha256 candidate_id configuration_profile gpu_index port evidence_dir args_file env_file scenario_contract freeze_sha256 base_release_candidate_report_sha256; do
    [[ -n ${!required} ]] || die "missing --${required//_/-}"
done

readonly sha_re='^[0-9a-f]{64}$'
[[ $binary_sha256 =~ $sha_re && $binary_sha256 != "$ZERO_SHA256" ]] || die '--binary-sha256 must be a non-zero lowercase SHA-256'
[[ $model_tree_sha256 =~ $sha_re && $model_tree_sha256 != "$ZERO_SHA256" ]] || die '--model-tree-sha256 must be a non-zero lowercase SHA-256'
[[ $freeze_sha256 =~ $sha_re && $freeze_sha256 != "$ZERO_SHA256" ]] || die '--freeze-sha256 must be a non-zero lowercase SHA-256'
[[ $base_release_candidate_report_sha256 =~ $sha_re && $base_release_candidate_report_sha256 != "$ZERO_SHA256" ]] || die '--base-release-candidate-report-sha256 must be a non-zero lowercase SHA-256'
[[ $candidate_id =~ ^riley-(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)-rc([1-9][0-9]*)$ ]] || die '--candidate-id must be canonical riley-X.Y.Z-rcN'
[[ $configuration_profile == max-performance-exact ]] || die '--configuration-profile must be max-performance-exact'
[[ $gpu_index =~ ^[0-9]+$ ]] || die '--gpu-index must be a non-negative ordinal'
[[ $port =~ ^[0-9]+$ ]] && ((port >= 1024 && port <= 65535)) || die '--port must be from 1024 through 65535'
for numeric in max_gpu_memory_mib startup_timeout_seconds shutdown_timeout_seconds audit_wait_seconds; do
    [[ ${!numeric} =~ ^[1-9][0-9]*$ ]] || die "--${numeric//_/-} must be a positive integer"
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

require_absolute '--binary' "$binary"
require_absolute '--model-dir' "$model_dir"
require_absolute '--evidence-dir' "$evidence_dir"
require_regular_file '--args-file' "$args_file"
require_regular_file '--env-file' "$env_file"
require_regular_file '--scenario-contract' "$scenario_contract"
args_file=$(/usr/bin/realpath -e "$args_file") || die '--args-file cannot be resolved'
env_file=$(/usr/bin/realpath -e "$env_file") || die '--env-file cannot be resolved'
scenario_contract=$(/usr/bin/realpath -e "$scenario_contract") || die '--scenario-contract cannot be resolved'

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(CDPATH= cd -- "${script_dir}/../.." && pwd -P)
case $evidence_dir in
    "$repo_root"|"$repo_root"/*) die '--evidence-dir must be outside the source tree' ;;
esac

declare -a loaded_arguments=()
declare -a loaded_environment=()
readonly env_key_re='^[A-Z_][A-Z0-9_]*$'

load_arguments() {
    local line
    while IFS= read -r line || [[ -n $line ]]; do
        [[ -n $line && $line != *$'\r'* ]] || die '--args-file contains an empty or CR-terminated argument'
        case $line in
            serve|--help|-h|--version|--model|--model=*|--bind|--bind=*|--device|--device=*|--sampling-backend|--sampling-backend=*|--c02-candidate-id|--c02-candidate-id=*|--c02-configuration-profile|--c02-configuration-profile=*|--c02-startup-artifact|--c02-startup-artifact=*|--c02-audit-dir|--c02-audit-dir=*|--c02-shutdown-artifact|--c02-shutdown-artifact=*|--shutdown-on-stdin|--shutdown-on-stdin=*|--c02-*)
                die "--args-file attempts to override a runner-owned argument: $line"
                ;;
        esac
        loaded_arguments+=("$line")
    done <"$args_file"
}

load_environment() {
    local line key value
    local environment_keys='|'
    while IFS= read -r line || [[ -n $line ]]; do
        [[ $line != *$'\r'* && $line == *=* ]] || die '--env-file contains an invalid KEY=VALUE entry'
        key=${line%%=*}
        value=${line#*=}
        [[ $key =~ $env_key_re ]] || die "--env-file has an invalid key: $key"
        [[ $environment_keys != *"|${key}|"* ]] || die "--env-file repeats a key: $key"
        environment_keys+="${key}|"
        case $key in
            PATH|HOME|SHELL|BASH_ENV|ENV|CDPATH|IFS|LD_PRELOAD|LD_AUDIT|LD_LIBRARY_PATH|PYTHON*|CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|RILEY_SHUTDOWN_METRICS_PATH|RILEY_C02_*|RILEY_FREEZE_SHA|RILEY_GATE_E_REPORT_SHA|RILEY_CONFIGURATION_SHA|RILEY_BASE_RELEASE_CANDIDATE_REPORT_SHA)
                die "--env-file contains a forbidden lifecycle control key: $key"
                ;;
            RUST_LOG|RUST_BACKTRACE|CUDA_MODULE_LOADING|CUDA_CACHE_DISABLE|CUDA_CACHE_PATH|CUDA_FORCE_PTX_JIT|CUBLAS_WORKSPACE_CONFIG|NCCL_DEBUG)
                ;;
            *)
                die "--env-file key is outside the frozen lifecycle allowlist: $key"
                ;;
        esac
        [[ $value != *$'\n'* && $value != *$'\r'* ]] || die "--env-file value is not single-line: $key"
        loaded_environment+=("$key=$value")
    done <"$env_file"
}

load_arguments
load_environment

scratch_dir=$(/usr/bin/mktemp -d /tmp/riley-c02-lifecycle.XXXXXX)
server_pid=
server_start_ticks=

stop_server_after_failure() {
    local pid=${server_pid:-}
    local ticks=${server_start_ticks:-}
    local attempt=0
    local state signal_status
    [[ -n $pid ]] || return 0
    if [[ ! $ticks =~ ^[1-9][0-9]*$ ]]; then
        printf '%s: refusing to signal a server without a bound start tick\n' "$SCRIPT_NAME" >&2
        server_pid=
        return 0
    fi
    if server_identity_state "$pid" "$ticks"; then
        state=0
    else
        state=$?
    fi
    case $state in
        0)
            if signal_server_if_current "$pid" "$ticks" TERM; then
                while ((attempt < shutdown_timeout_seconds)); do
                    if server_identity_state "$pid" "$ticks"; then
                        state=0
                    else
                        state=$?
                    fi
                    ((state == 0)) || break
                    /usr/bin/sleep 1
                    attempt=$((attempt + 1))
                done
                # A child can become a zombie during the final sleep above.
                # Re-read through the pidfd guard before choosing KILL so a
                # stale successful poll can never prompt a signal.
                if ((state == 0)); then
                    if server_identity_state "$pid" "$ticks"; then
                        state=0
                    else
                        state=$?
                    fi
                fi
                if ((state == 0 && attempt >= shutdown_timeout_seconds)); then
                    if signal_server_if_current "$pid" "$ticks" KILL; then
                        wait "$pid" 2>/dev/null || true
                    else
                        printf '%s: refused unsafe forced shutdown of server PID %s\n' "$SCRIPT_NAME" "$pid" >&2
                    fi
                elif ((state == 3)); then
                    wait "$pid" 2>/dev/null || true
                else
                    printf '%s: could no longer authenticate server PID %s during cleanup\n' "$SCRIPT_NAME" "$pid" >&2
                fi
            else
                signal_status=$?
                if ((signal_status == 3)); then
                    wait "$pid" 2>/dev/null || true
                else
                    printf '%s: refused unsafe cleanup signal for server PID %s\n' "$SCRIPT_NAME" "$pid" >&2
                fi
            fi
            ;;
        3)
            wait "$pid" 2>/dev/null || true
            ;;
        *)
            printf '%s: could not authenticate server PID %s during cleanup\n' "$SCRIPT_NAME" "$pid" >&2
            ;;
    esac
    server_pid=
}

cleanup() {
    local status=$?
    trap - EXIT
    stop_server_after_failure
    case ${scratch_dir:-} in
        /tmp/riley-c02-lifecycle.*) [[ -d $scratch_dir ]] && /bin/rm -rf -- "$scratch_dir" ;;
        '') ;;
        *) printf '%s: retained unexpected scratch path %s\n' "$SCRIPT_NAME" "$scratch_dir" >&2 ;;
    esac
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

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
# -I intentionally drops the script directory.  Restore only this
# runner-derived, checked directory rather than accepting PYTHONPATH or an
# ambient module search path, so sibling provenance modules remain importable.
sys.path.insert(0, directory)
sys.argv = [script, *sys.argv[2:]]
runpy.run_path(script, run_name="__main__")
' "$program" "$@"
}

run_private_v5_raw_finalizer() {
    local module_path="$repo_root/ci/release/finalize_c02_lifecycle_v5_raw.py"
    [[ -f $module_path && ! -L $module_path ]] || \
        die "private v5 raw finalizer is missing or unsafe: $module_path"
    /usr/bin/env -i \
        PATH=/usr/bin:/bin \
        LC_ALL=C \
        TZ=UTC \
        PYTHONDONTWRITEBYTECODE=1 \
        /usr/bin/python3 -B -I -S -c '
import os
import stat
import sys
from pathlib import Path

module_path, evidence_root, bridge_report, candidate, freeze, base_report = sys.argv[1:]
metadata = os.lstat(module_path)
if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
    raise SystemExit("unsafe private v5 raw finalizer")
directory = os.path.dirname(module_path)
if not os.path.isabs(module_path) or not directory:
    raise SystemExit("private v5 raw finalizer must be absolute")
sys.path.insert(0, directory)
import finalize_c02_lifecycle_v5_raw as finalizer
import provenance_v2_common as common

try:
    report = finalizer._finalize_authenticated_v5_raw_once(
        evidence_root=Path(evidence_root),
        bridge_report_path=Path(bridge_report),
        candidate_id=candidate,
        freeze_sha256=freeze,
        base_release_candidate_report_sha256=base_report,
    )
except (finalizer.C02LifecycleV5RawFinalizationError, OSError) as error:
    print(f"C02 lifecycle v5 raw finalization refused: {error}", file=sys.stderr)
    raise SystemExit(2)
sys.stdout.buffer.write(common.canonical_json_bytes(report) + b"\\n")
' "$module_path" "$evidence_dir" "$bridge_report_path" "$candidate_id" "$freeze_sha256" "$base_release_candidate_report_sha256"
}

run_process_guard() {
    run_python "$repo_root/ci/release/c02_lifecycle_process_guard_v1.py" "$@"
}

# Return 0 only for the original child, 3 if it is gone or the numeric PID
# was reused, and 2 for an unavailable/malformed identity channel. Callers
# must never signal on either nonzero result.
server_identity_state() {
    local pid=$1
    local expected_ticks=$2
    local observed_ticks status
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
    local pid=$1
    local expected_ticks=$2
    local signal_name=$3
    run_process_guard \
        --pid "$pid" \
        --expected-start-ticks "$expected_ticks" \
        --signal "$signal_name"
}

verify_launch_inputs() {
    local phase=$1
    run_python "$repo_root/ci/release/verify_c02_lifecycle_launch_inputs_v1.py" \
        --binary "$binary" \
        --binary-sha256 "$binary_sha256" \
        --model-dir "$model_dir" \
        --model-tree-sha256 "$model_tree_sha256" \
        --phase "$phase"
}

verify_launch_inputs pre-launch >"$scratch_dir/launch-inputs-pre.json"

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

preflight_gpu

run_python "$repo_root/ci/release/prepare_c02_lifecycle_evidence_v5.py" \
    --evidence-root "$evidence_dir" \
    --scenario-contract "$scenario_contract" \
    --candidate-id "$candidate_id" \
    --configuration-profile "$configuration_profile" >"$scratch_dir/evidence-preparation.json"

server_log="$evidence_dir/server.log"
startup_artifact="$evidence_dir/startup-artifact.json"
source_audit_dir="$evidence_dir/$SOURCE_AUDIT_DIRECTORY"
shutdown_artifact="$source_audit_dir/shutdown.json"
[[ ! -e $server_log && ! -L $server_log ]] || die 'server log path unexpectedly exists'

(
    exec /usr/bin/env -i \
        PATH=/usr/bin:/bin \
        LC_ALL=C \
        TZ=UTC \
        "${loaded_environment[@]}" \
        "$binary" serve \
        --model "$model_dir" \
        --bind "127.0.0.1:${port}" \
        --device "$gpu_index" \
        --sampling-backend gpu-greedy \
        --c02-candidate-id "$candidate_id" \
        --c02-configuration-profile "$configuration_profile" \
        --c02-startup-artifact "$startup_artifact" \
        --c02-audit-dir "$source_audit_dir" \
        --c02-shutdown-artifact "$shutdown_artifact" \
        "${loaded_arguments[@]}"
) </dev/null >"$server_log" 2>&1 &
server_pid=$!
server_start_ticks=$(run_process_guard --pid "$server_pid" --read-start-ticks) || \
    die 'could not bind a start tick to the launched server process'
[[ $server_start_ticks =~ ^[1-9][0-9]*$ ]] || \
    die 'process guard returned an invalid server start tick'

wait_for_ready() {
    local attempt status identity_status
    for ((attempt = 0; attempt < startup_timeout_seconds; attempt++)); do
        # Authenticate before every loopback probe. A pre-existing service
        # must not be treated as this launch merely because it answers readyz.
        if server_identity_state "$server_pid" "$server_start_ticks"; then
            :
        else
            identity_status=$?
            if ((identity_status == 3)); then
                wait "$server_pid" 2>/dev/null || true
                server_pid=
                die 'server exited or its PID was reused before readiness; inspect retained server.log'
            fi
            die 'could not authenticate the launched server during readiness'
        fi
        if status=$(/usr/bin/curl --noproxy '*' --http1.1 --silent --show-error \
            --connect-timeout 1 --max-time 2 --output /dev/null --write-out '%{http_code}' \
            "http://127.0.0.1:${port}/readyz" 2>/dev/null); then
            if [[ $status == 200 ]]; then
                # The responder can change while curl is in flight. Bind the
                # success result to the original child immediately before
                # declaring the service ready.
                if server_identity_state "$server_pid" "$server_start_ticks"; then
                    return 0
                else
                    identity_status=$?
                fi
                if ((identity_status == 3)); then
                    wait "$server_pid" 2>/dev/null || true
                    server_pid=
                    die 'server exited or its PID was reused while readiness was observed; inspect retained server.log'
                fi
                die 'could not authenticate the launched server after readiness response'
            fi
        fi
        /usr/bin/sleep 1
    done
    die "server did not become ready within ${startup_timeout_seconds}s"
}

wait_for_ready

config_endpoint="http://127.0.0.1:${port}/v1/config"
completion_endpoint="http://127.0.0.1:${port}/v1/completions"
metrics_endpoint="http://127.0.0.1:${port}/v1/c02/metrics"

"$repo_root/ci/release/run_remote_c02_config_endpoint_observation_v1.sh" \
    --endpoint "$config_endpoint" \
    --server-pid "$server_pid" \
    --gpu-index "$gpu_index" \
    --evidence-root "$evidence_dir" \
    --capture-name "$CONFIG_CAPTURE_NAME" >"$scratch_dir/config-capture.json"

bridge_report_path="$scratch_dir/config-bridge-replay.json"
run_python "$repo_root/ci/release/check_c02_config_bridge_v1.py" \
    --evidence-root "$evidence_dir" \
    --endpoint-path "$CONFIG_CAPTURE_NAME/raw/config-endpoint.json" \
    --startup-artifact-path "$(basename -- "$startup_artifact")" \
    --session-path "$CONFIG_CAPTURE_NAME/session.json" \
    --expected-candidate-id "$candidate_id" \
    --expected-configuration-profile "$configuration_profile" >"$bridge_report_path"

derive_configuration_sha256() {
    /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C TZ=UTC /usr/bin/python3 -B -I -S -c '
import hashlib
import json
import os
import re
import stat
import sys

path, candidate, profile, pid_text, start_ticks_text, gpu_index_text = sys.argv[1:]
flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
before = os.lstat(path)
if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
    raise SystemExit("unsafe config bridge diagnostic report")
fd = os.open(path, flags)
try:
    opened = os.fstat(fd)
    if (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (before.st_dev, before.st_ino, before.st_mode, before.st_nlink, before.st_size, before.st_mtime_ns, before.st_ctime_ns):
        raise SystemExit("config bridge diagnostic report changed while opening")
    raw = b""
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        raw += block
    if os.fstat(fd).st_size != len(raw):
        raise SystemExit("config bridge diagnostic report changed while reading")
finally:
    os.close(fd)
after = os.lstat(path)
if (after.st_dev, after.st_ino, after.st_mode, after.st_nlink, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (before.st_dev, before.st_ino, before.st_mode, before.st_nlink, before.st_size, before.st_mtime_ns, before.st_ctime_ns):
    raise SystemExit("config bridge diagnostic report changed while reading")
try:
    report = json.loads(raw.decode("utf-8"))
except Exception as error:
    raise SystemExit(f"invalid config bridge diagnostic report: {error}")
if raw != json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"):
    raise SystemExit("config bridge diagnostic report is not canonical")
if set(report) != {"schema_version", "status", "qualification_status", "candidate_id", "runtime_identity", "configuration_evidence", "target", "reason_codes"}:
    raise SystemExit("config bridge diagnostic report field set is invalid")
if report["schema_version"] != "riley.c02-config-bridge-replay.v1" or report["status"] != "bound" or report["qualification_status"] != "not-run" or report["candidate_id"] != candidate or report["reason_codes"] != []:
    raise SystemExit("config bridge diagnostic report identity is invalid")
identity = report["runtime_identity"]
if not isinstance(identity, dict) or set(identity) != {"configuration_profile", "configuration_sha256"} or identity["configuration_profile"] != profile:
    raise SystemExit("config bridge diagnostic runtime identity is invalid")
configuration_sha256 = identity["configuration_sha256"]
if not isinstance(configuration_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", configuration_sha256) is None or configuration_sha256 == "0" * 64:
    raise SystemExit("config bridge diagnostic configuration SHA is invalid")
target = report["target"]
if not isinstance(target, dict) or set(target) != {"server_pid", "server_start_ticks", "gpu_index", "gpu_uuid", "listener_port", "listener_inode"}:
    raise SystemExit("config bridge diagnostic target is invalid")
if target["server_pid"] != int(pid_text) or target["server_start_ticks"] != int(start_ticks_text) or target["gpu_index"] != int(gpu_index_text) or type(target["server_start_ticks"]) is not int or target["server_start_ticks"] < 1 or type(target["listener_port"]) is not int or not 1024 <= target["listener_port"] <= 65535 or type(target["listener_inode"]) is not int or target["listener_inode"] < 1 or not isinstance(target["gpu_uuid"], str) or re.fullmatch(r"GPU-[0-9A-Fa-f-]+", target["gpu_uuid"]) is None:
    raise SystemExit("config bridge diagnostic target drifted")
print(configuration_sha256)
' "$bridge_report_path" "$candidate_id" "$configuration_profile" "$server_pid" "$server_start_ticks" "$gpu_index"
}

configuration_sha256=$(derive_configuration_sha256)

"$repo_root/ci/release/run_remote_c02_raw_soak_scenarios_v1.sh" \
    --endpoint "$completion_endpoint" \
    --server-pid "$server_pid" \
    --candidate-id "$candidate_id" \
    --configuration-profile "$configuration_profile" \
    --configuration-sha256 "$configuration_sha256" \
    --evidence-root "$evidence_dir" \
    --capture-name "$FALLBACK_CAPTURE_NAME" \
    --audit-dir-name "$SOURCE_AUDIT_DIRECTORY" \
    --scenario-contract "$evidence_dir/$FROZEN_CONTRACT_NAME" \
    --repository-root "$repo_root" \
    --audit-wait-seconds "$audit_wait_seconds" >"$scratch_dir/fallback-capture.json"

# There is deliberately no work between the one native-fallback completion and
# this one-sample observation. A later version must introduce a distinct timing
# contract before it can accept aggregate or interleaved scenarios.
"$repo_root/ci/release/run_remote_c02_observations_v2.sh" \
    --endpoint "$metrics_endpoint" \
    --server-pid "$server_pid" \
    --gpu-index "$gpu_index" \
    --evidence-root "$evidence_dir" \
    --capture-name "$FALLBACK_OBSERVATION_NAME" \
    --interval-seconds 1 \
    --sample-count 1 >"$scratch_dir/observation.json"

shutdown_server_successfully() {
    local pid=$server_pid
    local ticks=$server_start_ticks
    local attempt=0
    local identity_status signal_status server_status
    [[ -n $pid && $ticks =~ ^[1-9][0-9]*$ ]] || \
        die 'server identity disappeared before graceful shutdown'
    if server_identity_state "$pid" "$ticks"; then
        identity_status=0
    else
        identity_status=$?
    fi
    if ((identity_status != 0)); then
        if ((identity_status == 3)); then
            wait "$pid" 2>/dev/null || true
        fi
        server_pid=
        die 'server exited, was reused, or could not be authenticated before graceful shutdown'
    fi
    if signal_server_if_current "$pid" "$ticks" TERM; then
        signal_status=0
    else
        signal_status=$?
    fi
    if ((signal_status != 0)); then
        if ((signal_status == 3)); then
            wait "$pid" 2>/dev/null || true
        fi
        server_pid=
        die 'could not safely deliver SIGTERM to the bridged server process'
    fi
    while ((attempt < shutdown_timeout_seconds)); do
        if server_identity_state "$pid" "$ticks"; then
            identity_status=0
        else
            identity_status=$?
        fi
        ((identity_status == 0)) || break
        /usr/bin/sleep 1
        attempt=$((attempt + 1))
    done
    # Avoid acting on the state from before the final sleep: a server that
    # already exited or became a zombie must be reaped, not sent SIGKILL.
    if ((identity_status == 0)); then
        if server_identity_state "$pid" "$ticks"; then
            identity_status=0
        else
            identity_status=$?
        fi
    fi
    if ((identity_status == 0)); then
        if signal_server_if_current "$pid" "$ticks" KILL; then
            wait "$pid" 2>/dev/null || true
        fi
        server_pid=
        die "server did not exit within ${shutdown_timeout_seconds}s; no raw terminal manifest may be emitted"
    fi
    if ((identity_status != 3)); then
        server_pid=
        die 'could not authenticate the server while waiting for graceful shutdown'
    fi
    if wait "$pid"; then
        server_status=0
    else
        server_status=$?
    fi
    server_pid=
    ((server_status == 0)) || die "server exited with status ${server_status}; no raw terminal manifest may be emitted"
}

shutdown_server_successfully
verify_launch_inputs post-exit >"$scratch_dir/launch-inputs-post.json"

run_python "$repo_root/ci/release/verify_c02_lifecycle_shutdown_v1.py" \
    --evidence-root "$evidence_dir" \
    --candidate-id "$candidate_id" \
    --configuration-profile "$configuration_profile" >"$scratch_dir/shutdown-check.json"

# The private finalizer opens one fresh root FD and holds its EX lock across
# only the fixed v5 request -> terminal binder normal-return edge. It does not
# call the public writer/binder wrappers and cannot publish a lifecycle receipt.
# A terminal-marker fsync ambiguity exits nonzero; its visible pair is never a
# reason to rerun this finalizer or emit a success record.
run_private_v5_raw_finalizer >"$scratch_dir/v5-raw-manifest.json"

printf '%s\n' "C02 native-fallback raw evidence completed at ${evidence_dir}; qualification_status=not-run"
