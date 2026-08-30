#!/usr/bin/bash
# Produce the five native candidate runs for the PR-16 release performance
# gate.  CUDA/model work is restricted to the designated server-4096 host.

# Do this before even parsing --help.  The first Bash may have evaluated
# BASH_ENV, so ambient $0 must not select the script that the clean Python
# supervisor later executes while it owns the GPU lock.
if [[ ${BASH_SOURCE[0]:-} != "$0" ]]; then
    builtin printf '%s\n' 'release performance: runner must be executed, not sourced' >&2
    return 2 2>/dev/null || exit 2
fi
if [[ ${BASH_SOURCE[0]:-} != /* ]]; then
    builtin printf '%s\n' 'release performance: runner must be invoked by absolute path' >&2
    exit 2
fi

set -euo pipefail
set -o noclobber
umask 022
IFS=$' \t\n'

readonly PERFORMANCE_RUNNER_PATH="${BASH_SOURCE[0]}"

usage() {
    /bin/cat <<'EOF'
usage: bash /absolute/path/to/ci/run_remote_release_performance.sh \
  --optimizer-image sha256:... \
  --source-revision FULL_40_CHARACTER_COMMIT \
  --expected-source-archive-sha256 LOWERCASE_SHA256 \
  --profile-binary PATH \
  --expected-profile-binary-sha256 LOWERCASE_SHA256 \
  --model-dir PATH \
  --expected-model-tree-sha256 LOWERCASE_SHA256 \
  --optimizer-correctness-report PATH \
  --expected-optimizer-correctness-report-sha256 LOWERCASE_SHA256 \
  --output-dir NEW_ABSOLUTE_PATH
EOF
}

if (($# == 0)); then
    usage >&2
    exit 2
fi
if [[ $1 == -h || $1 == --help ]]; then
    usage
    exit 0
fi

# The first shell may already have evaluated BASH_ENV.  Before any release
# control-plane work, replace it through env -i with a pinned Python supervisor.
# The supervisor alone retains the canonical no-follow flock.  Its Bash child
# receives the descriptor only long enough to authenticate the direct parent,
# then closes it before any other process can be spawned.
if [[ ${1:-} != --gpu-lock-supervised ]]; then
    exec /usr/bin/env -i \
        PATH=/usr/bin:/bin \
        LC_ALL=C \
        TZ=UTC \
        HOME=/home/psyche \
        /usr/bin/python3.10 -I -S -E -c '
import ctypes
import fcntl
import hashlib
import os
import secrets
import signal
import stat
import subprocess
import sys
import time

PYTHON_SHA256 = "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
BASH_SHA256 = "59474588a312b6b6e73e5a42a59bf71e62b55416b6c9d5e4a6e1c630c2a9ecd4"
DOCKER_SHA256 = "29be5f37ee7fcb32bed170244a7d94f2eb94d272912e0bbe9328374e2eb4b7f6"
LOCK_PATH = "/var/tmp/riley-server-4096-gpu-evidence.lock"
LABEL = "org.riley.release-performance-supervisor"
PR_SET_PDEATHSIG = 1

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def require_reviewed_bootstrap(path, expected):
    metadata = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != 0
            or metadata.st_mode & 0o022 or file_sha256(path) != expected):
        raise SystemExit(f"release performance: unsafe reviewed bootstrap tool: {path}")

if not os.path.samefile("/proc/self/exe", "/usr/bin/python3.10"):
    raise SystemExit("release performance: supervisor executable path changed")
require_reviewed_bootstrap("/usr/bin/python3.10", PYTHON_SHA256)
require_reviewed_bootstrap("/usr/bin/bash", BASH_SHA256)
require_reviewed_bootstrap("/usr/bin/docker", DOCKER_SHA256)

flags = (os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
         | os.O_NONBLOCK | os.O_CLOEXEC)
lock_fd = os.open(LOCK_PATH, flags, 0o600)
os.set_inheritable(lock_fd, False)
metadata = os.fstat(lock_fd)
if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600):
    raise SystemExit("release performance: unsafe shared GPU lock inode")
named = os.stat(LOCK_PATH, follow_symlinks=False)
if (named.st_dev, named.st_ino) != (metadata.st_dev, metadata.st_ino):
    raise SystemExit("release performance: shared GPU lock path changed while opening")
try:
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("release performance: another GPU evidence capture holds the host lock")
named = os.stat(LOCK_PATH, follow_symlinks=False)
if (named.st_dev, named.st_ino) != (metadata.st_dev, metadata.st_ino):
    raise SystemExit("release performance: shared GPU lock path changed while locking")

supervisor_pid = os.getpid()
supervisor_token = secrets.token_hex(32)
forwarded_signals = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
previous_signal_mask = signal.pthread_sigmask(signal.SIG_BLOCK, forwarded_signals)
child_pid = os.fork()
if child_pid == 0:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        os._exit(125)
    if os.getppid() != supervisor_pid:
        os._exit(125)
    os.setsid()
    os.set_inheritable(lock_fd, True)
    signal.pthread_sigmask(signal.SIG_SETMASK, previous_signal_mask)
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "TZ": "UTC",
        "HOME": "/home/psyche",
        "RILEY_PERF_SUPERVISOR_PID": str(supervisor_pid),
        "RILEY_PERF_SUPERVISOR_EXE": "/usr/bin/python3.10",
        "RILEY_PERF_SUPERVISOR_LOCK_FD": str(lock_fd),
        "RILEY_PERF_SUPERVISOR_LOCK_ID": f"{metadata.st_dev}:{metadata.st_ino}",
        "RILEY_PERF_SUPERVISOR_TOKEN": supervisor_token,
    }
    os.execve(
        "/usr/bin/bash",
        ["/usr/bin/bash", sys.argv[1], "--gpu-lock-supervised", *sys.argv[2:]],
        environment,
    )

def forward_signal(signum, _frame):
    try:
        os.killpg(child_pid, signum)
    except ProcessLookupError:
        try:
            os.kill(child_pid, signum)
        except ProcessLookupError:
            pass

for forwarded_signal in forwarded_signals:
    signal.signal(forwarded_signal, forward_signal)
signal.pthread_sigmask(signal.SIG_SETMASK, previous_signal_mask)

wait_status = None
while wait_status is None:
    try:
        _waited_pid, wait_status = os.waitpid(child_pid, 0)
    except InterruptedError:
        continue

try:
    os.killpg(child_pid, signal.SIGTERM)
except ProcessLookupError:
    pass

if os.WIFEXITED(wait_status) and os.WEXITSTATUS(wait_status) == 0:
    raise SystemExit(0)

docker_environment = {
    "PATH": "/usr/bin:/bin",
    "LC_ALL": "C",
    "TZ": "UTC",
    "HOME": "/home/psyche",
    "DOCKER_CONFIG": "/nonexistent/riley-release-performance-docker-config",
}

def run_docker(arguments):
    return subprocess.run(
        ["/usr/bin/docker", *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=docker_environment,
        close_fds=True,
        timeout=15,
        check=False,
    )

cleanup_warning_printed = False
while True:
    try:
        listed = run_docker([
            "container", "ls", "--all", "--quiet", "--no-trunc",
            "--filter", f"label={LABEL}={supervisor_token}",
        ])
        if listed.returncode != 0:
            cleanup_error = listed.stderr.strip()
            raise RuntimeError(cleanup_error)
        container_ids = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
        if not container_ids:
            break
        active_container_found = False
        cleanup_error = "active labeled container remains"
        for container_id in container_ids:
            if (len(container_id) != 64
                    or any(character not in "0123456789abcdef" for character in container_id)):
                cleanup_error = "Docker returned an unsafe container identity"
                active_container_found = True
                break
            inspected = run_docker([
                "container", "inspect", "--format",
                "{{ index .Config.Labels \"org.riley.release-performance-supervisor\" }} {{.State.Status}}",
                container_id,
            ])
            inspected_fields = inspected.stdout.split()
            if (inspected.returncode != 0 or len(inspected_fields) != 2
                    or inspected_fields[0] != supervisor_token):
                cleanup_error = "container identity changed during cleanup"
                active_container_found = True
                break
            container_status = inspected_fields[1]
            if container_status in ("exited", "dead"):
                continue
            if container_status not in ("created", "running", "paused", "restarting", "removing"):
                cleanup_error = "container has an unknown active state"
                active_container_found = True
                break
            active_container_found = True
            removed = run_docker(["container", "rm", "--force", "--volumes", container_id])
            if removed.returncode != 0:
                cleanup_error = "could not remove supervised performance container"
                break
        if not active_container_found:
            break
    except Exception as error:
        cleanup_error = str(error)
    if not cleanup_warning_printed:
        print(
            f"release performance: retaining GPU lock until cleanup succeeds: {cleanup_error}",
            file=sys.stderr,
        )
        cleanup_warning_printed = True
    time.sleep(1)

if os.WIFEXITED(wait_status):
    raise SystemExit(os.WEXITSTATUS(wait_status))
if os.WIFSIGNALED(wait_status):
    raise SystemExit(128 + os.WTERMSIG(wait_status))
raise SystemExit(125)
' "${PERFORMANCE_RUNNER_PATH}" "$@"
fi
shift

[[ ${RILEY_PERF_SUPERVISOR_PID:-} =~ ^[1-9][0-9]*$ ]] || {
    echo 'release performance: supervisor PID was not authenticated' >&2
    exit 2
}
[[ ${PPID} == "${RILEY_PERF_SUPERVISOR_PID}" ]] || {
    echo 'release performance: supervisor is not the direct parent' >&2
    exit 2
}
[[ ${RILEY_PERF_SUPERVISOR_EXE:-} == /usr/bin/python3.10 ]] \
    && [[ /proc/${PPID}/exe -ef /usr/bin/python3.10 ]] || {
        echo 'release performance: supervisor executable differs from reviewed Python' >&2
        exit 2
    }
[[ ${RILEY_PERF_SUPERVISOR_LOCK_FD:-} =~ ^[0-9]+$ ]] \
    && ((RILEY_PERF_SUPERVISOR_LOCK_FD >= 3)) || {
    echo 'release performance: supervisor lock descriptor is invalid' >&2
    exit 2
}
[[ ${RILEY_PERF_SUPERVISOR_LOCK_ID:-} =~ ^[0-9]+:[0-9]+$ ]] || {
    echo 'release performance: supervisor lock inode identity is invalid' >&2
    exit 2
}
[[ ${RILEY_PERF_SUPERVISOR_TOKEN:-} =~ ^[0-9a-f]{64}$ ]] || {
    echo 'release performance: supervisor cleanup token is invalid' >&2
    exit 2
}
readonly PERF_SUPERVISOR_PID=${RILEY_PERF_SUPERVISOR_PID}
readonly PERF_SUPERVISOR_LOCK_FD=${RILEY_PERF_SUPERVISOR_LOCK_FD}
readonly PERF_SUPERVISOR_LOCK_ID=${RILEY_PERF_SUPERVISOR_LOCK_ID}
readonly PERF_SUPERVISOR_TOKEN=${RILEY_PERF_SUPERVISOR_TOKEN}
[[ /proc/${PERF_SUPERVISOR_PID}/fd/${PERF_SUPERVISOR_LOCK_FD} -ef \
    /var/tmp/riley-server-4096-gpu-evidence.lock ]] || {
    echo 'release performance: supervisor does not hold the canonical lock inode' >&2
    exit 2
}
[[ /proc/$$/fd/${PERF_SUPERVISOR_LOCK_FD} -ef \
    /var/tmp/riley-server-4096-gpu-evidence.lock ]] || {
    echo 'release performance: authentication lock descriptor was not inherited' >&2
    exit 2
}
parent_fd_flags=
parent_flock_pid=
while IFS=$' \t' read -r fdinfo_key fdinfo_value fdinfo_type fdinfo_kind fdinfo_mode fdinfo_pid fdinfo_rest; do
    case "${fdinfo_key}" in
        flags:) parent_fd_flags=${fdinfo_value} ;;
        lock:)
            if [[ ${fdinfo_type} == FLOCK \
                && ${fdinfo_kind} == ADVISORY \
                && ${fdinfo_mode} == WRITE ]]; then
                parent_flock_pid=${fdinfo_pid}
            fi
            ;;
    esac
done <"/proc/${PERF_SUPERVISOR_PID}/fdinfo/${PERF_SUPERVISOR_LOCK_FD}"
[[ ${parent_fd_flags} =~ ^0[0-7]+$ ]] || {
    echo 'release performance: supervisor fdinfo flags are malformed' >&2
    exit 2
}
parent_fd_flags_value=$((8#${parent_fd_flags#0}))
(( (parent_fd_flags_value & 8#2000000) != 0 )) \
    && (( (parent_fd_flags_value & 8#4000) != 0 )) || {
        echo 'release performance: supervisor lock lacks CLOEXEC or NONBLOCK' >&2
        exit 2
    }
[[ ${parent_flock_pid} == "${PERF_SUPERVISOR_PID}" ]] || {
    echo 'release performance: supervisor does not own the kernel flock' >&2
    exit 2
}
eval "exec ${PERF_SUPERVISOR_LOCK_FD}>&-"
[[ ! -e /proc/$$/fd/${PERF_SUPERVISOR_LOCK_FD} ]] || {
    echo 'release performance: Bash retained the supervisor lock descriptor' >&2
    exit 2
}
unset \
    RILEY_PERF_SUPERVISOR_PID \
    RILEY_PERF_SUPERVISOR_EXE \
    RILEY_PERF_SUPERVISOR_LOCK_FD \
    RILEY_PERF_SUPERVISOR_LOCK_ID \
    RILEY_PERF_SUPERVISOR_TOKEN

for unsafe_name in \
    BASH_ENV ENV CDPATH LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH PYTHONPATH PYTHONHOME \
    PYTHONINSPECT PYTHONSTARTUP PYTHONWARNINGS \
    GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY \
    GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_CONFIG_GLOBAL GIT_CONFIG_SYSTEM \
    GIT_CONFIG_COUNT GIT_CEILING_DIRECTORIES GIT_SSH GIT_SSH_COMMAND \
    GIT_EXEC_PATH GIT_CONFIG_PARAMETERS GIT_COMMON_DIR GIT_NAMESPACE \
    GIT_SHALLOW_FILE GIT_REPLACE_REF_BASE \
    DOCKER_HOST DOCKER_CONTEXT DOCKER_CONFIG DOCKER_TLS_VERIFY DOCKER_CERT_PATH
do
    if [[ -n ${!unsafe_name+x} ]]; then
        echo "release performance: unsafe inherited control-plane variable: ${unsafe_name}" >&2
        exit 2
    fi
done
export PATH=/usr/bin:/bin
export LC_ALL=C TZ=UTC
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
hash -r

if declare -F | /usr/bin/grep -Eq ' (basename|bash|cat|chmod|cmp|cp|df|dirname|docker|env|find|git|grep|head|hostname|install|mawk|mkdir|nvidia-smi|python3|sed|sha256sum|sleep|sort|stat|tail|tar|timedatectl|tr|uname|wc)$'; then
    echo 'release performance: inherited command function is forbidden' >&2
    exit 2
fi

readonly DESIGNATED_HOSTNAME='psyche-MS-7D91'
readonly DESIGNATED_GPU_UUID='GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0'
readonly GPU_LOCK_PATH='/var/tmp/riley-server-4096-gpu-evidence.lock'
readonly MAX_PREFLIGHT_ATTEMPTS=41
readonly PREFLIGHT_RETRY_SECONDS=30
readonly BASH_BIN=/usr/bin/bash
readonly DOCKER_BIN=/usr/bin/docker
readonly GIT_BIN=/usr/bin/git
readonly HOSTNAME_BIN=/usr/bin/hostname
readonly INSTALL_BIN=/usr/bin/install
readonly NVIDIA_SMI_BIN=/usr/bin/nvidia-smi
readonly PYTHON_BIN=/usr/bin/python3.10
readonly SHA256SUM_BIN=/usr/bin/sha256sum
readonly TAR_BIN=/usr/bin/tar

require_shared_gpu_lock() {
    local descriptor_path="/proc/${PERF_SUPERVISOR_PID}/fd/${PERF_SUPERVISOR_LOCK_FD}"
    local descriptor_id named_id fdinfo_key fdinfo_value fdinfo_type
    local fdinfo_kind fdinfo_mode fdinfo_pid fdinfo_rest flock_pid=
    [[ ${PPID} == "${PERF_SUPERVISOR_PID}" ]]
    [[ /proc/${PERF_SUPERVISOR_PID}/exe -ef /usr/bin/python3.10 ]]
    descriptor_id=$(/usr/bin/stat -Lc '%d:%i' -- "${descriptor_path}")
    named_id=$(/usr/bin/stat -c '%d:%i' -- "${GPU_LOCK_PATH}")
    test "${descriptor_id}" = "${PERF_SUPERVISOR_LOCK_ID}"
    test "${named_id}" = "${PERF_SUPERVISOR_LOCK_ID}"
    while IFS=$' \t' read -r fdinfo_key fdinfo_value fdinfo_type fdinfo_kind fdinfo_mode fdinfo_pid fdinfo_rest; do
        if [[ ${fdinfo_key} == lock: \
            && ${fdinfo_type} == FLOCK \
            && ${fdinfo_kind} == ADVISORY \
            && ${fdinfo_mode} == WRITE ]]; then
            flock_pid=${fdinfo_pid}
        fi
    done <"/proc/${PERF_SUPERVISOR_PID}/fdinfo/${PERF_SUPERVISOR_LOCK_FD}"
    [[ ${flock_pid} == "${PERF_SUPERVISOR_PID}" ]]
}

optimizer_image=
source_revision=
expected_source_archive_sha256=
profile_binary=
expected_profile_binary_sha256=
model_dir=
expected_model_tree_sha256=
optimizer_correctness_report=
expected_optimizer_correctness_report_sha256=
output_dir=
active_container=

cleanup_container() {
    if [[ -n ${active_container} ]]; then
        "${DOCKER_BIN}" container rm --force --volumes "${active_container}" >/dev/null 2>&1 || true
    fi
}
trap cleanup_container EXIT

while (($#)); do
    case "$1" in
        --optimizer-image)
            (($# >= 2)) || { usage >&2; exit 2; }
            optimizer_image=$2
            shift 2
            ;;
        --source-revision)
            (($# >= 2)) || { usage >&2; exit 2; }
            source_revision=$2
            shift 2
            ;;
        --expected-source-archive-sha256)
            (($# >= 2)) || { usage >&2; exit 2; }
            expected_source_archive_sha256=$2
            shift 2
            ;;
        --profile-binary)
            (($# >= 2)) || { usage >&2; exit 2; }
            profile_binary=$2
            shift 2
            ;;
        --expected-profile-binary-sha256)
            (($# >= 2)) || { usage >&2; exit 2; }
            expected_profile_binary_sha256=$2
            shift 2
            ;;
        --model-dir)
            (($# >= 2)) || { usage >&2; exit 2; }
            model_dir=$2
            shift 2
            ;;
        --expected-model-tree-sha256)
            (($# >= 2)) || { usage >&2; exit 2; }
            expected_model_tree_sha256=$2
            shift 2
            ;;
        --optimizer-correctness-report)
            (($# >= 2)) || { usage >&2; exit 2; }
            optimizer_correctness_report=$2
            shift 2
            ;;
        --expected-optimizer-correctness-report-sha256)
            (($# >= 2)) || { usage >&2; exit 2; }
            expected_optimizer_correctness_report_sha256=$2
            shift 2
            ;;
        --output-dir)
            (($# >= 2)) || { usage >&2; exit 2; }
            output_dir=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

for value in \
    "${optimizer_image}" \
    "${source_revision}" \
    "${expected_source_archive_sha256}" \
    "${profile_binary}" \
    "${expected_profile_binary_sha256}" \
    "${model_dir}" \
    "${expected_model_tree_sha256}" \
    "${optimizer_correctness_report}" \
    "${expected_optimizer_correctness_report_sha256}" \
    "${output_dir}"
do
    test -n "${value}" || { usage >&2; exit 2; }
done

revision_re='^[0-9a-f]{40}$'
sha_re='^[0-9a-f]{64}$'
image_re='^sha256:[0-9a-f]{64}$'
[[ ${source_revision} =~ ${revision_re} ]] || {
    echo 'release performance: source revision must be a lowercase 40-character SHA' >&2
    exit 2
}
[[ ${optimizer_image} =~ ${image_re} ]] || {
    echo 'release performance: optimizer image must be an immutable sha256 ID' >&2
    exit 2
}
for digest in \
    "${expected_source_archive_sha256}" \
    "${expected_profile_binary_sha256}" \
    "${expected_model_tree_sha256}" \
    "${expected_optimizer_correctness_report_sha256}"
do
    [[ ${digest} =~ ${sha_re} ]] || {
        echo 'release performance: trusted digests must be lowercase SHA-256 values' >&2
        exit 2
    }
done
case "${output_dir}" in
    /*) ;;
    *) echo 'release performance: output directory must be absolute' >&2; exit 2 ;;
esac

declare -a trusted_tool_args=()
reviewed_tools=(
    'mawk|/usr/bin/mawk|dc157030a32367742480403025a6f731275b07d039238d167ade535e6f3eb98e'
    'basename|/usr/bin/basename|3c19cca8e2630f570580104778cc1e3398811c4c57e3252f0727ce411ab0ad22'
    'bash|/usr/bin/bash|59474588a312b6b6e73e5a42a59bf71e62b55416b6c9d5e4a6e1c630c2a9ecd4'
    'cat|/bin/cat|210ffa7daedb3ef6e9230d391e9a10043699ba81080ebf40c6de70ed77e278ba'
    'chmod|/usr/bin/chmod|e624a2e918718e570f989dd05b219278c9fa7ae3b3ab8830302b2d98e0c7dca8'
    'cmp|/usr/bin/cmp|b355472d3c90ea94d11ebb8b750e6946ccd348edc6fca4aefc1235c3994ef791'
    'cp|/usr/bin/cp|8da5881bb59f65673bc22b3a09b0d663b19bc0e785cf986b05d41b8222449ec2'
    'df|/usr/bin/df|b06fe81669b9383abed94bb5cae1cb7a63c6e02801b1b7dd1c08d7d2c8987e86'
    'dirname|/usr/bin/dirname|674a6c35e9ece6a6ac62e6442e3c65f391f8a1a8d1537bdd4b2203423ec16e94'
    'docker|/usr/bin/docker|29be5f37ee7fcb32bed170244a7d94f2eb94d272912e0bbe9328374e2eb4b7f6'
    'env|/usr/bin/env|85036540673319c6c2f54233fd2b9e45a8a71246b51cc96c4e6ab8ee6c419eb0'
    'find|/usr/bin/find|791b89c8bffb8101fd7d4d212b80af66a2332834b05a42721104eb47e8fa2eb1'
    'git|/usr/bin/git|587ef21868c948b883993e23209b86a72a6ddc06aab1545c697ffc31075acd4a'
    'grep|/usr/bin/grep|73abb4280520053564fd4917286909ba3b054598b32c9cdfaf1d733e0202cc96'
    'head|/usr/bin/head|9e457645cdcfd74ee0a9688b25b7b017d8d393233a0c0bdf3bef3c57a1238ce2'
    'hostname|/usr/bin/hostname|d254481d352a5a2b55848a4aeac6002ad594d4ab605e7f1fd49a25683b33559e'
    'install|/usr/bin/install|519a00d199d07da6028ec5a9800d92c562934582a2ea1793b2cbc378a85c1439'
    'mkdir|/usr/bin/mkdir|bd2f081ac37d653181332bd27f35a6041dbf215a7957f65838a9cbec9e64928b'
    'nvidia-smi|/usr/bin/nvidia-smi|22964713c1701fb62b4dd10b26b0dd25d174e100af5bda20c65e0b0fcc32b3be'
    'python3|/usr/bin/python3.10|7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86'
    'sed|/usr/bin/sed|42e2ce00721556ff9d371778fc36adcbb7c1697f65c3f996c6c9b28206dba565'
    'sha256sum|/usr/bin/sha256sum|7645c8e76d75515ccb75c9086bdcf0d4071f2985f380f249253ead7d7c6810b3'
    'sleep|/usr/bin/sleep|b9aec374a2b2a175a182f615291ad408820b7fb8c663a184e37fa3492d3f8eff'
    'sort|/usr/bin/sort|0fc26ce295e8e549635da2129e389f63685745b3be7c1737db6251a296f1cd78'
    'stat|/usr/bin/stat|9b571b54bd2f17f5fbb841e1886c2d364f5138a02533f4ac3dbfbdaf4dddbea3'
    'tail|/usr/bin/tail|d686c3513b6ecbcc6ac826383bd4b8b0f00aa6500d8d3d5e593687a3dee8fce0'
    'tar|/usr/bin/tar|fd0d62eed19efd3e115aa1be44160f89d777cd1e6d6d8eb0ce7c8bdc879f59e2'
    'timedatectl|/usr/bin/timedatectl|a1d1298afc514e7143d1a7a4c0039ce1256871faf33fe356fd9063dd283df5d9'
    'tr|/usr/bin/tr|24f53bbf7e48b1be3b71f20cf29963a44dbf084aafe5301f0ed1425b91d1c60c'
    'uname|/usr/bin/uname|37df0311d0e24169abfd166bc6018d40b87306f7ff64d9eec256c8331ac26347'
    'wc|/usr/bin/wc|504463c7a12780b7439321be6e67f43ab61a3ff429cbf916c0722d19f98692a8'
)
for reviewed_tool in "${reviewed_tools[@]}"; do
    IFS='|' read -r tool tool_path tool_sha256 <<<"${reviewed_tool}"
    test -f "${tool_path}" && test ! -L "${tool_path}" && test -x "${tool_path}"
    test "$(/usr/bin/stat -c '%u' -- "${tool_path}")" = 0
    tool_mode=$(/usr/bin/stat -c '%a' -- "${tool_path}")
    (( (8#${tool_mode} & 8#022) == 0 ))
    test "$("${SHA256SUM_BIN}" "${tool_path}" | /usr/bin/mawk '{print $1}')" = "${tool_sha256}" || {
        echo "release performance: reviewed tool digest mismatch: ${tool}" >&2
        exit 2
    }
    trusted_tool_args+=(--tool "${tool}=${tool_path}")
done
require_shared_gpu_lock || {
    echo 'release performance: supervisor no longer owns the canonical GPU lock' >&2
    exit 2
}

repository_root=$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "${repository_root}"
test "$("${HOSTNAME_BIN}")" = "${DESIGNATED_HOSTNAME}" || {
    echo "release performance may run only on server-4096 (${DESIGNATED_HOSTNAME})" >&2
    exit 1
}

require_exact_clean_checkout() {
    test "$("${GIT_BIN}" rev-parse --verify 'HEAD^{commit}')" = "${source_revision}" || {
        echo 'release performance: checkout HEAD differs from selected revision' >&2
        return 1
    }
    test -z "$("${GIT_BIN}" status --porcelain=v1 --untracked-files=all)" || {
        echo 'release performance: checkout must be completely clean' >&2
        return 1
    }
}
require_exact_clean_checkout

profile_binary=$(cd "$(/usr/bin/dirname "${profile_binary}")" && pwd -P)/$(/usr/bin/basename "${profile_binary}")
optimizer_correctness_report=$(cd "$(/usr/bin/dirname "${optimizer_correctness_report}")" && pwd -P)/$(/usr/bin/basename "${optimizer_correctness_report}")
model_dir=$(cd "${model_dir}" && pwd -P)
test -f "${profile_binary}" && test ! -L "${profile_binary}" && test -x "${profile_binary}"
test -f "${optimizer_correctness_report}" && test ! -L "${optimizer_correctness_report}"
test -d "${model_dir}" && test ! -L "${model_dir}"
for path in "${profile_binary}" "${optimizer_correctness_report}" "${model_dir}" "${output_dir}"; do
    [[ ${path} != *,* ]] || {
        echo "release performance: Docker bind paths may not contain commas: ${path}" >&2
        exit 2
    }
done

output_parent=$(cd "$(/usr/bin/dirname "${output_dir}")" && pwd -P)
output_leaf=$(/usr/bin/basename "${output_dir}")
output_dir="${output_parent}/${output_leaf}"
test ! -e "${output_dir}" && test ! -L "${output_dir}"
case "${output_dir}" in
    "${repository_root}"|"${repository_root}/"*)
        echo 'release performance: output must be outside the source checkout' >&2
        exit 2
        ;;
esac
/usr/bin/mkdir -m 0700 "${output_dir}"
/usr/bin/mkdir -m 0700 \
    "${output_dir}/containers" \
    "${output_dir}/inputs" \
    "${output_dir}/preflight" \
    "${output_dir}/run-evidence"
/usr/bin/mkdir -m 0700 "${output_dir}/inputs/docker-config"
export DOCKER_CONFIG="${output_dir}/inputs/docker-config"

source_archive="${output_dir}/source.tar"
profile_snapshot="${output_dir}/inputs/riley-profile"
optimizer_report_snapshot="${output_dir}/inputs/optimization-correctness-report.json"
model_snapshot="${output_dir}/inputs/model"
/usr/bin/mkdir -m 0700 "${model_snapshot}"
"${GIT_BIN}" -c tar.umask=0002 archive \
    --format=tar \
    --output="${source_archive}" \
    "${source_revision}"
test "$("${SHA256SUM_BIN}" "${source_archive}" | /usr/bin/mawk '{print $1}')" = \
    "${expected_source_archive_sha256}" || {
        echo 'release performance: generated source archive differs from trusted digest' >&2
        exit 1
    }
test "$("${GIT_BIN}" get-tar-commit-id <"${source_archive}")" = "${source_revision}"
"${INSTALL_BIN}" -m 0555 -- "${profile_binary}" "${profile_snapshot}"
"${INSTALL_BIN}" -m 0444 -- "${optimizer_correctness_report}" "${optimizer_report_snapshot}"
/usr/bin/cp -a -- "${model_dir}/." "${model_snapshot}/"
/usr/bin/find "${model_snapshot}" -type d -exec /usr/bin/chmod 0555 {} +
/usr/bin/find "${model_snapshot}" -type f -exec /usr/bin/chmod 0444 {} +
/usr/bin/chmod 0444 "${source_archive}"

resolved_image_id=$("${DOCKER_BIN}" image inspect --format '{{.Id}}' "${optimizer_image}")
test "${resolved_image_id}" = "${optimizer_image}" || {
    echo 'release performance: optimizer image did not resolve to the trusted ID' >&2
    exit 1
}
test "$("${DOCKER_BIN}" image inspect --format '{{.Os}}/{{.Architecture}}' "${optimizer_image}")" = linux/amd64
image_inspect="${output_dir}/optimizer-image-inspect-before.json"
"${DOCKER_BIN}" image inspect "${optimizer_image}" >"${image_inspect}"
gpu_csv="${output_dir}/gpu.csv"
"${NVIDIA_SMI_BIN}" --id="${DESIGNATED_GPU_UUID}" \
    --query-gpu=name,uuid,pci.bus_id,memory.total,driver_version,compute_cap \
    --format=csv,noheader,nounits >"${gpu_csv}"

run_release_python() {
    /usr/bin/env -i \
        PATH=/usr/bin:/bin \
        PYTHONHASHSEED=0 \
        PYTHONNOUSERSITE=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        LC_ALL=C \
        TZ=UTC \
        "${PYTHON_BIN}" \
        "${repository_root}/ci/release/run_release_python.py" \
        "$@"
}

accepted_preflight=
run_preflight() {
    local pair_index=$1 attempt=1 status stdout stderr
    local attempt_dir="${output_dir}/preflight/run-${pair_index}"
    /usr/bin/mkdir -m 0700 "${attempt_dir}"
    while ((attempt <= MAX_PREFLIGHT_ATTEMPTS)); do
        stdout=$(printf '%s/attempt-%03d.stdout' "${attempt_dir}" "${attempt}")
        stderr=$(printf '%s/attempt-%03d.stderr' "${attempt_dir}" "${attempt}")
        set +e
        RILEY_PREFLIGHT_OUTPUT_ROOT="${output_dir}" \
            "${BASH_BIN}" "${repository_root}/benchmarks/scripts/preflight.sh" \
            >"${stdout}" 2>"${stderr}"
        status=$?
        set -e
        if ((status == 0)); then
            accepted_preflight=${stdout}
            return 0
        fi
        if /usr/bin/grep -Eq '^preflight: start temperature [0-9]+ C exceeds 50 C$' "${stderr}" \
            && ((attempt < MAX_PREFLIGHT_ATTEMPTS)); then
            /usr/bin/sleep "${PREFLIGHT_RETRY_SECONDS}"
            attempt=$((attempt + 1))
            continue
        fi
        /bin/cat "${stderr}" >&2
        return "${status}"
    done
    return 1
}

validator_common=(
    "${repository_root}/ci/release/validate_release_performance_runner.py"
    --source-revision "${source_revision}"
    --source-archive "${source_archive}"
    --expected-source-archive-sha256 "${expected_source_archive_sha256}"
    --profile-binary "${profile_snapshot}"
    --expected-profile-binary-sha256 "${expected_profile_binary_sha256}"
    --optimizer-image-id "${optimizer_image}"
    --image-inspect "${image_inspect}"
    --gpu-csv "${gpu_csv}"
    --model-dir "${model_snapshot}"
    --expected-model-tree-sha256 "${expected_model_tree_sha256}"
    --optimizer-correctness-report "${optimizer_report_snapshot}"
    --expected-optimizer-correctness-report-sha256 "${expected_optimizer_correctness_report_sha256}"
)

require_snapshot_permissions_and_hashes() {
    test "$(/usr/bin/stat -c '%a' -- "${source_archive}")" = 444
    test "$(/usr/bin/stat -c '%a' -- "${profile_snapshot}")" = 555
    test "$(/usr/bin/stat -c '%a' -- "${optimizer_report_snapshot}")" = 444
    test -z "$(/usr/bin/find "${model_snapshot}" -type d ! -perm 0555 -print -quit)"
    test -z "$(/usr/bin/find "${model_snapshot}" -type f ! -perm 0444 -print -quit)"
    test -z "$(/usr/bin/find "${model_snapshot}" -mindepth 1 ! -type d ! -type f -print -quit)"
    test "$("${SHA256SUM_BIN}" "${source_archive}" | /usr/bin/mawk '{print $1}')" = \
        "${expected_source_archive_sha256}"
    test "$("${SHA256SUM_BIN}" "${profile_snapshot}" | /usr/bin/mawk '{print $1}')" = \
        "${expected_profile_binary_sha256}"
    test "$("${SHA256SUM_BIN}" "${optimizer_report_snapshot}" | /usr/bin/mawk '{print $1}')" = \
        "${expected_optimizer_correctness_report_sha256}"
}

revalidate_immutable_inputs() {
    local label=$1 preflight_path=$2 output_path=$3
    require_snapshot_permissions_and_hashes || {
        echo "release performance: immutable input permissions/hash drift at ${label}" >&2
        return 1
    }
    run_release_python "${validator_common[@]}" --mode preflight \
        --preflight "${preflight_path}" >"${output_path}"
}

trim_gpu_value() {
    /usr/bin/sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' <<<"$1"
}

gpu_processes() {
    "${NVIDIA_SMI_BIN}" --id="${DESIGNATED_GPU_UUID}" \
        --query-compute-apps=pid --format=csv,noheader,nounits
}

append_gpu_monitor_sample() {
    local receipt=$1 capture_id=$2 container_id=$3 stage=$4 sample_index=$5
    local row power_limit graphics_clock memory_clock temperature memory_used
    local processes pid process_field=none
    row=$("${NVIDIA_SMI_BIN}" --id="${DESIGNATED_GPU_UUID}" \
        --query-gpu=power.limit,clocks.applications.graphics,clocks.applications.memory,temperature.gpu,memory.used \
        --format=csv,noheader,nounits)
    IFS=, read -r power_limit graphics_clock memory_clock temperature memory_used <<<"${row}"
    power_limit=$(trim_gpu_value "${power_limit}")
    graphics_clock=$(trim_gpu_value "${graphics_clock}")
    memory_clock=$(trim_gpu_value "${memory_clock}")
    temperature=$(trim_gpu_value "${temperature}")
    memory_used=$(trim_gpu_value "${memory_used}")
    test "${power_limit}" = 450.00
    test "${graphics_clock}" = '[N/A]'
    test "${memory_clock}" = '[N/A]'
    [[ ${temperature} =~ ^[0-9]+$ ]] && ((temperature <= 95))
    [[ ${memory_used} =~ ^[0-9]+$ ]] && ((memory_used <= 24564))

    processes=$(gpu_processes)
    if [[ -n ${processes//[[:space:]]/} ]]; then
        process_field=
        while IFS= read -r pid; do
            pid=$(trim_gpu_value "${pid}")
            [[ ${pid} =~ ^[1-9][0-9]*$ ]] || return 1
            if ! /usr/bin/grep -Fq -- "${container_id}" "/proc/${pid}/cgroup" 2>/dev/null; then
                printf 'release performance: foreign CUDA PID %s during %s\n' "${pid}" "${stage}" >&2
                return 1
            fi
            if [[ -n ${process_field} ]]; then
                process_field+=';'
            fi
            process_field+="container:${pid}"
        done <<<"${processes}"
    fi
    if [[ ${stage} != running ]]; then
        test "${process_field}" = none
        ((memory_used <= 256))
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${capture_id}" "${container_id}" "${stage}" "${sample_index}" \
        "${power_limit}" "${graphics_clock}" \
        "${memory_clock}" "${temperature}" "${memory_used}" "${process_field}" \
        >>"${receipt}"
}

monitor_running_container_gpu() {
    local container_id=$1 capture_id=$2 receipt=$3 sample_index=1 observed_container=0 running
    while :; do
        running=$("${DOCKER_BIN}" inspect --format '{{.State.Running}}' "${container_id}") || return 1
        [[ ${running} == true ]] || break
        if ! append_gpu_monitor_sample "${receipt}" "${capture_id}" "${container_id}" running "${sample_index}"; then
            "${DOCKER_BIN}" kill "${container_id}" >/dev/null 2>&1 || true
            return 1
        fi
        if /usr/bin/tail -n 1 "${receipt}" | /usr/bin/grep -Eq ',container:[0-9]+(;container:[0-9]+)*$'; then
            observed_container=1
        fi
        sample_index=$((sample_index + 1))
        /usr/bin/sleep 1
    done
    ((observed_container == 1)) || {
        echo 'release performance: monitor observed no candidate CUDA process' >&2
        return 1
    }
    printf '%s' "${sample_index}"
}

declare -a preflight_snapshots=()
declare -a raw_runs=()
declare -a raw_snapshots=()
declare -a raw_snapshot_sha256=()
declare -a container_inspect_before=()
declare -a container_inspect_after=()
declare -a gpu_monitor_receipts=()
declare -a capture_ids=()
declare -a execution_receipt_outputs=()
for pair_index in 1 2 3 4 5; do
    require_shared_gpu_lock || {
        echo 'release performance: shared GPU lock inode changed before a run' >&2
        exit 1
    }
    capture_id=$(printf '%s:%s\n' "${PERF_SUPERVISOR_TOKEN}" "${pair_index}" \
        | "${SHA256SUM_BIN}" | /usr/bin/mawk '{print $1}')
    [[ ${capture_id} =~ ^[0-9a-f]{64}$ ]]
    capture_ids+=("${capture_id}")
    run_preflight "${pair_index}"
    receipt_dir="${output_dir}/run-${pair_index}"
    run_evidence_dir="${output_dir}/run-evidence/run-${pair_index}"
    /usr/bin/mkdir -m 0700 "${receipt_dir}"
    /usr/bin/mkdir -m 0733 "${run_evidence_dir}"
    "${INSTALL_BIN}" -m 0444 -- "${accepted_preflight}" "${receipt_dir}/preflight.txt"
    preflight_snapshots+=("${receipt_dir}/preflight.txt")
    execution_receipt_outputs+=("${receipt_dir}/execution-receipt.json")
    revalidate_immutable_inputs accepted-preflight \
        "${receipt_dir}/preflight.txt" \
        "${output_dir}/preflight/run-${pair_index}/accepted-validation.json"

    container_dir="${output_dir}/containers/run-${pair_index}"
    /usr/bin/mkdir -m 0700 "${container_dir}"
    container_id=$("${DOCKER_BIN}" create \
        --restart no \
        --entrypoint /bin/bash \
        --user 0:0 \
        --workdir /workspace \
        --no-healthcheck \
        --network none \
        --gpus "device=${DESIGNATED_GPU_UUID}" \
        --read-only \
        --cap-drop ALL \
        --security-opt no-new-privileges:true \
        --pids-limit 512 \
        --tmpfs /tmp:rw,nosuid,nodev,noexec,size=2147483648 \
        --label "org.riley.release-performance-supervisor=${PERF_SUPERVISOR_TOKEN}" \
        --env "RILEY_PERF_PAIR_INDEX=${pair_index}" \
        --env "RILEY_PERF_CAPTURE_ID=${capture_id}" \
        --env "RILEY_PERF_SOURCE_REVISION=${source_revision}" \
        --env "RILEY_PERF_SOURCE_ARCHIVE_SHA256=${expected_source_archive_sha256}" \
        --env "RILEY_PERF_PROFILE_BINARY_SHA256=${expected_profile_binary_sha256}" \
        --env "RILEY_PERF_OPTIMIZER_REPORT_SHA256=${expected_optimizer_correctness_report_sha256}" \
        --env "RILEY_PERF_OPTIMIZER_IMAGE_SHA256=${optimizer_image#sha256:}" \
        --env "RILEY_PERF_MODEL_TREE_SHA256=${expected_model_tree_sha256}" \
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
        --mount "type=bind,source=${profile_snapshot},destination=/input/riley-profile,readonly" \
        --mount "type=bind,source=${optimizer_report_snapshot},destination=/input/optimizer-correctness-report.json,readonly" \
        --mount "type=bind,source=${model_snapshot},destination=/model,readonly" \
        --mount "type=bind,source=${run_evidence_dir},destination=/evidence" \
        --mount type=volume,destination=/workspace,volume-nocopy \
        "${optimizer_image}" \
        -ceu 'test -z "$(/usr/bin/find /workspace -mindepth 1 -print -quit)"; /usr/bin/tar --extract --file /input/source.tar --directory /workspace; cd /workspace; exec /bin/bash ci/release/run_release_performance_once.sh')
    [[ ${container_id} =~ ^[0-9a-f]{64}$ ]]
    active_container=${container_id}
    inspect_before="${receipt_dir}/container-inspect-before.json"
    inspect_after="${receipt_dir}/container-inspect-after.json"
    "${DOCKER_BIN}" inspect "${container_id}" >"${inspect_before}"
    container_inspect_before+=("${inspect_before}")
    revalidate_immutable_inputs immediate-pre-start \
        "${receipt_dir}/preflight.txt" \
        "${container_dir}/input-validation-pre-start.json"
    gpu_monitor="${receipt_dir}/gpu-monitor.csv"
    printf '%s\n' 'capture_id,container_id,stage,sample_index,power_limit_w,graphics_clock_mhz,memory_clock_mhz,temperature_c,memory_used_mib,compute_processes' \
        >"${gpu_monitor}"
    append_gpu_monitor_sample "${gpu_monitor}" "${capture_id}" "${container_id}" pre_start 0
    gpu_monitor_receipts+=("${gpu_monitor}")

    set +e
    "${DOCKER_BIN}" start --attach "${container_id}" \
        >"${container_dir}/stdout.log" 2>"${container_dir}/stderr.log" &
    attach_pid=$!
    set -e
    container_running=false
    for _start_attempt in {1..100}; do
        if [[ $("${DOCKER_BIN}" inspect --format '{{.State.Running}}' "${container_id}") == true ]]; then
            container_running=true
            break
        fi
        kill -0 "${attach_pid}" 2>/dev/null || break
        /usr/bin/sleep 0.1
    done
    [[ ${container_running} == true ]] || {
        wait "${attach_pid}" || true
        echo "release performance: candidate container ${pair_index} never entered running state" >&2
        exit 1
    }
    monitor_next_index_path="${container_dir}/monitor-next-index"
    monitor_running_container_gpu "${container_id}" "${capture_id}" "${gpu_monitor}" \
        >"${monitor_next_index_path}" 2>"${container_dir}/monitor.stderr" &
    monitor_pid=$!
    set +e
    wait "${attach_pid}"
    container_status=$?
    wait "${monitor_pid}"
    monitor_status=$?
    set -e
    if ((monitor_status != 0)); then
        /bin/cat "${container_dir}/monitor.stderr" >&2
        echo "release performance: GPU monitor rejected candidate run ${pair_index}" >&2
        exit 1
    fi
    monitor_next_index=$(<"${monitor_next_index_path}")
    append_gpu_monitor_sample "${gpu_monitor}" "${capture_id}" "${container_id}" post_exit "${monitor_next_index}"
    revalidate_immutable_inputs post-exit \
        "${receipt_dir}/preflight.txt" \
        "${container_dir}/input-validation-post-exit.json"
    "${DOCKER_BIN}" inspect "${container_id}" >"${inspect_after}"
    container_inspect_after+=("${inspect_after}")
    "${DOCKER_BIN}" container rm --volumes "${container_id}" >/dev/null
    active_container=
    if ((container_status != 0)); then
        /bin/cat "${container_dir}/stderr.log" >&2
        echo "release performance: candidate run ${pair_index} failed" >&2
        exit "${container_status}"
    fi
    raw_run="${run_evidence_dir}/candidate-${pair_index}.json"
    test -f "${raw_run}" && test ! -L "${raw_run}" && test -s "${raw_run}"
    test "$(/usr/bin/stat -c '%a' -- "${raw_run}")" = 444
    raw_sha_before=$("${SHA256SUM_BIN}" "${raw_run}" | /usr/bin/mawk '{print $1}')
    raw_snapshot="${receipt_dir}/candidate.json"
    "${INSTALL_BIN}" -m 0444 -- "${raw_run}" "${raw_snapshot}"
    raw_sha_after=$("${SHA256SUM_BIN}" "${raw_run}" | /usr/bin/mawk '{print $1}')
    test "${raw_sha_before}" = "${raw_sha_after}"
    test "${raw_sha_before}" = "$("${SHA256SUM_BIN}" "${raw_snapshot}" | /usr/bin/mawk '{print $1}')"
    /usr/bin/chmod 0500 "${run_evidence_dir}"
    raw_runs+=("${raw_run}")
    raw_snapshots+=("${raw_snapshot}")
    raw_snapshot_sha256+=("${raw_sha_before}")
    "${SHA256SUM_BIN}" "${raw_run}" "${raw_snapshot}" >"${container_dir}/candidate.sha256"
    require_exact_clean_checkout
done

post_image_inspect="${output_dir}/optimizer-image-inspect-after.json"
require_shared_gpu_lock || {
    echo 'release performance: shared GPU lock inode changed before final replay' >&2
    exit 1
}
"${DOCKER_BIN}" image inspect "${optimizer_image}" >"${post_image_inspect}"
/usr/bin/cmp --silent "${image_inspect}" "${post_image_inspect}" || {
    echo 'release performance: optimizer image inspection changed during the run' >&2
    exit 1
}
test "$("${DOCKER_BIN}" image inspect --format '{{.Id}}' "${optimizer_image}")" = "${optimizer_image}"
require_exact_clean_checkout

final_preflight_args=(--preflight "${preflight_snapshots[@]}")
for array_index in 0 1 2 3 4; do
    test "$("${SHA256SUM_BIN}" "${raw_runs[${array_index}]}" | /usr/bin/mawk '{print $1}')" = \
        "${raw_snapshot_sha256[${array_index}]}"
    test "$("${SHA256SUM_BIN}" "${raw_snapshots[${array_index}]}" | /usr/bin/mawk '{print $1}')" = \
        "${raw_snapshot_sha256[${array_index}]}"
done
run_release_python \
    "${validator_common[@]}" \
    --mode final \
    --image-inspect-after "${post_image_inspect}" \
    "${final_preflight_args[@]}" \
    --run "${raw_runs[@]}" \
    --container-inspect-before "${container_inspect_before[@]}" \
    --container-inspect-after "${container_inspect_after[@]}" \
    --gpu-monitor "${gpu_monitor_receipts[@]}" \
    --supervisor-token "${PERF_SUPERVISOR_TOKEN}" \
    --capture-id "${capture_ids[@]}" \
    --execution-receipt-output "${execution_receipt_outputs[@]}" \
    --runner-manifest-output "${output_dir}/runner-manifest.json" \
    "${trusted_tool_args[@]}" \
    >"${output_dir}/raw-validation.json"

for array_index in 0 1 2 3 4; do
    test "$("${SHA256SUM_BIN}" "${raw_runs[${array_index}]}" | /usr/bin/mawk '{print $1}')" = \
        "${raw_snapshot_sha256[${array_index}]}"
    test "$("${SHA256SUM_BIN}" "${raw_snapshots[${array_index}]}" | /usr/bin/mawk '{print $1}')" = \
        "${raw_snapshot_sha256[${array_index}]}"
done

receipt_names=(
    gpu.csv
    optimizer-image-inspect-after.json
    optimizer-image-inspect-before.json
    runner-manifest.json
)
for pair_index in 1 2 3 4 5; do
    receipt_names+=(
        "run-${pair_index}/candidate.json"
        "run-${pair_index}/container-inspect-after.json"
        "run-${pair_index}/container-inspect-before.json"
        "run-${pair_index}/execution-receipt.json"
        "run-${pair_index}/gpu-monitor.csv"
        "run-${pair_index}/preflight.txt"
    )
done
(
    cd "${output_dir}"
    printf '%s\n' "${receipt_names[@]}" | /usr/bin/sort | while IFS= read -r receipt_name; do
        "${SHA256SUM_BIN}" -- "${receipt_name}"
    done >SHA256SUMS
)
/usr/bin/chmod 0444 "${output_dir}/SHA256SUMS"
run_release_python -c '
import json
import sys
sys.path.insert(0, sys.argv[2])
import check_release_performance as performance
receipt = performance.load_runner_receipt_root(sys.argv[1])
print(json.dumps({
    "schema_version": "riley.release-performance-receipt-replay.v1",
    "status": "passed",
    "container_ids": receipt["container_ids"],
    "workspace_volume_names": receipt["workspace_volume_names"],
    "gpu_monitors": receipt["gpu_monitors"],
    "executions": receipt["executions"],
    "raw_runs": receipt["derived"]["raw_runs"],
    "runner_manifest": receipt["manifest"],
}, sort_keys=True, indent=2, allow_nan=False))
' "${output_dir}" "${repository_root}/benchmarks/scripts" \
    >"${output_dir}/receipt-replay.json"
for array_index in 0 1 2 3 4; do
    test "$("${SHA256SUM_BIN}" "${raw_runs[${array_index}]}" | /usr/bin/mawk '{print $1}')" = \
        "${raw_snapshot_sha256[${array_index}]}"
    test "$("${SHA256SUM_BIN}" "${raw_snapshots[${array_index}]}" | /usr/bin/mawk '{print $1}')" = \
        "${raw_snapshot_sha256[${array_index}]}"
done
echo "release performance raw evidence complete: ${output_dir}"
