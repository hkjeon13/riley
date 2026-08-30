#!/usr/bin/env bash
# Verify only the authenticated parent/child GPU-lock handoff required by a
# future RC3 Gate E producer. This script deliberately creates no evidence
# and does not run any Gate E action.

if [[ ${BASH_SOURCE[0]:-} != "$0" ]]; then
    /usr/bin/printf '%s\n' 'run_remote_rc3_gate_e_session_v1.sh: must be executed, not sourced' >&2
    return 2 2>/dev/null || exit 2
fi
if [[ ${BASH_SOURCE[0]:-} != /* ]]; then
    /usr/bin/printf '%s\n' 'run_remote_rc3_gate_e_session_v1.sh: must be invoked by absolute path' >&2
    exit 2
fi

set -euo pipefail
set -o noclobber
umask 077
IFS=$' \t\n'

readonly SCRIPT_NAME='run_remote_rc3_gate_e_session_v1.sh'
readonly SCRIPT_PATH="${BASH_SOURCE[0]}"
readonly GPU_LOCK_PATH='/var/tmp/riley-server-4096-gpu-evidence.lock'
readonly SUPERVISOR_LOCK_FD=9

usage() {
    /bin/cat <<'EOF'
usage: bash /absolute/path/to/ci/release/run_remote_rc3_gate_e_session_v1.sh --supervisor-smoke-test

This is only an authenticated supervisor smoke test for a future RC3 Gate E
producer. It obtains the shared host GPU-evidence lock, proves the clean
parent/child handoff, prints one diagnostic to stderr, and exits.

It does not select or query a GPU, create an evidence root, capture artifacts,
invoke a subproducer, replay Gate E, publish a receipt, or make a
qualification decision. A successful smoke test means only that the
authenticated supervisor handoff completed; no Gate E action was run.

The script itself must be invoked by absolute path so an ambient shell hook
cannot change the child script identity before the authenticated handoff.
EOF
}

outer_die() {
    /usr/bin/printf '%s\n' "$SCRIPT_NAME: $*" >&2
    exit 2
}

# Keep every invalid outer invocation away from the shared host lock.
preflight_invocation() {
    (($# == 1)) || outer_die 'usage requires exactly one --supervisor-smoke-test option'
    [[ $1 == --supervisor-smoke-test ]] || outer_die "unknown option: $1"
}

if (($# == 0)); then
    usage >&2
    exit 2
fi
if (($# == 1)) && [[ $1 == --help || $1 == -h ]]; then
    usage
    exit 0
fi

# The parent owns the no-follow host lock throughout the authenticated child.
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
    raise SystemExit(f"RC3 Gate E session supervisor: cannot open GPU lock safely: {error}")
try:
    if opened_fd != LOCK_FD:
        os.dup2(opened_fd, LOCK_FD, inheritable=False)
        os.close(opened_fd)
    lock_fd = LOCK_FD
    metadata = os.fstat(lock_fd)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600):
        raise SystemExit("RC3 Gate E session supervisor: unsafe shared GPU lock inode")
    named = os.stat(LOCK_PATH, follow_symlinks=False)
    if (named.st_dev, named.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("RC3 Gate E session supervisor: GPU lock path changed while opening")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("RC3 Gate E session supervisor: another GPU evidence capture holds the host lock")
    named = os.stat(LOCK_PATH, follow_symlinks=False)
    if (named.st_dev, named.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("RC3 Gate E session supervisor: GPU lock path changed while locking")

    supervisor_pid = os.getpid()
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
            "RILEY_RC3_GATE_E_SESSION_SUPERVISOR_PID": str(supervisor_pid),
            "RILEY_RC3_GATE_E_SESSION_SUPERVISOR_EXE": "/usr/bin/python3",
            "RILEY_RC3_GATE_E_SESSION_SUPERVISOR_LOCK_FD": str(lock_fd),
            "RILEY_RC3_GATE_E_SESSION_SUPERVISOR_LOCK_ID": f"{metadata.st_dev}:{metadata.st_ino}",
        }
        script = os.path.abspath(sys.argv[1])
        os.execve(
            "/usr/bin/bash",
            ["/usr/bin/bash", script, "--gpu-lock-supervised", *sys.argv[2:]],
            environment,
        )

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
' "$SCRIPT_PATH" "$@"
fi
shift

[[ ${RILEY_RC3_GATE_E_SESSION_SUPERVISOR_PID:-} =~ ^[1-9][0-9]*$ ]] || outer_die 'supervisor PID was not authenticated'
[[ $PPID == "${RILEY_RC3_GATE_E_SESSION_SUPERVISOR_PID}" ]] || outer_die 'supervisor is not the direct parent'
[[ ${RILEY_RC3_GATE_E_SESSION_SUPERVISOR_EXE:-} == /usr/bin/python3 ]] || outer_die 'supervisor executable identity is invalid'
[[ /proc/$PPID/exe -ef /usr/bin/python3 ]] || outer_die 'supervisor executable differs from expected Python'
[[ ${RILEY_RC3_GATE_E_SESSION_SUPERVISOR_LOCK_FD:-} == "$SUPERVISOR_LOCK_FD" ]] || outer_die 'supervisor lock descriptor is invalid'
[[ ${RILEY_RC3_GATE_E_SESSION_SUPERVISOR_LOCK_ID:-} =~ ^[0-9]+:[0-9]+$ ]] || outer_die 'supervisor lock identity is invalid'
[[ /proc/$PPID/fd/$SUPERVISOR_LOCK_FD -ef "$GPU_LOCK_PATH" ]] || outer_die 'supervisor does not hold canonical GPU lock inode'
[[ /proc/$$/fd/$SUPERVISOR_LOCK_FD -ef "$GPU_LOCK_PATH" ]] || outer_die 'authenticated lock descriptor was not inherited'
observed_lock_id=$(/usr/bin/stat -Lc '%d:%i' "/proc/$$/fd/$SUPERVISOR_LOCK_FD") || outer_die 'cannot inspect inherited supervisor lock descriptor'
[[ $observed_lock_id == "${RILEY_RC3_GATE_E_SESSION_SUPERVISOR_LOCK_ID}" ]] || outer_die 'inherited supervisor lock identity differs from parent'
if ! /usr/bin/grep -Eq "^lock:.*FLOCK[[:space:]]+ADVISORY[[:space:]]+WRITE[[:space:]]+$PPID([[:space:]]|$)" "/proc/$PPID/fdinfo/$SUPERVISOR_LOCK_FD"; then
    outer_die 'supervisor does not own kernel GPU flock'
fi
exec 9>&-
[[ ! -e /proc/$$/fd/$SUPERVISOR_LOCK_FD ]] || outer_die 'Bash retained supervisor lock descriptor'
unset RILEY_RC3_GATE_E_SESSION_SUPERVISOR_PID \
    RILEY_RC3_GATE_E_SESSION_SUPERVISOR_EXE \
    RILEY_RC3_GATE_E_SESSION_SUPERVISOR_LOCK_FD \
    RILEY_RC3_GATE_E_SESSION_SUPERVISOR_LOCK_ID \
    BASH_ENV ENV CDPATH
export PATH=/usr/bin:/bin
export LC_ALL=C TZ=UTC
hash -r

(($# == 1)) || outer_die 'authenticated supervisor received invalid action'
[[ $1 == --supervisor-smoke-test ]] || outer_die "authenticated supervisor received invalid action: $1"
/usr/bin/printf '%s\n' 'authenticated supervisor smoke test completed; no Gate E action was run' >&2
