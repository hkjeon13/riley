#!/usr/bin/bash
# Build and run the Python-free PR16 soak derivative only on server-4096.
# The final release image remains the promoted subject; this layer adds only
# the process-observation tools required by ci/run_release_soak.sh.

set -euo pipefail
umask 077
IFS=$' \t\n'

DESIGNATED_HOSTNAME=psyche-MS-7D91
DESIGNATED_GPU_UUID=GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0
DESIGNATED_GPU_ROW="NVIDIA GeForce RTX 4090, ${DESIGNATED_GPU_UUID}, 8.9, 24564, 580.173.02"
MODEL_ID=HuggingFaceTB/SmolLM2-135M
MODEL_REVISION=93efa2f097d58c2a74874c7e644dbc9b0cee75a2
MODEL_CONFIG_SHA256=1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843
MODEL_WEIGHTS_SHA256=80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1
MODEL_TOKENIZER_JSON_SHA256=9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
MODEL_TOKENIZER_AGGREGATE_SHA256=51666963fa4cef6fbd450fc7ec5f70e483717757e0fcc2a5956f097d3915c4db
NATIVE_CORRECTNESS_GATE=smollm2-fp32-bf16-native-e0-v3
NATIVE_CORRECTNESS_SCHEMA=1.0.0

usage() {
    /bin/cat >&2 <<'EOF'
usage: ci/run_remote_release_soak.sh \
  --release-image-id sha256:... \
  --source-revision COMMIT \
  --source-archive PATH \
  --expected-source-archive-sha256 HEX \
  --release-binary PATH \
  --expected-release-binary-sha256 HEX \
  --model-dir PATH \
  --expected-model-tree-sha256 HEX \
  --materialized-manifest PATH \
  --expected-manifest-sha256 HEX \
  --correctness-golden PATH \
  --expected-correctness-golden-sha256 HEX \
  --native-correctness-report PATH \
  --expected-native-correctness-report-sha256 HEX \
  --test-image-tag NAME:TAG \
  --output-dir PATH

The materialized manifest is the adaptation point for separately reviewed
golden arguments. It must differ from the checked-in template only in the two
golden SHA-256 fields. The manifest, E2E golden, and native correctness report
each require independently reviewed digests; none can authorize the others.
EOF
}

if (($# == 0)); then
    usage
    exit 2
fi
if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi

# The caller's first Bash may already have evaluated BASH_ENV or imported an
# exported function. The authorized command therefore starts with env -i (see
# RELIABILITY_SOAK_TEST_LAYER.md), and this bootstrap immediately re-execs the
# real run under a closed environment while retaining one no-follow descriptor
# for the shared server-4096 GPU-evidence lock.
if [[ ${1:-} != --soak-gpu-lock-supervised ]]; then
    reviewed_python_sha256=$(
        /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C TZ=UTC HOME=/home/psyche \
            /usr/bin/sha256sum /usr/bin/python3.10
    )
    reviewed_python_sha256=${reviewed_python_sha256%% *}
    test "$reviewed_python_sha256" = \
        7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86 || {
        echo 'release soak: reviewed python3.10 digest mismatch' >&2
        exit 2
    }
    exec /usr/bin/env -i \
        PATH=/usr/bin:/bin \
        LC_ALL=C \
        TZ=UTC \
        HOME=/home/psyche \
        /usr/bin/python3.10 -I -S -c '
import fcntl
import ctypes
import os
import signal
import stat
import sys

lock_path = "/var/tmp/riley-server-4096-gpu-evidence.lock"
flags = (os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
         | os.O_CLOEXEC)
descriptor = os.open(lock_path, flags, 0o600)
metadata = os.fstat(descriptor)
if (
    not stat.S_ISREG(metadata.st_mode)
    or metadata.st_nlink != 1
    or metadata.st_uid != os.getuid()
    or stat.S_IMODE(metadata.st_mode) != 0o600
):
    raise SystemExit("release soak: unsafe shared GPU lock inode")
try:
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("release soak: another GPU evidence capture holds the host lock")
environment = {
    "PATH": "/usr/bin:/bin",
    "LC_ALL": "C",
    "TZ": "UTC",
    "HOME": "/home/psyche",
    "RILEY_SOAK_SUPERVISOR_PID": str(os.getpid()),
    "RILEY_SOAK_SUPERVISOR_LOCK_FD": str(descriptor),
}
supervisor_pid = os.getpid()
child = os.fork()
if child == 0:
    libc = ctypes.CDLL(None, use_errno=True)
    PR_SET_PDEATHSIG = 1
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        raise SystemExit("release soak: PR_SET_PDEATHSIG failed")
    if os.getppid() != supervisor_pid:
        raise SystemExit("release soak: lock supervisor died during bootstrap")
    os.close(descriptor)
    os.execve(
        "/usr/bin/bash",
        ["/usr/bin/bash", sys.argv[1], "--soak-gpu-lock-supervised", *sys.argv[2:]],
        environment,
    )

def forward(signum, _frame):
    try:
        os.kill(child, signum)
    except ProcessLookupError:
        pass

for forwarded_signal in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
    signal.signal(forwarded_signal, forward)
while True:
    try:
        _, wait_status = os.waitpid(child, 0)
        break
    except InterruptedError:
        continue
if os.WIFEXITED(wait_status):
    raise SystemExit(os.WEXITSTATUS(wait_status))
raise SystemExit(128 + os.WTERMSIG(wait_status))
' "$0" "$@"
fi
shift

[[ ${RILEY_SOAK_SUPERVISOR_PID:-} =~ ^[1-9][0-9]*$ ]]
[[ ${RILEY_SOAK_SUPERVISOR_LOCK_FD:-} =~ ^[0-9]+$ ]]
test "$PPID" = "$RILEY_SOAK_SUPERVISOR_PID" || {
    echo 'release soak: clean lock supervisor parent was bypassed' >&2
    exit 2
}
/usr/bin/python3.10 -I -S -c '
import os
import re
import stat
import sys

supervisor_pid = int(sys.argv[1])
descriptor = int(sys.argv[2])
bash_pid = int(sys.argv[3])
if os.getppid() != bash_pid:
    raise SystemExit("release soak: supervisor proof has the wrong Bash parent")
if os.readlink(f"/proc/{supervisor_pid}/exe") != "/usr/bin/python3.10":
    raise SystemExit("release soak: lock supervisor executable is not reviewed python3.10")
children = {
    int(value)
    for value in open(
        f"/proc/{supervisor_pid}/task/{supervisor_pid}/children",
        encoding="ascii",
    ).read().split()
}
if bash_pid not in children:
    raise SystemExit("release soak: Bash is not a direct supervisor child")
lock_path = "/var/tmp/riley-server-4096-gpu-evidence.lock"
metadata = os.stat(lock_path, follow_symlinks=False)
descriptor_metadata = os.stat(f"/proc/{supervisor_pid}/fd/{descriptor}")
if (
    not stat.S_ISREG(metadata.st_mode)
    or metadata.st_nlink != 1
    or metadata.st_uid != os.getuid()
    or stat.S_IMODE(metadata.st_mode) != 0o600
    or (metadata.st_dev, metadata.st_ino)
       != (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
):
    raise SystemExit("release soak: supervisor lock descriptor/path identity mismatch")
fdinfo = open(
    f"/proc/{supervisor_pid}/fdinfo/{descriptor}", encoding="ascii"
).read()
flags_match = re.search(r"^flags:\s+([0-7]+)$", fdinfo, re.MULTILINE)
if flags_match is None or not (int(flags_match.group(1), 8) & os.O_CLOEXEC):
    raise SystemExit("release soak: supervisor lock descriptor lacks FD_CLOEXEC")
lock_pattern = rf"^lock:\s+\d+:\s+FLOCK\s+ADVISORY\s+WRITE\s+{supervisor_pid}\s+"
if re.search(lock_pattern, fdinfo, re.MULTILINE) is None:
    raise SystemExit("release soak: supervisor does not hold the kernel flock")
' "$RILEY_SOAK_SUPERVISOR_PID" "$RILEY_SOAK_SUPERVISOR_LOCK_FD" "$$"
unset RILEY_SOAK_SUPERVISOR_PID RILEY_SOAK_SUPERVISOR_LOCK_FD

export PATH=/usr/bin:/bin
export LC_ALL=C TZ=UTC
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
test "$PATH" = /usr/bin:/bin
test "$HOME" = /home/psyche
hash -r

for unsafe_name in $(compgen -e); do
    case "$unsafe_name" in
        BASH_ENV|BASHOPTS|CDPATH|CURL_HOME|ENV|GLOBIGNORE|IFS|SHELLOPTS|XDG_CONFIG_HOME| \
        GCONV_PATH|GLIBC_TUNABLES|LOCPATH|MALLOC_TRACE|NLSPATH|POSIXLY_CORRECT| \
        PYTHONHOME|PYTHONPATH|PYTHONINSPECT|PYTHONSTARTUP|PYTHONWARNINGS| \
        CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES| \
        BASH_FUNC_*|LD_*|GIT_*|DOCKER_*|BUILDX_*)
            case "$unsafe_name" in
                GIT_CONFIG_GLOBAL|GIT_CONFIG_NOSYSTEM) ;;
                *)
                    echo "release soak: unsafe inherited control-plane variable: $unsafe_name" >&2
                    exit 2
                    ;;
            esac
            ;;
    esac
done

while read -r _ _ function_name; do
    [[ $function_name == usage ]] || {
        echo "release soak: inherited command function is forbidden: $function_name" >&2
        exit 2
    }
done < <(declare -F)

readonly BASH_BIN=/usr/bin/bash
readonly BASENAME_BIN=/usr/bin/basename
readonly CAT_BIN=/bin/cat
readonly CHMOD_BIN=/usr/bin/chmod
readonly CMP_BIN=/usr/bin/cmp
readonly CP_BIN=/usr/bin/cp
readonly DATE_BIN=/usr/bin/date
readonly DIRNAME_BIN=/usr/bin/dirname
readonly DOCKER_BIN=/usr/bin/docker
readonly ENV_BIN=/usr/bin/env
readonly FIND_BIN=/usr/bin/find
readonly GIT_BIN=/usr/bin/git
readonly GREP_BIN=/usr/bin/grep
readonly HOSTNAME_BIN=/usr/bin/hostname
readonly JQ_BIN=/usr/bin/jq
readonly MAWK_BIN=/usr/bin/mawk
readonly MKDIR_BIN=/usr/bin/mkdir
readonly NVIDIA_SMI_BIN=/usr/bin/nvidia-smi
readonly OD_BIN=/usr/bin/od
readonly PYTHON_BIN=/usr/bin/python3.10
readonly SHA256SUM_BIN=/usr/bin/sha256sum
readonly SORT_BIN=/usr/bin/sort
readonly STAT_BIN=/usr/bin/stat
readonly TAIL_BIN=/usr/bin/tail
readonly TAR_BIN=/usr/bin/tar
readonly TEE_BIN=/usr/bin/tee
readonly TR_BIN=/usr/bin/tr
readonly WC_BIN=/usr/bin/wc

reviewed_tools=(
    'bash|/usr/bin/bash|59474588a312b6b6e73e5a42a59bf71e62b55416b6c9d5e4a6e1c630c2a9ecd4'
    'basename|/usr/bin/basename|3c19cca8e2630f570580104778cc1e3398811c4c57e3252f0727ce411ab0ad22'
    'cat|/bin/cat|210ffa7daedb3ef6e9230d391e9a10043699ba81080ebf40c6de70ed77e278ba'
    'chmod|/usr/bin/chmod|e624a2e918718e570f989dd05b219278c9fa7ae3b3ab8830302b2d98e0c7dca8'
    'cmp|/usr/bin/cmp|b355472d3c90ea94d11ebb8b750e6946ccd348edc6fca4aefc1235c3994ef791'
    'cp|/usr/bin/cp|8da5881bb59f65673bc22b3a09b0d663b19bc0e785cf986b05d41b8222449ec2'
    'date|/usr/bin/date|08b85d43067bcd15edb0882d5372a8b5635e211f76b62ccc4d575f2ed4920e18'
    'dirname|/usr/bin/dirname|674a6c35e9ece6a6ac62e6442e3c65f391f8a1a8d1537bdd4b2203423ec16e94'
    'docker|/usr/bin/docker|29be5f37ee7fcb32bed170244a7d94f2eb94d272912e0bbe9328374e2eb4b7f6'
    'env|/usr/bin/env|85036540673319c6c2f54233fd2b9e45a8a71246b51cc96c4e6ab8ee6c419eb0'
    'find|/usr/bin/find|791b89c8bffb8101fd7d4d212b80af66a2332834b05a42721104eb47e8fa2eb1'
    'git|/usr/bin/git|587ef21868c948b883993e23209b86a72a6ddc06aab1545c697ffc31075acd4a'
    'grep|/usr/bin/grep|73abb4280520053564fd4917286909ba3b054598b32c9cdfaf1d733e0202cc96'
    'hostname|/usr/bin/hostname|d254481d352a5a2b55848a4aeac6002ad594d4ab605e7f1fd49a25683b33559e'
    'jq|/usr/bin/jq|858a84f22b39317f13a57b4b91e535925c1b4f819d9bb2864361df4ad6acb00f'
    'mawk|/usr/bin/mawk|dc157030a32367742480403025a6f731275b07d039238d167ade535e6f3eb98e'
    'mkdir|/usr/bin/mkdir|bd2f081ac37d653181332bd27f35a6041dbf215a7957f65838a9cbec9e64928b'
    'nvidia-smi|/usr/bin/nvidia-smi|22964713c1701fb62b4dd10b26b0dd25d174e100af5bda20c65e0b0fcc32b3be'
    'od|/usr/bin/od|8831c6be1e0b0a7c8c01e2f939b03d8d1d144e238c6b8e0a5d9d1a8c367ac910'
    'python3|/usr/bin/python3.10|7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86'
    'sha256sum|/usr/bin/sha256sum|7645c8e76d75515ccb75c9086bdcf0d4071f2985f380f249253ead7d7c6810b3'
    'sort|/usr/bin/sort|0fc26ce295e8e549635da2129e389f63685745b3be7c1737db6251a296f1cd78'
    'stat|/usr/bin/stat|9b571b54bd2f17f5fbb841e1886c2d364f5138a02533f4ac3dbfbdaf4dddbea3'
    'tail|/usr/bin/tail|d686c3513b6ecbcc6ac826383bd4b8b0f00aa6500d8d3d5e593687a3dee8fce0'
    'tar|/usr/bin/tar|fd0d62eed19efd3e115aa1be44160f89d777cd1e6d6d8eb0ce7c8bdc879f59e2'
    'tee|/usr/bin/tee|eb219ccfbdad53064135a4101d4f56f0d9e5f7f1cd20c032b29e3604264cf79b'
    'tr|/usr/bin/tr|24f53bbf7e48b1be3b71f20cf29963a44dbf084aafe5301f0ed1425b91d1c60c'
    'wc|/usr/bin/wc|504463c7a12780b7439321be6e67f43ab61a3ff429cbf916c0722d19f98692a8'
)
for reviewed_tool in "${reviewed_tools[@]}"; do
    IFS='|' read -r tool_name tool_path tool_sha256 <<<"$reviewed_tool"
    test -f "$tool_path" && test ! -L "$tool_path" && test -x "$tool_path"
    test "$("$STAT_BIN" -c '%u' -- "$tool_path")" = 0
    tool_mode=$("$STAT_BIN" -c '%a' -- "$tool_path")
    (( (8#$tool_mode & 8#022) == 0 ))
    test "$("$SHA256SUM_BIN" "$tool_path" | "$MAWK_BIN" '{print $1}')" = "$tool_sha256" || {
        echo "release soak: reviewed tool digest mismatch: $tool_name" >&2
        exit 2
    }
done

basename() { "$BASENAME_BIN" "$@"; }
chmod() { "$CHMOD_BIN" "$@"; }
cmp() { "$CMP_BIN" "$@"; }
cp() { "$CP_BIN" "$@"; }
date() { "$DATE_BIN" "$@"; }
dirname() { "$DIRNAME_BIN" "$@"; }
docker() { "$DOCKER_BIN" "$@"; }
find() { "$FIND_BIN" "$@"; }
git() { "$GIT_BIN" "$@"; }
grep() { "$GREP_BIN" "$@"; }
hostname() { "$HOSTNAME_BIN" "$@"; }
jq() { "$JQ_BIN" "$@"; }
mawk() { "$MAWK_BIN" "$@"; }
mkdir() { "$MKDIR_BIN" "$@"; }
nvidia-smi() { "$NVIDIA_SMI_BIN" "$@"; }
od() { "$OD_BIN" "$@"; }
sha256sum() { "$SHA256SUM_BIN" "$@"; }
sort() { "$SORT_BIN" "$@"; }
stat() { "$STAT_BIN" "$@"; }
tail() { "$TAIL_BIN" "$@"; }
tar() { "$TAR_BIN" "$@"; }
tee() { "$TEE_BIN" "$@"; }
tr() { "$TR_BIN" "$@"; }
wc() { "$WC_BIN" "$@"; }
readonly -f basename chmod cmp cp date dirname docker find git grep hostname jq mawk mkdir nvidia-smi od sha256sum sort stat tail tar tee tr wc

release_image_id=
source_revision=
source_archive=
expected_source_archive_sha256=
release_binary=
expected_release_binary_sha256=
model_dir=
expected_model_tree_sha256=
materialized_manifest=
expected_manifest_sha256=
correctness_golden=
expected_correctness_golden_sha256=
native_correctness_report=
expected_native_correctness_report_sha256=
test_image_tag=
output_dir=
active_container=

while (($#)); do
    case "$1" in
        --release-image-id)
            (($# >= 2)) || { usage; exit 2; }
            release_image_id=$2
            shift 2
            ;;
        --source-revision)
            (($# >= 2)) || { usage; exit 2; }
            source_revision=$2
            shift 2
            ;;
        --source-archive)
            (($# >= 2)) || { usage; exit 2; }
            source_archive=$2
            shift 2
            ;;
        --expected-source-archive-sha256)
            (($# >= 2)) || { usage; exit 2; }
            expected_source_archive_sha256=$2
            shift 2
            ;;
        --release-binary)
            (($# >= 2)) || { usage; exit 2; }
            release_binary=$2
            shift 2
            ;;
        --expected-release-binary-sha256)
            (($# >= 2)) || { usage; exit 2; }
            expected_release_binary_sha256=$2
            shift 2
            ;;
        --model-dir)
            (($# >= 2)) || { usage; exit 2; }
            model_dir=$2
            shift 2
            ;;
        --expected-model-tree-sha256)
            (($# >= 2)) || { usage; exit 2; }
            expected_model_tree_sha256=$2
            shift 2
            ;;
        --materialized-manifest)
            (($# >= 2)) || { usage; exit 2; }
            materialized_manifest=$2
            shift 2
            ;;
        --expected-manifest-sha256)
            (($# >= 2)) || { usage; exit 2; }
            expected_manifest_sha256=$2
            shift 2
            ;;
        --correctness-golden)
            (($# >= 2)) || { usage; exit 2; }
            correctness_golden=$2
            shift 2
            ;;
        --expected-correctness-golden-sha256)
            (($# >= 2)) || { usage; exit 2; }
            expected_correctness_golden_sha256=$2
            shift 2
            ;;
        --native-correctness-report)
            (($# >= 2)) || { usage; exit 2; }
            native_correctness_report=$2
            shift 2
            ;;
        --expected-native-correctness-report-sha256)
            (($# >= 2)) || { usage; exit 2; }
            expected_native_correctness_report_sha256=$2
            shift 2
            ;;
        --test-image-tag)
            (($# >= 2)) || { usage; exit 2; }
            test_image_tag=$2
            shift 2
            ;;
        --output-dir)
            (($# >= 2)) || { usage; exit 2; }
            output_dir=$2
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

for required_value in \
    "$release_image_id" \
    "$source_revision" \
    "$source_archive" \
    "$expected_source_archive_sha256" \
    "$release_binary" \
    "$expected_release_binary_sha256" \
    "$model_dir" \
    "$expected_model_tree_sha256" \
    "$materialized_manifest" \
    "$expected_manifest_sha256" \
    "$correctness_golden" \
    "$expected_correctness_golden_sha256" \
    "$native_correctness_report" \
    "$expected_native_correctness_report_sha256" \
    "$test_image_tag" \
    "$output_dir"
do
    test -n "$required_value" || { usage; exit 2; }
done

sha_re='^[0-9a-f]{64}$'
image_re='^sha256:[0-9a-f]{64}$'
git_re='^[0-9a-f]{40}$'
[[ $release_image_id =~ $image_re ]] || {
    echo "release image must be an immutable sha256 image ID" >&2
    exit 2
}
[[ $source_revision =~ $git_re ]] || {
    echo "source revision must be a full lowercase commit" >&2
    exit 2
}
for digest in \
    "$expected_source_archive_sha256" \
    "$expected_release_binary_sha256" \
    "$expected_model_tree_sha256" \
    "$expected_manifest_sha256" \
    "$expected_correctness_golden_sha256" \
    "$expected_native_correctness_report_sha256"
do
    [[ $digest =~ $sha_re ]] || {
        echo "trusted soak bindings must be lowercase SHA-256 values" >&2
        exit 2
    }
done
[[ $test_image_tag =~ ^[a-z0-9][a-z0-9._/-]*:[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ]] || {
    echo "test image tag must be an explicit repository:tag" >&2
    exit 2
}
case "$output_dir" in
    /*) ;;
    *) echo "output directory must be absolute" >&2; exit 2 ;;
esac
for path_value in \
    "$source_archive" \
    "$release_binary" \
    "$model_dir" \
    "$materialized_manifest" \
    "$correctness_golden" \
    "$native_correctness_report" \
    "$output_dir"
do
    [[ $path_value != *$'\n'* && $path_value != *','* ]] || {
        echo "soak paths must not contain newlines or commas" >&2
        exit 2
    }
done

actual_hostname=$(hostname)
test "$actual_hostname" = "$DESIGNATED_HOSTNAME" || {
    echo "release soak may run only on server-4096 (${DESIGNATED_HOSTNAME}), got ${actual_hostname}" >&2
    exit 1
}
gpu_rows=$(nvidia-smi \
    --query-gpu=name,uuid,compute_cap,memory.total,driver_version \
    --format=csv,noheader,nounits)
test "$(wc -l <<<"$gpu_rows" | mawk '{$1=$1;print}')" -eq 1
test "$gpu_rows" = "$DESIGNATED_GPU_ROW" || {
    echo "release soak host is not the designated RTX 4090 environment" >&2
    exit 1
}
IFS=, read -r actual_gpu_name actual_gpu_uuid actual_compute_capability actual_memory_total_mib actual_driver_version <<<"$gpu_rows"
actual_gpu_name=$(mawk '{$1=$1;print}' <<<"$actual_gpu_name")
actual_gpu_uuid=$(mawk '{$1=$1;print}' <<<"$actual_gpu_uuid")
actual_compute_capability=$(mawk '{$1=$1;print}' <<<"$actual_compute_capability")
actual_memory_total_mib=$(mawk '{$1=$1;print}' <<<"$actual_memory_total_mib")
actual_driver_version=$(mawk '{$1=$1;print}' <<<"$actual_driver_version")
[[ $actual_memory_total_mib =~ ^[0-9]+$ ]]

require_gpu_idle() {
    local stage=$1 compute_pids
    compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) || {
        echo "GPU compute-process query failed during $stage" >&2
        return 1
    }
    test -z "$compute_pids" || {
        echo "designated GPU is not idle during $stage" >&2
        return 1
    }
}
require_gpu_idle initial-preflight

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repository_root"
resolved_revision=$(git rev-parse --verify "${source_revision}^{commit}")
test "$resolved_revision" = "$source_revision"

require_exact_clean_checkout() {
    test "$(git rev-parse --verify 'HEAD^{commit}')" = "$resolved_revision" || {
        echo "checkout HEAD differs from the selected soak revision" >&2
        return 1
    }
    test -z "$(git status --porcelain=v1 --untracked-files=all)" || {
        echo "release soak checkout must be completely clean" >&2
        return 1
    }
}
require_exact_clean_checkout

canonical_file() {
    local input=$1 parent name
    test -f "$input"
    test ! -L "$input"
    parent=$(cd "$(dirname "$input")" && pwd -P)
    name=$(basename "$input")
    printf '%s/%s\n' "$parent" "$name"
}

source_archive_input=$(canonical_file "$source_archive")
release_binary=$(canonical_file "$release_binary")
materialized_manifest_input=$(canonical_file "$materialized_manifest")
correctness_golden_input=$(canonical_file "$correctness_golden")
native_correctness_report_input=$(canonical_file "$native_correctness_report")
model_dir_input=$(cd "$model_dir" && pwd -P)
test -d "$model_dir_input"
test ! -L "$model_dir_input"

sha256_file() {
    sha256sum "$1" | mawk '{print $1}'
}

write_model_manifest() {
    local root=$1 output=$2 model_file relative count=0
    test ! -e "$output" && test ! -L "$output"
    : >"$output"
    while IFS= read -r -d '' model_file; do
        relative=${model_file#"$root"/}
        [[ $relative =~ ^[A-Za-z0-9._/+@=-]+$ ]] || {
            echo "model path uses an unsafe alphabet: $relative" >&2
            return 1
        }
        printf '%s  %s\n' "$(sha256_file "$model_file")" "$relative" >>"$output"
        count=$((count + 1))
    done < <(find "$root" -type f -print0 | sort -z)
    test "$count" -gt 0
}

snapshot_regular_file() {
    "$PYTHON_BIN" -I -S - "$1" "$2" <<'PY'
import os
import stat
import sys

source_path, destination_path = sys.argv[1:]
source_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
destination_flags = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    | os.O_CLOEXEC
)
source = os.open(source_path, source_flags)
try:
    if not stat.S_ISREG(os.fstat(source).st_mode):
        raise SystemExit("release soak: snapshot source is not a regular file")
    destination = os.open(destination_path, destination_flags, 0o444)
    try:
        while True:
            chunk = os.read(source, 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(destination, view)
                if written <= 0:
                    raise SystemExit("release soak: snapshot write made no progress")
                view = view[written:]
        os.fchmod(destination, 0o444)
        os.fsync(destination)
    finally:
        os.close(destination)
finally:
    os.close(source)
PY
}

# Materialize every mutable input into the new evidence directory before
# validating or using its contents. Rechecks below make accidental same-UID
# replacement/restoration races observable; the designated host account itself
# remains part of the trusted execution boundary.
if [[ -e $output_dir || -L $output_dir ]]; then
    echo "refusing to reuse reliability soak output: $output_dir" >&2
    exit 1
fi
mkdir -m 0700 "$output_dir"
output_dir=$(cd "$output_dir" && pwd -P)
case "$output_dir" in
    "$repository_root"|"$repository_root"/*)
        echo "reliability soak output must be outside the source checkout" >&2
        exit 1
        ;;
esac
test "$(stat -c '%a' "$output_dir")" = 700
runtime_receipts="$output_dir/runtime-receipts"
mkdir -m 0700 "$runtime_receipts"
container_evidence="$output_dir/container-evidence"
mkdir -m 0777 "$container_evidence"
chmod 0777 "$container_evidence"
test "$(stat -c '%a' "$container_evidence")" = 777 || {
    echo "container evidence bind must be writable by production USER 65532:65532" >&2
    exit 1
}
docker_config="$output_dir/docker-config"
mkdir -m 0700 "$docker_config"
export DOCKER_CONFIG="$docker_config"

source_archive="$output_dir/source-archive.tar"
materialized_manifest="$output_dir/materialized-reliability-soak-v1.json"
correctness_golden="$output_dir/python-free-e2e-correctness-golden.json"
native_correctness_report="$output_dir/native-correctness-report.json"
snapshot_regular_file "$source_archive_input" "$source_archive"
snapshot_regular_file "$materialized_manifest_input" "$materialized_manifest"
snapshot_regular_file "$correctness_golden_input" "$correctness_golden"
snapshot_regular_file "$native_correctness_report_input" "$native_correctness_report"

model_snapshot="$output_dir/model-snapshot"
mkdir -m 0700 "$model_snapshot"
cp --recursive --no-preserve=mode,ownership,timestamps -- "$model_dir_input/." "$model_snapshot/"
if find "$model_snapshot" -mindepth 1 ! -type d ! -type f -print -quit | grep -q .; then
    echo "model snapshot contains a link or non-regular entry" >&2
    exit 1
fi
find "$model_snapshot" -type f -exec "$CHMOD_BIN" 0444 {} +
find "$model_snapshot" -type d -exec "$CHMOD_BIN" 0555 {} +
model_dir=$model_snapshot

archive_magic=$(od -An -tx1 -N4 "$source_archive" | tr -d ' \n')
case "$archive_magic" in
    1f8b*|425a*|fd377a58|28b52ffd|504b*)
        echo "source archive must be an uncompressed tar stream" >&2
        exit 1
        ;;
esac
archive_size=$(stat -c '%s' "$source_archive")
test "$archive_size" -gt 0 && ((archive_size % 512 == 0)) || {
    echo "source archive is not an uncompressed tar stream" >&2
    exit 1
}
archive_ustar_magic=$(od -An -tx1 -j257 -N6 "$source_archive" | tr -d ' \n')
test "$archive_ustar_magic" = 757374617200 || {
    echo "source archive lacks the uncompressed git-archive ustar header" >&2
    exit 1
}
tar --list --file "$source_archive" >/dev/null

test "$(sha256_file "$source_archive")" = "$expected_source_archive_sha256" || {
    echo "source archive differs from its trusted digest" >&2
    exit 1
}
archive_revision=$(git get-tar-commit-id <"$source_archive")
test "$archive_revision" = "$resolved_revision" || {
    echo "source archive does not bind the selected revision" >&2
    exit 1
}
canonical_manifest_member=benchmarks/soak/reliability-soak-v1.json
test "$(tar --list --file "$source_archive" | mawk -v expected="$canonical_manifest_member" '$0 == expected {count += 1} END {print count + 0}')" -eq 1
canonical_manifest="$output_dir/canonical-reliability-soak-v1.json"
tar --extract --to-stdout --file "$source_archive" -- "$canonical_manifest_member" >"$canonical_manifest"
chmod 0444 "$canonical_manifest"
test -x "$release_binary"
test "$(sha256_file "$release_binary")" = "$expected_release_binary_sha256" || {
    echo "standalone release binary differs from its trusted digest" >&2
    exit 1
}
test "$(sha256_file "$materialized_manifest")" = "$expected_manifest_sha256" || {
    echo "materialized soak manifest differs from its trusted digest" >&2
    exit 1
}
test "$(sha256_file "$correctness_golden")" = "$expected_correctness_golden_sha256" || {
    echo "E2E correctness golden differs from its independent trusted digest" >&2
    exit 1
}
test "$(sha256_file "$native_correctness_report")" = "$expected_native_correctness_report_sha256" || {
    echo "native correctness report differs from its independent trusted digest" >&2
    exit 1
}

jq -e '.schema_version == "riley.reliability-soak-manifest.v1" and .contract_id == "pr16-release-soak-v1"' \
    "$materialized_manifest" >/dev/null
normalized_template=$(jq -cS \
    '.golden.generated_sha256 = ("0" * 64) | .golden.provenance_sha256 = ("0" * 64)' \
    "$canonical_manifest")
normalized_materialized=$(jq -cS \
    '.golden.generated_sha256 = ("0" * 64) | .golden.provenance_sha256 = ("0" * 64)' \
    "$materialized_manifest")
test "$normalized_materialized" = "$normalized_template" || {
    echo "materialized manifest changes more than the reviewed golden digests" >&2
    exit 1
}
jq -e \
    '.golden.generated_sha256 | test("^[0-9a-f]{64}$") and . != ("0" * 64)' \
    "$materialized_manifest" >/dev/null
jq -e \
    '.golden.provenance_sha256 | test("^[0-9a-f]{64}$") and . != ("0" * 64)' \
    "$materialized_manifest" >/dev/null
test "$(jq -r '[.requests[].model] | unique | join("\n")' "$materialized_manifest")" = "$MODEL_ID"

# The materialized soak contract is not a trust anchor. Bind it to the actual
# independently reviewed E2E golden and native E0 report before the long run.
jq -e \
    'keys == ["config_sha256","correctness_gate_id","correctness_report_sha256","expected_greedy_text_sha256","max_tokens","model_id","model_revision","prompt","schema_version","source_revision","tokenizer_aggregate_sha256","tokenizer_json_sha256","weights_sha256"]' \
    "$correctness_golden" >/dev/null
jq -e \
    --arg gate "$NATIVE_CORRECTNESS_GATE" \
    --arg source_revision "$resolved_revision" \
    --arg model_id "$MODEL_ID" \
    --arg model_revision "$MODEL_REVISION" \
    --arg config_sha256 "$MODEL_CONFIG_SHA256" \
    --arg weights_sha256 "$MODEL_WEIGHTS_SHA256" \
    --arg tokenizer_aggregate_sha256 "$MODEL_TOKENIZER_AGGREGATE_SHA256" \
    --arg tokenizer_json_sha256 "$MODEL_TOKENIZER_JSON_SHA256" \
    --arg native_report_sha256 "$expected_native_correctness_report_sha256" \
    '.schema_version == "riley.python-free-release-e2e-golden.v1"
     and .correctness_gate_id == $gate
     and .correctness_report_sha256 == $native_report_sha256
     and .source_revision == $source_revision
     and .model_id == $model_id
     and .model_revision == $model_revision
     and .config_sha256 == $config_sha256
     and .weights_sha256 == $weights_sha256
     and .tokenizer_aggregate_sha256 == $tokenizer_aggregate_sha256
     and .tokenizer_json_sha256 == $tokenizer_json_sha256
     and (.prompt | type == "string" and length > 0 and length <= 16384
          and (contains("\n") | not) and (contains("\r") | not))
     and (.max_tokens | type == "number" and floor == . and . >= 2 and . <= 1024)
     and (.expected_greedy_text_sha256 | test("^[0-9a-f]{64}$"))' \
    "$correctness_golden" >/dev/null

empty_tree_sha256=$(printf '' | sha256sum | mawk '{print $1}')
jq -e \
    --arg schema "$NATIVE_CORRECTNESS_SCHEMA" \
    --arg gate "$NATIVE_CORRECTNESS_GATE" \
    --arg source_revision "$resolved_revision" \
    --arg clean_sha256 "$empty_tree_sha256" \
    --arg model_id "$MODEL_ID" \
    --arg model_revision "$MODEL_REVISION" \
    --arg config_sha256 "$MODEL_CONFIG_SHA256" \
    --arg weights_sha256 "$MODEL_WEIGHTS_SHA256" \
    --arg tokenizer_sha256 "$MODEL_TOKENIZER_AGGREGATE_SHA256" \
    '.schema_version == $schema
     and .gate_id == $gate
     and .status == "pass"
     and .bindings.candidate_git_revision == $source_revision
     and .bindings.candidate_git_status_sha256 == $clean_sha256
     and (.bindings.candidate_executable_sha256
          | type == "string" and test("^[0-9a-f]{64}$"))
     and .bindings.model_id == $model_id
     and .bindings.model_revision == $model_revision
     and .bindings.config_sha256 == $config_sha256
     and .bindings.weights_sha256 == $weights_sha256
     and .bindings.tokenizer_sha256 == $tokenizer_sha256' \
    "$native_correctness_report" >/dev/null

golden_generated_sha256=$(jq -er '.expected_greedy_text_sha256' "$correctness_golden")
golden_prompt=$(jq -er '.prompt' "$correctness_golden")
golden_max_tokens=$(jq -er '.max_tokens' "$correctness_golden")
golden_profile=$(jq -er '.golden.request_profile' "$materialized_manifest")
test "$(jq -er '.golden.generated_sha256' "$materialized_manifest")" = "$golden_generated_sha256" || {
    echo "soak generated digest does not match the independently reviewed E2E golden" >&2
    exit 1
}
test "$(jq -er '.golden.provenance_sha256' "$materialized_manifest")" = "$expected_native_correctness_report_sha256" || {
    echo "soak provenance digest does not hash the independently reviewed native report" >&2
    exit 1
}
test "$(jq -er --arg profile "$golden_profile" '.requests[$profile].model' "$materialized_manifest")" = "$MODEL_ID"
test "$(jq -er --arg profile "$golden_profile" '.requests[$profile].prompt' "$materialized_manifest")" = "$golden_prompt"
test "$(jq -er --arg profile "$golden_profile" '.requests[$profile].max_tokens' "$materialized_manifest")" = "$golden_max_tokens"
test "$(jq -er --arg profile "$golden_profile" '.requests[$profile].temperature' "$materialized_manifest")" = 0

test "$(sha256_file "$model_dir/config.json")" = "$MODEL_CONFIG_SHA256"
test "$(sha256_file "$model_dir/model.safetensors")" = "$MODEL_WEIGHTS_SHA256"
test "$(sha256_file "$model_dir/tokenizer.json")" = "$MODEL_TOKENIZER_JSON_SHA256"

resolved_release_image_id=$(docker image inspect --format '{{.Id}}' "$release_image_id")
test "$resolved_release_image_id" = "$release_image_id" || {
    echo "release image resolution differs from the trusted immutable ID" >&2
    exit 1
}
test "$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$release_image_id")" = linux/amd64
test "$(docker image inspect --format '{{.Config.User}}' "$release_image_id")" = 65532:65532 || {
    echo "release image does not retain the reviewed production USER 65532:65532" >&2
    exit 1
}
if docker image inspect "$test_image_tag" >/dev/null 2>&1; then
    echo "refusing to replace an existing reliability soak test image tag" >&2
    exit 1
fi

printf '%s\n' "$gpu_rows" >"$runtime_receipts/host-gpu.csv"
printf '%s\n' "$release_image_id" >"$output_dir/release-image-id.txt"
printf '%s\n' "$resolved_revision" >"$output_dir/source-revision.txt"
materialized_manifest_copy=$materialized_manifest
correctness_golden_copy=$correctness_golden
native_correctness_report_copy=$native_correctness_report

model_snapshot_pre_manifest="$output_dir/model-snapshot-SHA256SUMS.pre"
write_model_manifest "$model_snapshot" "$model_snapshot_pre_manifest"
test "$(sha256_file "$model_snapshot_pre_manifest")" = "$expected_model_tree_sha256" || {
    echo "model tree differs from its trusted canonical digest" >&2
    exit 1
}
if find "$model_snapshot" -type f ! -perm 0444 -print -quit | grep -q . \
    || find "$model_snapshot" -type d ! -perm 0555 -print -quit | grep -q .; then
    echo "model snapshot permissions are not fixed read-only" >&2
    exit 1
fi

verify_input_snapshots() {
    local stage=$1 stage_manifest="$output_dir/model-snapshot-SHA256SUMS.${stage}"
    test -f "$source_archive" && test ! -L "$source_archive"
    test -f "$materialized_manifest" && test ! -L "$materialized_manifest"
    test -f "$correctness_golden" && test ! -L "$correctness_golden"
    test -f "$native_correctness_report" && test ! -L "$native_correctness_report"
    test "$(stat -c '%a' "$source_archive")" = 444
    test "$(stat -c '%a' "$materialized_manifest")" = 444
    test "$(stat -c '%a' "$correctness_golden")" = 444
    test "$(stat -c '%a' "$native_correctness_report")" = 444
    test "$(sha256_file "$source_archive")" = "$expected_source_archive_sha256"
    test "$(sha256_file "$materialized_manifest")" = "$expected_manifest_sha256"
    test "$(sha256_file "$correctness_golden")" = "$expected_correctness_golden_sha256"
    test "$(sha256_file "$native_correctness_report")" = "$expected_native_correctness_report_sha256"
    if find "$model_snapshot" -mindepth 1 ! -type d ! -type f -print -quit | grep -q . \
        || find "$model_snapshot" -type f ! -perm 0444 -print -quit | grep -q . \
        || find "$model_snapshot" -type d ! -perm 0555 -print -quit | grep -q .; then
        echo "input snapshot type or permissions changed during $stage" >&2
        return 1
    fi
    write_model_manifest "$model_snapshot" "$stage_manifest"
    test "$(sha256_file "$stage_manifest")" = "$expected_model_tree_sha256"
    cmp --silent "$model_snapshot_pre_manifest" "$stage_manifest"
}

build_context="$output_dir/test-layer-build-context"
mkdir -m 0700 "$build_context"
context_members=(
    ci/release/ReliabilitySoak.Dockerfile
    ci/run_release_soak.sh
    benchmarks/soak/reliability-soak-v1.json
)
for context_member in "${context_members[@]}"; do
    context_count=$(tar --list --file "$source_archive" | mawk -v expected="$context_member" '$0 == expected {count += 1} END {print count + 0}')
    test "$context_count" -eq 1 || {
        echo "source archive must contain one regular $context_member" >&2
        exit 1
    }
done
tar --extract \
    --file "$source_archive" \
    --directory "$build_context" \
    --no-same-owner \
    --no-same-permissions \
    -- "${context_members[@]}"
if find "$build_context" -mindepth 1 ! -type d ! -type f -print -quit | grep -q .; then
    echo "minimal test-layer build context contains a link or special file" >&2
    exit 1
fi
expected_context_files=$(printf './%s\n' "${context_members[@]}" | sort)
actual_context_files=$(cd "$build_context" && find . -type f -print | sort)
test "$actual_context_files" = "$expected_context_files"
for context_member in "${context_members[@]}"; do
    test -f "$build_context/$context_member" && test ! -L "$build_context/$context_member"
done
find "$build_context" -type f -exec "$CHMOD_BIN" 0444 {} +
find "$build_context" -type d -exec "$CHMOD_BIN" 0555 {} +
build_context_pre_manifest="$output_dir/test-layer-build-context-SHA256SUMS.pre"
(
    cd "$build_context"
    sha256sum "${context_members[@]}"
) >"$build_context_pre_manifest"

docker image inspect "$release_image_id" >"$runtime_receipts/release-image-inspect.json"
jq -e \
    --arg image_id "$release_image_id" \
    --arg expected_path "/opt/riley/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --arg expected_ld_library_path "/usr/local/cuda/lib64" \
    'def environment_map:
       reduce (.[] | capture("^(?<name>[^=]+)=(?<value>.*)$")) as $item
         ({}; if has($item.name) then error("duplicate release environment name")
              else .[$item.name] = $item.value end);
     def forbidden($name):
       ($name == "BASH_ENV" or $name == "BASHOPTS" or $name == "CDPATH"
        or $name == "ENV" or $name == "GCONV_PATH" or $name == "GLOBIGNORE"
        or $name == "GLIBC_TUNABLES" or $name == "IFS" or $name == "LD_AUDIT"
        or $name == "LD_PRELOAD" or $name == "LOCPATH" or $name == "MALLOC_TRACE"
        or $name == "NLSPATH" or $name == "POSIXLY_CORRECT" or $name == "SHELLOPTS"
        or $name == "HOME" or $name == "CURL_HOME" or $name == "XDG_CONFIG_HOME"
        or $name == "PYTHONHOME"
        or $name == "PYTHONPATH" or $name == "CUDA_VISIBLE_DEVICES"
        or ($name | startswith("BASH_FUNC_")) or ($name | startswith("GIT_"))
        or ($name | startswith("DOCKER_")) or ($name | startswith("BUILDX_"))
        or (($name | startswith("LD_")) and $name != "LD_LIBRARY_PATH"));
     length == 1 and .[0] as $image
     | ($image.Config.Env | environment_map) as $environment
     | $image.Id == $image_id
       and $image.Os == "linux" and $image.Architecture == "amd64"
       and $image.Config.User == "65532:65532"
       and ($image.Config.WorkingDir | type) == "string"
       and (($image.Config.Labels == null) or
            (($image.Config.Labels | type) == "object" and
             all($image.Config.Labels[]; type == "string")))
       and $environment.PATH == $expected_path
       and $environment.NVIDIA_VISIBLE_DEVICES == "all"
       and $environment.NVIDIA_DRIVER_CAPABILITIES == "compute,utility"
       and $environment.LD_LIBRARY_PATH == $expected_ld_library_path
       and all($environment | keys[]; forbidden(.) | not)' \
    "$runtime_receipts/release-image-inspect.json" >/dev/null || {
    echo "release image has an unsafe or unexpected inherited build environment" >&2
    exit 1
}
release_working_directory=$(jq -er '.[0].Config.WorkingDir' "$runtime_receipts/release-image-inspect.json")
release_environment=$(jq -c \
    'def environment_map:
       reduce (.[] | capture("^(?<name>[^=]+)=(?<value>.*)$")) as $item
         ({}; if has($item.name) then error("duplicate release environment name")
              else .[$item.name] = $item.value end);
     .[0].Config.Env | environment_map' \
    "$runtime_receipts/release-image-inspect.json")
release_labels=$(jq -c '.[0].Config.Labels // {}' "$runtime_receipts/release-image-inspect.json")
test "${DOCKER_BUILDKIT:-1}" != 0 || {
    echo "the soak derivative requires Docker BuildKit" >&2
    exit 1
}
# BuildKit treats a raw local sha256 ID in FROM as a registry name. Create one
# deterministic, collision-closed local reference, while retaining the
# original image ID as the only provenance binding. The reference is kept for
# evidence replay and is checked before and after the build.
base_image_tag="riley-soak-release-base:${resolved_revision}-${release_image_id#sha256:}"
if docker image inspect "$base_image_tag" >/dev/null 2>&1; then
    test "$(docker image inspect --format '{{.Id}}' "$base_image_tag")" = "$release_image_id" || {
        echo "existing candidate/image base tag resolves to a different image ID" >&2
        exit 1
    }
    base_image_tag_state=reused
else
    docker image tag "$release_image_id" "$base_image_tag"
    base_image_tag_state=created
fi
resolved_base_image_id=$(docker image inspect --format '{{.Id}}' "$base_image_tag")
test "$resolved_base_image_id" = "$release_image_id"
printf '%s\n' "$base_image_tag" >"$output_dir/base-image-tag.txt"
printf '%s\n' "$base_image_tag_state" >"$output_dir/base-image-tag-state.txt"
docker image inspect "$base_image_tag" >"$output_dir/base-image-inspect-pre.json"

test "$(sha256_file "$source_archive")" = "$expected_source_archive_sha256"
build_context_immediate_manifest="$output_dir/test-layer-build-context-SHA256SUMS.immediate-pre-build"
(
    cd "$build_context"
    sha256sum "${context_members[@]}"
) >"$build_context_immediate_manifest"
cmp --silent "$build_context_pre_manifest" "$build_context_immediate_manifest"

DOCKER_BUILDKIT=1 docker build \
    --no-cache \
    --pull=false \
    --file "$build_context/ci/release/ReliabilitySoak.Dockerfile" \
    --build-arg "RILEY_RELEASE_IMAGE_REF=${base_image_tag}" \
    --build-arg "RILEY_RELEASE_IMAGE_ID=${release_image_id}" \
    --build-arg "RILEY_SOURCE_REVISION=${resolved_revision}" \
    --build-arg "RILEY_SOURCE_ARCHIVE_SHA256=${expected_source_archive_sha256}" \
    --build-arg "RILEY_RELEASE_BINARY_SHA256=${expected_release_binary_sha256}" \
    --tag "$test_image_tag" \
    "$build_context" 2>&1 | tee "$output_dir/test-layer-build.log"

post_build_base_image_id=$(docker image inspect --format '{{.Id}}' "$base_image_tag")
test "$post_build_base_image_id" = "$release_image_id" || {
    echo "candidate/image base tag changed while building the test layer" >&2
    exit 1
}
docker image inspect "$base_image_tag" >"$output_dir/base-image-inspect-post.json"

test_image_id=$(docker image inspect --format '{{.Id}}' "$test_image_tag")
[[ $test_image_id =~ ^sha256:[0-9a-f]{64}$ ]]
test "$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$test_image_id")" = linux/amd64
test "$(docker image inspect --format '{{.Config.User}}' "$test_image_id")" = 65532:65532
test "$(docker image inspect --format '{{index .Config.Labels "org.riley.reliability-soak.release-image-id"}}' "$test_image_id")" = "$release_image_id"
test "$(docker image inspect --format '{{index .Config.Labels "org.riley.reliability-soak.source-revision"}}' "$test_image_id")" = "$resolved_revision"
test "$(docker image inspect --format '{{index .Config.Labels "org.riley.reliability-soak.source-archive-sha256"}}' "$test_image_id")" = "$expected_source_archive_sha256"
test "$(docker image inspect --format '{{index .Config.Labels "org.riley.reliability-soak.release-binary-sha256"}}' "$test_image_id")" = "$expected_release_binary_sha256"
docker image inspect "$test_image_id" >"$runtime_receipts/test-layer-image-inspect.json"
printf '%s\n' "$test_image_id" >"$output_dir/test-layer-image-id.txt"
release_rootfs=$(jq -c '.[0].RootFS.Layers' "$runtime_receipts/release-image-inspect.json")
test_rootfs=$(jq -c '.[0].RootFS.Layers' "$runtime_receipts/test-layer-image-inspect.json")
jq -en --argjson release "$release_rootfs" --argjson test_layer "$test_rootfs" \
    '$release | type == "array" and length > 0
     and ($test_layer | type == "array" and length > ($release | length))
     and $test_layer[0:($release | length)] == $release' >/dev/null || {
    echo "derived test layer does not extend the exact release image rootfs" >&2
    exit 1
}
expected_test_image_environment=$(jq -cn \
    --argjson release "$release_environment" \
    '$release + {DEBIAN_FRONTEND:"noninteractive",LC_ALL:"C",TZ:"UTC"}')
expected_test_image_labels=$(jq -cn \
    --argjson release "$release_labels" \
    --arg release_image_id "$release_image_id" \
    --arg source_revision "$resolved_revision" \
    --arg source_archive_sha256 "$expected_source_archive_sha256" \
    --arg release_binary_sha256 "$expected_release_binary_sha256" \
    '$release + {
       "org.riley.reliability-soak.release-image-id":$release_image_id,
       "org.riley.reliability-soak.source-revision":$source_revision,
       "org.riley.reliability-soak.source-archive-sha256":$source_archive_sha256,
       "org.riley.reliability-soak.release-binary-sha256":$release_binary_sha256
     }')
jq -e --arg image_id "$test_image_id" \
    --arg working_directory "$release_working_directory" \
    --argjson expected_environment "$expected_test_image_environment" \
    --argjson expected_labels "$expected_test_image_labels" \
    'length == 1
     and (.[0].Config.Env |
       reduce (.[] | capture("^(?<name>[^=]+)=(?<value>.*)$")) as $item
         ({}; if has($item.name) then error("duplicate test environment name")
              else .[$item.name] = $item.value end)) == $expected_environment
     and .[0].Id == $image_id
     and .[0].Config.User == "65532:65532"
     and .[0].Config.WorkingDir == $working_directory
     and .[0].Config.Labels == $expected_labels
     and .[0].Config.Entrypoint == ["/opt/riley-soak/ci/run_release_soak.sh"]
     and .[0].Config.Cmd == []' \
    "$runtime_receipts/test-layer-image-inspect.json" >/dev/null
expected_container_environment=$(jq -cn \
    --arg source_revision "$resolved_revision" \
    --arg source_archive_sha256 "$expected_source_archive_sha256" \
    --arg release_binary_sha256 "$expected_release_binary_sha256" \
    --arg release_image_sha256 "${release_image_id#sha256:}" \
    --arg model_tree_sha256 "$expected_model_tree_sha256" \
    --arg model_id "$MODEL_ID" \
    --arg model_revision "$MODEL_REVISION" \
    --argjson image_environment "$expected_test_image_environment" \
    'def environment_map:
         reduce (.[] | capture("^(?<name>[^=]+)=(?<value>.*)$")) as $item
           ({}; if has($item.name) then error("duplicate image environment name")
                else .[$item.name] = $item.value end);
     $image_environment + {
       RILEY_SOAK_MANIFEST:"/run-input/reliability-soak-v1.json",
       RILEY_SOAK_OUTPUT:"/evidence/run",
       RILEY_SOURCE_REVISION:$source_revision,
       RILEY_SOURCE_ARCHIVE_SHA256:$source_archive_sha256,
       RILEY_BINARY_SHA256:$release_binary_sha256,
       RILEY_IMAGE_SHA256:$release_image_sha256,
       RILEY_MODEL_SHA256:$model_tree_sha256,
       RILEY_MODEL_ID:$model_id,
       RILEY_MODEL_REVISION:$model_revision,
       RILEY_SOAK_FINAL_METRICS_JSON:"/evidence/final-metrics.json",
       RILEY_SOAK_BINARY:"/opt/riley/bin/riley",
       RILEY_SOAK_MODEL_PATH:"/model",
       RILEY_SOAK_BIND:"127.0.0.1:18080",
       NVIDIA_DRIVER_CAPABILITIES:"compute,utility",
       ALL_PROXY:"",FTP_PROXY:"",HTTP_PROXY:"",HTTPS_PROXY:"",NO_PROXY:"",
       all_proxy:"",ftp_proxy:"",http_proxy:"",https_proxy:"",no_proxy:""
     }')

container_name="riley-soak-${resolved_revision:0:12}-$(date -u +%Y%m%dT%H%M%SZ)"
if docker inspect "$container_name" >/dev/null 2>&1; then
    echo "refusing to replace existing soak container: $container_name" >&2
    exit 1
fi
printf '%s\n' "$container_name" >"$output_dir/container-name.txt"

capture_interrupted_container() {
    local running
    if [[ -z ${active_container:-} ]]; then
        return
    fi
    running=$(docker inspect --format '{{.State.Running}}' "$active_container" 2>/dev/null || true)
    if [[ $running == true ]]; then
        docker stop --time 30 "$active_container" >/dev/null 2>&1 || true
    fi
    if [[ ! -e $output_dir/container-interrupted-inspect.json ]]; then
        docker inspect "$active_container" >"$output_dir/container-interrupted-inspect.json" 2>/dev/null || true
    fi
    if [[ ! -e $output_dir/container-interrupted.log ]]; then
        docker logs --timestamps "$active_container" >"$output_dir/container-interrupted.log" 2>&1 || true
    fi
}

handle_interruption() {
    local status=$1
    capture_interrupted_container
    active_container=
    trap - EXIT INT TERM
    exit "$status"
}

trap capture_interrupted_container EXIT
trap 'handle_interruption 130' INT
trap 'handle_interruption 143' TERM

validate_container_contract() {
    local inspect_path=$1 expected_status=$2 expected_exit_code=$3 post_run=$4
    jq -e \
        --arg id "$container_id" \
        --arg name "/$container_name" \
        --arg image "$test_image_id" \
        --arg user "65532:65532" \
        --arg status "$expected_status" \
        --argjson exit_code "$expected_exit_code" \
        --arg model_source "$model_snapshot" \
        --arg manifest_source "$materialized_manifest_copy" \
        --arg evidence_source "$container_evidence" \
        --arg gpu_uuid "$DESIGNATED_GPU_UUID" \
        --arg working_directory "$release_working_directory" \
        --argjson expected_labels "$expected_test_image_labels" \
        --argjson expected_environment "$expected_container_environment" \
        --argjson post_run "$post_run" \
        'def environment_map:
           reduce (.[] | capture("^(?<name>[^=]+)=(?<value>.*)$")) as $item
             ({}; if has($item.name) then error("duplicate container environment name")
                  else .[$item.name] = $item.value end);
         def docker_timestamp:
           capture("^(?<year>[0-9]{4})-(?<month>[0-9]{2})-(?<day>[0-9]{2})T(?<hour>[0-9]{2}):(?<minute>[0-9]{2}):(?<second>[0-9]{2})(?:\\.(?<fraction>[0-9]{1,9}))?Z$") as $time
           | ("\($time.year)-\($time.month)-\($time.day)T\($time.hour):\($time.minute):\($time.second)Z" | fromdateiso8601)
             + (("0." + ($time.fraction // "0")) | tonumber);
         length == 1 and .[0] as $container
         | $container.Id == $id
           and $container.Name == $name
           and $container.Image == $image
           and $container.Path == "/opt/riley-soak/ci/run_release_soak.sh"
           and $container.Args == []
           and (($container.Created | docker_timestamp) > 0)
           and $container.Config.Image == $image
           and $container.Config.User == $user
           and $container.Config.Entrypoint == ["/opt/riley-soak/ci/run_release_soak.sh"]
           and $container.Config.Cmd == []
           and $container.Config.WorkingDir == $working_directory
           and $container.Config.Healthcheck == {Test:["NONE"]}
           and $container.Config.Labels == $expected_labels
           and ($container.Config.Env | environment_map) == $expected_environment
           and $container.State.Status == $status
           and $container.State.Running == false
           and $container.State.Paused == false
           and $container.State.Restarting == false
           and $container.State.OOMKilled == false
           and $container.State.Dead == false
           and $container.State.Pid == 0
           and $container.State.ExitCode == $exit_code
           and $container.State.Error == ""
           and (if $post_run then
                  (($container.State.StartedAt | docker_timestamp) >= ($container.Created | docker_timestamp))
                  and (($container.State.FinishedAt | docker_timestamp) > ($container.State.StartedAt | docker_timestamp))
                  and ((($container.State.FinishedAt | docker_timestamp) - ($container.State.StartedAt | docker_timestamp)) >= 26100)
                else
                  $container.State.StartedAt == "0001-01-01T00:00:00Z"
                  and $container.State.FinishedAt == "0001-01-01T00:00:00Z"
                end)
           and $container.RestartCount == 0
           and $container.HostConfig.NetworkMode == "none"
           and $container.HostConfig.PidMode == "host"
           and $container.HostConfig.IpcMode == "private"
           and $container.HostConfig.UTSMode == ""
           and $container.HostConfig.UsernsMode == ""
           and $container.HostConfig.CgroupnsMode == "private"
           and $container.HostConfig.Runtime == "runc"
           and $container.HostConfig.ReadonlyRootfs == true
           and $container.HostConfig.AutoRemove == false
           and $container.HostConfig.Privileged == false
           and ($container.HostConfig.CapAdd // []) == []
           and $container.HostConfig.CapDrop == ["ALL"]
           and $container.HostConfig.SecurityOpt == ["no-new-privileges:true"]
           and $container.HostConfig.PidsLimit == 8192
           and $container.HostConfig.PublishAllPorts == false
           and ($container.HostConfig.PortBindings // {}) == {}
           and $container.HostConfig.RestartPolicy == {Name:"no",MaximumRetryCount:0}
           and $container.HostConfig.Binds == null
           and $container.HostConfig.DeviceCgroupRules == null
           and $container.HostConfig.Devices == []
           and $container.HostConfig.ExtraHosts == null
           and $container.HostConfig.GroupAdd == null
           and $container.HostConfig.Links == null
           and $container.HostConfig.Sysctls == null
           and $container.HostConfig.VolumesFrom == null
           and ($container.HostConfig.Tmpfs["/tmp"] | split(",") | sort)
               == ["nodev","noexec","nosuid","rw","size=67108864"]
           and ($container.HostConfig.DeviceRequests | length) == 1
           and ($container.HostConfig.DeviceRequests[0] | keys)
               == ["Capabilities","Count","DeviceIDs","Driver","Options"]
           and $container.HostConfig.DeviceRequests[0].Driver == ""
           and $container.HostConfig.DeviceRequests[0].Count == 0
           and $container.HostConfig.DeviceRequests[0].DeviceIDs == [$gpu_uuid]
           and $container.HostConfig.DeviceRequests[0].Capabilities == [["gpu"]]
           and $container.HostConfig.DeviceRequests[0].Options == {}
           and (($container.NetworkSettings.Networks // {}) | keys) == ["none"]
           and ([$container.Mounts[] | {Type,Source,Destination,Mode,RW,Propagation}] | sort_by(.Destination))
               == [
                    {Type:"bind",Source:$evidence_source,Destination:"/evidence",Mode:"",RW:true,Propagation:"rprivate"},
                    {Type:"bind",Source:$model_source,Destination:"/model",Mode:"",RW:false,Propagation:"rprivate"},
                    {Type:"bind",Source:$manifest_source,Destination:"/run-input/reliability-soak-v1.json",Mode:"",RW:false,Propagation:"rprivate"}
                  ]' \
        "$inspect_path" >/dev/null
}

container_id=$(docker create \
    --name "$container_name" \
    --restart no \
    --no-healthcheck \
    --user 65532:65532 \
    --network none \
    --pid host \
    --gpus "device=${DESIGNATED_GPU_UUID}" \
    --read-only \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --pids-limit 8192 \
    --tmpfs /tmp:rw,nosuid,nodev,noexec,size=67108864 \
    --env "RILEY_SOAK_MANIFEST=/run-input/reliability-soak-v1.json" \
    --env "RILEY_SOAK_OUTPUT=/evidence/run" \
    --env "RILEY_SOURCE_REVISION=${resolved_revision}" \
    --env "RILEY_SOURCE_ARCHIVE_SHA256=${expected_source_archive_sha256}" \
    --env "RILEY_BINARY_SHA256=${expected_release_binary_sha256}" \
    --env "RILEY_IMAGE_SHA256=${release_image_id#sha256:}" \
    --env "RILEY_MODEL_SHA256=${expected_model_tree_sha256}" \
    --env "RILEY_MODEL_ID=${MODEL_ID}" \
    --env "RILEY_MODEL_REVISION=${MODEL_REVISION}" \
    --env "RILEY_SOAK_FINAL_METRICS_JSON=/evidence/final-metrics.json" \
    --env "RILEY_SOAK_BINARY=/opt/riley/bin/riley" \
    --env "RILEY_SOAK_MODEL_PATH=/model" \
    --env "RILEY_SOAK_BIND=127.0.0.1:18080" \
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
    --mount "type=bind,source=${model_snapshot},destination=/model,readonly" \
    --mount "type=bind,source=${materialized_manifest_copy},destination=/run-input/reliability-soak-v1.json,readonly" \
    --mount "type=bind,source=${container_evidence},destination=/evidence" \
    "$test_image_id")
[[ $container_id =~ ^[0-9a-f]{64}$ ]]
printf '%s\n' "$container_id" >"$output_dir/container-id.txt"
active_container=$container_name
docker inspect "$container_name" >"$runtime_receipts/container-inspect-pre.json"
validate_container_contract "$runtime_receipts/container-inspect-pre.json" created 0 false

container_binary_copy="$output_dir/container-riley"
docker cp "$container_name:/opt/riley/bin/riley" "$container_binary_copy"
test -f "$container_binary_copy" && test ! -L "$container_binary_copy"
test "$(sha256_file "$container_binary_copy")" = "$expected_release_binary_sha256" || {
    echo "actual container binary differs from the reviewed release binary" >&2
    exit 1
}
chmod 0444 "$container_binary_copy"

# This receipt was generated by the source-bound Dockerfile's pre/post apt
# closure comparison. Copy it from the actual created container before start,
# then enforce the same canonical format consumed by the offline checker.
runtime_closure_receipt="$runtime_receipts/release-runtime-closure.tsv"
docker cp \
    "$container_name:/opt/riley-soak/release-runtime-closure.tsv" \
    "$runtime_closure_receipt"
test -f "$runtime_closure_receipt" && test ! -L "$runtime_closure_receipt"
test "$(stat -c '%a' "$runtime_closure_receipt")" = 444
mawk -F '\t' '
    function normalized_absolute(path) {
        return substr(path, 1, 1) == "/" \
            && substr(path, 1, 2) != "//" \
            && path !~ /\/\.\.?(\/|$)/
    }
    NF != 4 { exit 1 }
    $1 !~ /^[A-Za-z0-9_+.\/-]+$/ { exit 1 }
    $2 == "NOT_FOUND" {
        if ($1 != "libcuda.so.1" || $3 != "-" || $4 != "-") exit 1
        unresolved += 1
        rows += 1
        next
    }
    !normalized_absolute($2) || !normalized_absolute($3) { exit 1 }
    length($4) != 64 || $4 ~ /[^0-9a-f]/ { exit 1 }
    substr($1, 1, 1) == "/" {
        if ($1 != $2 || index($1, "ld-linux") == 0) exit 1
        loaders += 1
    }
    { rows += 1 }
    END { if (rows < 1 || rows > 1024 || loaders != 1 || unresolved != 1) exit 1 }
' "$runtime_closure_receipt"
sort -u "$runtime_closure_receipt" | cmp --silent - "$runtime_closure_receipt"
release_runtime_closure_sha256=$(sha256_file "$runtime_closure_receipt")

verify_input_snapshots immediate-pre-start
require_gpu_idle immediate-pre-start
docker start "$container_name" >/dev/null
docker wait "$container_name" >"$output_dir/container-exit-code.txt"
docker inspect "$container_name" >"$runtime_receipts/container-inspect-post.json"
docker logs --timestamps "$container_name" >"$output_dir/container.log" 2>&1
container_evidence_export="$output_dir/container-evidence-export"
mkdir -m 0700 "$container_evidence_export"
docker cp "$container_name:/evidence/." "$container_evidence_export"
active_container=

container_status=$(<"$output_dir/container-exit-code.txt")
[[ $container_status =~ ^[0-9]+$ ]]
if ((container_status != 0)); then
    echo "release soak container failed with status $container_status; preserved as $container_name" >&2
    exit "$container_status"
fi
validate_container_contract "$runtime_receipts/container-inspect-post.json" exited 0 true
jq -e -s \
    'length == 2 and all(.[]; length == 1)
     and .[0][0] as $pre
     | .[1][0] as $post
     | ["Id","Name","Image","Path","Args","Created","Config","HostConfig","Mounts"]
     | all(. as $field | $pre[$field] == $post[$field])' \
    "$runtime_receipts/container-inspect-pre.json" \
    "$runtime_receipts/container-inspect-post.json" >/dev/null

verify_input_snapshots post

test -f "$container_evidence_export/run/run.json"
test -f "$container_evidence_export/run/events.jsonl"
test -f "$container_evidence_export/final-metrics.json"
test "$(jq -r '.source.git_commit' "$container_evidence_export/run/run.json")" = "$resolved_revision"
test "$(jq -r '.source.source_archive_sha256' "$container_evidence_export/run/run.json")" = "$expected_source_archive_sha256"
test "$(jq -r '.source.binary_sha256' "$container_evidence_export/run/run.json")" = "$expected_release_binary_sha256"
test "$(jq -r '.source.image_sha256' "$container_evidence_export/run/run.json")" = "${release_image_id#sha256:}"
test "$(jq -r '.source.model_sha256' "$container_evidence_export/run/run.json")" = "$expected_model_tree_sha256"
jq -e 'select(.kind == "run_end" and .status == "success")' \
    < <(tail -n 1 "$container_evidence_export/run/events.jsonl") >/dev/null
run_json_sha256=$(sha256_file "$container_evidence_export/run/run.json")
events_jsonl_sha256=$(sha256_file "$container_evidence_export/run/events.jsonl")
require_exact_clean_checkout

jq -nS \
    --arg hostname "$actual_hostname" \
    --arg gpu_name "$actual_gpu_name" \
    --arg gpu_uuid "$actual_gpu_uuid" \
    --arg compute_capability "$actual_compute_capability" \
    --argjson memory_total_mib "$((10#$actual_memory_total_mib))" \
    --arg driver_version "$actual_driver_version" \
    --arg git_revision "$resolved_revision" \
    --arg source_archive_sha256 "$expected_source_archive_sha256" \
    --arg release_binary_sha256 "$expected_release_binary_sha256" \
    --arg model_tree_sha256 "$expected_model_tree_sha256" \
    --arg manifest_sha256 "$expected_manifest_sha256" \
    --arg correctness_golden_sha256 "$expected_correctness_golden_sha256" \
    --arg native_correctness_report_sha256 "$expected_native_correctness_report_sha256" \
    --arg run_json_sha256 "$run_json_sha256" \
    --arg events_jsonl_sha256 "$events_jsonl_sha256" \
    --arg release_runtime_closure_sha256 "$release_runtime_closure_sha256" \
    --arg release_image_id "$release_image_id" \
    --arg test_layer_image_id "$test_image_id" \
    --arg container_id "$container_id" \
    --arg container_name "$container_name" \
    --argjson container_exit_code "$container_status" \
    '{schema_version:"riley.reliability-soak-launcher-receipt.v3",
      host:{hostname:$hostname,gpu_name:$gpu_name,gpu_uuid:$gpu_uuid,compute_capability:$compute_capability,memory_total_mib:$memory_total_mib,driver_version:$driver_version},
      source:{git_revision:$git_revision,source_archive_sha256:$source_archive_sha256,release_binary_sha256:$release_binary_sha256,model_tree_sha256:$model_tree_sha256,manifest_sha256:$manifest_sha256,correctness_golden_sha256:$correctness_golden_sha256,native_correctness_report_sha256:$native_correctness_report_sha256},
      evidence:{run_json_sha256:$run_json_sha256,events_jsonl_sha256:$events_jsonl_sha256,release_runtime_closure_sha256:$release_runtime_closure_sha256},
      images:{release_image_id:$release_image_id,test_layer_image_id:$test_layer_image_id},
      container:{id:$container_id,name:$container_name,exit_code:$container_exit_code}}' \
    >"$runtime_receipts/launcher-receipt.json"
jq -e \
    'keys == ["container","evidence","host","images","schema_version","source"]
     and .schema_version == "riley.reliability-soak-launcher-receipt.v3"
     and (.host | keys) == ["compute_capability","driver_version","gpu_name","gpu_uuid","hostname","memory_total_mib"]
     and (.source | keys) == ["correctness_golden_sha256","git_revision","manifest_sha256","model_tree_sha256","native_correctness_report_sha256","release_binary_sha256","source_archive_sha256"]
     and (.evidence | keys) == ["events_jsonl_sha256","release_runtime_closure_sha256","run_json_sha256"]
     and (.images | keys) == ["release_image_id","test_layer_image_id"]
     and (.container | keys) == ["exit_code","id","name"]
     and (.host.memory_total_mib | type) == "number"
     and (.container.exit_code | type) == "number"
     and ([.host.hostname,.host.gpu_name,.host.gpu_uuid,.host.compute_capability,.host.driver_version,
           .source.git_revision,.source.source_archive_sha256,.source.release_binary_sha256,
           .source.model_tree_sha256,.source.manifest_sha256,.source.correctness_golden_sha256,
           .source.native_correctness_report_sha256,.evidence.run_json_sha256,.evidence.events_jsonl_sha256,
           .evidence.release_runtime_closure_sha256,
           .images.release_image_id,.images.test_layer_image_id,
           .container.id,.container.name] | all(type == "string"))' \
    "$runtime_receipts/launcher-receipt.json" >/dev/null

expected_runtime_receipts=$(printf '%s\n' \
    container-inspect-post.json \
    container-inspect-pre.json \
    host-gpu.csv \
    launcher-receipt.json \
    release-image-inspect.json \
    release-runtime-closure.tsv \
    test-layer-image-inspect.json | sort)
if find "$runtime_receipts" -mindepth 1 ! -type f -print -quit | grep -q .; then
    echo "runtime receipt directory contains a link or special entry" >&2
    exit 1
fi
actual_runtime_receipts=$(cd "$runtime_receipts" && find . -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | sort)
test "$actual_runtime_receipts" = "$expected_runtime_receipts"
test "$(wc -l <"$runtime_receipts/host-gpu.csv" | mawk '{$1=$1;print}')" -eq 1
test "$(<"$runtime_receipts/host-gpu.csv")" = "$gpu_rows"
chmod 0444 "$runtime_receipts"/*

(
    cd "$output_dir"
    sha256sum \
        materialized-reliability-soak-v1.json \
        python-free-e2e-correctness-golden.json \
        native-correctness-report.json \
        source-archive.tar \
        test-layer-build-context-SHA256SUMS.pre \
        test-layer-build-context-SHA256SUMS.immediate-pre-build \
        model-snapshot-SHA256SUMS.pre \
        model-snapshot-SHA256SUMS.immediate-pre-start \
        model-snapshot-SHA256SUMS.post \
        container-riley \
        container-evidence-export/run/run.json \
        container-evidence-export/run/events.jsonl \
        container-evidence-export/final-metrics.json \
        runtime-receipts/host-gpu.csv \
        runtime-receipts/launcher-receipt.json \
        runtime-receipts/release-runtime-closure.tsv \
        runtime-receipts/release-image-inspect.json \
        runtime-receipts/test-layer-image-inspect.json \
        runtime-receipts/container-inspect-pre.json \
        runtime-receipts/container-inspect-post.json \
        >launcher-SHA256SUMS
)
trap - EXIT INT TERM
printf '%s\n' riley.remote-release-soak.completed.v1 >"$output_dir/completed"

echo "release soak completed; persistent container: $container_name"
echo "release soak evidence: $output_dir"
