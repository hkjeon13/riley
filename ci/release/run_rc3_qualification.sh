#!/usr/bin/env bash
# Finalize one externally frozen RC3 qualification decision.
#
# This wrapper deliberately does not SSH, launch CUDA, or rerun a partial gate.
# Execute the existing remote producers on server-4096 first, then run this
# once from the exact clean candidate checkout.  The fresh decision directory
# makes a failed candidate terminal: use a new freeze for another attempt.

set -euo pipefail
umask 077
IFS=$' \t\n'

usage() {
    /bin/cat <<'EOF'
usage: bash ci/release/run_rc3_qualification.sh \
  --freeze /absolute/path/riley-X.Y.Z-rcN.freeze.json \
  --expected-candidate-sha256 LOWERCASE_SHA256 \
  --evidence-root /absolute/path/existing-evidence-root \
  --decision-dir /absolute/path/new-empty-decision-directory
EOF
}

freeze=''
expected_candidate_sha256=''
evidence_root=''
decision_dir=''

while (($#)); do
    case "$1" in
        --freeze|--expected-candidate-sha256|--evidence-root|--decision-dir)
            (($# >= 2)) || { usage >&2; exit 2; }
            case "$1" in
                --freeze)
                    [[ -z ${freeze} ]] || { echo 'duplicate --freeze' >&2; exit 2; }
                    freeze=$2
                    ;;
                --expected-candidate-sha256)
                    [[ -z ${expected_candidate_sha256} ]] || {
                        echo 'duplicate --expected-candidate-sha256' >&2; exit 2;
                    }
                    expected_candidate_sha256=$2
                    ;;
                --evidence-root)
                    [[ -z ${evidence_root} ]] || { echo 'duplicate --evidence-root' >&2; exit 2; }
                    evidence_root=$2
                    ;;
                --decision-dir)
                    [[ -z ${decision_dir} ]] || { echo 'duplicate --decision-dir' >&2; exit 2; }
                    decision_dir=$2
                    ;;
            esac
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n ${freeze} && -n ${expected_candidate_sha256} && -n ${evidence_root} && -n ${decision_dir} ]] || {
    usage >&2
    exit 2
}
[[ ${freeze} == /* && ${evidence_root} == /* && ${decision_dir} == /* ]] || {
    echo 'freeze, evidence root, and decision directory must be absolute paths' >&2
    exit 2
}
[[ ${expected_candidate_sha256} =~ ^[0-9a-f]{64}$ ]] || {
    echo 'expected candidate SHA-256 must be lowercase hexadecimal' >&2
    exit 2
}
[[ -f ${freeze} && ! -L ${freeze} ]] || {
    echo 'freeze must be a regular non-symlink file' >&2
    exit 2
}
[[ -d ${evidence_root} && ! -L ${evidence_root} ]] || {
    echo 'evidence root must be a real directory' >&2
    exit 2
}
[[ ! -e ${decision_dir} && ! -L ${decision_dir} ]] || {
    echo 'decision directory must not already exist; RC3 decisions are create-only' >&2
    exit 2
}

repository_root=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo 'must run from a Git candidate checkout' >&2
    exit 2
}
repository_root=$(cd -- "${repository_root}" && pwd -P)
[[ -f ${repository_root}/ci/release/check_rc3_qualification.py ]] || {
    echo 'candidate checkout is missing the RC3 checker' >&2
    exit 2
}

# mkdir itself is the create-only claim.  Do not remove this directory after a
# failure: it is the durable decision boundary for this candidate.
mkdir -- "${decision_dir}"
report=${decision_dir}/rc3-qualification.json

python3 "${repository_root}/ci/release/check_rc3_qualification.py" \
    --freeze "${freeze}" \
    --expected-candidate-sha256 "${expected_candidate_sha256}" \
    --evidence-root "${evidence_root}" \
    --repository-root "${repository_root}" \
    --report "${report}"
