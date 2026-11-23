#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
root=$(dirname "$(dirname "$stage")")
previous="$root/evidence/w3-rag-profile-20261119.FE1ZUFeE"
w5="$root/evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
package="$root/evidence/w4-schema-ids-20261123.ZsUYnqkm"
holder="$root/evidence/w2-sm17-20261118.6UMfeHh2"
first="$root/evidence/w3-package-coder14-20261124.8hMR98cC"
runner="$root/tools/vllm-venv/bin/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$root" && sha256sum --strict -c "$stage/pins.sha256")
(cd "$w5" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
test "$(cat "$package/campaign.exit")" = 0
mkdir tools gpu6 gpu7
cp "$previous/holder.py" .
cp "$previous"/tools/{run_profile,profile_workload,profile_summary,load,compare_runner}.py tools/
python3 -B - <<'PY'
from pathlib import Path
p = Path("tools/run_profile.py")
source = p.read_text()
lease = '/ "gpu7.status.json"'
budget = '"--max-num-seqs", "32",'
assert source.count(lease) == source.count(budget) == 1
p.write_text(source.replace(lease, '''/ f"gpu{env['W3_GPU_INDEX']}.status.json"''')
             .replace(budget, budget + ' "--max-num-batched-tokens", "2048",'))
PY
sha256sum holder.py tools/*.py >runtime.sha256
"$runner" -B -c 'from importlib.metadata import version; assert version("vllm") == "0.27.1+cu129"; print(version("vllm"))' >vllm-version.txt
# The first attempt verified the full checkpoint before its driver import failed.
(cd "$first" && sha256sum --strict -c result.sha256 >/dev/null)
test "$(cat "$first/campaign.exit")" = 1
test "$(cat "$first/checkpoint.log")" = "checkpoint verify: verified"
cp "$first/checkpoint.log" checkpoint.log
sha256sum "$first"/{campaign.exit,dispatch.sha256,result.sha256,checkpoint.log,gpu6/riley.log,gpu7/vllm.log} >first-attempt.sha256
export PYTHONPATH="$stage/tools:$w5/harness/ci:$package/src/ci"
export PYTHONDONTWRITEBYTECODE=1
"$runner" -B -c 'import run_profile, check_distribution_gpu'

reservations() {
    for index in 6 7; do
        "${control[@]}" --index "$index" status >"gpu$index/$1.json"
    done
}

arm() {
    local index=$1 engine=$2 rc=0
    printf 'START gpu%s %s %s\n' "$index" "$engine" "$(date -u +%FT%TZ)"
    W3_GPU_INDEX="$index" "${control[@]}" --index "$index" run -- \
        "$runner" -B arm.py "$engine" >"gpu$index/$engine.log" 2>&1 || rc=$?
    printf 'END gpu%s %s %s exit=%s\n' "$index" "$engine" "$(date -u +%FT%TZ)" "$rc"
    return "$rc"
}

round() {
    local left right rc=0
    arm 6 "$1" &
    left=$!
    arm 7 "$2" &
    right=$!
    wait "$left" || rc=1
    wait "$right" || rc=1
    return "$rc"
}

reservations before
rc=0
if round riley vllm; then
    reservations between
    round vllm riley || rc=1
else
    rc=1
fi
reservations after
(cd "$root" && sha256sum --strict -c "$stage/pins.sha256")
find gpu6 gpu7 -type f \( -name '*.json' -o -name '*.log' \) \
    ! -path '*/vllm-cache/*' ! -path '*/torch-cache/*' -print0 |
    sort -z | xargs -0 sha256sum >result.sha256
sha256sum checkpoint.log vllm-version.txt runtime.sha256 first-attempt.sha256 >>result.sha256
exit "$rc"
