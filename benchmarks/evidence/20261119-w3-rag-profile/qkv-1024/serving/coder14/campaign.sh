#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
prior="$evidence/w3-qkv-serving-20261119.MhydzNsX"
warm="$evidence/w5-metrics-warm-20261120.tLp6iwIe"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
model="$root/models/Qwen2.5-Coder-14B-Instruct"
python=/usr/bin/python3
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$prior/campaign.exit")" = 0
(cd "$prior" && sha256sum -c "$stage/reused.sha256" binary.sha256) >prior-check.log
(cd "$model" && sha256sum -c "$stage/model.sha256") >model-check.log
"$python" -B - "$model" <<'PY' >geometry.json
import json
from pathlib import Path
import sys
model = Path(sys.argv[1])
config = json.loads((model/"config.json").read_bytes())
checkpoint = json.loads((model/"riley-checkpoint.json").read_bytes())
q, kv, k = config["num_attention_heads"], config["num_key_value_heads"], config["hidden_size"]
d = config.get("head_dim", k//q)
n = (q+2*kv)*d
assert (q,kv,k,d,n) == (40,8,5120,128,7168)
print(json.dumps({"model":checkpoint["source_model"],"revision":checkpoint["source_revision"],
                  "gemm":{"m":1024,"n":n,"k":k},"query_heads":q,"kv_heads":kv,"head_dim":d}, indent=2))
PY
mkdir -p tools target/release
cp -- "$prior/tools/"{run_profile_retry,profile_workload,load,profile_summary,profile_process,compare_runner}.py tools/
cp -- "$prior/holder.py" .
ln -- "$prior/target/release/riley" target/release/riley
cp -- "$prior/binary.sha256" .
sha256sum -c binary.sha256
printf 'WAIT %s %s\n' "$(basename "$warm")" "$(date -u +%FT%TZ)"
deadline=$((SECONDS+3600))
until [[ -s "$warm/campaign.exit" ]]; do
    if ! ps -p 2883662 -o args= | grep -Fq -- "$warm/campaign.sh"; then
        [[ -s "$warm/campaign.exit" ]] && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 20
done
printf 'PREDECESSOR_TERMINAL %s %s\n' "$(basename "$warm")" "$(cat "$warm/campaign.exit")"

status() {
    local index=$1 output=$2
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status >"$output"
    "$python" -B - "$index" "$output" <<'PY'
from pathlib import Path
import json
import sys
index=int(sys.argv[1])
pid,ticks={6:(2431228,"6339570869"),7:(2431237,"6339570875")}[index]
x=json.loads(Path(sys.argv[2]).read_bytes())
assert (x["physical_index"],x["pid"],x["start_ticks"],x["phase"])==(index,pid,ticks,"idle")
assert x["allocation_bytes"]==x["torch_reserved_bytes"]==20*1024**3 and not x["other_processes"]
PY
}
for mode in riley-first qkv-first qkv-second riley-second; do
    status 6 "gpu6-before-$mode.json"
    status 7 "idle-before-$mode.json"
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    rc=0
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
         "$python" -B "$stage/tools/run_profile_retry.py" coder14 "$mode" >"$mode.controller.log" 2>&1 || rc=$?
    mkdir -p "$stage/coder14-$mode"
    status 6 "coder14-$mode/gpu6-after.json"
    status 7 "coder14-$mode/idle-after.json"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
printf '0\n' >measurements.exit
"$python" -B compare_coder.py "$stage" verified.json
sha256sum verified.json measurements.exit geometry.json binary.sha256 \
    coder14-*/{receipt,plan,server-exit,idle-after,gpu6-after,token-id-probe,rag-c1,rag-c8-s1,rag-c8-s2,wave-sync,wave-50ms,short-c32}.json \
    {gpu6-before,idle-before}-*.json >result.sha256
