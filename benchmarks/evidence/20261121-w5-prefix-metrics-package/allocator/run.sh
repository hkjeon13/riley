#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
stock="$evidence/w5-prefix-metrics-package-20261121.nEFyze5I"
run="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
prior="$evidence/w5-current-package-20261121.f6SEbjvA"
rag="$evidence/w5-current-rag-20261121.hvh3VS67"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
golden="$base/src/benchmarks/evidence"
runner="$(dirname "$evidence")/tools/vllm-venv/bin/python"
cpu="$stock/cpu-evidence/report.json"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c allocator-dispatch.sha256 build-result.sha256
test "$(cat build.exit)" = 0
while ! test -f "$run/campaign.exit"; do
    ps -p "$(cat "$run/campaign.pid")" -o stat= | grep -Eq '^[^Z]+$'
    sleep 5
done
test "$(cat "$run/campaign.exit")" = 0
(cd "$run" && sha256sum --strict -c dispatch.sha256 runtime-tools.sha256 result.sha256 >/dev/null)
(cd "$stock" && test "$(cat build.exit)" = 0 && sha256sum --strict -c build-result.sha256 >/dev/null)
export PYTHONPATH="$run/harness/ci:$stock/src/ci:$rag:$prior/src/ci"
"$runner" -B verify.py "$run" >stock-verified.json
"$runner" -B - <<'PY'
from pathlib import Path
from verify import read, require
soaks = read(Path("stock-verified.json"))["soaks"]
if soaks["release_gate"] == "passed":
    print("Stock RSS qualification passed. No allocator GPU trial is needed.")
    raise SystemExit(78)
require(all(
    failure == "rss_growth_second_half_at_most_zero"
    for row in soaks["modes"].values() for failure in row["failed_checks"]
), "only a measured strict RSS miss calls for this allocator trial")
PY
printf '%s  %s\n' b32bd641dab4e9516a8d3b45de2d77fbde91ad64acad893153f4b6c2090b72a6 "$golden/20261119-w4-long/riley/capture.json.gz" \
    220f13a7306518f996480c608733623475471ff3721ea3aeb895a2d5f91abd11 "$golden/20261119-w4-holdout/holdout.json.gz" \
    12d508ed51f03509432e6302b8d10855f56125f4f6b7f31a559cff03eb10b611 "$golden/20261119-w4-holdout/riley/capture.json.gz" | sha256sum --strict -c
"$runner" -B allocator.py bind "$stage" "$cpu" >gpu-checker-input.json

reservations() {
    local phase=$1
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"idle-$phase.json"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status >"gpu7-$phase.json"
    "$runner" -B - "$phase" <<'PY'
import sys
from pathlib import Path
from verify import check_reservations
check_reservations(Path.cwd(), sys.argv[1])
PY
}

run_arm() {
    local label=$1 index=$2 rc=0
    printf 'START %s %s %s\n' "$label" "$index" "$(date -u +%FT%TZ)"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" run -- \
        "$runner" -B allocator.py run "$stage" "$cpu" "$base" "$label" >"$label.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" >"$label.exit"
    printf 'END %s %s %s exit=%d\n' "$label" "$index" "$(date -u +%FT%TZ)" "$rc"
    test -s "$label/report.json"
}

reservations before
run_arm cache-off 6 &
off=$!
run_arm cache-on 7 &
on=$!
rc=0
wait "$off" || rc=1
wait "$on" || rc=1
reservations after-soaks
test "$rc" = 0
run_arm accuracy 6
reservations after
test "$(cat accuracy.exit)" = 0
"$runner" -B allocator.py verify "$stage" "$cpu" >verified.json
sha256sum verified.json stock-verified.json gpu-checker-input.json \
    {cache-off,cache-on,accuracy}.exit {cache-off,cache-on,accuracy}/{report,api,shutdown}.json \
    cache-{off,on}/soak.json {cache-off,cache-on,accuracy}-allocator-{before,after}.json \
    idle-{before,after-soaks,after}.json gpu7-{before,after-soaks,after}.json >result.sha256
