#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$stage")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
preceding="$evidence/w4-hf-mlp-input-20261119.rOd2VDeo"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
cuda=/usr/local/cuda-12.4
trap 'printf "%d\n" "$?" >"$stage/campaign.exit.tmp"; mv -- "$stage/campaign.exit.tmp" "$stage/campaign.exit"' EXIT
cd "$stage"
sha256sum -c dispatch.sha256
cp -- "$base/src/benchmarks/profiling/gemm_anchor_search.cu" .
sha256sum -c source-base.sha256
git apply --check qkv1024.patch
git apply qkv1024.patch
sha256sum -c source.sha256
/usr/bin/python3 -B - "$base/src/kernels/src/gemm.cu" <<'PY' >pinned-config.json
from hashlib import sha256
from pathlib import Path
import json, re, sys

raw = Path(sys.argv[1]).read_bytes()
source = raw.decode()
anchors = source.split("constexpr AnchorOverride kAnchorOverrides[] = {", 1)[1].split("};", 1)[0]
exact = [[int(n) for n in re.findall(r"\d+", line)] for line in anchors.splitlines()
         if "{80, 5120, 7168, " in line]
wide = [[int(n) for n in re.findall(r"\d+", line)] for line in source.splitlines()
        if "{80, 5120, 7168, 513, 1024, " in line]
assert exact == [[80, 5120, 7168, 128, 21, 18, 18, 1, 0]]
assert wide == [[80, 5120, 7168, 513, 1024, 21, 20, 11, 3, 1]]
print(json.dumps({"gemm_source_sha256": sha256(raw).hexdigest(), "anchor": exact[0], "wide_table_entry": wide[0]}))
PY
export LD_LIBRARY_PATH="$cuda/lib64"
"$cuda/bin/nvcc" --version >toolchain.txt
nice -n 15 "$cuda/bin/nvcc" -O3 -std=c++17 -arch=sm_80 gemm_anchor_search.cu -lcublasLt -o qkv1024 >build.log 2>&1
sha256sum qkv1024 >binary.sha256
printf 'WAIT %s %s\n' "$preceding" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 14400))
until test -s "$preceding/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$preceding/campaign.pid")" | grep -Fq -- "$preceding/campaign.sh"; then
        test -s "$preceding/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 30
done
status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status
}
status >idle-before.json
printf 'START %s\n' "$(date -u +%FT%TZ)"
rc=0
/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
    /usr/bin/timeout 420s /usr/bin/env RILEY_SEARCH_WORKSPACE_MIB=64 "$stage/qkv1024" 5120 7168 qkv1024 \
    >probe.log 2>probe.err || rc=$?
status >idle-after.json
/usr/bin/python3 -B - <<'PY'
import json
from pathlib import Path
before, after = [json.loads(Path(f"idle-{phase}.json").read_bytes()) for phase in ("before", "after")]
keys = ("pid", "start_ticks", "gpu", "physical_index", "phase", "allocation_bytes", "torch_reserved_bytes")
assert all(before[key] == after[key] for key in keys)
assert after["physical_index"] == 7 and after["phase"] == "idle" and not after["other_processes"]
assert after["allocation_bytes"] == after["torch_reserved_bytes"] == 20 * 1024**3
PY
printf 'END %s exit=%d\n' "$(date -u +%FT%TZ)" "$rc"
if (( rc != 0 )); then exit "$rc"; fi
/usr/bin/python3 -B summarize_probe.py probe.log >result.json
sha256sum result.json pinned-config.json binary.sha256 probe.{log,err} idle-{before,after}.json >result.sha256
