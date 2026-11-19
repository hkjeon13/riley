#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
root=$(dirname "$stage")
base="$root/w1-dist-d3d8fe6e5.YnZWdh5l"
harness="$root/w5-dist-off-retry.ORBlgo3W/harness"
image=sha256:0a907408ec73fc8b9b00f440037305ce1bfb00e05bd8d3ef3096fcee59849ea1
library=/tmp/riley-w5-libjemalloc.so.2
probe=/tmp/riley-w5-allocator-probe
startup=/tmp/riley-w5-start
conf=abort_conf:true,dirty_decay_ms:0,muzzy_decay_ms:0,oversize_threshold:0
sha256sum -c dispatch.sha256
printf '%s  %s\n' \
    144eaa64b9da0319760c407ebe330b46d9d03ef9952fc067d48221cc9832e45c libjemalloc2_5.2.1-4ubuntu1_amd64.deb \
    c769f7591006b37ac8893a69a82fc40d5ac831e8e109670878c4e4d44c8a6ba9 libjemalloc.so.2 \
    92f884b05c1101d4bfb6c49df9289483b698e4e64d88fdc1090705998a17de33 "$base/cpu-evidence/report.json" |
    sha256sum -c -
(cd "$harness" && sha256sum -c "$stage/inputs.sha256")
test "$(docker image inspect --format '{{.Config.User}}' "$image")" = 65532:65532
cc -std=c11 -O2 -Wall -Wextra -Werror -o allocator-probe allocator_probe.c -ldl
chmod 0555 allocator-probe start_riley.sh
base_run=(docker run --rm --pull=never --network none --env NVIDIA_VISIBLE_DEVICES=void)
bind_lib=(--mount "type=bind,source=$stage/libjemalloc.so.2,target=$library,readonly")
bind_probe=(--mount "type=bind,source=$stage/allocator-probe,target=$probe,readonly")
"${base_run[@]}" "${bind_probe[@]}" --entrypoint "$probe" "$image" >probe-glibc.json
"${base_run[@]}" "${bind_probe[@]}" "${bind_lib[@]}" --env "LD_PRELOAD=$library" \
    --env "MALLOC_CONF=$conf" --entrypoint "$probe" "$image" >probe-jemalloc.json

stub="$base/driver/libcuda.so.1"
test -f "$stub" && test ! -L "$stub"
/usr/bin/python3 -B - "$base/cpu-evidence/report.json" "$stub" <<'PYTHON'
from hashlib import sha256
import json
from pathlib import Path
import sys

report, stub = map(Path, sys.argv[1:])
assert json.loads(report.read_bytes())["cpu_cli_driver_stub_sha256"] == sha256(stub.read_bytes()).hexdigest()
PYTHON
runtime_stub=/run/riley-ci-driver/libcuda.so.1
flags=(--mount "type=bind,source=$stub,target=$runtime_stub,readonly" --env LD_LIBRARY_PATH=/run/riley-ci-driver)
"${base_run[@]}" "${flags[@]}" "$image" --version >version-glibc.txt
"${base_run[@]}" "${flags[@]}" "${bind_lib[@]}" \
    --mount "type=bind,source=$stage/start_riley.sh,target=$startup,readonly" \
    --entrypoint "$startup" "$image" --version >version-jemalloc.txt
cmp version-glibc.txt version-jemalloc.txt
"${base_run[@]}" "${flags[@]}" "${bind_lib[@]}" --entrypoint /bin/sh "$image" -ec \
    'LD_DEBUG=bindings LD_BIND_NOW=1 LD_PRELOAD="$1" MALLOC_CONF="$2" exec /opt/riley/bin/riley --version' \
    check "$library" "$conf" >version-binding.txt 2>loader-bindings.log
cmp version-glibc.txt version-binding.txt
/usr/bin/python3 -B - "$library" <<'PY' >cpu-preflight.json
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

lib = sys.argv[1]
reference, candidate = (json.loads(Path(f"probe-{name}.json").read_bytes()) for name in ("glibc", "jemalloc"))
assert reference["malloc_provider"].endswith("/libc.so.6") and reference["version"] == ""
assert candidate["malloc_provider"] == lib and candidate["version"].startswith("5.2.1")
assert all(candidate[key] == 0 for key in ("dirty_decay_ms", "muzzy_decay_ms", "oversize_threshold"))
assert candidate["rss_allocated_kib"] - candidate["rss_ready_kib"] >= 29 * 1024
assert candidate["rss_allocated_kib"] - candidate["rss_partially_freed_kib"] >= 24 * 1024
pattern = re.compile(r"binding file /opt/riley/bin/riley \[0\] to " + re.escape(lib)
                     + r" \[0\]: normal symbol .(malloc|realloc|free|calloc).")
bindings = sorted(set(pattern.findall(Path("loader-bindings.log").read_text())))
assert {"malloc", "realloc", "free"} <= set(bindings)
print(json.dumps({"same_riley_cli_version": Path("version-glibc.txt").read_text().strip(),
                  "riley_symbols_resolved_to_jemalloc": bindings, "service_uid_gid": "65532:65532",
                  "glibc": reference, "jemalloc": candidate,
                  "loader_binding_log_sha256": sha256(Path("loader-bindings.log").read_bytes()).hexdigest()},
                 indent=2))
PY
sha256sum cpu-preflight.json start_riley.sh libjemalloc.so.2 >preflight.sha256
cat cpu-preflight.json
