#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
prior="$(dirname "$stage")/w3-wide-prefill-order-20261124.cy7XlWYP"
sha256sum --strict -c build-dispatch.sha256
(cd "$prior"
    test "$(cat campaign.exit)" = 0
    printf '%s  %s\n' \
        363b6b74595a877213c081c1e04e0f2f4636544c95074d091632fe98f6139856 campaign-dispatch.sha256 \
        7953b13a5c2fb94e5a6be0804ec0159b691c4dff1878fd1744ee9df09eab26a5 result.sha256 |
        sha256sum --strict -c
    sha256sum --strict -c campaign-dispatch.sha256 result.sha256 >/dev/null)
cp -- "$prior"/{campaign.sh,configure.py,holdout.py} .
python3 -B - <<'PY'
import ast
from pathlib import Path
configure = Path("configure.py")
text = configure.read_text()
anchor = '''    arm.write_text(source)'''
addition = '''    trace_anchor = '        "--env", f"RILEY_W3_PREFILL_FIFO='
    if source.count(trace_anchor) != 1:
        raise ValueError("missing pinned trace environment setting")
    source = source.replace(trace_anchor, '        "--env", "RILEY_W3_RAG_STEPS=1",\\n' + trace_anchor)
    ast.parse(source)
    arm.write_text(source)'''
assert text.count(anchor) == 1
text = text.replace(anchor, addition)
ast.parse(text)
configure.write_text(text)
PY
sha256sum campaign.sh configure.py holdout.py >campaign-dispatch.sha256
exec bash campaign.sh
