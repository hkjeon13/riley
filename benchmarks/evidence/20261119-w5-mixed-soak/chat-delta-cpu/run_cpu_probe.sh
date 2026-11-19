#!/usr/bin/env bash
set -euo pipefail
evidence=$(cd -- "$(dirname -- "$0")" && pwd -P)
repository=$(git -C "$evidence" rev-parse --show-toplevel)
revision=d3d8fe6e5b1c528bff51c0008cdc24544c99f862
workspace=$(mktemp -d "${TMPDIR:-/tmp}/riley-chat-delta.XXXXXXXX")
printf 'CPU comparison directory %s\n' "$workspace"

mkdir "$workspace/base" "$workspace/candidate"
git -C "$repository" archive "$revision" >"$workspace/source.tar"
for mode in base candidate; do
    tar -xf "$workspace/source.tar" -C "$workspace/$mode"
done
gzip -dc "$evidence/chat-delta.patch.gz" >"$workspace/candidate.patch"
(
    cd "$workspace/candidate"
    git apply --check "$workspace/candidate.patch"
    git apply "$workspace/candidate.patch"
)
for mode in base candidate; do
    source_dir="$workspace/$mode/crates/riley-server/src"
    cp -- "$evidence/chat_frame_probe.rs" "$source_dir/"
    printf '\n#[cfg(test)]\nmod chat_frame_probe;\n' >>"$source_dir/lib.rs"
done
python3.12 - "$workspace/base/crates/riley-server/src/chat_frame_probe.rs" <<'PY'
from pathlib import Path
import sys

probe = Path(sys.argv[1])
text = probe.read_text()
start = text.index("    if direct", text.index("fn frame("))
end = text.index("\n}\n", start)
probe.write_text(text[:start] + "    let _ = direct;\n    shaper.shape(&encoder.encode_event(event).unwrap())" + text[end:])
PY
for mode in base candidate; do
    (
        cd "$workspace/$mode"
        cargo test --locked --offline -p riley-server --features server --lib \
            chat_frame_probe::compare_chat_frame_wire_and_allocations -- --nocapture --test-threads=1
    ) >"$workspace/$mode.log" 2>&1
done
python3.12 - "$workspace" <<'PY'
from pathlib import Path
import sys

folder = Path(sys.argv[1])
def cases(file):
    result = {}
    for line in file.read_text().splitlines():
        if "FRAME_PROBE " in line:
            name, *fields = line.split("FRAME_PROBE ", 1)[1].split()
            result[name] = dict(field.split("=", 1) for field in fields)
    return result

base, candidate = (cases(folder / f"{mode}.log") for mode in ("base", "candidate"))
assert base.keys() == candidate.keys() and len(base) == 7, "missing CPU cases"
for name, row in candidate.items():
    for field in ("old_calls", "old_bytes", "sha256"):
        assert row[field] == base[name][field], (name, field)
    assert row["different"] == "0", name
    print(name, row["old_calls"], row["new_calls"], row["old_bytes"], row["new_bytes"], row["sha256"])
assert "FLOAT_SWEEP samples=4096 different=0" in (folder / "candidate.log").read_text()
PY
