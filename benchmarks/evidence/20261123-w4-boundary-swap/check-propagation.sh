#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
expected=$(shasum -a 256 "$here/result.sha256" | cut -d' ' -f1)
dispatch=$(shasum -a 256 "$here/dispatch.sha256" | cut -d' ' -f1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -c 'set -euo pipefail
cd /mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-boundary-swap-gpu-20261123.OhUoSQsC
test \"\$(cat campaign.exit)\" = 0
printf \"%s  result.sha256\\n%s  dispatch.sha256\\n\" \"\$1\" \"\$2\" | sha256sum --strict -c >/dev/null
sha256sum --strict -c dispatch.sha256 result.sha256 >/dev/null
export PYTHONPATH=../w4-long-code-gpu-20261122.QAbtnxKV:../w4-long-code-build-20261122.LO5rPFy0:../w4-long-code-build-20261122.LO5rPFy0/tools
exec ../../tools/vllm-venv/bin/python -B -
' bash '$expected' '$dispatch'" <<'PY'
from pathlib import Path
from struct import iter_unpack, pack, unpack
import json

from trial import BOUNDARIES, OLD_KEYS, delta, digest, require, trace


def values(raw):
    return [value for (value,) in iter_unpack("<H", raw)]


def as_float(value):
    return unpack("<f", pack("<I", value << 16))[0]


order = list(OLD_KEYS)
for before, norm in (("attention_residual", "post_attention_norm"), ("layer_residual", "next_input_norm")):
    order.insert(order.index((before, "0")), (norm, "0"))
roles = ("baseline", "candidate")
word_changes = {}
directions = {}
for comparison, (_, keys) in BOUNDARIES.items():
    parent = Path(comparison)
    own = {side: trace(parent / side / "calibration") for side in roles}
    if comparison == "attention31":
        for key in keys:
            word_changes[key[0]] = [
                {"index_zero_based": index, "baseline_bits": f"0x{a:04x}", "candidate_bits": f"0x{b:04x}",
                 "baseline_value": as_float(a), "candidate_value": as_float(b)}
                for index, (a, b) in enumerate(zip(values(own["baseline"][key]),
                                                   values(own["candidate"][key]), strict=True)) if a != b
            ]
    directions[comparison] = {}
    after = order[max(order.index(key) for key in keys) + 1:]
    for recipient, donor in (roles, roles[::-1]):
        swapped = trace(parent / recipient / "cross")
        donor_rows = own[donor]
        require(all(swapped[key] == donor_rows[key] for key in keys), "donor rows differ after replacement")
        first_index = next(i for i, key in enumerate(after) if swapped[key] != donor_rows[key])
        first = after[first_index]
        directions[comparison][recipient] = {
            "donor": donor,
            "equal_before_next_difference": [{"stage": label, "layer": layer} for label, layer in after[:first_index]],
            "first_difference_from_donor": {"stage": first[0], "layer": int(first[1]),
                                            **delta(swapped[first], donor_rows[first])},
        }

print(json.dumps({"archive_checksums_sha256": digest(Path("result.sha256")), "attention_words": word_changes,
                  "directions": directions}, indent=2, allow_nan=False))
PY
