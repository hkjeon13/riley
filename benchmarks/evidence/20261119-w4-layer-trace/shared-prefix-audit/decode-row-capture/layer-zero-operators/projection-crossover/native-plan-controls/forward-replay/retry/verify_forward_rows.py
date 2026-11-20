"""Verify the saved projection and final-logit rows without exporting tensors."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import math
from pathlib import Path
import struct


RECORDED = "recorded"
REPLAY = "rank-2"
BASELINE = "baseline"
MODES = (BASELINE, RECORDED, REPLAY, "baseline-repeat")
ADDRESSABLE_VOCABULARY = 151665


def checksum(data):
    return sha256(data).hexdigest()


def captured_rows(folder):
    raw = (folder / "rows.bf16").read_bytes()
    header, *index = (folder / "index.tsv").read_text().splitlines()
    fields = header.split()
    assert fields[0] == "#" and fields[3:] == ["0", "1", "1"]
    start = 0
    rows = {}
    for line in index:
        stage, layer, offset, count = line.split("\t")
        assert int(offset) == start and int(count) > 0 and int(count) % 2 == 0
        end = start + int(count)
        assert (stage, layer) not in rows and end <= len(raw)
        rows[stage, layer] = raw[start:end]
        start = end
    assert start == len(raw)
    return raw, rows


def values_le(bytes_):
    return [struct.unpack("<f", b"\0\0" + bytes_[pos:pos + 2])[0] for pos in range(0, len(bytes_), 2)]


def verify(root, reference):
    report_bytes = (root / "verified.json").read_bytes()
    measured = json.loads(report_bytes)
    plan_bytes = (reference / "verified.json").read_bytes()
    assert checksum(plan_bytes) == measured["controls_report_sha256"]
    plans = json.loads(plan_bytes)["plans"]
    assert plans[RECORDED]["config"]["id"] == 21
    assert plans[REPLAY]["config"] == {**plans[RECORDED]["config"], "id": 31}

    output = {}
    assert measured["mode_order"] == list(MODES)
    for case, claim in measured["cases"].items():
        rows = {}
        raw_trace = {}
        case_summary = {}
        for mode in MODES:
            raw_trace[mode], rows[mode] = captured_rows(root / "riley" / "rows" / f"{case}.{mode}")
            assert checksum(raw_trace[mode]) == claim["trace_sha256"][mode]
            qkv = rows[mode]["qkv_prebias", "0"]
            plan = REPLAY if mode == REPLAY else RECORDED
            assert checksum(qkv) == plans[plan]["cases"][case]["sha256"]
            logits = rows[mode]["logits", "-"]
            assert len(logits) == 152064 * 2
            values = values_le(logits[:ADDRESSABLE_VOCABULARY * 2])
            assert all(map(math.isfinite, values))
            profiled_token = claim["selected_token"]["replay" if mode == REPLAY else BASELINE]
            highest = max(values)
            assert values[profiled_token] == highest
            assert all(values[int(token)] == score for token, score in claim["scores"][mode].items())
            hf = claim["selected_token"]["hf_reference"]
            case_summary[mode] = {
                "largest_logit_token_id": profiled_token,
                "max_tie_count": values.count(highest),
                "hf_reference_rank": 1 + sum(score > values[hf] for score in values),
                "selected_minus_hf_logit": highest - values[hf],
            }
        assert raw_trace[BASELINE] == raw_trace[RECORDED] == raw_trace["baseline-repeat"]
        before, after = rows[BASELINE], rows[REPLAY]
        assert before.keys() == after.keys()
        for row in claim["row_differences"]:
            key = row["stage"], row["layer"]
            a, b = before[key], after[key]
            assert len(a) == len(b)
            changed = sum(a[i:i + 2] != b[i:i + 2] for i in range(0, len(a), 2))
            assert changed == row["different_bf16_words"]
        probe_count = len(before["qkv_prebias", "0"]) // 2
        assert probe_count == 4608 and len(claim["row_differences"]) == len(before)
        output[case] = case_summary
    assert len(output) == 2
    return {"verified_report_sha256": checksum(report_bytes),
            "addressable_vocabulary": ADDRESSABLE_VOCABULARY,
            "all_reported_row_difference_counts_reproduced": True,
            "cases": output}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.reference), indent=2, allow_nan=False))
