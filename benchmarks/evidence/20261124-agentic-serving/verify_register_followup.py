"""Validate the independent numerical reference and the full-register reduction trials."""

import argparse
import json
import sys
import tarfile
from hashlib import sha256
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "live"))
from agent_tool_aggregate import project_unique_integer_max
from agent_workflow import fingerprint
from full_register import body, case, parse
from verify_long_tile import read, receipt


def check_files(root):
    if (root / "exit-status").read_text().strip() != "0" or not (
        root / "released"
    ).exists():
        raise ValueError("experiment incomplete")
    for line in (root / "inputs.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("harness file mismatch")


def run_receipt(root, name, mode, gpu):
    args, device, binary = receipt(root / name, mode)
    lease = read(root / name / "lease-after.json")
    if (
        binary != read(HERE / "long-tool-records-verified.json")["binary_sha256"]
        or lease["physical_index"] != gpu
        or lease["other_processes"]
        or "--prefix-cache-blocks" in args
        or device
        != (root / name / "device.csv").read_text().splitlines()[1].split(",")[0]
    ):
        raise ValueError("unexpected GPU, binary or cache configuration")
    return args, device


def reference(root):
    check_files(root)
    reg = read(root / "register-reference-registration.json")
    if reg != read(HERE / "register-reference-registration.json"):
        raise ValueError("reference registration changed")
    hf = read(root / "hf.json")
    lease = read(root / "hf-lease-after.json")
    page_args, _ = run_receipt(root, "page", "page-v1", 6)
    tile_args, dev = run_receipt(root, "tile", "fixed-tile-unsplit-v1", 7)
    if (
        page_args != tile_args
        or hf["dtype"] != "torch.bfloat16"
        or hf["attention"] != "sdpa"
        or lease["phase"] != "idle"
        or lease["other_processes"]
        or lease["gpu"] != dev
        or hf["gpu"] != dev
    ):
        raise ValueError("unmatched witness or reference state")
    manifest = read(root / "checkpoint.json")
    files = {r["path"]: r["sha256"] for r in manifest["files"]}
    if (
        set((root / "validated-files.txt").read_text().splitlines())
        != {f"{p}: OK" for p in files}
        or set((root / "model-files.sha256").read_text().splitlines())
        != {f"{h}  {p}" for p, h in files.items()}
        or not any(p.endswith(".safetensors") for p in files)
    ):
        raise ValueError("incomplete checkpoint verification")
    rows = {}
    captures = {k: read(root / k / "witness.json") for k in ("page", "tile")}
    with tarfile.open(HERE / "full-register-raw.tar.gz") as archive:
        for kind, seed, count in reg["cases"]:
            name = f"{kind}-{seed}"
            left, right = [captures[m][name] for m in ("page", "tile")]
            q, records, expected, schema = case(kind, seed, count)
            q = body(q, records, schema, False, False)
            q["return_token_ids"] = True
            h = hf["cases"][name]
            ids = {}
            correct = {}
            for m, g, r in (("page", 6, left), ("tile", 7, right)):
                original = json.load(
                    archive.extractfile(f"./gpu{g}-{m}/full-register.json")
                )
                (old,) = [
                    v
                    for v in original
                    if v["kind"] == kind
                    and v["seed"] == seed
                    and not v["strict"]
                    and not v["table"]
                ]
                x = r["measurement"]
                if (
                    r["answer"] != old["answer"]
                    or r["tool_calls"]
                    or x["status"] != 200
                    or not x["done_ns"]
                    or x["input_sha256"] != fingerprint(q)
                    or x["prompt_ids_sha256"] != h["prompt_ids_sha256"]
                    or x["usage"]["prompt_tokens"] != h["prompt_tokens"]
                    or r["expected"] != h["expected"]
                    or h["expected"] != expected
                ):
                    raise ValueError("unmatched rendered prompt or original replay")
                ids[m] = x["generated_ids"]
                correct[m] = (
                    fingerprint(parse(r["answer"])) == fingerprint(expected)
                    and x["finish"] == "stop"
                )
            hi = h["generated_ids"]
            correct["HF"] = (
                fingerprint(parse(h["answer"])) == fingerprint(expected)
                and hi[-1] in hf["eos"]
            )
            diff = next(
                (
                    i
                    for i, (a, b) in enumerate(
                        zip(ids["page"], ids["tile"], strict=False)
                    )
                    if a != b
                ),
                None,
            )
            d = h["first_difference"]
            if diff is None:
                if d is not None or ids["page"] != ids["tile"]:
                    raise ValueError("unaccounted output divergence")
            elif (
                d["index"] != diff
                or d["page"] != ids["page"][diff]
                or d["tile"] != ids["tile"][diff]
                or min(d["page_gap"], d["tile_gap"]) < 0
            ):
                raise ValueError("incorrect teacher-forced token prefix")
            rows[name] = {
                "tokens": h["prompt_tokens"],
                "correct": correct,
                "first_riley_difference": d,
                "same_generated_ids_as_HF": {m: v == hi for m, v in ids.items()},
                "end_to_end_first_difference_as_HF": {
                    m: next(
                        (
                            i
                            for i, (a, b) in enumerate(zip(v, hi, strict=False))
                            if a != b
                        ),
                        None,
                    )
                    for m, v in ids.items()
                },
            }
    return {
        "checkpoint_sha256": sha256(
            (root / "checkpoint.json").read_bytes()
        ).hexdigest(),
        "torch": hf["torch"],
        "transformers": hf["transformers"],
        "cases": rows,
    }


def reduction(root):
    for records in (
        [],
        [{}],
        [False],
        [{"amount": True}],
        [{"amount": 1.5}],
        [{"amount": 8}, {"amount": 8}],
        [{"amount": 4}, {"other": 5}],
    ):
        if project_unique_integer_max(records, key="amount") is not records:
            raise ValueError("ambiguous or malformed integer query lost source data")
    records = [
        {"amount": -4},
        {"amount": -4},
        {"amount": -1, "details": {"source": "x"}},
    ]
    result = project_unique_integer_max(records, key="amount")
    if len(result) != 1 or result[0] is not records[2]:
        raise ValueError("unique winner after a temporary tie lost its provenance")
    check_files(root)
    reg = read(root / "register-reduction-registration.json")
    if reg != read(HERE / "register-reduction-registration.json"):
        raise ValueError("reduction registration changed")
    out = {}
    both = {}
    settings = None
    for gpu in (6, 7):
        args, _ = run_receipt(root, f"gpu{gpu}", "page-v1", gpu)
        if settings is not None and settings != args:
            raise ValueError("configuration drift")
        settings = args
        capture = iter(read(root / f"gpu{gpu}" / "reduction.json"))
        cases = {}
        visit = 0
        end = 0
        for kind, specs in reg["cases"].items():
            for seed, count in specs:
                key = f"{kind}-{seed}"
                pair = {}
                order = (
                    ("full", "reduced")
                    if (gpu + visit) % 2 == 0
                    else ("reduced", "full")
                )
                visit += 1
                for policy in order:
                    row = next(capture)
                    req, records, answer, _ = case(kind, seed, count)
                    if kind == "scan":
                        selected = [
                            r
                            for r in records
                            if r["service"] in ("restored", "suspended")
                        ]
                    else:
                        selected = [
                            r for r in records if r["approved"] == answer["approved"]
                        ]
                        if (
                            len(selected) != 1
                            or selected[0]["source"] != answer["source"]
                        ):
                            raise ValueError("ambiguous or incorrect oracle")
                    original = json.dumps(records, separators=(",", ":"))
                    text = (
                        json.dumps(selected, separators=(",", ":"))
                        if policy == "reduced"
                        else original
                    )
                    req["messages"][-1]["content"] = text
                    m = row["measurement"]
                    if (
                        (
                            row["policy"],
                            row["expected"],
                            row["kind"],
                            row["seed"],
                            row["count"],
                        )
                        != (policy, answer, kind, seed, count)
                        or row["sent_content_hash"] != fingerprint(text)
                        or row["original_hash"] != fingerprint(original)
                        or m["input_sha256"] != fingerprint(req)
                        or row["start_ns"] <= end
                    ):
                        raise ValueError("source or order mismatch")
                    complete = bool(
                        m.get("done_ns")
                        and m.get("finish")
                        and m.get("status") == 200
                        and m.get("content_ns")
                        and not row.get("error")
                    )
                    events = m.get("content_ns", [])
                    if (
                        complete
                        and not row["start_ns"]
                        <= row["prepared_ns"]
                        <= m["sent_ns"]
                        <= m["request_written_ns"]
                        <= events[0]
                        <= events[-1]
                        <= m["done_ns"]
                    ):
                        raise ValueError("invalid request timing")
                    usage = m.get("usage", {})
                    if usage.get("prompt_tokens_details", {}).get("cached_tokens", 0):
                        raise ValueError("cached input")
                    end = m.get("done_ns", m["sent_ns"])
                    sig = m["input_sha256"], usage.get("prompt_tokens")
                    if sig != both.setdefault((key, policy), sig):
                        raise ValueError("input mismatch across devices")
                    pair[policy] = {
                        "complete": complete,
                        "correct": complete
                        and m["finish"] == "stop"
                        and not row["calls"]
                        and fingerprint(parse(row["answer"])) == fingerprint(answer),
                        "input": usage.get("prompt_tokens"),
                        "output": usage.get("completion_tokens"),
                        "first_ms": (events[0] - row["start_ns"]) / 1e6
                        if events
                        else None,
                        "last_ms": (events[-1] - row["start_ns"]) / 1e6
                        if events
                        else None,
                        "prepare_ms": (row["prepared_ns"] - row["start_ns"]) / 1e6,
                        "finish": m.get("finish"),
                    }
                f, r = pair["full"], pair["reduced"]
                pair["pass"] = bool(
                    f["complete"]
                    and r["correct"]
                    and r["input"] < f["input"]
                    and r["first_ms"] <= 0.8 * f["first_ms"]
                    and r["last_ms"] <= 0.8 * f["last_ms"]
                )
                cases[key] = pair
        if next(capture, None) is not None:
            raise ValueError("extra requests")
        out[str(gpu)] = cases
    return {
        "gpus": out,
        "all_pass": all(p["pass"] for cases in out.values() for p in cases.values()),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trial", choices=("reference", "reduction"))
    p.add_argument("root", type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            (reference if a.trial == "reference" else reduction)(a.root), indent=2
        )
    )
