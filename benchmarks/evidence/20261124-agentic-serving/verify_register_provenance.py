"""Check exact source/value binding against the original register on both GPUs."""

import argparse
import json
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "live"))
from agent_workflow import fingerprint
from full_register import case, obj, parse
from verify_register_followup import check_files, read, run_receipt


def verify(root):
    check_files(root)
    reg = read(root / "register-provenance-registration.json")
    if reg != read(HERE / "register-provenance-registration.json"):
        raise ValueError("registration changed")
    previous = read(HERE / "register-reduction-verified.json")
    results = {}
    signatures = {}
    with tarfile.open(HERE / "register-reduction-raw.tar.gz") as prior:
        for gpu in (6, 7):
            run_receipt(root, f"gpu{gpu}", "page-v1", gpu)
            old = {
                f"{r['kind']}-{r['seed']}": r
                for r in json.load(prior.extractfile(f"./gpu{gpu}/reduction.json"))
                if r["policy"] == "reduced"
            }
            captures = iter(read(root / f"gpu{gpu}" / "provenance.json"))
            cases = {}
            i = 0
            end = 0
            for kind, specs in reg["cases"].items():
                for seed, count in specs:
                    key = f"{kind}-{seed}"
                    pair = {}
                    first_strict = bool((gpu + i) % 2)
                    i += 1
                    for strict in (first_strict, not first_strict):
                        q, records, answer, _ = case(kind, seed, count)
                        full = json.dumps(records, separators=(",", ":"))
                        subset = (
                            [
                                r
                                for r in records
                                if r["service"] in ("suspended", "restored")
                            ]
                            if kind == "scan"
                            else [
                                r
                                for r in records
                                if r["approved"] == answer["approved"]
                            ]
                        )
                        if len(subset) != (2 if kind == "scan" else 1):
                            raise ValueError("ambiguous input")
                        q["messages"][-1]["content"] = json.dumps(
                            subset, separators=(",", ":")
                        )
                        if strict:
                            allowed = sorted(r["source"] for r in subset)
                            entry = obj(
                                {
                                    "source": {"type": "string", "enum": allowed},
                                    "approved": {"type": "integer"},
                                }
                            )
                            schema = (
                                obj({"suspended": entry, "restored": entry})
                                if kind == "scan"
                                else entry
                            )
                            q["response_format"] = {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "tool_result",
                                    "strict": True,
                                    "schema": schema,
                                },
                            }
                        r = next(captures)
                        m = r["measurement"]
                        if (
                            (
                                r["kind"],
                                r["seed"],
                                r["count"],
                                r["strict"],
                                r["expected"],
                            )
                            != (kind, seed, count, strict, answer)
                            or r["wire"] != fingerprint(q)
                            or m["input_sha256"] != r["wire"]
                            or r["full_hash"] != fingerprint(full)
                            or r["start_ns"] <= end
                        ):
                            raise ValueError(
                                "wire payload, original data or schedule differs"
                            )
                        end = m.get("done_ns", m["sent_ns"])
                        events = m.get("content_ns", [])
                        complete = bool(
                            events
                            and m.get("done_ns")
                            and m.get("status") == 200
                            and not r.get("error")
                            and not r["calls"]
                        )
                        usage = m.get("usage", {})
                        if (
                            complete
                            and not r["start_ns"]
                            <= m["sent_ns"]
                            <= m["request_written_ns"]
                            <= events[0]
                            <= events[-1]
                            <= end
                        ):
                            raise ValueError("missing event timestamps")
                        if usage.get("prompt_tokens_details", {}).get(
                            "cached_tokens", 0
                        ):
                            raise ValueError("unexpected warm prefix")
                        signature = r["wire"], usage.get("prompt_tokens")
                        if signature != signatures.setdefault((key, strict), signature):
                            raise ValueError("cross-GPU input changed")
                        if (
                            not strict
                            and key in old
                            and r.get("answer") != old[key].get("answer")
                        ):
                            raise ValueError("earlier free generation changed")
                        pair["schema" if strict else "free"] = {
                            "correct": complete
                            and m.get("finish") == "stop"
                            and fingerprint(parse(r.get("answer", "")))
                            == fingerprint(answer),
                            "input": usage.get("prompt_tokens"),
                            "output": usage.get("completion_tokens"),
                            "error": r.get("error"),
                            "first_ms": (events[0] - r["start_ns"]) / 1e6
                            if events
                            else None,
                            "last_ms": (events[-1] - r["start_ns"]) / 1e6
                            if events
                            else None,
                        }
                    c, f = pair["schema"], pair["free"]
                    pair["pass"] = c["correct"] and c["input"] == f["input"]
                    if key in old:
                        base = previous["gpus"][str(gpu)][key]["full"]
                        pair["pass"] = (
                            pair["pass"]
                            and c["first_ms"] < 0.8 * base["first_ms"]
                            and c["last_ms"] < 0.8 * base["last_ms"]
                        )
                    cases[key] = pair
            if next(captures, None) is not None:
                raise ValueError("unexpected extra requests")
            results[str(gpu)] = cases
    return {
        "gpus": results,
        "all_pass": all(
            pair["pass"] for cases in results.values() for pair in cases.values()
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(verify(p.parse_args().root), indent=2))
