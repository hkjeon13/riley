"""Check tool ownership, final prompt IDs, user-visible time and strict reader guards."""

import argparse
import json
import subprocess
from pathlib import Path

from agent_workflow import (
    append_tool_result,
    fingerprint,
    fixture,
    history_messages,
    percentile,
    turn_request,
)
from verify_catalog_engines import application, summarize
from verify_coder_pressure import check_manifest, check_server, read


def prompt_ids(profiler, tokenizer, raw):
    requests = []
    measured = []
    for w in raw:
        for j in w["jobs"]:
            work = j["work"]
            if len(work["turns"]) != 3 or not work["turns"][-1].get("usage"):
                continue
            seed = work["seed"]
            tools, prompt, docs = fixture(seed, 2)
            messages = history_messages(seed, work["history_entries"]) + [
                {"role": "user", "content": prompt}
            ]
            for i, row in enumerate(work["turns"][:2]):
                append_tool_result(messages, [row["call"]], docs, i)
                assert (
                    row["tool_ready_ns"] - row["tool_start_ns"]
                    >= work["tool_delay_s"] * 1e9
                )
            req = turn_request("m", messages, tools, None)
            final = work["turns"][-1]
            assert fingerprint(req) == final["input_sha256"]
            requests.append(req)
            measured.append((seed, final["usage"]["prompt_tokens"]))
    native = subprocess.run(
        [profiler, tokenizer],
        input="".join(json.dumps(r) + "\n" for r in requests),
        text=True,
        capture_output=True,
        check=True,
    )
    digests = {}
    for (seed, count), s in zip(measured, native.stdout.splitlines(), strict=True):
        ids = json.loads(s)["ids"]
        assert len(ids) == count and seed not in digests
        digests[seed] = fingerprint(ids)
    return digests


def verify(root, profiler, tokenizer):
    check_manifest(root)
    cfg = read(root / "named-burst-registration.json")
    results = {}
    args = None
    for gpu, order in cfg["orders"].items():
        arms = results[gpu] = {}
        identity = {}
        for mode in order:
            d = root / f"gpu{gpu}" / mode
            check_server(
                d,
                "13b589df5c19a33c685ca1388398905f46a6cc566cbd1bed68f06b798080ca9f",
                gpu,
            )
            assert read(d / "warm.json")["ok"]
            cli = (d / "server-arguments.txt").read_text()
            if args is None:
                args = cli
            assert cli == args
            raw = read(d / "waves.json")
            waves = application(raw, cfg, application_targets=mode == "application")
            identity[mode] = prompt_ids(profiler, tokenizer, raw)
            arms[mode] = {
                "by_wave": [summarize([w]) for w in waves],
                "high": summarize(waves[2:]),
                "waves": waves,
            }
        for mode in order:
            readers = [
                r["first_ms"]
                for w in arms[mode]["waves"][2:]
                for r in w["rows"]
                if r["reader"] and "first_ms" in r
            ]
            arms[mode]["high"]["reader_first_p95_ms"] = percentile(readers, 0.95)
        a, b = [arms[m]["high"] for m in ("model", "application")]
        pair = identity["model"].keys() & identity["application"].keys()
        arms["paired_prompt_ids"] = len(pair)
        arms["gates"] = {
            "same_inputs": len(pair) == sum(size for size, _ in cfg["waves"])
            and all(identity["model"][k] == identity["application"][k] for k in pair),
            "preserve_valid": set(b["failed_basic"]) <= set(a["failed_basic"]),
            "better_goodput": b["warm_slo_per_gpu_hour"] > a["warm_slo_per_gpu_hour"],
            "more_slo": b["slo"] > a["slo"],
            "faster_last": a["last_p95_ms"] is not None
            and b["last_p95_ms"] is not None
            and b["last_p95_ms"] < a["last_p95_ms"],
        }
        for gate, key in (
            ("reader_gap", "reader_sse_gap_p95_ms"),
            ("reader_first", "reader_first_p95_ms"),
        ):
            arms["gates"][gate] = (
                a[key] is not None and b[key] is not None and b[key] <= 1.05 * a[key]
            )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "profiler", "tokenizer", "out"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    a.out.write_text(
        json.dumps(verify(a.root, a.profiler, a.tokenizer), indent=2, allow_nan=False)
        + "\n"
    )
