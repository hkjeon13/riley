"""Recompute externally observed serving and fixed decode parity from both engines' SSE records."""

import argparse
import itertools
import json
from hashlib import sha256
from pathlib import Path
from statistics import median

from agent_workflow import answer_format, fingerprint, fixture, percentile
from fixed_agent_decode import freeze
from verify_coder_catalog import answer_checks, verify_job
from verify_coder_pressure import (
    check_manifest,
    check_server,
    qualifies,
    read,
    work_latency,
)


def application(raw, cfg, *, application_targets=False):
    waves = []
    for wave, (size, base) in zip(raw, cfg["waves"], strict=True):
        assert (size, base) == (wave["size"], wave["base"])
        assert (
            wave["before"]["clock_ns"]
            <= wave["origin_ns"]
            < wave["drained_ns"]
            <= wave["after"]["clock_ns"]
        )
        rows = []
        for i, job in zip(range(size), wave["jobs"], strict=True):
            reader = i % 4 == 0
            assert job["index"] == i and job["role"] == (
                "reader" if reader else "target"
            )
            seed = base + i
            t0 = wave["origin_ns"] + 30_000_000 * i
            work = job["work"]
            assert (
                job["scheduled_ns"] == t0
                and t0 <= work["start_ns"] <= work["end_ns"] <= wave["drained_ns"]
            )
            facts = verify_job(
                "selected",
                job,
                seed,
                reader,
                application_tools=application_targets and not reader,
            )
            times = work_latency(work, t0)
            fmt = answer_format(
                work["turns"][-1].get("answer", ""), fixture(seed, 2)[2]
            )
            rows.append(
                {
                    "seed": seed,
                    "reader": reader,
                    "valid": facts["valid"],
                    "slo": qualifies(facts["valid"], times),
                    "words": fmt["explanation_words"],
                    "tool_calls": facts["verified_tool_calls"],
                    "error": work.get("error"),
                    "tool_model_ms": [
                        (t["done_ns"] - t["sent_ns"]) / 1e6
                        for t in work["turns"][:2]
                        if t.get("done_ns")
                    ],
                    "tool_wall_ms": [
                        (t["tool_ready_ns"] - t["tool_start_ns"]) / 1e6
                        for t in work["turns"][:2]
                        if t.get("tool_ready_ns")
                    ],
                    **answer_checks(work),
                    **times,
                }
            )
        waves.append(
            {"duration_s": (wave["drained_ns"] - wave["origin_ns"]) / 1e9, "rows": rows}
        )
    return waves


def summarize(waves):
    rows = [r for w in waves for r in w["rows"]]
    readers = [r for r in rows if r["reader"]]

    def q(group, key):
        return percentile([r[key] for r in group if r.get(key) is not None], 0.95)

    n = sum(r["slo"] for r in rows)
    targets = [r for r in rows if not r["reader"] and r.get("completed")]
    stage_medians = {
        k: median(vals)
        for k in ("before_final_submission_ms", "ttft_ms", "stream_ms")
        if (vals := [r[k] for r in targets])
    }
    for i in range(2):
        for k in ("tool_model_ms", "tool_wall_ms"):
            vals = [r[k][i] for r in targets if len(r[k]) > i]
            if vals:
                stage_medians[f"{k}_{i}"] = median(vals)
    return {
        "jobs": len(rows),
        "long_stage_medians_ms": stage_medians,
        "long_submitted_after_first_deadline": sum(
            r["before_final_submission_ms"] >= 5000 for r in targets
        ),
        "long_answer_checks": {
            k: sum(r[k] for r in rows if not r["reader"])
            for k in ("all_three_values", "both_sources", "at_least_64_tokens")
        },
        "valid": sum(r["valid"] for r in rows),
        "failed_basic": [r["seed"] for r in rows if not r["valid"]],
        "completed": sum(bool(r.get("completed")) for r in rows),
        "slo": n,
        "slo_readers": sum(r["slo"] for r in readers),
        "strict90": sum(r["words"] == 90 for r in rows),
        "first_p95_ms": q(rows, "first_ms"),
        "last_p95_ms": q(rows, "last_ms"),
        "reader_sse_gap_p95_ms": q(readers, "gap_p95_ms"),
        "output_tokens": sum(r.get("tokens") or 0 for r in rows),
        "warm_slo_per_gpu_hour": n * 3600 / sum(w["duration_s"] for w in waves),
    }


def fixed(d, requests):
    if (d / "fixed-error.json").exists():
        return {"failure": read(d / "fixed-error.json")}
    trials = read(d / "fixed.json")
    assert len(trials) == 5
    cells = {}
    for (role, req), ident in zip(requests.items(), trials[:2], strict=True):
        assert ident["phase"] == "identity"
        r = ident["row"]
        assert r["name"] == role and r["input_sha256"] == fingerprint(
            {**req, "return_token_ids": True}
        )
        assert (
            r["prompt_ids_sha256"]
            and len(r["generated_ids"]) == r["usage"]["completion_tokens"] == 64
        )
        assert r["finish"] == "length"
        cells[role] = {
            "prompt_ids_sha256": r["prompt_ids_sha256"],
            "prompt_tokens": r["usage"]["prompt_tokens"],
            "generated_ids": r["generated_ids"],
            "tpot": [],
            "gaps": [],
            "drifts": [],
        }
    for i, trial in zip(range(3), trials[2:], strict=True):
        assert (trial["phase"], trial["index"]) == ("measured", i)
        one, two = trial["rows"]
        assert one["content_ns"][0] < two["sent_ns"] < one["content_ns"][-1]
        for (role, req), r in zip(requests.items(), trial["rows"], strict=True):
            assert r["name"] == role and r["input_sha256"] == fingerprint(req)
            c = cells[role]
            assert r["finish"] == "length" and r["usage"]["completion_tokens"] == 64
            assert r["usage"]["prompt_tokens"] == c["prompt_tokens"]
            events = r["content_ns"]
            assert len(events) >= 2 and events == sorted(events)
            c["tpot"].append((events[-1] - events[0]) / 1e6 / 63)
            c["gaps"].extend((b - a) / 1e6 for a, b in itertools.pairwise(events))
            c["drifts"].append(abs(r["schedule_error_ms"]))
    return {
        role: {
            "tokens": c["prompt_tokens"],
            "prompt_ids": c["prompt_ids_sha256"],
            "output_ids_hash": fingerprint(c["generated_ids"]),
            "median_tpot_ms": median(c["tpot"]),
            "gap_p95_ms": percentile(c["gaps"], 0.95),
            "max_arrival_drift_ms": max(c["drifts"]),
        }
        for role, c in cells.items()
    }


def verify(root):
    check_manifest(root)
    cfg = read(root / "coder-catalog-comparison.json")
    frozen = read(root / "frozen.json")
    requests = {
        name: freeze(w, "m") for name, w in zip(("reader", "long"), frozen, strict=True)
    }
    results = {}
    for gpu, order in cfg["orders"].items():
        arms = results[gpu] = {}
        for engine in order:
            d = root / f"gpu{gpu}" / engine
            assert read(d / "warm.json")["ok"]
            lease = read(d / "lease-after.json")
            assert lease["phase"] == "idle" and lease["physical_index"] == int(gpu)
            if engine == "riley":
                check_server(
                    d,
                    "13b589df5c19a33c685ca1388398905f46a6cc566cbd1bed68f06b798080ca9f",
                    gpu,
                )
                cli = (d / "server-arguments.txt").read_text().splitlines()
                for flag, value in [
                    ("--max-active-sequences", "8"),
                    ("--max-sequence-tokens", "8192"),
                    ("--kv-blocks", "4096"),
                    ("--inference-profile", "native-bf16-varlen-v1"),
                ]:
                    assert cli[cli.index(flag) + 1] == value
                assert cli[cli.index("--model") + 1].endswith(
                    "/Qwen2.5-Coder-7B-Instruct"
                )
            else:
                cli = read(d / "command.json")
                assert cli[2].endswith("/Qwen2.5-Coder-7B-Instruct")
                model = read(root / "checkpoint-config.json")
                kv = (
                    4096
                    * 16
                    * 2
                    * 2
                    * model["num_hidden_layers"]
                    * model["num_key_value_heads"]
                    * (model["hidden_size"] // model["num_attention_heads"])
                )
                for flag, value in [
                    ("--max-num-seqs", "8"),
                    ("--max-model-len", "8192"),
                    ("--dtype", "bfloat16"),
                    ("--kv-cache-memory-bytes", str(kv)),
                ]:
                    assert cli[cli.index(flag) + 1] == value
                assert (
                    "--enable-prefix-caching" in cli
                    and "--exclude-tools-when-tool-choice-none" in cli
                )
            waves = application(read(d / "waves.json"), cfg)
            arms[engine] = {
                "high": summarize(waves[2:]),
                "by_wave": [summarize([w]) for w in waves],
                "waves": waves,
                "fixed": fixed(d, requests),
                "gpu_uuid": lease["gpu"],
            }
            if engine == "vllm":
                arms[engine]["version"] = (d / "version.txt").read_text().strip()
        assert arms["riley"]["gpu_uuid"] == arms["vllm"]["gpu_uuid"]
        r, v = [arms[e]["high"] for e in ("riley", "vllm")]
        arms["application_gates"] = {
            "slo_count": r["slo"] >= v["slo"],
            "goodput": r["warm_slo_per_gpu_hour"] >= v["warm_slo_per_gpu_hour"],
            "last_p95": r["last_p95_ms"] <= 1.05 * v["last_p95_ms"],
            "preserved_valid": set(r["failed_basic"]) <= set(v["failed_basic"]),
        }
        rf, vf = [arms[e]["fixed"] for e in ("riley", "vllm")]
        ratios = {}
        if "failure" not in rf and "failure" not in vf:
            for role in requests:
                a, b = rf[role], vf[role]
                assert a["tokens"] == b["tokens"] and a["prompt_ids"] == b["prompt_ids"]
                ratios[role] = {
                    k: a[k] / b[k] for k in ("median_tpot_ms", "gap_p95_ms")
                }
                ratios[role]["within_5_percent"] = all(
                    v <= 1.05 for v in ratios[role].values()
                )
                ratios[role]["same_generated_ids"] = (
                    a["output_ids_hash"] == b["output_ids_hash"]
                )
        arms["fixed_ratios"] = ratios
    return results


def verify_attempt(root):
    assert (root / "exits.log").read_text().split() == ["1", "1"]
    for line in (root / "inputs.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert sha256((root / name).read_bytes()).hexdigest() == digest
    models = {
        k: v
        for v, k in (
            line.split(maxsplit=1)
            for line in (root / "model.sha256").read_text().splitlines()
        )
    }
    assert models["config.json"] == sha256(
        (root / "checkpoint-config.json").read_bytes()
    ).hexdigest() and any(name.endswith(".safetensors") for name in models)
    assert "31.05/39.38 GiB" in (root / "gpu6/vllm/server.log").read_text()
    assert "unadmitted process" in (root / "gpu7/vllm.controller.log").read_text()
    d = root / "gpu7/riley"
    check_server(
        d, "13b589df5c19a33c685ca1388398905f46a6cc566cbd1bed68f06b798080ca9f", 7
    )
    assert read(d / "warm.json")["ok"]
    waves = application(
        read(d / "waves.json"), read(root / "coder-catalog-comparison.json")
    )
    reqs = {
        role: freeze(w, "m")
        for role, w in zip(("reader", "long"), read(root / "frozen.json"), strict=True)
    }
    return {
        "paired": False,
        "vllm_version": (root / "gpu6/vllm/version.txt").read_text().strip(),
        "failures": {
            "6": "vLLM startup free memory below default reservation",
            "7": "lease manager refused unadmitted process",
        },
        "gpu7_riley": {
            "high": summarize(waves[2:]),
            "by_wave": [summarize([w]) for w in waves],
            "fixed": fixed(d, reqs),
            "other_processes_after": read(d / "lease-after.json")["other_processes"],
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--initial-attempt", action="store_true")
    a = p.parse_args()
    a.out.write_text(
        json.dumps(
            verify_attempt(a.root) if a.initial_attempt else verify(a.root),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
