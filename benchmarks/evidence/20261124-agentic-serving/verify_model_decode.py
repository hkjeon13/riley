"""Score independent model crossovers from original SSE records and checkpoint digests."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import median

from agent_workflow import (
    append_tool_result,
    fingerprint,
    fixture,
    history_messages,
    turn_request,
)
from fixed_agent_decode import freeze
from verify_decode import compare


def read(path):
    return json.loads(path.read_text())


def natural(
    work,
    seed,
    notes,
    depth,
    *,
    delay=0.1,
    execute_tool=None,
    application_tools=False,
    final_transform=None,
):
    tools, prompt, docs = fixture(seed, notes)
    assert (
        work["seed"],
        work["history_entries"],
        work["document_pairs"],
        work["tool_delay_s"],
    ) == (seed, depth, notes, delay)
    assert work["fixture_sha256"] == fingerprint([prompt, docs])
    history = history_messages(seed, depth) + [{"role": "user", "content": prompt}]
    calls = 0
    assert 1 <= len(work["turns"]) <= 3
    for index, (row, name) in enumerate(
        zip(work["turns"], ("list_reports", "read_reports", None))
    ):
        local = bool(name and application_tools)
        key = "planned_request_sha256" if local else "input_sha256"
        request = turn_request("m", history, tools, name)
        if name is None and final_transform is not None:
            request = final_transform(request)
        assert row[key] == fingerprint(request)
        if local:
            assert row["dispatch"] == "application"
            assert {
                "sent_ns",
                "response_id",
                "usage",
                "observed_calls",
                "content_ns",
                "finish",
            }.isdisjoint(row)
            assert row["dispatch_ns"] <= row["tool_start_ns"] <= row["tool_ready_ns"]
        else:
            assert row.get("dispatch") != "application"
        if application_tools and index:
            previous = work["turns"][index - 1]
            assert (
                previous["tool_ready_ns"] <= row["dispatch_ns" if local else "sent_ns"]
            )
        if name:
            observed = [row["call"]] if local else row.get("observed_calls") or []
            if len(observed) != 1 or observed[0]["function"]["name"] != name:
                break
            expected = (
                {} if index == 0 else {"first": docs[0]["id"], "second": docs[1]["id"]}
            )
            if json.loads(observed[0]["function"]["arguments"]) != expected:
                break
            calls += 1
            append_tool_result(history, observed, docs, index, execute_tool)
    final = work["turns"][-1]
    text = final.get("answer", "").replace("*", "")
    values = [d["p95_ms"] for d in docs]
    valid = bool(
        calls == 2
        and final["turn"] == 2
        and final.get("done_ns")
        and final.get("finish") == "stop"
        and final.get("usage", {}).get("completion_tokens", 0) >= 64
        and all(d["id"].lower() in text.lower() for d in docs)
        and all(
            re.search(rf"\b{label}\s*:\s*{v}\s*ms\b", text, re.IGNORECASE)
            for label, v in zip(
                ("before", "after", "reduction"), [*values, values[0] - values[1]]
            )
        )
    )
    assert valid == work["ok"]
    events = final.get("content_ns") if final["turn"] == 2 else []
    return {
        "seed": seed,
        "valid": valid,
        "verified_tool_calls": calls,
        "error": work.get("error"),
        "before_final_submission_ms": (final["sent_ns"] - work["start_ns"]) / 1e6
        if events
        else None,
        "final_submission_to_first_ms": (events[0] - final["sent_ns"]) / 1e6
        if events
        else None,
        "final_stream_ms": (events[-1] - events[0]) / 1e6 if events else None,
        "first_final_ms": (events[0] - work["start_ns"]) / 1e6 if events else None,
        "last_final_ms": (events[-1] - work["start_ns"]) / 1e6 if events else None,
        "final_tokens": final.get("usage", {}).get("completion_tokens")
        if final["turn"] == 2
        else None,
    }


def check_frozen_requests(root, directory):
    expected = {}
    for history in (0, 200):
        original = read(root / "frozen" / f"workflows-{history}.json")
        for name, work in zip(
            ("reader", "long-tool-result"), original[-2:], strict=True
        ):
            expected[(history, name)] = freeze(work, "m")
    for wave in read(directory / "fixed.json"):
        is_identity = wave["phase"] == "identity"
        for row in [wave["result"]] if is_identity else wave["results"]:
            req = expected[wave["history"], row["name"]]
            body = {**req, "return_token_ids": True} if is_identity else req
            assert row["input_sha256"] == fingerprint(body)
            assert row["done_ns"] and row["status"] == 200


def normal(folder):
    result = {}
    for entry in read(folder / "fixed.json"):
        if entry["phase"] != "normal":
            continue
        for row in entry["results"]:
            data = result.setdefault(
                (entry["history"], row["name"]),
                {"cache": [], "ttft": [], "last": [], "texts": []},
            )
            details = row["usage"].get("prompt_tokens_details") or {}
            data["texts"].append(row["text_sha256"])
            data["cache"].append(details.get("cached_tokens"))
            data["ttft"].append((row["content_ns"][0] - row["sent_ns"]) / 1e6)
            data["last"].append((row["content_ns"][-1] - row["sent_ns"]) / 1e6)
    return result


def verify(root):
    reg = read(root / "model-decode-registration.json")
    result = {}
    for gpu, model in reg["models"].items():
        directory = root / f"gpu{gpu}"
        digests = {
            name: digest
            for digest, name in (
                l.split() for l in (directory / "files.sha256").read_text().splitlines()
            )
        }
        files = read(directory / "checkpoint.json")["files"]
        assert all(digests[f["path"]] == f["sha256"] for f in files)
        assert (
            any(p.endswith(".safetensors") for p in digests)
            and "tokenizer_config.json" in digests
        )
        config_path = directory / "config.json"
        assert (
            hashlib.sha256(config_path.read_bytes()).hexdigest()
            == digests["config.json"]
        )
        config = read(config_path)
        dim = (
            config.get("head_dim")
            or config["hidden_size"] // config["num_attention_heads"]
        )
        expected_bytes = (
            4096
            * 16
            * 2
            * 2
            * config["num_hidden_layers"]
            * config["num_key_value_heads"]
            * dim
        )
        launches = []
        for n, engine in enumerate(reg["order"][gpu]):
            path = directory / f"{n}-{engine}"
            lease = read(path / "lease-after.json")
            assert lease["physical_index"] == int(gpu) and lease["phase"] == "idle"
            if engine == "riley":
                cli = (path / "server-arguments.txt").read_text().splitlines()
                assert (
                    "--prefill-burst-steady-budget" in cli
                    and "native-bf16-varlen-v1" in cli
                )
                assert (path / "binary.sha256").read_text().split()[
                    0
                ] == "13b589df5c19a33c685ca1388398905f46a6cc566cbd1bed68f06b798080ca9f"
                assert Path(cli[cli.index("--model") + 1]).name == model
                assert cli[cli.index("--kv-blocks") + 1] == "4096"
                assert all(
                    read(path / "shutdown.json")[k] == 0
                    for k in (
                        "active_requests",
                        "waiting_requests",
                        "kv_allocated_blocks",
                    )
                )
                version = (path / "binary.sha256").read_text().split()[0]
            else:
                cli = read(path / "command.json")
                assert Path(cli[2]).name == model and "--enable-prefix-caching" in cli
                assert "--enable-prompt-tokens-details" in cli
                assert (
                    int(cli[cli.index("--kv-cache-memory-bytes") + 1]) == expected_bytes
                )
                version = (path / "version.txt").read_text().strip()
            if read(path / "fixed-exit.json") == 0:
                check_frozen_requests(root, path)
            works = read(path / "natural.json")
            assert len(works) == 2
            launches.append(
                {
                    "order": n,
                    "engine": engine,
                    "version": version,
                    "fixed_exit": read(path / "fixed-exit.json"),
                    "natural": [
                        natural(w, *case)
                        for w, case in zip(
                            works, [(761, 2, 0), (762, 96, 100)], strict=True
                        )
                    ],
                }
            )
        pairs = []
        for start in (0, 2):
            arms = {
                k["engine"]: directory / f"{k['order']}-{k['engine']}"
                for k in launches[start : start + 2]
            }
            a, b = arms["riley"], arms["vllm"]
            if any(k["fixed_exit"] for k in launches[start : start + 2]):
                pairs.append(
                    {
                        "passed": False,
                        "error": "fixed workload incomplete; see launch logs",
                    }
                )
                continue
            try:
                comp = compare(a, b)
            except ValueError as error:
                pairs.append({"passed": False, "error": str(error)})
                continue
            ra, rb = normal(a), normal(b)
            for cell in comp["cells"]:
                key = cell["history"], cell["stream"]
                va, vb = ra[key], rb[key]
                cell["matched_output_texts_of_three"] = sum(
                    x == y for x, y in zip(va["texts"], vb["texts"], strict=True)
                )
                cell["warm_cached_riley_vllm"] = [va["cache"], vb["cache"]]
                cell["cache_matched"] = va["cache"] == vb["cache"] and all(
                    type(v) is int and v > 0 for v in va["cache"]
                )
                cell["first_visible_p50_ms_riley_vllm"] = [
                    median(r["ttft"]) for r in (va, vb)
                ]
                cell["last_visible_p50_ms_riley_vllm"] = [
                    median(r["last"]) for r in (va, vb)
                ]
            comp["passed"] = all(
                c["cache_matched"]
                and c["largest_arrival_drift_ms"] < 3
                and max(c["tpot_ratio"], c["gap_ratio"]) <= 1
                for c in comp["cells"]
            )
            pairs.append(comp)
        result[model] = {
            "gpu": gpu,
            "launches": launches,
            "pairs": pairs,
            "screen_passed": all(p["passed"] for p in pairs),
        }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    print(json.dumps(verify(p.parse_args().folder), indent=2))
