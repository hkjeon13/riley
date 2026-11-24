"""Audit the D128 decode-score intervention against its prespecified crossover."""

import argparse
import json
import tarfile
from hashlib import sha256
from pathlib import Path
from statistics import median

import numpy as np
from agent_workflow import fingerprint
from full_register import case, parse
from verify_register_layers import compare, read, tensors

HERE = Path(__file__).resolve().parent


def require(value, why):
    if not value:
        raise ValueError(why)


def digest(path):
    return path.read_text().split()[0]


def check_receipt(path, gpu, binary):
    lease = read(path / "lease-after.json")
    end = read(path / "shutdown.json")
    require(
        lease["physical_index"] == gpu
        and lease["phase"] == "idle"
        and not lease["other_processes"]
        and lease["allocation_bytes"] == 20 * 2**30
        and lease["gpu"] in (path / "device.csv").read_text()
        and digest(path / "binary.sha256") == binary,
        f"{path}: binary or device",
    )
    require(
        read(path / "ready.json")["accepting"]
        and not any(
            end[k]
            for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
        )
        and not any(end["allocation"].values()),
        f"{path}: lifecycle",
    )
    require(
        "RILEY_NATIVE_ATTENTION mode=page-v1" in (path / "server.log").read_text()
        and all(v == 0 for v in read(path / "metrics.json")["prefix_cache"].values()),
        f"{path}: execution mode or prefix reuse",
    )
    return lease["gpu"]


def inspect_measurement(row, ref, expected, cap):
    m = row["measurement"]
    ids = m["generated_ids"]
    t = m["content_ns"]
    n = len(ids)
    require(
        row["expected"] == expected
        and not row["calls"]
        and m["status"] == 200
        and m["sent_ns"] <= m["request_written_ns"] <= t[0] <= t[-1] <= m["done_ns"]
        and t == sorted(t)
        and n == m["usage"]["completion_tokens"]
        and m["usage"]["prompt_tokens"] == ref["prompt_tokens"]
        and m["prompt_ids_sha256"] == ref["prompt_ids_sha256"],
        "incomplete or mismatched model call",
    )
    require(
        (n == cap and m["finish"] == "length") or (n <= cap and m["finish"] == "stop"),
        "length or terminal condition",
    )
    return {
        "tokens": n,
        "id_sha256": fingerprint(ids),
        "exact": m["finish"] == "stop"
        and fingerprint(parse(row["answer"])) == fingerprint(expected),
        "hf_tokens_identical": ids == ref["generated_ids"],
        "first_content_ms": (t[0] - m["sent_ns"]) / 1e6,
        "last_content_ms": (t[-1] - m["sent_ns"]) / 1e6,
        "tpot_ms": (t[-1] - t[0]) / 1e6 / (n - 1),
    }


def inspect_trace(root, previous, hf, control_binary, candidate_binary):
    require(control_binary != candidate_binary, "identical intervention binaries")
    check_receipt(root / "trace", 6, candidate_binary)
    old = read(previous / "page/witness.json")
    new = read(root / "trace/witness.json")
    out = {}
    for name, spec in read(previous / "register-layer-registration.json")[
        "cases"
    ].items():
        m = new[name]["measurement"]
        ids = m["generated_ids"]
        n = spec["difference_index"]
        earlier = old[name]["measurement"]
        require(
            m["prompt_ids_sha256"] == earlier["prompt_ids_sha256"]
            and ids[:n] == earlier["generated_ids"][:n],
            "different decode prefix",
        )
        for phase in ("prefill", "decode"):
            point = f"{name}-{phase}"
            base = previous / "page/trace" / point
            cand = root / "trace/trace" / point
            a, meta = tensors(base)
            b, selected = tensors(cand)
            require(
                meta == selected
                and np.array_equal(a["query_rope", 0], b["query_rope", 0]),
                point,
            )
            gold = np.load(root / "oracle" / f"{point}.npy", allow_pickle=False).ravel()
            before = compare(a["attention_context", 0], gold)["relative_l2"]
            after = compare(b["attention_context", 0], gold)["relative_l2"]
            expected = read(root / "oracle-comparisons.json")[point]["modes"]["page"][
                "standard"
            ]
            require(
                np.isclose(before, expected["relative_l2"]),
                "independent oracle mismatch",
            )
            identical = (base / "rows.bf16").read_bytes() == (
                cand / "rows.bf16"
            ).read_bytes()
            if phase == "prefill":
                require(identical, "different last prefill")
            i = 0 if phase == "prefill" else n
            require(int(b["logits", -1].argmax()) == ids[i], "different traced choice")
            out[point] = {
                "position": meta[0],
                "input_token": meta[1],
                "whole_row_identical": identical,
                "layer0_q_identical": True,
                "base_attention_relative_l2": before,
                "candidate_attention_relative_l2": after,
                "base_token": earlier["generated_ids"][i],
                "candidate_token": ids[i],
                "hf_token": hf[name]["generated_ids"][i],
            }
    return out


def verify(root, previous):
    for f in ("exit-status", "build.exit", "hf.exit"):
        require((root / f).read_text().strip() == "0", f)
    require((root / "released").is_file(), "campaign incomplete")
    for line in (root / "start-inputs.sha256").read_text().splitlines():
        h, filename = line.split()
        require(sha256((root / filename).read_bytes()).hexdigest() == h, filename)
    for f in (
        "register-score-registration.json",
        "register-score-holdout.json",
        "register-decode-score.patch",
    ):
        require((root / f).read_bytes() == (HERE / f).read_bytes(), f)
    require(
        read(root / "oracle-comparisons.json")
        == read(HERE / "register-kv-verified.json"),
        "changed gold",
    )
    kv_lines = (root / "oracle/kv-check.log").read_text().splitlines()
    require(
        len(kv_lines) == 16 and all(x.endswith(": OK") for x in kv_lines), "KV digests"
    )
    ref = read(root / "hf-holdout.json")
    hf = ref["cases"]
    lease = read(root / "hf-lease.json")
    require(
        lease["gpu"] == ref["gpu"]
        and lease["physical_index"] == 7
        and lease["phase"] == "idle"
        and not lease["other_processes"],
        "HF lifecycle",
    )
    with tarfile.open(HERE / "register-reference-raw.tar.gz") as archive:
        old_ref = json.load(archive.extractfile("./hf.json"))
        require(
            all(old_ref[k] == ref[k] for k in ("torch", "transformers")), "HF versions"
        )
        old_hf = old_ref["cases"]
        weights = archive.extractfile("./model-files.sha256").read()
        require(
            (root / "weights.sha256").read_bytes() == weights,
            "different model manifest",
        )
        checks = (root / "weights.checked").read_text().splitlines()
        require(
            checks
            == [f"{line.split()[1]}: OK" for line in weights.decode().splitlines()],
            "weights changed",
        )
        old_page = json.load(archive.extractfile("./page/witness.json"))
        for name, old in old_hf.items():
            require(
                all(
                    old[k] == hf[name][k]
                    for k in ("generated_ids", "answer", "prompt_ids_sha256")
                ),
                name,
            )
    specs = read(root / "register-score-holdout.json")["cases"]
    expect = {}
    for kind, seed, count in specs:
        _, _, expected, _ = case(kind, seed, count)
        expect[f"{kind}-{seed}"] = expected
    require(set(hf) == set(expect), "HF case inventory")
    base_binary = digest(previous / "binary.sha256")
    candidate_binary = digest(root / "binary.sha256")
    traces = inspect_trace(root, previous, hf, base_binary, candidate_binary)
    results, by_gpu, args, reproducible = {}, {}, None, {}
    spans = {}
    bodies = {}
    traced = read(root / "trace/witness.json")
    for gpu in (6, 7):
        times = {}
        for arm, binary in (("control", base_binary), ("candidate", candidate_binary)):
            path = root / f"gpu{gpu}-{arm}"
            check_receipt(path, gpu, binary)
            argv = (path / "server-arguments.txt").read_text().splitlines()
            if args is None:
                args = argv
            require(
                argv == args and "--prefix-cache-blocks" not in argv,
                "unmatched serving parameters",
            )
            for k, v in (
                ("--native-attention-mode", "page-v1"),
                ("--prefill-chunk-tokens", "256"),
                ("--max-sequence-tokens", "32768"),
                ("--batch-token-budget", "256"),
            ):
                require(argv[argv.index(k) + 1] == v, k)
            records = read(path / "score.json")
            require(set(records) == set(expect), "request inventory")
            rows = {}
            span = []
            for name, expected in expect.items():
                pair = records[name]
                require(
                    set(pair) == {"fixed", "natural"}
                    and expected == hf[name]["expected"],
                    name,
                )
                fixed = pair["fixed"]["measurement"]
                natural = pair["natural"]["measurement"]
                require(
                    len(fixed["generated_ids"]) == 24
                    and fixed["finish"] == "length"
                    and fixed["generated_ids"] == natural["generated_ids"][:24]
                    and len(fixed["content_ns"]) == 24,
                    f"{name}: token-equivalent cap",
                )
                if arm == "control" and name in old_page:
                    require(
                        natural["generated_ids"]
                        == old_page[name]["measurement"]["generated_ids"],
                        "control changed prior output",
                    )
                if arm == "candidate" and name in traced:
                    require(
                        natural["generated_ids"]
                        == traced[name]["measurement"]["generated_ids"],
                        "instrumentation changed candidate output",
                    )
                rows[name] = {}
                for step, row in pair.items():
                    cap = (
                        24
                        if step == "fixed"
                        else (96 if name.startswith("lookup") else 128)
                    )
                    rows[name][step] = inspect_measurement(row, hf[name], expected, cap)
                    m = row["measurement"]
                    span.extend((m["sent_ns"], m["done_ns"]))
                    witness = (m["generated_ids"], row["answer"], m["input_sha256"])
                    require(
                        bodies.setdefault((name, step), m["input_sha256"])
                        == m["input_sha256"],
                        "different HTTP body between arms",
                    )
                    key = (arm, name, step)
                    require(
                        reproducible.setdefault(key, witness) == witness,
                        "device-dependent output",
                    )
            spans[gpu, arm] = min(span), max(span)
            results[f"gpu{gpu}-{arm}"] = rows
            times[arm] = median(v["fixed"]["tpot_ms"] for v in rows.values())
        first, second = (
            ("control", "candidate") if gpu == 6 else ("candidate", "control")
        )
        require(spans[gpu, first][1] < spans[gpu, second][0], "wrong crossover order")
        base_rows = results[f"gpu{gpu}-control"]
        candidate_rows = results[f"gpu{gpu}-candidate"]
        ratios = {
            name: candidate_rows[name]["fixed"]["tpot_ms"]
            / base_rows[name]["fixed"]["tpot_ms"]
            for name in expect
        }
        by_gpu[gpu] = {
            **times,
            "ratio": times["candidate"] / times["control"],
            "median_paired_ratio": median(ratios.values()),
            "per_prompt_ratio": ratios,
        }
    summary = {}
    for name, expected in expect.items():
        summary[name] = {
            "hf_exact": fingerprint(parse(hf[name]["answer"])) == fingerprint(expected)
        }
        for arm in ("control", "candidate"):
            m = results[f"gpu6-{arm}"][name]["natural"]
            summary[name][arm] = {
                k: m[k] for k in ("exact", "hf_tokens_identical", "tokens")
            }
    lost = [
        name
        for name, r in summary.items()
        if r["control"]["exact"] and not r["candidate"]["exact"]
    ]
    gained = [
        name
        for name, r in summary.items()
        if not r["control"]["exact"] and r["candidate"]["exact"]
    ]
    return {
        "requests": 72,
        "independent_prompts": len(expect),
        "cases": summary,
        "trace": traces,
        "fixed_24_tpot_median_ms": by_gpu,
        "correct_lost": lost,
        "correct_gained": gained,
        "original_correct_preserved": not (set(old_hf) & set(lost)),
        "fixed_tpot_gate": all(v["ratio"] <= 1.02 for v in by_gpu.values()),
        "trace_gate": all(
            v["candidate_attention_relative_l2"] < v["base_attention_relative_l2"]
            for k, v in traces.items()
            if k.endswith("-decode")
        ),
        "binaries_sha256": {"control": base_binary, "candidate": candidate_binary},
        "measurements": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score_root", type=Path)
    parser.add_argument("baseline_layer_root", type=Path)
    a = parser.parse_args()
    print(json.dumps(verify(a.score_root, a.baseline_layer_root), indent=2))
