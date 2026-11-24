"""Verify the queued fixed-tile comparison against its build and frozen workloads."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import tarfile
from tempfile import TemporaryDirectory

FROZEN = "20261124-w3-short-prefill-continuation"
HISTORY = "20261124-w3-c32-rag-history"
PACKAGE = "20261124-w4-pointer-syntax/a100/cpu-evidence/report.json"
ORDERS = {"6": ["control", "candidate", "control"], "7": ["candidate", "control", "candidate"]}
BINARY_NAMES = {"riley-diagnostic", "flashinfer-attention-test"}
MANIFESTS = (
    "dispatch.sha256", "build-result.sha256", "build-wait-script.sha256",
    "campaign-dispatch.sha256", "campaign-wait-script.sha256", "runtime.sha256", "result.sha256",
)
RECEIPTS = {
    "build-wait.pid", "build-wait.started", "build-wait.exit", "build.exit", "build.log",
    "campaign-wait.exit", "campaign.pid", "campaign.started", "campaign.exit",
    "campaign.log", "compiler.log", "gpu-test-build.log", *MANIFESTS,
}
ENTRY = re.compile(r"([a-f0-9]{64})  ((?:[A-Za-z0-9][A-Za-z0-9_.-]*/)*[A-Za-z0-9][A-Za-z0-9_.-]*)")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def verify_archive(archive, root, reg):
    names = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for entry in bundle.getmembers():
            require(entry.isfile() and entry.name not in names, "duplicate or nonregular archive entry")
            names.add(entry.name)
        bundle.extractall(root, filter="data")
    expected = RECEIPTS.copy()
    binaries = {}
    for manifest in MANIFESTS:
        entries = {}
        for line in (root / manifest).read_text().splitlines():
            match = ENTRY.fullmatch(line)
            require(match, f"invalid checksum member in {manifest}")
            fingerprint, name = match.groups()
            require(name not in entries, f"duplicate checksum member in {manifest}: {name}")
            entries[name] = fingerprint
            if name in BINARY_NAMES:
                require(manifest in ("build-result.sha256", "runtime.sha256"),
                        f"invalid binary receipt in {manifest}")
                require(binaries.setdefault(name, fingerprint) == fingerprint,
                        f"runtime binary differs from the build: {name}")
            else:
                require(name in names and digest(root / name) == fingerprint,
                        f"archived checksum differs: {name}")
                expected.add(name)
        require(entries and (manifest not in ("build-result.sha256", "runtime.sha256")
                             or BINARY_NAMES <= entries.keys()), f"incomplete {manifest}")
    require(expected == names, "archive contents differ from the registered receipts")
    fingerprints = {
        "dispatch.sha256": "build_dispatch_sha256",
        "campaign-dispatch.sha256": "campaign_dispatch_sha256",
        "wide-fixed-tile.patch": "patch_sha256",
        "schedules.json": "schedules_sha256",
    }
    for filename, key in fingerprints.items():
        require(digest(root / filename) == reg[key], f"registration differs at {filename}")
    for process in ("build-wait", "campaign"):
        key = process.replace("-", "_")
        require(int((root / f"{process}.pid").read_text()) == reg[f"{key}_pid"]
                and (root / f"{process}.started").read_text().strip() == reg[f"{key}_started_at"],
                f"unbound process receipt: {process}")
    for phase in ("build-wait", "build", "campaign-wait", "campaign"):
        require((root / f"{phase}.exit").read_text().strip() == "0", f"incomplete phase {phase}")
    return binaries


def expected_server(plan, reg, mode):
    argv = list(plan["server"])
    budget = reg["row_budgets"][mode]
    for key, value in {
        "--max-sequence-tokens": str(reg["max_sequence_tokens"]),
        "--batch-token-budget": str(budget),
        "--prefill-chunk-tokens": str(budget),
    }.items():
        argv[argv.index(key) + 1] = value
    if mode == "candidate":
        argv[argv.index("--batch-shape-buckets") + 1] += f",{budget}"
    return {**plan, "server": argv}


def check_reservations(root, frozen, gpu):
    old = json.load(frozen.extractfile(f"gpu{gpu}/after.json"))
    for phase in ("before", "after-kernel", "between-1", "between-2", "after"):
        current = read(root / f"gpu{gpu}/{phase}.json")
        require(current["phase"] == "idle" and not current["other_processes"]
                and current["allocation_bytes"] == current["torch_reserved_bytes"] == 20 * 2**30
                and all(current[key] == old[key] for key in ("gpu", "physical_index", "pid", "start_ticks")),
                f"original GPU reservation differs for {gpu}/{phase}")


def compare_short_vllm(workload, reference, reg, percentile, release_timed):
    name = "c32-riley-arrivals"
    rows, current = workload[name]
    archived = json.load(reference.extractfile(f"gpu7/coder14-vllm/{name}.json"))
    previous, stats = archived["measured"], archived["statistics"]
    keys = ("request_index", "input_sha256", "prompt_tokens", "scheduled_write_ns")
    pairs = list(zip(rows, previous, strict=True))
    require(stats["requests"] == stats["ok"] == len(previous) and not stats["errors"]
            and not stats["rejected_429"] and not archived["length_mismatches"]
            and all(all(a[key] == b[key] for key in keys) for a, b in pairs),
            "saved C32 vLLM input, schedule, or outcome differs")
    vllm_tpot = percentile([
        sum(row["content_gaps_s"]) / max(row["completion_tokens"] - 1, 1) * 1000
        for row in previous
    ], .5)
    require(vllm_tpot == stats["tpot_p50_ms"], "saved C32 vLLM TPOT differs from its samples")
    errors = {
        "riley": percentile([abs(row["write_error_ms"]) for row in rows], .95),
        "vllm": percentile([abs(row["write_error_ms"]) for row in previous], .95),
        "between": percentile([abs(a["write_error_ms"] - b["write_error_ms"]) for a, b in pairs], .95),
    }
    eligible = release_timed and max(errors.values()) <= reg["write_error_p95_limit_ms"]
    ratio = current["tpot_p50_ms"] / vllm_tpot if eligible else None
    return {"label": name, "abs_write_error_p95_ms": errors, "tpot_p50_ratio": ratio,
            "target_at_most_1_03": ratio <= 1.03 if eligible else None}


def verify(root, archive, reg, evidence):
    binaries = verify_archive(archive, root, reg)
    package_path = evidence / PACKAGE
    package = read(package_path)
    frozen_path = evidence / FROZEN / "raw-results.tar.gz"
    parent = read(evidence / FROZEN / "registration.json")
    require(reg["orders"] == ORDERS and reg["row_budgets"] == {"control": 1024, "candidate": 2048}
            and reg["max_sequence_tokens"] == 4096, "row/context trial or order changed")
    require(reg["source_revision"] == package["source_revision"]
            and reg["package_image_id"] == package["image_id"]
            and reg["package_cpu_report_sha256"] == digest(package_path)
            and reg["frozen_replay_archive_sha256"] == digest(frozen_path), "registered baseline differs")
    require(all(reg[key] == parent[key] for key in (
        "holdout_requests_per_arm", "write_error_p95_limit_ms", "decode_tpot_p50_max_regression_pct_exclusive",
        "short_ttft_p50_max_regression_pct_exclusive", "short_throughput_max_drop_pct_exclusive",
        "short_start_target_s", "rag_start_target_s", "history_release_tolerance_s",
    )), "frozen comparison gates changed")

    history = runpy.run_path(str(evidence / HISTORY / "verify.py"))
    native = runpy.run_path(str(evidence / history["NATIVE"] / "verify.py"))
    baseline = runpy.run_path(str(evidence / history["BASELINE"] / "verify.py"))
    repeated = runpy.run_path(str(evidence / "20261124-w3-c32-native-repeat/verify.py"))
    log = (root / "campaign.log").read_text()
    kernel = (root / "gpu6/flashinfer-attention-test.log").read_text()
    require(re.search(r"^END_KERNEL gpu6 \S+ exit=0$", log, re.M)
            and "test flashinfer_rows_are_batch_invariant_and_close_to_the_page_path ..." in kernel
            and "flashinfer query_heads=40 key_value_heads=8 pages=60" in kernel
            and "kv_chunk=512 generated=256 done" in kernel
            and "1 passed; 0 failed" in kernel, "FlashInfer boundary test has no successful result")
    events = re.findall(r"^(START|END) gpu([67]) ([123]) (control|candidate) \S+(?: exit=(\d+))?$", log, re.M)
    outcomes = {}
    with (tarfile.open(frozen_path, "r:gz") as frozen,
          tarfile.open(evidence / repeated["VLLM"], "r:gz") as vllm):
        for name in ("client.py", "holdout.py", "replay.py", "schedules.json", "cpu-placement.json"):
            require((root / name).read_bytes() == frozen.extractfile(name).read(), f"replay asset differs: {name}")
        for gpu, order in ORDERS.items():
            expected = [(kind, str(index), mode, code) for index, mode in enumerate(order, 1)
                        for kind, code in (("START", ""), ("END", "0"))]
            require([(k, i, m, c) for k, g, i, m, c in events if g == gpu] == expected, f"arm order differs on GPU {gpu}")
            check_reservations(root, frozen, gpu)

            def previous(name):
                return json.load(frozen.extractfile(f"gpu{gpu}/rag/1/{order[0]}/{name}.json"))

            old_plan, old_receipt, expected_tokens = (
                previous(name) for name in ("plan", "receipt", "token-id-holdout")
            )
            old_rows = {
                name: [(row["request_index"], row["output_sha256"]) for row in previous(name)["measured"]]
                for name in old_plan["cells"]
            }
            arms = []
            for index, mode in enumerate(order, 1):
                path = root / f"gpu{gpu}/rag/{index}/{mode}"
                plan = expected_server(old_plan, reg, mode)
                replay = {**reg, "mode_flag": ""}
                workload, tokens, shapes = native["verify_arm"](
                    path, plan, old_receipt, replay, binaries["riley-diagnostic"], "control",
                    int(gpu) + 38, baseline,
                )
                repeated["verify_metrics"](path, workload, baseline["percentile"])
                stats = baseline["current_batch_shapes"]({"metrics": read(path / "shutdown.json")})
                require(max(stats) == reg["row_budgets"][mode], f"server shape ceiling differs in {path}")
                wide = stats.get(2048, (0, 0, 0))
                shapes.update({"wide_2048_steps": wide[0], "wide_2048_gpu_time_ms": wide[1] / 1e6})
                history_file = path / "history.json"
                timing = history["check_history"](history_file, tuple(plan["cells"]), reg)
                rag_started = read(history_file)["launch_s"][history["RAG"][0]]
                timing["first_rag_release_s"] = rag_started
                timing["rag_release_timed"] = abs(rag_started - reg["rag_start_target_s"]) <= reg["history_release_tolerance_s"]
                timing["release_timed"] = timing["release_timed"] and timing["rag_release_timed"]
                arms.append({
                    "mode": mode, "workload": workload, "batch_shapes": shapes,
                    "history": timing,
                    "changed_output_indexes_vs_baseline": {
                        name: [new["request_index"] for (index, old), new in zip(old_rows[name], rows, strict=True)
                               if index != new["request_index"] or old != new["output_sha256"]]
                        for name, (rows, _) in workload.items()
                    },
                    "token_ids_match_baseline": tokens == expected_tokens,
                    "token_ids_batch_invariant": tokens["c1"] == tokens["c8"],
                })
            alias = tuple("reserve" if mode == "candidate" else mode for mode in order)
            comparisons = repeated["compare"](arms, alias, reg, baseline)
            for comparison in comparisons.values():
                for key in ("first_mode", "second_mode"):
                    if comparison[key] == "reserve":
                        comparison[key] = "candidate"
            outcomes[gpu] = {"order": order, "comparisons": comparisons,
                             "arms": [{k: v for k, v in arm.items() if k != "workload"} for arm in arms]}
            if gpu == "7":
                outcomes[gpu]["frozen_vllm"] = [
                    repeated["compare_vllm"](arm["workload"], history["RAG"], vllm, reg,
                                              baseline["percentile"], arm["history"]["release_timed"])
                    for arm in arms
                ]
                outcomes[gpu]["frozen_vllm_c32"] = [
                    compare_short_vllm(arm["workload"], vllm, reg, baseline["percentile"],
                                       arm["history"]["release_timed"])
                    for arm in arms
                ]
    return {"source_revision": reg["source_revision"], "binaries": binaries, "gpus": outcomes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("registration", type=Path)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    with TemporaryDirectory() as scratch:
        print(json.dumps(verify(Path(scratch), args.archive, read(args.registration), args.evidence_root),
                         indent=2, allow_nan=False))
