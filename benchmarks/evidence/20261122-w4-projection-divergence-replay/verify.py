"""Check archived exactness and summarize fixed-history score comparisons."""

from collections import Counter
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import sys


ARMS = ("baseline", "attention31", "fp32_down21")
VARIANTS = {"attention31": "o31-serving", "fp32_down21": "fp32-down-serving"}
BACKENDS = ("sdpa", "eager")
SUITES = {"long": 18, "holdout": 11}
CONTROL = {("long", "3"), ("long", "14"), ("holdout", "numbers-1023")}


def read(path):
    return json.loads(path.read_bytes())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def hash_ids(ids):
    return sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def common_prefix(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def normalize(ids, finish, eos):
    return ids[:-1] if finish == "stop" and ids and ids[-1] in eos else ids


def stage_inputs(path):
    checksums = dict(line.split("  ", 1)[::-1] for line in (path / "result.sha256").read_text().splitlines())
    expected = {"replay.json", "replay.log", "checkpoint-verify.log",
                *(f"gpu{index}-{point}.json" for index in (6, 7) for point in ("before", "after"))}
    require(set(checksums) == expected, "unexpected remote artifact set")
    for filename, expected_sha in checksums.items():
        data = (gzip.decompress((path / "replay.log.gz").read_bytes()) if filename == "replay.log"
                else (path / filename).read_bytes())
        require(sha256(data).hexdigest() == expected_sha, f"remote artifact differs: {filename}")
    require((path / "checkpoint-verify.log").read_text().strip() == "checkpoint verify: verified",
            "missing integrity check of model files")

    holder = {}
    for index in (6, 7):
        receipts = [read(path / f"gpu{index}-{point}.json") for point in ("before", "after")]
        states = [(r["physical_index"], r["pid"], r["start_ticks"], r["gpu"]) for r in receipts]
        require(states[0] == states[1] and states[0][0] == index, f"GPU {index} holder changed")
        require(all(r["phase"] == "idle" and not r["other_processes"]
                    and r["allocation_bytes"] == r["torch_reserved_bytes"] == 20 * 2**30 for r in receipts),
                f"GPU {index} did not return to its 20 GiB reservation")
        holder[str(index)] = {"pid": states[0][1], "start_ticks": states[0][2], "gib": 20}
    return holder


def verify_scores(case, point, eos, baseline, reference):
    offset = point["step"]
    require(offset < len(baseline) and point["history_sha256"] == hash_ids(baseline[:offset])
            and point["baseline_id"] == baseline[offset], f"fixed history changed at {case}:{offset}")
    normalized = normalize(reference["hf_tokens"], reference["hf_finish"], eos)
    archived_id = normalized[offset] if offset < len(normalized) and normalized[:offset] == baseline[:offset] else None
    require(point["archived_hf_id_on_same_history"] == archived_id, "archived HF history alignment changed")
    for backend in BACKENDS:
        row = point[backend]
        keys, values = row["selected_scores"].keys(), row["selected_scores"].values()
        require(str(point["baseline_id"]) in keys and all(str(i) in keys for i in point["candidate_ids"].values())
                and math.isfinite(row["max_logit"]) and math.isfinite(row["margin"])
                and row["margin"] >= 0 and all(math.isfinite(value) and value <= row["max_logit"] for value in values)
                and row["argmax_ids"] and row["argmax_ids"] == sorted(set(row["argmax_ids"]))
                and (row["margin"] == 0) == (len(row["argmax_ids"]) > 1)
                and all((int(token) in row["argmax_ids"]) == (value == row["max_logit"])
                        for token, value in row["selected_scores"].items()), f"invalid {backend} score bounds")
        if point["archived_control"]:
            prior = reference["first_divergence"]
            require(point["archived_control"] == prior and prior["offset"] == offset
                    and all(row["selected_scores"][key] == value
                            for key, value in prior[f"hf_{backend}_replay"]["selected_scores"].items()),
                    "archived replay score changed")
    return archived_id is None or archived_id in point["sdpa"]["argmax_ids"]


def exact_counts(rows, reference):
    field = f"{reference}_exact"
    return {
        arm: {"exact": sum(row["matches"][arm][field] for row in rows),
              "gained": [row["case"] for row in rows if not row["matches"]["baseline"][field] and row["matches"][arm][field]],
              "lost": [row["case"] for row in rows if row["matches"]["baseline"][field] and not row["matches"][arm][field]]}
        for arm in ARMS
    }


def verify(path):
    holder = stage_inputs(path)
    raw = read(path / "replay.json")
    require(raw["checkpoint_sha256"] == "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f",
            "wrong checkpoint")
    root = path.parent
    source = root / "20261119-w4-layer-trace/shared-prefix-audit/decode-row-capture"
    archives = {arm: read(source / filename / "verified-summary.json") for arm, filename in VARIANTS.items()}
    sections, decisions = {}, []
    controls = set()
    same_history_misses = []
    for suite, expected in SUITES.items():
        rows = [row for row in raw["cases"] if row["suite"] == suite]
        require(len(rows) == expected, f"incomplete {suite}")
        fixture = root / f"20261119-w4-{suite}"
        hf = read(fixture / "hf/reference.json")
        eos = hf["eos_token_ids"]
        require(raw["runtime"] == hf["runtime"], "reference runtime drifted")
        old = json.loads(gzip.decompress((fixture / "riley/capture.json.gz").read_bytes()))["baseline"]
        vllm = json.loads(gzip.decompress((fixture / "vllm/results.json.gz").read_bytes()))["reference"]
        known_splits = {arm: {row["case"]: row["token_common_prefix"]
                              for row in doc["suites"][suite]["baseline_to_candidate"]["c1"]["differences"]}
                        for arm, doc in archives.items()}
        for row, gold, ref, vref in zip(rows, old, hf["cases"], vllm, strict=True):
            case = row["case"]
            require(case == str(ref.get("label", ref["case"])), "cases changed order")
            require(row["input_sha256"] == hash_ids(gold["prompt_tokens"]) and vref["prompt_tokens"] == gold["prompt_tokens"]
                    and row["input_tokens"] == ref["prompt_tokens"] == len(gold["prompt_tokens"]), "prompt IDs changed")
            baseline = normalize(gold["tokens"], gold["finish"], eos)
            hf_ids = normalize(ref["hf_tokens"], ref["hf_finish"], eos)
            require((row["matches"]["baseline"]["hf_exact"], row["matches"]["baseline"]["hf_prefix"])
                    == (baseline == hf_ids, common_prefix(baseline, hf_ids)), "baseline HF comparison changed")
            vref_ids = normalize(vref["tokens"], vref["finish"], eos)
            require((row["matches"]["baseline"]["vllm_exact"], row["matches"]["baseline"]["vllm_prefix"])
                    == (baseline == vref_ids, common_prefix(baseline, vref_ids)), "baseline vLLM comparison changed")
            for arm in VARIANTS:
                archived = next(item for item in archives[arm]["suites"][suite]["hf"]["cases"] if item["case"] == case)
                require((row["matches"][arm]["hf_exact"], row["matches"][arm]["hf_prefix"])
                        == (archived["candidate"]["exact"], archived["candidate"]["common_prefix"]), "HF comparison changed")
            steps = [point["step"] for point in row["points"]]
            require(steps == sorted(set(steps)), "duplicated or unordered replay points")
            found = {arm: p["step"] for p in row["points"] for arm in p["candidate_ids"]}
            require(found == {arm: data[case] for arm, data in known_splits.items() if case in data}, "split coverage changed")
            for point in row["points"]:
                if not verify_scores(case, point, eos, baseline, ref):
                    same_history_misses.append([suite, case, point["step"]])
                if point["archived_control"]:
                    controls.add((suite, case))
                for arm, candidate in point["candidate_ids"].items():
                    scores = {backend: point[backend]["selected_scores"] for backend in BACKENDS}
                    baseline_key = str(point["baseline_id"])
                    delta = {backend: values[str(candidate)] - values[baseline_key] for backend, values in scores.items()}
                    preference = {name: "candidate" if value > 0 else "baseline" if value < 0 else "tie"
                                  for name, value in delta.items()}
                    decisions.append({"suite": suite, "case": case, "step": point["step"], "arm": arm,
                                      "baseline_id": point["baseline_id"], "candidate_id": candidate,
                                      "archived_hf_id_on_same_history": point["archived_hf_id_on_same_history"],
                                      "candidate_minus_baseline": delta, "preference": preference,
                                      "hf_transition": [row["matches"][tag]["hf_exact"] for tag in ("baseline", arm)],
                                      "vllm_transition": [row["matches"][tag]["vllm_exact"] for tag in ("baseline", arm)]})
        sections[suite] = {ref: exact_counts(rows, ref) for ref in ("hf", "vllm")}
    require(controls == CONTROL and len(raw["cases"]) == sum(SUITES.values()), "missing positive controls or cases")
    points = [point for row in raw["cases"] for point in row["points"]]
    return {"requests": len(raw["cases"]), "score_positions": len(points),
            "divergence_positions": sum(bool(point["candidate_ids"]) for point in points),
            "changed_inputs": sum(any(point["candidate_ids"] for point in row["points"]) for row in raw["cases"]),
            "same_history_sdpa_checks": sum(point["archived_hf_id_on_same_history"] is not None for point in points),
            "runtime": raw["runtime"], "same_history_sdpa_archived_choice_misses": same_history_misses,
            "reservations": holder, "exact_counts": sections, "decisions": decisions,
            "first_token_decisions": sum(row["step"] == 0 for row in decisions),
            "preferences": dict(sorted(Counter("/".join(row["preference"].values()) for row in decisions).items()))}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
