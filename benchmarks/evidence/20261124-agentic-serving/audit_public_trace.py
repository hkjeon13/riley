"""Audit token budgets and prospectively predict long tool waits from public metadata."""

import argparse
import bisect
import collections
import gzip
import hashlib
import json
import math
import tarfile
from pathlib import Path


def quantile(values, q):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)]


def distribution(values):
    return {
        "n": len(values),
        "p50": quantile(values, 0.5),
        "p90": quantile(values, 0.9),
        "p95": quantile(values, 0.95),
    }


def fits(call):
    tok = call.get("tokens") or {}
    prompt, output = tok.get("prompt"), tok.get("completion")
    if any(type(v) is not int or v < 0 for v in (prompt, output)):
        return None
    return output <= 384 and prompt + output <= 8192


def audit(path, registration):
    with path.open("rb") as archive:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: archive.read(1024 * 1024), b""):
            hasher.update(chunk)
        digest = hasher.hexdigest()
    if digest != registration["sha256"]:
        raise ValueError("public release digest mismatch")
    want = {f"date=2026-06-01/shard-{i:04}.jsonl.gz" for i in range(8)}
    found = set()
    counts = collections.Counter()
    prompts = []
    outputs = []
    deltas = []
    chain = []
    ratios = []
    train = collections.defaultdict(list)
    heldout = []
    cases = []
    large = []
    long_context = []
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.name not in want:
                continue
            found.add(member.name)
            with gzip.GzipFile(fileobj=archive.extractfile(member)) as lines:
                for line in lines:
                    session = json.loads(line)
                    counts["sessions"] += 1
                    for turn in session["turns"]:
                        counts["turns"] += 1
                        calls = sorted(turn["llm_calls"], key=lambda c: c["timestamp"])
                        chain.append(len(calls))
                        times = [c["timestamp"] for c in calls]
                        for call in calls:
                            counts["llm_calls"] += 1
                            if call.get("result") != "Success":
                                counts["llm_not_success"] += 1
                                continue
                            ok = fits(call)
                            if ok is None:
                                counts["success_missing_token_counts"] += 1
                                continue
                            t = call["tokens"]
                            prompts.append(t["prompt"])
                            outputs.append(t["completion"])
                            counts["success_known_tokens"] += 1
                            counts["within_current_limits"] += ok
                            counts["within_32k_and_2048"] += (
                                t["prompt"] + t["completion"] <= 32768
                                and t["completion"] <= 2048
                            )
                            if t["completion"] > 0:
                                ratios.append(t["prompt"] / t["completion"])
                            counts["within_128k_and_2048"] += (
                                t["prompt"] + t["completion"] <= 131072
                                and t["completion"] <= 2048
                            )
                            cached = t.get("cached")
                            if type(cached) is int and 0 <= cached <= t["prompt"]:
                                deltas.append(t["prompt"] - cached)
                        for batch in turn["tool_batches"]:
                            counts["batches"] += 1
                            stamp = batch.get("timestamp", "")
                            # In the released data several LLM calls in a turn can
                            # share the same message_id. Require two unique temporal
                            # neighbors and a matching request id. Ties are ambiguous.
                            pos = bisect.bisect_left(times, stamp)
                            if (
                                pos == 0
                                or pos >= len(calls)
                                or times[pos] == stamp
                                or (
                                    pos + 1 < len(times)
                                    and times[pos + 1] == times[pos]
                                )
                                or (pos >= 2 and times[pos - 2] == times[pos - 1])
                            ):
                                counts["ambiguous_or_unbracketed_batch_time"] += 1
                                continue
                            previous, nxt = calls[pos - 1], calls[pos]
                            if (
                                not batch.get("request_id")
                                or previous.get("message_id") != batch["request_id"]
                            ):
                                counts["previous_id_mismatch"] += 1
                                continue
                            duration = batch.get("duration_ms")
                            names = tuple(
                                sorted(
                                    f["name"].lower()
                                    for f in batch["function_calls"]
                                    if f.get("name")
                                )
                            )
                            if (
                                not stamp.startswith("2026-06-01T")
                                or not names
                                or nxt.get("initiator_type") != "agent"
                                or previous.get("result") != "Success"
                                or nxt.get("result") != "Success"
                                or not isinstance(duration, (int, float))
                                or not math.isfinite(duration)
                                or duration < 0
                            ):
                                continue
                            if stamp[11:13] < "12":
                                hours, minutes, seconds = map(
                                    int, stamp[11:19].split(":")
                                )
                                day_ms = (
                                    3600 * hours + 60 * minutes + seconds + 1
                                ) * 1000
                                if day_ms + duration < 12 * 3600 * 1000:
                                    train[names].append(duration)
                            else:
                                heldout.append(
                                    (
                                        names,
                                        duration,
                                        fits(nxt),
                                        nxt["tokens"].get("prompt"),
                                        nxt["tokens"].get("completion"),
                                        nxt["tokens"].get("cached"),
                                    )
                                )
            if found == want:
                break
    if found != want:
        raise ValueError("registered shard missing")
    hints = {
        names: quantile(values, 0.1)
        for names, values in train.items()
        if len(values) >= 20
    }
    evaluated = []
    for names, actual, fit, pt, ct, cached in heldout:
        forecast = hints.get(names)
        selected = forecast is not None and forecast >= 350
        evaluated.append((selected, actual, fit))
        extended = (
            not fit
            and isinstance(pt, int)
            and isinstance(ct, int)
            and pt > 0
            and ct > 0
            and pt + ct <= 32768
            and ct <= 2048
        )
        long = (
            isinstance(pt, int)
            and isinstance(ct, int)
            and 32768 < pt + ct <= 131072
            and 0 < ct <= 2048
        )
        if (fit or extended or long) and forecast is not None:
            # Deterministic random-ranking sample avoids selecting for favorable errors.
            rank = hashlib.sha256(
                json.dumps([len(evaluated), names, actual, pt, ct]).encode()
            ).hexdigest()
            output = long_context if long else (cases if fit else large)
            output.append(
                (
                    rank,
                    {
                        "actual_ms": actual,
                        "estimated_ms": forecast,
                        "next_prompt_tokens": pt,
                        "next_cached_tokens": cached,
                        "next_completion_tokens": ct,
                    },
                )
            )

    def tally(rows):
        return {
            "batches": len(rows),
            "forecast_long": sum(r[0] for r in rows),
            "actual_long": sum(r[1] >= 350 for r in rows),
            "forecast_long_actual_short": sum(r[0] and r[1] < 350 for r in rows),
            "wait_ms": distribution([r[1] for r in rows]),
        }

    return {
        "archive_sha256": digest,
        "shards": sorted(found),
        "counts": counts,
        "prompt_tokens": distribution(prompts),
        "completion_tokens": distribution(outputs),
        "uncached_tokens_where_known": distribution(deltas),
        "input_to_output_ratio_positive_output": distribution(ratios),
        "llm_calls_per_user_turn": distribution(chain),
        "training_batches": sum(map(len, train.values())),
        "usable_tool_shapes": len(hints),
        "heldout": tally(evaluated),
        "heldout_within_current_limits": tally([r for r in evaluated if r[2]]),
        "long_context_sample": [v for _, v in sorted(long_context)[:4]],
        "extended_context_sample": [v for _, v in sorted(large)[:8]],
        "timing_replay_sample": [v for _, v in sorted(cases)[:16]],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    args = p.parse_args()
    config = json.loads(
        Path(__file__).with_name("public-trace-registration.json").read_text()
    )
    print(json.dumps(audit(args.archive, config), indent=2))
