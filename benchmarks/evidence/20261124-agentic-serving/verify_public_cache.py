"""Compare known public token counts to measured GPU cache reuse and costs."""

import argparse
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def latency(row):
    return [(row["content_ns"][i] - row["sent_ns"]) / 1e6 for i in (0, -1)]


def summarize(root):
    manifest = read(root / "public-cached-shapes.json")["rows"]
    cold = read(root / "cold1536/shapes.json")
    result = {}
    for name in ("cold1536", "warm1536", "warm2048"):
        folder = root / name
        if (
            read(folder / "lease-after.json")["phase"] != "idle"
            or not read(folder / "ready.json")["ready"]
        ):
            raise ValueError("incomplete server launch")
        shutdown = read(folder / "shutdown.json")
        if any(shutdown["allocation"].values()) or any(
            shutdown[k]
            for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
        ):
            raise ValueError("server did not drain")
    for cap in ("warm1536", "warm2048"):
        values = []
        for m, c, w in zip(
            manifest, cold, read(root / cap / "continuations.json"), strict=True
        ):
            a, b = w["prior"], w["continuation"]
            if (
                w["case"] != m["index"]
                or a["input_sha256"] != m["prefix_sha256"]
                or c["request_sha256"] != b["input_sha256"]
                or b["input_sha256"] != m["request_sha256"]
            ):
                raise ValueError("changed prompt identity")
            ok = (
                all(r.get("complete") for r in (c, a, b))
                and b["status"] == a["status"] == c["http_status"] == 200
            )
            row = {
                "case": m["index"],
                "mode": cap,
                "intended_cache": m["next_cached_tokens"],
                "whole_block_upper": m["reusable_whole_blocks_tokens"],
                "complete": ok,
            }
            if ok:
                start, end = latency(b)
                elapsed = (a["done_ns"] - a["sent_ns"]) / 1e6
                if (
                    not c["prompt_exact"]
                    or b["usage"]["prompt_tokens"] != m["encoded_prompt_tokens"]
                ):
                    raise ValueError("actual prompt length differs")
                hits = (b["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                row.update(
                    observed_cache=hits,
                    cold_cache=(c["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                    cold_first_ms=c["first_content_ms"],
                    warm_first_ms=start,
                    first_saved_ms=c["first_content_ms"] - start,
                    cold_last_ms=c["last_content_ms"],
                    warm_last_ms=end,
                    last_saved_ms=c["last_content_ms"] - end,
                    preparation_elapsed_ms=elapsed,
                    extra_elapsed_including_preparation_ms=elapsed
                    + end
                    - c["last_content_ms"],
                    warm_output=b["usage"]["completion_tokens"],
                    cold_output=c["usage"]["completion_tokens"],
                    requested_output=m["next_completion_tokens"],
                    completion_digest=b["answer_sha256"],
                )
            else:
                row["errors"] = [r.get("error") for r in (c, a, b)]
            values.append(row)
        result[cap] = values
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(summarize(p.parse_args().root), indent=2))
