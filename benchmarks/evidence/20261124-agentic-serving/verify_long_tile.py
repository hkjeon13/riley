"""Compare bounded long-context attention routes against real server receipts."""

import argparse
import json
import re
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def receipt(arm, mode):
    args = (arm / "server-arguments.txt").read_text().splitlines()
    if "--native-attention-mode" in args:
        index = args.index("--native-attention-mode")
        if args[index + 1] != mode:
            raise ValueError(f"unexpected attention argument in {arm}")
        del args[index : index + 2]
    if f"RILEY_NATIVE_ATTENTION mode={mode}" not in (arm / "server.log").read_text():
        raise ValueError(f"unexpected attention receipt in {arm}")
    lease = read(arm / "lease-after.json")
    state = read(arm / "shutdown.json")
    if not (
        read(arm / "ready.json")["ready"]
        and lease["phase"] == "idle"
        and not any(state["allocation"].values())
        and not any(
            state[k]
            for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
        )
    ):
        raise ValueError(f"incomplete lifecycle in {arm}")
    binary = (arm / "binary.sha256").read_text().split()[0]
    if binary != (arm.parent / "binary.sha256").read_text().split()[0]:
        raise ValueError(f"binary mismatch in {arm}")
    return args, lease["gpu"], binary


def compare(reference, target, manifest, same_binary):
    base_args, base_gpu, base_bin = receipt(reference, "page-v1")
    test_args, test_gpu, test_bin = receipt(target, "fixed-tile-unsplit-v1")
    if (
        base_args != test_args
        or base_gpu != test_gpu
        or (same_binary and base_bin != test_bin)
    ):
        raise ValueError("configuration, device or paired binary mismatch")
    shapes = []
    for src, left, right in zip(
        read(manifest)["rows"],
        read(reference / "shapes.json"),
        read(target / "shapes.json"),
        strict=True,
    ):
        values = []
        for row in (left, right):
            if (
                row["request_sha256"] != src["request_sha256"]
                or row["next_completion_tokens"] != src["next_completion_tokens"]
            ):
                raise ValueError("public input or output budget mismatch")
            usage = row.get("usage") or {}
            values.append(
                {
                    "successful": row.get("http_status") == 200
                    and row.get("complete")
                    and usage.get("prompt_tokens") == src["encoded_prompt_tokens"]
                    and row.get("prompt_exact")
                    and not row.get("error"),
                    "status": row.get("http_status"),
                    "error": row.get("error"),
                    "cached": (usage.get("prompt_tokens_details") or {}).get(
                        "cached_tokens", 0
                    ),
                    "output": usage.get("completion_tokens"),
                    "finish": row.get("finish"),
                    "first_ms": row.get("first_content_ms"),
                    "last_ms": row.get("last_content_ms"),
                    "answer_sha256": row.get("answer_sha256"),
                }
            )
        base, test = values
        shapes.append(
            {
                "input": src["encoded_prompt_tokens"],
                "budget": src["next_completion_tokens"],
                "page": base,
                "fixed": test,
                "cache_matched": base["cached"] == test["cached"],
                "answer_hash_matches": (base["answer_sha256"] == test["answer_sha256"])
                if base["answer_sha256"] and test["answer_sha256"]
                else None,
            }
        )
    needles = []
    for left, right in zip(
        read(reference / "retrieval.json"), read(target / "retrieval.json"), strict=True
    ):
        if (
            left["expected"] != right["expected"]
            or left["measurement"]["input_sha256"]
            != right["measurement"]["input_sha256"]
        ):
            raise ValueError("retrieval prompt changed")
        pair = {}
        for name, row in (("page", left), ("fixed", right)):
            m = row["measurement"]
            contents = m.get("content_ns", [])
            nums = re.findall(r"\d+", row.get("answer", ""))
            pair[name] = {
                "answer": row.get("answer"),
                "expected": row["expected"],
                "correct": m.get("status") == 200
                and bool(m.get("done_ns"))
                and nums == list(map(str, row["expected"]))
                and row["correct"],
                "input": (m.get("usage") or {}).get("prompt_tokens"),
                "first_ms": (contents[0] - m["sent_ns"]) / 1e6 if contents else None,
                "error": row.get("error") or m.get("stream_error"),
            }
        needles.append(pair)
    if len(needles) != 2 or not shapes:
        raise ValueError("incomplete cohort")
    return {
        "physical_gpu": test_gpu,
        "page_binary": base_bin,
        "fixed_binary": test_bin,
        "shutdown_zero_and_leases_idle": True,
        "shapes": shapes,
        "retrieval": needles,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("previous", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "qwen_7b": compare(
                    args.root / "qwen-page",
                    args.root / "qwen-tile",
                    args.root / "public-cached-shapes.json",
                    True,
                ),
                "llama_8b": compare(
                    args.previous / "native",
                    args.root / "llama-tile",
                    args.root / "public-long-shapes.json",
                    False,
                ),
            },
            indent=2,
        )
    )
