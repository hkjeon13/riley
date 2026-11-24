"""Match the experimental binary, token lengths, retrieval answers and resource receipts."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def summarize(root, name):
    arm = root / name
    binary = (root / "binary.sha256").read_text().split()[0]
    if (arm / "binary.sha256").read_text().split()[0] != binary or read(
        arm / "lease-after.json"
    )["phase"] != "idle":
        raise ValueError("incomplete or mismatched binary receipt")
    s = read(arm / "shutdown.json")
    zero = all(
        s[k] == 0
        for k in ("waiting_requests", "active_requests", "kv_allocated_blocks")
    ) and not any(s["allocation"].values())
    retrieval = []
    for row in read(arm / "retrieval.json"):
        measurement = row["measurement"]
        answer = row.get("answer", "")
        expected = list(map(str, row["expected"]))
        correct = all(n in answer for n in expected) and answer.index(
            expected[0]
        ) < answer.index(expected[1])
        if correct != row["correct"]:
            raise ValueError("retrieval validator differs")
        result = {
            "correct": correct,
            "status": measurement.get("status"),
            "error": row.get("error"),
        }
        if measurement.get("stream_error"):
            result["stream_error"] = measurement["stream_error"]
        if measurement.get("done_ns") and measurement.get("content_ns"):
            result.update(
                prompt_tokens=measurement["usage"]["prompt_tokens"],
                completion_tokens=measurement["usage"]["completion_tokens"],
                first_ms=(measurement["content_ns"][0] - measurement["sent_ns"]) / 1e6,
                last_ms=(measurement["content_ns"][-1] - measurement["sent_ns"]) / 1e6,
            )
        retrieval.append(result)
    if len(retrieval) != 2:
        raise ValueError("missing retrieval cases")
    results = {
        "binary_sha256": binary,
        "scope": "isolated experimental page attention binary on Llama 3.1 8B, no production promotion",
        "retrieval": retrieval,
        "shutdown_zero": zero,
    }
    if name == "native":
        cases = read(root / "public-long-shapes.json")["rows"]
        rows = read(arm / "shapes.json")
        if [r["request_sha256"] for r in cases] != [r["request_sha256"] for r in rows]:
            raise ValueError("missing public cases")
        results["public_cases"] = [
            {
                k: r.get(k)
                for k in (
                    "index",
                    "next_prompt_tokens",
                    "next_completion_tokens",
                    "http_status",
                    "complete",
                    "prompt_exact",
                    "output_exact",
                    "usage",
                    "finish",
                    "error",
                    "first_content_ms",
                    "last_content_ms",
                )
            }
            for r in rows
        ]
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("arm", choices=("probe", "deadline", "native"))
    a = p.parse_args()
    print(json.dumps(summarize(a.root, a.arm), indent=2))
