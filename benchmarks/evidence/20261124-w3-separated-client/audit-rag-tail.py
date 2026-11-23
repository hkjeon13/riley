"""Find the requests behind the two valid GPU 7 RAG p95 misses."""

from hashlib import sha256
import json
from pathlib import Path
import tarfile


HERE = Path(__file__).resolve().parent


def read_rows(archive, checksums, engine, label):
    path = f"gpu7/coder14-{engine}/{label}.json"
    raw = archive.extractfile(path).read()
    if sha256(raw).hexdigest() != checksums[path]:
        raise ValueError(f"capture checksum differs for {engine}/{label}")
    rows = json.loads(raw)["measured"]
    by_id = {row["request_index"]: row for row in rows}
    if len(by_id) != len(rows) or len(rows) != 64:
        raise ValueError(f"expected 64 distinct requests in {engine}/{label}")
    return by_id


def count_pending(row, run, boundary):
    at = row["sent_at"]
    return sum(other["sent_at"] < at < other["start"] + other[boundary] for other in run.values())


def audit():
    pairs = json.loads((HERE / "gpu7-verified.json").read_bytes())["by_gpu"]["7"]
    labels = [label for label, pair in pairs.items()
              if label.startswith("rag-") and pair["valid_matched_comparison"] and not pair["ttft_p95_stage_met"]]
    results = {}
    with tarfile.open(HERE / "initial-results.tar.gz", "r:gz") as archive:
        checksums = {path: digest for digest, path in
                     (line.split(maxsplit=1) for line in archive.extractfile("result.sha256").read().decode().splitlines())}
        for label in labels:
            riley, vllm = (read_rows(archive, checksums, engine, label) for engine in ("riley", "vllm"))
            keys = ("input_sha256", "prompt_tokens", "completion_tokens", "scheduled_write_ns")
            if riley.keys() != vllm.keys() or any(riley[i][k] != vllm[i][k] for i in riley for k in keys):
                raise ValueError(f"replayed requests differ for {label}")
            ordered = sorted(riley.values(), key=lambda row: row["ttft"])
            p95_index = int(len(ordered) * .95)
            if ordered[p95_index]["ttft"] * 1000 != pairs[label]["riley"]["ttft_p95_ms"]:
                raise ValueError(f"p95 row disagrees with the verified comparison for {label}")
            rows = []
            for rank, row in enumerate(ordered[p95_index:], start=p95_index + 1):
                peer = vllm[row["request_index"]]
                rows.append({"riley_rank_ascending": rank, "request_index": row["request_index"],
                             "prompt_tokens": row["prompt_tokens"], "riley_ttft_ms": row["ttft"] * 1000,
                             "vllm_ttft_ms": peer["ttft"] * 1000, "matched_delta_ms": (row["ttft"] - peer["ttft"]) * 1000,
                             "preceding_request_pending_completion": {"riley": count_pending(row, riley, "e2e"),
                                                                         "vllm": count_pending(peer, vllm, "e2e")},
                             "preceding_request_pending_first_content": {"riley": count_pending(row, riley, "ttft"),
                                                                            "vllm": count_pending(peer, vllm, "ttft")}})
            results[label] = rows
    return results


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))
