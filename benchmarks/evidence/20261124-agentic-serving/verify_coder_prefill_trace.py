"""Close the client-to-first-output wall clock and score page minus tile on matching prefills."""

import argparse
import json
from pathlib import Path

from join_trace import join
from verify_model_decode import check_frozen_requests, natural, read

SEEDS = (761, 762, 833, 861, 902, 931)
STAGES = (
    "client_to_render_ms",
    "normalize_ms",
    "render_to_submit_after_normalize_ms",
    "submit_to_queued_ms",
    "command_queue_ms",
    "admission_ms",
    "admit_to_first_prefill_ms",
    "prefill_span_ms",
    "prefill_end_to_first_client_event_ms",
)


def arm(root, gpu, mode):
    d = root / f"gpu{gpu}" / mode
    digest = (root / "trace-server.sha256").read_text().split()[0]
    assert (d / "binary.sha256").read_text().split()[0] == digest
    assert read(d / "lease-after.json")["phase"] == "idle"
    assert read(d / "lease-after.json")["physical_index"] == gpu
    assert all(
        read(d / "shutdown.json")[k] == 0
        for k in ("kv_allocated_blocks", "active_requests", "waiting_requests")
    )
    log = (d / "server.log").read_text()
    mode_id = "fixed-tile-unsplit-v1" if mode == "tile" else "page-v1"
    assert f"RILEY_NATIVE_ATTENTION mode={mode_id}" in log
    cli = (d / "server-arguments.txt").read_text().splitlines()
    if "--native-attention-mode" in cli:
        assert cli[cli.index("--native-attention-mode") + 1] == mode_id
    else:
        assert (
            mode == "tile"
            and cli[cli.index("--inference-profile") + 1] == "native-bf16-varlen-v1"
        )
    assert Path(cli[cli.index("--model") + 1]).name == "Qwen2.5-Coder-7B-Instruct"
    check_frozen_requests(root, d)
    raw = read(d / "natural.json")
    assert len(raw) == len(SEEDS)
    probes = read(d / "identity.json")
    timeline = {(a["workflow"], a["turn"]): a for a in join(log, raw, details=True)}
    rows = {}
    for seed, work in zip(SEEDS, raw, strict=True):
        metrics = natural(
            work, seed, 2 if seed == 761 else 96, 0 if seed == 761 else 100
        )
        final = work["turns"][-1]
        assert (
            final["turn"] == 2 and final["status"] == 200 and final["finish"] == "stop"
        )
        t = timeline[work["workflow_id"], 2]
        ident = probes[str(seed)]
        assert final["input_sha256"] == ident["original_input_sha256"]
        assert final["usage"]["prompt_tokens"] == ident["prompt_tokens"]
        ttft = metrics["final_submission_to_first_ms"]
        assert abs(sum(t[k] for k in STAGES) - ttft) < 1e-5
        rows[seed] = {
            **t,
            **metrics,
            **ident,
            "ttft_ms": ttft,
            "cached": (final["usage"].get("prompt_tokens_details") or {}).get(
                "cached_tokens", 0
            ),
        }
    return rows


def verify(root):
    assert (root / "build.exit").read_text().strip() == "0" and (
        root / "exits.log"
    ).read_text().split() == ["0", "0"]
    pairs = []
    raw = {}
    for gpu in (6, 7):
        arms = {m: arm(root, gpu, m) for m in ("page", "tile")}
        raw[gpu] = arms
        for seed in SEEDS:
            a, b = [arms[m][seed] for m in ("page", "tile")]
            shapes = lambda r: [
                (c["rows"], c["end_position"], c["members"]) for c in r["chunks"]
            ]
            assert (
                a["prompt_ids_sha256"] == b["prompt_ids_sha256"]
                and a["cached"] == b["cached"]
                and shapes(a) == shapes(b)
            )
            assert not a["shared_prefill_steps"] and not b["shared_prefill_steps"]
            delta = {
                k: a[k] - b[k]
                for k in (
                    *STAGES,
                    "ttft_ms",
                    "prefill_participating_gpu_ms",
                    "tokenizer_path_ms",
                    "final_stream_ms",
                    "last_final_ms",
                )
            }
            gpu_fraction = (
                delta["prefill_participating_gpu_ms"] / delta["ttft_ms"]
                if delta["ttft_ms"] > 0
                else None
            )
            pairs.append(
                {
                    "gpu": gpu,
                    "seed": seed,
                    "new_rows": a["prefill_rows"],
                    "cache": a["cached"],
                    "delta_ms": delta,
                    "gpu_fraction": gpu_fraction,
                    "dominant_gpu": gpu_fraction is not None and gpu_fraction >= 0.8,
                }
            )
    return {
        "arms": raw,
        "pairs": pairs,
        "original_seed_gpu_dominant": all(
            p["dominant_gpu"] for p in pairs if p["seed"] == 762
        ),
        "other_long_gpu_dominant": sum(
            p["dominant_gpu"] for p in pairs if p["seed"] not in (761, 762)
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(verify(parser.parse_args().root), indent=2))
