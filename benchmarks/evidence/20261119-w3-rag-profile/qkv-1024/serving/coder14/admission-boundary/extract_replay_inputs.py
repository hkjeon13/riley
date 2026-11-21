"""Extract receipt-bound Coder-14B scheduler inputs before the first decode."""

from argparse import ArgumentParser
import json
from pathlib import Path

from analyze_prefill_turns import reconstruct
from compare_coder import ORDER
from compare_profile_arms import digest, read, require
from summarize_request_stages import read_phases


CELL = "wave-50ms"
INDICES = set(range(40, 48))


def extract(root, source_report):
    require((root / "campaign.exit").read_text().strip() == "0", "source trace did not complete")
    expected = read(source_report)
    require(expected["remote_stage"] == root.name and expected["order"] == list(ORDER), "wrong source report")
    cases = {}
    for mode in ORDER:
        folder = root / f"coder14-{mode}"
        log = folder / "server.log"
        archive = folder / f"{CELL}.json"
        report = expected["arms"][mode]
        require(digest(log) == report["server_log_sha256"]
                and digest(archive) == report["cells"][CELL]["trace_source_sha256"], "bound trace bytes differ")
        phases = read_phases(log)
        steps, turns, admitted, lengths = reconstruct(phases)
        wave = {row["request_id"]: row["request_index"] for row in read(archive)["measured"]
                if row["request_index"] in INDICES}
        require(set(wave.values()) == INDICES and len(wave) == len(INDICES), "wave capture is incomplete")
        by_index = {index: request for request, index in wave.items()}
        origin = admitted[by_index[40]]
        first = turns[by_index[40]][0][0]
        prefix = steps[first:first + 9]
        require(len(prefix) == 9 and all(len(step.chunks) == 1 for step in prefix),
                "the prefix has a different number of scheduler selections")
        observations = []
        for step in prefix:
            chunk, = step.chunks
            require(chunk.request in wave and chunk.batch_rows == 1024 and chunk.decode_rows == chunk.final == 0,
                    "prefix contains non-wave work or a decode turn")
            observations.append([wave[chunk.request], chunk.first_token, chunk.next_token])
        for index in INDICES:
            require(lengths[by_index[index]] >= 1024, "prefill fixture has an unexpected prompt length")
        cases[mode] = {
            "server_log_sha256": digest(log),
            "traced_wave_sha256": digest(archive),
            "admissions_ns": {index: admitted[by_index[index]] - origin for index in sorted(INDICES)},
            "prompt_tokens": {index: lengths[by_index[index]] for index in sorted(INDICES)},
            "step_windows_ns": [[step.start - origin, step.end - origin] for step in prefix],
            "observed_prefill": observations,
        }
    return {"source_scope": "First nine only-prefill turns of the eight-request fifth wave.",
            "source_receipt_sha256": digest(source_report), "source_stage": root.name, "arms": cases}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("source_report", type=Path)
    args = parser.parse_args()
    print(json.dumps(extract(args.root, args.source_report), indent=2))
