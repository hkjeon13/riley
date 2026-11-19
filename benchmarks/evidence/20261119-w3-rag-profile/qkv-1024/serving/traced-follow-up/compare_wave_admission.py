"""Reconstruct and replay the first nine prefill turns of the delayed W3 wave."""

from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess

from analyze_prefill_turns import reconstruct
from compare_profile_arms import digest, read, require
from compare_traced_qkv import FIELDS, ORDER, check_campaign, read_arm
from summarize_request_stages import read_phases

CELL = "wave-50ms"
TARGET = 41
BOUNDARY_REQUEST = 45
WAVE = range(40, 48)
PREFIX_STEPS = 9


def replay(binary, admissions, lengths, windows):
    rows = [(stamp, index, lengths[index]) for index, stamp in sorted(admissions.items())]
    rows += [(step.start, step.end) for step in windows]
    fixture = "".join(" ".join(map(str, row)) + "\n" for row in rows)
    result = subprocess.check_output([binary], input=fixture, text=True)
    return [list(map(int, row.split())) for row in result.splitlines()]


def arm(root, previous, mode, binary):
    binding, observed, log_hash = read_arm(root, previous, mode)
    directory = root / f"qwen14-{mode}"
    cell = read(directory / f"{CELL}.json")
    witnesses = {row["request_id"]: row for row in cell["measured"]}
    require(cell["statistics"]["config"]["stagger_ms"] == 50
            and cell["statistics"]["config"]["concurrency"] == 8
            and cell["statistics"]["config"]["max_tokens"] == 128, "unexpected wave configuration")
    flags = binding["server"]
    expected_flags = {"--max-active-sequences": "32", "--max-sequence-tokens": "8192",
                      "--batch-token-budget": "1024", "--prefill-chunk-tokens": "1024",
                      "--prefill-burst-steady-budget": "256", "--prefill-burst-max-decode-rows": "32",
                      "--prefill-burst-min-prompt-tokens": "512"}
    require(all(flags[flags.index(flag) + 1] == value for flag, value in expected_flags.items())
            and not any(flag in flags for flag in ("--prefix-cache-blocks", "--prefill-coalescing-tokens",
                                                    "--prefill-plateau-rows"))
            and binding["identity"]["common_kv_blocks"] == 2576, "unexpected replay scheduler configuration")
    events = read_phases(directory / "server.log")
    steps, turns, admitted, lengths = reconstruct(events)
    wave = {row["request_index"]: key for key, row in witnesses.items() if row["request_index"] in WAVE}
    require(set(wave) == set(WAVE), "incomplete delayed wave")
    offset = turns[wave[40]][0][0]
    absolute = steps[offset:offset + PREFIX_STEPS]
    expect = [[witnesses[chunk.request]["request_index"], chunk.first_token, chunk.next_token]
              for step in absolute for chunk in step.chunks]
    require(len(absolute) == len(expect) == PREFIX_STEPS
            and all(len(step.chunks) == 1 and step.chunks[0].decode_rows == step.chunks[0].final == 0
                    and step.chunks[0].batch_rows == 1024 for step in absolute)
            and all(row[0] in WAVE for row in expect), "early wave includes different scheduler work")
    prompt_lengths = {index: lengths[identifier] for index, identifier in wave.items()}
    require(all(prompt_lengths[i] == witnesses[key]["prompt_tokens"] for i, key in wave.items()),
            "wave prompt length disagrees with matched request")
    origin = admitted[wave[40]]
    arrival = {i: admitted[key] - origin for i, key in wave.items()}
    windows = [step._replace(start=step.start - origin, end=step.end - origin) for step in absolute]
    reference = replay(binary, arrival, prompt_lengths, windows)
    require(reference == expect, f"{mode}: replayed decisions differ from the observed prefix")

    selection = windows[2]
    require(expect[2] == [TARGET, 0, 1024], "third wave step is not the first focal prefill")
    request_id = wave[BOUNDARY_REQUEST]
    (send, _), = events[request_id]["command_send"]
    sent = send - origin
    boundary = arrival[BOUNDARY_REQUEST] < selection.start
    if boundary:
        replay_admission = selection.end + 1
        require(replay_admission < min(windows[3].start, arrival[46], arrival[47]),
                "cannot move the request just after the third step")
    else:
        replay_admission = max(windows[1].end, arrival[44]) + 1
        require(replay_admission < selection.start, "cannot move the request just before the third step")
    require((sent < selection.start) == boundary and sent <= arrival[BOUNDARY_REQUEST],
            "command and admission straddle different plans")
    changed = replay(binary, {**arrival, BOUNDARY_REQUEST: replay_admission}, prompt_lengths, windows)
    target = turns[wave[TARGET]]
    return binding, [[row[key] for key in FIELDS] for row in cell["measured"]], {
        "server_log_sha256": log_hash, "traced_cell_sha256": digest(directory / f"{CELL}.json"),
        "target_prompt_tokens": prompt_lengths[TARGET],
        "request_45_send_minus_wave_step_3_start_ns": sent - selection.start,
        "request_45_admit_minus_wave_step_3_start_ns": arrival[BOUNDARY_REQUEST] - selection.start,
        "request_45_admit_minus_wave_step_3_return_ns": arrival[BOUNDARY_REQUEST] - selection.end,
        "original_45_admission_ns": arrival[BOUNDARY_REQUEST], "moved_45_admission_ns": replay_admission,
        "wave_step_3_start_ns": selection.start, "wave_step_3_end_ns": selection.end,
        "cpu_replay_inputs": {"admissions_ns": arrival, "prompt_tokens": prompt_lengths,
                              "step_windows_ns": [[step.start, step.end] for step in windows]},
        "observed_early_plans": expect, "scheduler_cpu_replay": reference,
        "scheduler_cpu_replay_with_only_45_moved": changed,
        "target_prefill_steps": [
            {"wave_step": position - offset + 1, "row_range": [part.first_token, part.next_token],
             "decode_rows": part.decode_rows, "batch_rows": part.batch_rows,
             "peers": [{"request_index": witnesses[peer.request]["request_index"],
                        "rows": peer.next_token - peer.first_token}
                       for peer in steps[position].chunks if peer.request != wave[TARGET]]}
            for position, part in target
        ],
        "verified_request_streams": observed[CELL]["complete_streams_equal_to_uninstrumented"],
    }


def compare(root, previous, binary, scheduler_source):
    check_campaign(root)
    rows = {mode: arm(root, previous, mode, binary) for mode in ORDER}
    same = rows[ORDER[0]]
    require(all(row[:2] == same[:2] for row in rows.values()), "arm inputs or output identity changed")
    require((root / "binary.sha256").read_text().split()[0] == same[0]["sha256"]["riley"],
            "unexpected GPU trace binary")
    require(all(rows[f"riley-{order}"][2]["scheduler_cpu_replay_with_only_45_moved"]
                == rows[f"qkv-{order}"][2]["observed_early_plans"]
                and rows[f"qkv-{order}"][2]["scheduler_cpu_replay_with_only_45_moved"]
                == rows[f"riley-{order}"][2]["observed_early_plans"] for order in ("first", "second")),
            "moving one request did not reproduce the other branch")
    prefixes = [row[2]["observed_early_plans"] for row in rows.values()]
    first = next(i for i in range(PREFIX_STEPS) if len({tuple(row[i]) for row in prefixes}) > 1)
    return {"scope": "Observed admission and prefill order, with the original scheduler replayed before the first decode.",
            "cell": CELL, "wave_request_indices": list(WAVE), "target_request": TARGET,
            "admission_changed_in_counterfactual": BOUNDARY_REQUEST,
            "first_changed_prefill_wave_step": first + 1,
            "scheduler_inputs_sha256": {name: digest(scheduler_source.with_name(name))
                                        for name in ("scheduler.rs", "config.rs")},
            "cpu_replay_binary_sha256": digest(binary),
            "gpu_trace_binary_sha256": same[0]["sha256"]["riley"],
            "arms": {mode: row[2] for mode, row in rows.items()}}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("uninstrumented", type=Path)
    parser.add_argument("cpu_replay_binary", type=Path)
    parser.add_argument("scheduler_source", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.trace, args.uninstrumented, args.cpu_replay_binary, args.scheduler_source),
                     indent=2, allow_nan=False))
