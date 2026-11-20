"""Bound promised KV from every measured and warm-up prompt in the joined trace."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys


def verify(initial, retry, helpers):
    sys.path.insert(0, str(helpers))
    from summarize_request_stages import read_phases

    def number(arguments, flag):
        return int(arguments[arguments.index(flag) + 1])

    result = {}
    for arm, stage in (("riley-first", initial), ("qkv-first", retry)):
        root = stage / f"qwen14-{arm}"
        plan = json.loads((root / "plan.json").read_bytes())
        receipt = json.loads((root / "receipt.json").read_bytes())
        assert receipt["status"] == "passed"
        assert json.loads((root / "server-exit.json").read_bytes())["exit"] == 0
        kv = number(plan["server"], "--kv-blocks")
        active = number(plan["server"], "--max-active-sequences")
        assert kv == receipt["common_kv_blocks"] == 2576 and active == 32
        sizes = {}
        phases = read_phases(root / "server.log")
        served = {request: events for request, events in phases.items() if events.get("first_delta")}
        earliest = min(events["front_submit"][0][0] for events in served.values())
        startup = [events for events in phases.values() if events.get("front_submit") and not events.get("first_delta")]
        assert all(events["prefill_return"][-1][0] < earliest for events in startup)
        for request, events in served.items():
            assert len(events["front_submit"]) == len(events["first_delta"]) == 1
            finals = [fields[1] for _, fields in events["prefill_start"] if fields[2] == 1]
            assert len(finals) == 1
            sizes[request] = finals[0]
        expected = sum(item["requests"] + item["warmup"] for item in plan["cells"].values())
        assert len(sizes) == expected
        for name, cell in plan["cells"].items():
            raw = (root / f"{name}.json").read_bytes()
            assert sha256(raw).hexdigest() == receipt["results"][name]
            samples = json.loads(raw)["measured"]
            assert all(sample["completion_tokens"] <= cell["max_tokens"] and
                       sample["prompt_tokens"] == sizes[sample["request_id"]] for sample in samples)
        concurrent = max(item["concurrency"] for item in plan["cells"].values())
        max_output = max(item["max_tokens"] for item in plan["cells"].values())
        block_size = 16
        blocks_per_request = (max(sizes.values()) + max_output + block_size - 1) // block_size
        bound = concurrent * blocks_per_request
        assert concurrent <= active and bound <= kv
        result[arm] = {
            "phase_requests_with_warmup": expected, "prior_startup_requests": len(startup),
            "maximum_prompt_tokens": max(sizes.values()),
            "client_concurrency_upper_bound": concurrent, "configured_active_sequence_limit": active,
            "max_output_tokens_per_request": max_output, "max_promised_blocks_all_live_requests": bound,
            "physical_kv_blocks": kv, "guaranteed_kv_block_headroom": kv - bound,
            "server_log_sha256": sha256((root / "server.log").read_bytes()).hexdigest(),
        }
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for item in ("initial", "retry", "helpers"):
        parser.add_argument(item, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.initial, args.retry, args.helpers), indent=2))
