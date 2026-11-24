"""Check cancellation, unicode boundaries, fixed system messages and renderer recovery on two checkpoints."""

import argparse
import json
from pathlib import Path

from budgeted_tool_prefix import PromptProfiler, budgeted_preview


def check(binary, directory, llama):
    results = []
    sysmsg = {"role": "system", "content": "Keep the source identifiers unchanged."}
    msg = {"role": "user", "content": "안녕하세요. 🧪 cafe\u0301 경과 23ms. " * 190}
    base = {
        "model": "m",
        "messages": [sysmsg, msg],
        "max_tokens": 1,
        "tool_choice": "none",
        "temperature": 0,
        "stream": True,
    }
    orig = json.dumps(base, sort_keys=True)
    with PromptProfiler(binary, directory, llama3=llama) as native:
        try:
            native.ids({"messages": [], "model": 42})
        except ValueError:
            pass
        else:
            raise AssertionError("invalid request was not rejected")
        full = native.ids(base)
        for cap in (1, 15, 192, 512, len(full), len(full) + 16):
            chosen, report = budgeted_preview(base, cap, native.ids)
            if chosen is None:
                assert report["decision"] == "no_matching_block"
            else:
                assert chosen["messages"][0] == sysmsg
                assert msg["content"].startswith(chosen["messages"][1]["content"])
                ids = native.ids(chosen)
                assert len(ids) <= cap and ids[:16] == full[:16]
                if cap >= len(full):
                    assert chosen is base
            results.append(report)
        assert any(r["decision"] == "partial" for r in results)
        calls = []

        def cancel_while_rendering(body):
            calls.append(1)
            return native.ids(body)

        assert (
            budgeted_preview(
                base, 512, cancel_while_rendering, cancelled=lambda: bool(calls)
            )[1]["decision"]
            == "cancelled"
        )
        assert len(calls) == 1 and orig == json.dumps(base, sort_keys=True)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("binary", "qwen", "llama", "out"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    a.out.write_text(
        json.dumps(
            {
                "qwen": check(a.binary, a.qwen, False),
                "llama3": check(a.binary, a.llama, True),
            },
            indent=2,
        )
        + "\n"
    )
