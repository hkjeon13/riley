"""Measure cold retrieval, real prefix continuation and an overlapping final stream."""

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent_workflow import stream


def request(text, limit):
    return {
        "model": "m",
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": limit,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def archive_text(label, size, left, right):
    return (
        f"Archive {label}. The reference number for ALDER is {left}. Other entries follow. "
        + (" valley" if label == "north" else " ridge") * size
        + f" The reference number for BIRCH is {right}. "
    )


def run(address, output):
    rows = {}

    def measure(req, numbers=(), visible=None):
        data = {}
        try:
            answer, calls = stream(
                address, req, data, first_visible=visible, timeout_s=180
            )
            data["answer"] = answer
            data["valid"] = bool(
                data.get("done_ns")
                and data.get("content_ns")
                and not calls
                and answer.strip()
            )
            if numbers:
                data["valid"] &= re.findall(r"\d+", answer) == list(numbers)
        except (ValueError, OSError) as exc:
            data.update(valid=False, error=str(exc))
        return data

    def save():
        output.write_text(json.dumps(rows, indent=2) + "\n")

    final = request(
        "Write a detailed historical survey of the development of railways in Europe, from their origins to the present. Describe each era in depth.",
        192,
    )
    rows["final_solo"] = measure(final)
    save()
    ask = "Return only the two reference numbers in order, ALDER and then BIRCH."
    north = archive_text("north", 12100, "486273", "705148")
    rows["cold"] = measure(request(north + ask, 40), ("486273", "705148"))
    save()
    rows["continuation"] = measure(
        request(north + "An auditor is checking the record. " + ask, 40),
        ("486273", "705148"),
    )
    save()
    with ThreadPoolExecutor(max_workers=1) as pool:
        visible = threading.Event()
        user = pool.submit(measure, final, (), visible)
        if not visible.wait(30):
            raise RuntimeError("final stream did not start")
        south = archive_text("south", 26700, "230961", "874215")
        rows["mixed_prefill"] = measure(request(south + ask, 40), ("230961", "874215"))
        rows["final_mixed"] = user.result()
    save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("out", type=Path)
    a = p.parse_args()
    run(a.address, a.out)
