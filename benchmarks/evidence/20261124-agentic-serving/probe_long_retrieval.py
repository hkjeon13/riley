"""Test value retrieval at both ends of a long context using two independent records."""

import argparse
import json
from pathlib import Path

from agent_workflow import stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("out", type=Path)
    parser.add_argument("--filler", type=int, default=45200)
    args = parser.parse_args()
    address, out = args.address, args.out
    rows = []
    for filler_size, early, late in [
        (128, 193671, 502894),
        (args.filler, 736251, 914603),
    ]:
        content = (
            f"The catalog value of ALPHA is {early}. Background records follow. "
            + " river" * filler_size
            + f" End of background. The catalog value of ZULU is {late}. "
            + "Return only the two catalog values, ALPHA first and ZULU second."
        )
        req = {
            "model": "m",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 40,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        measurement = {}
        row = {"filler_tokens_approx": filler_size, "expected": [early, late]}
        try:
            answer, calls = stream(address, req, measurement, timeout_s=900)
            row.update(
                answer=answer,
                correct=(
                    not calls
                    and str(early) in answer
                    and str(late) in answer
                    and answer.index(str(early)) < answer.index(str(late))
                ),
            )
        except (ValueError, OSError) as error:
            row.update(correct=False, error=str(error))
        row["measurement"] = measurement
        rows.append(row)
        Path(out).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
