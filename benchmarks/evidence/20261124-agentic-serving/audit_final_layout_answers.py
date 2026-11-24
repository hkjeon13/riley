"""Check what each long-history writer actually claimed about percentiles."""

import argparse
import json
import re
from pathlib import Path

from agent_workflow import fingerprint

# Post-hoc witnesses, deliberately narrower than a general semantics grader.
# Neither between-period percentile threshold identifies paired faster requests,
# changes in the maximum, or a change in the slowest fraction as a whole.
UNSUPPORTED = (
    "95% of the service responses were faster in the after report compared to the before report",
    "95% of the service responses are now faster",
    "the longest 5% of responses being notably shorter",
    "the changes implemented have effectively reduced the latency for the majority of the service requests",
    "effectively reduced the latency for the majority of service responses",
    "95% of the service response times were 84 ms before the changes and 12 ms after the changes",
    "a notable decrease in the longest observed response times",
)
PREFIX = re.compile(
    r"^before:\s*\d+\s*ms\.\s*after:\s*\d+\s*ms\.\s*reduction:\s*\d+\s*ms\.",
    re.IGNORECASE,
)


def audit(report):
    rows = []
    for group, cells in report["cells"].items():
        if not group.startswith("200-"):
            continue
        for arm, work in cells.items():
            w = work["long-tool-result"]
            text = w.get("answer", "").replace("*", "").strip()
            normalized = " ".join(text.lower().split())
            header = PREFIX.match(text)
            tail = text[header.end() :] if header else None
            hits = [phrase for phrase in UNSUPPORTED if phrase.lower() in normalized]
            rows.append(
                {
                    "group": group,
                    "arm": arm,
                    "answer_sha256": fingerprint(text),
                    "automatic_ok": w["ok"],
                    "matched_unsupported_claims": hits,
                    "explanation_whitespace_words": len(tail.split())
                    if tail is not None
                    else None,
                }
            )
    return {
        "method": "Post-hoc exact phrase search on normalized whitespace. An unmatched answer is ungraded. Count explanation words after the three required labeled numbers only.",
        "writer_outputs": len(rows),
        "rows": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    print(json.dumps(audit(json.loads(a.report.read_bytes())), indent=2))
