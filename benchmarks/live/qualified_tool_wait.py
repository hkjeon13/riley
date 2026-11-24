"""Run the registered wait-hint policy and preserve incorrect predictions as distinct cells."""

import argparse
import json
from pathlib import Path

from tool_wait_policy import ToolWaitPolicy
from tool_wait_trial import measure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("out", type=Path)
    p.add_argument("registration", type=Path)
    p.add_argument(
        "--include-completed-call",
        action="store_true",
        help="Include the validated tool call and an empty tool result in the preview.",
    )
    p.add_argument(
        "--release-on-preview",
        action="store_true",
        help="Allow the next preparation when the previous preview finishes, while its user answer can still be streaming.",
    )
    args = p.parse_args()
    config = json.loads(args.registration.read_text())
    order = config["order"].get(args.out.name, [])
    rows = []
    for group, enabled in order:
        policy = ToolWaitPolicy()
        for label, offset, actual, hint, peer in config["scenarios"]:
            base = config["seeds"][group] + offset
            result = measure(
                args.address,
                enabled,
                label,
                base,
                actual,
                base + 1 if peer else None,
                policy=policy,
                estimated_wait_ms=hint,
                include_completed_call=args.include_completed_call,
                release_on_preview=args.release_on_preview,
            )
            rows.append({"group": group, "result": result})
            (args.out / "wait-policy.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
