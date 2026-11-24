"""Application adapter for the report comparison's explicitly requested numerical opening."""

import json
import re


def report_prefix(request):
    """Read the actual completed call and its result. Use ordinary generation on ambiguity."""
    if request.get("tool_choice") != "none":
        return ""
    messages = request["messages"]
    if len(messages) < 2:
        return ""
    invocation, result = messages[-2:]
    if invocation.get("role") != "assistant" or result.get("role") != "tool":
        return ""
    try:
        (call,) = invocation["tool_calls"]
        if (
            not call["id"]
            or call["id"] != result["tool_call_id"]
            or call["function"]["name"] != "read_reports"
        ):
            return ""
        args = json.loads(call["function"]["arguments"])
        if args != {"first": "report-A", "second": "report-B"}:
            return ""
        rows = json.loads(result["content"])
        if not isinstance(rows, list) or len(rows) != 2:
            return ""
        values = {}
        for row in rows:
            metric = row["p95_ms"]
            if (
                type(metric) is not int
                or metric < 0
                or row["statistic"] != "service response latency"
            ):
                return ""
            values[row["id"]] = metric
        before, after = values[args["first"]], values[args["second"]]
    except (KeyError, TypeError, ValueError):
        return ""
    return (
        f"Before: {before} ms. After: {after} ms. Reduction: {before - after} ms.\n\n"
        f"The measured p95 changed from {before} ms ({args['first']}) "
        f"to {after} ms ({args['second']})."
    )


def prefix_request(request, *, guide_continuation=False):
    prefix = report_prefix(request)
    if not prefix:
        return request
    prepared = {**request, "riley_assistant_prefix": prefix}
    if guide_continuation:
        lead = prefix.split("\n\n", 1)[1]
        lead_words = len(re.findall(r"[A-Za-z0-9]+(?:[-’'][A-Za-z0-9]+)*", lead))
        remaining = 90 - lead_words
        instruction = (
            f"The application supplies the requested opening and the first {lead_words} "
            f"words of the explanation. Continue the same paragraph with {remaining} more "
            "words. Explain the meaning of the 95th percentile, interpret the measured "
            "change, and say what these two reports do and do not establish. Use only "
            "these measurements. Do not repeat the opening or add a heading."
        )
        prepared["messages"] = [
            *request["messages"],
            {"role": "user", "content": instruction},
        ]
    return prepared
