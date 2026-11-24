"""Experimental presentation of long tool results in the two-report performance task."""

import json

P95_EVIDENCE = {
    "statistic": "95th-percentile response time (milliseconds)",
    "interpretation": "Each value is a threshold at or below which approximately 95% of that observation period's requests finished.",
    "comparison": "The difference describes the change in the two percentile thresholds. These reports do not contain paired per-request outcomes, mean or maximum latency, a statistical significance test, or evidence of the cause of the change.",
}


def present(request, mode):
    if request["tool_choice"] != "none":
        return request
    message = request["messages"][-1]
    if message["role"] != "tool" or len(message["content"].encode()) < 4096:
        return request
    request = {**request, "chat_template_kwargs": {"tools": request["tools"]}}
    if mode == "defined":
        content = json.dumps(
            {
                "metric_definition": P95_EVIDENCE,
                "reports": json.loads(message["content"]),
            }
        )
        request["messages"] = [
            *request["messages"][:-1],
            {**message, "content": content},
        ]
    return request
