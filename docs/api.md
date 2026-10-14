# HTTP API

Riley speaks HTTP/1.1 and implements the OpenAI *completions* shape. There is
no chat-completions endpoint: send the already-templated prompt as text.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/completions` | Generate text, optionally as a stream |
| `GET` | `/v1/models`, `/v1/models/{id}` | The served model |
| `GET` | `/healthz` | Liveness; always 200 while the process serves HTTP |
| `GET` | `/readyz` | Readiness; 503 until the model is loaded and while shutting down |
| `GET` | `/metrics` | Operational counters as JSON |
| `GET` | `/v1/config` | The effective runtime configuration, when the server was started with one |

## `POST /v1/completions`

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | string | required | Must equal the served model ID (`--model-id`) |
| `prompt` | string | required | One UTF-8 string. Arrays and token-ID prompts are rejected |
| `max_tokens` | integer | 16 | Bounded by `--max-output-tokens` and the context size |
| `temperature` | number | 1.0 | 0 to 2. Use 0 for greedy decoding |
| `top_p` | number | 1.0 | Greater than 0, at most 1 |
| `seed` | integer | none | Makes sampling reproducible for that request |
| `stop` | string or array | none | Stop strings |
| `stream` | boolean | false | Server-sent events when true |
| `stream_options.include_usage` | boolean | false | Adds a final usage-only chunk; needs `stream` |
| `return_token_ids` | boolean | false | Adds the generated token IDs to each choice |

`n`, `best_of`, `echo`, `presence_penalty` and `frequency_penalty` are accepted
only at their neutral values (1, 1, false, 0, 0). `logprobs`, `suffix` and
unknown fields are rejected with an explicit error instead of being ignored.

Response:

```json
{
  "id": "cmpl-…",
  "object": "text_completion",
  "created": 1791800000,
  "model": "qwen3b",
  "choices": [
    { "text": " Paris.", "index": 0, "logprobs": null, "finish_reason": "stop" }
  ],
  "usage": { "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8 }
}
```

`finish_reason` is `stop` (end-of-sequence token or a stop string) or `length`
(`max_tokens` reached).

### Streaming

With `"stream": true` the response is `text/event-stream`. Each event is a
`data: {…}` line carrying a chunk with the same fields as above; the last
content chunk carries the `finish_reason`, and the stream ends with
`data: [DONE]`.

Closing the connection cancels the request: its slot and KV-cache blocks are
released at the next scheduler step.

### Sampling

Sampling is request-local. Each request owns its random stream, so a request's
output does not depend on which other requests share its batch. With
`--sampling-backend gpu-greedy`, requests with `temperature: 0` pick the token
on the GPU; other requests in the same server fall back to the CPU sampler.

## Errors

Errors use the OpenAI envelope:

```json
{ "error": { "message": "…", "type": "invalid_request_error", "param": "max_tokens", "code": "…" } }
```

| Status | When |
|---|---|
| 400 | Malformed JSON, an invalid value, or an unsupported parameter |
| 404 | Unknown path or model |
| 405 | Wrong method for a known path |
| 408 | The request timed out or was cancelled |
| 411, 413, 415, 431 | Missing `Content-Length`, body too large, not `application/json`, headers too large |
| 429 | The waiting queue is full |
| 503 | The server is not ready or is shutting down |
| 500 | Internal error |

## `GET /metrics`

A JSON object, free of prompt text:

- `active_requests`, `waiting_requests`, `kv_allocated_blocks`
- `allocation`: live native allocations (device and pinned host memory)
- `batch_shapes`: how many steps ran at each batch shape, with GPU time
- `counters`: `cancellations`, `disconnects`, `overloads`, `dropped_observations`

## `GET /healthz`, `GET /readyz`

```json
{ "status": "ok", "ready": true, "accepting": true, "active_requests": 0, "waiting_requests": 0 }
```
