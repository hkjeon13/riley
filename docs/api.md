# HTTP API

Riley speaks HTTP/1.1 and implements the OpenAI *chat completions* and
*completions* shapes. With `--api-key-env NAME` every `/v1` endpoint needs
`Authorization: Bearer <value of NAME>`; `/healthz`, `/readyz` and `/metrics`
stay open for probes. The value is read from the environment, never from a
command-line argument. Without the option there is no authentication, so bind
to a private address or put a proxy in front.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | Answer a conversation, optionally as a stream |
| `POST` | `/v1/completions` | Generate text from a raw prompt, optionally as a stream |
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
| `logprobs` | integer | none | 0 to 5 top alternatives per token, see [log probabilities](#log-probabilities) |
| `user` | string | none | Accepted and ignored |

`n`, `best_of`, `echo`, `presence_penalty` and `frequency_penalty` are accepted
only at their neutral values (1, 1, false, 0, 0). `suffix` and unknown fields
are rejected with an explicit error instead of being ignored.

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

### Log probabilities

Values are natural logs of the raw model distribution: the softmax of the
logits before temperature, `top_p` and any structured-output mask. A sampled or
constrained token can therefore have a lower value than the top alternative.
Ties in `top_logprobs` go to the lower token ID. `/v1/completions` returns
`tokens`, `token_logprobs`, `top_logprobs` and `text_offset` arrays;
`/v1/chat/completions` returns `content` entries with `token`, `logprob`,
`bytes` and `top_logprobs`. Streamed chunks carry the entry of their own token.
The end-of-sequence token, when generated, is included. With
`--sampling-backend gpu-greedy`, a request without a grammar has each token's
row summarized on the GPU (its maximum, partial sums of the normalizer and the
top tokens), and the CPU finishes the values from that summary. They have the
same bytes as the full-row CPU pass, which other requests still use.

## `POST /v1/chat/completions`

```json
{ "model": "qwen3-4b", "messages": [{ "role": "user", "content": "Hi" }], "max_tokens": 64 }
```

The server renders `messages` with the checkpoint's chat template
(`chat_template.jinja`, else the `chat_template` of `tokenizer_config.json`)
under Transformers' Jinja settings, with `bos_token`, `eos_token`,
`add_generation_prompt` (default `true`), `strftime_now` (UTC) and
`raise_exception`, then encodes the result without adding the tokenizer's own
begin-of-text token, as `apply_chat_template` does. The prompt then follows
the completion path, so `max_tokens` (or `max_completion_tokens`, default
`--max-output-tokens`), `temperature`, `top_p`, `seed`, `stop`, `stream`,
`stream_options.include_usage` and `return_token_ids` behave as there.

- `content` is a string, `null`, or a list of `{"type": "text", "text": …}`
  parts; messages carry `role`, `content` and optionally `name`, and for tools
  `tool_calls` and `tool_call_id`.
- `chat_template_kwargs` adds template variables, e.g.
  `{"enable_thinking": false}` for Qwen3.
- `logprobs: true` with `top_logprobs` 0 to 20 returns
  [log probabilities](#log-probabilities); `user` is accepted and ignored.
- Image parts, `functions`, `function_call` and `n > 1` are rejected with 400.
  A template's own error (`raise_exception`, e.g. roles not alternating) is a
  400 with code `invalid_messages`.
- A checkpoint without a chat template answers 404 `chat_template_unavailable`.

The response is a `chat.completion` with `choices[0].message` =
`{"role": "assistant", "content": …}`. Streamed chunks are
`chat.completion.chunk` objects whose `delta` carries `role` once, then
`content`; the finish chunk has an empty `delta` and the `finish_reason`.

### Structured output

`response_format` takes `{"type": "text"}`, `{"type": "json_object"}` or
`{"type": "json_schema", "json_schema": {"name": …, "schema": {…}}}`. The
output is held to the schema token by token, so a response that finishes with
`stop` is valid. One cut off by `max_tokens` finishes with `length` and is not:
leave room, and bound free-text fields with `maxLength`.

Output is compact JSON with properties in schema order and at most one space
after `:` and `,`. Enforced keywords:

| Type | Keywords |
|---|---|
| any | `type` (string or array), `enum`, `const`, `anyOf`, `oneOf`, one-element `allOf`, `$ref` to `#/…` including recursion |
| `object` | `properties`, `required`, `additionalProperties` (as `false`, or as a schema when there are no `properties`), `minProperties` and `maxProperties` without `properties` |
| `array` | `items`, `minItems`, `maxItems` |
| `string` | `minLength`, `maxLength`, `format` of `date`, `time`, `date-time` or `uuid`, `pattern` |
| `integer` | `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum` |
| `number` | `minimum: 0` |

Properties outside `properties` are never generated. `oneOf` is treated as
`anyOf`. Annotations (`title`, `description`, `default`, `examples`, `$schema`,
`$comment`) are ignored. Every other keyword, such as `uniqueItems`, `not` or
`multipleOf`, is refused with 400 `invalid_schema` and the keyword's name: a
schema is never accepted and then not enforced. Repetition bounds above 1,024
are refused too.

`pattern` supports literals, `.`, classes with ranges and `^` negation, `\\d`,
`\\w`, `\\s` (a space), groups, `|`, and the greedy quantifiers `*`, `+`, `?` and
`{m,n}`. As in JSON Schema an unanchored pattern may match anywhere in the
string. Lookaround, backreferences, lazy quantifiers and `\\b` are refused, and
`pattern` cannot be combined with `format` or length bounds.

### Tools

`tools` and `tool_choice` follow the OpenAI shape. The tools go to the chat
template as the `tools` variable, and the output format comes from the
template: `<tool_call>` blocks for Qwen, a bare `{"name": …, "parameters": …}`
object for Llama 3. A template that mentions neither answers 400.

| `tool_choice` | Behaviour |
|---|---|
| absent, `"auto"` | The model decides; a well-formed call is parsed out of its text |
| `"none"` | Tools are hidden from the template |
| `"required"` | One call of any tool, with arguments held to its `parameters` schema |
| `{"type": "function", "function": {"name": …}}` | One call of that tool, likewise |

A parsed reply has `finish_reason: "tool_calls"`, and `message.tool_calls` with
`arguments` as a JSON string. With `"required"`, `parallel_tool_calls: true`
lets the model write several calls; a greedy model may repeat a call until
`max_tokens`. With `"auto"`, arguments are the model's own text and are not
held to the schema. Text that does not parse stays in `content`, and a call cut
off by `max_tokens` is dropped if a complete one precedes it. When streaming,
text before a call streams as `content`, and the call arrives whole in the
finish chunk. Send results back as `{"role": "tool", "tool_call_id": …,
"content": …}` after the assistant message.

### Cost of per-request logits

Greedy requests pick their token on the GPU. A request with a grammar uses that
token when the grammar admits it, and otherwise pays for one row of logits on
the CPU. A request with log probabilities always does, which is much slower. The
other requests in the batch stay on the GPU path either way. See
[the measurements](serving-validation.md#structured-output-and-log-probabilities).

## Errors

Errors use the OpenAI envelope:

```json
{ "error": { "message": "…", "type": "invalid_request_error", "param": "max_tokens", "code": "…" } }
```

| Status | When |
|---|---|
| 400 | Malformed JSON, an invalid value, an unsupported parameter, or a schema that cannot be enforced |
| 401 | A bearer token is required and missing or wrong |
| 404 | Unknown path or model |
| 405 | Wrong method for a known path |
| 408 | The request timed out or was cancelled |
| 411, 413, 415, 431 | Missing `Content-Length`, body too large, not `application/json`, headers too large |
| 429 | The waiting queue is full |
| 503 | The server is not ready or is shutting down |
| 500 | Internal error |

When the [prefix cache](prefix-cache.md) served part of a prompt, `usage` also
carries `"prompt_tokens_details": { "cached_tokens": N }`. The field is absent
otherwise. `prompt_tokens` always counts the whole prompt.

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
