# Hub model ids and `/v1/chat/completions` (2026-10-29)

Image `riley-dist:cu124-chat` (`sha256:6c188588d9a9…`, 2.34 GB) built on
`aws-a100` from `02fe91ef0` plus the files of this commit, without the
unrelated uncommitted `crates/riley-model/src/config.rs` edit of that checkout.
Runs used A100 GPU 6 under the holder lease.

## Chat and Hub ids (`verify_chat.py`, `chat-report.json`)

Each case ran `serve --model <Hub id or preset>` with `--user` and an
EFS-backed `/models`, so the first start planned the model from Hub metadata
and downloaded it. For three conversations (single user turn, system + user,
user/assistant/user) the prompt token IDs Riley fed the model
(`return_token_ids`) were compared with Transformers 5.19.0
`apply_chat_template(..., add_generation_prompt=True, tokenize=True)` on the
same downloaded files, and streamed content with non-streamed content.

| Model | Ready | Prompt IDs = HF | Stream = non-stream |
|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | 39 s | 3/3 | 3/3 |
| Qwen/Qwen3-0.6B | 55 s | 3/3 | 3/3 |
| Qwen/Qwen3-0.6B, `enable_thinking: false` | 21 s | 3/3 | 3/3 |
| HuggingFaceTB/SmolLM2-360M-Instruct | 33 s | 3/3 | 3/3 |
| unsloth/Llama-3.2-1B-Instruct (`strftime_now` date) | 108 s | 3/3 | 3/3 |
| Qwen/Qwen2.5-Coder-1.5B-Instruct (not a preset) | 110 s | 3/3 | 3/3 |
| preset `llama-3.2-1b` (new chat files) | 104 s | 3/3 | 3/3 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | refused | — | — |

`/v1/completions` answered on every served model and every container stopped
with exit 0. DeepSeek-R1-Distill was refused by the plan preflight before any
download (`unsupported config field use_mrope=false`), as designed; accepting
that neutral field is separate pending work.

## Completions unchanged (`verify_image.py`, `regression-report.json`)

The new image served smollm2-135m, qwen2.5-0.5b, qwen3-4b and llama-3.2-1b
with the preset defaults, as did the measured host binary
`riley-mc-b74361d3`; greedy `/v1/completions` outputs were identical for 4 of
4 models.

## Presets

The six Llama/SmolLM2 presets gained `tokenizer_config.json` (and
`chat_template.jinja` for Llama 3.2) from `riley-checkpoint plan` at their
existing commits; every previously listed file kept its size and SHA-256.
`generation_config.json` was not added, so their stop behaviour stays that of
the measured runs.
