# Independent synthetic free-generation gate

2026-09-13, native/runtime base `0007ccb9168abcf15861f7962704b919d52db9dd`.
**Strict greedy equivalence fails on new inputs. General model quality remains unaccepted, and serving performance is unmeasured.**

## Test contract

The Rust test creates eight independent synthetic token sequences with seed 9131701 and lengths 1/15/16/17/127/128/129/511. IDs are in the fixed model vocabulary. Four repetitions produce 32 concurrent requests; each generates 32 tokens. Both engines use their own selected next token, with no teacher forcing and no EOS early stop. Existing exact and experimental consistent-decode v2 use the same loaded checkpoint, mixed scheduling, 1024-token budget and 128-token prefill chunk. Separate sessions run sequentially; this is not a throughput measurement. Input/output tokens are preserved in [tokens.json](tokens.json).

| Check | Result |
| --- | --- |
| Requests / generated tokens per backend | 32 / 1024 |
| Different completed sequences | 12/32, representing 3/8 unique prompts |
| Different token positions | 152/1024 |
| First divergences | prompt length 15 at index 23; length 16 at 27; length 128 at 5 |
| Repeated-prompt invariance within each backend | Pass for all 24 repeated requests |
| Scheduler settlement / close / zero retained CUDA allocations | Pass for both runs |
| Strict independent greedy equivalence | **FAIL, cargo exit 101** |
| Natural-language quality or vLLM serving comparison | Not measured |

The 152 differing positions include continuation divergence after different tokens have entered the two models. They are not 152 independent numerical errors or a 14.8% quality-loss estimate. This failure is intentionally retained as an ignored GPU gate; it is not changed to pass by weakening its assertion. It contradicts broad exact interchangeability despite the earlier fixture passing.

## Independent FP32 observation at the first divergence

An offline Hugging Face eager FP32 model (torch 2.13.0+cu130, transformers 5.15.1, TF32 disabled) loads the same local checkpoint with no remote code/download. It receives the identical prompt plus the shared continuation before the first different token. Only the three unique divergence points are evaluated.

| Prompt length / generated index | Exact token | FlashInfer token | FP32 argmax | FP32 log-probability: exact / FlashInfer |
| --- | ---: | ---: | ---: | --- |
| 15 / 23 | 655 | 1739 | 1739 | -2.335262 / -2.282461 |
| 16 / 27 | 725 | 4776 | 4776 | -2.890056 / -2.846811 |
| 128 / 5 | 41 | 37 | 41 | -2.230413 / -2.304424 |

Two observations favor the FlashInfer token and one favors the existing token under this reference. They are selected synthetic cases, not a held-out quality score. FP32 is also a different numerical contract from BF16 serving. These observations motivate a broader independent reference comparison rather than treating every baseline disagreement as quality degradation. They do not promote the candidate.

The reference script is offline only. Rust serving never invokes Python, PyTorch, a Python worker or an RPC to one. Native implementation is unchanged in this batch.

## Evidence and reproduction

[Source hashes](sources.json) match local and remote test/reference scripts. The recorded Rust log contains both full generated token sets and the final failed assertion. An initial run without raw-token printing returned the same counts; the recorded run retained sequences for independent analysis. The FP32 log contains raw logits/top tokens and log-probabilities.

With the existing pinned CUDA/FlashInfer/checkpoint environment:

```sh
cargo test -p riley-scheduler --features cuda --test flashinfer_free_generation_gpu -- --ignored --nocapture
python benchmarks/analysis/flashinfer_first_divergence_reference.py --model MODEL_PATH --generation-log RUST_LOG_PATH
```

The first command currently fails its exact-equivalence gate, as documented. The second is an offline observation and does not turn that failure green.

## Next decision

Keep the explicit experimental profile and existing exact serving default. Evaluate separate natural-language prompts using the same independent reference for baseline and candidate, with numerical/quality criteria declared before inspecting that dataset. Preserve nonfinite, request isolation and batch invariance checks. Only claim serving improvement from a matched baseline/new-Riley/vLLM serving report at a meaningful integration milestone; no performance result is inferred here.
