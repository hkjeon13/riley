# Natural-language numerical screen: relative gate not met

2026-09-13; runtime/native baseline `7d47cf80324a700a1821e486f1c89b0a93054019` (no inference implementation changes in this batch).
**NLL improved, FP32-reference KL worsened; the predeclared relative screen fails. The experimental profile remains unpromoted.**

## Fixed protocol

[Protocol](../../fixtures/flashinfer-natural-v1/PROTOCOL.md) and [eight authored passages](../../fixtures/flashinfer-natural-v1/passages.json) were written and copied to the remote host before execution. Each is tokenized locally without special tokens: first 32 tokens are the prompt, next 32 are human target tokens. [Input tokens](cases.json), tokenizer/source hashes and per-passage metrics are preserved. These are short newly authored English passages, not a standardized or representative quality benchmark.

Existing exact Riley and consistent-decode v2 consume the identical target history. Rust alone runs scheduler/model inference and writes BF16 logits. A separate offline Python script loads the same checkpoint into HF eager FP32 with TF32 disabled and evaluates the same causal prediction positions. Python is not invoked by Rust serving.

The frozen rule required both aggregate target NLL and KL(FP32 || engine) to be no greater than baseline, plus finite logits and complete outputs/resource cleanup. The criteria were not adjusted after inspection.

## Results (256 target positions)

| Metric | Existing exact Riley | Experimental FlashInfer v2 | Direction |
| --- | ---: | ---: | --- |
| Mean target NLL (nats/token) | 2.95928943 | 2.95213500 | Candidate lower by 0.00715443 |
| Mean KL from FP32 (nats) | 0.0007591519 | 0.0007855655 | Candidate higher by 0.0000264136 |
| FP32 argmax matches | 244/256 | 243/256 | Candidate lower by 1 |
| Finite logits / full output counts / allocation release | Pass | Pass | Both |
| Relative screen | Reference | **FAIL** | KL condition not met |
| vLLM serving performance | Not measured | Not measured | No performance conclusion |

Candidate NLL is lower on six passages and higher on two. Candidate KL is lower on four and higher on four. [metrics.json](metrics.json) contains every passage, avoiding a conclusion from only favorable examples. The sample is small, FP32 is a different numerical contract, and these differences are not a general language-quality verdict. An NLL improvement does not override the failed predeclared KL condition; one fewer argmax match is also not a calibrated quality-loss percentage.

## Verification and artifacts

The Rust GPU fixture completed both backends, 256 output rows each, and zero retained CUDA allocations. Each raw logit file is 25,165,824 bytes (8 × 32 × 49,152 × 2); both raw files remain at `/tmp/riley-opt-260912/natural-screen-v1/{baseline,candidate}.bf16` and their SHA256 values are in metrics.json. Aggregate arithmetic and input lengths were independently reconciled after copying results. [sources.json](sources.json) confirms changed local/remote source equality, including the predeclared protocol. Binary/dependency-file hashes are retained separately.

The evaluator's exit success means metrics were produced; its authoritative `relative_screen_passed` is **false**. No memcheck or serving-latency claim is made from this run. The prior full32 memcheck evidence remains separate.

## Reproduction

Prepare cases with the offline tool's `prepare` mode, pointing `--model` to the pinned local checkpoint, `--passages` to the authored JSON, and `--directory` to a new output directory. Set `RILEY_NATURAL_SCREEN_DIR` to that directory alongside the existing pinned CUDA/FlashInfer/checkpoint environment, then run:

```sh
cargo test -p riley-scheduler --features cuda --test flashinfer_natural_logits_gpu -- --ignored --nocapture
python benchmarks/analysis/flashinfer_natural_screen.py evaluate --model MODEL_PATH --directory OUTPUT_DIRECTORY
```

## Decision

Keep existing exact serving as default. Preserve the failed screen and the separate free-generation exact-equivalence failure. Compare precision/backend alternatives against the same fixed contracts rather than tuning acceptance thresholds to this result. Any exploratory performance measurement must explicitly disclose the unaccepted numerical profile and cannot establish the final correctness-preserving serving goal. Formal baseline/new-Riley/vLLM reporting remains due at a meaningful serving integration milestone.
