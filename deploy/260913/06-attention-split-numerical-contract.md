# PR06 attention split: implementation and numerical gate

Status: native and experimental model integration implemented; model strict generation gate failed. Serving integration and qualification remain incomplete.

The [compatibility probe](../../benchmarks/results/20260914-attention-split-compatibility/README.md) rejects independent local-max BF16 partial merging under the existing exact profile. It does not reject every context-parallel implementation. Existing QK context splitting and eight output-dimension CTAs are already implemented; neither is a new optimization.

## Next implementation batch

1. Build an optional context-partitioned decode attention backend with FP32 partial maximum, denominator and value accumulation. Retain BF16 Q/K/V storage; specify every probability conversion and reduction order in its source/profile identity. Start with scalar FP32 probability/value arithmetic as the independent numerical control, not as a presumed fast serving path. A faster MMA implementation must pass the same gates against that control and the model reference.
2. Allocate bounded per-context partial scratch once, include split count and capacity in graph identity, and reduce partials with explicit empty-split handling. Validate ragged pages, inactive rows, non-multiple lengths, zero/one/max active rows, split changes and repeated graph replay. Unsupported extents must be rejected before enqueue; no allocation or Python call in replay.
3. Integrate through a distinct Rust → C ABI → CUDA experimental profile only after native correctness and timing. Keep the current exact backend available. Select short-context fallback using measured crossover including extra launches, scratch and merge; do not count fallback execution as candidate coverage.
4. Run full-model teacher forcing, independent generation, cache/batch invariance and lifecycle gates before matched serving measurement. Compare the current frozen Riley baseline, new profile and vLLM in reversed orders at a meaningful milestone. Include long-context workloads because the earlier C32 short-context matrix cannot establish a context-split benefit.

Do not resume rejected normalize-once or compensation tile variants without new evidence. Native speedup is insufficient to justify model integration if the native numerical contract fails.

## Immutable acceptance rules

Reuse the [existing natural protocol](../../benchmarks/fixtures/flashinfer-natural-v1/PROTOCOL.md), eight frozen passages and tokenization. Both aggregate NLL and KL(FP32 || candidate) must be no greater than the matched exact baseline; no tolerance is added. Preserve all per-passage metrics, full finite BF16 logits, input/model/tokenizer/source/binary hashes, FP32 reference configuration and zero-allocation cleanup receipt. Do not compare a changed model or fixture with historical scalar metrics.

The [metric adjudicator](../../benchmarks/analysis/check_attention_natural_screen.py) recalculates the existing aggregate and rejects malformed or inconsistent reports. Exit 0 means the relative metric screen passed, exit 1 means it failed, exit 2 means evidence is invalid. It does **not** verify raw-logit hashes against files, resource cleanup, backend execution, generation or serving. The producer's successful exit alone is insufficient: the historical evaluator exits successfully even when its metric screen fails.

Existing exact-profile logits and strict free-generation gates remain unchanged. For the separate experimental profile, report differences explicitly and retain the existing strict generation acceptance requirement; passing NLL/KL alone cannot waive it. A failed profile may be timed as a disclosed experiment but cannot satisfy correctness-preserving qualification. No calibrated non-exact generation acceptance policy has been established by this small dataset, and this document does not invent one.

For numerical isolation, compare partials against an independent higher-precision attention reference and include the known BF16 counterexample. Record errors rather than retrospectively choosing a tolerance. End-to-end acceptance remains governed by the existing model gates above, not an arbitrary primitive epsilon.

## Hardware, completion and rollback

Run supported native/model/serving paths on SM89. Compile SM90a and SM100a paths where available; runtime tests for absent Hopper/Blackwell/multi-GPU hardware remain explicit skips. Do not skip observed failures. Keep raw profiler databases private and avoid concurrent GPU workloads during measurement; restore the recorded Blender instances afterward and keep all three viewer services available.

Completion requires model correctness, demonstrated candidate execution, bounded scratch/lifecycle behavior, long-context serving improvement and no short-context regression. Report throughput, TTFT, TPOT and P95/P99 versus both Riley and vLLM. Default promotion additionally requires the broader serving goal; a passing numerical screen does not confer promotion. Rollback selects the existing exact profile and releases candidate graph/scratch through the normal owner lifecycle.

## Native control outcome — 2026-09-14

[Native report](../../benchmarks/results/20260914-attention-split-fp32/README.md): partial/merge FP32 control, bounded scratch, host extent checks and isolated graph probe implemented. 120 native combinations completed; bounded memcheck/racecheck 0, SM90a/SM100a compile passed with runtime skips. Long-context P×V+merge timings are promising, but short-context overhead regresses. Sampled FP64 errors and legacy differences are reported without declaring numerical acceptance. Rust profile/graph owner, full-model gates and matched serving are still pending.

## Model integration outcome — 2026-09-14

[모델 결과](../../benchmarks/results/20260914-attention-split-model/README.md): Rust owner/C ABI/graph identity와 2,613,252-byte retained workspace를 연결했다. 초기 pure-decode-only 연결은 batch invariance에 실패했으며, mixed decode에도 같은 계산을 적용한 v2에서 해당 불변성이 복구됐다. 자연어 NLL/KL은 기존 사전 기준을 통과했지만 strict 독립 생성은 168/1,024 tokens 불일치로 실패했다. 전체 모델 memcheck는 오류 0이며 생성 assertion의 exit 101과 구분한다. 기본값·서버 선택은 추가하지 않았고 serving/vLLM 검증 및 PR06 승격은 미완료다. 기존 정확성 기준을 완화하거나 합성 long-context 시간만으로 도입하지 않는다.
