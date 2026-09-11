# C11 — LM-head와 Greedy Output Fusion

**상태:** Planned  
**의미 등급:** `E0` — 이 PR은 temperature 0 greedy만 다룸  
**한 가지 목적:** small-batch decode에서 full vocabulary logits materialization과 별도 argmax pass를 제거하고 LM-head 계산 중 deterministic top-1을 산출한다.

[이전: C10](10-transformer-subgraph-fusion.md) | [목차](README.md) | [다음: C12](12-tenant-safe-prefix-cache.md)

## 1. 범위 제한

이름에 sampling이 포함되지만 stochastic top-k/top-p RNG sampling은 이 PR에 넣지 않는다. 동일 PR에서 `E0` greedy와 `E1` distribution-preserving sampling을 섞지 않는다.

후속 stochastic GPU sampling은 별도 admission/contract에서 다음을 먼저 고정해야 한다.

- request-local RNG algorithm/version
- exact logits processing order
- snapshot/restore/fork
- rejected/cancelled branch 소비 정책
- distribution 검증

C11의 production candidate는 오직 greedy fast path다.

## 2. 현재 문제

현재 GPU greedy는 full LM-head logits를 device memory에 만든 뒤 별도 argmax kernel이 token/status를 생성한다. D2H는 작아졌지만 다음 비용은 남아 있다.

- full `[rows, vocab]` logits write
- logits buffer VRAM/workspace
- 별도 argmax kernel launch
- vocab 전체의 두 번째 read

small-batch decode에서는 LM-head 자체와 logits memory traffic이 TPOT의 유의미한 비중이 될 수 있다.

## 3. 목표 실행

```text
hidden row
  -> LM-head tiled GEMM/GEMV
  -> tile-local (max_value, token_id, status)
  -> deterministic global reduction
  -> {token_id, status}
```

full logits tensor는 greedy fast path에서 materialize하지 않는다. exact CPU/full-logits fallback과 debug/quality capture path는 유지한다.

## 4. Reduction contract

partial state:

```rust
struct GreedyPartial {
    value: f32,
    token_id: u32,
    non_finite_seen: bool,
}
```

merge order와 tie rule:

1. non-finite policy는 현재 GPU greedy contract와 동일
2. 더 큰 logit 선택
3. 값이 동일하면 더 작은 token ID 선택
4. signed zero 비교 의미 고정
5. padding vocabulary ID는 후보에서 제외

reduction tree가 block/grid에 따라 달라져도 tie/token semantics가 deterministic해야 한다.

## 5. 구현 후보

### 후보 A — cuBLASLt output + fused epilogue 불가 시 streamed chunk

vocabulary를 고정 chunk로 나누어 chunk GEMM과 partial top-1을 수행하고 logits chunk를 재사용 가능한 작은 workspace에만 둔다.

장점: cuBLASLt 유지.  
위험: 여러 GEMM submission과 chunk loop.

### 후보 B — CUTLASS/custom LM-head top-1

LM-head matmul tile의 accumulator에서 local max를 계산하고 global partial만 write한다.

장점: full logits write 제거.  
위험: model shape별 GEMM 성능·유지보수 부담.

선택 순서:

```text
cuBLASLt baseline profile
-> streamed/chunk prototype
-> CUTLASS prototype
-> same end-to-end campaign
-> 승리한 구현만 registry candidate
```

NVIDIA general GEMM을 무조건 custom으로 대체하지 않는다.

## 6. Tied weight와 layout

- tied embedding/LM-head storage ownership 유지
- transposed/non-transposed safetensors layout을 load-time descriptor로 정규화
- vocabulary logical size와 padded physical size 분리
- model revision/weight hash가 graph/implementation signature에 포함
- output token ID가 tokenizer addressable vocabulary 범위인지 검증

## 7. Capability

최초 지원:

- BF16 weight/activation
- FP32 accumulator
- active rows `1,2,4,8` 우선, profile 후 `16,32`
- temperature 0
- repetition penalty 1
- finish token masking 없음 또는 prevalidated closed mask capability

unsupported 조건에서는 current full-logits CPU/GPU path로 fallback한다.

## 8. Registry

```text
LmHeadGreedy:
  lm-head-top1-<backend>-v1
  semantic class: E0
  fallback: lm-head-full-logits-gpu-argmax
```

implementation ID는 C07 graph signature에 포함한다. full-logits graph와 fused-top1 graph는 공유하지 않는다.

## 9. Correctness

### Low-level

- first/last vocabulary maximum
- all-negative
- exact tie across tiles/blocks
- signed zero
- NaN, +Inf, -Inf
- padded vocabulary sentinel
- odd vocabulary tail
- repeated deterministic output

### End-to-end

- SmolLM2/Qwen
- prompt/output matrix
- graph disabled/enabled
- active rows `1,2,4,8`
- 128-step token exact
- full-logits fallback과 generated hash exact
- cancellation/commit failure/close

가능하면 sampled token뿐 아니라 debug mode에서 selected token의 logit을 full reference와 비교한다.

## 10. Memory contract

- greedy mode full logits persistent/workspace를 제거하거나 다른 path와 명시적으로 공유
- fallback용 full logits buffer를 항상 예약할지 on-demand cold pool로 둘지 사전 결정
- hot path allocation 금지
- graph retained bytes와 KV capacity trade-off 기록

fallback buffer를 제거해 memory를 절약하더라도 unsupported request가 runtime allocation을 유발하면 안 된다. startup에 bounded fallback workspace를 준비하거나 해당 server profile에서 unsupported sampling을 admission 단계에 거부한다.

## 11. Performance campaign

- baseline: current LM-head + GPU argmax
- candidate A/B 개별 arm
- primary `c1/p128/o32`
- required `c4,c8`, `p4096/o128`
- model sizes diagnostic 135M과 경쟁용 최소 0.5B 하나

Metric:

- LM-head GPU duration
- logits bytes written/read
- kernel/GEMM launch count
- TPOT/TTFT/E2E/throughput
- workspace/peak VRAM/KV blocks
- graph instantiate/replay cost

## 12. Promotion gate

- failure/token/status mismatch 0
- greedy token exact 128 steps 이상
- LM-head+selection GPU time improvement `>= 15%`
- end-to-end primary TPOT improvement `>= 5%`
- c8 throughput ratio `>= 0.95`
- TTFT/long p95 ratio `<= 1.05`
- peak VRAM ratio `<= 1.00` 목표, 최대 `1.05`
- hot allocation 0

## 13. 예상 파일 변경

```text
kernels/src/lm_head_*.cu 또는 CUTLASS integration
kernels/include/riley_cuda.h
crates/riley-cuda/src/primitives.rs
crates/riley-runtime/src/pattern.rs
crates/riley-runtime/src/llama/executor/output.rs
crates/riley-runtime/src/llama/executor/graph.rs
crates/riley-runtime/tests/lm_head_greedy_gpu.rs
benchmarks/results/<campaign>/...
```

## 14. 실패와 rollback

- capability mismatch: current full-logits path
- candidate pre-dispatch failure: fallback 가능
- candidate launch 후 mutation 불명확: 자동 full-logits 재실행 금지
- 운영 rollback: `LmHeadGreedy` implementation override를 reference로 전환

## 15. 완료 정의

온도 0 greedy 요청에서 full logits materialization 없이 동일 token/status를 내고, end-to-end 5% gate와 memory/lifecycle 기준을 통과해 registry 승격 여부를 결정할 때 완료다.
