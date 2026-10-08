# PR-K — post-graph profile-selected kernel/fusion cards

**목적:** 새 graph/candidate 상태에서 실제 최대 병목 하나만 고르고, 한 PR에 한 optimization만
구현한다. profiler 없이 C09/C10/C11을 시작하지 않는다.

**선행:** G03C correctness, G04C dispatch contract, B02 Tier D. graph가 성능 gate 미달로
`not-promoted`여도 eager와 experimental graph를 별도 arm으로 profile할 수 있다.

## 공통 discovery PR: PR-K00

**범위:** instrumentation와 benchmark artifact만 변경한다.

- `c1/p128/o32`, `c8/p128/o32`, `c1/p4096/o128`에서 eager/graph path의 kernel duration,
  CUDA API gap, LM-head logits bytes, projection/attention time, VRAM/KV capacity를 수집한다.
- fresh process/ABBA, warmup 분리, exact candidate receipt를 유지한다.
- output은 ranked bottleneck 하나와 candidate 선택/비선택 근거다. kernel implementation은 넣지 않는다.

**선택 규칙:** primary TPOT의 유의미한 비중을 차지하고 해당 operation의 exact fallback과
correctness contract가 명확한 후보만 다음 PR로 간다. 예상 최대 절감이 candidate별 promotion
threshold보다 작으면 구현하지 않고 `not-selected`로 기록한다.

## 후보 PR-K01: C11 LM-head + greedy fusion

**시작 조건:** LM-head logits materialization/write가 selected bottleneck.

**범위:** temperature 0 greedy에서 full logits materialization 없이 tile-local max/token/status를
deterministic reduction해 작은 output만 남긴다. stochastic sampling, masked finish token, generic LM head는 fallback이다.

**변경 표면:** `kernels/src/lm_head_*.cu`, CUDA ABI/wrapper, `executor/output.rs`, runtime pattern entry,
GPU parity tests, immutable benchmark artifact.

**gate:** 128-step greedy token/status exact, LM-head+selection GPU time `>=15%`, E2E TPOT `>=5%`,
c8 throughput `>=0.95`, TTFT/long p95 `<=1.05`, peak VRAM `<=1.05`, hot allocation 0.

## 후보 PR-K02: C09 packed QKV 또는 gate/up projection

**시작 조건:** projection memory traffic/launch가 selected bottleneck이며 exact packed weight identity를
cold prepare에서 보장할 수 있음.

**범위:** one projection family only; Python-free cold packing, original/packed weight ownership,
GEMM plan compatibility, exact reference fallback.

**gate:** BF16/token/KV parity, packed-owner close/accounting 0, VRAM/KV non-regression, selected operation
and primary TPOT의 사전 선언 threshold 통과. QKV와 gate/up을 같은 PR에서 합치지 않는다.

## 후보 PR-K03: C10 subgraph fusion

**시작 조건:** profiler가 residual+norm, RoPE+KV write, SiLU×gate 중 정확히 하나를 선택.

**범위:** one semantic pattern and one model geometry; rounding/alias/lifetime contract와 reference fallback.

**gate:** candidate 단독 `>=5%` paired improvement, token/KV numeric gate, all common regression gate.
generic grouped CTA와 과거 residual+RMSNorm prototype은 gate 미달 이력이 있으므로 새 evidence 없이는 재도입하지 않는다.

## 공통 종료

각 후보는 `promoted`, `experimental/not-promoted`, `rejected` 중 하나로 종료한다. 다음 후보 또는
M4/M5 재실행은 이 candidate revision을 기준으로 다시 profile/campaign을 만든다.

한 후보가 promoted되면 다른 후보를 이어서 구현하기 전에 K00을 새 revision에서 다시 실행한다.
selected candidate가 없으면 B01/B02 재-freeze 없이 현재 graph candidate로 B03/B04에 진행한다.
