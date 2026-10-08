# Riley Decode 성능 개선 구현 보고서

**작성일:** 2026-08-27
**기준 commit:** `436dad64526f228292c24cec73a652a72e0b1e38`
**작업 branch:** `codex/decode-performance-optimization`
**현재 판정:** Phase 1~3과 cooperative shared-KV attention의 `ai-assistant` RTX 4090 correctness 및 5쌍 AB/BA M1 성능 gate 통과
**성능 완료 판정:** M1 달성 — TPOT p50 42.7% 개선. M2/M3, vLLM 동일 campaign 및 production 승격은 아직 판정하지 않음.

## 1. 요약

[decode 성능 개선 계획](./decode-performance-improvement-plan.md)의 첫 구조 개선 세 가지와
profile-guided GQA decode attention 개선을 구현했다.

1. 실제 active row 수를 사용하는 dense shape bucket
2. token과 metadata 6종을 한 번에 보내는 packed asynchronous H2D
3. GPU exact greedy argmax와 token/status-only D2H
4. SmolLM2 `QH=9/KVH=3`의 BF16 K/V tile을 한 번만 읽는 shared-KV attention과,
   세 query-head warp의 cooperative tile preload

기존 경로는 production 기본값이자 rollback으로 그대로 유지한다. 현재 기본값은
`fixed-max + synchronous + cpu`이며, fast path는 명시적인 selection에서만 활성화된다.
로컬에는 CUDA toolkit/GPU가 없으므로, native compile·정합성·성능은 지정 SSH alias
`ai-assistant`의 RTX 4090에서 clean source archive와 release ELF를 고정해 검증했다.

## 2. 구현 범위

### 2.1 Phase 1 — active-row shape bucket

- 하나의 executor가 weights, paged KV cache, RoPE table, forward buffer를 공유한다.
- bucket별로 execution plan과 GEMM plan만 cold prepare한다.
- active row 수 `T` 이상인 가장 작은 prepared shape `M`을 선택한다.
- non-power-of-two maximum은 마지막 catch-all bucket으로 유지한다.
- 기본 bucket은 `1,2,4,...,max_input_tokens`이고, 최대 10개의 custom bucket도
  `1` 시작, 엄격 증가, 마지막 값 `max_input_tokens` 조건으로 cold validation한다.
- output gather도 선택된 `M`을 사용한다.
- 마지막 성공 iteration의 active/selected/padding row와 bucket별 hit count를 allocation 없이
  조회할 수 있다.
- shape별 GEMM 탐색은 사용자가 설정한 workspace cap을 그대로 사용한다. 각 shape가 요구한
  workspace의 최댓값만 cold prepare에서 한 번 shared buffer로 확보하고, hot path에서는
  grow하거나 다시 할당하지 않는다.
- bucket GEMM은 최대 row(`M=max`) plan이 선택한 opaque `cublasLtMatmulAlgo_t`를
  child descriptor에 `cublasLtMatmulAlgoCheck`로 재검증해 사용한다. metadata를 다시
  조합하는 heuristic은 사용하지 않는다. 호환하지 않는 child shape는 그 bucket을 준비하지
  않고, 다음 anchored bucket 또는 최대 row plan으로 exact fallback한다.
- fixed maximum 경로는 변경 없이 rollback으로 남는다.
- 서버 CLI:

      --batch-shape-policy fixed-max
      --batch-shape-policy power-of-two
      --batch-shape-buckets 1,2,4,8,...,512

기본값은 `fixed-max`다.

운영 관측도 같은 정책 경계에 연결했다.

- CUDA worker는 cold path에서 최대 10개 bucket의 고정 배열 collector를 만든다.
- `active_rows`는 request 수가 아니라 flattened input-token row이며, scheduler plan의
  `total_tokens()`와 일치하는지 확인한다.
- scheduler commit 성공과 iteration metric 생성이 모두 확인된 iteration만 hit 및 latency에
  포함한다. D2H, sampling, result 또는 commit 실패는 성공 통계에 섞이지 않는다.
- bucket별 hit, active/selected/padding row, CUDA execution latency의 total/average/max/last를
  live `/metrics`와 shutdown JSON에 공개한다.
- counter overflow나 shape 불일치는 inference를 실패시키지 않고 sticky
  `metrics_degraded`로 격리한다.
- `fixed-max`도 동일 schema에서 bucket 하나로 기록되므로 baseline/candidate의 관측 형식이
  달라지지 않는다.

### 2.2 Phase 2 — packed asynchronous metadata H2D

- token ID와 metadata 6종을 aligned host slab에 pack한다.
- cold prepare에서 일반 host pack buffer, pinned host slab, device slab을 각각 한 번
  할당한다.
- 모든 host shape/offset/range 검증을 H2D enqueue 전에 수행한다.
- active command batch 안에서 pinned host slab을 device slab으로 한 번 복사한다.
- 같은 stream에서 device subspan을 embedding, batch metadata, output gather가 소비한다.
- H2D마다 별도 completion token이나 synchronize를 만들지 않는다.
- command-batch finish가 pinned/device resource lease의 단일 완료 경계다.
- public Rust lifetime이 pinned/device buffer borrow를 command stream lifetime에 묶어,
  command-batch 완료 전 drop, close 또는 mutable reuse를 컴파일 단계에서 차단한다.
- pre-dispatch 실패와 command submission 시작 이후 실패를 구분한다. batch 전체 poison은
  실제 submission이 시작된 뒤의 실패에만 적용하고, 기존 operation별 poison 계약은 유지한다.
- 기존 token+metadata 6종 동기 업로드는 그대로 유지한다.
- 서버 CLI:

      --metadata-transport synchronous
      --metadata-transport packed-async

기본값은 `synchronous`다. `packed-async`는 `iteration-batch` completion과 함께만
허용하며, `per-operation` 조합은 device mutation 전 cold validation에서 거부한다.

### 2.3 Phase 3 — exact GPU greedy

- BF16 `[rows, vocabulary]` logits를 GPU에서 deterministic argmax한다.
- 동률은 CPU contract와 동일하게 가장 작은 token ID를 선택한다.
- NaN 또는 infinity가 하나라도 있는 row는 `{u32::MAX, NON_FINITE}` status를 반환한다.
- 결과 ABI는 row별 U32 두 단어 `{token_id, status}`다.
- greedy path는 RNG를 사용하지 않는다.
- host D2H는 full logits 대신 row당 8 bytes만 전송한다.
- scheduler, server, native benchmark는 cold-reserved token workspace를 반복 재사용하며
  성공, sampling 실패, abort, commit 실패 경로에서도 allocation ownership을 회수한다.
- invalid status 또는 vocabulary 밖 성공 token은 native-integrity 오류로 보고 executor를
  poison한다. 정상적인 NON_FINITE row 오류는 별도로 보고한다.
- 다음 조건을 모두 만족하는 request iteration에만 fast path를 적용한다.

  - 명시적으로 GPU greedy backend를 활성화함
  - temperature가 정확히 0
  - repetition penalty가 정확히 1
  - masked finish token이 없음
  - tokenizer가 model vocabulary 전체를 address할 수 있음

- 조건이 하나라도 맞지 않으면 기존 full-logits/CPU sampler를 dispatch 전에 선택한다.
- 서버 CLI:

      --sampling-backend cpu
      --sampling-backend gpu-greedy

기본값은 `cpu`다.

## 3. 공정 비교 경로

`riley-profile`과 `check_native_profile_pair.py`에 다음 독립 비교 축을 추가했다.

| 비교 목적 | baseline | candidate |
|---|---|---|
| shape | `batch_shape_policy=fixed-max` | `batch_shape_policy=power-of-two` |
| metadata | `metadata_transport=synchronous` | `metadata_transport=packed-async` |
| output | `greedy_output=cpu-logits` | `greedy_output=gpu-token` |
| Phase 1~3 결합 | `decode_fast_path=fixed-sync-cpu` | `decode_fast_path=bucket-packed-gpu` |

각 축은 독립 correctness gate ID를 사용한다. checker는 baseline/candidate role을 반대로
기록하거나 서로 다른 축을 섞은 evidence를 incomparable로 거부한다.

## 4. 검증 현황

### 4.1 통과한 로컬 검증

- `riley-cuda` feature-off unit test: 70 passed
- `riley-cuda` host runtime/source contract test: 9 passed
- `riley-cuda` compile-fail doctest: 17 passed
- `riley-runtime` no-default all-targets: 122 unit + 1 architecture-boundary passed
- `riley-scheduler` no-default all-targets: 27 unit + 14 simulation passed
- `riley-server` no-default target: 1 passed
- `riley-server` server feature: 46 library + 9 CLI passed
- native profile pair checker: 21 passed
- reliability, Python-free release E2E, release candidate/evidence, native profile checker:
  193 Python contract tests passed
- C11 public ABI layout compile: passed
- CUDA feature Rust type-check/strict Clippy: 이번 변경의 CUDA library, GPU test,
  scheduler integration, server/profiler target passed
- `riley-model` 사용자 변경 diff hash 보존:
  `d8b3420ac30bba5e1265a84c22af39c8a0507519f4f1bef21643e31d63b128a6`

CUDA feature type-check에는 native CUDA build만 건너뛰는 임시 build-script 분기를 사용했고,
검사 직후 제거했다. 그 분기는 최종 diff에 포함되지 않는다. 따라서 이 결과는 Rust의
feature-gated API/type 계약만 증명하며 CUDA C++ compile/link 또는 GPU 실행을 증명하지
않는다. 아래 4.2의 별도 GPU 실행 결과가 native 경로를 보완한다.

참고로 저장소 전체 CUDA `--all-targets -D warnings`는 이번 변경과 무관하고 수정하지 않은
`primitives_gpu.rs`의 `items_after_statements` 2건과 `smollm_cache_empty_gpu.rs`의
`float_cmp` 2건에서 중단된다. 이번에 추가·수정한 CUDA target과 server 전체 target은
strict Clippy를 통과했다.

### 4.2 `ai-assistant` RTX 4090에서 통과한 GPU gate

- 환경: NVIDIA GeForce RTX 4090 (compute capability 8.9), CUDA runtime 12.8.1,
  cuBLAS 12.8.4.1, immutable image
  `sha256:f51d74009d8a5abd2aa0115ab51967aca200f99c0c0ffbafcff603212af258c1`
- native CUDA C++ compile/link: passed
- BF16 GPU greedy argmax semantic test: 3 passed
- pinned/device memory 및 packed H2D lifecycle test: 7 passed
- scheduler prefill→decode plan/commit integration: passed
- real CUDA HTTP lifecycle E2E: passed
- 최대-row anchor의 opaque cuBLASLt algorithm이 SmolLM2 projection 5종,
  `M={1,2,4,8,16,32,64,128}`에서 zero-padded `M=256` 결과의 active-row BF16 bytes와
  동일한지 확인하는 low-level test: passed
- 같은 test에서 서로 다른 N/policy 및 다른 CUDA context의 anchor 사용은 거부됨을 확인
- active-row bucket의 prefill, mixed prefill/decode, KV boundary와 shape-history가
  fixed-max logits/top-1/KV continuation과 일치하는 GPU test: passed

이 결과는 child M마다 새 heuristic algorithm을 고르는 이전 방식이 아니라, maximum plan의
opaque algorithm을 그대로 보존한 anchored 방식의 결과다. child descriptor가 해당 algorithm을
지원하지 않으면 `NotSupported`으로 bucket을 제외하며, M-specific algorithm으로 바꾸지 않는다.

### 4.3 추가·유지한 correctness 범위

- active-row decode `T={1,2,8,17}`
- prefill, mixed prefill/decode, KV boundary `15→16→17`
- shape history `128→1→8→256→1`
- fixed-max와 bucket mode의 logits/top-1/KV continuation parity
- synchronous, iteration-batch synchronous, packed-async의 multi-step parity
- packed H2D allocation/close와 fallback lifecycle
- low-level BF16 argmax의 tie, first/last token, signed zero, all-negative,
  NaN/+Inf/-Inf, command-batch ordering
- scheduler greedy download와 output-slot routing

기존 release evidence가 고정한 ignored test inventory와 filtered-count를 유지하도록 새
검증은 기존 gate를 확장하거나 별도 required-feature target으로 분리했다.

### 4.4 baseline에서 이미 드러난 별도 GPU gate 문제

전체 ignored `llama_batch_gpu` 모음에는 이번 변경 이전 commit에서도 같은 방식으로 실패하는
두 개의 stale gate가 있다. candidate 회귀로 분류하거나 threshold를 낮춰 통과시키지 않았다.

- fixed-37 fixture의 실제 SHA-256은
  `57b41e8faf7cc044e9eeb235011109224cb6cddfd19719920641a1662cc41fd3`인데 test에는
  `87333a1859be45a2f8e7563d898dde5e64256ccc03ca4da3cab90def07dd3c95`가 고정돼 있다.
- 32-step generic numeric comparison은 baseline과 candidate가 동일 token을 내도 step 20의
  cosine threshold에서 동일하게 실패한다 (`0.997528… < 0.997903…`).

이 두 gate는 별도 fixture/threshold 정비 작업으로 남기고, 이번 decode fast path의
correctness 증거에는 새로 추가한 exact anchor test와 active-row end-to-end parity test를
사용한다.

### 4.5 아직 실행하지 못한 검증

- Phase 0 current/vLLM default/vLLM eager AB/BA baseline
- Phase 1, 2, 3 개별 paired 성능 gate
- c8·장문맥 회귀 cell과 deployed production HTTP E2E
- CUDA Graph capture/replay lifecycle gate 및 M2/M3 gate

### 4.6 최종 M1 paired 성능 및 NCU 증거

최종 source `436dad64526f228292c24cec73a652a72e0b1e38`를 tar archive
`385107119fef5ebe65a80632bc3b4c765a77fea9bac5a07feb51331fc3ac03fa`로 고정하고,
`ai-assistant` RTX 4090에서 만든 release ELF
`a6870817235c80342878258a689973d979ebdcd0e04bc2638235a82a135b1ffc`를 사용했다.
각 arm은 warmup 5회 뒤 30 requests를 측정했고, B,C,C,B,B,C,C,B,B,C 순서의 독립 process
5쌍으로 실행했다. strict checker는 failures=0, dropped traces=0과 모든 p95/throughput/
primary gate를 통과했다.

| 지표 | fixed-sync-cpu median | cooperative candidate median | 변화 |
|---|---:|---:|---:|
| TTFT p50 | 5.450 ms | 4.033 ms | 25.98% 개선 |
| TPOT p50 | 7.166 ms | 4.109 ms | **42.66% 개선** |
| E2E p50 | 227.595 ms | 131.404 ms | 42.26% 개선 |
| 처리량 | 140.535 tok/s | 243.536 tok/s | 73.29% 증가 |
| host execute | 6.704 ms | 3.941 ms | 41.22% 개선 |

TPOT p95 candidate/baseline ratio는 `0.5724`, TTFT p95 ratio는 `0.7368`이다. 원본 10개 JSON,
replay script와 checker report는
[`cooperative-kv-profile`](../benchmarks/results/20260827T101334Z-riley-decode-fastpath-pr16/cooperative-kv-profile/)에 있고,
정합성 receipt는
[`correctness-report-cooperative-shared-kv-heads.json`](../benchmarks/results/20260827T101334Z-riley-decode-fastpath-pr16/correctness-report-cooperative-shared-kv-heads.json)에 있다.

NCU `--launch-skip 500 --launch-count 600` attribution에서는 최종
`ragged_paged_attention_gqa_shared_kv_kernel`의 steady decode 30회가 합계 `1.993888 ms`,
평균 `66.463 µs/call`이었다. 직전 serial shared-KV prototype의 같은 kernel은 30회 합계
`3.055296 ms`, 평균 `101.843 µs/call`이었다. NCU window는 AB/BA 성능 판단용이 아니라
병목 귀속용이므로, 채택 근거는 위 strict paired profile이며 NCU는 cooperative preload가
attention 메모리 대기 시간을 실제로 줄였다는 보조 증거다.

이 결과로 M1의 30% TPOT 목표는 초과 달성했다. 다만 사용자가 제시한 vLLM 1.12 ms와는
동일 campaign 비교가 아니며, 수치상 아직 약 3.7배의 TPOT gap이 남는다. 이 보고서는
vLLM 우위가 해소됐다고 주장하지 않는다.

## 5. Phase 4와 Phase 5 중단 판정

### 5.1 CUDA Graph

Phase 4는 아직 시작하지 않는다.

- M1 paired 성능 gate는 통과했지만, final candidate의 host execute와 CUDA stream span은
  960 iteration 기준 각각 `3.9406 s`, `3.9359 s`로 차이가 약 `4.8 µs/iteration`뿐이다.
  따라서 graph는 lifecycle/CPU jitter 정리에는 가치가 있어도 M2의 큰 TPOT 개선을 단독으로
  만들 수 없다는 것이 현재 증거다.
- 현재 packed slab allocation 자체는 안정적이지만, 실제 metadata 길이에 따라 일부
  subspan offset이 달라질 수 있다. graph key에 전체 layout signature를 포함하거나,
  fixed-offset layout 또는 graph node parameter update를 설계해야 한다.
- 기존 embedding path 내부의 host transfer/synchronization도 capture-safe 경계로
  분리되어야 한다.
- graph capture/replay 전용 lifecycle test 없이 capture/replay owner를 production path에
  넣지 않는다.

따라서 Phase 4는 M1 통과 후에도 stable-layout/capture lifecycle 설계와 별도 graph-value
가설이 필요한 `Deferred` 상태로 기록한다.

### 5.2 Kernel fusion

Phase 5의 attention profiling slice는 실행했다. generic grouped CTA는 legacy보다 느려
승격하지 않았고, shared-KV와 cooperative preload만 strict paired M1 gate를 통과했다.
그 외 fusion 후보는 아직 시작하지 않는다.

- QKV, gate/up, SiLU×gate 중 어느 후보를 먼저 선택할지는 M1 이후 새 profile이
  결정해야 한다.
- Q/K/V width가 다른 M>1 layout, BF16 intermediate rounding, packed weight ownership을
  GPU에서 확인하지 않은 채 fusion을 기본 경로에 넣을 수 없다.
- 계획서의 후보별 단독 5% 개선 gate를 현재 환경에서 측정할 수 없다.

따라서 나머지 Phase 5는 `Deferred until a new post-M1 GPU profile selects one candidate`로 기록한다.

## 6. 실행한 성능 campaign

`ai-assistant`의 분리된 candidate workspace에서 다음 순서로 진행한다.

1. scoped implementation을 clean candidate commit으로 고정하고 exact archive를 전달한다.
2. 같은 profile ELF, model/tokenizer hash, GPU UUID, container image를 receipt에 기록한다.
3. `fixed-sync-cpu` baseline과 `bucket-packed-gpu` candidate를 5쌍 AB/BA 순서로 실행한다.
4. warm-up 5회 후 30회 측정으로 TPOT, E2E, throughput, TTFT, allocation 및 correctness
   report hash를 수집한다.
5. strict checker로 M1 combined gate를 판정했다. 5쌍 모두 통과했고 primary paired median
   개선은 41.20%였다.
6. default 승격은 하지 않았다. vLLM 동일 campaign, c8/장문맥, production HTTP와 M2/M3 gate가
   남아 있으므로 `fixed-max + synchronous + cpu` rollback/default를 유지한다.

## 7. Commit/Push 기준

사용자는 구현 완료 후 commit/push를 요청했다. scoped implementation은 exact clean commits로
고정했고, 최종 paired performance evidence와 문서를 별도 commit으로 추가한 뒤 함께 push한다.

커밋 시에는 이 작업의 파일만 정확히 stage한다. 기존 `crates/riley-model/**` 변경은
사용자 작업으로 간주하여 stage, commit, 정리하지 않는다.
