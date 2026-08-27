# Riley Decode 성능 개선 계획

**상태:** M1 달성 / Phase 1~3 및 cooperative shared-KV attention의 로컬·`ai-assistant` RTX 4090 correctness와 5쌍 AB/BA 성능 gate 통과 / M2~M3 후속 작업 대기
**작성일:** 2026-08-27
**대상:** SmolLM2-135M BF16, RTX 4090, concurrency 1의 짧은 decode를 시작점으로 한 Riley native runtime
**기본 원칙:** correctness 우선, Python-free production 유지, 단계별 독립 측정, exact fallback과 롤백 경로 유지

## 1. 문서 목적

현재 Riley는 첫 토큰 시간(TTFT)은 vLLM보다 빠르지만, 다음 토큰 시간(TPOT)이 약
6.4배 느리다. 이 문서는 그 차이를 한 번에 여러 최적화로 덮지 않고, 병목을 독립적인
PR과 검증 gate로 나누어 개선하기 위한 실행 계획이다.

이 계획은 사용자 승인을 받아 구현 단계로 전환됐다. 실제 진행 상태와 검증 결과,
중단 gate는 [구현 보고서](./decode-performance-implementation-report.md)에 기록한다.
계획서의 threshold와 완료 정의는 결과를 본 뒤 완화하지 않는다.

## 2. 현재 기준선과 문제 정의

사용자가 제시한 warm c1/p128/o32 결과는 다음과 같다.

| 지표 | Riley | vLLM 0.27.1 | 차이 |
|---|---:|---:|---:|
| TTFT p50 | 5.45 ms | 6.91 ms | Riley 21.1% 빠름 |
| TPOT p50 | 7.16 ms | 1.12 ms | vLLM 6.42배 빠름 |
| E2E p50 | 227.53 ms | 41.53 ms | vLLM 5.48배 빠름 |
| 처리량 median | 140.65 tok/s | 770.25 tok/s | vLLM 5.48배 빠름 |
| 실패 | 0/150 | 0/150 | 동일 |

출력 32토큰에서 E2E와 처리량은 별도 병목이 아니라 TPOT의 산술 결과다.

    Riley E2E ≈ 5.45 + 31 × 7.16 = 227.41 ms
    Riley throughput ≈ 32 / 0.22753 = 140.64 tok/s

따라서 최적화의 primary metric은 warm c1/p128/o32의
`request mean TPOT p50`으로 고정한다. E2E와 concurrency 1 처리량은 이를 확인하는
secondary metric으로 사용한다. concurrency 1의 output tok/s를 고동시성 포화
처리량과 같은 의미로 해석하지 않는다.

### 2.1 현재 증거

| 원인 | 현재 증거 | 판단 |
|---|---|---|
| 고정 dense row 수 | 모든 iteration이 `M=max_input_tokens`로 실행되고 서버 기본값이 512다. c1 decode는 실제 1행과 padding 511행을 사용한다. | 최우선 구조 문제 |
| 많은 eager launch | decode 1회에 kernel 517개, NCU 합산 GPU kernel 시간은 약 2.368 ms다. | host launch/gap 문제 |
| 동기식 metadata 전송 | token과 metadata 배열 6개가 개별 업로드되며 각 copy helper가 completion을 기다린다. | CUDA Graph 선행 병목 |
| full logits D2H | SmolLM2의 BF16 vocab row 49,152개, 즉 98,304 bytes를 토큰마다 host로 가져온다. | 작은 batch에서 불필요한 경계 |
| CPU sampling | full-vocab host buffer를 초기화·순회한 뒤 greedy token을 선택한다. | GPU top-1 fast path 필요 |
| vLLM 실행 방식 | 비교 실행은 `enforce_eager=False`, AOT compile, full/piecewise CUDA Graph capture size 1/2를 사용한다. | graph replay 격차 확인 |

근거 소스:

- [고정-M batch executor](../crates/riley-runtime/src/llama/batch_executor.rs)
- [서버 batch-token-budget 기본값과 executor 연결](../crates/riley-server/src/main.rs)
- [PR15 profiling 결과](../deploy/15-profiling-and-optimization.md)
- [iteration 실행과 logits download](../crates/riley-scheduler/src/execution.rs)
- [CPU sampling contract](../crates/riley-runtime/src/sampling.rs)
- [vLLM 실행 receipt](../benchmarks/results/20260824T192344Z-vllm-repeatability-pr01-v2-run003/runs/run-01/cell-01-warm-c1-p128-o32/benchmark.stdout.txt)

### 2.2 baseline 주의사항

체크인된 Riley release performance baseline은 과거 PR15 commit
`1826d09fd28825582f22a2d49390f6ed047562ea`에 묶여 있고, vLLM repeatability
artifact는 다른 Riley repository revision에서 생성됐다. 작성 시점 HEAD도 사용자
작업이 있는 dirty worktree이므로 새 성능 기준선으로 사용할 수 없다.

구현 시작 전 반드시 다음을 새로 만든다.

1. 승인된 clean Riley candidate commit
2. 같은 날, 같은 RTX 4090, 같은 model/token IDs에서 측정한 Riley current baseline
3. 같은 campaign에서 측정한 vLLM optimized default baseline
4. 원인 귀속용 vLLM eager/no-graph baseline

과거 artifact는 회귀와 역사 비교용으로 보존하며 새 결과로 덮어쓰지 않는다.

## 3. 목표와 성공 기준

### 3.1 최종 목표

1. c1 decode에서 실제 active row 수에 맞는 dense shape를 실행한다.
2. steady-state decode의 host kernel submission을 CUDA Graph replay 1회로 줄인다.
3. greedy 요청에서 full-vocab D2H와 CPU full-vocab scan을 제거한다.
4. exact eager/CPU path를 항상 롤백 경로로 유지한다.
5. TTFT, mixed batch, c8 처리량, long-context 정확성을 희생하지 않는다.

### 3.2 단계 목표

아래 절대값은 현재 사용자 측정 7.16 ms를 기준으로 한 **사전 목표**다. Phase 0에서
동일 revision의 clean baseline을 다시 고정한 뒤, 환경이 달라졌다면 측정 전에만
baseline-relative 값으로 재결합한다. 결과를 본 뒤 threshold를 완화하지 않는다.

| milestone | 목표 | 현재 수치 기준 해석 |
|---|---|---:|
| M1 — active-row 실행 | TPOT p50 30% 이상 개선 | 5.01 ms 이하 — **달성: 4.109 ms, 42.7% 개선** |
| M2 — graph-ready path | TPOT p50 2배 이상 개선 | 3.58 ms 이하 |
| M3 — release candidate | TPOT p50 2.0 ms 이하, E2E p50 72 ms 이하, 처리량 445 tok/s 이상 | vLLM gap 1.8배 이내 |
| Stretch | TPOT p50 1.4 ms 이하 | vLLM gap 약 1.25배 이내 |

각 PR의 기본 promotion gate는 다음과 같다.

- failure, dropped trace, token mismatch: 모두 0
- 사전 선언한 numeric correctness gate: 통과
- primary metric arm median과 AB/BA paired median: 모두 5% 이상 개선
- 복잡도가 큰 CUDA Graph 후보: paired TPOT 15% 이상 개선
- TTFT p95 candidate/current ratio: 1.05 이하
- 비대상 필수 workload의 TPOT/E2E p95 ratio: 1.05 이하
- c8 throughput candidate/current ratio: 0.95 이상
- peak VRAM 증가: 5% 이하이며 usable KV block capacity 감소 없음
- hot-loop host/device allocation 증가: 0
- owner close 후 live allocation: 0

5% 미만인 enabling change는 단독 production default로 승격하지 않는다. 다만 후속
CUDA Graph의 필수 선행 변경이라면 opt-in 상태로 유지하고, 합쳐진 candidate가 10%
이상 개선되는지 별도 판정한다.

### 3.3 2026-08-27 M1 실행 결과

동일 clean commit, model/tokenizer, GPU UUID, container image와 `c1/p128/o32/greedy`
workload에서 `fixed-sync-cpu`와 `bucket-packed-gpu`를 B,C,C,B,B,C,C,B,B,C 순서로
각 5회 독립 실행했다. 최종 candidate는 active-row/packed metadata/GPU greedy에 더해,
SmolLM2의 GQA `QH=9, KVH=3`에서 KV tile을 query-head warps가 한 번만 읽는 cooperative
attention을 사용한다.

| 지표 | baseline median | candidate median | 변화 |
|---|---:|---:|---:|
| TTFT p50 | 5.450 ms | 4.033 ms | 26.0% 개선 |
| TPOT p50 | 7.166 ms | 4.109 ms | **42.7% 개선** |
| E2E p50 | 227.595 ms | 131.404 ms | 42.3% 개선 |
| 처리량 | 140.535 tok/s | 243.536 tok/s | 73.3% 증가 |
| failure / dropped trace | 0 / 0 | 0 / 0 | 통과 |

checker의 TPOT p95 ratio는 `0.5724`, TTFT p95 ratio는 `0.7368`이며 primary host execute
paired median 개선은 `41.2%`다. raw run과 strict report는
[`cooperative-kv-profile`](../benchmarks/results/20260827T101334Z-riley-decode-fastpath-pr16/cooperative-kv-profile/)에 append-only로 보관한다.
이 결과는 M1을 충족하지만, vLLM과의 동일 campaign 비교 및 M2/M3 graph·장문맥·c8 gate를
대체하지 않는다.

## 4. 범위와 비범위

### 4.1 포함 범위

- active token 수 기반 dense-row shape bucket
- shared weights/KV를 유지하는 multi-shape GEMM plan
- packed pinned metadata와 asynchronous H2D
- exact GPU greedy top-1과 token-only D2H
- Python-free CUDA Graph capture/replay
- profiler가 다시 선택한 QKV, gate/up, elementwise fusion
- 동일 benchmark contract와 append-only performance evidence

### 4.2 비범위

- quantization, speculative decoding, KV pruning 등 A1/E1 최적화
- 모델 구조, checkpoint 의미 또는 학습 방식 변경
- Python, PyTorch, Triton runtime dependency의 production 추가
- TPOT 개선과 무관한 API/UX 변경
- short-context 증거만으로 long-context 또는 대형 모델 일반 우위 주장
- 효과 측정 전 FlashAttention 계열 backend를 무조건 도입
- 사용자 작업이 있는 현재 `riley-model` 변경의 수정·정리·commit

## 5. 목표 실행 구조

현재 경로:

    Scheduler iteration
      → 실제 token T개를 M=512로 padding
      → token + metadata 개별 sync H2D
      → 517개 eager kernel submit
      → [output rows, vocabulary] 전체 logits sync D2H
      → CPU greedy sampling
      → scheduler commit / stream publish

목표 경로:

    Scheduler iteration
      → immutable mixed plan과 실제 active rows T 확정
      → 최소 shape bucket M 선택
      → packed metadata 1회 async H2D
      → pure-decode 지원 bucket이면 CUDA Graph replay
         └─ bucketed dense graph + GPU greedy
      → token/status record만 D2H
      → scheduler commit / stream publish
      → unsupported shape·sampling이면 exact eager/CPU fallback

### 5.1 핵심 설계 결정: full executor 복제 금지

bucket마다 `PreparedLlamaBatchExecutor` 전체를 만들면 uploaded weights와 KV cache가
중복될 수 있다. 따라서 executor는 계속 단일 owner로 유지한다.

권장 내부 구조:

    PreparedLlamaBatchExecutor
      ├─ shared uploaded weights
      ├─ shared paged KV cache
      ├─ shared RoPE tables
      ├─ max-capacity forward buffers/workspace
      ├─ max-capacity metadata buffers
      └─ shape variants
           ├─ M=1 execution/GEMM plans
           ├─ M=2 execution/GEMM plans
           ├─ M=4 execution/GEMM plans
           └─ ... M=512

각 shape variant는 plan과 GEMM descriptor만 소유한다. weights, KV, scratch의 실체는
공유한다. GEMM workspace는 모든 bucket 요구량의 최댓값 하나만 할당한다.

### 5.2 shape bucket 정책

기본 후보:

    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

512는 현재 기본 `batch-token-budget`일 때의 마지막 bucket이다. 운영 설정이 다르면
bucket은 1부터 시작해 strictly increasing이어야 하고, 마지막 값은 configured
iteration token budget과 정확히 같아야 한다.

선택 규칙:

    T = packed.total_input_tokens()
    M = T 이상인 가장 작은 bucket

예:

| iteration | 실제 T | 선택 M |
|---|---:|---:|
| c1 decode | 1 | 1 |
| c8 decode | 8 | 8 |
| prefill 128 | 128 | 128 |
| decode 1 + prefill 128 | 129 | 256 |
| 257~512 rows | 257~512 | 512 |

Scheduler plan을 prefill과 decode로 분리하지 않는다. 현재 mixed iteration의 row 순서,
output slot, KV reservation과 commit 원자성을 보존하고 runtime이 flattened T만 보고
shape를 고른다.

## 6. 단계별 구현 계획

### Phase 0 — clean baseline과 attribution 고정

**목적:** 코드 변경 전에 7.16 ms의 critical path를 재현하고 각 비용을 닫는다.

#### 작업

1. clean candidate 시작 commit을 고정한다.
2. current Riley와 vLLM 0.27.1을 같은 campaign에서 AB/BA 교차 실행한다.
3. vLLM을 optimized default와 eager/no-graph 두 mode로 측정한다.
4. Riley iteration에 다음 NVTX 또는 동등 timing span을 추가한다.

   - scheduler plan
   - metadata pack
   - H2D enqueue와 wait
   - eager submit 또는 graph launch
   - GPU completion
   - logits/token D2H
   - sampling
   - commit
   - client-visible token

5. Nsight Systems로 CUDA API, launch gap, memcpy, synchronization을 수집한다.
6. NCU는 output=2 sentinel의 두 번째 decode iteration만 사용해 kernel attribution을
   수집한다.
7. 다음 accounting을 p50 기준 ±5% 이내로 닫는다.

       TPOT ≈ GPU kernel work
            + host launch/gap
            + H2D/D2H와 synchronization
            + sampling
            + scheduler/commit

#### 변경 후보

- [riley-profile](../crates/riley-server/src/bin/riley-profile.rs)
- [server benchmark adapter](../crates/riley-server/src/benchmark.rs)
- [scheduler metrics](../crates/riley-scheduler/src/metrics.rs)
- benchmark schema/checker와 append-only result writer

#### 완료 조건

- 동일 조건 5 independent process × 30 measured 요청에서 격차 재현
- failure 및 dropped trace 0
- Riley current의 selected M=512, kernel count=517를 receipt로 확인
- vLLM default의 compile/CUDA Graph 설정을 stdout receipt로 확인
- performance 변경 없이 current baseline artifact 생성

#### 중단 조건

같은 조건에서 TPOT 격차가 재현되지 않거나 provenance가 다르면 최적화를 시작하지 않고
benchmark contract부터 수정한다.

### Phase 1 — active-row shape bucket

**목적:** decode 1토큰이 M=512 GEMM을 실행하는 구조를 제거한다.

**의미 보존 등급:** reference systems optimization. Active row의 수학적 의미는
동일하며 inactive padding 계산만 제거한다.

#### 1A. 관측과 롤백 설정

다음 runtime 설정을 추가한다.

    --batch-shape-policy fixed-max
    --batch-shape-policy power-of-two
    --batch-shape-buckets 1,2,4,8,16,32,64,128,256,512

초기 기본값은 `fixed-max`다. iteration metric에 다음을 기록한다.

- active_rows
- selected_dense_rows
- padding_rows
- bucket hit count
- bucket별 execution latency

이 단계에서는 여전히 M=512만 실행하여 관측 추가가 성능과 결과를 바꾸지 않는지
확인한다.

#### 1B. 단일-shape ownership refactor

`execute_fixed_graph`가 `forward.plan`과 `forward.gemms`를 암묵적으로 참조하지
않고 다음을 명시적으로 받도록 분리한다.

- selected execution plan
- selected GEMM plans
- shared uploaded weights
- shared forward buffers
- shared KV/RoPE

먼저 shape list를 `[512]`로만 두고 기존 결과가 가능한 범위에서 byte exact인지
확인한다. close와 prepare failure 경로에서 shape plan은 각각 한 번, shared
weights/KV/buffers는 정확히 한 번 정리되어야 한다.

#### 1C. power-of-two bucket 활성화

1. cold prepare에서 각 M의 execution/GEMM plan을 만든다.
2. 모든 bucket은 동일한 weight owner와 reduction profile을 사용한다.
3. scratch buffer는 max-M 크기로 유지하고 selected M의 prefix span만 kernel에 넘긴다.
4. `padded_tokens[..M]`만 초기화하고 `[..T]`에 실제 token을 복사한다.
5. token H2D도 M개만 수행한다.
6. output gather의 input row count도 max 512가 아니라 selected M을 사용한다.
7. bucket 선택은 첫 H2D 및 KV mutation 전에 끝낸다.
8. post-dispatch 오류에서 다른 bucket/fixed-max로 자동 재시도하지 않는다.

#### 관련 소스

- [batch executor ownership/execution](../crates/riley-runtime/src/llama/batch_executor.rs)
- [forward plans와 GEMM plans](../crates/riley-runtime/src/llama/forward.rs)
- [Llama execution plan](../crates/riley-runtime/src/llama/plan.rs)
- [packed mixed-batch metadata](../crates/riley-runtime/src/llama/batch.rs)
- [scheduler iteration adapter](../crates/riley-scheduler/src/execution.rs)
- [server CLI/config wiring](../crates/riley-server/src/main.rs)

#### 필수 invariant

1. `T == sum(row input token counts) == packed.total_input_tokens()`
2. selected M은 항상 T 이상인 최소 bucket이다.
3. `[0,T)`만 active이고 `[T,M)`만 zero-padding이다.
4. `[M,max_M)`의 stale scratch는 어떤 kernel span에서도 참조하지 않는다.
5. work item 순서, request ID, block table과 output slot을 재정렬하지 않는다.
6. 모든 output token index는 T보다 작다.
7. padding row는 KV write 대상이 아니다.
8. 모든 shape plan은 같은 model/weights/RoPE/reduction profile을 사용한다.
9. paged KV cache는 executor 전체에 정확히 하나다.
10. post-dispatch CUDA 오류는 shared KV의 부분 변경 가능성 때문에 owner 전체를 poison한다.

#### 테스트

- bucket 선택: T={1,2,3,8,9,127,128,129,511,512}
- invalid bucket: 0, duplicate, unsorted, max bucket 누락
- decode-only: T={1,2,8,17}
- prefill-only: T={1,127,128,129,512}
- mixed: decode 1 + prefill {1,127,128}
- KV block boundary: 15→16, 16→17
- shape history: 128→1→8→256→1
- fixed-max와 bucket mode의 logits tolerance 및 generated token 비교
- output slot, KV logical length와 written block hash 비교
- fault injection 후 poison/abort/close/reuse 계약

#### 성능 gate

- primary TPOT p50와 throughput 30% 이상 개선
- c1/p128/o32에서 prefill M=128, decode M=1을 trace로 확인
- c8/p128/o32에서 decode M=8을 확인
- TTFT p50/p95 5% 초과 악화 없음
- weight 및 KV device bytes 중복 0
- bucket plan/workspace VRAM 증가가 사전 예산 이내

#### 롤백

재시작 시 `--batch-shape-policy fixed-max`를 사용한다. 실행 도중 실패한 iteration을
fixed-max로 다시 실행하는 것은 KV double-write 위험 때문에 금지한다.

### Phase 2 — packed asynchronous metadata H2D

**목적:** token과 metadata의 여러 작은 synchronous copy를 한 번의 stream-ordered
copy로 바꾸고 CUDA Graph가 읽을 안정적인 device address를 만든다.

#### 작업

1. token IDs와 metadata 6종을 versioned/aligned host slab에 pack한다.
2. cold prepare에서 pinned host slab과 device slab을 한 번 할당한다.
3. host validation과 offset/length 검증을 모두 끝낸 뒤 H2D 1회를 enqueue한다.
4. copy 직후 synchronize하지 않고 같은 stream ordering으로 model graph가 데이터를
   보게 한다.
5. DMA 완료 전 host slab이 덮어써지지 않도록 resource lease를 iteration completion
   ledger에 유지한다.
6. 1차 구현은 single slab으로 제한한다. 효과가 증명된 뒤에만 double buffering으로
   iteration N 실행과 N+1 metadata preparation을 겹친다.
7. device pointer와 layout version은 graph key에 포함한다.

#### 위험과 검증

- DMA 완료 전 pinned slab reuse
- validation failure 뒤 stale metadata 사용
- alignment, padding, endianness와 ABI version 불일치
- partial upload 뒤 graph replay
- double-buffer slot과 graph pointer 불일치

#### 성능 gate

- pre-model H2D call: token+6회에서 1회로 감소
- dispatch 전 copy-related stream synchronize: 0
- H2D bytes: 실제 payload와 alignment padding 외 5% 이상 증가 금지
- hot-path allocation: 0
- 단독 paired TPOT 또는 metadata/upload wall 5% 이상 개선

단독 개선이 5% 미만이면 production default로 승격하지 않고 graph enabling path로만
유지한다.

#### 롤백

    --metadata-transport synchronous
    --metadata-transport packed-async

partial failure와 lifetime을 fail closed로 증명하지 못하면 synchronous path를 유지한다.

### Phase 3 — exact GPU greedy와 token-only D2H

**목적:** 기본 greedy 요청에서 매 토큰 98,304-byte full logits D2H와 CPU vocab scan을
제거한다.

#### 최초 fast-path eligibility

다음 조건을 모두 만족할 때만 GPU greedy를 사용한다.

- temperature=0
- allowed mask가 없는 `AllowAll`
- repetition penalty=1
- top-k/top-p 등 stochastic filtering 불필요
- API가 full logits 또는 별도 logprobs를 요구하지 않음
- output row 수가 prepared bound 이내

그 외 요청은 기존 CPU sampler와 full-logits download로 fallback한다.

#### 구현 계약

1. BF16 logits를 GPU에서 검사하고 argmax한다.
2. 동점은 현재 CPU contract처럼 작은 token ID를 선택한다.
3. NaN/Inf, all-masked, invalid metadata를 status bit로 반환한다.
4. greedy path는 RNG를 한 번도 소비하지 않는다.
5. host에는 row별 `{token_id,status}` record만 가져온다.
6. 다음 decode가 같은 token을 다시 필요로 하므로 device token buffer 연결 지점을
   별도로 제공한다.
7. stop/cancel/detokenize와 scheduler commit은 기존 host 경계를 유지한다.
8. stochastic sampling 이동은 별도 후속 계획으로 남긴다.

#### 테스트

- 전체 31-case native correctness와 multi-step golden token
- 동률 최고값에서 최소 token ID
- 첫/마지막 vocab index
- 음수/양수 BF16 극값
- NaN/Inf error
- all-masked error
- repetition penalty와 mask가 있는 요청의 CPU fallback
- greedy RNG draw=0
- c1/c8 output slot 순서
- stop token, EOS ignore, cancellation 직전 commit

#### 성능 gate

- D2H: 98,304×O bytes에서 최대 16×O bytes
- hot-path host logits Vec allocation: 0
- CPU sampling time: 90% 이상 감소
- token/status mismatch: 0
- primary paired TPOT 5% 이상 개선

#### 롤백

    --sampling-backend cpu
    --sampling-backend gpu-greedy

GPU argmax+completion이 full-row D2H+CPU greedy보다 5% 이상 빠르지 않거나 semantic
contract를 exact하게 닫지 못하면 CPU path를 기본값으로 유지한다.

### Phase 4 — CUDA Graph 기반 decode fast path

CUDA Graph는 517개의 GPU node를 자동으로 하나의 kernel로 합치지 않는다. 이 단계의
목적은 host가 517개 kernel을 매 토큰 다시 제출하는 대신, 준비된 graph를
`cudaGraphLaunch` 한 번으로 replay하게 하는 것이다. 실제 GPU node 수 감소는
Phase 5의 fusion에서 수행한다.

#### 4A. low-level CUDA Graph lifecycle

먼저 [riley-cuda](../crates/riley-cuda/src)와 native C ABI에 다음 최소 기능을 추가한다.

- stream capture begin/end
- graph instantiate
- graph exec launch
- graph/graph-exec destroy
- capture invalidation과 error mapping
- stable device allocation/address 검증
- Rust RAII ownership과 explicit close
- capture 중 금지 API의 fail-closed 검사

독립 GPU 테스트에서 작은 fixed kernel chain을 capture/replay하고 다음을 검증한다.

- eager와 replay 결과 동일
- replay N회 후 allocation 증가 0
- capture failure 뒤 stream 재사용
- graph close 뒤 live allocation 0
- context/stream close 순서와 error propagation

이 PR에서는 Llama executor를 graph로 바꾸지 않는다.

#### 4B. pure-decode graph integration

초기 지원 범위:

- BF16
- pure decode iteration
- active decode bucket 1, 2, 4, 8
- production reduction/residual profile
- GPU greedy fast path
- stable paged-KV와 metadata device addresses

prefill, mixed prefill+decode, unsupported sampling과 unsupported shape는 bucketed eager
path를 사용한다.

graph key에는 최소한 다음을 포함한다.

- model architecture/dimensions와 dtype
- selected M/output capacity
- reduction profile
- residual implementation
- sampling backend
- metadata slab slot/layout version
- KV layout version와 maximum context
- kernel ABI/build identity

실행 순서:

1. request와 scheduler plan을 host에서 모두 검증한다.
2. graph/eager 선택을 KV mutation 전에 확정한다.
3. packed metadata H2D를 같은 stream에 async enqueue한다.
4. 준비된 graph를 replay한다.
5. GPU greedy 결과 record를 D2H한다.
6. completion 확인 후에만 scheduler reservation을 commit한다.

graph capture는 startup/warmup 단계에서 수행하고 measured request 안에서는 하지 않는다.
readiness는 필수 graph가 준비됐거나 documented eager fallback이 준비된 뒤에만 true가
된다.

#### lifecycle 규칙

- replay 중 cancellation은 다음 안전 경계에서 처리한다.
- partial KV write 뒤 graph 오류가 나면 eager retry하지 않고 owner를 poison한다.
- stale graph pointer 또는 layout version mismatch는 dispatch 전 거부한다.
- graph cache eviction을 첫 구현에 넣지 않는다.
- bucket explosion을 막기 위해 1/2/4/8만 먼저 지원한다.
- graph memory reservation과 usable KV capacity를 allocation report에 노출한다.

#### 성능 gate

- warm steady state에서 graph hit rate 95% 이상
- measured request 중 capture 0회
- iteration당 수백 eager submissions가 graph launch 1회로 감소
- host submit+completion wall 80% 이상 감소
- primary TPOT paired median 15% 이상 개선
- throughput 10% 이상 개선
- graph memory로 usable KV capacity가 5% 넘게 감소하지 않음
- eager와 graph token/logits/lifecycle parity

10~15% 개선은 experimental로 유지하고, 10% 미만이면 production default 승격을
중단한다. host submit이 80% 줄었는데 TPOT가 10% 미만 개선되면 GPU kernel 자체가
다음 병목이라는 뜻이므로 bucket 확대보다 Phase 5 profiling으로 이동한다.

#### 롤백

    --decode-execution eager
    --decode-execution cuda-graph

### Phase 5 — profiling 기반 kernel 수와 GPU 시간 축소

Phase 1~4가 끝난 새 trace에서 GPU category별 시간을 다시 측정한다. 기존 517 inventory를
그대로 근거로 모든 fusion을 한 PR에 넣지 않는다.

#### 5A. QKV packed projection

현재 layer당 Q/K/V GEMM 3개를 packed projection 1개로 바꾼다.

작업:

- checkpoint 원본과 provenance는 유지
- device preparation에서 Q/K/V packed weight/bias를 생성
- output layout을 Q/K/V view offset으로 노출
- 기존 separate projection을 exact fallback으로 유지
- packed derived artifact와 source weight hash를 기록

예상 node 감소는 30 layer에서 GEMM 90→30, 즉 60개다. 실제 count와 시간은 NCU로
확인한다.

위험:

- Q와 KV width가 다른 layout의 offset 오류
- cuBLASLt algorithm 변경에 따른 reduction rounding
- packed weight 중복 상주
- SmolLM 전용 가정을 Qwen에 잘못 적용

gate:

- source weight provenance 유지
- original device buffers 정리 후 상주 weight overhead 1% 이하
- raw-logit numeric gate와 multi-step token gate 통과
- QKV kernel duration과 intermediate DRAM write 감소
- 단독 paired TPOT 5% 이상 개선

#### 5B. gate/up packed projection

gate/up weight를 pack해 GEMM 2개를 1개로 만든다. 이 변경만 독립적으로 측정한다.

gate:

- layer당 gate/up GEMM 2→1
- layout/alias/workspace 테스트
- numeric/token parity
- 단독 paired TPOT 5% 이상 개선

#### 5C. SiLU × gate fusion

SiLU와 gated multiply 두 elementwise kernel을 하나로 합친다.

gate:

- register pressure, occupancy, DRAM bytes 측정
- extreme BF16 입력과 numeric parity
- 단독 paired TPOT 5% 이상 개선

#### 5D. 후순위 후보

새 profile이 근거를 제공할 때만 다음을 각각 별도 PR로 검토한다.

- RoPE + paged KV write
- residual + RMSNorm 재검토
- LM-head + top-1 연계
- output gather 제거

기존 residual+RMSNorm fusion은 paired 개선 약 1.61%로 promotion gate에 미달했으므로
단독 우선순위가 아니다.

독립 rollback:

    --qkv-projection separate|packed
    --gate-up-projection separate|packed
    --activation-gate separate|fused

### Phase 6 — long-context와 high-concurrency 후속 최적화

c1/p128/o32 목표를 달성한 뒤 다음 workload에서 병목을 다시 선택한다.

- c1/p4096/o128: long prefill/decode
- c8/p128/o32: scheduler와 batch throughput
- c4/p1024/o128: balanced workload
- mixed prefill/decode steady state

attention이 실제 top bottleneck일 때만 canonical ragged attention과 fixed37 또는
Flash-style backend를 비교한다. paged KV라는 개념 자체를 원인으로 간주하지 않는다.

CPU/GPU overlap과 double-buffer metadata도 Nsight Systems에서 GPU idle gap이 남을 때만
추가한다.

## 7. 공정한 benchmark 계획

### 7.1 두 비교 트랙

1. **Production-vs-production**

   Riley production candidate와 vLLM optimized default를 비교한다. 이 결과만 사용자
   대상 경쟁 성능 표에 사용한다.

2. **Attribution-only**

   다음처럼 한 축씩만 바꾸어 원인을 분리한다.

   - Riley fixed-max eager vs bucketed eager
   - CPU sampling vs GPU greedy
   - synchronous metadata vs packed async
   - Riley eager vs CUDA Graph
   - vLLM eager vs CUDA Graph default

   이 결과는 원인 분석용이며 production 우위 주장에 사용하지 않는다.

engine-core와 HTTP serving 결과도 별도 표와 timing boundary를 사용한다.

### 7.2 필수 decision cells

| 목적 | 상태 | concurrency | prompt | output | primary |
|---|---|---:|---:|---:|---|
| decode overhead | warm | 1 | 128 | 32 | TPOT p50/p95 |
| medium prompt | warm | 1 | 1024 | 32 | TTFT + TPOT |
| long context | warm | 1 | 4096 | 128 | E2E + TTFT |
| scheduler throughput | warm | 8 | 128 | 32 | aggregate tok/s + tail ITL |
| balanced | warm | 4 | 1024 | 128 | throughput + E2E p95 |
| startup | cold | 1 | 128 | 32 | load/readiness + first TTFT |

최적화 merge는 이 필수 cell로 판정한다. 전체 Riley-vLLM parity 또는 일반 우위 주장은
[benchmark matrix](../benchmarks/matrix.yaml)의 전체 predeclared cell을 실행한 뒤에만
허용한다.

### 7.3 고정 조건

- 동일 RTX 4090와 GPU UUID
- 동일 driver, CUDA runtime/toolkit, power/clock 정책
- 같은 SmolLM2 revision, weights/tokenizer hash
- BF16, TP=1
- exact pretokenized prompt IDs
- greedy, fixed output, ignore EOS
- prefix cache disabled
- warm: process별 warmup 5 + measured 30
- 최소 5 independent process, AB/BA 순서 교차
- 최종 경쟁 campaign은 10 process pair 권장
- thermal preflight와 다른 CUDA process 부재 확인

### 7.4 통계 계약

- quantile: 기존 checker와 같은 R7
- 먼저 run별 p50/p95를 계산한 뒤 run-level 통계를 계산
- pooled 150 rows를 독립 표본처럼 취급하지 않음
- CV: run medians/p95 사이의 sample standard deviation / absolute mean
- current/candidate ratio는 pair 단위로 계산
- fixed seed 10,000-resample bootstrap 95% CI
- primary paired ratio의 CI upper bound는 1.0 미만
- outlier 임의 삭제 금지
- preflight/provenance drift는 failed가 아니라 incomparable
- E2E와 throughput을 TPOT와 독립적인 세 가지 승리로 중복 계산하지 않음

### 7.5 수집 지표

Product:

- TTFT p50/p95/p99
- request mean TPOT p50/p95/p99
- pooled ITL distribution
- E2E p50/p95/p99
- output tok/s
- failure와 token count

Host/CUDA:

- scheduler plan/commit ns
- metadata pack/H2D ns
- `cudaLaunchKernel`, `cudaGraphLaunch` call/time
- memcpy calls/bytes
- stream/event synchronize count/time
- D2H bytes/token
- CPU sampling ns

GPU/memory:

- kernel/node inventory
- category별 GPU duration
- launch gap
- DRAM bytes와 occupancy
- graph reservation
- pinned/device slab bytes
- peak VRAM
- usable KV block capacity
- startup/capture 시간

## 8. correctness 및 lifecycle 검증

### 8.1 공통 correctness

- 기존 native E0 correctness gate 전 case 통과
- canonical current와 candidate generated IDs exact
- shape 변경은 사전 고정 tolerance 내 raw logits parity
- GPU greedy는 CPU greedy와 token/status exact
- output row 수와 slot mapping exact
- KV logical length, block ownership과 content checksum 일치
- NaN/Inf와 invalid metadata fail closed
- hot-loop allocation 0
- close 후 allocation 0

### 8.2 요청 lifecycle

- pure prefill, pure decode, mixed iteration
- cancellation before dispatch
- cancellation during replay
- client disconnect
- stop token/EOS/ignore EOS
- overload와 waiting queue
- fault injection before KV mutation
- fault injection after partial KV mutation
- poison된 owner의 재사용 거부
- graph capture failure 뒤 eager path 재사용
- shutdown 중 graph/stream/context close

### 8.3 모델 범위

최초 성능 gate는 SmolLM2-135M으로 수행한다. production default 승격 전에는 최소한
다음 architecture boundary를 확인한다.

- SmolLM2/Llama head geometry
- Qwen2.5의 QKV/rope/model boundary
- context 1, 128, 1024, 4096, 8191
- concurrency 1, 2, 4, 8

지원하지 않는 architecture/shape는 silent fallback이 아니라 capability 판정 후
documented exact eager path를 사용한다.

## 9. artifact와 release 증거

모든 결과는 [benchmark artifact 계약](../benchmarks/README.md)에 따라 append-only로
저장한다.

권장 campaign 구조:

    benchmarks/results/<UTC>-riley-vllm-decode-fair-v1-<nonce>/
      README.md
      comparison-contract.json
      campaign-manifest.json
      statistics-report.json
      correctness/
      lanes/
        riley-current/
        riley-candidate/
        vllm-default/
        vllm-eager/
      profiling/
        micro/
        ncu/
        nsys/
      external-artifacts.json
      SHA256SUMS

각 raw row와 manifest는 다음을 bind한다.

- campaign/run/pair/cell/trial ID
- timing-boundary version
- source commit와 dirty=false
- source archive, executable, native library, image hash
- model/weights/tokenizer/prompt token hash
- GPU/driver/CUDA/clock/thermal receipt
- runtime flags와 graph/sampling/shape policy
- generated token hash
- cache fingerprint 전후

대형 NCU/NSYS trace, binaries와 images는 immutable external storage의 URI, byte size,
SHA-256과 retention을 기록한다. 작은 report와 raw JSONL/CSV만 repository에 보존한다.

report status는 다음으로 닫는다.

- `passed`: 모든 correctness/performance gate 통과
- `failed`: comparable한 측정이 threshold에 미달
- `incomparable`: workload/environment/provenance drift
- `error`: schema, digest, missing evidence 또는 실행 오류

기존 `performance-baseline-v1.json`은 수정하지 않는다. 최종 candidate가 통과하면
predecessor hash를 가진 `performance-baseline-v2.json`을 새로 추가한다. Riley 자기
회귀 baseline과 Riley-vLLM competitive baseline은 별도 artifact로 유지한다.

## 10. 단계 의존성과 예상 복잡도

| 단계 | 의존성 | 복잡도 | 기본값 승격 조건 |
|---|---|---|---|
| Phase 0 baseline | 없음 | M | attribution과 comparable baseline 완료 |
| Phase 1 shape bucket | Phase 0 | L | TPOT 30% 개선, 정확성/VRAM gate |
| Phase 2 async metadata | Phase 0, Phase 1 interface | M | 단독 5% 또는 graph combined 10% |
| Phase 3 GPU greedy | Phase 0 | M | exact token/status, TPOT 5% |
| Phase 4A graph runtime | Phase 0 | L | lifecycle/RAII GPU test |
| Phase 4B decode graph | Phase 1~4A | L | hit 95%, TPOT 15%, safe fallback |
| Phase 5 fusion | Phase 4 이후 재profile | M~L/PR | 각 candidate 단독 5% |
| Phase 6 long/c8 | M3 달성 후 | L | workload별 독립 gate |

Phase 2와 Phase 3은 Phase 1 이후 병렬 개발할 수 있다. Phase 4B는 안정적인 shape,
device pointer와 capture-safe output 경계가 필요하므로 Phase 1~4A가 모두 완료된 뒤
통합한다.

## 11. 위험 관리

| 위험 | 영향 | 예방/대응 |
|---|---|---|
| bucket별 full executor 생성 | weight/KV 중복, VRAM 급증 | single owner + plan/GEMM variant만 추가 |
| stale scratch 참조 | 잘못된 logits/KV 오염 | prefix span invariant와 shape-history test |
| graph stale pointer | crash 또는 silent corruption | stable allocation, layout/version graph key |
| graph 실패 후 eager retry | KV double-write | post-dispatch retry 금지, owner poison |
| GPU argmax semantic drift | token 불일치 | tie/NaN/mask/penalty fixture와 CPU fallback |
| async slab lifetime 오류 | stale metadata 또는 UAF | pinned owner lease와 completion ledger |
| graph bucket 폭증 | startup/VRAM 증가 | decode 1/2/4/8부터 시작, eviction 미도입 |
| fusion weight 중복 | VRAM/startup 회귀 | derived packed owner와 original device buffer release |
| TTFT 희생 | 첫 토큰 UX 회귀 | 모든 PR에서 TTFT p95 ≤1.05 gate |
| c1만 최적화 | c8/mixed 회귀 | 필수 c8/balanced/mixed regression cells |
| benchmark 계보 혼합 | 잘못된 우위 주장 | 동일 campaign, clean commit, digest binding |

## 12. PR별 공통 제출 형식

각 PR은 [PR 작업·리뷰 계약](../deploy/00-pr-contract.md)에 따라 다음을 포함한다.

1. 문제와 한 가지 primary 질문
2. 범위와 명시적 비범위
3. reference 또는 E0 의미 보존 등급
4. profiler evidence와 가설
5. reference/rollback 구현
6. optimized implementation
7. correctness 및 lifecycle 결과
8. microbenchmark
9. profiler attribution
10. paired end-to-end 결과
11. memory/runtime dependency 변화
12. append-only artifact와 hash
13. runtime flag와 rollback 절차

한 PR에서 다음을 함께 성공으로 묶지 않는다.

- shape bucket + CUDA Graph
- GPU sampling + stochastic sampling
- QKV packing + gate/up packing
- 여러 elementwise fusion
- short-context attention + long-context attention

## 13. 최종 완료 정의

다음이 모두 충족돼야 decode 성능 개선 프로그램을 완료로 본다.

- [ ] clean current baseline과 vLLM default/eager baseline이 같은 campaign에 존재
- [ ] 7.16 ms critical-path attribution이 ±5% 이내로 닫힘
- [ ] c1 decode가 M=1, c8 decode가 M=8을 사용
- [ ] full executor/weight/KV duplicate allocation 없음
- [ ] greedy D2H가 row당 최대 16 bytes
- [ ] steady-state decode graph hit rate 95% 이상
- [ ] measured request 안에서 graph capture 0
- [ ] correctness, generated token, KV와 lifecycle gate 모두 통과
- [ ] failure와 dropped trace 0
- [ ] TTFT와 c8/mixed regression gate 통과
- [ ] M3 성능 목표 또는 승인된 baseline-relative 목표 달성
- [ ] production HTTP E2E에서 동일 개선 확인
- [ ] Python-free build/runtime 유지
- [ ] eager/CPU/fixed-max rollback 실제 실행 검증
- [ ] append-only artifact와 checker replay 통과
- [ ] 새 baseline은 기존 baseline을 덮어쓰지 않고 version-up

## 14. 승인 후 첫 실행 순서

승인 후에도 곧바로 CUDA Graph부터 구현하지 않는다.

1. Phase 0 clean baseline과 Nsight Systems attribution
2. Phase 1A 관측/rollback 설정
3. Phase 1B single-shape ownership refactor
4. Phase 1C active-row bucket
5. Phase 2 async metadata와 Phase 3 GPU greedy를 독립 검증
6. Phase 4A CUDA Graph lifecycle
7. Phase 4B batch-1 decode graph integration
8. 새 profile을 근거로 Phase 5 candidate를 하나씩 선택
9. 최종 동일 revision에서 full matrix, HTTP, release evidence 재수집

각 단계의 gate를 통과하지 못하면 다음 단계로 넘어가지 않고, 해당 가설을
`Rejected`, `Deferred` 또는 `Blocked`로 기록한다.
