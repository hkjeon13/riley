# PR 09 — 연속 KV Cache 기반 단일 요청 Decode

**상태:** Implemented — `d5a30e5b863b606ffdd6bee6592dcb92cd88902b` 원격 GPU 검증 완료<br>
**선행 조건:** [PR 08](08-prefill-attention.md)  
**다음:** [PR 10 — Paged KV Manager](10-paged-kv-manager.md)

[← 이전](08-prefill-attention.md) | [목차](README.md) | [다음 →](10-paged-kv-manager.md)

## 목적

한 요청에 대해 prefill 후 token 하나씩 생성할 수 있는 contiguous/static KV cache와 decode attention을 구현한다. decode attention은 긴 KV 범위를 작은 chunk로 처리하고 PR 08의 `(m, l, n)` 부분합을 병합할 수 있는 exact 경로를 사용한다.

## 이번 단계의 cache

단순성을 위해 요청별 연속 cache를 사용한다.

```text
K: [layer][kv_head][max_seq][head_dim]
V: [layer][kv_head][max_seq][head_dim]
```

최대 길이는 요청 시작 시 고정한다. Paged allocation은 PR 10으로 미룬다.

구현은 K/V를 별도 BF16 device buffer로 사전 할당한다. `batch=1`은 public runtime
contract로 고정돼 cache layout에서 제거했다. Logical length는 모든 layer가 성공한 뒤에만
증가하며, capacity 오류는 K/V와 logical state를 변경하기 전에 반환한다.

## 필수 동작

- cache 사전 할당
- prefill K/V write
- decode position에 K/V append
- 현재 logical length 관리
- boundary 검사
- request reset/drop
- cache on/off parity

## Decode attention

- query length 1 우선
- GQA/MQA head mapping
- cached K/V read
- causal mask의 implicit 처리 가능
- output projection 전 layout 규약 고정
- KV range별 online softmax partial state
- 여러 range의 associative merge

reference decode path와 optimized path를 분리한다.

## `DecodePartialState` 계약

KV 범위 하나의 출력은 정규화된 attention output이 아니라 다음 부분합이다.

```rust
struct DecodePartialState {
    max_score: Accumulator,
    exp_sum: Accumulator,
    weighted_value_sum: VectorAccumulator,
}
```

merge는 PR 08의 online softmax 공식과 동일하다. 이 구조를 contiguous KV에서 먼저 검증하면 PR 10의 paged block과 이후 split-K decode에도 같은 reducer를 재사용할 수 있다.

필수 규칙:

- logical KV 순서와 physical chunk 순서를 구분
- empty chunk와 fully masked chunk 표현
- accumulator dtype 고정
- merge 결과를 마지막 한 번만 normalize
- 부분 output을 먼저 normalize한 뒤 평균내는 잘못된 구현 금지

## 구현 순서

1. 전체 contiguous KV를 한 번에 읽는 reference decode
2. 동일 KV를 고정 chunk로 나누는 partial-state reference
3. CPU 또는 단순 CUDA merge 검증
4. optimized decode backend 연결
5. chunk 수와 CTA partition을 runtime capability로 선택
6. one-pass와 split-range 결과 비교

## 구현 결과

- Public runtime에 `PreparedLlamaDecode`, decode config/report/phase/error와 cache layout
  metadata를 추가했다.
- Prefill K/V write와 single-token append는 low-level CUDA ABI로 분리했다.
- Reference attention은 BF16 QK, scale, softmax, AV의 materialized 4-kernel 경로다.
- Optimized D64 attention은 FP32 partial-state producer와 ordered reducer/normalizer의
  2-kernel 경로다. 근사 attention이 아니지만 연산 순서가 달라 reference와 bit-exact라고
  주장하지 않는다.
- Partial state storage는 `(max_score, exp_sum, weighted_value_sum[D])`이며
  `[range, query_head, D + 2]` FP32 contiguous layout을 사용한다. 이 reducer ABI는 PR 10의
  physical page range도 logical 순서의 descriptor로 공급할 수 있게 cache addressing과
  분리했다.
- Decode workspace, hidden scratch와 partial state는 prepare 시 할당한다. Steady-state
  decode에는 device allocation이 없다.
- `reset`은 allocation을 유지한 채 logical state를 초기화하고, `drop`은 runtime-owned
  allocation accounting을 복구한다.

## 검증

- full forward의 마지막 token logits와 prefill+decode logits 비교
- cache on/off greedy token 일치
- one-range와 multi-range decode 비교
- chunk boundary를 1, head-dependent 값, 임의 값으로 변경
- merge 순서 변경에 대한 dtype별 tolerance
- 1, 2, 32, 128 step decode
- max sequence boundary
- EOS 직전 cache length
- reset 후 재사용
- 서로 다른 prompt를 순차 실행했을 때 데이터 오염 없음

검증은 실행 가능한 source snapshot `d5a30e5…`를 read-only로 mount한
`server-4096`의 RTX 4090/sm89에서 수행했다. Container network는 비활성화했고 모든 Cargo
command는 `--locked --offline`이었다. 로컬에서는 GPU·model을 실행하지 않았다.

### Correctness 결과

- Direct CUDA test 4개가 7개 shape matrix를 통과했다. Materialized 대 CPU max-abs
  tolerance는 `0.03125`, optimized 대 materialized tolerance는 `0.0625`다.
- Logical length 33에서 optimized 7-token multi-range와 64-token one-range output을 GPU에서
  직접 비교했고 max abs `0.000000000`으로 일치했다.
- SmolLM2-135M cache/full 32-step의 33개 row와 128-step의 129개 row에서 greedy top-1
  mismatch는 0이고 prefill row는 byte-exact였다.
- 32-step reference/optimized paired run의 33개 row도 greedy top-1 mismatch가 0이었다.
- Cache/full worst diagnostic은 32-step cosine `0.997812344493`, max abs `0.59375`,
  mean abs `0.280234733`; 128-step은 cosine `0.997812375627`, max abs `1.5`, mean abs
  `0.455652977`였다.
- 위 model numeric 값은 M=1 decode와 M=S full-forward cuBLASLt reduction 차이를 포함한다.
  PR 01의 FP32/relative-error/31-prompt gate를 다시 실행하지 않았으므로 E0 numeric
  재인증으로 해석하지 않는다. 이 PR의 model gate는 same-shape prefill byte parity와 greedy
  semantic parity다.

### Boundary와 lifecycle 결과

- 1, 2, 32, 128 step과 logical length 1/2/31/32/33를 검사했다.
- 8,064-token prefill + 128 decode로 capacity 8,192에 도달한 뒤 다음 call이 mutation 전
  capacity error를 반환했다.
- Reset 뒤 같은 prompt replay와 다른 prompt fresh run은 각각 byte-exact였고 allocation은
  안정적이었다.
- Implicit drop 뒤 device allocation accounting은 0으로 복귀했다.
- Compute Sanitizer memcheck는 direct attention과 lifecycle 모두 `0 errors, 0 bytes leaked`,
  racecheck는 `0 hazards, 0 errors, 0 warnings`였다.
- Workspace all-features는 `111 passed, 0 failed, 40 ignored`, strict Clippy는 통과했다.

EOS stop/sampling policy는 비범위다. Test harness는 greedy token을 다음 입력으로 사용해
cache append와 최대 logical length까지의 안전성을 검증했다.

## 성능

- per-token latency
- kernel launch count
- K/V read bandwidth
- partial-state bytes와 merge 비용
- CPU launch overhead
- GPU idle gap
- allocation 없는 steady-state 여부

긴 context에서 decode는 계산량보다 KV memory traffic이 지배할 수 있으므로 FLOPs뿐 아니라 실제 읽은 KV bytes를 기록한다.

### 측정 결과

준비된 model의 logits download 제외 per-token wall time은 다음과 같다. Native primitive와
GEMM call이 각각 stream을 synchronize하므로 CUDA event 안에도 CPU-induced GPU idle이 포함될
수 있다. Nsight Systems가 없어 CPU launch overhead와 GPU idle gap은 분리하지 못했다.

| 실행 | samples | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| optimized cache/full run | 32 | 3.424913 ms | 3.681939 ms | 3.866996 ms |
| reference paired run | 32 | 3.309245 ms | 3.466826 ms | 3.486636 ms |
| optimized paired run | 32 | 3.412714 ms | 3.569305 ms | 3.620120 ms |
| optimized 128-step | 128 | 4.202020 ms | 5.083333 ms | 5.175827 ms |
| optimized 8,064→8,192 | 128 | 10.914296 ms | 11.003411 ms | 11.985214 ms |

Paired functional run에서 optimized p50/p95는 reference보다 각각 `3.1267%`/`2.9560%`
느렸다. Clock 고정과 전용 warmup benchmark가 없으므로 성능 우열로 일반화하지 않는다.

Attention kernel은 optimized 2개, reference 4개다. 각 cuBLASLt call이 kernel 하나를
launch한다고 가정한 source-level 조건부 최소치는 30-layer token당 optimized 546개,
reference 606개다. cuBLASLt auxiliary launch를 포함한 profiler-wide exact count가 아니다.

Logical length 8,065의 첫 layer Nsight Compute 결과는 producer DRAM read `6,203,520 B` /
`121,184 ns`, reducer read `181,760 B` / `244,448 ns`였다. Logical K/V는
`6,193,920 B`, active partial state는 `152,064 B`; producer logical bandwidth는
`51.111698 GB/s`였다. 최대 model cache는 `188,743,680 B`다.

원문과 checksum, parseable event는
[PR 09 run001 artifact](../benchmarks/results/20260825T074051Z-rustinfer-contiguous-decode-pr09-run001/README.md)에
보존한다. Remote append-only root는 다음과 같다.

```text
server-4096:/home/psyche/rustinfer-artifacts/pr09/d5a30e5b863b606ffdd6bee6592dcb92cd88902b
SHA256SUMS sha256=ba73fab1901058a3271998508588ffa718660a208f717d102e1198dedc4244d5
```

## 비범위

- 여러 요청
- block pool
- prefix sharing
- offload
- KV page pruning 또는 근사 attention
- sampling 정책
- API streaming

## 완료 기준

- [x] 단일 요청에서 prefill+N decode가 정확함
- [x] cache on/off greedy sequence 일치
- [x] one-range와 multi-range 부분합 결과가 허용 오차 내 일치
- [x] decode hot path에 device allocation 없음
- [x] 최대 길이 초과가 안전한 오류로 종료
- [x] request drop 후 VRAM accounting 복귀
- [x] partial-state ABI가 PR 10 paged block에서도 재사용 가능하게 문서화됨

[← 이전](08-prefill-attention.md) | [목차](README.md) | [다음 →](10-paged-kv-manager.md)
