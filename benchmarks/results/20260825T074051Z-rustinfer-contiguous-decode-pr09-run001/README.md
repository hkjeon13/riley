# PR 09 contiguous single-request decode evidence — run001

`d5a30e5b863b606ffdd6bee6592dcb92cd88902b`의 contiguous/static KV cache와
단일 요청 decode는 RTX 4090/sm89에서 correctness, lifecycle, boundary, sanitizer와
profile 검사를 통과했다. 모든 CUDA·모델 실행은 `server-4096`의 network-disabled
container에서 수행했으며 로컬에서는 GPU나 모델을 실행하지 않았다.

## 결론

Prepared runtime은 layer별 BF16 K/V를 `[kv_head, max_seq, head_dim]`의 연속 storage에
사전 할당한다. Prefill은 prompt K/V를 쓰고, 각 decode는 다음 slot에 K/V를 append한 뒤
전체 layer가 성공했을 때만 logical length를 commit한다. Reset은 allocation을 재사용하고,
capacity 초과는 cache나 logical length를 바꾸기 전에 오류가 된다. 명시적 reset과 drop 뒤
allocation accounting이 원래 값으로 돌아오는 것도 검증했다.

Decode attention에는 4-kernel materialized reference와 D64 전용 2-kernel chunked-online
optimized 경로가 있다. Optimized 경로는 KV range마다 FP32 `(m,l,n)` partial state를 만들고
공통 reducer가 logical range 순서로 merge한 뒤 한 번만 normalize한다. 근사 attention이
아니지만 BF16 연산 순서가 reference와 달라 bit-exact라고 주장하지 않는다.

## Correctness

Direct GPU attention test 4개가 7개 shape matrix를 통과했다. CPU reference 대비
materialized output의 max-abs gate는 `0.03125`, optimized 대비 materialized gate는
`0.0625`였다. Logical length
33을 7-token range로 나눈 multi-range와 64-token one-range optimized output을 GPU에서 직접
비교한 결과 max abs는 `0.000000000`으로 `0.062500000` gate 안이었다.

Pinned SmolLM2-135M에서 cache 없는 full-forward prefix와 cache 기반 prefill+decode를 매
step 비교했다. 32-step 실행은 33개 row, 128-step 실행은 129개 row 모두 greedy top-1이
일치했고 prefill row는 byte-exact였다. Numeric 차이는 diagnostic으로 기록했다.

| 비교 | rows | top-1 mismatch | worst cosine | max abs | max mean abs |
|---|---:|---:|---:|---:|---:|
| cache vs full, 32 decode | 33 | 0 | 0.997812344493 | 0.593750000 | 0.280234733 |
| cache vs full, 128 decode | 129 | 0 | 0.997812375627 | 1.500000000 | 0.455652977 |
| optimized vs reference, 32 decode | 33 | 0 | 0.998950330082 | 0.507812500 | 0.261816807 |

이 값에는 PR 01의 FP32 comparator, relative metric, 31-prompt corpus를 다시 적용하지 않았다.
따라서 PR 01 E0 numeric gate 재인증이 아니라, 동일 shape prefill byte parity와 greedy sequence
semantic parity, 그리고 direct attention의 사전 고정 max-abs gate를 통과한 결과다.

## Latency와 launch 구조

표의 per-token 값은 준비된 model에서 logits download를 제외한 decode call wall time이며,
native primitive마다 stream synchronize가 포함된다. GPU event 구간에도 CPU가 primitive를
차례로 제출하는 동안의 GPU idle이 포함될 수 있다. `host outside event`는 event 바깥의 짧은
호스트 시간일 뿐 전체 CPU launch overhead가 아니다.

| 실행 | samples | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| optimized, cache/full parity run | 32 | 3.424913 ms | 3.681939 ms | 3.866996 ms |
| materialized reference, paired run | 32 | 3.309245 ms | 3.466826 ms | 3.486636 ms |
| optimized, paired run | 32 | 3.412714 ms | 3.569305 ms | 3.620120 ms |
| optimized, 128-step run | 128 | 4.202020 ms | 5.083333 ms | 5.175827 ms |
| optimized, 8,064→8,192 near-limit | 128 | 10.914296 ms | 11.003411 ms | 11.985214 ms |

같은 paired run에서 optimized p50은 reference보다 `3.1267%`, p95는 `2.9560%` 느렸다.
Clock 고정과 전용 warmup benchmark가 없는 기능 검증 실행이므로 backend 우열로 일반화하지
않는다.

Attention 자체는 optimized 2개, reference 4개 kernel로 관찰됐다. Source-level로 각
cuBLASLt call이 kernel 하나를 낸다고 조건부 계산하면 SmolLM2 30-layer token은 optimized
최소 546개, reference 최소 606개 launch다. cuBLASLt auxiliary kernel이 있을 수 있어 전체
호출의 profiler-wide 정확한 launch 수가 아니다. Nsight Systems가 설치되지 않아 CPU launch
overhead와 GPU idle gap을 별도로 분리하지 못했다.

## KV traffic과 partial state

Nsight Compute 2025.1.1로 logical length 8,065의 첫 layer optimized attention을 targeted
profile했다.

| kernel | grid | duration | measured DRAM read | registers/thread |
|---|---:|---:|---:|---:|
| partial-state producer | 64×9 | 121,184 ns | 6,203,520 B | 26 |
| reducer/normalizer | 1 | 244,448 ns | 181,760 B | 29 |
| 합계 | — | 365,632 ns | 6,385,280 B | — |

이 shape의 logical K/V payload는 `6,193,920 B`, active partial state는
`64 × 9 × (64 + 2) × 4 = 152,064 B`다. Producer logical read bandwidth는
`51.111698 GB/s`, 두 kernel duration 합 기준 logical bandwidth는 `16.940312 GB/s`,
measured aggregate DRAM bandwidth는 `17.463679 GB/s`였다. 최대 8,192-token model cache는
`188,743,680 B`다. Profiler replay duration은 위 raw latency와 직접 비교하지 않는다.

## Boundary와 lifecycle

- 1, 2, 32, 128 decode step과 direct logical lengths 1/2/31/32/33를 검사했다.
- 8,064-token prefill 뒤 128회 decode로 정확히 capacity 8,192에 도달했다.
- 다음 decode는 mutation 전 capacity error를 반환했고 allocation count/bytes는 변하지 않았다.
- reset 뒤 같은 prompt 결과는 byte-exact였고 다른 prompt의 fresh run과도 byte-exact였다.
- implicit drop 뒤 device allocation accounting은 0으로 복귀했다.
- Optimized workspace는 미리 할당되며 32-step에서 capacity 1, 128-step에서 capacity 2,
  near-limit에서 capacity 64의 partial state를 사용했다. Steady-state decode에 device
  allocation은 없었다.

Sampling 정책과 EOS stop 처리는 이번 PR의 범위가 아니다. Test harness는 greedy token을
선택해 다음 입력으로 사용했고, cache boundary와 logical commit 동작을 검증했다.

## 검증 환경과 명령

- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9, 24,564 MiB
- driver 580.173.02, CUDA runtime 12.8.1, nvcc 12.8.93
- Rust/Cargo 1.85.0, `RUSTINFER_CUDA_ARCHITECTURES=89`
- image ID: `sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`
- container network disabled; Python/Python3/Nsight Systems absent
- Nsight Compute 2025.1.1, Compute Sanitizer 2025.1

모든 Cargo command는 `--locked --offline`을 사용했고 source는 read-only, target과 사전
복사한 Cargo registry만 writable하게 mount했다. 주요 검증은 다음과 같다.

```text
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo test --release -p rustinfer-cuda --features cuda --test decode_attention_gpu -- --ignored --test-threads=1 --nocapture
cargo test --release -p rustinfer-runtime --features cuda --test llama_decode_gpu -- --ignored --test-threads=1 --nocapture
compute-sanitizer --tool memcheck ... decode_attention_gpu
compute-sanitizer --tool racecheck ... decode_attention_gpu
compute-sanitizer --tool memcheck ... llama_decode_gpu lifecycle
ncu --csv --metrics ... decode_attention_gpu
ncu --csv --metrics ... llama_decode_gpu near_limit
```

Workspace all-features 결과는 `111 passed, 0 failed, 40 ignored`였다. 실제 GPU 실행은 direct
attention 4개, model core 3개, 128-step 1개, near-limit 1개가 모두 통과했다. Memcheck 두
경로와 lifecycle memcheck는 `0 errors, 0 bytes leaked`, racecheck는 `0 hazards, 0 errors,
0 warnings`였다.

## Artifact와 provenance

Version-controlled [raw-events.jsonl](raw-events.jsonl)은 224개 decode latency sample,
128개 near-limit sample, correctness/lifecycle/metadata/profile event를 lossless decimal 또는
integer로 보존한다. [metadata.json](metadata.json)은 source, 환경, 구현 계약, 결과와 제한을
담는다. 원문 stdout, sanitizer, Nsight CSV와 source metadata는 다음 append-only 경로에 있다.

```text
server-4096:/home/psyche/rustinfer-artifacts/pr09/d5a30e5b863b606ffdd6bee6592dcb92cd88902b
SHA256SUMS sha256=ba73fab1901058a3271998508588ffa718660a208f717d102e1198dedc4244d5
source tar sha256=f3e3dee283c6ef9ee74b689d43b163cf7bbf58e0fd9f4f0803b588dc78c78fc8
```

최종 evidence root의 16개 payload checksum은 로컬로 복사한 뒤 다시 검증했다. 이전
`0df3167…` evidence에는 잘못된 test operator invocation 두 건이 보존돼 있지만 최종 root에는
성공한 명령과 결과만 있다. 그보다 앞선 개발 중 pairwise model numeric threshold를 적용한
실패를 통해 M=1 cuBLASLt FMA와 M=S tensor-core reduction이 같은 oracle가 아님을 확인했고,
최종 gate는 위 semantic/direct contract로 고정했다.

## 제한

- 단일 request/batch 1만 지원한다. Block pool, paging, 여러 요청, prefix sharing은 PR 10
  이후 범위다.
- D64 optimized kernel만 제공하며 다른 head dimension은 materialized reference capability를
  사용하거나 명시적으로 실패한다.
- Native primitive와 GEMM host call은 각각 동기식이다. CUDA Graph와 비동기 enqueue 최적화는
  아직 없다.
- 정확한 whole-call kernel launch 수와 CPU launch/GPU idle 분리는 기록하지 못했다.
- Cache/full numeric 값은 PR 01 E0 재인증 결과가 아니다.
- Allocation accounting은 runtime이 소유한 device buffer bytes이며 observed peak VRAM이 아니다.
- sm89와 기록된 driver/toolkit만 검증했다. 다른 architecture는 해당 AOT target으로 재빌드해
  같은 gate를 실행해야 한다.
