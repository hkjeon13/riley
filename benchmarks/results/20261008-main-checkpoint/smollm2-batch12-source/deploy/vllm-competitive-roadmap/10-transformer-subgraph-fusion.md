# C10 — Profile-selected Transformer Subgraph Fusion

**상태:** Planned  
**의미 등급:** `E0`  
**한 가지 목적:** C07~C09 이후 profiler가 선택한 반복 transformer subgraph 하나를 fuse하여 end-to-end 병목을 줄인다.

[이전: C09](09-packed-projection-weights.md) | [목차](README.md) | [다음: C11](11-lm-head-sampling-fusion.md)

## 1. 중요한 범위 규칙

이 문서는 두 후보군을 설명하지만 실제 merge PR은 **한 후보만** 구현한다.

- 후보 A: packed QKV output의 split/position transform/KV write 경계
- 후보 B: packed Gate-Up output의 activation/gate multiply 경계

post-C09 profile에서 primary bottleneck으로 선택된 하나만 C10에 들어간다. 둘 다 유효하면 두 번째 후보는 별도 `C10-follow-up` PR로 분리한다. correctness와 aggressive optimization을 한 PR에 섞거나 서로 다른 fusion의 효과를 합산해 단독 효과를 숨기지 않는다.

## 2. 공통 가설

반복 subgraph 사이의 intermediate global-memory write/read와 kernel launch를 제거하면, 단일 primitive 최적화보다 model layer 전체의 TPOT/TTFT에 더 큰 효과가 있다.

승격 근거는 microbenchmark가 아니라 paired end-to-end 결과다.

## 3. 후보 선택을 위한 profiling

C09 accepted candidate를 기준으로 다음을 Nsight Systems/Compute와 Riley trace에서 수집한다.

- pattern별 GPU duration과 호출 수
- intermediate read/write bytes
- launch/API time
- register/shared-memory/occupancy
- graph node inventory
- sequence length와 active-row bucket별 비중
- c1 latency와 c8 throughput에서의 상대 병목

선택 기준:

```text
expected removable time/bytes가 가장 큼
E0 semantics와 exact fallback을 닫을 수 있음
대표 GPU/model geometry에서 재사용 가능
graph compatibility를 보존 가능
```

결과를 보기 전에 후보별 promotion threshold는 동일하게 고정한다.

# 후보 A — QKV Split + RoPE + Paged-KV Write

## 4A. 목표 경로

현재 개념 경로:

```text
packed QKV GEMM
  -> Q/K/V segment materialization
  -> indexed RoPE on Q
  -> indexed RoPE on K
  -> paged KV write K/V
  -> attention
```

후보 경로:

```text
packed QKV GEMM
  -> fused split + Q/K RoPE + paged K/V write
  -> Q output view + attention-ready KV
```

초기 구현은 GEMM 자체를 custom kernel로 대체하지 않는다. cuBLASLt packed output 뒤의 split/RoPE/KV write를 하나의 CUDA C++ kernel로 fuse한다. profiler가 증명하지 않은 CUTLASS epilogue 통합은 후속 단계다.

## 5A. Contract

- standard/partial RoPE capability를 명시
- Q/K head count와 width 차이 지원
- absolute position과 rope theta/scaling metadata 고정
- page table version과 physical block bounds 검증
- V는 변환 없이 exact segment에서 KV cache로 write
- Q는 attention backend가 요구하는 layout으로 출력
- inactive/padded rows는 KV mutation 금지

## 6A. Geometry

최초 지원 후보:

- head dimension 64/128 중 profiler-selected 범위
- GQA/MHA closed head ratio
- active-row `1,2,4,8,16,32`
- BF16 input/output, FP32 trig/reduction contract

unsupported dynamic/multimodal RoPE는 separate fallback이다.

# 후보 B — SwiGLU Activation + Gate Multiply

## 4B. 목표 경로

현재 개념 경로:

```text
packed Gate-Up GEMM
  -> Gate segment
  -> Up segment
  -> SiLU(Gate)
  -> activation * Up
  -> Down GEMM
```

후보 경로:

```text
packed Gate-Up GEMM
  -> fused SiLU(Gate) * Up
  -> Down GEMM input
```

초기 구현은 packed GEMM output을 소비하는 fused CUDA C++ epilogue kernel이다. CUTLASS GEMM epilogue에 직접 결합하는 후보는 separate kernel이 5% gate를 통과하고 추가 memory round trip이 여전히 top bottleneck일 때만 후속으로 비교한다.

## 5B. Contract

- exact SiLU formula/order와 BF16/FP32 staging 정의
- gate/up segment offset과 padding 검증
- signed zero, NaN/Inf propagation contract
- output layout이 down GEMM input descriptor와 일치
- inactive row zero/sentinel 보존

## 7. Registry와 fallback

C08 registry에 선택된 candidate 하나를 추가한다.

```text
semantic pattern
implementation ID
capability predicate
semantic class E0
workspace bytes
graph capability
exact separate fallback ID
```

runtime flag 또는 implementation override로 fused/separate를 비교할 수 있어야 한다. capability `Unknown`은 fused candidate를 선택하지 않는다.

## 8. Graph compatibility

C07 graph signature에는 implementation ID와 layout revision이 포함된다.

- fused/separate graph를 같은 signature로 재사용 금지
- graph capture 전 kernel attributes와 static address 검증
- candidate poison 시 matching graph도 poison
- fallback은 별도 eager 또는 prepared fallback graph를 사용

fusion을 위해 graph correctness gate를 우회하지 않는다.

## 9. Numeric correctness

### Standalone

- FP32 oracle와 BF16 fixed tolerance
- extreme value, signed zero, NaN/Inf
- alignment/odd dimensions/padding
- repeated byte determinism 가능한 범위

### Model

- SmolLM2/Qwen
- prefill/decode
- prompt `18,128,1024,4096,near-limit`
- active rows `1,2,4,8,16,32`
- 32/128-step greedy token exact
- graph/eager, fused/separate alternating
- KV boundary와 permuted block table — 후보 A

canonical tolerance는 결과를 본 뒤 변경하지 않는다.

## 10. Performance campaign

baseline은 C09 accepted separate implementation이다.

- independent process 5쌍 AB/BA
- c1/p128/o32 primary
- c8/p128/o32 throughput
- c1/p4096/o128 long
- graph disabled와 enabled를 별도 비교

Metric:

- selected pattern GPU time
- kernel/node count
- intermediate HBM bytes
- TTFT/TPOT/E2E/throughput
- register/shared-memory/occupancy
- workspace/peak VRAM

## 11. Promotion gate

- failure/token mismatch/dropped trace 0
- selected pattern paired latency improvement `>= 10%`
- end-to-end primary TPOT 또는 TTFT paired improvement `>= 5%`
- c8 throughput ratio `>= 0.95`
- non-target p95 ratio `<= 1.05`
- peak VRAM ratio `<= 1.05`
- graph hit/capture 회귀 없음
- hot allocation 0

pattern microbenchmark만 10% 개선되고 end-to-end 5%를 넘지 못하면 default 승격하지 않는다.

## 12. 기술 escalation

```text
separate CUDA C++
-> fused CUDA C++ consumer kernel
-> profiler 재검증
-> CUTLASS epilogue prototype
-> end-to-end 이점이 있을 때만 production CUTLASS
```

universal custom GEMM, Triton runtime, NVRTC는 이 PR에 포함하지 않는다.

## 13. 예상 파일 변경

선택 후보에 필요한 파일만 수정한다.

```text
kernels/src/batch_primitives.cu 또는 새 fused_*.cu
kernels/include/riley_cuda.h
crates/riley-cuda/src/primitives.rs
crates/riley-runtime/src/pattern.rs
crates/riley-runtime/src/llama/executor/dispatch.rs
crates/riley-runtime/tests/*_fusion_gpu.rs
benchmarks/results/<campaign>/...
```

## 14. 실패와 롤백

- pre-dispatch unsupported/mismatch: separate fallback
- post-dispatch completion 확인 오류: candidate/graph poison, 안전한 다음 iteration부터 separate
- completion 미확정: executor poison, 자동 재실행 금지
- 운영 rollback: implementation override 또는 runtime flag로 separate

## 15. 완료 정의

post-C09 profile이 선택한 반복 subgraph 하나가 fixed E0/token gate와 end-to-end 5% promotion gate를 통과하거나, 미달하여 `not-promoted`로 명확히 종료될 때 완료다.
