# PR 16 — 신뢰성, Soak Test, 첫 Release Gate

**상태:** Planned  
**선행 조건:** [PR 15](15-profiling-and-optimization.md)  
**다음:** [PR 17 — 확장 Gate](17-extension-gates.md)

[← 이전](15-profiling-and-optimization.md) | [목차](README.md) | [다음 →](17-extension-gates.md)

## 목적

짧은 benchmark가 아니라 장시간 서비스에서 메모리, 오류, 취소, 과부하와 성능 회귀를 검증해 첫 release candidate를 만든다. 최적화의 의미 보존 등급과 함께 **Python-free production runtime 경계**를 release gate로 확인한다.

## 첫 Release의 Runtime Dependency 정책

운영 패키지에 허용되는 구성:

- Rust executable/library
- native CUDA C++ library
- CUDA Driver/Runtime
- cuBLASLt와 명시된 NVIDIA native dependency
- compile된 CUTLASS/custom kernel artifact
- model/tokenizer/config artifact

운영 패키지와 server process에서 금지:

- Python interpreter 또는 virtual environment
- PyTorch
- Hugging Face Transformers
- Python subprocess fallback
- Triton Python JIT/compiler
- pickle 또는 Python class artifact

Triton AOT artifact나 NVRTC를 production에 포함하려면 이 단계 이전에 별도 승인 문서와 dependency/실패 정책이 있어야 한다. 초기 release 기본값은 `nvcc`로 compile된 native kernel이다.

## Python-free Release Test

Python이 설치되지 않은 clean container 또는 host에서 다음을 수행한다.

1. production build 또는 release artifact 설치
2. `config.json`, tokenizer artifact, safetensors load
3. prefill
4. 여러 token decode
5. greedy와 sampling
6. streaming API
7. cancellation
8. model unload/shutdown
9. golden token 비교

추가 검사:

- `ldd`, loader inspection 또는 동등한 native dependency 목록
- process tree에 Python child 없음
- filesystem/package에 PyTorch/Transformers wheel 없음
- `PATH`에 Python이 없어도 startup 성공
- optional Python-generated checkpoint artifact를 Python 없이 load

이 gate를 통과하지 못하면 release candidate가 아니다.

## 첫 Release의 의미 정책

첫 stable release의 기본 경로는 다음만 허용한다.

- reference path
- 검증을 완료한 `E0` exact optimization

다음은 기본 비활성 또는 release 범위 밖이다.

- `E1` distribution-preserving experimental path
- `A1` approximate path
- `M1` model-changing path

`E1` 또는 `A1`을 preview로 포함한다면 다음이 필요하다.

- 명시적인 feature flag
- exact fallback
- request/run metadata에 사용 여부 기록
- 별도 benchmark와 correctness/quality report
- 운영 중 즉시 비활성화 가능한 configuration

## 신뢰성 시나리오

- 장시간 steady load
- burst 후 idle 반복
- 짧은 요청과 긴 요청 혼합
- 높은 cancellation 비율
- client disconnect
- 잘못된 요청 반복
- KV capacity 근접
- model load/unload 반복
- server graceful shutdown/restart
- CUDA 오류 주입 가능한 범위
- exact backend 간 runtime switch
- experimental flag on/off 반복
- fallback을 유발하는 unsupported shape/metadata
- Python 없는 runtime 반복 실행

## 검증 항목

### 메모리

- host RSS 추이
- VRAM high-water mark
- free block 복귀
- fragmentation
- pinned memory
- metadata leak
- optional KV sidecar lifetime
- fallback 전환 후 workspace 회수

### 동시성

- deadlock/livelock
- channel backlog
- scheduler starvation
- shutdown race
- duplicate finish event
- RNG state가 다른 request 사이에서 공유되지 않음
- backend fallback 중 stream ordering 유지

### 오류 격리

한 요청의 오류가 다음을 유발하지 않아야 한다.

- process abort
- 다른 요청 cache 손상
- 다른 요청 RNG 오염
- stream 영구 오류
- block pool 불일치
- stale optional metadata 재사용
- Python fallback/subprocess 기동
- 민감 정보 로그 노출

### 의미 보존 등급별 검증

#### `E0`

- reference와 장시간 token-level 회귀
- 여러 partition, graph bucket, backend fallback
- extreme logits와 긴 context

#### `E1` preview

- acceptance와 residual correction invariant
- request-local RNG snapshot/restore
- distribution 검증 artifact
- target/draft 오류 시 exact target path 복귀

#### `A1` preview

- error budget configuration validation
- exact fallback 비율
- approximation 사용 여부 metric
- epsilon 또는 rank/bit-width별 quality regression
- 잘못되거나 stale한 summary에서 exact path 전환

## Release artifact

- supported GPU/CUDA matrix
- supported model family와 제약
- native dependency manifest
- Python-free build/startup 검증 결과
- build instructions
- benchmark report
- semantic class별 feature table
- default enable/disable 상태
- approximation/error-budget configuration
- known limitations
- configuration reference
- operational metrics
- upgrade/rollback guide

Optional Python reference/calibration 도구는 별도 development package로 배포하며 production artifact에 포함하지 않는다.

## 성능 회귀 gate

허용 threshold는 PR 01 기준선에 상대적으로 정의한다.

예:

- TTFT p50/p95
- TPOT p50/p95
- throughput
- peak VRAM
- CPU utilization
- exact fallback overhead
- optional metadata overhead

noise보다 작은 threshold를 두지 않는다. 환경 편차를 먼저 측정한다.

## 비범위

- 새로운 모델 family
- quantization stable enablement
- MoE
- multi-GPU
- speculative decoding stable enablement
- approximate attention의 기본 활성화
- Python runtime fallback
- 기능 추가를 통한 benchmark 개선

## 완료 기준

- [ ] 정해진 soak scenario 통과
- [ ] host/VRAM 누수 징후 없음
- [ ] overload와 cancellation 안정
- [ ] release build 재현 가능
- [ ] Python 없는 clean 환경에서 end-to-end generation과 API test 통과
- [ ] native dependency manifest에 Python/PyTorch/Transformers/Triton JIT가 없음
- [ ] stable 기본 경로가 reference 또는 검증된 `E0`로 제한됨
- [ ] experimental feature의 flag, 표기, exact fallback 검증
- [ ] supported/unsupported 범위 명확
- [ ] rollback 절차 검증
- [ ] Gate E와 Python-free Gate 승인 후 첫 tag 생성 가능

[← 이전](15-profiling-and-optimization.md) | [목차](README.md) | [다음 →](17-extension-gates.md)
