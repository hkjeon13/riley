# PR 16 — 신뢰성, Soak Test, 첫 Release Gate

**상태:** Active
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

#### PR 04 CUDA memory fault-injection gate

PR 04의 정상·validation 경로와 Compute Sanitizer만으로 직접 만들 수 없는 다음 native
실패 분기는 test-only injectable backend로 강제한다.

- allocation create가 실패한 뒤 rollback `cudaFree`/`cudaFreeHost` 결과가 모호한 경우
- explicit close의 `cudaFree*` 결과가 모호한 경우
- copy completion은 확인됐지만 token에 저장된 submission/deferred CUDA 오류가 있는 경우
- query/synchronize 또는 current-context 복원 실패로 copy completion이 미확정인 경우

각 fault case는 shared primary context와 같은 process의 후속 leak gate를 오염시키지
않도록 별도 subprocess에서 실행한다. Allocation 해제가 확인되지 않으면 live
bytes/count 또는 context child가 남아 context close를 거부해야 한다. Copy 완료가
확인된 deferred 오류는 stream/device/pinned reservation을 정확히 한 번 해제하면서 원래
CUDA 오류를 반환해야 한다. 완료가 확인되지 않은 오류는 native active token과 Rust
busy 상태를 남겨 reuse/free를 거부해야 한다. 각 subprocess evidence는 double
decrement/free가 없고 오류를 성공으로 잘못 보고하지 않았음을 함께 검증한다.

### 의미 보존 등급별 검증

#### `E0`

- reference와 canonical correctness corpus(31 case)의 token-level exact 회귀
- 여러 partition, graph bucket, backend fallback
- extreme logits와 긴 context

이 유한 canonical correctness gate는 soak와 독립적으로 실행·판정한다.

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

허용 threshold는 첫 serving 경로와 동일한 request identity, model, GPU/toolchain,
warmup/measurement 횟수를 갖춘 PR 15의 accepted command-batch 기준선에 상대적으로
정의한다. PR 01은 oracle·환경·초기 kernel 기준선의 provenance로 유지하지만, 당시에는
PR 13~15의 scheduler/API/iteration-batch 경로가 없었으므로 release serving ratio의 직접
분모로 사용하지 않는다. 고정 기준선과 threshold는
`benchmarks/release/performance-baseline-v1.json`이 authoritative하다.

예:

- TTFT p50/p95
- TPOT p50/p95
- throughput
- peak VRAM
- CPU utilization
- exact fallback overhead
- optional metadata overhead

noise보다 작은 threshold를 두지 않는다. 환경 편차를 먼저 측정한다.

## Canonical E0 GEMM Release 진단

첫 candidate 승인을 위한 canonical correctness 재검증은 cuBLASLt reduction policy를 model-global
switch가 아닌 prepared plan class별 cold contract로 고정한다. `server-4096`의 RTX 4090에서
HF CUDA 13.1과 native CUDA 12.8 raw first heuristic을 S=18/128/1024/4096/8064에 대해
비교했으며 algorithm/tile/stage/split-K/reduction/workspace가 모두 일치했다.

- 검증된 BF16 dense-576 Llama: q/o, k/v, down에서 reviewed `OUTPUT_TYPE`/`INPLACE` 보존
- 같은 profile의 gate/up과 LM head: strict `NONE`
- 모든 M=1 decode plan: strict `NONE`
- Qwen2 및 미검증 Llama geometry: 전 plan strict `NONE`
- `COMPUTE_TYPE`과 unknown C ABI flag: fail closed

`INPLACE`가 실제로 선택되는 S=1024/4096/8064는 process 내부 각 100회와 fresh process
각 10회에서 first-layer hidden, full BF16 logits와 token이 byte-identical했다. 이 관측은
pinned GPU/toolchain/workspace contract에 한정한다. 최종 release 승인은 이 진단만으로
대체하지 않고 동일한 candidate 대상 revision의 전체 native correctness와 Qwen
regression, Python-free, reproducible build, performance 및 soak evidence를 각각 요구한다.
전체 native correctness와 Qwen regression은 soak와 독립적인 단기 gate이며, soak
실행 여부와 무관하게 판정한다.

## 이번 순차 구현 실행의 Release-owner Soak 예외

2026-08-27 사용자는 이번 7시간 15분 soak 재실행을 생략하도록 지시한 뒤, 이전 soak를
한 차례 수행했다는 운영 판단을 근거로 첫 Riley 릴리스를 명시적으로 승인했다. 이전
실행은 최종 Riley revision, source archive, binary와 v2 raw evidence로 교차 결합된
artifact가 아니므로 이 저장소의 fail-closed checker에서는 `Gate E passed` 또는
`soak-qualified` 증거로 간주하지 않는다.

따라서 이 승인은 checker나 threshold를 완화하지 않는 **release-owner prerelease
예외**다. 태그와 release notes는 candidate-bound soak 및 final candidate report가
없다는 사실을 공개해야 한다. Soak launcher, raw-evidence packaging, replay checker와
동일-revision gate는 그대로 유지하며, 정식 Gate E qualification에는 원래 계약의
전체 evidence를 다시 실행해야 한다.

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
