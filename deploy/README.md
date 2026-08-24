# rustinfer 단계별 실행 계획

이 폴더는 `rustinfer`의 방대한 설계 문서를 **한 번에 하나씩 검토·구현·검증할 수 있는 PR 단위**로 분해한다.

기준 문서:

- [프로젝트 비전](../README.md)
- [Transformers 연산 모듈 분석](../docs/transformers-model-operation-analysis.md)

현재 원칙은 **구현보다 기존 구조 분석과 검증 계약을 먼저 확정하는 것**이다. 앞 단계의 승인 기준을 통과하지 못하면 다음 단계로 넘어가지 않는다.

## 진행 상태 표기

- `Planned`: 아직 시작하지 않음
- `Active`: 현재 작업 중
- `Blocked`: 선행 조건 또는 검증 실패
- `Complete`: 승인 기준을 통과하고 병합됨
- `Deferred`: 범위 밖으로 명시적으로 연기

## 전체 순서

| 순서 | 문서 | PR의 한 가지 목적 | 완료 시 얻는 결과 |
|---:|---|---|---|
| 00 | [PR 계약](00-pr-contract.md) | 모든 PR의 크기·검증·롤백 규칙과 수학적 최적화 등급 확정 | 리뷰 가능한 공통 작업 방식 |
| 01 | [기준선과 재현성](01-baseline-and-reproducibility.md) | 하드웨어·모델·정확도·성능·근사 허용 기준 고정 | 비교 가능한 benchmark contract |
| 02 | [Workspace와 CI](02-workspace-and-ci.md) | Rust/CUDA 프로젝트 뼈대만 구축 | 빌드·정적검사·테스트 기반 |
| 03 | [CUDA Host Runtime](03-cuda-host-runtime.md) | Rust에서 CUDA를 안전하게 호출 | device/stream/event/FFI smoke path |
| 04 | [Tensor와 메모리](04-tensor-and-memory.md) | GPU 메모리 lifetime과 layout 정의 | `DeviceBuffer`, `TensorView`, workspace |
| 05 | [모델 로딩과 Canonical IR](05-model-loading-and-ir.md) | HF 설정·가중치를 실행 가능한 IR로 변환 | Llama 계열 모델 기술자와 weight map |
| 06 | [핵심 Primitive](06-core-primitives.md) | 모델 조립 전 핵심 연산 정확도 확보 | GEMM adapter, embedding, norm, RoPE 등 |
| 07 | [Llama 기준 Forward](07-llama-reference-forward.md) | cache 없는 단일 prefill logits 일치 | 최초 end-to-end GPU forward |
| 08 | [Prefill Attention](08-prefill-attention.md) | online softmax를 포함한 정확한 attention backend 분리 | score matrix를 만들지 않는 prefill 경로 |
| 09 | [단일 요청 Decode](09-single-request-decode.md) | 결합 가능한 부분합 기반 decode와 연속 KV cache | 단일 요청 autoregressive core |
| 10 | [Paged KV 관리](10-paged-kv-manager.md) | block 단위 KV 할당·회수와 확장 가능한 metadata ABI | batching·후속 page 최적화용 cache substrate |
| 11 | [Sampling과 Generation](11-sampling-and-generation.md) | 재현 가능한 RNG 계약과 token generation 완성 | 단일 요청 생성 API core |
| 12 | [Qwen 호환성](12-qwen-compatibility.md) | 두 번째 모델 family로 IR 재사용 검증 | 모델별 복제 없는 확장성 검증 |
| 13 | [Scheduler와 Continuous Batching](13-scheduler-and-batching.md) | 여러 요청을 GPU step으로 구성 | Rust-native serving control plane |
| 14 | [API와 Streaming](14-api-and-streaming.md) | OpenAI 호환 서비스 경계 제공 | 취소·backpressure 포함 서버 |
| 15 | [Profiling과 최적화](15-profiling-and-optimization.md) | 측정된 병목만 exact fusion 또는 명시적 오차 예산으로 개선 | 성능 회귀 방지와 최적 fast path |
| 16 | [신뢰성 및 Release Gate](16-reliability-and-release.md) | 의미 보존 등급별 장시간 안정성과 배포 가능성 검증 | 첫 release candidate |
| 17 | [확장 Gate](17-extension-gates.md) | Quantization 변환, speculative, sparse attention, SSM 등 진입 조건 정의 | 범위 폭증 방지용 후속 로드맵 |

## 수학적 최적화 배치 원칙

수학적으로 연산을 치환할 수 있다는 이유만으로 초기 core에 넣지 않는다. 모든 최적화는 다음 네 등급 중 하나를 선언한다.

| 등급 | 의미 | 기본 정책 | 배치 단계 |
|---|---|---|---|
| `E0` | 실수 연산에서 동일한 대수적 재구성 | reference parity 후 사용 가능 | PR 08·09·15 |
| `E1` | 목표 확률분포를 보존하는 확률적 알고리즘 | RNG·분포 검증 후 opt-in | PR 17 speculative |
| `A1` | 오차 상한 또는 품질 예산이 있는 근사 | 기본 비활성, exact fallback 필수 | PR 15·17 |
| `M1` | 재학습·calibration 등 모델 의미를 변경 | 별도 연구 트랙 | PR 17 이후 |

현재 계획에 반영한 항목:

- **Online softmax와 associative partial reduction:** PR 08·09
- **Paged KV의 선택적 page 통계 확장 지점:** PR 10
- **RNG snapshot/restore와 sampling 의미 계약:** PR 11
- **Query-aware page pruning의 오차 예산 실험:** PR 15·17
- **대각 스케일링·직교/Hadamard 회전을 이용한 저정밀화:** PR 17
- **Rejection sampling 기반 speculative decoding과 동적 lookahead:** PR 17
- **Jacobi/lookahead decoding:** PR 17의 별도 experimental gate
- **Associative scan 기반 Mamba/SSM:** PR 17
- **SVD/저랭크 weight·KV 압축:** PR 17의 조건부 연구 gate

## 주요 Gate

### Gate A — 기준선 고정

PR 01이 완료되기 전에는 “더 빠르다”는 주장을 하지 않는다.

### Gate B — 수치 정확성

PR 07에서 동일 checkpoint의 logits가 기준 구현과 허용 오차 내에서 일치해야 한다.

### Gate C — 단일 요청 생성

PR 11에서 KV cache on/off 결과, greedy token sequence, RNG 재현성과 streaming 종료 동작이 검증되어야 한다.

### Gate D — 서비스 가능

PR 14에서 취소, 연결 종료, 과부하, 오류 응답이 GPU 자원 누수 없이 처리되어야 한다.

### Gate E — 첫 릴리스

PR 16의 soak test와 성능 회귀 gate를 통과해야 release tag를 만든다. `A1` 근사는 첫 릴리스 기본 경로가 아니다.

## 실행 규칙

1. 번호 순서대로 진행한다.
2. 각 PR은 문서에 적힌 **한 가지 목적**만 수행한다.
3. 다음 단계에 필요한 interface는 정의할 수 있지만 구현을 미리 끌어오지 않는다.
4. 성능 개선 PR에는 변경 전후 수치와 측정 환경을 반드시 포함한다.
5. 모든 `unsafe`와 CUDA FFI는 작은 경계에 가둔다.
6. 외부 backend를 먼저 사용하고, profiler가 증명한 병목만 custom kernel 대상으로 삼는다.
7. 한 PR에서 correctness와 aggressive optimization을 동시에 하지 않는다.
8. `E0`, `E1`, `A1`, `M1` 중 의미 보존 등급을 선언하지 않은 최적화는 병합하지 않는다.
9. 근사 최적화는 error budget, exact fallback, feature flag와 결과 표기를 가져야 한다.

다음 문서: [00 — PR 계약](00-pr-contract.md)