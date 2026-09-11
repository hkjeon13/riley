# Riley 구현 언어·라이브러리와 Runtime Dependency 경계

> 상태: Architecture decision  
> 적용 범위: production server, inference runtime, CUDA kernels, 개발·분석 도구  
> 관련 계획: [`deploy/`](../deploy/README.md)

## 1. 최종 결정

Riley의 **production runtime과 inference path는 Python-free**로 구성한다.

```text
Production Runtime
────────────────────────────────────────
Rust
CUDA C++
CUDA Driver/Runtime API
cuBLASLt
CUTLASS — 필요성과 성능이 검증된 경우
검증된 native attention backend

Python 의존성 없음
PyTorch 의존성 없음
Transformers 의존성 없음
Triton Python runtime/JIT 의존성 없음
```

Python은 삭제 대상이 아니라 다음과 같이 **비교·연구·오프라인 도구 전용**으로 사용한다.

```text
Development / Reference / Offline Tools
────────────────────────────────────────
Python
PyTorch
Hugging Face Transformers
NumPy / SciPy
선택적 Triton prototype

운영 서버 프로세스에는 포함하지 않음
```

핵심 문장은 다음과 같다.

> Python은 Riley가 맞는지 비교하고 모델을 분석·변환하기 위해 사용할 수 있지만, 요청을 처리하고 모델을 추론하는 모듈에는 필요하지 않다.

---

## 2. 언어와 라이브러리의 책임 분담

| 영역 | 기본 기술 | 책임 |
|---|---|---|
| API server | Rust | HTTP, SSE, request validation, cancellation, backpressure |
| Scheduler | Rust | admission, batching, iteration plan, queue state |
| KV metadata | Rust | block lifetime, refcount, block table, rollback, eviction policy |
| Model runtime | Rust | execution plan, shape, workspace, backend dispatch |
| Model parsing | Rust | `config.json`, `tokenizer.json`, `safetensors`, canonical IR |
| Tokenization | Rust library | encode/decode, special token 처리 |
| CUDA orchestration | Rust | device/context/stream/event/graph lifetime |
| FFI boundary | C ABI | Rust와 CUDA C++ 사이의 안정적인 함수 호출 규약 |
| Custom GPU operation | CUDA C++ | norm, RoPE, KV access, sampling, reductions, scan |
| Dense GEMM | cuBLASLt | QKV/O projection, MLP, LM head의 기본 경로 |
| Fused/quantized/grouped GEMM | CUTLASS/CuTe | profiler가 필요성을 증명한 특수 GEMM |
| Attention | native CUDA backend | prefill/decode attention; capability 기반 dispatch |
| Kernel prototype | Triton, 선택 사항 | 아이디어 검증과 비교 실험; production 기본 경로 아님 |
| Reference 결과 | Python/PyTorch/Transformers | golden logits, hidden state, token sequence 생성 |
| Checkpoint 변환 | Python 또는 Rust offline tool | weight rename/packing/rotation/분해 artifact 생성 |
| Calibration·수치 분석 | Python/NumPy/SciPy/PyTorch | quantization, SVD, activation statistics |
| Benchmark 분석 | Python 또는 독립 도구 | 통계, 그래프, 회귀 분석; server와 분리 |

### C와 C++의 구분

- 실제 GPU kernel 구현 언어는 **CUDA C++**이다.
- Rust에 노출되는 외부 함수만 `extern "C"` ABI를 사용한다.
- C ABI를 사용한다는 것이 kernel을 C로 구현한다는 뜻은 아니다.

```text
Rust safe API
  ↓
Rust unsafe FFI wrapper
  ↓ extern "C"
CUDA C++ wrapper
  ↓
cuBLASLt / CUTLASS / custom CUDA kernel
```

---

## 3. Production dependency graph

허용하는 dependency 방향은 다음과 같다.

```text
riley-server
      ↓
riley-scheduler
      ↓
riley-runtime
      ├─ riley-model
      ├─ riley-tensor
      └─ riley-cuda
              ↓ C ABI
       native CUDA library
              ├─ CUDA Runtime/Driver
              ├─ cuBLASLt
              ├─ CUTLASS-generated kernels
              └─ custom CUDA C++
```

금지하는 runtime dependency:

```text
riley runtime → Python interpreter
riley runtime → PyTorch
riley runtime → Transformers
riley runtime → Python subprocess
riley runtime → pickle artifact
riley runtime → runtime Triton Python compiler
```

Rust server가 실패했을 때 Python Transformers를 운영 fallback으로 호출하지 않는다. 실패는 명확한 오류 또는 native exact fallback으로 처리한다.

---

## 4. Production에서 Python을 제거하는 이유

Python 자체가 항상 느리기 때문이라는 단순한 이유가 아니다. 운영 경로에서 제거하는 목적은 다음과 같다.

- GIL과 Python object lifecycle을 scheduler hot path에서 제거
- Python GC와 interpreter 상태에 의한 latency variance 제거
- 단일 native binary 또는 명확한 native package로 배포
- CUDA stream과 device allocation lifetime을 직접 관리
- PyTorch allocator와 graph/runtime 정책에 강하게 결합되지 않음
- dependency와 보안 표면 축소
- CPU orchestration profiling과 메모리 accounting 단순화
- 실패 시 native stack 내부에서 일관된 cleanup 수행

Rust를 사용한다고 GPU kernel이 자동으로 빨라지는 것은 아니다. Rust의 이점은 **host runtime의 예측 가능성, concurrency, ownership, 상태 관리**에서 얻는다.

---

## 5. Python이 필요한 개발 영역

### 5.1 Hugging Face reference 생성

PyTorch/Transformers를 실행해 다음 golden artifact를 만든다.

- token IDs
- 선택 layer의 hidden states 또는 checksum
- Q/K/V와 norm output 일부
- final logits
- top-k tokens
- greedy generation
- KV cache on/off 결과

이 reference는 Rust 구현을 검증하기 위한 외부 기준이다. Python reference를 production fallback으로 사용하지 않는다.

### 5.2 모델 구조 분석

Transformers의 `modeling_*.py`, `modular_*.py`, config mapping을 분석할 때 Python AST와 생태계를 사용할 수 있다. 분석 결과는 JSON/CSV/Markdown 같은 정적 artifact로 저장한다.

### 5.3 Checkpoint 변환과 packing

오프라인으로 다음을 수행할 수 있다.

- weight name normalization
- Q/K/V packing
- gate/up packing
- tensor transpose 또는 execution layout 변환
- shard 병합·분할
- 대각 scaling과 rotation
- SVD/low-rank decomposition
- quantization calibration과 변환

운영 runtime은 변환 결과를 표준 artifact로 직접 읽는다.

### 5.4 Triton prototype

Triton은 다음에 유용하다.

- online softmax prototype
- RMSNorm와 fused elementwise 실험
- paged attention 아이디어 검증
- grouped GEMM 비교
- tile/warp/layout 탐색

그러나 Triton prototype이 빠르다는 사실만으로 production dependency가 되지 않는다. 기본 절차는 다음이다.

```text
Triton prototype
  ↓ correctness + profiler
CUDA C++ / CUTLASS production port
  ↓ end-to-end validation
native runtime integration
```

Triton을 production에 남기려면 별도 architecture decision이 필요하다. 최소 조건:

- Python 없는 AOT artifact loading
- target GPU와 compiler version 고정·검증
- stream, CUDA Graph, allocator semantics 검증
- cold-start와 kernel cache 운영 계획
- CUDA C++ 대비 유지보수·성능 이점 입증

초기 release에서는 Triton production runtime을 사용하지 않는다.

---

## 6. Artifact 경계

Python/offline tool과 Rust runtime 사이에는 언어 객체가 아니라 명시적 artifact만 전달한다.

권장 형식:

- `config.json`
- `tokenizer.json`
- `.safetensors`
- 변환 manifest JSON
- calibration JSON 또는 safetensors
- benchmark CSV/JSONL
- kernel selection JSON

금지 또는 지양:

- Python pickle
- 임의 Python class serialization
- 실행 시 `trust_remote_code`
- Python module path가 있어야 해석되는 checkpoint
- source revision이 없는 변환 결과

변환 manifest 예시:

```json
{
  "format": "riley-checkpoint-v1",
  "source_model": "org/model",
  "source_revision": "immutable-revision",
  "converter_revision": "git-sha",
  "transforms": ["packed_qkv", "packed_gate_up"],
  "dtype": "bf16",
  "weight_files": ["model.safetensors"]
}
```

모든 offline artifact에는 source model, source revision, tool revision, transform parameters를 기록한다.

---

## 7. Kernel 구현 선택 순서

GPU operation은 다음 순서로 선택한다.

### 7.1 Dense GEMM

```text
1. cuBLASLt
2. cuBLASLt algorithm/epilogue tuning
3. CUTLASS — 필요한 fusion·dtype·layout이 없을 때
4. custom CUDA GEMM — 원칙적으로 최후 수단
```

범용 GEMM을 처음부터 직접 구현하지 않는다.

### 7.2 Non-GEMM primitive

```text
1. 명확한 CUDA C++ reference
2. 검증된 native library backend
3. profiler 기반 custom/fused CUDA C++
```

대상:

- embedding gather
- RMSNorm/LayerNorm
- RoPE
- KV write/gather
- logits processing/sampling
- MoE dispatch metadata
- scan/reduction

### 7.3 Attention

```text
1. correctness-first score-matrix reference
2. 검증된 native CUDA attention backend
3. target workload의 공백이 확인된 경우 custom CUDA C++
```

Prefill과 decode를 별도 mode로 취급한다. Paged cache, head layout, dtype, CUDA Graph capability를 backend key에 포함한다.

### 7.4 CUTLASS 도입 조건

CUTLASS는 다음 중 하나가 측정된 경우에만 도입한다.

- cuBLASLt에서 필요한 epilogue/fusion을 표현할 수 없음
- quantized GEMM format 지원이 부족함
- MoE grouped GEMM이 병목
- 특수 weight layout이 반복적인 변환을 유발
- target shape에서 end-to-end 이점이 재현됨

### 7.5 NVRTC

초기에는 `nvcc`로 미리 compile한 kernel을 배포한다. NVRTC runtime specialization은 다음 조건 이후의 별도 단계다.

- static kernel matrix가 지나치게 커짐
- shape specialization의 성능 이점이 측정됨
- compile cache, timeout, failure fallback, artifact provenance가 설계됨
- production 환경의 compiler dependency가 승인됨

---

## 8. 권장 저장소 구조

```text
riley/
├── crates/                       # production Rust
│   ├── riley-core/
│   ├── riley-cuda/
│   ├── riley-tensor/
│   ├── riley-model/
│   ├── riley-runtime/
│   ├── riley-scheduler/
│   └── riley-server/
│
├── kernels/                      # production native GPU code
│   ├── CMakeLists.txt
│   ├── include/                  # C ABI headers
│   ├── src/                      # CUDA C++ kernels
│   └── cutlass/                  # 필요한 경우의 CUTLASS ops
│
├── tools/
│   ├── python/                   # optional offline/reference tools
│   │   ├── reference/
│   │   ├── architecture/
│   │   ├── checkpoint/
│   │   ├── calibration/
│   │   ├── quantization/
│   │   ├── low_rank/
│   │   └── benchmark/
│   └── native/                   # 필요 시 Rust/C++ offline tools
│
├── experiments/
│   └── triton/                   # optional prototype, production 비의존
│
├── benchmarks/
├── docs/
└── deploy/
```

`tools/python/`과 `experiments/triton/`은 production crate dependency graph 밖에 둔다.

---

## 9. Build와 Release Gate

### 9.1 Rust/CUDA build

다음 명령은 Python이 없는 환경에서도 성공해야 한다.

```bash
cargo build --release --features cuda,server
```

빌드가 Python executable, Python headers, PyTorch C++ extension 또는 Triton compiler를 요구하면 production boundary 위반이다.

### 9.2 Server startup

다음 동작 중 Python subprocess를 생성하거나 Python module을 import하지 않아야 한다.

```bash
riley serve --model /models/example
```

### 9.3 Python-free runtime test

CI 또는 release container에서 다음을 검증한다.

- Python이 설치되지 않은 이미지에서 server 시작
- 표준 HF artifact 또는 Riley 변환 artifact 로딩
- tokenizer encode/decode
- prefill와 decode
- streaming response
- cancellation과 cleanup
- golden token 결과
- `ldd` 또는 동등한 dependency inspection
- child process 목록에 Python 없음

### 9.4 Offline tool test

Python tool은 별도 optional job에서 실행한다.

- lock된 Python dependency
- source/tool revision 기록
- deterministic artifact 또는 허용된 오차
- generated artifact를 Rust가 Python 없이 읽는 integration test

### 9.5 Release package

운영 패키지는 다음으로 구성한다.

- Rust executable/library
- native CUDA shared/static library
- 필요한 CUDA/cuBLASLt runtime dependency 설명
- model/tokenizer artifacts
- configuration

Python virtual environment, PyTorch wheel, Transformers package는 production package에 포함하지 않는다.

---

## 10. PR 단계 반영

| 단계 | 기술 경계 |
|---|---|
| PR 01 | Python reference 환경과 Python-free Riley benchmark 환경 분리 |
| PR 02 | Rust/CUDA production workspace와 optional `tools/python`, `experiments/triton` 분리 |
| PR 03 | Rust host runtime + C ABI + CUDA C++ smoke kernel |
| PR 05 | Rust-native config/safetensors/tokenizer loader; Python fallback 금지 |
| PR 06 | cuBLASLt 기본, CUDA C++ primitive, CUTLASS 도입 gate |
| PR 08/09 | native attention backend와 CUDA C++ production 경로 |
| PR 15 | Triton prototype → native port, CUTLASS/custom CUDA escalation, NVRTC 지연 |
| PR 16 | Python-free binary/container release gate |
| PR 17 | Python은 calibration·SVD·rotation 등 offline 연구 도구로만 사용 |

---

## 11. 명시적 비목표

- Python을 프로젝트에서 완전히 금지하는 것
- 모든 offline 도구를 Rust로 재작성하는 것
- Triton을 사용하지 않는 것
- cuBLASLt/CUTLASS보다 범용 GEMM을 새로 작성하는 것
- Python Transformers를 운영 fallback으로 사용하는 것

Python은 개발 속도와 비교 신뢰도를 높이는 도구로 유지한다. 다만 그 편의성이 production inference dependency로 전이되지 않도록 경계를 강제한다.

---

## 12. 완료 정의

이 결정이 지켜졌다고 판단하려면 다음이 성립해야 한다.

- [ ] runtime crate dependency graph에 Python/PyTorch/Transformers가 없음
- [ ] release build가 Python 없는 환경에서 성공
- [ ] server가 Python 없는 환경에서 model load와 generation 수행
- [ ] Rust가 `config.json`, `tokenizer.json`, `safetensors`를 직접 처리
- [ ] CUDA kernel은 CUDA C++로 구현되고 C ABI 뒤에 격리
- [ ] dense GEMM 기본 경로가 cuBLASLt
- [ ] CUTLASS/custom CUDA는 profiler와 별도 PR로 도입
- [ ] Triton은 초기 production dependency가 아님
- [ ] Python reference/offline tool 결과는 표준 artifact로 전달
- [ ] Python fallback 없이 native exact/error path가 존재
