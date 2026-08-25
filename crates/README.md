# Production crate boundaries

이 디렉터리의 일곱 crate만 production Cargo workspace member다. Dependency는
아래 방향으로만 흐르며 모든 crate의 default feature는 비어 있다.

```text
rustinfer-server
  -> rustinfer-scheduler
      -> rustinfer-runtime
          -> rustinfer-model -> rustinfer-tensor
          -> rustinfer-tensor
          -> rustinfer-cuda (optional `cuda`)
      -> rustinfer-core
  -> rustinfer-runtime
  -> rustinfer-model
  -> rustinfer-core

rustinfer-tensor -> rustinfer-core
rustinfer-tensor -> rustinfer-cuda (optional `cuda`)
rustinfer-cuda   -> rustinfer-core
```

| crate | 책임 | 알면 안 되는 계층 |
|---|---|---|
| `rustinfer-core` | 작은 공통 오류·value contract | CUDA, model, scheduler, HTTP |
| `rustinfer-cuda` | native C ABI의 좁은 Rust 경계 | tensor/model/runtime/server |
| `rustinfer-tensor` | tensor shape/layout/view/storage ownership 경계 | model architecture, HTTP |
| `rustinfer-model` | 향후 canonical model IR 경계 | scheduler, HTTP |
| `rustinfer-runtime` | model/tensor/backend orchestration 경계 | scheduler, HTTP |
| `rustinfer-scheduler` | bounded admission과 continuous-batching state 경계 | HTTP representation |
| `rustinfer-server` | binary, HTTP DTO/transport, model/runtime/scheduler composition root | native C ABI 세부 구현 |

`tools/python`, `tools/native`, `experiments/triton`은 workspace member가 아니다.
Python/Triton은 Cargo dependency나 build-script 입력도 아니다. Python 결과는
JSON, safetensors, CSV/JSONL 같은
명시적 artifact로만 production 경계를 넘는다. `python`과 `triton`이라는 Cargo
feature는 금지한다.

## Feature ownership

Root의 default member는 `rustinfer-server` 하나다. 따라서 다음 root 명령의
feature는 server가 소유하고 아래 crate로 명시적으로 전달한다.

```bash
cargo build --locked --release --features cuda,server
```

- `server`: `rustinfer` binary와 정확히 고정된 optional `serde`/`serde_json` HTTP
  직렬화 경계를 활성화한다.
- `cuda`: composition root가 scheduler와 runtime 양쪽에 명시적으로 전달하고,
  runtime/tensor → cuda로 이어진다.
- `bench`, `experimental`: 아직 구현을 활성화하지 않는 예약 opt-in 경계다.

## Panic과 error 원칙

- 예상 가능한 공통 configuration·artifact 오류는 `rustinfer_core::Result`로
  반환한다. `rustinfer-cuda`의 ABI/build metadata API도 호환성을 위해 이 타입을
  유지한다. Device/context/stream/event host-runtime API는 CUDA domain·stage·native
  status를 잃지 않도록 `CudaResult<T>`/`CudaError`를 사용하며, 상위 runtime
  경계에서 공통 오류로 변환한다. Library 입력 오류에 `panic!`을 사용하지 않는다.
- Binary는 오류를 stderr와 non-zero exit status로 변환한다.
- Debug는 `panic=unwind`, overflow check on이다. Cargo test harness도 unwind를
  사용하며, test profile은 별도로 지원되지 않는 `panic` 설정을 두지 않는다.
- Release는 `panic=abort`, thin LTO, codegen unit 1이다.
- C ABI를 가로질러 unwind하지 않는다. Rust/CUDA wrapper는 status나 검증 가능한
  return value를 `Result`로 변환한다.
- 내부 불변식에 대한 panic이 필요하면 불변식과 검증 근거를 가까이 문서화한다.
