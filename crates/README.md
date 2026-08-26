# Production crate boundaries

이 디렉터리의 production crate는 아래 일곱 개로 고정된다. 추가 development
workspace member `rustinfer-native`는 non-default이며 production dependency graph에
들어가지 않는다. 모든 crate의 default feature는 비어 있다.

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

`rustinfer-native`는 development-only calibration ABI/parser library와
`rustinfer-native` calibration binary를 소유한다. Binary는 정확히 `cuda` feature를
요구하며, 이 feature는 optional `rustinfer-cuda`, `rustinfer-model`,
`rustinfer-runtime`과 `rustinfer-cuda/{cuda,nvml}`, `rustinfer-runtime/cuda`를
활성화한다. Production crate는 이 development crate에 의존할 수 없다.

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

- `server`: `rustinfer` binary, 정확히 고정된 optional `serde`/`serde_json` HTTP
  직렬화 경계와 POSIX 종료 신호를 동기적으로 처리하는 optional `libc` 경계를
  활성화한다.
- `cuda`: composition root가 scheduler와 runtime 양쪽에 명시적으로 전달하고,
  runtime/tensor → cuda로 이어진다.
- `bench`: non-default native profile evidence producer와 그 고정 JSON
  직렬화 경계를 활성화한다. 실행 binary는 추가로 `cuda`를 요구하며 production
  `rustinfer` artifact에는 포함되지 않는다.
- `experimental`: 아직 구현을 활성화하지 않는 예약 opt-in 경계다.
- `rustinfer-native/cuda`: Python-free calibration producer를 gate하는 명시적
  development feature다. 다음 locked release build만 binary를 만들며, non-default
  package이므로 `--package rustinfer-native` 없이 root build에 포함되지 않는다.

  ```bash
  cargo build --locked --release --package rustinfer-native \
    --no-default-features --features cuda --bin rustinfer-native
  ```

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
