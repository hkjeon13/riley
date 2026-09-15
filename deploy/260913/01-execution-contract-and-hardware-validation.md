# PR 01 — 실행 backend 계약과 하드웨어 검증 기반

상태: **구현 진행 중**. 하드웨어 preflight 정책과 실제 CUDA metadata 조회 연결을 추가했으며, backend plan identity·instruction/topology probe·검증 harness 통합은 남아 있다. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

장비와 backend가 늘어도 동일한 실행·정확성 계약을 적용하고, 미지원 장비 때문에 개발을 막거나 검증을 통과로 오인하지 않게 한다.

## 의존성과 변경 위치

선행: 없음.

예상 수정 위치: crates/riley-cuda, crates/riley-runtime, benchmarks의 capability·결과 기록 경계. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. device/architecture/instruction capability와 GPU 수·peer topology를 조회하고 backend 요구조건과 대조한다.
2. backend plan identity에 model revision, dtype, KV layout, shape 범위, device 및 workspace 요구를 포함한다.
3. 실행 결과를 pass/fail/skip으로 분리하고 required/observed capability·skip 이유·실제 선택 backend를 기록한다.
4. 현행 exact 경로와 신규 numerical/quantized 경로의 검증 기준을 분리하여 결과를 보기 전에 고정한다.

## 범위 경계

새 attention kernel, 여러 GPU 실제 실행, 모델 지원 확장은 이 PR에 넣지 않는다.

## Correctness·수명 계약

기존 경로의 수치 기준을 약화하지 않는다. capability가 부족하면 명시적으로 거부하거나 검증된 fallback만 선택한다. fallback 성공은 전용 backend 성공이 아니다.

## 검증과 하드웨어 skip

capability 조합 및 fallback 선택, plan identity mismatch, required-GPU CI에서 잘못된 skip을 실패로 처리하는 테스트. 4090 기존 실행 smoke. 없는 SM90/Blackwell/multi-GPU 실행은 skip하고 가능한 compile-only 결과를 별도로 기록한다.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

공통 계약·검증 결과 스키마가 사용되고 기존 backend 동작이 유지되면 기반 PR 완료. 이 PR 자체에 성능 개선 배수를 요구하지 않는다.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

기존 backend 선택으로 복귀하고 새 결과 필드는 호환 가능한 부가 필드로 유지한다.

## 연구 근거

[FlashInfer API](https://docs.flashinfer.ai/api/attention.html), [TRT-LLM attention](https://nvidia.github.io/TensorRT-LLM/features/attention.html). 논문 성능 배수는 Riley의 예상 개선율이 아니다.

## 현재 구현 증거와 실행 방법

`crates/riley-cuda/src/hardware_validation.rs`는 명시된 architecture 목록·선택 GPU 수·메모리 하한을 대조한다. `CudaRuntime::validation_devices()`는 실제 CUDA metadata에서 관측값을 얻는다. peer topology·instruction feature 검사는 아직 연결하지 않았으며 이 결과로 optimized backend를 허용하면 안 된다.

실행기는 `crates/riley-cuda/examples/hardware_validation.rs`다. 예를 들어 Hopper 전용 GPU 검증 명령을 감쌀 때:

```sh
cargo run -p riley-cuda --features cuda --example hardware_validation -- --arch 9.0 --devices 1 -- ./gpu-validation-command
```

`./gpu-validation-command`는 설명용 자리표시자이며 실제 테스트 실행 파일과 인자로 교체한다. `CUDA_VISIBLE_DEVICES`로 선택한 device 집합 전체를 검사하고 자식 프로세스가 같은 환경을 상속한다. `--arch` 반복 지정은 명시적 허용 목록이며 forward binary compatibility를 가정하지 않는다. 필수 장비 CI는 `--required`를 추가한다.

exit 77은 장비 요구 불충족 skip, exit 1은 오류/실패다. CUDA 초기화 오류와 자식 테스트의 실패·exit 77은 skip으로 숨기지 않는다. 실제 실행은 shell 문자열 결합 없이 argument 배열을 전달한다. 로그에는 requirement·observed·mismatch·명령을 남긴다. 자식 프로세스 exit 0은 명령 성공이며 테스트 개수·ignored 수·GPU 실행 여부까지 증명하지 않는다. 최종 harness는 그 상세 결과를 별도로 검사해야 한다.

CPU library 테스트 89개와 runner 테스트 4개가 통과했다. 4090에서 실행기를 통해 실제 CUDA fill 테스트 1개가 통과했다. Hopper 선택적 조건과 GPU 2개 조건은 exit 77, Hopper 필수 조건은 exit 1을 확인했다. instruction/topology probe, plan identity, 수치 계약 통합은 진행 중이다. 이번 결과는 GPU 성능이나 PR01 전체 완료 증거가 아니다.

원격 검증 raw: [GPU runner 결과](../../benchmarks/results/20260913-execution-contract/hardware-validation-gpu.jsonl), [revision·파일 hash](../../benchmarks/results/20260913-execution-contract/manifest.json). 원격은 V56 기반 별도 worktree이며 로컬과 신규 두 소스의 SHA256을 대조했다. 기존 V56 checkout은 clean을 유지했다.

### Peer access 구현 진행

`CudaRuntime::can_access_peer`와 C ABI의 방향별 CUDA Driver 조회를 추가했다. context 생성이나 peer enable을 수행하지 않는다. `--peer-access`는 선택 GPU 집합의 모든 방향을 요구하며, 누락·중복 관측으로 접근 가능 판정을 하지 않는다. native query 오류는 실패로 전파한다. 이 full-mesh 조건은 선택적 검증 조건이며 모든 distributed backend의 필수 topology로 강제하는 정책이 아니다.

CPU library 90개와 실행기 4개가 통과했다. 원격 CUDA 빌드 성공 후 4090 한 장에서 `--peer-access`는 DeviceCount 불충족으로 exit 77을 반환했다. [실행 로그](../../benchmarks/results/20260913-execution-contract/peer-access-single-device.log). 실제 두 GPU 사이 접근 조회·통신은 장비 부재로 미검증이다. instruction capability·backend plan identity·수치 계약 통합은 아직 남아 있다.

### 기존 plan identity 재사용과 numerical evidence

현재 `llama/executor/graph.rs`의 GraphSignature는 model revision, device/runtime/ABI, dtype, geometry, KV/metadata layout, implementation/GEMM/reduction policy, iteration shape를 이미 포함한다. graph registry는 fingerprint만이 아니라 전체 signature를 비교하고 보유 byte 상한을 검사한다. 중복 registry/key를 새로 만들지 않는다. `graph_dispatch_cpu`, `graph_registry_cpu`, `graph_registry_dispatch_cpu`의 22개 테스트가 통과했다. 이는 해당 기존 graph 경로의 증거이며 원격 variable serving 경로 전체에 대한 증거로 확대하지 않는다.

owned decode executor의 evidence label을 실제 DecodeNumericalProfile과 연결했다. canonical과 HF 경로를 모두 `existing`으로 표시하던 부분을 `canonical-v1`/`hf-smollm2-v1`로 구분하며 `vllm-smol-p128-v1`은 유지한다. 연산 및 tolerance는 변경하지 않는다. local riley-cuda CPU 90개 통과, 원격 V56 worktree의 CUDA runtime compile 확인 완료. local/remote 파일 차이 때문에 runtime 변경은 동일한 네 의미 단위로 이식했으며 파일 전체를 덮어쓰지 않았다.

신규 backend instruction 검증과 variable serving 경로의 실행 계약 연결은 아직 남아 있다. PR01 전체 완료나 serving 개선을 주장하지 않는다.

### 첫 구현 batch 검토

누락·중복·선택 GPU 밖의 peer 관측은 잘못된 evidence이므로 optional 검사에서도 실패한다. 접근 불가가 정상 조회된 경우와 장비 수 부족만 optional skip이 가능하다. 수정 후 CPU 94개가 재통과했다. 원격 GPU 증거는 각 manifest의 당시 파일 hash 범위이며, 마지막 CPU 정책 보강을 새 GPU 성능 측정으로 표현하지 않는다.

이 batch는 하드웨어 preflight/peer query/실행기/numerical evidence를 포함한다. PR01 전체 완료와 분리해 커밋하며, 남은 backend capability와 variable serving 계약 연결은 계속 진행한다.
