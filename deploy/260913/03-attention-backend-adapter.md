# PR 03 — FlashInfer paged attention 실행층 통합

상태: **구현 중 — Rust/native 연결 및 관측 반례 수정, 일반 품질·serving 미검증**. 공통 계약은 [README](README.md)를 따른다.

## 문제와 가설

수작업 attention 변형의 반복 대신 shape별 검증된 backend를 선택할 수 있는 실행층을 만든다.

## 의존성과 변경 위치

선행: 01; 02와 독립 개발 가능.

예상 수정 위치: riley-runtime attention plan/dispatch, riley-cuda C ABI, kernels adapter, benchmarks/correctness. 경로는 착수 시 실제 checkout과 대조한다.

## 하나의 optimization batch

1. Rust/C++ adapter의 stream·workspace·owner 계약을 구현한다.
2. 현재 HND [page,head,token,dim] KV를 연결하고 필요한 변환이 있으면 비용을 명시한다.
3. plan/workspace 재사용과 graph capture를 연결한다.
4. pure decode·prefill 지원 shape를 명시하고 기존 경로 fallback을 유지한다.

## 범위 경계

POD 혼합 자원 정책, FFN 교체, BF16 검사 tolerance의 임의 완화는 제외한다.

## Correctness·수명 계약

현재 직접 probe에서 context1은 같지만 16/398/4096은 FlashInfer 두 backend 모두 Riley와 bitwise 불일치했다. 따라서 drop-in exact 통과로 간주하지 않는다. full logits·greedy·batch invariant·FP32 오차와 사전 정의된 수치 계약을 평가한다. 현행 exact 모드에서는 불일치 backend를 선택하지 않는다.

## 검증과 하드웨어 skip

실제 Q9/KV3/head64/page16/BF16, 비연속 page, ragged 길이, context 경계, graph replay 입력 갱신, mask·KV 격리. primitive 이후 full-model과 serving 검증. 4090 지원 backend 실행; SM90+ 전용 backend는 조건부 skip.

하드웨어 부재만으로 구현을 보류하지 않는다. 미지원 runtime 검사는 이유와 capability를 명시해 skip한다. 실행된 실패를 skip으로 바꾸지 않으며 build·mock·GPU·serving 증거를 구별한다.

## 완료·승격 기준

수치 계약 및 end-to-end 검증을 통과한 지원 조합만 등록한다. eager/graph 일치만으로 채택하지 않는다. README 성능 게이트 적용.

구현 완료, 장비 검증 대기, 성능 승격 여부를 각각 기록한다. PR이 병합 가능하더라도 검증되지 않은 경로는 기본값으로 켜지 않는다.

## 롤백

backend flag로 기존 attention 복귀. 새 workspace는 해당 stream 완료 후 해제한다.

## 연구 근거

[FlashInfer](https://arxiv.org/abs/2501.01005), [FlashAttention](https://arxiv.org/abs/2205.14135). 논문 성능 배수는 Riley의 예상 개선율이 아니다.


## 착수 우선순위 보강

[비동기 시간 예산 분석](../../benchmarks/results/20260913-overlap-headroom/README.md)에 따라 다음 구현은 이 PR의 실제 모델 adapter를 우선한다. PR 02 전체 완료를 기다리는 의존성은 추가하지 않는다. 기존 FlashInfer primitive 호환 검사는 actual-model correctness나 serving 이득의 증거가 아니다. HND/page16 무복사 경로, explicit numerical profile, graph-compatible plan 수명, 기존 exact fallback을 함께 연결한 뒤 검증하며 품질·성능 결과 없이 기본값으로 승격하지 않는다.

## Native adapter 진행 증거 (2026-09-13)

[검증 기록](../../benchmarks/results/20260913-flashinfer-native-adapter/README.md): FlashInfer 0.6.16.post3 CUDA-core unsplit decode를 별도 shared library로 구현했다. HND KV는 복사하지 않고 33,456-byte device metadata를 갱신한다. 1/4/16/32 rows, ragged context, graph replay와 metadata 갱신을 4090에서 검사했다. SM89/90a/100a AOT 빌드는 통과했지만 SM90a/100a runtime 검사는 장비 부재로 skip이다.

Native-only memcheck는 0 errors. Python FlashInfer 포함 memcheck는 cuGetProcAddress_v2 34건으로 실패했고, adapter를 로드하지 않은 import-only 대조군에서도 동일하게 재현됐다. 실패를 통과로 바꾸지 않는다.

현재 CMake·Rust owner·model recorder·server에는 등록하지 않았다. 다음 batch는 (1) pinned optional dependency/build와 명시적 numerical profile, (2) context/stream/extent/lease를 검증하는 owner/recorder 연결, (3) 전체 모델 logits·greedy·batch invariant 및 기존 exact fallback, (4) 동일 workload serving 비교다. 기존 scratch는 layer마다 덮어쓰므로 metadata는 별도 retained buffer로 수명을 보장한다. 이 검증만으로 성능 개선이나 수치 계약 통과를 주장하지 않는다.

## Native 모델·빌드 연결 진행

[후속 빌드 증거](../../benchmarks/results/20260913-flashinfer-model-build/README.md): 헤더 digest로 고정한 optional CMake/Cargo build와 별도 workspace를 받는 30-layer native 모델 진입점을 구현했다. Enabled 전체 archive 빌드 및 같은 build directory에서 disabled로 복귀하는 빌드가 통과했다. 기존 exact entry는 새 backend를 선택하지 않는다. Rust recorder/명시적 profile 연결, 새 full-model GPU 실행 및 serving 비교는 여전히 미완료다.

## 실제 모델 수치 gate 결과

[모델 관측 증거](../../benchmarks/results/20260913-flashinfer-model-observation/README.md): 별도 retained workspace를 검증하는 Rust/native recorder와 explicit experimental factory를 연결했다. Partial 224개 argmax는 reference와 같았지만 32-request 관측은 4,096개 중 16개 불일치했다. 동일 prompt·teacher-forced history의 요청끼리도 서로 다른 argmax를 내어 batch invariance 반례가 확인됐다. Native 모델 partial memcheck는 0 errors다.

**수치 gate는 실패이며 기본값 승격 및 serving 성능 주장은 하지 않는다.** Pure-decode FlashInfer와 mixed 기존 backend 사이 연산 일관성을 다음에 조사한다. 이것은 아직 원인 확정이 아니다. 관측 harness 종료 성공을 수치 계약 통과로 해석하거나 tolerance를 사후 완화하지 않는다.

## Mixed/pure decode 연산 통일 결과

[Stage consistency 증거](../../benchmarks/results/20260913-flashinfer-stage-consistency/README.md): stage 분리 대조군에서는 기존 experimental FlashInfer도 토큰/요청 간 logits 불일치가 사라졌다. Mixed batch의 decode 행까지 같은 FlashInfer 연산을 적용한 v2에서는 원래 mixed scheduler를 유지하면서 4,096/4,096 argmax 일치와 3,968/3,968 요청 간 full-logit 일치를 확인했다. Full32 native memcheck는 0 errors다. Metadata의 final indptr 및 임시 출력 정렬 오류도 실제 실패를 보존하고 수정했다.

Workspace 78,256 bytes, Q/KV 무복사와 decode 출력 scatter 1회/layer가 추가됐다. 이 비용은 serving에서 측정해야 한다. 독립 prompt/free-running 품질 및 serving 비교는 미완료이며 numerical_profile_accepted=false를 유지한다.

사용자 제약: 실제 실행은 Rust → C/C++ ABI → CUDA로 유지하며 Rust ↔ Python serving 흐름은 도입하지 않는다. Python은 빌드 및 오프라인 검증에만 허용한다.

## 독립 free-running 입력 gate

[독립 생성 검증](../../benchmarks/results/20260913-flashinfer-free-generation/README.md): 새로운 synthetic prompt 8개를 4회 반복한 32-request 실행에서 strict greedy equivalence는 실패했다(12/32 sequence, 152/1024 token positions 불일치). 각 backend 내부의 반복 prompt 결과는 같았다. 최초 분기 3곳의 독립 HF FP32 reference는 FlashInfer 선택 2곳, 기존 선택 1곳과 일치했으며, 이것만으로 일반 품질 우열을 판정하지 않는다. Default 승격은 보류하며 별도 자연어 입력에서 baseline/candidate를 같은 reference로 평가한다. Rust↔Python serving 호출은 추가하지 않았다.

## 고정 자연어 screen 결과

[자연어 수치 screen](../../benchmarks/results/20260913-flashinfer-natural-screen/README.md): 사전 고정한 8개 문장, 256개 target에서 NLL은 2.95928943→2.95213500으로 개선됐지만 KL(FP32 || engine)은 0.0007591519→0.0007855655로 증가했다. 사전 기준의 두 지표 모두 baseline 이하 조건은 실패했다. 기준을 완화하지 않고 기본값 승격을 보류한다. 이 작은 표본은 일반 품질 우열의 증거가 아니며, 다른 precision/backend 대안을 동일 계약에서 검토한다. Serving/vLLM 성능 비교는 아직 미측정이다.
