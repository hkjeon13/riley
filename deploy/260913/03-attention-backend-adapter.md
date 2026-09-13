# PR 03 — FlashInfer paged attention 실행층 통합

상태: **구현 중 — standalone native adapter 검증, 모델·serving 통합 미완료**. 공통 계약은 [README](README.md)를 따른다.

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
