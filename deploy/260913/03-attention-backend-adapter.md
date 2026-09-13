# PR 03 — FlashInfer paged attention 실행층 통합

상태: **구현 중 — Rust/native 연결 및 관측 반례 수정, 일반 품질 미통과·실험적 serving 측정 완료**. 공통 계약은 [README](README.md)를 따른다.

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

## 실험적 serving 비교 결과

[동일 바이너리 V7 / FlashInfer / vLLM 비교](../../benchmarks/results/20260913-flashinfer-serving-screen/README.md)를 완료했다. C32 natural 확장 교차 screen에서 FlashInfer throughput은 V7 대비 +7.24%, vLLM 대비 −5.36%이며 median TPOT는 vLLM보다 17.82% 길다. TTFT P99는 V7보다 나빠졌다. 품질 gate 실패는 유지하며, loopback 전용 `flashinfer-smol-experimental-v2`는 명시적 진단 선택지다. 지원·승격 backend 등록이나 기본값 변경이 아니다. 실제 serving 이득은 있지만 PR03 및 전체 목표의 완료 조건은 충족하지 못했다.

## Native paged-prefill adapter 진행

[Prefill native 검증](../../benchmarks/results/20260913-flashinfer-prefill-native/README.md): 기존 HND KV 및 packed-query metadata를 FlashInfer causal paged-prefill kernel에 직접 연결했다. 32요청·1,024 query row·context4096·비연속 page·graph replay를 검증했다. 원본 kernel의 output shared-memory 단계에서 racecheck warning을 확인했고, 고정 원본을 검증한 별도 빌드 overlay에 warp barrier 하나를 추가해 expanded probe의32 warnings를0으로 만들었다. 설치된 원본은 변경하지 않았다. 868,032 출력 값은 보정 전후 bitwise 일치하며 patched memcheck도0 errors다. SM89/90a/100a AOT 통과, 후자의 runtime은 장비 부재 skip이다.

모델 통합 시 이 검증된 overlay를 prefill object에 강제하고, mixed의 decode row는 선택된 decode backend로 유지해야 한다. 현재 raw adapter는 모든 supplied query를 처리하므로 그대로 mixed 전체에 붙이지 않는다. Rust owner·graph identity·stage routing 및 기존 full-model 품질 gate·serving 비교가 남아 있다. 합성 attention 오차0.01 기준은 기존 자연어 품질 gate를 대체하지 않는다.

## Prefill optional archive 연결

[빌드 연결 검증](../../benchmarks/results/20260913-flashinfer-prefill-build/README.md): 원본 dependency lock과 GPU 검증된 patched header digest를 모두 유지하는 생성 overlay를 CMake에 연결했다. Prefill만 overlay를 사용하며 decode는 원본을 유지한다. 전체 archive, header/receipt 변경 거부, 동일 디렉터리 enabled→disabled→enabled 빌드를 검증했다. Archive-linked GPU probe의868,032개 값은 이전 patched library와 bitwise 일치하고 memcheck/racecheck 모두0이다. Cargo native 입력 및 내부 ABI를 등록했지만 model recorder/serving 선택에는 아직 연결하지 않았다. 다음 구현은 retained workspace와 mixed prefill-only routing이며 모델 수치·serving gate가 여전히 필요하다.

## Mixed prefill-only metadata 준비

[Routing 검증](../../benchmarks/results/20260913-flashinfer-prefill-routing/README.md): 별도 native planner가 V7 요청별 stage를 기준으로 prefill tile만 등록한다. Packed offset은 유지하며 decode 출력은 건드리지 않는다. 1-token prefill과 decode를 구분하고, all-decode·invalid stage·잘못된 decode 길이·오류 후 graph 재사용을 포함한10 replay를 검증했다. 기존 prefill849,600개 값 bitwise 일치, matched decode18,432개 값 미변경, memcheck/racecheck0이다. 모델 owner/recorder 연결 전 단계이며 기존 decode 품질 gate 실패와 serving 기본값은 그대로다.

## Prefill-only 전체 모델 연결과 수치 gate

[전체 모델 결과](../../benchmarks/results/20260913-flashinfer-prefill-model/README.md): 별도 retained workspace·native recorder·Rust factory·graph identity를 연결했다. Mixed의 prefill만 FlashInfer로 처리하고 pure/mixed decode는 기존 V7을 유지한다. 30-layer full-model 실행과 native memcheck0 errors, 종료 후 allocation0을 확인했다. 기존 V7 자연어 logits는 이전 baseline과 bitwise 일치한다.

Strict free generation은32요청 중8요청·1,024토큰 중56토큰 불일치로 실패했다. 반복 prompt invariance는 통과했다. 고정 자연어256 target의 NLL은2.95928943→2.95632435로 낮아졌지만 KL은0.0007591519→0.0008375119로10.32% 증가해 기존 사전 gate도 실패했다. 서버 선택/기본값 승격은 하지 않는다. 다음 검토는 동일 intermediate Q/K/V의 수치 차이와 정밀도/backend 대안이며, 기존 gate를 완화하거나 primitive 통과로 대체하지 않는다. 실제 serving 비교와 최종 성능 목표는 여전히 미완료다.

## 동일 입력 정밀도 대조

[정밀도 screen](../../benchmarks/results/20260913-prefill-precision/README.md): 기존 attention body, FlashInfer BF16, 동일 BF16 입력값을 FP16으로 계산 후 BF16 출력으로 되돌리는 별도 native 실험을 비교했다. 입력값은 float32 dump hash로 동일함을 확인했다. 3개 Q/K scale에서 FP16 실험의 FP64-reference RMSE는 FI BF16보다0.90~24.41% 낮았다. 새 FP16 primitive memcheck/racecheck는0이다. 이는 합성 attention 오차 감소이며 serving speedup이나 모델 품질 통과가 아니다. 이전 full-model KL 실패는 유지한다. 다음 batch는 BF16 KV 저장을 유지하는 명시적 변환, FP16 범위/정밀도 계약, 독립 graph identity 및 full-model gate를 함께 평가하고 변환 비용까지 serving에서 측정한다.

## BF16 잔차 보정과 실제 serving 판정

[보정 batch 결과](../../benchmarks/results/20260913-prefill-residual/README.md): FlashInfer의 rounded-probability denominator를 FP32로 유지하고, BF16 probability의 반올림 잔차를 추가 P×V MMA로 누적하는 후보를 별도 빌드에서 평가했다. BF16 Q/K/V 저장을 유지한다. 고정 자연어 NLL·KL 사전 screen은 모두 통과했지만 독립 greedy는148/1,024토큰 차이로 실패하며 general quality는 미승인이다. Whole-model native memcheck0 errors다.

C32 natural·두 역순·engine별1,536 retained 비교에서 V7/후보/vLLM throughput은10,420.8/9,637.2/11,802.2 tokens/s, median TPOT는2.928/3.170/2.385ms다. 후보는 V7보다7.52% 느리고 tails도 악화되어 도입을 보류한다. 기본값은 유지한다. 선택된 primitive는239 registers/thread와32KiB shared memory를 사용하지만 local memory는0이므로, 다른 미사용 변형의 spill을 이번 회귀 원인으로 단정하지 않는다. 다음 attention batch는 query tile/work 분배와 residual fragment 수명을 함께 조정하고 동일 모델 gate·serving 비교를 반복한다. 합성오차 개선만으로 성공을 선언하지 않는다.

## Q16 work distribution batch 판정

[Q16 비교 보고](../../benchmarks/results/20260913-prefill-q16/README.md): query tile·work slot·KV-warp shared-storage 동기화를 함께 조정했다. 수정 후 mixed memcheck/racecheck 및 whole-model memcheck는0이다. 초기 race24건과 Q32 configuration 거부는 실패 증거로 보존했다. 고정 자연어 NLL/KL은 유지됐지만 독립 생성232/1,024토큰 차이, serving reference1,033/1,536 일치로 품질 승격은 불가하다.

동일 C32 natural 두 역순 비교에서 V7/Q128/Q16/vLLM은10,464.6/9,592.5/10,113.7/11,850.6 tokens/s다. Q16은 Q128보다5.43% 빠르지만 V7보다3.35%, vLLM보다14.66% 느리다. 기본값은 유지하며 이 compensation 계열의 추가 미세 tile 조정은 멈춘다. 다음 영역은 PR07/PR19 persistent layer execution feasibility이며, 기존 수치 연산·dependency/scratch ownership·full-model 호출을 함께 다룬 후 의미 있는 serving milestone에서 비교 표를 반복한다. 해당 구조의 성능 이득은 아직 미측정이다.

## 다음 착수: native precision 경로의 모델 검증

[Attention 실행 대안 분석](../../benchmarks/results/20260913-attention-task-costs/README.md)에서 task remapping과 두 softmax 공유안 모두 C16/C32 native 회귀를 보였다. 다음에는 기존 same-input FP16 primitive 근거를 출발점으로, pinned FlashInfer의 실제 mixed-dtype load/compute 경로를 확인한다. BF16 KV 저장·명시적 변환 및 FP16 범위/subnormal 계약·retained owner/graph identity를 하나의 batch로 다루고, 기존 full-model 수치 gate와 serving 비교를 완료한다. 현재 FP16 모델 품질·serving 이득은 미검증이며, BF16 bit reinterpret 또는 사후 tolerance 완화는 허용하지 않는다.


### BF16 저장 / FP16 register 변환 native 진단

[검증 기록](../../benchmarks/results/20260913-prefill-half-register/README.md): 명시적인 Q/K/V fragment 및 probability 변환, FP16 범위 오류 status, 별도 검증 overlay를 구현했다. 일반·mixed oracle, 15개 nonfinite/overflow 주입, memcheck/racecheck를 통과했다. SM90a/SM100a compile 통과, runtime은 장비 부재 skip이다. 기본 backend는 유지한다. 다음은 변환 경계·full-model 품질·status 전달·workspace/graph 계약 통합이며 새 serving 결과는 아직 없다.
