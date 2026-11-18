# W2. 비동기 스텝과 CPU/GPU 작업 중첩

작성일: 2026-11-13\
선행 조건: [W1 검증 복구](01-validation-and-release.md), [W4 정확성 기준](04-correctness-and-schema.md)\
종합 계획: [진행 단계·공통 가드레일](README.md)

2026-11-18 사전 확인: [착수 범위·GPU 준비 기록](02-async-step-preflight.md)을 추가했다. 코드 조사·소유권 설계는 CPU에서 진행할 수 있다. 실제 비동기 실행 변경의 검증과 채택은 GPU 기준 측정 재현 뒤에 수행한다. W1 로컬 검사 복구는 완료했고 실제 Linux CI 확인은 남아 있다.

`aws-a100` GPU 7에서 기준선과 Nsight trace를 재현했다. 기존 `after-submit-v1` 옵션의 비교와 GPU 6·7 예약 상태는 [W2 GPU 실행 결과](02-async-step-result.md)에 기록했다.

## 1. 해결할 문제와 근거

SmolLM2-1.7B의 0.1.0 `reason-c32` 처리량은 vLLM 대비 0.968배, TPOT은 1.061배다. 이 모델은 MHA·KV head 32개·head dimension 64로 작은 GQA 모델과 병목이 다르다.

[목표 문서의 프로파일](../goal.md)에 따르면 생성 32행 GPU 스텝은 약 6.1ms이고 attention 2.5ms, gate|up 1.3ms, down 0.73ms 등으로 구성된다. 스텝 사이 GPU 유휴 약 0.45ms가 있고 vLLM과의 TPOT 차이는 약 0.39ms다. 이는 중첩을 조사할 근거이지 유휴 시간을 전부 제거하거나 0.39ms를 회복할 수 있다는 증명은 아니다.

같은 목표 문서의 **2026-11-14 추가 확인**은 이 해석을 좁힌다. nsys 아래 스텝 경계 유휴 중앙값은 476µs였지만, nsys 없이 host trace로 본 직렬 구간은 약 5.8ms 스텝당 100µs였다. 이를 전부 숨기는 이상적 가정도 약 +1.7%(처리량 비율 0.968→약 0.985)의 추정 상한이다. host trace도 관측 비용이 있으므로 이 수치를 새로운 실측 이득으로 취급하지 않는다. 0.45ms 전체 회복을 목표로 구현을 시작하지 않고, trace OFF/host trace/nsys를 구분해 기준선을 다시 확인한다.

동일 바이트의 QKV GEMM 대체와 step budget 변경은 각각 이득이 2% 미만이어서 기존 항목은 구조 한계로 닫혔다. KV chunk 변경도 진단 이득이 작고 출력이 달라 채택하지 않았다. 이 작업은 같은 설정 탐색 대신 호스트 실행 구조라는 새 가설을 검증한다.

## 2. 코드 조사 지도

| 소스 | 볼 부분 | 질문 |
|---|---|---|
| [server engine](../../crates/riley-server/src/engine.rs) | `step`, `sample_iteration`, `sampling_selection_for_plan`, 이벤트 발행 | GPU 완료 전후에 반드시 직렬이어야 하는 일은 무엇인가? |
| [scheduler 실행 어댑터](../../crates/riley-scheduler/src/execution.rs) | Prepared/Downloaded iteration, execute·commit·abort | 실행 중 소유권과 scheduler 상태를 어디서 확정하는가? |
| [scheduler](../../crates/riley-scheduler/src/scheduler.rs) | admission·plan·취소·KV·commit | 새 토큰·EOS를 알아야만 다음 계획을 만들 수 있는 부분은? |
| [batch executor](../../crates/riley-runtime/src/llama/batch_executor.rs) | GPU 제출·토큰/logits/summary 다운로드 | 동기 대기와 복사 범위가 어떻게 다른가? |
| [host trace](../../crates/riley-runtime/src/host_trace.rs) | PLAN/PACK/UPLOAD/SUBMIT/WAIT/DOWNLOAD/SAMPLE/COMMIT | 호스트 구간과 실제 GPU 타임라인이 대응하는가? |

현재 engine은 계획→실행·다운로드→샘플링→commit을 한 스텝에서 수행한다. 활성 프로파일·graph 경로·fallback을 기록해 어느 경로가 실제로 병목인지 확인한다.

## 3. 원인 분리 실험

### W2-1. 기준 프로파일 재현

1. SmolLM2-1.7B `reason-c32`를 승인 빌드와 동일한 모델 revision·KV·문맥·출력 길이로 재현한다.
2. `RILEY_HOST_STEP_TRACE=1`인 서버와 Nsight의 CUDA graph node trace를 사용해 호스트·GPU 구간을 대응시킨다. 이 환경변수는 현재 존재 여부로 활성화되므로 끌 때는 unset한다.
3. 프로파일 없는 동일 부하도 측정해 trace·로그·동기화가 만든 비용을 구분한다. host 평균 구간 합계와 GPU 유휴 실측이 같다고 가정하지 않는다.
4. graph capture/replay/eager 비율, D2H·H2D 크기, 동기화 횟수, OS wake-up, 이벤트 발행 시간을 기록한다.

다음 환경을 전용 실험 서버에 전달한다. GPU·nsys 경로와 서버 옵션은 manifest에서 고정한다.

```bash
export RILEY_HOST_STEP_TRACE=1
# 이 환경을 물려받는 실험 서버를 실행하고 측정한다.
# 측정 후 trace를 끈다.
unset RILEY_HOST_STEP_TRACE
```

### W2-2. 지연 성격에 따라 후보 선택

| 관찰 | 추가 조사 | 개선 후보 |
|---|---|---|
| GPU 종료 뒤 다음 제출까지 오래 걸림 | plan·commit·pack·event 발행의 임계 경로 | 완료 처리 분리, 제출 전 독립 작업 준비 |
| 제출했는데 GPU가 시작하지 않음 | stream dependency·event wait·드라이버 | 불필요 동기화 제거, 필요한 fence 유지 |
| logits·summary 복사 대기 집중 | 실제 필요한 데이터와 다운로드 범위 | 전용 pinned 버퍼, 복사·완료 확인 분리 |
| 유휴가 작고 attention이 지배적 | 문맥별 KV 대역폭·cache | 이 작업 종료 후 attention 별도 후보 |
| 관측을 켰을 때만 악화 | 로그·문자열 변환·lock | bounded 관측·이벤트 처리 비용 개선 |

## 4. 구현 순서와 소유권 계약

### W2-3. 동작을 유지한 제출/완료 경계 분리

먼저 실행 상태를 명시적으로 표현하되 기존과 동일한 위치에서 완료를 기다린다. `Prepared → Submitted → Completed → Sampled → Committed`를 제안 상태로 두고, 실패는 `Aborted`와 완료 여부를 모르는 terminal failure로 구분한다. 이 이름은 현재 존재하는 API·CLI가 아니다.

스텝 ID, 요청 ID, 출력 slot, 버퍼 lease, CUDA event, 오류를 하나의 소유권 묶음으로 유지한다. 제출된 스텝의 handle 때문에 buffer 수명이 끊기지 않게 한다. 이 단계는 출력·오류 경로 동일성을 먼저 검증하며 성능 이득을 요구하지 않는다.

### W2-4. GPU 결과와 독립적인 CPU 작업부터 중첩

- 새 요청 검증·토큰화, 이미 commit된 토큰의 직렬화, 메트릭 요약 등 독립 작업의 중첩 가능성을 측정한다.
- scheduler·KV를 변경하는 스레드와 시점을 명확히 하고 worker·큐를 제한해 backpressure를 유지한다.
- 이전 스텝 버퍼 소비와 다음 스텝 재사용이 충돌하면 bounded 이중 버퍼를 둔다. 크기·시작 시 메모리 비용을 측정한다.
- 첫 후보는 GPU에서 실행 중인 스텝을 하나로 제한한다. 중첩을 위해 미완료 요청을 임의 commit하지 않는다.

### W2-5. 토큰 의존성이 남으면 설계 재판단

동일 요청의 다음 decode는 이전 토큰·EOS·grammar 상태에 의존한다. 단순히 두 스텝을 enqueue하면 올바르지 않다. 독립 작업 중첩만으로 이득이 없으면 다음을 별도 설계한다.

1. GPU greedy 결과를 device의 다음 입력으로 연결하되 scheduler의 완료·취소·출력 slot 계약도 재설계한다.
2. 처음에는 일반 greedy·비구조화 요청으로 제한하고 grammar·logprobs·확률 샘플링·TP는 검증 전 기존 경로를 사용한다.
3. fallback의 대기·상태 정리를 명시하고 batch 전체가 CPU 경로로 바뀌는지도 측정한다.

이 단계는 초기 중첩보다 범위가 크다. 예상 이득과 검증 비용을 검토한 뒤 진행하며 speculative decoding이나 미확정 토큰 외부 발행을 도입하지 않는다.

## 5. 반드시 찾아야 할 실패

| 경계 | 재현 방법 | 기대 결과 |
|---|---|---|
| 제출 직후 취소·연결 종료 | 완료 fence 직전/직후 취소 | GPU가 읽는 KV·host buffer 유지, 완료 후 한 번 회수 |
| 출력 slot 재정렬 | prefill 완료와 decode 혼합·요청 순서 치환 | 다른 요청에 토큰 전달 0건, 기존 routing 검사 통과 |
| EOS·stop와 계획 경쟁 | 즉시 EOS·다중 토큰 stop·보이지 않는 토큰 | 추가 발행 없음, usage·finish 정확 |
| CUDA 완료 여부 불명 | 제출·event 오류·shutdown 타이밍 주입 | 완료 확인 전 재사용·해제·eager 재시도 금지 |
| 이중 버퍼 덮어쓰기 | 작은 큐·느린 소비자·연속 graph replay | CPU 소비·GPU 완료 fence 이후 재사용 |
| prefix 공유·해제 | 공유 block과 취소·eviction 혼합 | writer alias 없음, refcount·KV accounting 보존 |
| grammar/logprobs fallback | greedy·schema·logprobs 혼합 | 문서화한 경로와 출력·확률 계약 유지 |

기존 mixed-stage routing·KV alias·lease 검사를 재사용하고 새 동시성 경계에만 테스트를 추가한다. CPU 시뮬레이션과 실제 GPU 이벤트·메모리 검사를 모두 사용한다. TP=2는 단일 GPU 통과 뒤 검증하고 rank별 shutdown·오류도 포함한다.

## 6. 성능·완료 판정

기준 빌드 대비 후보 효과를 먼저 보고 같은 세션의 vLLM과 비교한다. SmolLM2-1.7B에서 시작해 작은 GQA, Qwen3 norm, 7B, 14B, TP=2로 확장한다.

- 기존 목표: reason-c32 처리량이 vLLM 동률 범위(±3%) 안으로 들어가는지 확인한다. 현재 0.968배를 측정 잡음으로 지우지 않는다.
- 채택: 반복 가능한 주 지표 이득과 기존 회귀·출력 가드레일 통과. 메모리·시작 시간·CPU 비용도 보고한다.
- 원인 확인: GPU 유휴 감소와 end-to-end 개선이 함께 보여야 한다. host trace만 좋아진 후보는 채택하지 않는다.
- 종료: 가드레일을 지키는 독립 후보 두 개가 각각 2% 미만이면 근거와 함께 닫는다.

산출물은 상태·소유권 설계, 임계 경로 before/after, 요청별 출력 비교, 오류·취소 검사, 단독 성능표, 메모리 상한, 채택 여부다. 기존 동기 경로로 복귀할 수 있게 시작하고 unknown-completion 상태에서 실행 중 경로 전환은 금지한다.

## 7. 다른 작업에 넘길 결과

W3에는 prefill/decode 스텝 비용과 상태 변경 경계를, W4에는 변경 전후 동일 입력·토큰을, W5에는 큐·실행 상태·fallback·대기 메트릭과 새 soak 시나리오를 넘긴다. W3와 결합할 때 기준/W2/W3/W2+W3 네 경우를 별도 비교한다.
