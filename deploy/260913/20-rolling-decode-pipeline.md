# PR20 — Rolling one-step-ahead decode pipeline

상태: **최신 실행 추적·코드 접점 확인 완료 / 구현 미착수**. PR02의 후속 실행 통합이다. [공통 계약](README.md)과 Rust → C ABI → CUDA 경계를 따른다.

## 근거와 목표

[현재 실행 추적](../../benchmarks/results/20260914-cache-residency-profile/README.md)에서 두 decode의 내부 선행 실행은 작동하지만 pair 뒤 GPU 간격 중앙값은 shared1.12ms/unique0.87ms다. 현재 server는 두 결과 fence·sample·commit 이후 다음 pair를 준비한다. 이 CPU 경계를 매 두 token마다 반복하는 구조를 바꾼다. 전체 간격을 CPU 오버헤드나 예상 가속률로 간주하지 않는다. Unique GPU span의83.55%는 prefill/mixed이므로 이 PR만으로 전체 목표를 달성한다고 가정하지 않는다.

[TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/1.2.0/torch/features/overlap_scheduler.html)의 다음 step 선행 실행과 [SGLang](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)의 future-token/event 방식에 근거한다. 라이브러리의 Python scheduler를 가져오지 않는다.

## 하나의 optimization batch

1. **Scheduler reservation과 순서 있는 부분 정산.** `scheduler/decode_window.rs`의 고정 first/second 상태를 진행 중인 append reservation과 committed prefix로 표현한다. 다음 step을 예약한 채 완료된 앞 step만 정산할 수 있어야 한다. GPU가 사용할 table·페이지·cache/off-batch owner는 별도 보존한다. 취소·EOS 후 미발행 suffix는 drain 후 폐기한다. 기존 ordinary/pair API의 동작은 유지한다.
2. **Native/Rust ticket 수명.** `variable_session.rs`와 `graph_buffered_transfer.inc`의 predecessor/successor 참조를 generation을 가진 bounded ring으로 연결한다. 초기 깊이는 현재 step+다음 step의2개다. result 소비 완료와 GPU future-input 소비 완료를 구분하고 둘 다 확인하기 전 slot을 재사용하지 않는다. Cookie/catalog/replay/iteration·owner 검증을 생략하지 않는다. 변경된 capability를 model/session identity에 포함한다.
3. **Rolling dispatch와 streaming.** `engine.rs::try_decode_window` 경계를 바꿔 n+1 실행 중 n 결과 처리·응답·n+2 준비를 진행한다. n+2를 제출하기 전에 n+1의 의존성과 reservation authority를 보유한다. 결과는 검증·정산 후 token 순서대로 발행한다. 여러 token을 끝까지 모아 전송해 평균 TPOT를 낮춘 것으로 보이게 하지 않는다.
4. **Stop/cancel/admission 및 측정.** speculative GPU 실행은 최대 한 step 선행으로 제한한다. stop 이후 suffix는 발행하지 않고 페이지를 drain 전 회수하지 않는다. 대기 prefill·새 요청·비greedy·출력 한도·page pressure 경계에서는 명시적으로 pipeline을 drain하고 일반 scheduler로 복귀한다. per-token dispatch/complete/publish 시점과 drain/fallback 이유를 기록한다.

## 착수 시 확인할 실제 계약

기존 `DecodeWindowState`의 tickets[2], `issued_successor`, `confirm_decode_window_commit`와 scheduler pending successor는 pair 전체 정산을 가정한다. Native future resolve는 predecessor ticket뿐 아니라 실제 result 저장소 재사용 시점을 보장해야 한다. 단순 배열 확장·result validator 우회는 금지한다. `SequenceState`가 reservation의 일부 commit 후 나머지 소유권을 유지할 수 있는지 먼저 확인하고, 불가능하면 해당 primitive를 같은 batch에서 추가한다. 단계별 내부 커밋은 가능하나 전체 통합 전 성능 개선·PR 완료로 보고하지 않는다.

## 검증·승격

Host 상태 검증: 1/2/3개 이상 연속 step, 15→16→17 page 경계, 마지막 token, EOS/stop, cancel 도착 시점, slot generation 재사용, off-batch cache reader, 실패·poison·부분 제출·부분 완료·drain. 이벤트 미완료를 timeout만으로 완료 처리하지 않는다.

GPU: 기존 ordinary/pair와 실제 모델 logits 및 greedy 출력 일치; bounded native memcheck/racecheck; staged fault injection과 취소 후 재사용. Full-model racecheck로 원격 OOM을 반복하지 않는다. Multi-GPU는 모든 참여 device의 완료가 확보돼야 retirement 가능하도록 계약을 둔다. Hopper/Blackwell 장비가 없으면 해당 runtime만 이유를 남겨 skip하며 CPU/build와4090 테스트는 실행한다.

Serving: 동일 모델·하드웨어·workload·720MiB KV의 prior/new/vLLM, shared/unique, C8/C32 및 active32 초과 대기 요청 축. Warmup과 역순 반복, throughput·TTFT·TPOT·E2E P95/P99에 더해 실제 token inter-arrival P95/P99와 오류율을 보고한다. 개선 여부는 serving으로 판단하며 profiler idle 감소만으로 승격하지 않는다. 취소·짧은 출력·prefill admission 지연이 나빠지면 원인 및 선택 범위를 기록한다. 장시간/open-loop 검증 없이 안정성 목표 완료로 표시하지 않는다.

## 범위와 롤백

Greedy dense path의 rolling execution에 한정한다. Draft 모델 speculation, GPU 무한 scheduler, 임의 window 길이, 모든 모델·수치 backend 지원은 별도다. 이는 전체 serving 목표의 대체 기준이 아니다. Opt-in 정책으로 연결하고 rollback은 신규 요청부터 기존 pair 정책으로 전환하되 진행 중 ticket은 drain한다.
