# PR20 — Rolling one-step-ahead decode pipeline

상태: **KV·scheduler·runtime·server 통합 및 C8/C32/C64 serving screen 완료 / 기본값 승격·전체 qualification 미완료**. PR02의 후속 실행 통합이다. [공통 계약](README.md)과 Rust → C ABI → CUDA 경계를 따른다.

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


## 구현 진행 — 예약 수명 기반

`SequenceState::extend_reservation`을 추가했다. 기존 pending append의 물리 페이지는 유지하면서 목표 길이를 늘리고 detached/pending nonce를 함께 갱신한다. OOM·nonce 소진·중간 page-generation 실패 시 기존 예약과 table을 보존한다. 이미 완료된 prefix는 기존 `commit_prefix`로 정산하며 suffix를 계속 보유한다. Dispatch된 GPU 작업은 이전 table의 immutable snapshot을 사용해야 하고 새 suffix 실행은 의존성을 따라 순서화해야 한다. 이 host 연산은 GPU fence나 token publication을 수행하지 않는다.

연속48회 진행과15/16/17·31 page 경계, 오래된 권한, 부분 할당 후 실패·재시도, poison 후 prefix 보존, cache/off-batch reader 수명을 검사했다. [검증 기록](../../benchmarks/results/20260914-rolling-reservation/README.md)에 실제 실행 결과를 기록한다. 아직 scheduler rolling 정산, result ring, native future-input 소비 수명 및 server streaming 통합은 미완료이며 새로운 serving benchmark는 실행하지 않았다.

다음 scheduler 연결에서는 `publish_committed_item`의 terminal 처리에 주의한다. 현재 helper는 stop/length/cancel이면 sequence를 닫을 수 있어 successor 실행 중 그대로 호출하면 안 된다. 앞 step의 token 발행과 GPU에서 사용하는 suffix의 retirement를 분리해야 한다. Batch 중간 정산 오류에서도 진행 중 페이지를 reclaim하지 않도록 실패 상태를 보유하고 전체 drain 이후 정리한다. 단순한 pair API 반복이나 terminal 경로 우회로 완료 처리하지 않는다.


## 구현 진행 — Scheduler rolling 전이

[Scheduler 검증 기록](../../benchmarks/results/20260914-rolling-scheduler/README.md): nonterminal 앞 token만 정산하는 `complete_decode_window_prefix`와 실행 중 successor를 새 first로 삼아 뒤 step 하나를 예약하는 `roll_decode_window`를 구현했다. 첫 결과는 보존하고 마지막 drain에서 중복 발행하지 않는다. 일부 row만 연장된 OOM에서도 모든 예약을 유지하며 quiesced abort로만 회수한다. 앞 결과 발행 이후 NotDispatched rollback은 거부한다. Waiting/출력·context limit에서는 drain 후 일반 scheduler로 복귀한다.

24회 연속 rolling·page 경계·descriptor authority·취소·terminal·부분 할당 실패·출력 중복 방지를 host fixture로 검증했다. GPU ticket 승격과 server의 실제 호출은 아직 연결하지 않았다. 기존 `try_decode_window`의 callback만 바꾸는 것으로 완료할 수 없다. Immutable scheduler authority의 borrow 종료, retained successor cookie/replay 승격, native 두 slot의 재사용 시점, 결과별 publication을 함께 연결하고 model/serving gate를 실행해야 한다.


## 구현 진행 — Runtime ticket 승격·실제 모델

[Runtime/GPU gate](../../benchmarks/results/20260914-rolling-runtime/README.md): 완료된 앞 replay만 승인하고 live successor ticket·cookie를 유지하는 승격 API를 구현했다. 새 successor만 fresh cookie/replay로 준비하며 준비 실패는 owner를 poison하고 close/drain 전 KV 회수를 막는다. 기존 native2-slot ring을 그대로 사용한다.

4090 실제 모델4개 요청에서 rolling21회, 총96개 생성 token이 직렬 실행과 같았다. 취소 mode는[5,24,24,24] token이 각각 직렬 prefix와 일치하며, 잘못된 cookie를 주입한 별도 mode도 GPU drain 후4개 요청 정리·allocation0을 확인했다. 이는 greedy 모델·수명 검사이며 full-logit·HTTP serving·C32·장시간 qualification은 아니다. 첫 private-type build 오류는 수정했고 GPU 실패를 skip으로 바꾸지 않았다.

다음 server 연결은 worker tick 사이에 rolling state를 유지해야 한다. 각 앞 token 정산 후 새 successor를 제출하고 해당 token event를 반환한다. 모든 rolling step을 한 호출에서 반복한 뒤 출력을 한꺼번에 반환하면 streaming 지연과 TPOT 측정이 왜곡되므로 금지한다. Prefix를 먼저 발행한 drain fallback은 첫 결과를 보존한 채 다음 tick에서 suffix를 정산한다. 기존 pair 경로는 유지하고 opt-in serving 비교 후 승격을 판단한다.


## Server 통합 및 milestone 결과

앞의 구현 진행 절은 단계별 당시 기록이다. 현재 server는 worker tick 사이에 window와 첫 결과·발행 상태를 유지하며 앞 token event를 매 tick 반환한다. `RILEY_ROLLING_DECODE=1`로만 활성화한다. Rust → C ABI → CUDA 경계를 유지한다.

[전체 비교표와 검증](../../benchmarks/results/20260914-rolling-serving/README.md): C8 shared throughput은 vLLM 대비13.78% 높지만 unique 및 C32/C64는 미달이다. C32 shared의 이전 Riley 대비 throughput12.52% 개선과 token interval P99의4.53% 악화를 함께 기록했다. C64는 client64/active32 대기 요청 조건이다. 세 조건 총12288개 retained 요청이 protocol-valid이고 Riley9216개는 이전 binary의 출력과 일치했다. Server105개·CUDA runtime9개 검사와 release build가 통과했다.

기본값 승격은 보류한다. Full-logit·광범위 모델 quality·장시간/open-loop·추가 fault injection은 남아 있다. Multi-GPU/Hopper/Blackwell runtime 검사는 해당 장비 부재로 미실행이다. 다음 영역은 기존 profile의 unique prefill/mixed 지배 및 high-concurrency GPU 실행 비용을 근거로 선택한다.
