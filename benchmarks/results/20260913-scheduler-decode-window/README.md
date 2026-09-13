# Scheduler two-decode window

기존 KV prefix 도구와 future-token wire를 실제 scheduler 계획·authority·settlement에 연결했다. 서버에서 자동 선택하지 않는 opt-in API이며, 두 GPU graph를 실행하는 runtime adapter는 아직 없다.

## 구현

- `plan_decode_window`: 현재 선택된 요청이 모두 decode이고 각 요청에 최소 두 출력이 남았을 때 두 iteration ID를 할당한다. 요청마다 최종 target을 공동 예약하고 첫 plan에는 prefix table만, 두 번째에는 full table과 future-token placeholder를 넣는다. admission/timeout 처리가 필요한 경우와 prefill/마지막 한 출력에서는 기존 planner로 fallback한다.
- `authorize_decode_window`: 두 plan의 ID·입력·slot·endpoint·block table을 실제 scheduler 예약과 대조한다. 두 단계의 immutable scheduler borrow를 함께 유지한다. 기존 단일-plan authority 및 `complete_iteration`은 window를 거절한다. `prepare_wire`는 이 authority에서 서로 다른 cookie와 실제 committed replay를 유지한 두 expectation 및 GPU reference를 만든다.
- `complete_decode_window_after_drain`: 두 runtime 결과의 ID·slot·token 범위를 모두 검증한 후에만 mutation한다. prefix commit 후 선행 stop은 첫 token만 공개하고 suffix를 폐기한다. deferred cancel은 두 출력을 공개하지 않는다. 계속되는 요청은 후행 commit과 두 번째 token을 처리한다. 두 execution이 모두 quiescent하고 runtime의 전체 identity/numerical 검증을 통과했다는 증거는 호출자가 제공해야 한다. API 이름 자체가 CUDA fence는 아니다.
- OOM 시 이미 확보한 앞 요청의 추가 page까지 rollback한다. 다음 ID는 성공한 두 plan에 대해서만 진행한다. abort는 첫 iteration ID를 사용하며 NotDispatched는 두 단계 모두 미제출, DeviceQuiescedMutationUnknown은 두 단계 모두 quiescent임을 뜻한다. outstanding iteration gauge는 window 동안 2를 표시한다. step별 metric은 scheduler 내부에 두 번 기록하고 단일 `IterationUpdates::iteration_metric`에는 두 step을 하나로 위장해 넣지 않는다.

## 검증

최종 로컬 scheduler 44개 검사와 Rustdoc 2개 검사가 통과했다. 원격 CUDA feature host 검사도 44개 모두 통과했으며, scheduler 소스 5개와 최종 로그·exit 파일의 SHA-256을 로컬과 원격에서 대조했다. Rustdoc에는 authority를 보유한 동안 mutable scheduler cancellation이 컴파일되지 않는 검사가 포함된다. 추가 host 검사는 두 token의 정확한 요청별 진행, page16→17, 선행 stop·deferred cancel, 두 번째 출력의 length 종료, 신규 admission fallback, malformed successor의 무변경 거절, abort/retry, 중간 OOM rollback, ragged C32의 64개 출력 및 outstanding gauge 2→0을 다룬다.

[최종 로컬 검사](tests-final.log), [borrow 수명 검사](doc-tests.log), [CUDA feature host 검사](cuda-host-final.log), [소스 hash](source-hashes.json).

초기 테스트 fixture의 chunk/context 설정 오류와 OOM 후 lifetime/high-water 통계까지 원복되어야 한다는 잘못된 assertion은 수정했다. OOM rollback은 live allocation/free 개수와 요청 상태를 복구하며, 실제 발생한 allocation 통계 이력은 유지한다. 실패·수정 결과 로그를 분리해 보존했다.

## 범위와 다음 단계

이는 실제 scheduler 상태 전이의 CPU 검사다. CUDA feature 빌드에서도 host 검사를 실행할 뿐, 두 iteration의 실제 GPU 완료·append-only 접근·exact 모델 결과를 증명하지 않는다. GPU ticket, 두 staging slot, resolver의 status 초기화 뒤/embedding 앞 삽입, 원본 result 및 sidecar 수명, graph fingerprint가 다음 통합 범위다.

현재 완료 API는 두 실행이 모두 끝난 뒤 token을 공개한다. 이 방식의 token 전달 간격과 tail latency는 실제 serving에서 측정해야 한다. 선행 결과를 더 일찍 공개하는 확장은 prefix publication과 terminal page reclaim을 분리해야 하며, 완료 증거를 생략해서는 안 된다.

Throughput·TTFT·TPOT·P95/P99의 새 serving 측정은 없다. 기본 서버 경로와 vLLM 비교표를 변경하지 않으며, PR02 및 전체 성능 목표를 완료 처리하지 않는다.
