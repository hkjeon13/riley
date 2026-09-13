# 두 iteration용 KV 예약 endpoint와 완료 append 폐기

PR02의 두 계획 연결에서 기존 `complete_iteration`을 그대로 재사용할 수 없음을 확인했다. 기존 경로는 전체 예약을 commit하고 terminal 요청의 page를 반환한다. 후행 GPU 작업이 남은 상태에서는 prefix commit과 suffix drain을 분리해야 한다.

## 변경

- `SequenceState::reserved_prefix_table`: 하나의 full-target 예약에서 앞쪽 endpoint만 조회한다. physical IDs는 sequence에서 빌리고 prefix occupancy만 caller scratch에 기록한다. full-target occupancy·logical length·pool accounting을 바꾸지 않는다. 범위·scratch·stale authority 오류는 scratch 변경 전에 거절한다.
- `OwnedBlockTable::copy_reserved_window`: scheduler transport용 두 endpoint의 불변 snapshot을 생성한다. strict하게 증가하는 두 append 길이를 요구한다. 선행 table이 후행 page나 valid count를 노출하지 않으며 reservation 자체는 유지한다. Host transport snapshot allocation은 존재하며 GPU activation 이중화가 아니다.
- `SequenceState::discard_completed_append`: 완료되고 append 범위만 쓴 suffix를 논리적으로 폐기한다. 현재 committed prefix와 그 page는 보존하고 변경된 tail sidecar를 무효화한다. 아직 쓰지 않은 예약용 rollback 및 mutation 범위가 불명확한 poison과 구별한다.

마지막 API는 CUDA fence가 아니다. 호출자는 GPU quiescence와 committed prefix 아래에 쓰지 않았다는 증거를 가져야 한다. EOS/cancel만으로 호출하면 안 된다. 이 전제는 CPU에서 자동으로 증명되지 않으며 후속 runtime ticket/owner 연결이 필요하다.

## 검증

최종 CPU KV 18개, scheduler 38개 통과. 28개 initial/prefix/target 경계 조합에서 prefix commit 중 page 반환 없음, append 폐기 후 prefix page 보존, 다음 append 재사용, 최종 pool allocation-zero를 확인했다. 별도 검사는 prefix table의 full-target 불변성·scratch 거절 무변경, stale nonce 거절, 변경된 tail sidecar 무효화, scheduler endpoint snapshot의 page16→17 경계를 다룬다.

[최종 KV 검사](kv-final.log), [최종 scheduler 검사](scheduler-final.log), [소스 hash](source-hashes.json). 첫 scheduler build의 닫는 brace 누락은 수정했고 실패 로그와 최종 통과 로그를 따로 보존했다. GPU kernel 변경이 없는 host 계약 검사이며 실제 GPU suffix 실행·EOS/cancel 통합·serving 성능 증거가 아니다.

## 남은 연결

`Scheduler::inflight`는 여전히 단일 계획이다. 이번 변경은 예약 endpoint와 transport/settlement 도구이며 두 계획 큐나 execution authority를 이미 구현했다는 뜻이 아니다. 다음에는 최종 target을 공동 예약한 두 plan의 authority, prefix 결과 반영, terminal 결과 보류 및 suffix drain 후 정리, source result/sidecar 수명을 연결한다. 기존 `complete_iteration`의 전체 commit을 앞선 결과 처리에 우회 사용하지 않는다.

기본 serving 동작과 vLLM 비교표는 변경하지 않는다. 실제 두-plan 모델 실행·overlap trace·동일 조건 serving 비교가 완료되기 전 성능 개선을 주장하지 않는다.
