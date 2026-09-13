# Prepared future window: remove duplicate descriptor work

[Host phase profile](../20260913-host-phase-profile/README.md)에서 드러난 중복 future descriptor 준비를 두 개의 연관된 변경으로 제거했다. 새 paired는 이전 paired보다 serving throughput이 **+3.10%** 개선됐다. Single과는 사실상 동률이며 vLLM 대비 **-10.79%**이므로 기본값은 single로 유지한다. 최종 목표는 미달성이다.

## Optimization batch

1. **검증된 소유 객체 전달**: `PreparedFutureWindow`가 first/second expectation과 그들로부터 검증·생성한 packet/reference를 함께 소유한다. getter는 immutable borrow만 반환하고 내부 분해는 runtime crate로 제한한다. Scheduler가 이미 만든 bytes를 execution adapter가 버리던 동작을 제거하고 runtime으로 그대로 이동한다. Runtime은 현재 owner·issued cookie·pure decode rows·retention 조건을 다시 확인한다. 임의의 expectation을 다른 cached packet과 섞는 API는 제공하지 않는다. 기존 `submit_decode_window`는 compatibility wrapper로 유지한다.
2. **검증된 borrow로 encoding**: 내부 `CheckedExpectation`은 검증된 expectation의 immutable borrow를 유지한다. Future successor structural validation 직후 이 proof로 encode하여 동일 전체 검증을 반복하지 않는다. 외부 `encode_into`는 여전히 전체 validation을 수행하며 잘못된 extent/authority를 buffer 변경 전에 거절한다. GPU result validation, first-stop/suffix discard, cancellation, commit/publication 순서는 변경하지 않았다.

## 실제 serving 비교

RTX 4090, SmolLM2-135M BF16, 동일 checkpoint/tokenizer/natural corpus, C32, 입력 16/128/398·출력 32/64/128 tokens. Lane마다 새 server와 warmup 192 + retained 768개. 순서는 `single → 이전 paired → 새 paired → vLLM → vLLM → 새 paired → 이전 paired → single`이다. 이전 paired binary는 최적화 전 계측 binary와 SHA256을 대조했으며 **이번에는 양쪽 모두 시간 계측을 껐다**. CUDA Graph/node profiler도 사용하지 않았다.

Throughput과 P50은 두 반복의 중앙값, P95/P99는 두 반복 중 더 큰 값이다. Latency 단위는 ms다.

| 경로 | Output tok/s | TTFT P50 | TPOT P50 | E2E P95 | E2E P99 |
|---|---:|---:|---:|---:|---:|
| single | 10,338.63 | 10.142 | 2.893 | 397.394 | 492.067 |
| 이전 paired | 10,036.53 | 10.886 | 2.952 | 404.735 | 476.177 |
| 새 paired | 10,347.99 | 10.874 | 2.834 | 396.290 | 509.770 |
| vLLM 0.27.1 | 11,599.06 | 20.096 | 2.312 | 347.660 | 474.229 |


두 반복 각각의 새/이전 paired throughput 변화는 +2.71% / +3.50%다. TPOT P50도 이전 paired 대비 약 4.0% 감소했다. 하지만 새 paired의 E2E P99 최대값은 이전 paired보다 컸고, single을 안정적으로 능가했다고 볼 근거도 없다. 단순히 이번 code 변경이 tail 증가의 직접 원인이라고 단정하지 않는다. 공유 호스트의 짧은 exploratory screen이며 긴 soak나 open-loop SLO qualification이 아니다.

vLLM의 이번 결과를 이전 날짜/환경 수치와 직접 이어 붙이지 않는다. 모델·길이·client 조건은 맞췄지만 KV capacity와 backend별 BF16 수치 경로는 다르다. 비교에 사용한 정확한 argv/env, binary/model/workload/client hash는 [preparation](evidence/prepared-window-c32-v1/preparation.json)과 lane별 launch JSON에 있다. [전체 반복별 값](evidence/prepared-window-c32-v1/completion.json), [P95/P99 포함 집계](comparison.json)를 보존한다.

## Correctness / validation

- [CPU 검사](tests.json): descriptor 51 passed / 1 ignored, scheduler 44 passed. 모든 byte 오염, stale replay, foreign page, slot/owner/cookie 등의 기존 rejection 검사를 포함한다. Immutable prepared binding 변경을 거절하는 Rustdoc compile-fail E0594 검사 1개도 통과했다.
- [실제 CUDA release build](evidence/phase-build.log), [exit 0](evidence/phase-build.exit). Host typecheck를 GPU 실행 증거로 대체하지 않았다.
- [GPU/HTTP smoke](evidence/prepared-window-smoke-v1/completion.json): single/새 paired 각각 warmup 96 + retained 96개가 기존 token/text/prompt/finish reference와 일치했다. 두 경로에서 stop-string 각 96개의 token/text/finish/usage 일치, 연결 취소 각 32개 이후 retained reference 일치도 확인했다. 새 paired는 실제 253개 window, 최대 폭 32를 실행했다.
- Full serving 8개 lane의 retained 6,144개에서 HTTP/SSE 오류 0건. Riley 6개 lane의 warmup+retained 5,760개는 reference와 모두 일치했다. vLLM retained token/text의 Riley reference exact는 1,121/1,536이며, 모든 경로의 prompt·finish·출력 길이는 같았다. Cross-engine bitwise 동일성이나 모델 품질 qualification 완료를 주장하지 않는다.
- 모든 최종 server exit 0, 측정 후 GPU compute process 없음. vLLM 종료 시 기존 resource_tracker semaphore 경고는 남았으며 장시간 안정성은 미검증이다.
- [소스 snapshot](source-hashes.json) 6개 변경 파일을 [원격 readback](remote-source-readback.json)과 대조했다. Smoke와 serving candidate binary hash도 일치한다.
- [Manifest](evidence/manifest.json)의 보관 hash를 모두 대조했다. 압축 request 자료는 frames를 제외한 token/text/usage/checks/arrival timestamp를 유지한다. 원본 full frames는 manifest의 persistent remote 경로에 남아 있다.

## 결정과 다음 범위

이 batch는 profiler 결과로 발견한 CPU 회귀를 줄였으며 opt-in paired 구현에 유지한다. 기본값 승격, vLLM 동등/우위, 목표 +15% 달성으로 표시하지 않는다. C16 및 Hopper/Blackwell/multi-GPU는 이번 batch에서 새로 측정하지 않았다. 새로운 CUDA kernel이나 하드웨어 전용 경로는 추가하지 않았다.

다음은 native execution/transfer/wait와 GPU kernel 비용을 구분하고, 기존 연구 계획의 GPU execution/attention 영역에서 더 큰 개선 여지를 평가하는 것이다. 현재 host profile의 native 시간을 순수 GPU kernel 시간으로 해석하지 않는다. Future construction은 이제 adapter에서 한 번 수행되므로 runtime `future_prepare=0`을 전체 descriptor 비용이 0이라는 뜻으로 읽지 않도록 분석기도 수정했다.

재실행: [controller](../../analysis/paired_decode_serving_screen.py)에 `--prior-paired-binary /data/riley-serving-260913-recovery/riley-before-descriptor-batch --warmup 192 --retained 768 --concurrency 32`를 지정한다. 원복/비교용 이전 binary를 보존하며, 사용 경로 변경은 새 manifest에 기록한다.
