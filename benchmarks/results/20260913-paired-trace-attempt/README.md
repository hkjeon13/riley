# Paired serving trace: collector blocked before workload

직전 [serving screen](../20260913-paired-serving-recovery/README.md)은 실제 성능 근거이며 paired 회귀 판단을 유지한다. 이번에는 그 원인을 구분하려 했지만 **유효한 GPU trace를 확보하지 못했다**. 새로운 throughput·latency·병목 비율을 제시하지 않는다. Serving binary와 기본값은 변경하지 않았다.

## 수행 결과

- Nsight Systems 2024.6.2.225의 `--version` 자체가 약 2분 동안 I/O 대기 상태였다가 정상 종료했다. 실제 PID 268698의 D 상태를 확인했다.
- [시도 1](paired-trace-v1.log): profiler PID 284787이 시작됐지만 서버 readiness가 90초 내 성립하지 않아 종료했다. 새 GPU/HTTP 요청 결과가 없다.
- [시도 2](paired-trace-v2.log): readiness 한계를 240초로 늘렸으나 [Nsight 내부 process probe](attempt-2/single.log)가 `Failed to probe the process (sync). Timeout: 2 sec`로 종료했다. [Receipt](attempt-2/single-receipt.json)의 exit는 1이고 당시 관찰된 프로세스 그룹 peak RSS는 약 93 MiB다. 이 예전 그룹 측정은 detached agent를 포함하지 않으므로 전체 수집기 메모리 상한 증거가 아니다.
- 별도 최소 검사 `nsys profile --trace=none --sample=none --cpuctxsw=none /bin/sleep 1`도 45초 + TERM 이후 5초 제한 안에 끝나지 않아 [exit 137](nsys-launcher-check.exit)로 정리됐다. `timeout`의 강제 종료이며 OOM이라고 해석하지 않는다.
- 같은 Riley release binary의 `--help`는 [0.03초, exit 0](riley-launcher-help.time)이었다. 따라서 서버의 decode 경로 실행 실패로 볼 근거가 없으며 최소 수집기 실행부터 복구해야 한다.
- 관찰 시 `/proc/pressure/io`의 full avg10이 약 65–73%였다. 수집기/호스트 문제는 관찰됐지만 디스크·Nsight·driver 중 직접 원인은 아직 확정하지 않았다. 다른 서비스나 GUI를 중단하지 않았다.

Nsight가 `--session-name profile-284787` / `profile-315441`의 agent를 별도 session으로 남기는 것도 관찰했다. 두 agent를 해당 작업의 세션 identity로 대조했다. 하나는 확인 중 자연 종료했고, 남은 PID 291431만 SIGTERM으로 종료했다. 최종 재확인에서 Nsight/trace controller와 GPU compute process는 남지 않았다.

## 다음 수집과 분석 도구

[수집기](../../analysis/paired_decode_serving_trace.py)는 single/paired 각각 실제 natural HTTP 요청 96개를 C32로 실행하고 reference 전체 일치를 요구한다. GPU kernel은 node 단위로 수집한다. 데이터는 profiler overhead와 client pacing을 포함하므로 serving benchmark로 사용하지 않는다.

초기 구현의 단순 process-group watchdog을 수정해 **정확한 Nsight detached session agent와 그 자손**도 추적한다. 500ms 간격으로 RSS를 관찰하여 8 GiB 초과 또는 360초 초과 시 작업 소유 프로세스만 종료한다. 이는 kernel/cgroup의 엄격한 메모리 제한이 아니며 sampling 사이의 overshoot를 막는 보장은 아니다. 신호 전 PID start time을 다시 대조하고 일반 종료 뒤의 detached agent도 정리한다. SQLite export는 별도 120초 timeout이며 RSS watchdog 범위 밖이다. 최종 수정된 controller는 Python 구문 검사와 소유권 fixture 검사를 통과했지만 실제 Nsight 성공 실행은 아직 없다.

[분석기](../../analysis/paired_decode_trace_analysis.py)는 CPU graph launch와 GPU activity correlation을 전부 대조하고, 한 stream의 겹치지 않는 graph들만 분석한다. 중간 80% launch count에서 future resolver를 가진 graph를 식별하여 `within_pair`, `after_pair`, `ordinary` 간격을 구분한다. 이와 함께 graph span, runtime API 비용, kernel별 합계를 출력한다. 완전한 실제 trace를 받기 전에는 이 분류로 성능 결론을 내리지 않는다.

검증: 새 분석기/프로세스 소유권 fixture 3개, 기존 graph gap accounting 검사 4개 통과. 연속 successor 등 잘못된 pair 배치는 거절한다. 외부 session agent와 그 자식은 작업 소유권에 포함하지 않는다.

## 다음 조치

호스트 I/O 상태가 회복되면 먼저 GPU를 쓰지 않는 최소 Nsight launch가 정상 종료하는지 확인한다. 그것이 통과해야 실제 single/paired trace를 다시 수집한다. Trace가 확보되면 pair 내부에서 줄어든 대기와 pair 종료 후 증가한 비용을 분리하고, 그 증거로 예약/metadata 재사용·제출/완료 분리 또는 GPU backend 변경 중 다음 optimization batch를 선택한다. 이번 실패를 근거로 serving 코드를 추측 수정하지 않는다. 목표는 계속 미완료다.

## 참고

[NVIDIA의 같은 process-probe 오류 진단 사례](https://forums.developer.nvidia.com/t/failed-to-start-pfofiling-on-k8s-pod-with-failed-to-probe-the-proces-message/297076)에서도 작은 실행으로 환경 문제를 분리한다. 해당 사례의 GPU Manager 원인이 이 호스트에도 적용된다는 증거는 없다. [공식 Nsight User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)는 수집/launch 동작의 참고 자료이며, 현재 설치 버전의 실제 실행 로그를 우선한다.
