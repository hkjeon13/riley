# Serving 재측정의 호스트 시작 조건

상태: 측정 절차 보강. 모델·GPU kernel·성능 성공 기준을 변경하지 않는다.

## 근거

이전 C32 projection CTA 비교에서 unique 역순의 CPU PSI some 비중은 candidate 24.23%, prior 14.77%였다. VLLM unique 첫 실행은 I/O 대기가 더 높으면서도 더 빨랐다. 따라서 I/O 대기 하나를 원인으로 단정하거나 성능을 보정하는 데 쓰지 않는다. 기존 결과는 버리지 않고 non-qualified 증거로 유지한다.

Serving을 실행하지 않은 후속 여섯 관측 구간에서도 I/O PSI full은 8.98–47.62%였다. 해당 구간에서 swap-in/out 증가는 없었다. 읽기 권한이 있는 생존 프로세스의 I/O 증가량에는 다른 프로젝트의 동기화 작업, Android emulator 및 MinIO 등이 관측됐다. 이 목록은 전체 프로세스를 포괄하지 않고 종료된 단기 프로세스도 빠지므로 인과관계나 독점 원인을 주장하지 않는다. 다른 작업은 중지하지 않는다.

## 변경 묶음

1. `host_pressure_probe.py`: PSI, memory/swap/major fault 변화, 읽을 수 있는 프로세스의 CPU/I/O counter를 수집한다. PID 재사용은 start time으로 제외하고 명령 인수·환경 변수는 저장하지 않는다. 분석용이며 timed serving 안에서는 실행하지 않는다.
2. `host_quiet_gate.py`: 다음 실행에 사전 선언하는 시작 조건. 2초 구간 세 개 연속으로 CPU some ≤5%, I/O full ≤5%, memory full ≤0.5%일 때 시작한다. 이는 실험 운영상의 기준이며 하드웨어 성능이나 완전한 격리를 보장하는 임계값이 아니다.
3. Serving controller에 `--host-quiet-timeout-seconds` 옵션을 추가한다. 0은 기존 동작이며, 새 비교에서는 모든 엔진의 warmup/retained 전에 동일한 값으로 켠다. 조건 판정은 GC 정리 이후, timed phase 이전이다. gate source와 각 샘플을 보존한다.

## 검증 및 해석

기존 controller의 모델·binary·GPU·메모리 사전 조건은 유지한다. 게이트 판정에는 throughput/latency를 입력하지 않는다. 시간 내 조건이 충족되지 않으면 시도 전체를 중단하고 이미 수집된 결과도 보존한다. 느린 lane만 골라 재실행하거나 선택적으로 통계를 합치지 않는다.

시작 조건 통과 후에도 부하는 바뀔 수 있다. 기존 before/after 호스트 기록과 실행 순서별 성능을 함께 공개하며, 시작 조건 통과만으로 결과를 qualified로 표시하지 않는다. 새 조건이 warmup 이후 cache residency에 영향을 줄 수 있으므로 대기 시간을 기록하고 과거 절대 수치와 직접 합치지 않는다.

실제 재측정은 동일 후보·prior·vLLM으로 C8/C16/C64 및 unique 재현성을 확인한다. 충분히 안정적인 비교가 나오기 전에는 projection CTA를 기본 경로로 승격하지 않는다. 새 시작 조건이 충족되지 않는 동안 GPU 측정을 반복하거나 다른 서비스를 중지하지 않는다.
