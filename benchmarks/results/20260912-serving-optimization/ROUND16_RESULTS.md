# Round16: fusion-history-v1 serving screening

동일 RTX 4090·SmolLM2-135M BF16·P128/O32·C1 조건에서 새 후보의 serving 개선을 확인했다. 세 비교 각각 3쌍, 각 프로세스 300 retained 요청으로 총 18개 프로세스·5,400 retained 응답을 측정했다. 원본 token 응답·시간·pair·완료 및 복구 증빙 재검증 결과는 `validated-complete`, 오류 0이다. 이는 screening이며 높은 concurrency나 최종 목표 달성 증거가 아니다.

| 오른쪽/왼쪽 비율 | Throughput | TTFT median | TPOT median | E2E median |
| --- | ---: | ---: | ---: | ---: |
| baseline-vllm | 1.033018 | 0.702621 | 1.055588 | 0.980944 |
| candidate-baseline | 1.076774 | 1.000207 | 0.915603 | 0.928244 |
| candidate-vllm | 1.122864 | 0.691610 | 0.966048 | 0.906024 |

비율은 각 비교의 세 paired ratio 중앙값이다. `baseline-vllm`은 baseline/vLLM, `candidate-baseline`은 candidate/baseline, `candidate-vllm`은 candidate/vLLM이다. Throughput은 클수록, latency는 작을수록 좋다.

후보/vLLM throughput은 +12.29% (세 쌍 +12.08~+12.99%), TTFT −30.84%, TPOT −3.40%였다. 후보/baseline은 throughput +7.68%, TPOT −8.44%, TTFT +0.02%로 사실상 동일했다. 따라서 새 후보를 확대 검증할 근거가 생겼다. Accepted Batch7을 아직 교체하지 않는다.

후보/vLLM E2E P95·P99 paired 중앙값 비율은 0.792257 / 0.716670이다. 프로세스당 300개 관측에서 계산한 tail이며 높은 concurrency P99 안정성으로 일반화하지 않는다. TTFT/TPOT는 HTTP token 전달 시각이며 같은 SSE frame의 토큰들은 관측된 도착 시각을 공유한다.

기존 Batch8의 회귀 구간을 근거로 과거 KV 경로 분리·주소 계산 이동·정렬된 BF16 읽기 결합을 한 optimization batch로 적용했다. 이 결과만으로 세 변경 각각의 기여도를 분리하거나 특정 SASS 명령이 원인이라고 주장하지 않는다.

18개 서버의 정리와 Blender 복구가 완료됐다. 복구 후 실제 PID 3081024/3081081/3081259, 포트 9876/9911/9887에서 명령·GUI 환경·start identity·listening을 확인했다. vLLM 종료 로그의 semaphore warning은 관찰됐지만 소유 프로세스 정리 검사는 통과했다.

다음 단계는 더 긴 반복 측정과 높은 concurrency 정확성 문제의 해소, 실제 다중 요청 실행 통합이다. +15% throughput, −10% TPOT 및 높은 concurrency 안정성 목표는 아직 미달성이다.

증빙: [분석 JSON](token-round16-analysis-v4.json), [완료](raw/token-serving-round16/completion.json), [복구](raw/blender-round16/verified.json), [controller 검증](round16-controller-validation.json).

분석 v1~v3 파일은 로컬 경로 매핑 누락 및 새 HTTP schema 연결 오류를 보존한 실패 결과다. 최종 v4는 실제 기록된 schema를 사용하고 원본 측정·receipt를 수정하지 않았다. 분석기 CPU 테스트 11개도 통과했다.
