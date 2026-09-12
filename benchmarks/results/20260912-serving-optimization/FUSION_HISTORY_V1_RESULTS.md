# Fusion-history-v1 검증 상태

Batch8의 실제 serving 회귀와 Round15의 RoPE/attention 구간 측정을 근거로 세 가지 연관 변경을 적용했다: 과거 KV와 현재 토큰 경로 분리, 반복 주소 계산 이동, 정렬된 인접 BF16 읽기 결합. Accepted Batch7은 유지한다.

새 원격 소스는 `4c5bcff43d1942fdd3c396b2b9bcd9df3bf63593`이며 Batch8의 직접 자식이다. 변경은 `kernels/src/graph_numerics.cu` 하나다. F108 인터페이스는 유지되며 새로운 소스와 바이너리 해시로 후보를 구분한다.

- 독립 GPU 수치 검사: 8,640개 통과. 103,680개 device allocation 전부 해제, live allocation 및 cleanup error 0.
- 실제 체크포인트 GPU 검사: 4개 통과. Prefill/decode logits 및 KV, retained reuse/cancel, scheduler block mapping 검사 포함.
- 서버 library: 72개 통과, 2개 ignored. Profile 도구: 10개 통과.
- 별도 read-only 재검증 명령도 성공했다. 원격 소스, build, 바이너리, 모델, private runtime, 원본 로그와 probe 증빙을 재확인했다. 로컬 복사본의 여섯 test log 해시도 receipt와 일치했다.

증빙: [build](raw/fusion-history-build-v1.json), [GPU probe](raw/fusion-history-probe-v1/receipt.json), [model tests](raw/fusion-history-model-tests-v1.json), [runner](qualify_fusion_history_v1.py).

이 검사는 성능 측정이 아니다. Blender를 종료하지 않았다. 새 후보의 CPU·GPU-greedy 두 경로 HTTP 검사 92개가 통과했다. Raw/default 응답, O1/O32/stop, 동시 요청 1/2/4/8, 연결 종료 후 재사용과 두 서버의 정리를 확인했다. 별도 `--validate-only`는 저장된 raw 응답을 다시 검사했고 성공했다. Baseline/vLLM serving 비교는 아직 실행하지 않았다. 후보 채택, 높은 concurrency 정확성·P95/P99 안정성, 전체 목표 달성을 주장하지 않는다.

Round16 세션 helper 생성과 현재 Blender PID 2730220/2730294/2730392의 read-only 검사도 완료했다. 종료·복원 lifecycle 함수는 기존 검증된 helper와 AST가 동일하다. 아직 Blender 종료나 성능 측정은 시작하지 않았다. 다음 단계는 새 model receipt 형식을 측정 controller에 명시적으로 연결하고 후보/baseline/vLLM 계획을 검증하는 것이다. 이전 측정 계획과 복구 기록은 덮어쓰지 않는다.

HTTP 증빙: [completion](raw/fusion-history-http-correctness-v1/completion.json), [검사 runner](fusion_history_http_check_v1.py). 로컬 복사본에서도 내부 HTTP artifact 참조 391개의 해시를 확인했다.
