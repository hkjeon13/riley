# W5-B A100 혼합 부하와 종료 검증

2026-11-19 KST에 `aws-a100`의 물리 GPU 7에서 Qwen2.5-7B-Instruct와 `native-bf16-varlen-v1`을 실행했다. 모델의 문맥 상한은 8,192, 출력 상한은 512토큰이다. CUDA 바이너리는 W4의 `f7c956d3c` 빌드와 SHA-256이 같다. [W5 계획](05-documentation-and-operations.md) · [W4 CUDA 빌드](04-structured-output-result.md) · [실행 인자와 해시](../../benchmarks/evidence/20261119-w5-mixed-soak/smoke/receipt.json)

## 90초 사전 검증

프롬프트 KV 캐시에 128블록을 할당했다. 초당 평균 8건의 합성 요청으로 일반·스키마·logprobs·필수 도구·장문·긴 출력·클라이언트 연결 종료를 섞었다. 장문 요청은 여덟 개의 프롬프트 앞부분을 반복 사용했다.

| 측정 | 결과 |
|---|---:|
| 정상 완료 | 691건 |
| 스키마 완료·독립 값 검사 | 90건 |
| 필수 도구 완료·인수 검사 | 40건 |
| 의도적으로 연결을 끊은 요청 | 39건 |
| 서버 disconnect / cancel 카운터 증가 | 각각 39건 |
| HTTP·SSE 오류와 429 | 0건 |
| drain 뒤 활성·대기·요청 KV 블록 | 각각 0 |
| 종료 뒤 device·pinned 할당과 바이트 | 각각 0 |

부하 직전과 drain 뒤 native 할당 메트릭이 같았고, `SIGTERM`을 받은 서버는 종료 코드 0으로 최종 메트릭을 남겼다. GPU 7의 동일 홀더는 이 검증 뒤 20GiB 대기 예약으로 복귀했다. 90초 구간에는 장시간 지연 가드레일을 적용하지 않았다. [요청 집계](../../benchmarks/evidence/20261119-w5-mixed-soak/smoke/soak.json) · [종료 메트릭](../../benchmarks/evidence/20261119-w5-mixed-soak/smoke/shutdown-metrics.json) · [예약 복귀](../../benchmarks/evidence/20261119-w5-mixed-soak/smoke/idle-after.json)

## 30분 본 실행

같은 바이너리·모델·입력 생성기에서 cache off 다음 cache on 순서로 각 30분을 실행한다. 2026-11-19 01:39 KST에 첫 조건을 시작했다. 판정에는 초기·후반 TTFT와 전체 완료 지연 p95, RSS, 완료·취소·오류 수, drain 뒤 native 할당과 종료 메트릭을 사용한다. [GPU 실행 도구](../../benchmarks/evidence/20261119-w5-mixed-soak/tools/run_soak.py) · [순차 실행 명령](../../benchmarks/evidence/20261119-w5-mixed-soak/campaign.sh)

범위는 합성 부하를 사용한 한 체크포인트의 서빙이다. 실제 운영 분포, cache hit·eviction 관측, 장애 주입, 이미지 패키징은 추가 검증 항목이다.
