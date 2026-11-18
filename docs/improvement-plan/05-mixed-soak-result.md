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

같은 바이너리·모델·입력 생성기와 16,384블록 KV 풀에서 30분씩 실행했다. 첫 조건은 2026-11-19 01:39 KST에 시작했고, 두 번째 조건에는 prefix cache 128블록을 할당했다. 두 조건의 바이너리·체크포인트·토크나이저·부하 생성기 해시와 서버 인자를 대조했다. [GPU 실행 도구](../../benchmarks/evidence/20261119-w5-mixed-soak/tools/run_soak.py) · [순차 실행 명령](../../benchmarks/evidence/20261119-w5-mixed-soak/campaign.sh)

| 측정 | cache off | cache on |
|---|---:|---:|
| 정상 완료 / 클라이언트 연결 종료 | 13,749 / 700 | 13,749 / 700 |
| 스키마 유효 완료 / 필수 도구 완료 | 1,739 / 708 | 1,739 / 708 |
| HTTP·SSE 오류와 429 | 0 | 0 |
| 서버 disconnect / cancel 증가 | 700 / 700 | 700 / 700 |
| TTFT p95 초기 → 후반 | 327.14 → 330.79ms (+1.11%) | 320.57 → 323.36ms (+0.87%) |
| 전체 완료 지연 p95 초기 → 후반 | 4,784.31 → 4,149.00ms (−13.28%) | 4,770.78 → 4,128.77ms (−13.46%) |
| RSS 후반 변화 | +0.75MiB | 0.00MiB |
| drain 뒤 활성·대기·요청 KV 블록 | 각각 0 | 각각 0 |
| 종료 뒤 device·pinned 할당과 바이트 | 각각 0 | 각각 0 |

두 조건 모두 p95 후반 증가 5% 이하와 자원 회수 조건을 통과했다. 부하 직전과 drain 뒤 native 할당값이 같았고 서버·클라이언트 모두 종료 코드 0을 남겼다. GPU 7은 각 조건 종료 뒤 동일 홀더의 20GiB 대기로 복귀했다. 전체 시험 종료 시점에는 GPU 6·7 모두 20GiB 예약 상태였다. [cache off 집계](../../benchmarks/evidence/20261119-w5-mixed-soak/cache-off/summary.json) · [cache on 집계](../../benchmarks/evidence/20261119-w5-mixed-soak/cache-on/summary.json) · [GPU 6 예약](../../benchmarks/evidence/20261119-w5-mixed-soak/gpu6-final.json) · [GPU 7 예약](../../benchmarks/evidence/20261119-w5-mixed-soak/gpu7-final.json)

각 조건의 운영 메트릭 32개 표본에서 24개 `batch_shapes` 슬롯 중 21개가 구성됐다. 비어 있는 슬롯은 모두 0이었고, 저장된 스텝 수와 GPU 시간 표본 수가 각 bucket에서 일치했다. 두 조건 모두 진단 링에서 오래된 요청 요약 13,425개가 제거됐다. 부하 측정의 오류 수와 표본 수는 별도로 기록됐다. [bucket 검증 집계](../../benchmarks/evidence/20261119-w5-mixed-soak/live-batch-shapes.json) · [cache off 원본](../../benchmarks/evidence/20261119-w5-mixed-soak/cache-off/soak.json.gz) · [cache on 원본](../../benchmarks/evidence/20261119-w5-mixed-soak/cache-on/soak.json.gz) · [양쪽 실행 근거 해시](../../benchmarks/evidence/20261119-w5-mixed-soak/SHA256SUMS)

검증 범위는 합성 부하를 사용한 한 체크포인트의 서빙이다. 실제 운영 분포, cache hit·eviction, 의도적 큐 포화 후 복구, 실제 장치 장애, 배포 이미지에서의 GPU 추론은 별도 검증 항목이다.
