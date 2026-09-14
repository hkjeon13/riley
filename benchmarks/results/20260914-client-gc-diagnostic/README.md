# Client GC 관측 진단 — shared C32

Riley 및 vLLM의 긴 구간 tail을 해석하기 전에 측정 클라이언트의 GC 구간을 관측했다. 서버 바이너리 및 GC 정책은 변경하지 않았다. 이 결과는 후보 backend 성능 비교를 대체하지 않는 진단이다.

## 조건과 결과

RTX 4090, SmolLM2-135M BF16, C32/active32, 같은 shared32 prompt 집합, prefix cache 활성, 양쪽 KV payload720MiB. 각 lane256 warmup 및8192 retained 요청, 순서는 control/vLLM 다음 vLLM/control이다. Riley는 projection 비활성인 현 binary control이며 rolling decode/prefill FFN/adaptive projection을 사용한다. Python은 오프라인 측정 클라이언트에만 사용한다.

| 실행 | throughput tok/s | TTFT P50 ms | TPOT P50 ms | E2E P99 ms | 최대 GC 구간 ms | P99 이상 요청 중 긴 GC와 겹침 | 10ms 이상 ITL 중 긴 GC와 겹침 |
|---|---:|---:|---:|---:|---:|---:|---:|
| shared-p0-control | 9565.5 | 14.50 | 2.74 | 343.02 | 441.99 | 82/82 | 248/21765 |
| shared-p0-vllm | 11061.8 | 29.17 | 1.79 | 346.12 | 435.11 | 82/82 | 222/621 |
| shared-p1-vllm | 11015.5 | 28.86 | 1.77 | 395.57 | 442.19 | 64/82 | 227/496 |
| shared-p1-control | 9526.1 | 14.56 | 2.75 | 326.92 | 441.52 | 82/82 | 256/21526 |

긴 GC는 callback start/end 구간10ms 이상으로 정의했다. Request 시작/끝 또는 연속 token arrival 사이와 GC 구간의 교집합이 있으면 해당 항목을 한 번 센다. 실제 latency에서 시간을 차감하지 않는다. 전체 P99 이상 요청328개 중310개(94.51%)가 긴 GC와 겹쳤다. Riley의10ms 이상 ITL43291개 중504개(1.16%)만 긴 GC와 겹쳤다. 따라서 큰 E2E tail에 클라이언트 영향이 있다는 강한 상관 증거지만, Riley의 대부분 ITL 지연이나 vLLM과의 성능 격차를 설명하지는 않는다.

전체 세대 GC의 최대 관측 구간은435–442ms다. GC callback 구간은 wall time이며 OS preemption을 포함할 수 있어 전부를 GC CPU 시간이나 독점 정지 시간으로 해석하지 않는다. Callback 자체 비용도 측정에 포함된다. GC를 끄거나 강제 수행하거나 threshold를 변경하지 않았다. Process peak RSS는 첫 lane 약657MiB에서 마지막 약2282MiB로 증가했다. `ru_maxrss`는 누적 high-water mark로 lane별 상주량 또는 메모리 누수 증명이 아니다.

## 검증과 증거

32768 retained 응답/1048576 output tokens의 SSE frame, prompt/token/text/finish/usage, arrival 순서와 phase 경계를 로컬에서 재검증했다. Riley16384 retained 응답은 frozen prior reference와 정확히 일치한다. vLLM의 reference agreement는 comparison.json에 별도 기록하며 동등 출력으로 간주하지 않는다. GC generation별 이벤트 수를 Python collection counter 차이와 대조하고 이벤트 경계·비중첩을 확인했다. 재계산한 모든 통계는 기록된 progress와 일치한다. `[DONE]`은 저장 frame에 없으며 hash-bound client의 종료 검사를 따른다.

측정4 lane 종료코드0 및 Blender3개 복구 receipt가 common archive에 있다. 웹 뷰어3개는 측정 대상에서 제외하여 계속 유지했다. 최초 실행은 제거된 unique corpus를 검사하던 잔여 assertion의 KeyError로 측정 전 실패했고 수정 후 새 디렉터리에서 재실행했다. 실패 traceback도 보존했다. vLLM 종료 로그의 leaked semaphore 경고는 보존했으며 server exit0을 경고 없음으로 표현하지 않는다.

[evidence/manifest.json](evidence/manifest.json)은5개 무손실 archive의 hash/size/member count를 담는다. 전체56개 파일의 hash·안전 경로·확장자 및 credential pattern 검사를 완료했다. common archive에 실행한 controller/client/analyzer/validator, 준비 조건, fixtures, lifecycle receipt와 결과를 보존했다. 원본 profiler/환경변수 전체 dump는 포함하지 않는다. 로컬 client와 실행 snapshot의 미사용 main 부분은 다르지만 실제 imported helper 부분은 byte-identical이다.

## 다음 판정

C8/C64 확대 전에 동일 binary·fixture·count·client로 기본 GC와 측정 구간 GC 비활성의 AB/BA intervention을 양 engine에 공통 적용한다. GC 상태는 finally에서 복구하며 강제 collection과 결과 직렬화는 측정 밖에서 수행한다. 모든 SSE 증거를 유지하고 RSS 및 오류·품질을 함께 확인한다. 한쪽 engine만 유리한 정책으로 비교하지 않는다. 기존 결과를 수정하거나 latency에서 GC 시간을 임의 차감하지 않는다.

Intervention으로 영향이 재현되면 명시적으로 versioned client 조건으로 전체 projection 비교를 다시 수행한다. 영향이 제거된 뒤에도 남는 ITL/throughput 병목을 서버 최적화 대상으로 삼는다. 이 진단만으로 기본 backend 승격, vLLM 목표 달성 또는 장기 안정성을 주장하지 않는다.
