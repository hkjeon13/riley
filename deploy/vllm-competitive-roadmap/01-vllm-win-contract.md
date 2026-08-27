# C01 — vLLM 승리 계약 v1

**상태:** Planned  
**의미 등급:** `reference`  
**한 가지 목적:** Riley와 vLLM을 같은 조건에서 비교하고 M4 parity/M5 win을 기계적으로 판정하는 immutable benchmark contract를 만든다.

[목차](README.md) | [다음: C02](02-rc3-candidate-qualification.md)

## 1. 배경

현재 저장소에는 재현 가능한 benchmark matrix와 vLLM 0.27.1 lane이 있지만, 최신 Riley fast path와 vLLM을 같은 candidate campaign에서 비교한 closed report는 없다. 기존 SmolLM2-135M 결과는 kernel과 runtime 병목을 빠르게 찾는 진단에는 유용하지만 일반적인 LLM serving 승리를 주장하기에는 모델 크기와 workload 범위가 좁다.

이 PR은 성능을 개선하지 않는다. 앞으로 어떤 결과가 나와야 `Riley가 vLLM보다 빠르다`고 말할 수 있는지를 결과를 보기 전에 고정한다.

## 2. 범위

### 포함

- competitor version pin 정책
- 동일 model/tokenizer/token IDs와 sampling 계약
- engine-only와 HTTP streaming E2E의 분리
- cold/warm, cache-off, controlled prefix-hit workload
- closed-loop concurrency와 open-loop arrival-rate workload
- TTFT, TPOT/ITL, E2E, throughput, SLO goodput, CPU, VRAM, KV capacity, failure 지표
- AB/BA 독립 process 실행과 thermal preflight
- M4/M5 판정 checker와 result schema
- diagnostic matrix와 competitive matrix 분리

### 비범위

- Riley runtime 또는 CUDA 코드 변경
- vLLM 내부 코드를 Riley에 복사
- prefix cache 구현
- quantization/speculative decoding 비교
- 결과를 본 뒤 threshold 조정

## 3. 비교 대상 고정 방식

하나의 campaign은 다음 두 vLLM 기준을 가진다.

1. `historical-baseline`: 저장소가 이미 고정한 버전. 장기 추세와 과거 재현성 확인용이다.
2. `current-competitor`: campaign 시작 시점의 승인된 vLLM release/tag와 dependency lock. 실제 M4/M5 분모다.

`current-competitor`는 exact tag, wheel hash, torch/CUDA dependency lock, runtime option dump를 저장한다. campaign 도중 버전을 바꾸지 않는다. 새 vLLM release를 비교하려면 새 campaign ID를 만든다.

## 4. workload 계층

### Tier D — Diagnostic

- SmolLM2-135M
- `c1/p128/o32/greedy`
- 목적: micro bottleneck attribution과 빠른 반복
- 승리 선언에는 사용하지 않음

### Tier C — Competitive latency

- 0.5B, 1~3B, 7~8B dense model 각 최소 1개
- concurrency `1,2,4,8`
- prompt `128,1024,4096`
- output `32,128,512`
- greedy 및 승인된 deterministic sampling 1종
- 목적: TTFT/TPOT/ITL 비교

### Tier S — Serving SLO

- concurrency `8,16,32`
- 짧은 prompt/긴 prompt 혼합
- open-loop arrival rate 단계적 증가
- cancellation 5~20%, client disconnect 포함 별도 cell
- 목적: p99와 goodput, overload 안정성

### Tier P — Prefix reuse

- cache-off
- exact 50% prefix hit
- exact 90% prefix hit
- tenant sharing domain 동일/상이 분리
- C12 완료 전 Riley arm은 unavailable로 명시하며, 일반 비교 cell에 섞지 않음

## 5. 동일성 계약

각 request는 다음 identity를 갖는다.

```text
model revision
weights SHA-256
tokenizer revision and file hashes
prompt token IDs SHA-256
requested output length
sampling parameters and seed
EOS policy
cache policy
arrival schedule ID
```

Riley와 vLLM의 generated token hash가 비교 가능한 cell에서는 동일해야 한다. backend별 BF16 reduction 차이로 raw logits exact가 불가능한 경우에도 해당 cell의 semantic gate와 top-1/token contract를 사전에 고정한다.

## 6. 측정 분리

### Engine-only

- pretokenized IDs 사용
- tokenizer/detokenizer 제외
- scheduler enqueue부터 generated token event까지 측정

### HTTP streaming E2E

- 동일 host의 loopback client
- connection setup 포함/제외를 별도 metric으로 기록
- first SSE/JSON chunk와 token publish timestamp 기록
- client backpressure를 정상 arm과 별도 cell로 분리

두 측정치를 합쳐 하나의 지표처럼 보고하지 않는다.

## 7. 핵심 metric

| 영역 | metric |
|---|---|
| Latency | TTFT p50/p95/p99, request mean TPOT p50/p95/p99, pooled ITL p95/p99, E2E p95/p99 |
| Capacity | output tok/s, scheduled tok/s, completed request/s |
| SLO | `TTFT <= X` 및 `ITL <= Y`를 동시에 만족한 output token goodput |
| Host | process-tree CPU%, scheduler CPU ns, CUDA API time |
| GPU | utilization, kernel launches/token, stream idle gap |
| Memory | peak VRAM, Riley/vLLM usable KV bytes, fragmentation proxy |
| Reliability | failure, timeout, cancellation completion, duplicate terminal, malformed output |

SLO threshold `X/Y`는 model tier별 contract에 기록하며 결과에 따라 변경하지 않는다.

## 8. M4/M5 checker

### M4 parity

모든 필수 Tier C cell에서:

```text
Riley TTFT p95 / vLLM TTFT p95 <= 1.03
Riley TPOT p95 / vLLM TPOT p95 <= 1.03
Riley failure_count == 0
Riley token_mismatch_count == 0
```

Tier S에서는 SLO goodput ratio가 `>= 0.97`이어야 한다.

### M5 win

사전 지정한 primary cell과 전체 필수 cell의 기하평균에서:

```text
TTFT p95 ratio <= 0.90
TPOT p95 ratio <= 0.90
SLO goodput ratio >= 1.10
peak VRAM ratio <= 1.05
failure_count == 0
```

특정 c1 cell 하나만 빠른 경우 M5가 아니다. 필수 cell 중 하나라도 fail-closed 조건을 위반하면 `partial-win` 또는 `not-passed`로 기록한다.

## 9. 예상 파일 변경

```text
benchmarks/competitive/README.md
benchmarks/competitive/contract-v1.json
benchmarks/competitive/contract-v1.schema.json
benchmarks/competitive/matrices/diagnostic-sm89-bf16-v1.json
benchmarks/competitive/matrices/latency-sm89-bf16-v1.json
benchmarks/competitive/matrices/serving-sm89-bf16-v1.json
benchmarks/competitive/lanes/riley.json
benchmarks/competitive/lanes/vllm-current.json
benchmarks/competitive/scripts/run_campaign.py
benchmarks/competitive/scripts/check_campaign.py
benchmarks/competitive/scripts/tests/*
```

기존 `benchmarks/matrix.yaml`과 historical evidence는 변경하거나 덮어쓰지 않는다.

## 10. 구현 단계

1. closed schema와 unknown-field rejection을 작성한다.
2. immutable request/workload identity와 hash 계산을 구현한다.
3. Riley/vLLM lane의 environment dump와 option dump를 동일 schema로 맞춘다.
4. AB/BA execution plan을 create-only로 생성한다.
5. thermal/clock/GPU process preflight를 fail-closed로 구현한다.
6. raw JSONL을 append-only로 수집한다.
7. percentile, ratio, geometric mean, SLO goodput checker를 구현한다.
8. synthetic fixture로 pass/fail/incomparable case를 테스트한다.
9. 실제 GPU에서는 작은 Tier D dry-run만 수행해 contract 생산 가능성을 검증한다. 성능 승리 판정은 이 PR 범위가 아니다.

## 11. 테스트

- duplicate key, unknown field, invalid SHA, dirty Git 거부
- 다른 model/tokenizer/request IDs 비교 거부
- 다른 GPU UUID/driver/CUDA campaign 혼합 거부
- baseline/candidate role 역전 탐지
- warmup을 measured sample로 포함한 결과 거부
- failure sample을 성공 percentile에 포함하지 않음
- 독립 run 수 부족 거부
- AB/BA 순서와 campaign plan drift 탐지
- NaN/Infinity metric 거부
- threshold 경계값 exact test

## 12. 승인 기준

- schema와 checker unit test 전부 통과
- 동일 fixture replay가 byte-identical closed report를 생성
- historical vLLM evidence를 read-only input으로 검증 가능
- Riley/vLLM dry-run이 같은 request identity를 생성
- M4/M5 결과가 `passed | partial-win | failed | incomparable` 중 하나로만 종료
- runtime production crate 변경 없음

## 13. 롤백

이 PR은 benchmark 전용 신규 경로만 추가한다. rollback은 `benchmarks/competitive/**`를 함께 revert한다. 이미 생성된 실제 campaign evidence는 삭제하지 않고 해당 contract version이 retired되었음을 별도 index에 기록한다.

## 14. 완료 정의

`Riley가 vLLM보다 빠르다`는 문장을 사람이 임의로 판단하지 않고, 동일 campaign의 closed report 하나로 M4/M5 여부를 판정할 수 있을 때 완료다.
