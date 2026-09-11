# C14 — Multi-model / Multi-hardware Competitive Matrix

**상태:** Planned  
**의미 등급:** `reference` benchmark/qualification  
**한 가지 목적:** 0.5B~8B dense model과 RTX 4090/H100에서 C01의 M4/M5/S1을 최종 판정하고 지원 범위를 공개한다.

[이전: C13](13-restartable-gpu-worker.md) | [목차](README.md)

## 1. 범위

이 PR은 성능 코드를 고치지 않는다. C07~C13에서 승격된 exact implementation과 stable configuration을 freeze한 뒤 동일 campaign을 실행한다. campaign 중 발견한 성능/정확성 문제는 별도 corrective PR로 분리하고 새 candidate로 matrix를 다시 실행한다.

## 2. 모델 선택 원칙

exact model ID/revision은 campaign 시작 전에 manifest로 고정한다. 최소 구성:

| Tier | 목적 | 조건 |
|---|---|---|
| 135M diagnostic | attribution/history | 기존 SmolLM2 유지, 승리 선언 제외 |
| 0.5B | small interactive | Qwen/Llama compatible dense |
| 1~2B | edge/single-GPU chat | dense, BF16 |
| 3B | 중간 규모 latency/throughput | dense, BF16 |
| 7~8B | 실사용 single-GPU 핵심 | 24GB 적재 가능한 approved model |

가능하면 Llama와 Qwen family를 모두 포함한다. 라이선스/접근 제한 model은 public 재현성 문서와 별도 availability를 명시한다.

각 model manifest:

```text
model ID
full revision SHA
config/tokenizer/weights hashes
architecture/IR signature
context limit
vocabulary size
tied weight 여부
RoPE/KV geometry
expected runtime capability
```

## 3. Hardware lane

### RTX 4090 / sm89

- 현재 primary 개발/latency lane
- exact GPU UUID, clock/power/thermal policy
- driver/CUDA/cuBLAS/Rust/image 고정

### H100 / sm90

- datacenter throughput/latency lane
- SXM/PCIe form factor를 혼합하지 않음
- BF16 우선
- sm90 AOT build와 capability receipt

B200/Blackwell은 후속 matrix다. H100 결과를 4090과 직접 ratio로 비교해 `빠르다`고 말하지 않는다. 각 hardware 안에서 Riley/vLLM을 비교한다.

## 4. Competitor lane

C01에 따라 campaign 시작 시점의 current vLLM release/tag를 고정한다.

- exact source/tag/wheel hash
- torch/CUDA dependency lock
- full runtime options
- graph/attention backend selection receipt
- prefix cache on/off
- sampling backend

Riley에 없는 기능을 vLLM에서 켜서 일반 exact baseline에 섞지 않는다. feature-specific lane은 별도 cell로 보고한다.

## 5. Matrix

### Latency cells

- concurrency `1,2,4,8`
- prompt `128,1024,4096,8192 가능한 범위`
- output `32,128,512`
- greedy
- cold/warm

### Serving cells

- concurrency `8,16,32`
- prompt length distribution short/medium/long 혼합
- output length distribution
- open-loop arrival rate: low/target/saturation/overload
- cancellation `0,10,30%`
- slow client 별도

### Prefix cells — C12 완료 시

- cache disabled
- controlled 50%/90% exact hit
- private domain
- different domain isolation

### Isolation cells — C13 완료 시

- in-process
- isolated worker
- fault/restart SLO

지원되지 않는 cell은 실패를 숨기지 않고 `unsupported`와 capability reason을 기록한다.

## 6. Warm/cold 정책

### Cold

- fresh process
- model state reset
- external dependency cache 정책 명시
- model load, graph prepare, weight packing, warmup을 분리 측정

### Warm

- warmup 5회 이상
- measured 30회 이상
- independent fresh process 5회
- AB/BA 교차

compile/JIT가 존재하는 competitor는 cache prime 정책을 Riley와 문서화하고 cold와 warm을 혼합하지 않는다.

## 7. Metric

### Primary

- TTFT p50/p95/p99
- request mean TPOT p50/p95/p99
- pooled ITL p95/p99
- E2E p95/p99
- SLO goodput

### Capacity/resource

- output/scheduled tokens per second
- completed requests/s
- CPU utilization
- GPU utilization/idle gap
- peak VRAM
- usable KV blocks/bytes
- graph retained bytes
- model load/prepare time

### Reliability

- failure/timeout
- token mismatch
- duplicate terminal
- cancellation completion latency
- worker restart/recovery
- memory/resource leak

## 8. 통계와 판정

- percentile: contract에 정의한 동일 방법
- independent process 5개 미만 거부
- failure arm을 성공 percentile에 포함하지 않음
- median-of-run 또는 predeclared aggregate 사용
- model/hardware 전체 summary는 arithmetic mean 대신 사전 정의한 geometric ratio 사용
- bootstrap confidence interval은 보조 지표이며 threshold 대체가 아님

M4/M5는 [`01-vllm-win-contract.md`](01-vllm-win-contract.md)의 기준을 그대로 사용한다.

## 9. 승리 범위 표현

허용 표현 예:

> RTX 4090, BF16, pinned Llama/Qwen-compatible 0.5B~3B, concurrency 1~8, prompt 128~4K의 필수 matrix에서 Riley가 vLLM campaign X 대비 M5를 통과했다.

금지 표현:

- 한 cell 결과로 `Riley가 vLLM보다 전반적으로 빠르다`
- 135M diagnostic 결과를 7B에 일반화
- 4090 결과를 H100/B200에 일반화
- prefix cache hit와 cache-off 결과 혼합
- failed/unsupported model을 summary에서 조용히 제외

## 10. Capability matrix

최종 report에는 다음 표를 생성한다.

```text
model family / size
GPU
context range
concurrency range
execution mode
attention/projection/MLP/LM-head implementation IDs
prefix cache support
worker isolation support
correctness status
M4/M5/S1 status
known limitations
```

지원 범위는 runtime startup capability receipt와 일치해야 한다.

## 11. Artifact 정책

Git에는 summary와 closed report, manifest/checksum만 체크인한다.

```text
benchmarks/competitive/results/<campaign>/README.md
campaign-manifest.json
closed-report.json
SHA256SUMS
selected profiler summaries
```

대용량 raw JSONL/NCU/NSYS/source archives는 append-only external evidence root에 저장하고 path/hash를 결합한다. immutable provenance 이름의 과거 `rustinfer` 문자열은 변경하지 않는다.

## 12. 예상 파일 변경

```text
benchmarks/competitive/models/*.json
benchmarks/competitive/hardware/*.json
benchmarks/competitive/matrices/*.json
benchmarks/competitive/lanes/*.json
benchmarks/competitive/scripts/run_full_matrix.py
benchmarks/competitive/scripts/check_full_matrix.py
benchmarks/competitive/scripts/tests/*
benchmarks/competitive/results/<campaign>/*
README.md 또는 release support matrix 링크
```

runtime source 변경은 금지한다.

## 13. 실행 절차

1. Riley candidate와 vLLM competitor freeze
2. model/hardware/dependency preflight
3. correctness/golden probe
4. diagnostic cells
5. latency matrix
6. serving saturation search
7. prefix/isolation feature cells
8. fault/S1 cells
9. raw evidence close/checksum
10. closed M4/M5/S1 report
11. human-readable support/limitation summary

중간 결과를 보고 remaining matrix를 줄이지 않는다. hardware 장애 등으로 중단되면 campaign은 incomplete이며 새 run ID로 재개한다.

## 14. 승인 기준

### Matrix completeness

- 모든 required cell이 `success | failed | unsupported`로 명시
- missing cell 0
- independent run/count/hash 검증
- same-campaign Riley/vLLM identity 일치

### Correctness

- 필수 supported cell token/failure mismatch 0
- model별 canonical/golden gate 통과

### Competitive

- M4/M5 checker 결과 생성
- partial-win을 full-win으로 표현하지 않음

### Stability

- C02 candidate qualification 유지
- C13 사용 시 fault/restart S1 통과
- memory/terminal/request invariants 위반 0

## 15. 실패 처리

matrix 중 runtime defect를 고치지 않는다.

- failing evidence 보존
- defect별 corrective PR
- 새 candidate/version
- 영향 cell만이 아니라 contract가 요구하는 full matrix 재실행 여부를 checker 정책에 따라 결정

서로 다른 candidate의 성공 cell을 조합해 final passed report를 만들지 않는다.

## 16. 롤백

benchmark-only PR이므로 runtime rollback은 없다. 잘못된 contract/report가 발견되면 기존 result를 삭제하지 않고 `invalidated` manifest와 새 schema/run을 추가한다.

## 17. 완료 정의

0.5B~8B와 RTX 4090/H100의 모든 required cell이 같은 campaign contract로 닫히고, Riley의 M4/M5/S1 통과 범위와 실패/unsupported 범위를 과장 없이 공개할 수 있을 때 완료다.
