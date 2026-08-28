# Riley vLLM 경쟁력 로드맵

**상태:** In progress — C01과 C02-P0는 clean remote branch에 별도 커밋됐고, C02-P1 provenance closure가 candidate freeze 전에 진행 중이다.
**작성 기준:** `main@1195cf20eef0bd6c3d72ac90437d308265e6f951`
**목표:** Riley가 제한된 우선 지원 범위에서 vLLM보다 더 낮은 지연, 더 높은 SLO goodput, 더 예측 가능한 오류 격리와 복구를 제공하도록 후속 작업을 독립 PR로 분해한다.

이 폴더는 기존 [`deploy/00-pr-contract.md`](../00-pr-contract.md), [`deploy/15-profiling-and-optimization.md`](../15-profiling-and-optimization.md), [`deploy/16-reliability-and-release.md`](../16-reliability-and-release.md), [`deploy/17-extension-gates.md`](../17-extension-gates.md)의 하위 실행 계획이다. 기존 gate와 충돌하는 경우 기존 문서의 fail-closed 규칙이 우선한다.

## 1. 현재 출발점

현재 Riley는 SmolLM2-135M BF16, RTX 4090, `c1/p128/o32/greedy`에서 active-row bucket, packed asynchronous metadata H2D, GPU greedy, cooperative shared-KV GQA attention을 결합하여 내부 baseline 대비 다음 결과를 확보했다.

| 지표 | 기존 경로 | M1 후보 | 변화 |
|---|---:|---:|---:|
| TTFT p50 | 5.450 ms | 4.033 ms | -25.98% |
| TPOT p50 | 7.166 ms | 4.109 ms | -42.66% |
| E2E p50 | 227.595 ms | 131.404 ms | -42.26% |
| 처리량 | 140.535 tok/s | 243.536 tok/s | +73.29% |

다만 이 결과는 최신 Riley와 vLLM을 같은 시점·같은 candidate·같은 campaign에서 직접 비교한 승리 증거가 아니다. 또한 `0.1.0-rc2`는 candidate-bound 전체 soak를 생략한 prerelease이므로 정식 Gate E qualification도 남아 있다.

## 2. 최초 경쟁 범위

Riley는 vLLM의 전체 기능을 복제하지 않고 다음 범위부터 이긴다.

| 축 | 최초 범위 |
|---|---|
| GPU | RTX 4090 우선, H100 확장 |
| 모델 | Llama/Qwen 호환 dense decoder |
| 크기 | 진단용 135M, 경쟁 판정용 0.5B·1~3B·7~8B |
| dtype | BF16 우선 |
| context | 128~8K, 후속 32K |
| concurrency | 1, 2, 4, 8, 16, 32 |
| output | 32, 128, 512 tokens |
| API | OpenAI 호환 streaming |
| 초기 비범위 | MoE, multimodal, multi-GPU, multi-LoRA |

## 3. 경쟁 milestone

| milestone | 판정 |
|---|---|
| M1 | 기존 Riley 대비 TPOT 30% 이상 개선 — 달성 |
| M2 | graph-ready decode 경로로 기존 Riley 대비 TPOT 2배 이상 개선 |
| M3 | 대표 c1 decode TPOT p50 2.0 ms 이하 |
| M4 — parity | 동일 campaign에서 Riley/vLLM TTFT·TPOT p95 ratio 각각 `<= 1.03` |
| M5 — win | TTFT·TPOT p95 ratio 각각 `<= 0.90`, SLO goodput ratio `>= 1.10`, peak VRAM ratio `<= 1.05` |
| S1 — stable win | candidate-bound soak와 오류 주입에서 잘못된 token·누수·hang·중복 종료 0 |

절대 지연 목표와 경쟁 상대 대비 비율을 함께 판정한다. vLLM 버전이 바뀌면 과거 절대값으로 승리를 선언하지 않고 같은 campaign에서 M4/M5를 다시 계산한다.

## 4. PR 순서

| ID | 문서 | 한 가지 목적 | 선행 조건 |
|---|---|---|---|
| C01 | [vLLM 승리 계약](01-vllm-win-contract.md) | 공정 비교와 M4/M5 판정 계약 고정 | 없음 |
| C02-P0 | [effective runtime configuration receipt](02a-effective-runtime-config-receipt.md) | cold-prepared `/v1/config`와 startup artifact raw evidence 구현 | C01 |
| C02-P1 | [provenance v2와 reconstructed rollback baseline](02b-c02-p1-provenance-v2.md) | v3 config bridge 뒤에 v4 serial-session raw provenance를 닫고 self-authored soak/rollback summary를 raw process/evidence provenance로 교체 | C02-P0 |
| C02 | [RC3 candidate qualification](02-rc3-candidate-qualification.md) | 최신 단일 revision의 정식 release gate 종료 | C01, C02-P0, C02-P1 |
| C03 | [Scheduler routing property fuzz](03-scheduler-output-routing-fuzz.md) | request-token routing invariant를 생성형 테스트로 고정 | C02 |
| C04 | [Llama executor 분리](04-llama-executor-refactor.md) | graph/fusion 작업 전에 거대 executor를 동작 보존 분리 | C03 권장 |
| C05 | [CUDA Graph ownership ABI](05-cuda-graph-ownership-abi.md) | capture/instantiate/replay/close의 native ownership 경계 구현 | C04 |
| C06 | [Graph signature dispatcher](06-graph-signature-dispatcher.md) | full/piecewise/eager 선택과 exact fallback 구현 | C05 |
| C07 | [Decode graph buckets](07-decode-graph-buckets.md) | `M=1..32` pure-decode graph fast path로 M2 판정 | C06 |
| C08 | [Executable pattern registry](08-executable-pattern-registry.md) | semantic IR과 kernel implementation 선택 분리 | C04, C07 결과 |
| C09 | [Packed QKV/Gate-Up weights](09-packed-projection-weights.md) | 중복 projection과 weight ownership 정리 | C08 |
| C10 | [Transformer subgraph fusion](10-transformer-subgraph-fusion.md) | QKV-RoPE-KV 및 MLP 반복 subgraph의 E0 fusion | C09 |
| C11 | [LM-head/sampling fusion](11-lm-head-sampling-fusion.md) | greedy 경로의 full logits materialization 제거 | C08, C10 권장 |
| C12 | [Tenant-safe prefix cache](12-tenant-safe-prefix-cache.md) | exact prefix reuse와 보안/lifetime 계약 구현 | C02, KV 안정화 |
| C13 | [Restartable GPU worker](13-restartable-gpu-worker.md) | CUDA/panic 실패를 API process에서 격리 | C02, C03 |
| C14 | [Multi-model/hardware matrix](14-multi-model-hardware-matrix.md) | 0.5B~8B·RTX4090/H100에서 M4/M5 최종 판정 | C07~C13 선택 결과 |

C12는 기존 PR 17 규칙 때문에 하나의 merge PR로 끝나지 않는다. 실제 병합은 `C12-A admission-only`와 `C12-B implementation`, 필요 시 schema v2 이후 `C12-C stable promotion`으로 분리한다.

## 5. 공통 PR 계약

모든 PR은 다음을 지킨다.

1. 한 PR은 한 가지 성능 또는 안정성 가설만 다룬다.
2. `reference`, `E0`, `E1`, `A1`, `M1` 중 의미 등급을 명시한다.
3. correctness 수정과 aggressive optimization을 같은 PR에 섞지 않는다.
4. production default 변경 전 exact fallback과 명시적 rollback flag를 유지한다.
5. threshold는 결과를 본 뒤 완화하지 않는다.
6. 성능 evidence는 clean Git commit, source archive, executable hash, model/tokenizer hash, GPU UUID, image, driver/CUDA 버전에 결합한다.
7. hot loop의 host/device allocation 증가는 허용하지 않는다.
8. 오류가 난 arm의 결과를 성공 표본에 포함하지 않는다.
9. unsupported shape, sampling, backend는 암묵적 근사가 아니라 exact fallback 또는 fail-closed로 처리한다.
10. Python/Triton은 reference와 prototype에만 사용하며 production dependency graph에 들어가지 않는다.

## 6. 공통 promotion gate

별도 문서가 더 강한 기준을 두지 않는 한 다음을 최소 기준으로 사용한다.

- failure, dropped trace, token routing mismatch: 모두 0
- canonical numeric/token correctness: 통과
- independent process 5회 이상, AB/BA 교차 실행
- primary metric arm median 및 paired median: 사전 목표 통과
- TTFT p95 candidate/current ratio: `<= 1.05`
- 비대상 필수 workload TPOT/E2E p95 ratio: `<= 1.05`
- c8 이상 대표 throughput candidate/current ratio: `>= 0.95`
- peak VRAM 증가: `<= 5%`
- usable KV block capacity 감소: 0 또는 사전 승인된 명시적 trade-off
- hot-loop allocation delta: 0
- owner close 후 Riley-owned live allocation: 0
- cancellation, client disconnect, overload에서 duplicate terminal event: 0

M4/M5 비교 PR에서는 current baseline 대신 같은 campaign의 vLLM arm을 분모로 사용한다.

## 7. 공통 산출물

각 PR은 최소 다음을 남긴다.

```text
설계 문서
변경 전/후 source boundary
정확성 report
성능 또는 신뢰성 raw evidence
기계 판독 가능한 closed report
환경·artifact provenance
운영 flag와 default 상태
rollback 절차
known limitations
```

승격하지 않은 후보도 append-only evidence에 보존한다. 실패한 실험을 삭제하거나 성공 결과로 덮어쓰지 않는다.

## 8. 권장 착수 순서

우선 `C01 → C02-P0 → C02-P1/v3 config bridge → C02-P1/v4 serial-session binder → C02-P1 lifecycle·semantic closure → C02 → C03 → C04`로 비교 기준, raw evidence surface, provenance closure, release 안정성, routing invariant, 코드 경계를 닫는다. 첫 C02-P1 lifecycle runner는 contract 1 scenario·observation 1회·v4 manifest 1개만 허용하며, multi-scenario timing/집계는 후속 versioned semantic contract로 미룬다. v4 binder와 fixture 테스트가 clean source로 고정되기 전에는 GPU capture나 candidate freeze를 시작하지 않는다. 이후 `C05 → C06 → C07`로 CUDA Graph M2를 판정한다. C07 이후 새 profile이 실제 병목을 선택할 때만 C09~C11의 fusion 순서를 확정한다. Prefix cache와 process 격리는 성능 숫자를 만들기 위한 우회가 아니라 별도의 serving 효율·안정성 축으로 판정한다.
