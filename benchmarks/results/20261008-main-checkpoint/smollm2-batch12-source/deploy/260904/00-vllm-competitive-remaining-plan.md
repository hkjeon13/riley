# Riley-vLLM 잔여 개선 PR 백로그

**기준:** `main@f1ecb0c` / 2026-09-04
**상태:** 이 폴더는 실행 가능한 PR 계획 정본이다. 구현, 원격 GPU 실행, candidate freeze,
root provisioning, 배포는 각 카드의 명시적 승인 조건 없이는 수행하지 않는다.

## 사용법

이 폴더만 읽고 다음 중 하나를 선택해 한 카드씩 진행한다.

1. 카드의 **선행 조건**을 확인한다.
2. 카드의 **변경 범위** 밖은 수정하지 않는다.
3. **구현 순서**와 **검증**을 모두 수행한다.
4. **완료/중단 판정** 중 하나를 evidence와 함께 기록한다.
5. 다음 카드는 이전 카드의 완료 조건을 충족한 경우에만 시작한다.

원본 로드맵은 역사/근거 자료일 뿐, 새 작업의 세부 지시는 이 폴더의 카드가 우선한다.

## 현재 기준선

| 항목 | 확인된 상태 | 해석 |
|---|---|---|
| M1 short decode | TPOT `7.166 → 4.109 ms`, 내부 paired gate 통과 | Riley 자체 개선이며 vLLM 승리 증거 아님 |
| C01 benchmark tooling | plan, materialized lane, adapter, journal, checker source/CPU 범위 완료 | actual campaign 없음 |
| C02 | P0/P1 source closure 완료 | actual Gate E/candidate qualification 없음 |
| C03/C04 | CPU/source contract 및 executor boundary 완료 | GPU parity/5-pair non-regression 없음 |
| C05/C06 | primitive graph ABI/parity, synthetic dispatch 완료 | full model graph/runtime dispatch 없음 |
| C07 | metadata/H2D/identity/evidence chain 진행 | attention owner binding과 full graph 없음 |

## 카드와 실행 순서

| 순서 | 문서 | 한 가지 목적 | 시작 조건 |
|---:|---|---|---|
| 0 | [08](08-c02-p2-qualification-prs.md) | native qualification 신뢰 경계와 no-GPU acceptance | 설계/source는 즉시, 설치는 승인 필요 |
| 1 | [01](01-c07-36-attention-owner-binding.md) | parent KV allocation과 attention graph ownership 결속 | source/CPU는 즉시 |
| 2 | [02](02-c07-capability-completion.md) | 나머지 primitive capability를 slot별 PR로 확정 | 01과 병렬 source 가능 |
| 3 | [03](03-c07-full-decode-graph.md) | M=1부터 bucket별 full graph capture/parity | 01·02 aggregate 및 08 실행 승인 |
| 4 | [04](04-c08-runtime-graph-registry.md) | 검증된 graph만 registry로 선택하고 eager fallback 유지 | 03 M=1 parity |
| 5 | [05](05-candidate-gpu-and-competitive-campaign.md) | candidate qualification, GPU regression, Tier D 진단 campaign | 08 no-GPU acceptance·실행 승인 |
| 6 | [06](06-profile-selected-kernel-prs.md) | Tier D profile로 선택한 한 후보만 최적화 | 03/04 및 05 Tier D |
| 7 | [05](05-candidate-gpu-and-competitive-campaign.md) | 최종 candidate로 Tier C/S M4/M5 campaign | 06 결정 후 candidate 재-freeze |
| 8 | [07](07-service-scope-and-release-prs.md) | prefix/restart/multi-model 및 release/default | core M4/M5 결과 후 |

01·02의 source work와 08의 design/source work는 병렬 가능하다. GPU acceptance는 08의 no-GPU
acceptance와 별도 실행 승인 뒤에만 한다. Tier D는 후보 선택용 진단이며, 06에서 코드가 바뀌면
candidate를 다시 freeze하고 Tier C/S를 처음부터 실행한다. 이 순서로 “M4/M5가 있어야 최적화하고,
최적화하면 M4/M5가 무효가 되는” 순환을 제거한다.

```text
Q01 ─┬─ Q02A ─ Q02B ─┐
     └─ Q03A ─ Q03B ─┴─ Q04 ─ RUN-Q05 ─ RUN-Q06
                                      │
G01 source ───────────────────────────┤─ G01 GPU ─ G02G ─┐
G02P + G02A/B/C/D/E/F source ─────────┴──────────────────┴─ G02H
                                                               │
                                         G03A ─ G03B ─ G03C ─ G04A/B/C/D ─ B01 ─ B02 Tier D
                                                           └─ G03D bucket PRs          │
                                                                                  K00 ─ one K PR
                                                                                         │
                                                    B01 re-freeze/requalify ─ B02 sanity
                                                                                         │
                                                                                B03 Tier C ─ B04 Tier S
                                                                                         │
                                                                                   S-track cards
```

G02P와 G02A~F의 GPU evidence도 RUN-Q06 이후에만 수집하지만 source/CPU 계약은 미리 진행할 수 있다.
G04A/B는 G03C 뒤, G04C/D는 B01 이전에 끝내 최종 candidate에 포함한다.

## 모든 카드에 공통인 규칙

- 계획 작성 시점의 dirty `crates/riley-model/**` 여섯 파일은 사용자 작업이다. future 상태를
  하드코딩하지 말고 매 PR 시작 시 `git status --short`로 다시 확인한다. unrelated 변경은 수정,
  stage, reset, clean, commit 대상에 포함하지 않는다.
- feature default는 `disabled` 또는 기존 eager path로 유지한다. opt-in 결과가 곧 default 승격은 아니다.
- failed/timeout/token mismatch sample은 percentile에서 숨기지 않는다. favorable report를 만들 수 없으면
  `incomparable`, `not-promoted`, `blocked` 중 하나로 끝낸다.
- full graph와 kernel candidate는 launch 후 상태가 불명확하면 재실행으로 보상하지 않는다.
  executor poison/요청 실패/eager fallback 경계는 각 카드에 적힌 대로 유지한다.
- PR마다 `git diff --check`, 수정 crate의 format/lint/unit test, 새 CPU contract test를 실행한다.
  GPU 성능 claim은 exact candidate, model/tokenizer, GPU UUID, driver/CUDA, runtime options,
  warmup, AB/BA ordering, raw artifact를 함께 남긴 경우에만 허용한다.
- 공통 실행 명령과 evidence 필드는 [09](09-common-validation-and-evidence.md)를 따른다.
- 각 PR/실행 종료 시 [STATUS.md](STATUS.md)의 상태와 evidence 링크만 갱신한다. 계획 본문의
  과거 상태 문구를 임의로 `Completed`로 고쳐 provenance를 잃지 않는다.

## 프로그램 완료 조건

다음 모두가 충족되어야 “vLLM 대비 성능 개선 완료”로 표현한다.

- candidate-bound execution authority와 actual Gate E qualification
- full decode graph 또는 profile-proven 대체 경로의 GPU parity와 eager rollback
- 현재 Riley와 pin된 current vLLM의 동일 campaign M4/M5 closed report
- short/long/concurrent/mixed/HTTP/cancellation/Llama/Qwen matrix의 regression gate
- VRAM, KV capacity, hot allocation, lifecycle, soak, deployed HTTP E2E 검증
- active default와 rollback 상태를 candidate receipt로 재현 가능하게 확인
