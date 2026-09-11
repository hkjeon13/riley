# 2026-09-04 계획 검토 기록

## 검토 범위

- `deploy/260904/*.md` 전체
- 현재 `main@f1ecb0c`의 executor/graph/capability 파일 존재 여부
- 기존 C01~C14 문서의 status와 C07-36 이후 미완료 경계
- M1/vLLM historical evidence의 동일-campaign 한계

## 발견한 문제와 반영 결과

| 발견 | 영향 | 반영 |
|---|---|---|
| C02-P2가 폴더 밖 문서에만 존재 | 이 폴더만으로 GPU/candidate 선행조건을 알 수 없음 | [08](08-c02-p2-qualification-prs.md)에 Q01~Q04와 RUN-Q05/Q06 추가 |
| G02가 14-slot completion을 한 PR로 묶음 | PR 크기 초과, missing primitive 발생 시 진행 방법 없음 | 실제 7 supported/7 unknown을 대조하고 [02](02-c07-capability-completion.md)를 G02A~H와 projection sub-slice로 분해 |
| 기존 7개 Supported를 executor owner-bound로 오인 가능 | primitive query만으로 full graph resource lifetime을 보장하지 못함 | G02P1~P7 owner-binding audit 추가 |
| full graph가 ownership/capture/executor/bucket을 한 PR로 묶음 | 리뷰·rollback·성능 귀속이 어려움 | [03](03-c07-full-decode-graph.md)을 G03A~D로 분해 |
| registry가 schema부터 metrics까지 한 PR | graph parity와 policy/default가 결합 | [04](04-c08-runtime-graph-registry.md)를 G04A~D로 분해 |
| Tier C/M4·M5와 profiler 순서가 모호 | optimization 후 이전 report가 무효가 되는 순환 | B02 Tier D → K00/K-track → candidate re-freeze → B03/B04로 재배치 |
| 코드 PR과 root/GPU 운영 실행이 섞임 | PR 완료가 operation authority로 오인될 수 있음 | PR-Q, RUN-Q, RUN-B를 명시적으로 구분 |
| M>1 graph bucket 후속이 문장 하나뿐 | C07 전체 완료를 M=1로 오인 | G03D1~D3 bucket별 카드 추가 |
| 공통 test/evidence 필드가 카드마다 다름 | source pass를 GPU/performance pass로 오인 | [09](09-common-validation-and-evidence.md) 추가 |
| 진행 상태 정본 없음 | 과거 계획 문구와 현재 source 상태가 충돌 | [STATUS.md](STATUS.md) 추가 |
| 현재 dirty 파일 수를 영구 규칙처럼 표기 | 미래 checkout 상태와 불일치 | 매 PR 시작 시 재확인하고 unrelated 변경 전체 보존으로 수정 |
| S-track 간 선행관계가 약함 | cache/worker/H100을 동시에 시작할 위험 | S02←S01, S04←S03, H100 승인 dependency 명시 |
| B03와 S05가 같은 RTX 4090 multi-model matrix를 중복 | 비용 증가와 서로 다른 report 발생 가능 | B03를 C14 첫 RTX 4090 lane, RUN-S05를 H100/추가 context 확장으로 분리 |
| C12 stable promotion과 C13 IPC fallback이 implementation에 묻힘 | default 승격 또는 shared-memory가 같은 PR로 확대될 수 있음 | RUN-S02P와 conditional S04B로 분리 |

## 검토 후 남은 의도적 blocker

아래는 문서 누락이 아니라 실제 외부 조건이다.

- reviewer/administrator가 승인한 Q01 immutable design record
- root-installed guardian/controller/ledger의 Q05 no-GPU acceptance
- Q06 GPU/Docker/capture authorization
- current vLLM competitor와 Tier C model/SLO concrete pin
- 전용 GPU 시간과 clean candidate/evidence root

이 blocker가 없어도 Q/G track의 source/CPU PR은 카드에 적힌 범위까지 진행할 수 있다.
GPU parity, candidate qualification, M4/M5는 blocker가 해소되기 전 완료로 올릴 수 없다.

## 현재 첫 착수 카드

서로 충돌 없이 병렬 가능한 첫 source 작업은 다음이다.

1. Q01 review contract/ABI freeze
2. G01 parent KV span attention owner source contract
3. G02P와 G02A~G02F의 slot별 owner/evidence adapter CPU contract

한 PR만 순차적으로 진행한다면 G01을 먼저 권장한다. 현재 C07의 명시적 첫 미완료 항목이며
G02G와 G02H, 이후 full graph를 직접 해제한다.
