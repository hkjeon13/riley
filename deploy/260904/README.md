# 260904 실행 계획 묶음

이 디렉터리는 Riley-vLLM 성능 개선을 **PR 크기**로 진행하기 위한 독립 작업 카드다.
상위 순서와 공통 규칙은 [00-vllm-competitive-remaining-plan.md](00-vllm-competitive-remaining-plan.md)를 따른다.

2026-09-04 전체 검토에서 다음을 보완했다.

- 폴더 밖 C02-P2 문서에만 있던 qualification 선행 작업을 [08](08-c02-p2-qualification-prs.md)로 옮겼다.
- capability completion, full graph, registry를 각각 여러 PR-sized slice로 다시 나눴다.
- Tier D 진단 → profiler/optimization → Tier C/S 최종 경쟁 campaign 순서로 고쳐 순환 의존성을 제거했다.
- 공통 검증과 evidence 규칙을 [09](09-common-validation-and-evidence.md)에 고정했다.
- 실제 진행 여부는 [STATUS.md](STATUS.md) 한 곳에서만 갱신한다.

## 빠른 선택표

| 하고 싶은 일 | 시작 문서 | 지금 가능한가 |
|---|---|---|
| qualification 신뢰 경계를 구현 | [08](08-c02-p2-qualification-prs.md) | 설계/source PR 가능, 설치·실행은 승인 필요 |
| full decode graph의 첫 실제 결손을 해결 | [01](01-c07-36-attention-owner-binding.md) | 예, source/CPU PR부터 가능 |
| graph를 model runtime에 연결 | [03](03-c07-full-decode-graph.md) | 01·02 및 GPU 승인 뒤 |
| graph 선택 policy/registry 도입 | [04](04-c08-runtime-graph-registry.md) | 03 GPU parity 뒤 |
| 진단용 Riley-vLLM 수치 생성 | [05](05-candidate-gpu-and-competitive-campaign.md) | 08 no-GPU acceptance 및 실행 승인 뒤 |
| TPOT을 더 낮출 kernel/fusion 구현 | [06](06-profile-selected-kernel-prs.md) | post-graph Tier D profile 뒤 |
| Tier C/S 경쟁 판정과 서비스 범위 확대 | [07](07-service-scope-and-release-prs.md) | 최종 optimization candidate 뒤 |

## PR 기록 템플릿

각 PR 설명에는 다음을 반드시 넣는다.

```text
Card: 260904-<문서 번호>/<PR ID>
Hypothesis: <사전 선언한 한 문장>
Scope: <수정 파일과 명시적 비범위>
Fallback/default: <existing eager path or disabled>
Validation: <실행한 CPU/GPU/test evidence>
Decision: promoted | not-promoted | rejected | blocked
Artifact: <immutable result directory or N/A for CPU-only>
```

`not-promoted`와 `rejected`는 실패가 아니라 유효한 완료 결과다. 성능 threshold를 측정 뒤에
완화하거나, 여러 PR의 작은 개선을 합산해 하나의 성능 claim으로 쓰지 않는다.

모든 카드의 공통 명령, clean-worktree 조건, GPU evidence 최소 필드는 [09](09-common-validation-and-evidence.md)를
따른다. 카드와 09가 충돌하면 더 엄격한 조건을 적용한다.
