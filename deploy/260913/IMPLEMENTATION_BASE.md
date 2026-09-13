# 실행 기준 코드

variable serving 작업의 기준은 실제 V56 측정 revision `030c207565eda3ec45533a166e6d9a201c8d1831`이다. 기존 main snapshot `0c045614`와 공통 조상이 없으며 crates/kernels 124개 파일이 다르다. snapshot 일부 파일과 V56 일부 파일을 임의로 조합하지 않는다.

원격 Git bundle로 V56 이력을 가져와 `codex/measured-v56`로 보존했다. 구현 브랜치는 `codex/260913-serving-integration`, 로컬 worktree는 `/Users/psyche/PycharmProjects/riley-serving-260913`이다. 원래 `/Users/psyche/PycharmProjects/riley` checkout의 unrelated dirty 변경은 그대로 보존했다.

- 계획 `894f0713`을 `6d2b2bc0`으로 이식했다. Git의 디렉터리 rename 추론이 연구 문서를 kernels/tests로 옮기려 해 해당 cherry-pick만 취소한 후 rename 추론 없이 원래 경로로 적용했다.
- PR01 첫 batch `0993ece6`을 `0779f9f4`로 이식했다. native/runtime 변경을 실제 V56 코드와 결합했다.
- V56 대비 application 변경은 PR01의 9개 파일, 693줄 추가/5줄 삭제다. V56 serving 알고리즘을 다른 후보로 교체하지 않았다.
- 이 worktree에서 riley-cuda library 92개 및 validation runner 4개 CPU 테스트가 통과했다. 기존 snapshot의 90개와 개수를 혼동하지 않는다.

이 결과는 기준 코드 정합성과 CPU 회귀 증거다. 새로운 serving benchmark 결과가 아니며, main과 자동 병합하지 않는다. 추후 main 통합 시에는 unrelated history를 억지로 합치지 말고 source diff와 검증 증거를 명시적으로 검토한다. 이후 PR01 잔여 계약과 PR02 비동기 실행은 이 기준에서 진행한다.
