# PR-N04 — 공유 prefix attention state 병합 검증

## 공통 적용 조건

[공통 계약과 실행 순서](README.md)를 따른다. 현재는 계획이며 구현·측정 완료를 의미하지 않는다. strict 기본 경로는 보존하고 새 실행은 opt-in으로 둔다. 1차 RTX4090 전체 GPU peak 20,000,000,000 bytes, BF16 weight/KV, Rust→native ABI→CUDA를 유지한다. runtime Python은 사용하지 않는다. Blender 종료 상태를 유지하며 복구하지 않는다.

## 목적과 선행 조건

N02 완료, 대상 workload에 충분히 긴 공유 prefix가 존재한다는 근거 확보. N03 통과는 필수 선행 조건이 아니지만 첫 실험은 N03 판정 후 진행한다. 공유가 없다면 구현을 건너뛴다.

## 변경 묶음

1. 기존 prefix page/owner metadata에서 공유 구간을 그룹화한다. 별도 radix cache나 KV 복제를 추가하지 않는다.
2. 공유 prefix 다중 query 실행과 개별 suffix 실행을 분리하고 output+log-sum-exp state를 안정적으로 병합한다.
3. 작은 owner 그룹/짧은 prefix/비공유는 native 대조군으로 fallback한다. grouping 비용과 suffix 증가를 포함해 dispatch 조건을 고정한다.
4. COW, owner 해제, 취소, page 재사용과 graph bucket 전환 중 상태 수명을 검증한다.

## 제한된 실험

(prefix tokens, suffix tokens, owners)=(0,1024,8),(2048,128,4),(8192,128,8),(8192,1024,8), 추가 owner1 대조군. 모델 context·20GB에 맞는 조합만 사용한다. 그룹 구성+두 attention 실행+merge의 합을 비교한다.

## 검증·진행 기준

실수에서 state 결합은 동치지만 BF16 반올림은 다를 수 있으므로 N01 수치 gate가 필수다. 빈 prefix/suffix, 전부 masked, page 경계와 COW 분리를 확인한다. 긴 공유 workload에 순이득이 있고 비공유 fallback 회귀가 없을 때 N06 조건부 후보로 선정한다.

산출물에는 이득이 사라지는 prefix/owner/suffix 경계와 cache 잔류분을 포함한 peak를 남긴다. 기존 두 owner의 strict KV 재사용과 구조적 차이를 설명한다. 공유 조건 결과를 범용 성능 향상으로 보고하지 않는다.

## 중단·롤백

grouping/merge가 절감량을 상쇄하면 보류한다. 기존 cache는 변경 없이 유지하고 native 비공유 실행으로 복귀한다. 다음 시도는 임의 threshold 탐색이 아니라 실제 공유 분포가 달라졌을 때 검토한다.
