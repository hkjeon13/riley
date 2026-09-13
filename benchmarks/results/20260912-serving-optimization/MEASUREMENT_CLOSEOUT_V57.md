# 마지막 측정 마무리 — V57 primitive / V56 working set

사용자 요청에 따라 준비되어 있던 측정까지만 완료하고 추가 구현을 중단했다. V57은 standalone primitive이며 serving binary에 통합하지 않았다. 원격 application checkout은 V56 commit `030c207565eda3ec45533a166e6d9a201c8d1831` 상태에서 clean임을 확인했다.

## Serving 판정

최종 serving 측정은 [V56 Round62](V56_SERVING_RESULTS.md)다. V56은 V52 대비 workload별 throughput −1.80%~+1.44%로 혼재한다. C32 natural에서 vLLM 대비 throughput −12.69%, TPOT +25.04%, P99 +16.34%로 목표 미달이다. V51 일반 reference 및 V52 비교 기준은 보존한다. V56 일반 승격 없음.

## V57 attention pair load

Q/K의 인접 BF16을 32-bit로 읽는 경로, shared probability pair load, 두 변경 결합, mixed shared stride padding을 비교했다. KV 저장 layout은 바꾸지 않았다.

Mixed 242 scenarios, context boundary 38 scenarios, decode 2,016 cases의 correctness 검사를 완료했다. mixed/boundary/decode memcheck 및 mixed/decode racecheck는 오류 0으로 종료했다. 자세한 개별 검사 범위와 결과는 원본 로그를 따른다.

Mixed timing은 600 records(30 conditions × 5 variants × 4 orders), decode는 320 records(20 conditions × 4 variants × 4 orders)다. mixed의 pair+padding은 baseline 대비 약 1.5~36.2% 빨랐다. probability pair 단독 이득은 거의 없었다. 아래 pure decode의 Q/K pair 결과처럼 효과는 조건별로 다르다. 이는 serving 향상을 입증하지 않는다. shared padding의 bank conflict 개선은 counter로 확인하지 않았으므로 원인을 단정하지 않는다.

| rows | context | baseline µs | Q/K pair µs | 시간 변화 |
|---:|---:|---:|---:|---:|
| 1 | 128 | 5.632 | 5.632 | +0.00% |
| 1 | 398 | 9.431 | 9.428 | -0.03% |
| 1 | 1024 | 15.846 | 15.340 | -3.19% |
| 1 | 4096 | 52.122 | 50.697 | -2.73% |
| 4 | 128 | 5.681 | 5.682 | +0.02% |
| 4 | 398 | 9.523 | 9.533 | +0.11% |
| 4 | 1024 | 15.973 | 15.467 | -3.16% |
| 4 | 4096 | 52.874 | 51.451 | -2.69% |
| 8 | 128 | 5.826 | 5.815 | -0.19% |
| 8 | 398 | 9.858 | 9.762 | -0.98% |
| 8 | 1024 | 16.512 | 15.829 | -4.14% |
| 8 | 4096 | 54.198 | 52.035 | -3.99% |
| 16 | 128 | 6.060 | 6.029 | -0.50% |
| 16 | 398 | 10.663 | 10.245 | -3.93% |
| 16 | 1024 | 17.833 | 16.608 | -6.87% |
| 16 | 4096 | 57.935 | 53.697 | -7.32% |
| 32 | 128 | 7.578 | 7.280 | -3.93% |
| 32 | 398 | 14.408 | 13.503 | -6.28% |
| 32 | 1024 | 27.228 | 24.968 | -8.30% |
| 32 | 4096 | 134.216 | 133.488 | -0.54% |

## V56 30-layer working-set probe

기존 gate와 split_rows<4>를 30개의 서로 다른 layer weight로 실행했다. weight 106,168,320 bytes, device L2 75,497,472 bytes. 출력·padding은 여섯 row 조건 모두 exact. 30-call graph, warmup 5회, replay 100회, 네 역순 order로 48 timing records를 얻었다. 아래 값은 layer당 시간이다.

| rows | 기존 gate µs | split_rows µs | 시간 변화 |
|---:|---:|---:|---:|
| 1 | 8.035 | 8.823 | +9.80% |
| 4 | 8.495 | 9.056 | +6.61% |
| 8 | 8.807 | 9.445 | +7.24% |
| 16 | 9.484 | 9.987 | +5.30% |
| 24 | 11.238 | 10.112 | -10.02% |
| 32 | 12.028 | 10.407 | -13.48% |

작은 batch에서는 5.3~9.8% 느려지고 rows24/32에서는 10.0~13.5% 빨라졌다. 따라서 “L2를 넘으면 모든 개선이 반전된다”는 설명은 성립하지 않는다. 실제 V56 serving profile에서 관측한 gate 회귀를 이 gate-only probe가 완전히 설명하지 못한다. 다른 연산이 섞인 cache 상태·동적 row 분포 등은 미확정이며 추가 실험은 중단했다.

## 보존한 증거와 종료 상태

[원자료](raw/final-measurements-v57/)에 소스, 빌드·검사 로그, 타이밍, 분석을 보존했다. 원격 원본 47개 파일의 SHA256을 로컬과 비교해 모두 일치함을 확인했다. archive SHA256: `9ba0afa17654d99958295ebabf9613e15c7cbde5fd4036f628a48ee9c76cc467`. [검증 receipt](raw/final-measurements-v57/remote-verification.json), [분석 JSON](raw/final-measurements-v57/analysis.json).

두 측정 controller 모두 exit 0. Blender 프로세스가 없음을 read-only로 확인했으며 복구하지 않았다. 추가 optimization batch, V57 application 통합, 새 serving run은 진행하지 않았다. 다음 방향은 [구조 중심 연구](RESEARCH_20260913.md)에 정리했다.
