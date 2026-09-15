# 연구 기반 다음 단계 실행 계획 — 2026-09-15

상태: **N01의 로컬 계약·수명주기 기반과 N02의 첫 synthetic projection control을 구현·로컬 검증했다.** 원격 GPU receipt, materialized 1.7B/3B 자산, native-BF16 품질 기준, full-model 및 serving benchmark는 아직 없다. [조사31](../31-method-feasibility-research.md)의 후보와 탈락 기준을 PR 단위로 구체화했으며, 향후 실행 순서는 이 문서가 [로드맵30](../30-next-stage-roadmap.md)의 개괄 순서보다 우선한다. 과거 완료/실패 기록은 변경하지 않는다.

## 실행 원칙

대규모 구현 전에 native operator/layer에서 가능성을 판정한다. 표준 BF16 대조군 없이 새 kernel의 이득을 평가하지 않는다. 후보마다 2~5개의 연관 변경을 묶고, 기본 두 실행 방식과 원인이 명확한 수정 한 차례로 실험 범위를 제한한다. 무제한 tile/threshold 탐색은 하지 않는다.

| 순서 | PR 문서 | 선행 조건 | 다음 단계로 가는 증거 |
|---|---|---|---|
| 1 | [N01 계약·실험 준비](01-contract-and-experiment-harness.md) | 없음 | 로컬 manifest·validator·lifecycle controller 완료; 원격 receipt 대기 |
| 2 | [N02 native BF16 대조군](02-native-bf16-control.md) | N01의 원격 receipt·품질 기준 | Qwen shape synthetic control 추가; SM89 실행·수치·비용 검증 대기 |
| 3 | [N03 GQA·context 분할](03-gqa-context-parallelism.md) | N02 | merge 포함 순이득·전체 기여 가능성 |
| 조건부 | [N04 공유 prefix state 병합](04-shared-prefix-cascade.md) | N02, 공유 workload 근거 | 공유 이득·비공유 fallback·COW 정확성 |
| 조건부 | [N05 RMSNorm–GEMM 재배치](05-rmsnorm-gemm-reorder.md) | N02, 제거 비용 상한 | 수치 계약·순이득·기여 가능성 |
| 4 | [N06 모델·serving 통합](06-model-serving-integration.md) | 앞선 판정에서 선택된 경로 | 1.7B/3B 수치·20GB·첫 serving 비교표 |
| 5 | [N07 SLO·동시성 채택 판정](07-serving-qualification.md) | N06 | 같은 조건의 vLLM throughput·latency·tail 비교 |
| 후속 | [H01/H02/D01 장비 확장](08-future-hardware.md) | native 계약, 실제 장비 | 장비별 runtime·모델·serving 증거 |

첫 구현 범위는 N01→N02→N03까지다. 그 결과에 따라 N06 진행 또는 N04/N05 조건부 실험을 선택한다. N03~N05를 전부 구현하는 것이 목표가 아니다. 후보가 모두 실패하면 native 기준의 범용 모델 평가 가치만 별도 판단하고 성능 개선을 선언하지 않는다.

## 현재 구현 상태와 해석 제한

- N01은 `benchmarks/next_stage/`의 모델·수치·20GB 계약 validator, `benchmarks/analysis/n01_experiment_lifecycle.py`의 five-phase receipt controller, schema와 unit test를 포함한다. controller는 실행 전에 descriptor/profile/materialized checkpoint의 실제 SHA-256, target model/profile linkage, duplicate JSON key와 manifest-relative working directory를 검증한다. N01 v1은 physical GPU 0 한 장만 허용한다. controller 자체에는 Blender lifecycle 동작이 없고 direct Blender/restore-helper argv는 거부한다.
- receipt의 GPU 값은 `nvidia-smi`로 얻은 **sampled observed peak**다. warmup과 timed-serving에 실행 중 poll이 없으면 receipt는 실패하며, poll이 있어도 연속 high-water 또는 serving performance/correctness qualification은 아니다.
- N02의 첫 control은 `crates/riley-cuda/tests/gemm_gpu.rs`에 있는 Qwen2.5-3B projection shape의 deterministic synthetic BF16 prepared-GEMM test다. 이는 full Qwen weights, attention, HTTP serving, 모델 품질, 또는 vLLM 비교가 아니다.
- 따라서 이 상태에서 성능 비교표는 만들지 않는다. 첫 표는 N06의 full-model serving 통합 이후 동일 workload에서 작성한다.

## 유지할 계약

- strict 경로 보존, 별도 native-BF16 profile, weight/KV BF16, Rust→C/C++ ABI→CUDA. runtime Python 금지. 오프라인 build/reference 도구는 가능하다.
- RTX4090 한 장의 전체 GPU peak 20,000,000,000 bytes 이하. weight/KV/workspace/graph/context/packing·로딩 중복/외부 사용량 포함, 추정 불확실성 1GB 이상. CPU RAM도 별도 확인한다.
- 135M strict 회귀 → SmolLM2-1.7B 연결 → Qwen2.5-3B 주 평가. 7B는 선택적 후속. 작은 모델이 항상 KV도 작다고 가정하지 않는다.
- 수치 기준은 성능 측정 전에 고정한다. 실수 동치나 BF16 dtype만으로 strict 출력 일치를 주장하지 않는다.
- native microbenchmark, 모델 품질, 실제 serving 성능을 구분한다. 현재 새 결과는 모두 미측정이다.

## 가능성 gate

R0는 소스·수치·SM89·ABI·메모리 지원, R1은 제한된 operator/layer 순이득, R2는 선택된 후보의 full-model/serving 검증이다. `T_new=T_unchanged+T_candidate+T_added`로 비용을 합산한다. 대상 비중 f, 해당 speedup s, 추가 비용 비율 h라면 직렬 근사 `S=1/(1-f+f/s+h)`를 사용하되 실제 serving 예측치로 단정하지 않는다. 비용 비중이 미측정이면 operator 유망까지만 판정한다.

## Blender 운영 변경 — 사용자 지시 반영

2026-09-15 기존 세 Blender 프로세스의 파일/argv identity를 확인하고 SIGTERM으로 종료했으며 모두 종료를 확인했다. 이후 GPU 사용 여부와 관계없이 **다시 복구하지 않는다**. 원격 receipt: `/data/riley-serving-260913-recovery/blender-restoration-260914/user-stop-no-restore-260915.json`.

새 실험 lifecycle에 Blender 복구를 완료 조건으로 넣지 않는다. 기존 역사적 archive의 복구 증거/검증은 수정하지 않는다. 종료 직후 GPU 사용량은 440MiB였으므로 Blender 종료를 GPU 메모리 0 또는 측정 환경 무간섭으로 해석하지 않는다. 기존 정적 웹 뷰어·Xvfb·다른 서비스는 종료 범위에 포함하지 않았다.

## 성능 보고와 종료

N06 첫 serving 통합과 N07 qualification에서 [기존 비교표 양식](../BENCHMARK_REPORT_TEMPLATE.md)에 모델/revision, 엔진/backend, workload, precision, memory, 반복/표본, throughput, TTFT/TPOT, P95/P99, 실패율, quality, raw 경로를 남긴다. 각 셀에서 vLLM 이상 throughput 및 이하 TTFT/TPOT가 최소 기준이고 +15%/+10% latency 감소가 목표다. 높은 concurrency의 tail·안정성도 확인한다. 미지원·미측정·미확정은 pass와 구분한다.

이번 단계에서는 로컬 N01/N02 기반을 구현·검증했으며, 다음 단계는 같은 contract로 원격 GPU operator-control receipt를 만든 뒤 full-model/serving 여부를 별도 판정하는 것이다.
