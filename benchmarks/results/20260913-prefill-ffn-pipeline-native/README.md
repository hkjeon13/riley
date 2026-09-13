# Prefill FFN pipeline — native gate passed, model integration pending

[384-request serving profile](../20260913-load-shape-profile/README.md)을 근거로 prefill gate/up와 down projection의 shared staging/copy-MMA pipeline을 함께 구현했다. Native 출력은 기존 kernel과 bitwise 같고 제한된 memcheck/racecheck는0 errors다. 작은M에서는 시간이 줄었지만 M398은 거의 동률이다. **전체 모델/serving 성능은 아직 측정하지 않았다.** Default backend 및 serving binary는 변경하지 않았다.

## 구현

`kernels/optional/prefill_ffn_pipeline.cuh`는16-row CTA의 입력을 warp 간 공유하고 K64 두-stage `cp.async` 전달을 다음 K16 MMA와 겹친다. Gate/up는4warp, down은2warp 출력 분할이다. 기존 gate/up BF16 rounding→SwiGLU와 down K320 구간별 BF16 rounding 및 마지막 K256 누적 순서를 유지한다. 전체 CTA가 copy commit/wait와 shared 재사용 barrier에 참여한다. 마지막 live M row 밖 입력은 안전한 주소에서 zero-fill하고 출력을 쓰지 않는다.

초기64-element row stride의 구현은 모든 correctness case가 통과했지만 native pair 시간이 회귀했다. 입력 row들이 같은 shared-bank 위치에 겹치는 배치였고,72 BF16 stride의 padding으로 변경했다. [초기 코드](initial-stride64.cuh)와 v1 build/probe 로그를 보존한다. Padding만 적용한 v2가 아래 결과다. 실제 bank-conflict counter는 수집하지 않았으므로 latency 회귀의 유일한 원인을 계측으로 확정했다고 주장하지 않는다.

Padding 후 실제 gate shared20,992B/register38, down shared8,704B/register39다. 두 kernel local memory0이며 ptxas에도 spill0이다. 이 리소스 사용량은 occupancy 또는 HBM 절감량의 증거가 아니다.

## Native 시간 — serving throughput과 구분

RTX4090 SM89, CUDA nvcc13.0.88, `-std=c++17 -O3 -arch=sm_89 -Xptxas=-v`. 동일 input/weight와 capacity1024에서 기존 gate/up→down graph와 후보 graph를 각각10회 warmup,100회 event timing했다. 순서 A/B와 B/A, 표는 두 값의 중앙값이다. Fixed one-layer weights가 warm인 native screen이며30-layer working set·model queue·TTFT/TPOT를 포함하지 않는다.

| Live M | 기존 두 kernel µs | 후보 µs | 시간 변화 |
|---:|---:|---:|---:|
| 32 | 21.734 | 17.521 | -19.39% |
| 128 | 22.810 | 19.236 | -15.67% |
| 398 | 48.404 | 48.220 | -0.38% |
| 512 | 57.605 | 53.775 | -6.65% |
| 1024 | 109.138 | 107.776 | -1.25% |

[Raw v2 결과](prefill-ffn-probe-v2.log), [집계](comparison.json). M398의0.38%, M1024의1.25% 차이를 안정적인 이득으로 보지 않는다. Native M32의19% 감소를 전체 serving 개선율로 사용하지 않는다. 이 단계에는 새 vLLM 비교표가 없다.

## Correctness·도구 결과

- 기존과 후보의 gate intermediate/down output 전체 및 inactive sentinel을 대조했다. Live M0/1/15/16/17/31/32/64/128/398/512/1024/1025, 입력 변경을 포함한 두 반복26case에서 mismatch0, inactive write0. M0/1025는 기존과 같이 no-work이며 descriptor 검증의 대체물이 아니다.
- [Memcheck](prefill-ffn-memcheck.log), [racecheck](prefill-ffn-racecheck.log): 각각 exit0, errors0/hazards0. 각각90초 timeout으로 두 native kernel 및 reference의 M0/17/398/1025만 실행했다. Whole-model sanitizer를 실행한 것이 아니다.
- SM90a·SM100a object compile 통과. [Manifest](manifest.json)에 source/binary/object SHA·크기와 nvcc version이 있다. 두 target runtime은 장비 부재로 미검증이다. 4090에서 실행된 실패를 skip으로 바꾸지 않았다.
- 측정 후 GPU compute process 없음. Candidate는 독립 probe에서만 실행되며 Rust serving에 Python이 들어가는 변경은 없다.

## 재현과 다음 단계

`benchmarks/analysis/prefill_ffn_pipeline_probe.cu`를 위 nvcc 옵션으로 빌드한다. 기본 실행은26 correctness case와 timing, `--sanitizer` 실행은4개 case만 수행한다. Build/readback log 및 object는 `ai-assistant:/data/riley-serving-260913-recovery`에도 남아 있다.

PR05 batch의 **native gate만 완료**했다. 다음은 별도 native/model graph profile·catalog digest, retained workspace/shape 계약, Rust opt-in backend 선택을 연결하는 작업이다. 이어서 전체30-layer logits/greedy/natural reference/stop/cancel을 검증하고 현재 single/직전 paired/새 후보/vLLM을 C32 및 client C64/active32 두 순서로 비교한다. 모델이나 serving이 회귀하면 승격하지 않는다. Native 이득이 확인된 shape만으로 workload를 바꾸거나 기존 품질 gate를 낮추지 않는다.
