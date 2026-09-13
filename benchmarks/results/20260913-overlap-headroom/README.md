# 비동기 serving의 시간 예산 재계산

2026-09-13. 새 serving run이나 성능 개선 결과가 아니다. 기존 V56(binary SHA256 `08cd627049f948b94e4426d933b7aeab883c93a385e3f2d603efe1e6a2fb54a8`) Nsight SQLite를 읽어 graph 사이의 시간을 재계산했다. 자연 workload의 receipt는 96 streaming requests에서 reference text/token IDs 일치를 기록하지만 성능 qualification은 아니다.

## 계산

CPU `cudaGraphLaunch*`와 kernel/memcpy/memset GPU activity를 correlation ID로 대응시켰다. launch가 누락되거나 여러 stream에 걸리거나 GPU graph span이 겹치면 이 단일 stream 분석을 거절한다. 세 trace의 CPU launch 525/305/92개가 모두 GPU activity와 대응했다. 수집 controller에는 명시적 warmup phase가 없으며 시작부와 drain 구간도 포함한다. 별도로 잘라 낸 steady-state 측정이 아니다.

각 graph의 첫 GPU activity부터 마지막 activity까지를 span으로 잡는다. 인접 span 사이의 간격만 제거 가능한 후보로 계산한다. graph 안에서 GPU activity가 없는 시간은 별도로 남기며 scheduler host bubble로 계산하지 않는다. GPU 활동의 겹치는 구간은 union으로 집계한다.

| trace | launches | 전체 구간 ms | graph span ms | graph 사이 간격 ms | 간격 비율 | 간격을 전부 없앤 trace 상한 |
|---|---:|---:|---:|---:|---:|---:|
| natural C16 | 525 | 1072.387 | 936.733 | 135.654 | 12.65% | +14.48% |
| natural C32 | 305 | 783.382 | 646.494 | 136.888 | 17.47% | +21.17% |
| fixed C32 | 92 | 269.133 | 223.858 | 45.275 | 16.82% | +20.22% |

상한은 `전체 구간 / graph span 합계`다. CPU 작업과 client pacing, profiler overhead, launch 지연까지 모든 graph 사이 간격이 사라진다는 과도하게 낙관적인 가정이다. 따라서 **unprofiled serving의 이론적 상한이나 예상 개선율로 해석하면 안 된다.** 이 자료만으로 실제 asynchronous serving이 목표에 도달할 수 없다고 증명하지 않는다.

natural C32의 간격 중앙값은 337.460 µs이며, 이전 GPU 완료 후 다음 CPU launch 진입까지의 구간 중앙값은 329.832 µs다. graph 내부 비활동 시간 48.688 ms는 앞의 136.888 ms와 구분했다. 이 간격 전부가 scheduler CPU 계산이라는 attribution은 하지 않았다.

## 구현 우선순위에 반영

별도 unprofiled round62 serving screen에서 V56 natural C32 throughput은 10433.829 token/s, vLLM은 11950.605 token/s다. V56에서 vLLM +15% 목표로 가려면 throughput +31.72%가 필요하고, vLLM 대비 TPOT −10% 목표에는 현재 V56 TPOT를 약 28.02% 줄여야 한다. 두 역순 order의 screen이며 장기 안정성 qualification이 아니다.

위 serving 수치와 Nsight trace 상한은 **서로 다른 실행**이다. 둘을 곱해서 미래 throughput을 예측하지 않는다. 다만 async만으로 전체 목표를 해결한다고 가정해 scheduler 재작성을 계속하는 것은 현재 근거보다 강한 판단이다.

- PR 02의 검증된 submit/ticket, 두 staging slot, KV 부분 commit은 유지한다. GPU future token과 두 plan 예약·순차 commit은 미완료다.
- 다음 구현의 우선순위를 PR 03 attention adapter로 옮겨 GPU graph 자체를 줄일 수 있는 후보를 실제 모델에 연결한다. 기존 연구의 attention 비중과 FlashInfer HND/page16 호환성 검사는 이 방향을 뒷받침하지만, 실제 모델/serving 이득은 아직 증명하지 않는다.
- FlashInfer의 BF16 결과가 기존 kernel과 byte-exact하지 않았다는 기존 관측을 유지한다. 별도 수치 계약·모델 수준 검증 없이 기존 exact 경로를 교체하거나 tolerance를 완화해서 통과시키지 않는다.
- adapter가 실제 모델에서 유효하면 현재 baseline 및 vLLM과 같은 모델·하드웨어·워크로드로 throughput/TTFT/TPOT/P95/P99를 비교한다. PR 02와의 결합 효과는 두 기능이 실제 serving에 연결된 뒤 별도로 검증한다.

## 근거 및 재현

[분석 코드](../../analysis/overlap_headroom.py), [계산 검사](../../analysis/test_overlap_headroom.py). `python3 benchmarks/analysis/overlap_headroom.py TRACE.sqlite --output RESULT.json`으로 JSON과 launch별 CSV를 생성한다. 원본 SQLite 경로와 SHA256은 각 JSON에 있다.

[natural C16](natural-c16.json), [natural C32](natural-c32.json), [fixed C32](fixed-c32.json), [natural receipt](natural-receipt.json), [fixed receipt](fixed-receipt.json), [round62 원본 분석](serving-round62-analysis.json), [독립 serving 목표 차이 계산](target-gap.json).

CPU interval 계산 검사 4개를 실행했다. 이 검사는 GPU activity union, graph 안/사이 시간 분리, 미리 제출된 launch의 음수 host gap 방지, overlap/불완전 window 거절을 확인하며 GPU 실행이나 serving 개선을 검증하지 않는다.
