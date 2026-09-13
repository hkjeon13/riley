# Shared probability storage compaction — native resource gate

기존 exponential FP32 배열과 별도 BF16 probability 배열의 중복 저장을 제거했다. **단일-warp mapped 경로는 대표 shape에서1.6–3.7% 빠르지만, POD 조합의 주요 회귀는 해소되지 않았다.** Kernel native 결과이며 serving 개선으로 승격하지 않는다.

## 하나의 resource-layout batch

- Small-query/decode와8-query prefill body에서 FP32 exponential을 한 번 저장한다. PV MMA의 probability operand를 읽는 시점에 같은 `__float2bfloat16_rn`을 적용해 별도 BF16 mirror를 제거한다. Denominator는 기존 FP32 배열·순서로 계산하며 reverse128 tile, causal mask, MMA 순서, nonfinite V fallback을 보존한다.
- 단일-warp direct mapping과 기존 SM-aware4-warp POD 배정에 같은 저장 규약을 적용했다. Direct 경로는 queue prepare 없이 기존과 같은 grid를 사용한다. POD 경로는 이전 queue/SM role 정책과 lifetime을 유지해 저장 변화의 효과를 분리한다.
- Audit/release를 compile-time 분리하고 dynamic input/metadata의 captured graph replay를 포함한 bitwise·coverage 검증을 실시했다. 기본 kernel과 Rust serving은 변경하지 않았다. Runtime에 Python을 추가하지 않았다.

POD 연구의 shared-memory 균형 요구는 [이전 native gate](../20260914-pod-mixed-native/README.md)에 있다. 이 batch는 그 실패에서 관측된 자원 부담에 대한 구현 실험이며 새로운 논문 배수나 하드웨어 counter 결과가 아니다.

## 같은 screen의 native timing

RTX4090, CUDA13.0, SM89/O3, capacity1024·9Q/3KV heads·D64·page16·최대 context4096. CUDA graph10warmup+50timed, mode0/1/2/3 및 역순. 두 값 중앙값이다. Mode0 기존,1 compact direct,2 직전 POD 교대,3 compact POD 교대. POD timing은 prepare 비용을 포함한다. One attention primitive이며 모델 working-set·serving 결과가 아니다.

| Prefill / decode / decode context | 기존 µs | Compact direct µs | 직전 POD µs | Compact POD µs |
|---|---:|---:|---:|---:|
| 128 / 31 / 512 | 44.28 | 43.56 | 67.21 | 67.30 |
| 398 / 31 / 1024 | 97.99 | 94.33 | 126.89 | 127.81 |
| 512 / 31 / 4096, prefill start3584 | 554.59 | 556.75 | 724.60 | 661.90 |
| 0 / 32 / 4096 | 335.45 | 330.60 | 524.40 | 528.10 |
| 1024 / 0, prefill start3072 | 700.18 | 701.52 | 702.48 | 665.42 |

[전체13 shape·변화율](comparison.json), [두 순서 raw](compact-mixed-probe-v2.log). Direct는 작은/중간 workload에서 약1–4% 낮지만 긴 prefill에서는0.2–0.4% 늘어 거의 동률이다. Compact POD가 일부 long shape에서 이전보다 나아도 대표 mixed에는 기존보다 약30–52% 느리고 다른 shape도 크게 회귀한다. 특정 row의 이득만 골라 채택하지 않는다. Hardware clock isolation/long soak는 없으며 작은 변화는 제한적으로 해석한다.

## Correctness와 자원

- [전체 check](compact-mixed-check-v2.log):13 shape×3 finite/query/V 상태,3후보의 audit/release 양쪽 총234개 full-output bitwise 비교 및117 coverage 검사 통과. Release는 미리 captured한 graph를 재사용한다. Empty,1/15/16/17/31/32/128/398/512/1024 query 경계, 최대4096 context, ragged owners/pages, inactive0x5555 sentinel, NaN V fallback을 포함한다.
- [Memcheck](compact-mixed-memcheck.log)0 errors, [racecheck](compact-mixed-racecheck.log)0 hazards. 각각90초 timeout, empty/P17-D4-context129/P128-D8-context256(start16)의3shape×3상태로 제한했다. 각각54개 output 비교·27 coverage 검사다. Whole-model sanitizer는 미실행이다.
- [Build resource](compact-mixed-build-v2.log): direct126 registers/4,096B shared/0 local, compact POD159 registers/16,392B shared/0 local. 기존 direct의127/6,144B와 이전 POD163/24,584B에 비해 storage는 약1/3 감소했다. Stack/spill은0이다. 이 값은 실제 HBM bytes 절감량이 아니다.
- CUDA occupancy API는 원래 direct14 CTA/14 warp per SM, compact POD3 CTA/12 warp를 보고했다. Shared memory를 줄여도4-warp POD의 register 제한이 남아 CTA 상한은 변하지 않았다. Achieved occupancy/stall counter로 해석하지 않는다.
- SM90a/SM100a object compile 통과. 해당 하드웨어 runtime 및 multi-GPU는 미검증이다. 장비 부재를 4090에서 실행된 실패와 혼동하지 않는다.
- [검증 집계](verification.json), [원본·binary/object/source SHA](compact-native-manifest.json). 원본은 `ai-assistant:/data/riley-serving-260913-recovery`에 유지한다. 내려받은 log와 최종 로컬 source SHA를 대조했다.

최초 compile은 이전 Plan type을 재사용하면서 ADL이 이전/new `prepare`·`execute`를 함께 찾는 ambiguity로 실패했다([로그](compact-mixed-build.log)). Namespace를 명시한v2에서 다시 build·전체 검사·timing·sanitizer·portable compile을 통과했다. 이 실패를 hardware skip으로 바꾸지 않았다.

## 판정과 남은 범위

저장 규약 자체는 native correctness gate를 통과했지만 **POD 목표인 대표 mixed attention 성능 개선은 미달**이다. Rust adapter/graph identity/전체 모델 검증·vLLM serving 비교는 아직 하지 않았다. 따라서 마지막 serving baseline 및 기본값은 유지한다. Direct의 작은 native 이득은 다음 의미 있는 attention backend batch에 합칠 후보로 남기고, 이것만으로 별도 serving 개선을 주장하지 않는다.

이제 shared mirror 제거만으로 POD resource envelope가 해결되지 않는다는 근거가 있다. 다음 단계는 prefill/decode phase별 live register 상태와 native backend/precision 계약을 함께 다뤄야 한다. 단순 CTA 상수나 time feedback threshold를 계속 탐색하는 것으로 대체하지 않는다. Hopper/Blackwell의 별도 실행 방식도 지원 계약에 포함하되 미검증 runtime을 완료로 표시하지 않는다. 전체 vLLM 초과 serving 목표는 계속 미완료다.
