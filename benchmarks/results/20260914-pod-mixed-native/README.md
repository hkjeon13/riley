# POD-style SM-aware mixed attention — native gate

POD의 CTA-parallel 배정 방식을 현재 Riley attention arithmetic에 적용한 native 후보를 만들었다. **대표 mixed shape에서 기존 대비 graph 시간이 약30–52% 늘어 serving에 연결하지 않는다.** POD 전체 구현이나 논문의 성능을 재현한 결과가 아니며, 기본 serving 경로는 바뀌지 않았다.

## 연구 적용과 구현 batch

[POD-Attention §4](https://arxiv.org/html/2410.18038v2#S4)는 SM별 CTA ticket과 작업 counter를 이용한 runtime role binding, proportional/alternating 배정, warp 단위 virtual decode CTA 및 shared-memory 균형을 설명한다. 2026-09-14 primary source를 확인했다. 원 논문의 FlashAttention 기반 tile/resource 선택과 Riley의 기존8-query/one-warp 구현은 다르므로 논문 배수를 이식 효과로 가정하지 않는다.

1. 현재 packed tile/head 작업을 prefill/decode queue로 분리하고 개수와 SM ticket/counter를 준비하는 CUDA plan을 추가했다. Parent metadata와 page extents는 caller가 검증·보유해야 하며 Plan은 stream별로 graph 완료까지 유지한다. Prototype은 replay마다 prepare를 실행하고 그 비용을 timing에 포함한다.
2. 원래 attention device body의 block ID, lane, shared storage를 명시적 인자로 바꿔4개의 독립 warp 작업을 한 CTA에서 실행한다. Prefill의8-query tile, 작은 prefill/decode의 single-query fallback, reverse128-token softmax, K16 MMA 순서와 BF16 rounding을 보존한다. Nonfinite V fallback도 그대로 사용한다. Default header/kernel은 편집하지 않았다.
3. Mode1은 queue packing만 적용한 static CTA 배정, Mode2는 SM ticket 교대, Mode3은 proportional SM 배정이다. Virtual decode 작업은 warp별 shared slice와 warp barrier를 사용한다. Role exhaustion은 다른 queue로 전환한다. 서로 다른 작업이 같은 출력에 쓰지 않는지 audit coverage로 검증한다.
4. Audit와 측정 kernel을 compile-time으로 분리하고 양쪽 bitwise를 검사한다. SM ID4096 이상은 명시적 error이며 그 하드웨어의 runtime을 지원한다고 주장하지 않는다. Workspace122,900B는 고정 prototype 상한이며 production allocator에 연결하지 않았다.

이는 논문 아이디어를 적용한 자체 CUDA prototype이다. 외부 POD 라이브러리나 Python inference runtime을 연결한 것이 아니다. 기존 Riley도 mixed 작업을 한 kernel에서 처리하므로 단순 launch fusion 자체를 신규 이득으로 주장하지 않는다.

## 최종 v4 native timing

RTX4090, CUDA13.0, `-std=c++17 -O3 -arch=sm_89`. Q/K/V BF16,9Q/3KV heads·D64, page16, capacity1024 고정, 최대 context4096. CUDA graph10warmup+50timed replay, A/B/C/D 및 역순, 두 값 중앙값. Candidate는 queue prepare+compute, baseline은 기존 mapped attention이며 empty CTA 비용도 포함한다. One primitive이며30-layer model/serving 시간이 아니다.

| Prefill rows / decode rows / decode context | 기존 µs | Static µs | SM 교대 µs | SM 비례 µs |
|---|---:|---:|---:|---:|
| 128 / 31 / 512 | 44.24 | 67.71 | 67.21 | 67.18 |
| 398 / 31 / 1024 | 97.97 | 131.45 | 127.84 | 128.93 |
| 512 / 31 / 4096, prefill start3584 | 555.19 | 728.35 | 723.58 | 721.17 |
| 0 / 32 / 4096 | 327.12 | 516.09 | 512.11 | 526.84 |
| 1024 / 0, prefill start3072 | 700.87 | 730.79 | 702.13 | 731.30 |

[전체13 shape 비교](comparison.json), [반복별 log](pod-mixed-probe-v4.log). Empty/one-token 경우에는 적은 inactive CTA 때문에 빨라지지만 serving 성공으로 계산하지 않는다. P31/D8/context1024에서는 SM 교대가−4.6%였으나 인접 P32에서는 크게 회귀했다. 이 일부 이득을 골라 채택하지 않는다. Shared-host event timing이며 hardware counter·clock isolation·model weight working-set proof가 아니다.

## 정확성·수명·자원 검증

- [전체 check-only 및 captured graph replay](pod-mixed-check-v5.log):13 shape×3 query/V 상태, audit/release 양쪽3후보 총234개 full output 비교에서 bitwise 차이0. Audit coverage117회가 모든 valid tile/head를 정확히1회 실행하고 padding 작업은0회임을 확인했다. Empty, small query tile 경계1/15/16/17/31/32,128/398/512/1024, ragged owner/page mapping, 최대4096context 및 NaN V fallback을 포함한다. 전체 output buffer의0x5555 padding sentinel도 기존과 비교했다. 동적 metadata·입력을 같은 allocation에 바꿔 기존 captured graph를 반복 실행해 검사했다.
- [Memcheck](pod-mixed-memcheck-v5.log)0 errors, [racecheck](pod-mixed-racecheck-v5.log)0 hazards. 각90초 timeout으로 empty, P17/D4/context129, P128/D8/context256(start16)의3shape×3상태에 한정했다. 각각54개 audit/release output 비교 및27 coverage 검사다. 전체 모델 racecheck는 실행하지 않았다.
- [SM89 build](pod-mixed-build-v5.log): candidate163 registers,24,584B shared,0 local/stack/spill. 기존 mapped attention은127 registers,6,144B shared. CUDA occupancy API의 이론상 상한은 기존14 CTA/14 warp, 후보3 CTA/12 warp per SM이다. 실제 achieved occupancy나 stall counter 측정은 아니다.
- Audit P398/D31에서는 같은 SM이 두 역할을 받은 수가 static1, 교대61, 비례60이었다. 배정 방식이 동작했음을 보여 주지만 **시간적으로 겹친 resident execution의 증거는 아니다**. Audit와 측정 버전을 섞어 co-location 성능을 주장하지 않는다.
- SM90a/SM100a 최종 source object compile 통과. Runtime은 해당 장비 부재 미검증이며 multi-GPU 실행은 미구현이다. 이 compile을 Hopper/Blackwell serving qualification으로 부르지 않는다.
- [검증 집계](verification.json), [파일·binary·object·source SHA manifest](pod-native-manifest.json). 원본은 `ai-assistant:/data/riley-serving-260913-recovery`에 유지한다. 내려받은 log와 로컬 source SHA를 원격과 대조했다.

## 과정과 판정

v1 timing 및 v2 nonfinite/sanitizer 결과를 보존했다. Audit runtime pointer를 NULL로만 두던 kernel의 register 배정 영향을 배제하려고 compile-time Audit specialization을 추가했다. 첫 v3 build는 template 쉼표를 받지 못한 probe macro 때문에 실패했다([로그](pod-mixed-build-v3.log)); variadic macro로 고친 v4에서 모든 검증과 timing을 다시 실행했다. 실제 compile 실패를 hardware skip으로 바꾸지 않았다. Audit 분리 후에도 registers163과 주요 회귀가 유지돼 audit code만의 문제는 아니었다. 이후 v5는 kernel을 변경하지 않고 release correctness 호출을 이미 captured된 graph replay로 강화했으며 full check·sanitizer·SM90a/SM100a compile을 다시 통과했다. 성능표는 동일 kernel의 v4 timing이며 v5 진단 수치를 섞지 않는다.

SM role assignment만으로 현재 body의 register/shared-memory 구조와 작은 tile의 비용을 해결하지 못했다. Static packing도 느리므로 SM atomic 배정만을 유일한 원인으로 볼 수 없다. 이는 resource-envelope 관측과 일관되는 해석이며, 정확한 회귀 비율의 인과 분해는 아니다.

**Native gate에서 탈락했으므로 Rust recorder/serving 선택에 연결하지 않았다.** 새 vLLM 비교·full-model·고부하·soak는 미실행이며 통과가 아니다. 마지막 유효 serving baseline은 [prefill FFN integration](../20260913-prefill-ffn-model-serving/README.md)의 opt-in 후보/기본 single이다. 직전 [time feedback](../20260914-mixed-time-policy/README.md) 회귀 후보도 비활성 유지한다.

다음 attention backend 작업은 CTA 개수만 미세 조정하지 않고 prefill/decode별 register·shared-memory 계약, head/query tile 및 native library precision 경계를 함께 재설계해야 한다. 현재 작은8-query body를 묶는 것과 원 논문의 compute/memory 균형을 갖춘 kernel 통합은 구분한다. 최종 목표인 실제 serving에서 vLLM 초과는 아직 미달성이다.
