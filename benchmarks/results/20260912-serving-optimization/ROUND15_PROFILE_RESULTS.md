# Round15: RoPE/KV + attention 결합 구간 진단

**Batch8의 fused 구간은 baseline의 두 enqueue를 함께 잰 구간보다 AB에서 17.188%, BA에서 17.496% 오래 걸렸다.** Operator timing을 끈 whole decode도 candidate가 느렸다. 이는 [Round14의 serving 회귀](ROUND14_RESULTS.md)가 나타난 구간을 좁히는 근거이며, 특정 SASS 명령·register pressure·occupancy를 원인으로 확정한 결과는 아니다. Batch8 기각과 Batch7 API baseline 유지 결정은 바뀌지 않는다.

## 직접 측정한 구간

Baseline은 원래 packed RoPE/KV 저장과 two-warp attention의 연속된 두 enqueue를 **하나의 CUDA event 구간**으로 감쌌다. Candidate는 fused enqueue 하나를 같은 방식으로 측정했다. 매 decode replay에서 30개 layer의 구간을 더한 뒤 retained 186개 replay의 중앙값을 구했다. 별도로 측정한 RoPE와 attention 중앙값을 합산하지 않았다.

| 실행 순서 | Baseline 합계 (ns) | Batch8 합계 (ns) | Batch8 / baseline | 차이 (ns) |
| --- | ---: | ---: | ---: | ---: |
| AB: baseline → candidate | 360,448 | 422,400 | 1.171875 | +61,952 |
| BA: candidate → baseline | 359,936 | 422,912 | 1.174964 | +62,976 |

아래는 같은 진단 바이너리에서 **operator timing off**와 on을 비교한 whole-graph 중앙값이다. Off도 기존 whole-graph timing은 유지하므로 완전히 계측을 제거한 serving 바이너리가 아니다. 괄호는 각 lane의 off-before 대비 on의 증가율이다.

| Lane / graph | Off-before (ms) | Combined AB (ms) | Combined BA (ms) | Off-after (ms) |
| --- | ---: | ---: | ---: | ---: |
| Baseline decode | 0.943216 | 1.070864 (+13.533%) | 1.074832 (+13.954%) | 0.943296 |
| Batch8 decode | 1.003232 | 1.126624 (+12.299%) | 1.135072 (+13.142%) | 1.009632 |
| Baseline prefill | 4.611040 | 4.602256 | 4.618128 | 4.617296 |
| Batch8 prefill | 4.611136 | 4.609824 | 4.616592 | 4.619264 |

On은 decode graph에 60개 event node를 추가했다. Prefill에는 operator event를 넣지 않았고 추가 stream synchronization도 없다. Decode off-before/after drift는 baseline **+0.008%**, candidate **+0.638%**다. On의 약 **12–14%** 교란 때문에 구간 값을 실제 serving latency나 순수 kernel 비용으로 대체할 수 없다. 두 AB/BA 쌍은 진단 범위이며 새 throughput·TTFT·TPOT 승리, tail 안정성 또는 candidate acceptance를 주장하지 않는다.

## 실행과 검증 범위

8개 독립 C1 프로세스를 `off baseline/candidate → combined AB → combined BA → off baseline/candidate` 순서로 실행했다. 각 프로세스는 P128/O32, GPU greedy, nonstream warmup 5개 + stream warmup 5개 + retained stream 6개다. 모든 **128개 HTTP 응답**이 strict prompt/output ID·text·finish·usage 검증을 통과했다. 전체 native **4,096 replay** 중 retained는 **1,536개**다. 프로세스별 replay **321–512**만 통계에 포함해 **6 prefill + 186 decode**를 남겼으며 선택된 30개 layer interval에 누락은 없다.

동일 RTX 4090와 private 580.173.02 runtime을 사용했다. GUI를 유지하고 허용된 Blender 세 개만 일시 중지했으며, 시작 온도 ≤48°C / idle GPU memory ≤512 MiB / foreign CUDA compute 없음 조건이다. Canonical headless 결과로 표시하지 않는다. Parent Round14의 source/model/reference 검증과 프로세스 종료·복원 gate를 그대로 확인한 뒤 진단했다. 기존 serving receipt는 변경하지 않았다.

| 진단 lane | 원본 source commit | 별도 build receipt |
| --- | --- | --- |
| Batch7 API baseline | `a179617070526068b66ba5627ba82a7151da8c64` | [decode7-combined-profile-build.json](raw/decode7-combined-profile-build.json) |
| Batch8 | `8329c1aeec6e013f581128888c536e15f8bf7300` | [decode8-profile-build.json](raw/decode8-profile-build.json) |

Generated `graph_resources.cu` SHA256은 baseline `91d473d01047dc8ad6dad475a567b92d3a290e33bb0a7ed20359d719642cd485`, candidate `604b99c668729667cd39ef3d42f3fcd5c0dfd8088cc8f3277ae4d6f85b27f372`다. 각 build receipt에 원본·계측 도구·binary 해시와 의도적인 instrumentation diff가 묶여 있다. 상세 계측 계약은 [PAIRED_ROPE_ATTENTION_PROFILE.md](PAIRED_ROPE_ATTENTION_PROFILE.md)에 보존한다.

[Completion](raw/paired-rope-attention-round15/completion.json)은 8개 프로세스 완료와 복원을 확인했고 [finalization](raw/paired-rope-attention-round15/finalization.json)의 failure는 `null`이다. [복사본 독립 검증](round15-copied-results-verification.json)은 native 4,096개와 HTTP 128개를 다시 읽어 검증했다. Python 3.10/3.13 합산 방식 차이로 파생 `.sum` 11개에서 최대 **1.79×10⁻⁷ ns** 차이만 기록됐으며, raw 값·percentile·최종 비교는 정확히 일치했다. 이 차이를 숨기거나 원본 JSON을 수정하지 않았다.

Blender는 **2730220 / 2730294 / 2730392**, birth ticks **84083612 / 84083660 / 84083711**, 포트 **9876 / 9911 / 9887**로 복원됐다. [복원 receipt](raw/blender-round15/verified.json)와 **2026-09-12 12:29:26 KST**의 [실제 프로세스 재확인](raw/round15-live-restoration-recheck.json)이 있다. 명령·GUI 환경·private GL readiness와 관측된 vendor mapping을 확인했다. CUDA는 첫 프로세스에만 로드됐고 나머지 두 개는 lazy loading 상태였다.

## 후속 다중 요청 구성요소 준비 상태

아래는 별도 GPU 수치 probe이며 Round15 timing이나 serving 통합 검증으로 집계하지 않는다.

| 구성요소 | 실제 결과 | 남은 범위 |
| --- | --- | --- |
| [Row attention](raw/multisequence-attention-probe/result.json) | Synthetic 34,560개 case와 retained graph transition 3개 모두 bitwise exact. Allocation 432,017개 전부 해제, live bytes·cleanup error 0. | Scheduler/graph/server와 실제 모델 다중 요청 통합 및 serving 성능 미검증. |
| [Row precise 연산](raw/multisequence-precise-probe/result.json) | Synthetic 3,456개 case와 retained graph transition 3개 모두 bitwise exact. Allocation 107,179개 전부 해제, live bytes·cleanup error 0. | 동일한 통합·실제 모델·성능 검증이 필요. |
| [Anchored projection V2](raw/multisequence-projection-probe-v2/run/receipt.json) | M1 anchor 5개는 허용됐지만 M2/M4 child 10개가 모두 unsupported. 계획된 1,089개 case는 전부 skipped, 실행 0개. | 수치 동등성 증거가 없으며 standalone 후보를 준비 중. 다른 heuristic으로 자동 대체하지 않았다. |

Probe의 `completed`는 실험 종료를 뜻한다. 특히 anchored projection의 `arithmetic_equal=false`와 `serving_qualified=false`를 성공으로 바꾸지 않는다. 통합 전 준비 중인 구성요소이며 새로운 optimization batch 채택이나 높은 concurrency 성공으로 계산하지 않는다.

주요 SHA256: [진단 비교](raw/paired-rope-attention-round15/comparison.json) `05be2a65ad78aa78f2e3346118c812d01e8172e82e1b094f43696ffaed1bc2bb`; [복사본 검증](round15-copied-results-verification.json) `bb064bcb6a0ea30e77fe5d37550ec8fd2e2713eb4d7afcdf88541edd44a955b8`; [완료](raw/paired-rope-attention-round15/completion.json) `fec1ddb624f8e774000620c02ec3987f0f200bd20cdf2a2fa5014ce2e536cb36`.
