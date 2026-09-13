# Mixed chunk sensitivity and execution-cost calibration

현재 scheduler에는 이미 decode-first mixed batching, 최대4개 prefill 요청, 고정 token budget과 chunk limit가 있다. PR04를 시작하며 고정 chunk 축소의 효과를 다른 변경과 분리했다. **512→128은 throughput−9.91%, 512→256은−1.68%**로 이번 workload에서 채택하지 않는다. 시간 기반 정책이나 POD kernel의 구현 완료를 뜻하지 않는다.

## 연구와 현재 적용 경계

[Sarathi-Serve](https://arxiv.org/abs/2403.02310)는 chunked prefill과 decode stall을 줄이는 scheduling을 사용한다. [POD-Attention](https://arxiv.org/abs/2410.18038)은 hybrid batch attention의 prefill/decode에 GPU 자원을 나눠 동시 실행한다. [NanoFlow](https://arxiv.org/abs/2408.12757)는 서로 다른 GPU 자원 사용을 겹치는 실행을 다룬다. 2026-09-14 primary source를 재확인했다. 기존 mixed batching을 POD 구현으로 부르지 않으며 논문 배수를 Riley 기대 성능으로 사용하지 않는다.

이번에는 checkpoint·kernel·paired execution·active32·iteration budget512를 고정하고 `--prefill-chunk-tokens`만128/256/512로 바꿨다. 총 graph capacity나 vLLM 설정을 함께 바꾸지 않았다. 이 실험은 causal sensitivity screen이며 vLLM 승격 비교가 아니다. 직전 동일 backend의 vLLM 비교는 [모델·serving batch](../20260913-prefill-ffn-model-serving/README.md)에 있다. 그 측정의 수치를 이번 denominator로 가져오지 않는다.

## Uninstrumented C32 screen

RTX4090, SmolLM2-135M BF16, frozen natural16/128/398 input 및32/64/128 output, prefill FFN paired. 각 lane warmup192+retained768, fresh server. Chunk512→128→256 및256→128→512. GUI 유지, Blender 종료, CPU isolation/clock lock 없음. Throughput/P50은 두 반복 중앙값, E2E P95/P99는 더 나쁜 반복이다.

| Chunk limit | Output tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| 512 기준 | 10,953.96 | 10.543 | 2.702 | 375.381 | 489.413 |
| 128 | 9,868.44 | 10.254 | 3.030 | 433.200 | 489.435 |
| 256 | 10,769.49 | 11.415 | 2.702 | 386.096 | 475.011 |

[반복별 결과](completion.json). Retained4,608개 오류0, warmup 포함5,760개 Riley reference exact. Chunk128은 TPOT 악화, chunk256은 P99 일부 감소 외에 throughput/TTFT/P95에서 불리하다. 긴 prefill·open-loop arrival·다른 모델로 일반화하지 않는다.

최초 harness가 잘못된 CLI 키 `--max-prefill-chunk-tokens`를 찾다가 server 실행 전에 실패했다. [실패 로그](evidence/initial-cli-option-failure.log)를 보존하고 실제 키로 수정한 v2에서 전체 screen을 완료했다. GPU 테스트 skip이나 runtime 회귀로 분류하지 않는다.

## Cost diagnostics

기존 `RILEY_SERVING_PHASE_TIMING=1`에만 bounded batch histogram을 추가했다. Key는 total rows의16단위 상한, decode rows, prefill request 수, 최대 target context의128단위 상한이다. 각 key의 성공한 ordinary iteration 수·execute host wall 합계·최소·최대를 shutdown 시 출력한다. Map4096 key 초과는 overflow count에 남기며 데이터를 몰래 버리지 않는다. Paired decode는 기존 pair phase 집계에 남고 이 histogram에서는 제외한다.

Execute wall에는 authority 준비·전송·GPU 대기·download/validation이 포함된다. 순수 GPU 시간, CPU 사용량, HBM traffic, percentile로 해석하면 안 된다. 시간 예산 정책에 바로 넣는 predictor가 아니라 비용 구간과 데이터 공백을 찾는 calibration 자료다. 진단 비활성 시 shape 수집 및 map 작업은 실행하지 않는다.

별도 짧은 진단 screen은 warmup32+retained192, 같은 chunk 두 순서다. 진단 수치를 위 uninstrumented throughput과 섞지 않는다. [분석기](../../analysis/mixed_batch_cost.py)는 sample count/overflow 보존과 합계 범위를 검증한다. 합계 손실·불가능한 mean을 거부하는3개 테스트와 기존 phase parser3개 테스트가 통과했다. CUDA release build11.40초 통과.

## Calibration 결과

[Shape별 비용](cost-shapes.json), [원본 manifest](evidence/raw-manifest.json). 6개 lane 모두 ordinary step 합계와 histogram+overflow가 일치하고 overflow0이다. 진단 retained1,152개와 warmup192개가 reference exact이며 모든 server exit0이다.

| Chunk | Mixed/prefill iteration 수 (두 순서) | 누적 execute wall ms (두 순서) |
|---|---:|---:|
| 512 | 138 / 134 | 642.33 / 626.25 |
| 128 | 211 / 195 | 873.13 / 814.94 |
| 256 | 156 / 148 | 691.71 / 664.27 |

각 lane 전체224요청을 포함한다. 작은 chunk는 iteration당 비용을 일부 낮춰도 iteration 수와 누적 비용이 늘었다. 이것은 uninstrumented 회귀와 일관되는 해석이며 진단 pacing·batch 구성이 달라 정확한 인과 분해나 GPU 비용 추정으로 주장하지 않는다. `cost-shapes.json`은 bucket 상한을 보관하므로 그 값을 실제 모든 요청의 context로 취급하지 않는다.

원본 full SSE frames는 `ai-assistant:/data/riley-serving-260913-recovery/chunk-sweep-c32-v2` 및 `chunk-cost-c32-v1`에 보존한다. 로컬 request gzip은 frames만 제거했다. 각 preparation의 binary/fixture/controller hash로 uninstrumented binary와 진단 binary를 구분한다. 진단 추가 외에 scheduler 선택·수치·kernel은 변경하지 않았다.

## 다음 결정

고정 chunk를 줄이는 정책은 채택하지 않는다. 다음 시간 예산 정책 batch는 (1) context·prefill/decode 구성에 따른 비용 추정, (2) decode를 매 iteration 진행시키면서 prefill 최소 진행량과 aging을 보존하는 선택, (3) graph capacity와 KV reservation을 함께 검증하는 admission 계약, (4) 긴 prompt·burst/open-loop workload에서 throughput뿐 아니라 queue delay와 SLO goodput 평가를 포함해야 한다. Cold/미관측 shape에서 추측 비용을 사용하지 않는 fallback도 필요하다. POD 실행은 지원·수치·workspace 계약을 별도로 통과해야 하며 실패한 FlashInfer 수치 profile을 승격하지 않는다.
