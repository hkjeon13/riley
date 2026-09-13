# V56 serving 결과 — 일부 개선, 추가 병목 확인

V56은 token-major V 기준에서 FFN gate/up의 row warp 배치를 변경하고, 두 projection merge를 residual/RMSNorm에 결합했다. 실제 serving의 자연 길이에서는 소폭 개선됐으나 fixed에서는 손해가 관찰됐다. 일반 성능 우위나 목표 달성으로 채택하지 않는다. V51 일반 비교 기준과 V52 frozen 기준을 보존한다.

동일 RTX4090/SmolLM2-135M BF16, V7 GPU greedy, admission C16/C32, budget512, fixed chunk128/natural chunk512. V51/V52/V56/vLLM0.27.1 두 역순, 32 lanes, 각각 warmup96/retained384를 사용했다. 총12,288 요청 실패0, Riley 기준출력9,216개 모두 일치. 엔진별 token 총량은 fixed7,680/natural28,672로 동일했다. vLLM 일부 출력은 reference와 다르므로 모든 cross-engine byte 일치를 주장하지 않는다. 두 번의 screen은 장시간 안정성이나 통계적 우위 검증이 아니다.

| Case | V51 tok/s | V52 tok/s | V56 tok/s | vLLM tok/s | V56 TPOT ms | vLLM TPOT ms | V56 P99 ms | vLLM P99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c16-fixed | 6146.358 | 6267.936 | 6155.395 | 5120.111 | 2.362 | 2.218 | 85.628 | 113.727 |
| c16-natural | 7805.284 | 7740.380 | 7847.702 | 7794.131 | 1.908 | 1.810 | 261.985 | 277.164 |
| c32-fixed | 7971.408 | 8218.024 | 8140.940 | 7412.664 | 3.561 | 2.804 | 141.876 | 167.288 |
| c32-natural | 10193.117 | 10285.685 | 10433.829 | 11950.605 | 2.874 | 2.299 | 405.391 | 348.449 |

V56 vs V52 throughput 변화는 순서대로 −1.80%, +1.39%, −0.94%, +1.44%다. Natural TPOT는 C16 −1.93%, C32 −1.41%였다. 그러나 vLLM 대비 TPOT는 여전히 모든 조건에서5.46–27.03% 높았다. C32 natural throughput은 vLLM보다12.69% 낮고 P99는16.34% 높다. 최종 목표 미달이다.

## 실제 모델 profile

V56 C32 natural decode graph의 median kernel count는336→276으로60개 줄었고 median graph 시간은 V52 1636.820→V56 1565.844µs다. Merge_norm은 평균1.824µs이며 이전 norm1.621µs와 projection merge약1.055µs를 합친 것보다 낮다. 반면 새 split_rows gate는9.080µs로 기존 gate8.341µs보다 느리다. 단일 kernel 반복에서의 개선과 실제 모델 결과가 달라, 여러 layer의 weights를 순환하는 working-set 실험을 준비했다. 아직 차이의 인과관계를 확정하지 않는다.

Profile의288 요청은 기준출력이 모두 일치했다. Middle80% window에서 prefill과 mixed를 합친 `prefill_or_mixed`와 pure decode를 구분한다. Trace runtime API 시간은 production latency가 아니며, 단계별 batch 구성 차이를 무시하고 serving 효과를 계산하지 않는다.

V57은 KV 배치를 바꾸지 않고 Q/K의 adjacent BF16 pair 읽기, probability pair 읽기, shared score/probability 배열 간격을 비교했다. Mixed primitive242 scenarios 및 context4096 boundary38 scenarios, decode2,016 cases와 예정된 timing이 완료됐다. Application 통합/성능 채택은 없다. Gate working-set 측정도 끝났으며 결과와 한계는 [측정 종료 기록](MEASUREMENT_CLOSEOUT_V57.md)에 정리했다.

## 검증과 artifact

V56 commit `030c207565eda3ec45533a166e6d9a201c8d1831`, binary SHA256 `08cd627049f948b94e4426d933b7aeab883c93a385e3f2d603efe1e6a2fb54a8`, build log SHA256 `8da582cc145389ec79360ca0b59434c975e10845fdb42efb58dbf93afebd0b3f`. Actual-model16 regression, V7 memcheck3, HTTP37, CPU/GPU fallback22 응답 일치. Gate 두 실험 각각1,632 cases, merge_norm816 cases, memcheck/racecheck 통과.

- `raw/ffn-v56-prototype`:139파일, SHA256 `ccc53b0439c1ed15bc74d8c7ca61dd8c49bee0f278f1ac3b1364eae645d734c7`.
- `raw/integration-v56`:32파일, SHA256 `1ba255906a1b69d4fac81c2aaf7e5def5ef9d7a720f21d2b79e9d35c35ea1cef`.
- `raw/serving-round62`:267파일, SHA256 `5fd838c76696bd6b84a4a79874982c575e70341b7dee868da69115c503e1ba26`.
- `raw/profile-v56`:35파일, SHA256 `d4f73042a0859207a247af647f3e0f8917cf40e5334c8d2661498011b7e74360`.

SHA256는 archive 해시이며, 내부 모든 파일의 size/SHA256도 local에서 검증했다. R62 controller54416과 profile/export controller61642는 정상 종료했다. Blender는 계속 종료 상태다.
