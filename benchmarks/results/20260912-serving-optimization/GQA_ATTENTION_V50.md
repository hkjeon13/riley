# V50 decode GQA attention batch

V49 Round56은 V48 대비 처리량을 개선했지만 vLLM 대비 TPOT가11.02~36.62% 느리다. 별도 natural C16/C32 trace에서 pure decode graph span은 각각69.38%/58.16%를 차지한다. C32 values10.918µs, scores7.349µs가 layer별로 반복된다. 이 증거를 근거로 두 연관 개선을 함께 검증한다.

1. 동일 KV head를 공유하는 query head3개를 QK MMA의 서로 다른 행으로 계산한다. 동일 K 읽기와 중복 MMA를 줄이며 기존 K16 네 번의 누적 순서를 유지한다.
2. 세 warp가 기존 순서로 각 head의 softmax를 계산하고, 한 warp의 MMA가 세 probability 행에 공통 V를 곱한다. reverse128 tile 순서, lane-local denominator, BF16 probability rounding, alpha rescaling을 유지한다.

격리 prototype은 원격 `/tmp/riley-opt-260912/gqa-attention-v50`에 있다. Application source는 아직 수정하지 않았다. V49 frozen binary를 serving 비교 기준으로 유지한다.

최초 primitive 검사는 active1..32, invalid0/33, context1..4096, disjoint permuted KV pages에서 single-row reference와 정확히 일치했다. 확장 검사는3 seeds ×7 context조건(mixed 및1/31/128/398/1024/4096) ×8 row조건(0/1/4/8/16/24/32/33) ×3 변경 조합=504case다. V49 baseline 대비 전체 score/output buffer 및 inactive padding을 byte 비교하여 모두 통과했고 memcheck 오류0이다.

Racecheck도504case 오류0/warning0으로 완료됐다. 첫 timing320records가 완료됐다. Timing은 기존/QK만/values만/둘다 네 variant, rows1/4/8/16/32, context128/398/1024/4096, 네 역순 반복,10warmup+100CUDA graph replay 조건이다. 이 결과는 커널 후보 선정 근거이며 serving 성능 입증은 아니다.

검사 및 timing이 통과한 뒤 실제 모델 full-logit/KV/greedy 및 scheduler/HTTP correctness를 검증하고, 동일 V49 및 vLLM serving 조건으로 비교해야 한다. Values 공유의 barrier 및 occupancy 비용 때문에 개선을 미리 가정하지 않는다. Blender는 종료 상태로 유지한다.


## 첫 비교 결과

| Rows | Context | 기존 µs | QK만 µs | 둘다 µs | 둘다 변화 |
|---|---:|---:|---:|---:|---:|
| 1 | 128 | 8.131 | 5.304 | 5.586 | -31.30% |
| 1 | 398 | 11.824 | 9.004 | 10.056 | -14.95% |
| 1 | 1024 | 17.971 | 15.913 | 17.869 | -0.57% |
| 1 | 4096 | 54.997 | 55.044 | 62.616 | 13.85% |
| 16 | 128 | 9.820 | 6.970 | 6.042 | -38.48% |
| 16 | 398 | 15.206 | 11.950 | 11.248 | -26.03% |
| 16 | 1024 | 25.987 | 20.516 | 19.738 | -24.05% |
| 16 | 4096 | 89.492 | 65.654 | 66.924 | -25.22% |
| 32 | 128 | 11.806 | 9.072 | 7.535 | -36.18% |
| 32 | 398 | 21.840 | 17.531 | 13.721 | -37.17% |
| 32 | 1024 | 44.265 | 33.869 | 25.164 | -43.15% |
| 32 | 4096 | 162.893 | 157.069 | 139.411 | -14.42% |

QK만 변경하면 대부분 개선하지만 values 공유는 낮은 rows에서 손해가 발생한다. 다음 실험은3 query head를 같은CTA의 독립 warp로 계산하여 block barrier 없이 처리하는 values variant다. 높은 rows에서 공유V의 이득을 유지하면서 낮은 rows의 실행 비용을 줄일 수 있는지 검증한다. 아직 application source 통합이나 serving 성능 주장은 없다.

첫 prototype 증빙42파일 SHA256 검증 완료. [Manifest](raw/gqa-v50-manifest.json), [분석](raw/gqa-v50/gqa-attention-v50/analysis.json). Archive SHA256 `4b375a88fe9011e8e075eb24731109c40acc802f27ce854423e49271dff2e743`.


## 독립 warp 조합 비교 및 통합 시작

추가840개 정확성 case와480개 timing records가 완료됐다. Variant5(grouped QK + independent three-warp values)는20개 timing 조건에서 모두 기존 대비 개선되어 후보로 선택했다. 값 공유 variant3보다 낮은 rows/긴 context에서 일관된 결과를 보인다. 일부32-row 중간 context에서 variant3이 더 빠르지만 현재는 조건별 분기를 추가하지 않는다.

| Rows | Context | 기존 µs | 선택 조합 µs | 변화 |
|---|---:|---:|---:|---:|
| 1 | 128 | 8.168 | 5.631 | -31.06% |
| 1 | 398 | 11.903 | 9.482 | -20.34% |
| 1 | 1024 | 18.033 | 15.831 | -12.21% |
| 1 | 4096 | 55.076 | 52.398 | -4.86% |
| 16 | 128 | 9.644 | 6.072 | -37.04% |
| 16 | 398 | 15.072 | 10.788 | -28.43% |
| 16 | 1024 | 26.050 | 17.964 | -31.04% |
| 16 | 4096 | 88.817 | 57.885 | -34.83% |
| 32 | 128 | 11.750 | 7.628 | -35.08% |
| 32 | 398 | 21.871 | 14.309 | -34.57% |
| 32 | 1024 | 44.098 | 27.115 | -38.51% |
| 32 | 4096 | 161.423 | 134.604 | -16.61% |

선택된 소스는 `kernels/src/decode_gqa_attention_v50.cuh`이며 grouped QK와 independent values만 포함한다. 기존 shared32 model의 grouped_attention 기본값은false이고, Mixed recorder의 새 compiled wrapper만true로 호출한다. V5/V6 numerical path는 유지한다. build dependency와 graph catalog source hash에 새 헤더를 포함했다.

원격 controller `qualify_gqa_v50.py`는 선택 조합을 포함한840case memcheck(통과), racecheck(진행 중), serving build, owned GPU test build, 전체 ignored GPU 회귀를 순차 실행한다. 관찰 session6088. 실행 중인 controller를 재시작하지 않는다. Application source 변경은 원격 격리 checkout에만 존재하며 아직 commit/frozen/serving qualification은 없다.

추가11파일 SHA256 검증 완료. [Manifest](raw/gqa-v50-independent-manifest.json), [분석](raw/gqa-v50-independent/gqa-attention-v50-independent/analysis.json). Archive SHA256 `94572580d0ac1da110e0ef810f01089bef340c6da74d562b1306472313148ce7`.


## V50 통합 검증 완료, Round57 실행 중

선택 조합을 포함한840case memcheck/racecheck 모두 오류0, 실제 모델 GPU 회귀16개 통과, V7 full/compact/partial3개 실제 모델 memcheck 오류0. HTTP37응답 기준 일치 및 invalid bound/stream disconnect 회복 검사 통과. CPU/GPU-greedy fallback22응답씩 전체 ordered response 일치.

원격 commit `44ccb8d0535e80e2f4b7fd09b84b770848ad4678`, frozen binary SHA256 `c618e2db269a1aa0e624f7d84036ee97b8ff8b45f82c6d58ebb6894e5b8a97c8`, build log SHA256 `776167e835a0f4ebfe0a140c53e9c37366dfdd591b85aa1a08569247dced6e2c`. 원격 checkout clean 확인 후 frozen했다. 이전 V49 binary는 보존한다.

[통합 patch](raw/integration-v50/gqa-v50.patch), [통합 manifest](raw/integration-v50-manifest.json). 36파일 SHA256 검증 완료; archive `28200c4fd61d7357df7f16c8f39ffb77b34fe958a20fb2292a5c35d59191f594`.

Round57은 V49/V50/vLLM ×C16/C32 ×fixed/natural ×2역순의24lane이다. 각96warmup+384retained, 두 Riley V7/GPU-greedy/budget512와 fixed chunk128/natural512를 동일하게 유지한다. 원격 controller `serving_screen_round57.py`, log `serving-round57-controller.log`, 관찰 session52679. 측정 중 다른GPU job이나 무거운build/export를 실행하지 않는다. 끝난 뒤 analyzer/export와 V50 trace로 다음 병목을 선정한다. Serving 결과가 나오기 전에는 개선 채택이나 목표 달성을 주장하지 않는다.


## Round57 완료

9,216 요청 실패0, Riley6,144 기준 일치. V50의V49대비 처리량 개선은C16+3.45~3.94%,C32+0.24~1.19%이며 vLLM 대비TPOT는 아직8.44~53.05% 느리다. C32 fixed P99도2.62% 악화해 전체 성능 개선을 단정하지 않는다. [최신 결과](V50_SERVING_RESULTS.md). 별도 V50 trace 실행 중이다.


V50 profile도288응답 기준 일치로 완료했다. C32fixed prefill/mixed가77.33%로 다음 병목을 확인했다. [최종 V50 serving/profile 결과](V50_SERVING_RESULTS.md). 현재 모든V50 controller는 terminal이며 실행 중인GPU측정은 없다.
