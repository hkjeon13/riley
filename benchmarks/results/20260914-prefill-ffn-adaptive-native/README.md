# Adaptive FFN dispatch native gate

같은 graph의 device live row로192 미만 M16,192 이상 M32를 선택하는 후보를 검증했다. Shared union으로 두 분기의 공간을 재사용하지만 커널 전체의 최대 자원량은 M32 기준이다. Gate/up·down을 함께 선택하며 원래 M16 kernel은 수정하지 않았다.

| Rows | M16 pair µs | Adaptive pair µs | 시간 변화 |
|---:|---:|---:|---:|
| 8 | 19.603 | 19.532 | -0.36% |
| 16 | 19.440 | 19.418 | -0.11% |
| 32 | 19.211 | 19.207 | -0.02% |
| 64 | 19.258 | 19.246 | -0.06% |
| 128 | 20.612 | 21.453 | +4.08% |
| 160 | 22.498 | 25.819 | +14.76% |
| 191 | 28.887 | 27.473 | -4.89% |
| 192 | 28.885 | 24.083 | -16.62% |
| 193 | 29.645 | 24.247 | -18.21% |
| 224 | 30.626 | 24.355 | -20.48% |
| 256 | 31.763 | 25.167 | -20.77% |
| 288 | 33.123 | 29.290 | -11.57% |
| 320 | 35.064 | 30.284 | -13.63% |
| 352 | 43.361 | 35.402 | -18.36% |
| 384 | 45.872 | 36.197 | -21.09% |
| 398 | 46.411 | 37.616 | -18.95% |
| 512 | 53.248 | 40.694 | -23.58% |
| 1024 | 107.095 | 79.031 | -26.21% |

192 이상에서11.6–26.2% 감소하고8–64에서는 거의 동률이지만128은4.08%,160은14.76% 느려졌다. 따라서 M16 branch 선택만으로 기존 kernel의 성능이 보존된다고 주장하지 않는다. Shared/register footprint가 두 branch에 공통인 구조가 가능한 원인이며 하드웨어 counter로 인과를 증명하지 않았다. Threshold 미세 조정을 반복하거나 이 회귀를 숨긴 채 승격하지 않는다.

26개 live-row 조건×2회×30 layer의1560개 출력/inactive 검사와52회 graph replay가 일치했다.191/192/193 선택 경계 포함, bounded memcheck/racecheck 통과,SM89 실행 및 SM90a/SM100a compile 통과다. 후자 runtime은 장비 부재 미검증이다. Actual weight fixture는 [이전 검증](../20260914-prefill-ffn-model-weight-native/README.md)의 같은90개 packed tensor이고 activation은 합성이다. Full-model/serving gate는 미완료다.

다음 통합 결정은 shared/register 자원까지 분리한 graph/backend 선택의 비용과 현재 unified dispatch의 실제 serving 영향을 비교해야 한다. Native 작은 shape 회귀만으로 전체 workload의 손익을 예측하지 않고, 지원 shape·graph identity·Rust 소유권과 full-model 수치 gate를 갖춘 opt-in으로 검증한다. Default는 유지한다. Blender3개는 측정 종료 후 복구했다.
