# FFN row reuse with actual model weights

SmolLM2-135M checkpoint의30 layer×gate/up/down BF16 weight90개를 추출하고 MMA layout packing의 역변환을 검증했다. Checkpoint 및 tensor/packed hash는 manifest에 보존했다. 입력 activation은 합성이므로 full-model 출력 검증이 아니다.

Timing graph는30개 layer의 서로 다른 weight를 순회한다. 각 pair 시간은100 replay×30 layer로 나눈 값이며, 두 역순 실행의 중앙값이다. 이전 단일 weight warm-cache 결과와 합산하지 않는다.

| Rows | M16 pair µs | M32 pair µs | 시간 변화 |
|---:|---:|---:|---:|
| 8 | 19.596 | 23.837 | +21.64% |
| 16 | 19.441 | 23.440 | +20.57% |
| 32 | 19.215 | 23.812 | +23.92% |
| 64 | 19.244 | 23.667 | +22.98% |
| 128 | 20.584 | 23.757 | +15.41% |
| 160 | 22.472 | 23.898 | +6.34% |
| 192 | 28.856 | 24.097 | -16.49% |
| 224 | 30.638 | 24.366 | -20.47% |
| 256 | 31.765 | 24.953 | -21.45% |
| 288 | 33.003 | 29.200 | -11.52% |
| 320 | 35.034 | 30.316 | -13.47% |
| 352 | 43.345 | 35.390 | -18.35% |
| 384 | 45.851 | 36.181 | -21.09% |
| 398 | 46.279 | 37.546 | -18.87% |
| 512 | 53.324 | 40.571 | -23.92% |
| 1024 | 107.520 | 78.801 | -26.71% |

측정한192 이상에서11.5–26.7% 감소했지만160 이하에서는6.3–23.9% 증가했다. 다음 dispatch 후보는192 미만 M16 유지,192 이상 M32 선택이다. 아직 측정하지 않은 경계191/193 및 row 전환을 검증하기 전 채택하지 않는다. Threshold는 현재4090/고정 shape의 후보이며 다른 GPU의 최적값으로 주장하지 않는다.

24개 live-row 조건×두 입력 갱신×30 layer의1440 gate/down 전체 출력·inactive 검사가 bitwise 일치했다.48회 graph replay의 최종 출력도 일치했다. 실제 모델 weight를 사용한 native 검증이며30-layer residual/attention 연결·자유 생성·serving gate는 남아 있다. 앞선 bounded sanitizer는 같은 kernel에 대한 합성 fixture 검증이고, 이번 실행에서 sanitizer를 재실행했다고 주장하지 않는다.

GPU 실행 exit0과 Blender3개 복구를 확인했다. 런타임은 Rust/CUDA를 유지하고 Python은 offline fixture 추출에만 사용한다. Graph-safe 선택, recorder identity/수명, full-model gate를 통과한 뒤 current/control/candidate/vLLM serving 비교를 진행한다.
