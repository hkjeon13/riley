# Prefill FFN row reuse native gate

M16 pipeline 대조군과 M32 weight reuse 후보를 같은 입력·weight에서 비교했다. 각 행의 K16 순서와 BF16 rounding을 유지한 gate/up·down 한 batch다. 모델·serving에는 연결하지 않았다.

| Live rows | M16 pair µs | M32 pair µs | 시간 변화 |
|---:|---:|---:|---:|
| 8 | 17.249 | 23.306 | +35.11% |
| 16 | 17.449 | 23.511 | +34.74% |
| 32 | 17.562 | 23.880 | +35.98% |
| 64 | 17.802 | 24.003 | +34.83% |
| 128 | 19.241 | 24.259 | +26.08% |
| 398 | 48.456 | 38.072 | -21.43% |
| 512 | 53.944 | 41.748 | -22.61% |
| 1024 | 107.694 | 79.862 | -25.84% |

M398/512/1024는 약21–26% 빨라졌지만 M8–128은 약26–37% 느려졌다. 전 shape 교체는 거부한다. 이 결과는 합성 weight 한 집합·warm graph100회·역순 두 실행의 중앙값으로, 실제30-layer weight 및 serving 효과를 뜻하지 않는다. 다음은 실제 모델 weight와128–398 사이 shape에서 crossover를 확인하고, 검증된 범위에만 M32를 선택하는 graph-safe dispatch와 기존 M16 fallback을 한 batch로 연결하는 것이다. 임의 threshold로 serving 승격하지 않는다.

32개 live-row/입력 갱신 graph replay에서 전체 gate/down bitwise mismatch0, inactive 변경0이다. Bounded memcheck/racecheck와 SM89 실행,SM90a/SM100a compile을 통과했다. Hopper/Blackwell runtime은 장비 부재로 미검증이다. 실제 shared gate25,600B/down13,312B, register42/38,local0이다. 행당 논리 weight copy 감소를 실제 HBM 절감으로 주장하지 않는다. 낮은 M의 회귀 원인은 CTA 병렬성 감소와 증가한 tile 작업이 가능한 설명이지만 occupancy counter로 증명하지 않았다.

GPU 측정 후 원래 Blender3개를 복구했다. 원본 native/sanitizer/build 로그와 source 해시는 evidence 및 receipt.json에 보존했다. Rust→Python serving 경로를 추가하지 않았다.
