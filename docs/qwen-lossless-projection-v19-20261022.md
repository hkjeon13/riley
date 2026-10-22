# BF16 무손실 projection 포맷 실험 v19 · 미채택

2026-10-22. 가중치 전송량을 더 줄이는 구조를 실제로 구현하고 GPU6에서 검증했다. 정확도는 통과했지만 **Gate/Up과 Down 모두 느려 서버에 통합하지 않는다**. 이 후보에 HTTP 처리량 개선 주장은 없다.

각 weight tile은 sign+mantissa8비트, 상위15개 exponent의 4비트 사전 인덱스, 8원소별 escape prefix, 원본 exponent escape로 저장한다. shared memory에서 원래 BF16 비트를 복원한 뒤 기존 WMMA/FP32 누산·BF16 경계를 실행한다. 오차를 허용한 양자화나 dtype 변환은 없다. Gate/Up은64×64, Down은64×32 및32×32타일을 비교했다. tile이64개 escape를 넘으면 원래 row-major 경로를 쓰도록 만들었으나, 실제3레이어에서는 fallback tile이0개여서 그 fallback의 GPU 실행은 이번 검증 범위에 포함되지 않는다.

실제 Qwen 레이어0/17/35의 포맷을 CPU에서 전수 bit 복원 검사했다. GPU는 R2/8/16/32와6개 finite 입력 상태에서 기존 current native/Lt 경로와 비교했다. 총360건(압축216건 +기존 K-major 대조144건) 모두 BF16 mismatch0이다. 원래 native/HF 전체 모델 정확도 기준을 이 연산 검사로 대체하지 않는다.

| 레이어0 · R16 · 상태0 | 현재 backend µs | 후보 µs | paired 지연 변화 |
|---|---:|---:|---:|
| Gate/Up 무압축 K-major | 83.211 | 79.266 | -4.74% |
| Gate/Up 무손실 포맷 | 83.157 | 109.011 | +31.09% |
| Down 무손실 W4 | 43.414 | 91.405 | +110.52% |
| Down 무손실 W2 | 43.395 | 93.235 | +114.86% |

시간은 5쌍의 before/after 교대 실행, CUDA Graph32연산 ×8replay이며 각 측정 전3warm replay다. 표의 각 µs는 중앙값, 변화는 pair별 비율의 중앙값이다. cold packing/upload는 이 시간에서 제외했다. 입력은 합성 finite activation이고 실제 가중치만 checkpoint에서 가져왔다.

Gate/Up 가중치 tile 저장량은99,614,720→82,104,320B(**17.58% 감소**), Down W4는49,807,360→41,635,840B(**16.41% 감소**)다. 원래 row-major/prefill 사본을 제거한 메모리 절감 결과는 아니다. 모든 R/레이어에서 압축 Gate/Up은 약31~35% 느렸고 Down W4는 약111~116% 느렸다. 세 레이어의 host packing은 Gate/Up 약0.45~0.58초, Down W4 약0.24~0.25초였다. cold 시간을 정상 HTTP 시간으로 합산하거나 속도 이득으로 환산하지 않는다.

이 구현에는 bit 복원 계산과 추가 shared-memory 동기화가 들어간다. `cuobjdump` 정적 결과에서 같은 자체 K-major 대조 대비 Gate R16 레지스터는62→93, Down W4 R16은56→70으로 증가했다. local spill은0이다. 복원·레지스터·동기화의 실제 기여도는 별도 hardware counter 없이 분리 확정하지 않는다. 전송량 감소만으로 빠른 backend라고 판단할 수 없다는 실제 반례다.

whole-device peak1379MiB, observer23개 표본에서 holder6과 이 유한 작업 외 GPU PID는 없었다. 기존 GPU6/7 holder PID/context는 유지했다. 정상 C16/C32 비교에는 연산상 이득이 확인된 무압축 K-major 후보만 진행한다.

[전체 코드·원본·컴파일 자원](../benchmarks/evidence/20261022-lossless-projection-v19-r2.zip) · [모든 pair 및 포맷 통계](../benchmarks/evidence/20261022-lossless-projection-v19/summary.json). Archive32,130B/21entries, SHA256 `a181dbbe835bbb14df33e7fe1dedc061d9fdc7504972e996a35367b9bbb8e0d3`. 최종3레이어 GPU 입력은 [원본 추출 identity](../benchmarks/evidence/20261022-weight-layout-v19-screen/weight-extraction.json)와 동일한 EFS BF16 파일이다.
