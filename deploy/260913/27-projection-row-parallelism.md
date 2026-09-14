# Decode projection 행 병렬화 실험

## 근거와 범위

C32 dense-wire profile의 pure decode에서 QKV·attention output projection·FFN down 합계는 shared 40.252 ms / graph 144.803 ms, unique 34.492 / 135.529 ms다. 이 세 연산만 2배 빨라진다고 가정하면 해당 graph 구간의 산술적 speedup은 1.161 / 1.146이다. Profile overhead·다른 serving 단계·host 비용을 반영하지 않은 가정이며 serving 예상치가 아니다. 원본 archive 해시와 계산은 `benchmarks/results/20260914-decode-projection-split-native/selection.json`에 기록한다.

현재 adaptive projection은 16행 초과 시 한 warp에서 두 M16 tile을 계산한다. 각 projection은 360 CTA지만 CTA당 한 warp다. 단순 CTA 수만으로 GPU 활용률을 판단할 수 없다. 이미 gate/up에 사용되는 행 분할을 이 세 projection의 실험에 적용한다. 이는 Stream-K의 K 분할을 도입하는 것이 아니라, 기존 수치 순서를 유지할 수 있는 M 방향 병렬화다.

## Optimization batch

1. QKV의 두 M16 tile을 각각 warp에 배정한다.
2. Attention output projection에도 같은 행 분할을 적용한다.
3. FFN down에도 적용하며 K320 BF16 partial 경계와 chunk-major 32-row scratch stride를 유지한다.

같은 CTA에 두 warp를 사용한다. 두 번째 warp의 가중치 load 증가와 작은 active batch에서의 비활성 warp 비용을 이득과 함께 비교한다. 기존 producer/consumer scratch 계약을 바꾸지 않는다. production dispatch는 변경하지 않으며 `kernels/optional/decode_projection_split_rows.cuh`와 native probe만 추가한다.

## 검증

- 기준은 이미 승격된 adaptive projection이다.
- 0..33 active rows, 세 입력 seed에서 모든 partial의 bitwise 일치, 비활성 행 및 guard 확인. 0/33행은 no-write 계약이다.
- Bounded compute-sanitizer memcheck.
- 30개 layer 크기의 서로 다른 synthetic weight 영역, CUDA Graph 90개 연산, warmup 20회와 측정 100회, 네 번의 순서 교대. 입력 seed가 다른 세 correctness 반복은 첫 layer weights를 사용한다. 실제 모델 품질 증거는 아니다.
- 1/8/16/17/24/32행을 비교해 경계 회귀를 확인한다.
- Native 이득이 확인된 뒤에만 실제 모델·stop/cancel·serving 검증으로 진행한다. Serving 비교에는 기존 메모리 및 GPU 격리 조건을 유지한다.

4090에서 실행한다. Hopper/Blackwell은 별도 실행 검증이 필요하며 이 결과로 성능을 추정하지 않는다. Python은 실험 orchestration에만 사용하고 Riley runtime에 연결하지 않는다. 실패하면 optional artifact를 보존하고 production baseline을 유지한다.

## 실행 결과와 다음 단계

동일 CTA의 두 warp 변형은 큰 행 수에서 6–8% 개선됐지만 작은 행 수에서 9–23% 회귀했다. 별도 CTA 변형은 16/17/24/32행에서 각각 13.31/25.71/18.85/18.38% 단축됐고 8행은 거의 동일, 1행은 5.65% 회귀했다. 두 변형 모두 102회 bitwise 검사 및 bounded memcheck를 통과했다. Raw logs·요약·binary/source 해시는 `benchmarks/results/20260914-decode-projection-split-native/`에 보존한다.

다음 단계는 별도 CTA 변형의 실제 모델 검증과 저부하 fallback 필요성 판단이다. 18.38%는 세 projection의 native 시간 감소이며 serving throughput 개선율이 아니다. production dispatch는 아직 변경하지 않았다.

## 실제 모델 및 serving 검증

실험 opt-in `RILEY_EXPERIMENT_PROJECTION_CTAS=1`을 graph capture에 연결했다. 기본값은 기존 adaptive projection이다. 실제 모델은 활성 용량 16·32에서 직렬 기준 생성 토큰과 일치했고, terminal/cancel 및 모든 실행 후 allocation 회수 검증을 통과했다.

C32 serving 12개 lane 및 warmup 포함 6,912개 응답을 재검산했다. Shared throughput은 기존 대비 +5.07%, TPOT는 -4.88%였다. Unique는 aggregate throughput +1.18%지만 순서별 +6.04%/-4.09%로 방향이 바뀌고 TPOT가 +6.40% 악화됐다. vLLM unique 실행 편차 및 host I/O pressure도 컸다. vLLM보다 모든 latency가 동등 이하인 목표는 미달이며 기본 경로로 승격하지 않는다.

다음 검증은 host 부하가 안정적인 조건에서 C8/C16/C64와 unique 재현성 확인이다. 작은 batch의 native 회귀가 실제 serving에도 나타나는지 확인하고, fallback은 비용까지 측정한 뒤 결정한다. 결과 표·source/binary 해시·요청별 검증은 `benchmarks/results/20260914-projection-cta-serving-c32/`에 보존한다. 메모리 사전 조건 실패는 이전 FFN tmpfs spool을 영구 보존한 후 재시도하여 해결했고, 종료 시 Blender 세 개를 복구했다.
