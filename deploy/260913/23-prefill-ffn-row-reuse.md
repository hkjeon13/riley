# PR 23 — Prefill FFN weight reuse across row tiles

상태: native 정확성·sanitizer 통과, 큰 M 개선/작은 M 회귀. 모델·serving 미검증. 기본 backend 변경 없음.

Projection 적용 후 [진단](../../benchmarks/results/20260914-projection-pipeline-profile/README.md)의 unique prefill/mixed에서 FFN은198.779ms, graph span의30.63%다. 기존 FFN pipeline은 gate/up과 down에서 M16마다 같은 weight를 다시 stage한다. 다음 batch는 새로운 shared staging 방식의 두 연산을 함께 평가한다.

1. M32 CTA에서 두 M16 입력 tile을 stage하여 한 번 가져온 weight를 공유한다. 입력 shared stride72 및 K64 double buffering은 유지한다.
2. gate/up과 down 모두 weight fragment를 한 번 읽고 두 독립 accumulator에 적용한다. K16 MMA 순서, gate/up BF16 round·SwiGLU, down K320/마지막K256 rounding과 merge 순서를 보존한다.
3. 양 kernel의 ragged/invalid live row 및 graph replay를 검증한다. Native gate 통과 후에만 capability·graph identity와 retained Rust C ABI recorder를 하나의 opt-in backend로 연결한다. Python serving 호출은 없다.

논리적으로 weight copy가 행당 절반이지만 실제 HBM traffic 절감은 counter 없이 주장하지 않는다. Shared 예산은 gate25,600B/down13,312B이며 register·occupancy·spill은 컴파일/실측으로 확인한다. CTA 감소로 낮은 row 수에서 회귀할 수 있으므로 M8/16/32/64/128/398/512/1024를 포함한다. Native 결과로 작은 상수 변형을 연속 탐색하지 않고 이득이 없으면 원인을 기록하고 중단한다.

검증: 현재 M16 pipeline과 전체 gate/down 출력 bitwise, inactive sentinel,0/1/15/16/17/31/32/33/63/64/65/128/398/512/1024 및 capacity 초과, 입력/live-row 갱신 graph replay. Bounded memcheck/racecheck,SM89 실행,SM90a/SM100a compile과 장비 부재 runtime skip을 구분한다. 합성 native pass는 실제30-layer weight·full-model logits/greedy·free generation/stop/cancel 및 serving 비교를 대체하지 않는다.

모델 연결 후 frozen projection baseline,같은 binary control,후보,vLLM을 동일 GC 통제/tmpfs 조건에서 비교한다. 먼저 C32 shared/unique 두 순서로 평가하고 이득·tail 확인 후 C8/C64 및 장기/open-loop로 확대한다. Throughput만 좋아지고 TTFT/TPOT/tail이 악화하면 승격하지 않는다. 롤백은 기존 M16 FFN backend 선택이다.

CUDA asynchronous copy와 memory lifetime 근거는 PR05에 보존된 연구를 사용한다. 새로운 library 도입이나 attention reduction 변경이 아니다.

## Native 결과

[전체 결과](../../benchmarks/results/20260914-prefill-ffn-row-reuse-native/README.md):32개 bitwise/inactive 조건 및 memcheck/racecheck 통과. M398/512/1024는 약21–26% 빨라졌지만 M8–128은 약26–37% 느려 전면 교체를 거부한다. 실제30-layer weight/crossover 평가 후 검증된 row 범위 선택과 기존 M16 fallback을 연결한다. Serving 비교 전 기본값 승격하지 않는다.

## 실제 모델 weight native 검증

[30-layer 결과](../../benchmarks/results/20260914-prefill-ffn-model-weight-native/README.md):1440개 layer별 출력과48회 graph replay가 일치했다. 측정한192행 이상에서11.5–26.7% 개선,160 이하에서는6.3–23.9% 회귀했다.192행을 선택 조건 후보로 삼고191/193 경계·행 수 전환과 기존 M16 fallback을 검증한다. Activation은 합성이며 full-model/serving 검증은 미완료다.
