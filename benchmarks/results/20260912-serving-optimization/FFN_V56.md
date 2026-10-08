# V56 FFN 및 projection merge/normalization batch

V54/V55 packed V 계열은 serving상 실질적 이득을 확인하지 못해 채택하지 않았다. 원격 source는 token-major V로 복귀했다. Restore commit `badd99adb9ed6263d78a3601917553fd13b1ff78`의 tree `938eb454f1406f7d84de72f2425c1077b24c2522`가 V52 tree와 정확히 같음을 검증했다. V51 일반 비교 기준과 V52 frozen 기준은 보존한다.

V55 natural C32 trace에서 decode gate/up/SwiGLU는 평균8.312µs, attention projection/down projection의 별도 merge는 각각 약1µs다. 이 batch는 gate/up 입력 재사용, shared intermediate/barrier 제거, 계산 row 범위와 load pipeline 조정, projection merge와 residual/RMSNorm 결합을 함께 다룬다. 기존 BF16 rounding과 합산 순서를 유지한다.

## Gate prototype

`decode-gate-v56`: single-warp M32 fusion, Steps4/2 및 active rows에 따른 M16/M32 경로를 비교했다. 1,632 finite/nonfinite/invalid-row/output-padding cases와 memcheck/racecheck가 통과했다. Adaptive Steps2는 small rows에서 약5–9% 빨랐지만 rows24/32에서16–18% 느렸다. 고정 M32 fusion은 크게 손해여서 그대로 채택하지 않았다. Register/local 자료에서 spill은 없었다.

`decode-gate-layout-v56`: small-row hybrid fallback, 두 warp가 각16 rows를 계산하는 Steps4/2를 비교했다. 추가1,632 cases와 memcheck/racecheck가 통과했다. 선택 후보는 `split_rows<4>`이며 baseline 대비 rows1/4/8/16/24/32의 시간 변화가 −2.19%, −0.21%, −2.40%, −4.41%, −7.35%, −8.85%다. 이는 여전히 microbenchmark이고 rows4의 미세 차이를 우위로 해석하지 않는다. 초기 실험과 baseline 절대시간이 달라 동일 실행 안의 비교만 사용한다.

두 gate controller(session61334,83297)는 완료됐다. 첫 초기 gate build는 테스트 macro와 loop 변수명 충돌로 실패했고, 수정 후 모든 검사를 수행했다. 실패 log도 보존한다.

## Merge + normalization prototype

`merge-norm-v56`는 attention projection과 FFN down projection이 사용하는 다섯 partial의 합산을 residual/RMSNorm kernel 앞부분에 결합한다. Projection 결과의 BF16 round-trip을 명시적으로 유지하고, mode1의 FP32 residual과 mode2의 BF16 residual 저장 및 기존 warp reduction 순서를 복제했다. Output/residual의 전체 바이트와 inactive padding을 비교한다.

최초 build는 host 데이터 생성에 device-only CUDA NaN/Inf 상수를 사용해 실패했다. Host용 `std::numeric_limits<float>`로 바꾸고 재시작했다. 현재 local exec session43873에서 build/correctness/memcheck/racecheck/timing이 순차 실행 중이다. 이전 session24438은 build 오류로 종료했다. 아직 merge+norm 결과나 V56 application 통합/성능 채택은 없다.

Serving Round61 session81317, 후속 profile/export session65770은 모두 정상 종료했다. FFN 측정과 serving을 겹치지 않는다. Blender는 계속 종료 상태다.
