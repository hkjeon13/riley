# PR22 — Prefill projection operand pipeline

상태: 격리 native batch 구현·SM89 GPU gate 완료, retained model/server opt-in 연결 및 full-model gate 완료, C32 긴 구간 검증 완료, C8/C64 및 승격 판정 대기. 기본 경로는 기존 rolling Riley이며 GQA staging은 비활성이다.

## 근거와 범위

[현재 control/GQA profile](../../benchmarks/results/20260914-gqa-staging-profile/README.md)에서 unique prefill/mixed graph 시간의 25.65%가 Q/K/V 및 attention-output projection이다. GQA attention 시간 감소는 선택 구간에서 5.08%이며 전체 graph 감소는 1.57%에 불과하다. Native 수치를 serving으로 확대하지 않고 다음 의미 있는 영역으로 이동한다.

`kernels/src/prefill_shape_projection.cuh`는 각 K16 MMA에서 A/B를 global memory로부터 직접 읽는다. Q/K/V는 K192, output은 K128마다 BF16 중간 rounding을 수행한다. 일반 GEMM의 단일 FP32 accumulation으로 교체하면 동일 계산 계약이 아니므로 cuBLAS 호출로 무조건 대체하지 않는다. 기존 FFN pipeline은 이미 별도 구현되어 있어 중복 작업하지 않는다.

[CUTLASS Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)의 계층형 tile 재사용과 double-buffered mainloop를 적용 근거로 한다. CUTLASS는 shared tile과 register fragment의 이중 버퍼링을 설명한다. 이 계획은 해당 원리를 기존 수치 recurrence에 적용하며 CUTLASS 라이브러리 전체를 사용했다고 표시하지 않는다.

## 하나의 optimization batch

1. Q/K/V/output projection의 weight를 모델 준비 시 한 번 tile layout으로 pack하고 retained owner 수명·byte budget·graph identity에 포함한다. Request 경로에서 pack/allocation을 하지 않는다. 행렬별 N/K 및 기존 output storage는 유지한다.
2. 공통 prefill projection mainloop에 A shared reuse와 A/B의 두 K64 buffer, 비동기 copy와 MMA overlap을 함께 구현한다. Shared stride는 bank mapping과 실제 resource 보고로 확인한다. K16 MMA 순서와 K192/K128 BF16 rounding·합산 순서는 보존한다. Invalid/inactive row는 안전한 source로 zero-fill하고 output을 쓰지 않는다.
3. 네 projection을 하나의 명시적 backend로 선택하고 dynamic live-row graph replay, 기존 fallback 및 source-bound identity를 연결한다. Pure decode와 다른 attention 선택을 변경하지 않는다. Native-only, full-model, serving 판정을 분리한다.

## 검증·승격·롤백

Native에서는 N576/K576/interval192, N192/K576/interval192, N576/K576/interval128을 함께 검사한다. M1/8/16/17/31/32/64/128/398/512, capacity padding, live-row 변화와 inactive sentinel, bitwise full output, bounded memcheck/racecheck, register/shared/spill 및 multi-layer weight 순환 측정을 포함한다. 준비 pack 비용·추가 상주 byte도 보고한다. 하나의 warm microbenchmark만으로 모델 연결을 승격하지 않는다.

Native 통과 후 기존 logits/greedy gate를 그대로 적용하고 현재 frozen rolling Riley 및 같은 binary control과 C8/C32/C64 shared/unique serving을 vLLM과 비교한다. TTFT/TPOT/E2E P95/P99 및 실제 SSE token-interval tail, stop/cancel/recovery와 오류를 함께 기록한다. 수치 실패의 tolerance를 바꾸지 않는다. 이득 미확인 시 기본값은 유지하고 증거와 원인을 정리한다.

Rust → C ABI → CUDA를 유지한다. SM89 runtime과 SM90a/SM100a compile을 구분하고 후자의 runtime만 장비 부재로 skip한다. Multi-GPU 장치별 owner 및 packed layout identity를 유지하되 테스트하지 않은 지원을 완료로 표시하지 않는다. GPU 시험 동안 세 Blender만 일시 중지 후 복구하고 3d/3dsol/3dfable 웹 서버는 계속 유지한다. 롤백은 backend 비활성 및 기존 weight owner 선택이며 checkpoint 파일을 덮어쓰지 않는다.

## Native gate 결과

[측정 보고](../../benchmarks/results/20260914-prefill-projection-native/README.md): weight packing, shared A 재사용과 두 K64 buffer를 구현했다. 78조건의 전체30-layer 출력과 graph replay가 bitwise 일치하며 memcheck/racecheck 오류0, SM90a/SM100a compile 통과다. 30개 weight를 순환하는 native graph에서 각 projection 시간이18.96–44.51% 감소했다. 실제 chained model 또는 serving 개선 수치가 아니다. 모델 연결과 품질·serving gate 전 기본값을 변경하지 않는다.

다음 연결은 기존273개 원본 weight 및90개 FFN tile 뒤에120개 projection tile을 별도 retained owner로 추가한다(총483 parent, 추가50.625MiB). 원본 weight는 decode/fallback용으로 유지하며 recorder의 extent·parent·layout 검증과 catalog source identity를 함께 확장해야 한다. 기존363-parent 경로의 계약을 전역 완화하지 않고 새 backend에서만 새 layout을 받는다. Pack 준비시간 및 peak allocation은 모델 gate에서 측정한다.

## Retained model 통합

[Rust/C ABI/model gate](../../benchmarks/results/20260914-prefill-projection-model/README.md)를 완료했다. 483-parent 전용 recorder와120개 tile 수명·content/source identity, 환경변수 `RILEY_PREFILL_PROJECTION_PIPELINE=1`을 연결했다. 2,359,296 BF16 logits bytes가 일치했고 full-model memcheck 오류0이다. Packing 준비 단계는 약0.94초, 추가 device weight는50.625MiB다. 최초 Rust wrapper parent-count 거부를 수정하고 최종 gate를 다시 통과했다. C32 serving은 별도 완료했으며 native 개선을 serving 개선으로 표시하지 않는다.

## C32 첫 serving screen

[전체 비교표와 실행별 수치](../../benchmarks/results/20260914-prefill-projection-serving/README.md): 후보의 동일 binary control 대비 throughput은 shared +1.74%, unique +10.08%다. 그러나 frozen prior shared가 두 순서 사이27.8% 하락했고 late host I/O pressure도 높아 확정 개선으로 승격하지 않는다. 4096 retained 요청의 protocol,3072 Riley 기준 일치, stop/cancel/recovery 각96건은 통과했다. vLLM throughput에는 여전히 미달한다. 후보를 더 수정하기 전에 host pressure를 lane별 기록하는 C32 재측정으로 효과를 확인하고 이후 C8/C64로 확장한다.

## Host 관측을 추가한 C32 재측정

[재측정 결과](../../benchmarks/results/20260914-projection-observed-serving/README.md): 후보의 frozen prior 대비 throughput은 shared +3.50%, unique +7.92%다. 기존 baseline은 이번 두 순서에서 안정적이지만 same-binary shared control 일부가 여전히 흔들렸다. IO pressure와 느린 lane이 일치하지 않아 인과를 단정하지 않는다. 실측 구간이0.67–2.35초에 불과함을 확인했으므로 후보 구현을 고정한 채 lane당8192 retained 요청으로 비교 시간을 늘린다. 원본 두 screen은 유지하고 긴 구간 결과와 섞지 않는다. C32 후 C8/C64로 확장하며 기본값 승격은 보류한다.

## 긴 구간 비교 실행

동일 candidate SHA `009697527d143357c3b3d1a2daa5d8b783298024ac1f8f2aa4ea388bbd3ad929` 및 frozen prior를 고정하고 C32 long comparison을 시작했다. 기존 observed controller에 `--warmup 256 --retained 8192`를 적용한다. Shared32 variant 집합은 유지하며 unique fixture는8192개 실측 요청을 모두 다른 초기 페이지로 구성하도록 확장한다. 실제 prompt 길이 범위는 새 fixture에서 산출하고 이전 짧은 screen과 합산하지 않는다. 전체16 lane의131,072 retained 요청(4,194,304 output tokens)을 검증한다.

새 archive helper는 completed/exit receipt를 확인한 뒤 common 및16개 lane의 무손실 archive를 각각64MiB 미만으로 보존한다. Long exporter는 manifest hash,256/8192 request count, 동일 controller/client·budget,reference/stop/cancel/recovery,PSI delta를 검증한다. 첫 prior shared 구간은 약28초였고 전체 비교는 진행 중이다. 완료 전 성능 승격 또는 안정성 통과로 표시하지 않는다.

긴 구간 exporter는 저장된 checks flag에만 의존하지 않고 fixture별 prompt/output token, text hash, finish, usage와 기록된 SSE frame을 다시 대조한다. 도착 시각 개수·순서 및 phase 시간 범위도 확인한다. 보강한 공통 validator는 기존 실제 응답2560개에서 검증했다. 전송 종료 `[DONE]`은 frames에 보존되지 않으므로 해당 transport 검사는 여전히 hash-bound client의 완료 조건에 의존한다. 현재 긴 C32 run은 같은 실행으로 계속 진행 중이며 재시작하지 않았다.

## 긴 C32 비교 완료

[전체 결과 및 원본 증거](../../benchmarks/results/20260914-projection-long-serving/README.md): 16개 lane,131,072 retained 요청/4,194,304 output tokens를 완료했다.98,304개 Riley 응답이 기준과 일치하며 stop/cancel/recovery 각96건이 통과했다. 보강한 verifier의 fixture·SSE frame·시간 경계 대조도 통과했다. Blender3개는 복구했고 웹 서버는 유지했다.

후보 throughput은 frozen prior 대비 shared +3.56%, unique +8.54%; 동일 binary control 대비 +4.01%, +8.37%다. 기존 Riley 대비 shared E2E P99는362.44→364.43ms(+0.55%)로 악화했고 다른 표의 latency 지표는 개선됐지만 vLLM throughput보다 각각9.57%,19.86% 낮으며 latency 전체 목표도 미달이다. 모델을 고정한 채 같은 긴 구간 평가를 C8/C64로 확장한다. 추가 모델·장기/open-loop·다른 GPU runtime gate는 미완료다.

긴 구간에서 모든 engine의 shared E2E P99가 짧은 screen보다 크게 늘었다. Client가 phase 전체의 SSE frame 객체를 보존하는 현재 경로와 host stall을 구분하기 위해, C8/C64 확대 전에 같은 frozen binary를 유지한 bounded client-pause diagnostic을 수행한다. GC timing은 이번 run에서 측정하지 않았으므로 원인으로 단정하거나 latency에서 임의 차감하지 않는다.

## Client GC 진단 완료

[관측 결과](../../benchmarks/results/20260914-client-gc-diagnostic/README.md): 고정 control/vLLM의 shared C32 역순4 lane32768 retained 응답을 검증했다. 최대 GC 구간435–442ms, P99 이상 요청328개 중310개가10ms 이상 GC와 겹쳤다. 다만 Riley의10ms 이상 ITL 중 GC와 겹친 것은1.16%이므로 서버 쪽 ITL 격차는 별도로 남는다. GC 구간은 wall time이며 인과·독점 CPU 정지로 단정하지 않는다. 기본 GC와 timed-phase GC 비활성의 양 engine 공통 AB/BA intervention으로 클라이언트 영향을 확인한 뒤 versioned 조건으로 후보 비교와 C8/C64를 진행한다. 기존 수치는 유지하고 임의 보정하지 않는다.
