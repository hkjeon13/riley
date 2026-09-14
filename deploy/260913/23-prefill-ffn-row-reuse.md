# PR 23 — Prefill FFN weight reuse across row tiles

상태: unified adaptive native/model/C32 serving 검증 완료, 전체 성능 목표 미달. Packed-row 계측 후 split graph native/model/C32 serving 검증 완료, 추가 이득1% 미만 및 전체 목표 미달. 기본 backend 변경 없음.

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

## Device adaptive dispatch native 결과

[선택 경계 검증](../../benchmarks/results/20260914-prefill-ffn-adaptive-native/README.md):1560 layer별 출력/52 graph replay와 bounded sanitizer 통과.192 이상 개선은 유지됐지만 M16을 선택한128/160에서4.08%/14.76% 회귀했다. 최대 shared/register footprint가 공통인 unified dispatch가 원래 M16 자원 특성을 보존하지 않는다. 기본값 승격을 보류하고 graph/backend 자원 분리 비용과 opt-in 모델/serving 손익을 평가한다.

## Opt-in 모델 통합

[모델 검증](../../benchmarks/results/20260914-ffn-adaptive-model/README.md):별도 recorder·graph identity·Rust session·엄격한 env 선택을 연결했다. BF16 logits3,244,032 bytes 일치/full-model memcheck0/CUDA release build 통과. 자유 생성·serving 결과는 아직 미완료이며 C32 prior/control/adaptive/vLLM16 lane 비교를 시작했다. Native 작은 행 회귀와 default 비승격은 유지한다.

## Adaptive FFN C32 serving 완료

[전체 표와 원본 검증](../../benchmarks/results/20260914-ffn-adaptive-serving-c32/README.md):131072 retained/4194304 tokens,4096 warmup,98304 Riley reference 일치,stop/cancel/recovery 각96건,16 exit0 및 Blender 복구를 검증했다. 후보 throughput은 prior 대비 shared −2.02%/unique +8.14%, 같은 binary control 대비 −0.80%/+6.46%다. vLLM 대비 −8.79%/−14.91%로 전체 목표 미달이다. Shared 회귀 및 vLLM tail 변동을 보존하고 default 비승격을 유지한다. 다음 판단은 실제 활성 prefill 행 분포와 graph/backend 자원 분리 비용을 근거로 하며 기존 작은 threshold 변형을 반복하지 않는다.

## Packed-row census 완료 및 다음 batch

[정확한 배치 행 계측](../../benchmarks/results/20260914-ffn-row-census/README.md):1280개 응답 reference 일치, packed histogram과 기존 batch count의 모든16-row bucket 일치/overflow0. Adaptive unique397회 중390회(98.24%)가192행 이상, shared111회 중38회(34.23%)다. FFN 선택값은 요청별 prefill 길이가 아니라 decode 행을 포함한 **전체 packed token rows**다. V1 요청별 계측은 fragmentation 자료로만 보존한다. 다음 batch는 원래 M16 kernel 자원 보존·M32 두 FFN kernel 분리·공유 buffer 기반 graph 선택을 함께 검토한다. Host-known rows를 사용하며 추가 GPU sync/산술순서 변경/threshold 탐색 없이 graph 소유권과 메모리 비용부터 확인한다. 상세 gate와 실패 시 후속 방향은 결과 문서에 기록했다.

## Split graph 구현 및 모델 검증

`RILEY_PREFILL_FFN_SPLIT=1`은 projection pipeline을 요구하고 unified adaptive flag와 동시 선택을 거부한다. Rust session과 catalog identity, C ABI recorder를 분리했다. 작은 packed batch는 기존 projection/M16 graph를 사용하고,192행 이상은 별도 M32 gate/up·down graph를 선택한다. 총 행 수는 V7 검증을 통과한 패킷 offset36에서 읽으므로 추가 device readback이나 synchronization은 없다.

새 graph는 full/compact 두 개이며 기존 ledger·stream·weight483개·device scratch를 공유한다. Buffered 슬롯의 graph 목록을4→6으로 확장하고, 원래 staging을 쓰는 슬롯은 원 graph를 빌리며 다른 슬롯은 기존 방식으로 host memcpy node를 재결합한다. Future decode는 계속 기존 compact decode graph index3만 사용한다. 종료 시 추가 graph도 stream drain 이후 파괴하고 오류 시 parent quarantine 규칙을 유지한다. 모델 buffer 중복 할당은 없지만 추가 CUDA graph/exec 내부 메모리와 capture 비용은 실측해야 한다.

로컬 `cargo check -p riley-server --features server,bench --bin riley` 및 `cargo test -p riley-server --features server,bench --lib` 통과(73 passed/1 ignored). 이 검사는 CUDA 경로를 검증하지 않는다. 원격 release CUDA build는 `ffn-split-build-v1`에서 exit0으로 완료됐다(바이너리 SHA256 `658c00b7b99b57d54a12dd345dd44a3ba705fdffa6749687eed9a8e1a04306cc`). 모델 test에191→192→191→193 및 작은/큰 행 교대, cached128행을 추가했다. Full logits와 memcheck 이후 compact/buffered/rolling serving correctness 및 frozen prior/control/candidate/vLLM 비교를 실행한다. 아직 split 성능이나 GPU correctness 통과를 주장하지 않는다.


Split serving 비교 준비: current 바이너리를 `riley-ffn-split-serving-v1`로 고정했으며 prior는 이전 측정의 unified adaptive 바이너리 `riley-ffn-adaptive-serving-v1`(SHA256 `3ae029380abc732a96f65e361548daef92a794ea9a74905dd6e245ccaf4786fc`)다. Prior만 adaptive=1, control은 두 FFN flag=0, split은 split=1/adaptive=0이다. 네 번째 lane은 vLLM이다. 각 lane C32/active32/warm256/ret8192, shared·unique와 역순16 lane 비교를 유지한다. 준비 완료 시간과 준비 직후 RSS/global GPU memory를 추가 수집하지만 이를 isolated graph allocation으로 해석하지 않는다. 모델 gate가 완료되기 전 serving을 시작하지 않는다. Controller/client 원본은 timed phase 전 tmpfs에 저장하고 source hash를 묶는다.

[Split 모델 검증](../../benchmarks/results/20260914-ffn-split-model/README.md):full logits4,128,768 BF16 bytes exact, full-model memcheck0 errors, 두 실행exit0 및 Blender 복구 완료. Compact/buffered/rolling serving parity와 성능은 이어지는16 lane 비교에서 검증하며 현재 미완료다.


## Split C32 결과 — FFN 추가 변형 중단

[전체 비교와 검증](../../benchmarks/results/20260914-ffn-split-serving-c32/README.md):16 lane/131072 retained/4194304 tokens,98304 Riley reference 일치,stop/cancel/recovery 각96 및 Blender 복구를 검증했다.184개 파일4,735,499,038bytes의 archive/materialization hash와 controller/client snapshot도 확인했다. V1은 vLLM 준비300초 timeout으로 실패했고 보존했다. V2는 모든 엔진에 동일한600초 준비 한도를 사전 적용해 전체16개를 새로 실행했다.

Split throughput은 unified prior 대비 shared+0.90%/unique+0.75%, 같은 binary M16 control 대비+0.41%/+6.98%다. Shared control 대비 방향은 순서에 따라+0.93%/−0.10%로 바뀐다. vLLM 대비−13.77%/−16.83%로 목표 미달이다. 준비 시간 중앙값4757.68ms vscontrol1002.17ms, global GPU readiness memory1916 vs1900, 서버 RSS803986 vs795000KiB가 관측됐다. 메모리 측정은 graph allocation 격리값이 아니며 host IO PSI4.37–27.37%도 함께 보존한다.

기본값 비승격. FFN threshold/tile 작은 변형을 더 이어가지 않는다. 다음 의미 있는 영역은 남아 있는 PR06 attention 작업 분배다. Native/모델 수치 계약과128-token recurrence/BF16 probability rounding을 먼저 검토하고 graph/scratch/serving을 묶어 평가한다. 기존 FFN 개선을 새 attention 성능으로 합산하지 않는다.
