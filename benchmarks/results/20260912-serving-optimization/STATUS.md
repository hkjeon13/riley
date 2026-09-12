# Serving 최적화 진행 상태

전체 목표는 **미달성**, 현재 채택 기준선은 **Batch7 API baseline**이며 **Batch8은 기각**했다. 완료된 Round14 C1/P128/O32 HTTP token-delivery 비교에서 baseline은 vLLM보다 throughput과 TTFT가 좋지만 median TPOT는 **5.424% 느리다**. +15% throughput 목표, TPOT 동등 이하, 높은 concurrency 안정성 검증은 남아 있다.

## Round14 C1 완료·Batch8 기각 / Round15 진단 완료

Round14는 **C1 세 비교 각각 5쌍, 총 15쌍·30개 독립 서버·30,000개 retained 요청**을 완료했다. 각 서버의 nonstream 5개·stream 5개 warmup도 모두 strict exact ID/text/usage 검증을 통과했다. C1 회귀가 반복돼 controller를 SIGINT로 종료했으며, 원래 **30-pair 계획은 수정하지 않은 채 의도적 미완료**로 남긴다. C2 서버는 한 개도 시작하지 않았다.

V3는 모든 raw 응답·pair 해시·정리·복원 검증을 통과한 C1 세 항목만 보고했다. 전체 campaign은 여전히 `incomplete`이고 성공 `completion.json`은 없다. 아래는 **다섯 paired ratio의 중앙값**이다. Throughput은 클수록, latency는 작을수록 좋다.

| 비율 방향 | Throughput | Token TTFT median | Token TPOT median | E2E median |
| --- | ---: | ---: | ---: | ---: |
| Baseline / vLLM | 1.037739 | 0.701692 | 1.054239 | 0.979405 |
| Batch8 / baseline | 0.942007 | 0.999462 | 1.073050 | 1.062003 |
| Batch8 / vLLM | 0.985882 | 0.693044 | 1.131935 | 1.036563 |

Baseline/vLLM 항목의 process 통계 중앙값은 throughput **918.839 / 885.684 tok/s**, token TTFT **5.169137 / 7.366680 ms**, token TPOT **0.948936 / 0.900157 ms**, E2E **34.634668 / 35.357063 ms**다. Batch8/baseline 다섯 쌍 모두 throughput과 median TPOT가 나빠졌고, paired 중앙값 기준 각각 **-5.799% / +7.305%**다. 따라서 Batch8을 채택하지 않고 baseline을 유지한다. 정확한 절대값·paired 범위·P95/P99와 grouped-frame 해석은 [ROUND14_RESULTS.md](ROUND14_RESULTS.md)에 있다. 이는 HTTP 전달 시각이며 과거 engine 시간과 구분한다. 1,000-request tail 관측도 높은 concurrency 안정성 증거는 아니다.

Round15는 8개 독립 프로세스·128개 strict HTTP 응답·4,096개 native replay를 완료했다. Baseline의 원래 RoPE/KV + attention 두 enqueue를 한 구간으로 측정한 30-layer 합계는 AB/BA **360,448 / 359,936 ns**, Batch8 fused 구간은 **422,400 / 422,912 ns**였다. Candidate가 **17.188% / 17.496%** 오래 걸렸으며, operator timing off whole decode도 baseline **0.943216 → 0.943296 ms**, candidate **1.003232 → 1.009632 ms**였다. Event on의 약 **12–14%** 교란이 있으므로 실제 serving 시간이나 SASS 원인으로 해석하지 않는다. 회귀 구간을 좁히는 진단이며 Batch8 기각 결정은 Round14 serving 결과에 근거한다. [ROUND15_PROFILE_RESULTS.md](ROUND15_PROFILE_RESULTS.md)에 집계 경계·원본 검증·별도 build 증빙을 기록했다.

Round15 종료 후 Blender 세 successor를 **2730220 / 2730294 / 2730392**, 포트 **9876 / 9911 / 9887**로 복원하고 **2026-09-12 12:29:26 KST**에 실제 프로세스를 재확인했다 (`raw/blender-round15/verified.json`, `raw/round15-live-restoration-recheck.json`). 명령·GUI 환경·private GL readiness와 관측된 vendor mapping을 검증했다. CUDA는 첫 프로세스에 로드됐고 나머지 두 개는 lazy loading 상태였다. Round14의 이전 복구 기록은 해당 receipt에 보존한다.

로컬에서는 후속 다중 요청 batch의 scheduler 정책을 격리 소스에서 구현해 CPU 57개, descriptor codec/소유권·전체 결과 검증 prototype은 CPU 22개를 통과했다. 후속 실제 GPU synthetic row attention **34,560개 + retained graph transition 3개**, precise 연산 **3,456개 + transition 3개**도 bitwise exact로 통과했다. 각각 allocation **432,017 / 107,179개**를 전부 해제했고 live bytes·cleanup error는 0이다. 반면 anchored projection V2는 M2/M4 10개 모두 unsupported여서 **1,089개 case를 전부 건너뛰었고 수치 동등성 증거가 없다**. 후속 standalone M2/M4 후보 20개는 모두 admission에 성공하고 2,178개 case를 실행했지만, 두 workspace arm 각각 BF16 word 861개가 M1 기준과 달라 채택하지 않았다. 실제 scheduler/graph/server·모델 다중 요청 통합과 serving 성능은 미검증이다. [Round15 문서의 probe 표](ROUND15_PROFILE_RESULTS.md#후속-다중-요청-구성요소-준비-상태), `batch9-scheduler-prototype/README.md`, `../../prototypes/multisequence_descriptor_v1/README.md`에 범위를 기록했다. 원본 model-loader 24개 파일은 재확인 결과 변경되지 않았다.

별도 HF eager/cache-off/B1 공통 prefix 진단도 완료했다. Output index 8은 BF16의 1443·2341이 동률이지만, index 29는 FP32/BF16 모두 638이 단독 최고값이었다. 이는 실제 vLLM graph logits가 아니므로 높은 concurrency 오류 판정이나 정답 기준 변경으로 사용하지 않는다. [COMMON_PREFIX_LOGITS_RESULTS.md](COMMON_PREFIX_LOGITS_RESULTS.md)에 전체 배열 증빙과 실제 vLLM graph logits를 수집해야 하는 범위를 기록했다.

Fusion-history-v1은 GPU 수치 8,640개, 실제 모델 GPU 4개, 서버 72개 및 profile 10개 검사를 통과했다. 별도 증빙 재검증도 성공했다. HTTP 92개 검사와 별도 raw 재검증도 통과했다. Round16 복구 helper 및 현재 successor read-only 검사는 완료했으나 serving 비교는 미실행이며 후보는 미채택이다. [FUSION_HISTORY_V1_RESULTS.md](FUSION_HISTORY_V1_RESULTS.md)에 새 소스·검증 범위를 기록했다.

## 완료된 Batch7 및 과거 측정

동일 RTX 4090 / SmolLM2-135M BF16 / c1 / 입력 128·출력 32 / GUI 유지 조건이다. 각 lane은 5개 독립 프로세스 × 30개 retained 요청이며 아래는 process별 값의 중앙값이다.

| HTTP 경로 | Output tok/s | E2E ms | First-text ms |
| --- | ---: | ---: | ---: |
| 기존 baseline | 62.376 | 512.918 | 441.425 |
| Batch1 stage 분리 (단독 기각) | 58.937 | 543.051 | 476.328 |
| Batch2 P128 병렬 prefill | 353.408 | 90.418 | 23.651 |
| Batch3 HTTP 대기 제거 | 394.541 | 80.974 | 14.269 |
| Batch4 packed decode | 420.459 | 76.013 | 14.319 |
| Batch5 attention 병렬화 | 654.542 | 48.725 | 14.285 |
| Batch6 M16 prefill | 804.884 | 39.569 | 5.181 |
| **Batch7 two-warp attention** | **916.999** | **34.806** | **5.174** |
| vLLM (Batch7 비교 쌍) | 894.053 | 35.424 | 7.438 |

Batch7는 Batch6 대비 HTTP throughput **+13.929%**, E2E **-12.037%**다. 다섯 쌍의 Riley/vLLM throughput 비율은 모두 **1.012–1.068**, 중앙값 **1.0315**다. Campaign 중앙값끼리 나눈 비율은 **1.0257**이며 두 집계를 구분한다.

별도 engine 측정은 Riley **939.925 output tok/service-s, TTFT 4.624290 ms / TPOT 0.948323 ms / E2E 34.025157 ms**, vLLM **826.963 tok/service-s, 7.185962 ms / 0.938211 ms / 36.401517 ms**다. Batch6 대비 TPOT **-14.548%**, TTFT **+0.044%로 사실상 동일**하다. 다섯 쌍 TPOT 비율의 중앙값은 **1.0122**이므로 TPOT 동등 성능을 아직 주장하지 않는다.

위 과거 30-request HTTP campaign의 first-text는 token TTFT가 아니며, 그 campaign에서는 HTTP TPOT를 측정하지 않았다. 후속 Round14는 별도 token-ID 전달 시각을 측정했다. Engine throughput은 요청 service time의 합으로 계산해 vLLM의 느린 tail 요청에도 영향을 받는다. HTTP wall throughput과 서로 다른 지표이며 median E2E의 역수도 아니다. Process당 30개 요청의 nearest-rank P95/P99는 높은 concurrency 안정성 증거가 아니다. 전체 분포와 과거 비교는 `batch7-comparison.json`에 있다.

Batch7 소스 `1a2be0df01fe49daa4d4db155ad5c44f34ead6df`는 F107/g11 구현이며 g04 numerical gate를 유지한다. 실제 GPU에서 **8,640개 synthetic attention 케이스**가 두 기존 oracle과 전체 576개 BF16 출력이 일치하고, 세 KV block 배치에서도 같은 결과를 냈다. Guard/input/oracle 불변성, **60,480개 allocation 해제**, zero live/error를 확인했다. 별도로 실제 모델 full logits/status, 96개 전체 KV snapshot, 6개 retained 요청·42개 invalid case, 취소·재사용, legacy GPU, profile CLI, 두 HTTP sampler 검증을 새로 통과했다. Qualification SHA256은 `4a12927e05a1596e548cda536469b79c273941f0586e25bf27c5760a38f149dd`다.

Batch6에서 측정한 30-layer attention 합계 **0.482304 ms**가 이번 batch의 근거다. Batch7 별도 whole-graph 진단은 prefill **4.779152 ms**, decode **0.961136 ms**다. 이 summary는 warmup을 포함한 6개 요청(6 prefill / 186 decode)이므로 serving 성능 수치로 사용하지 않는다. 세부 근거는 `ATTENTION_BATCH7.md`, 이전 prefill/attention 근거는 `PREFILL_PROFILE.md`와 `OPERATOR_PROFILE.md`에 보존한다.

추가 GPU allocation은 없으며 prefill **397**, decode **277** kernel을 유지한다. Batch4에서 추가한 packed weight/output **139,353,984 bytes** 비용은 계속 포함된다. 기존 model-loader 24개 파일의 시작 해시를 보존했고 로컬 commit/push는 하지 않았다.

Batch7의 12개 decode operator 진단을 완료했다. 30-layer attention 합계 중앙값은 **0.314368 ms**, gate/up은 **0.244736 ms**, RoPE/KV는 **0.107520 ms**였다. Operator timing을 끈 whole decode 전후 값은 **0.969760 / 0.970976 ms**다. 서로 다른 프로세스에서 측정한 연산군 값을 합산하거나 event 영향을 실제 serving 시간으로 해석하지 않는다. 상세 범위와 해시는 `DECODE_BATCH7_PROFILE.md`에 있다.

별도 HTTP concurrency runner는 17개 CPU/loopback 검증 후 Round12에서 7개 설정 탐색을 시작했다. C=1은 양쪽 256개 retained 요청을 완료했고 Riley **922.821**, vLLM **875.929 tok/s**였다. C=2 Riley는 **933.788 tok/s**, E2E **68.452 ms**, first-text **38.811 ms**로 256개를 통과했다. 이어진 vLLM C=2/budget128은 첫 nonstream warmup 2개가 기존 정답 텍스트/length 검증에 실패해 중단됐다. C=2 비교 쌍과 나머지 5개 설정은 미완료이며 높은 concurrency 성능을 주장하지 않는다. 기존 c1 receipt와 별도인 **768개 retained 요청·2개 실패 warmup**을 보존했다. 상세는 `CONCURRENCY_SCREEN.md`다.

Round12 중단 후 Blender 3개를 동일 명령·환경·포트로 복구하고 검증했다 (`raw/blender-round12-restore-verified.json`): PID **4177821 / 4177822 / 4177823**, 포트 **9876 / 9911 / 9887**. 이후 2026-09-12 09:44 KST에도 세 프로세스와 포트가 살아 있음을 확인했다. GUI 512 MiB / foreign CUDA compute 없음 / 시작 48°C 이하가 측정 조건이며 canonical 256 MiB 충족을 주장하지 않는다.

2026-09-12 06:43 KST 호스트 자동 업데이트가 NVIDIA 사용자 라이브러리를 **580.178.04**로 교체했지만 로드된 커널 모듈은 **580.173.02**여서 기본 NVML은 exit 18로 실패한다. 서명된 Ubuntu snapshot에서 기존 compute·GL 라이브러리를 `/tmp`에 추출했고, 별도 환경으로 실행한 NVML과 새 EGL·GLX offscreen context는 통과했다. 이 runtime 준비 과정에서는 시스템 패키지·커널 모듈·실행 중 Blender를 변경하지 않았다. `raw/driver-incident-host-audit.json`과 `raw/driver-runtime-gui-probe-v2/completion.json`에 근거가 있다. 당시의 새 context 생성 증빙은 이후 Round13·14에서 수행한 전체 Blender 복구 검증과 구분한다.

이 별도 runtime에서 vLLM C=2 응답 10개 진단에 이어 **7개 설정·123개 응답**을 추가 수집했다. 입력 ID·출력 32개·usage·length 종료는 모두 유효했다. C=2/budget256은 9개 모두 기존 정답 토큰과 일치했지만 C=4·8은 budget 증가만으로 일치하지 않았다. 일부 설정은 같은 서버의 offered C=1에서도 달랐다. 수치적 원인은 확정하지 않았고 성능 결과로 사용하지 않는다. `VLLM_OUTPUT_MATRIX.md`에 전체 범위와 차이를 기록했다. HTTP 토큰 ID·usage 출력 수정본의 별도 CUDA 빌드를 완료했고, 실제 모델 GPU 수치·수명 검사 4개, CUDA server lib 72개, profile 10개 및 기본 HTTP 두 sampler 검증을 통과했다. 새 opt-in 응답도 별도 g04 HTTP 검증 92개를 통과했다. 두 sampler·O32/O1/stop·빈 토큰 3개·C1/2/4/8·연결 종료 후 재사용을 확인했다. 이 프로필은 C02 audit와 호환되지 않아 source commit 경로·실제 unit/model GPU 검사·HTTP 정답 ID를 결합해 검증했으며 per-request C02 audit를 주장하지 않는다. 이 correctness 검사는 성능 측정이 아니며, 후속 Round14에서 별도 token-delivery 성능을 측정했다.

과거 원본은 `raw/`의 **16개 HTTP/engine campaign (80 pairs, 160 process runs, 4,800 retained requests)**와 source/binary/proof, tests, diagnostics, 복구 receipt에 보존했다. Round14의 별도 완결 C1 데이터 **15 pairs, 30 process runs, 30,000 retained requests**를 이 과거 계약에 합치거나 완결된 전체 campaign으로 집계하지 않는다. 짧은 고정 c1 workload 결과를 넓은 serving workload나 높은 concurrency 성공으로 일반화하지 않는다.

이후 검증한 Batch8은 RoPE·KV 저장·attention을 통합했다. 별도 source `8329c1aeec6e013f581128888c536e15f8bf7300` / F108 빌드를 완료했고 실제 GPU 8,640개 케이스에서 attention·Q·전체 K/V가 기존 precise RoPE + Batch7 attention과 비트 단위로 일치했다. 103,680개 allocation 해제, 입력·guard·비현재 KV 보존, block mapping 불변성도 통과했다. 이어 실제 모델 GPU 검사 4개(3개 프롬프트·96개 전체 KV snapshot·취소/재사용), server lib 72개와 profile 10개도 통과했다. Batch8의 기본/opt-in HTTP 검증 92개도 두 sampler에서 통과했다. 수치 및 HTTP correctness는 통과했지만 후속 Round14의 다섯 candidate/baseline 반복에서 serving 회귀가 확인돼 기각했다. 채택된 baseline은 Batch7 API baseline이다. 근거는 `FUSION_BATCH8.md`와 `raw/batch8-fusion-probe/`에 보존한다.

2026-09-12 10:54 KST Round13 토큰 측정은 첫 vLLM C1 lane에서 중단됐다. Warmup 10개와 retained 123개는 통과했으나 124번째 응답이 `[314,338]` 두 토큰을 하나의 정상 SSE 프레임에 담아 V1 측정기의 메시지당 1-token 제한에 걸렸다. 수치적 출력 오류로 확정하지 않으며 완성된 비교 쌍은 0개다. 새 grouped-token 수신 지원은 별도 V2 client/contract로 검증한 뒤 Round14에 적용했다. 이 Round13 부분 응답은 완성된 응답이나 성능 결과로 재해석하지 않는다. `raw/token-serving-round13/`에 원본을 보존했다.

과거 Round13 finalization에서 Blender 3개의 복구와 실제 관측된 private GL/compute 매핑을 검증했다. 새 PID는 **2234550 / 2234607 / 2234700**, 포트는 **9876 / 9911 / 9887**이며 round14 read-only preflight에서도 확인했다. 복구 receipt SHA256은 `9bf3b9d644015fec626122271d973e63fcab70f4ee9427320757d009375e1fb0`다.

## Round16 완료

C1 screening(3쌍 × 300요청, 18프로세스·5,400 retained 응답)을 완료하고 원본 재검증을 통과했다. 후보/vLLM은 throughput **+12.29%**, TTFT **−30.84%**, TPOT **−3.40%**다. 후보/baseline은 throughput **+7.68%**, TPOT **−8.44%**다. 확대 검증 대상이며 아직 baseline 교체나 전체 목표 달성을 주장하지 않는다. [ROUND16_RESULTS.md](ROUND16_RESULTS.md)에 비율·범위·tail 해석과 증빙을 기록했다.

Controller 3011507은 종료됐다. Blender successor PID **3081024 / 3081081 / 3081259**, 포트 **9876 / 9911 / 9887**의 실제 명령·GUI 환경·시작 identity·listening을 재확인했다. Round16 helper는 사용 완료됐으므로 다음 측정은 이 successor를 바탕으로 새 복구 세션을 준비해야 한다.

## 다중 요청 GEMM 후속 검사

기존 M1 알고리즘을 유지한 strided batch 2/4는 다섯 projection의 compact/padded stride **20개 모두 지원 검사에 통과**했다. 최소 정렬은 A/B/C/D 모두 2바이트, workspace는 0이었다. 아직 GEMM 실행이나 수치 동등성 검사는 하지 않았다. [MULTISEQUENCE_STRIDED_CAPS_RESULTS.md](MULTISEQUENCE_STRIDED_CAPS_RESULTS.md)에 실제 opaque algorithm·원본 build·결과를 기록했다. 다음은 실제 가중치 bitwise 검사다.

## M1 strided batch 수치 검사 완료

실제 121개 가중치 그룹의 compact/padded stride **2,178개 case가 BF16 bitwise exact**로 통과했다. 입력·가중치·메모리 가드와 자원 정리, 실행 전후 private runtime mapping 재검증도 통과했다. 이는 이전 M2/M4 수치 차이를 피할 통합 후보지만 아직 retained graph·scheduler·server 통합과 serving 성능 증거는 없다. [MULTISEQUENCE_STRIDED_ARITHMETIC_RESULTS.md](MULTISEQUENCE_STRIDED_ARITHMETIC_RESULTS.md)에 범위를 기록했다.

## M1 strided retained graph 검사 완료

2,178개 case에서 같은 graph exec를 원본→0→원본 입력으로 세 번씩 실행했다. **6,534 replay**, 최종 독립 M1 BF16 비교 불일치 0이다. 자원 정리도 통과했다. H2D 및 synchronize는 그래프 외부이고 전체 decode/scheduler 전이는 미검증이다. [MULTISEQUENCE_STRIDED_GRAPH_RESULTS.md](MULTISEQUENCE_STRIDED_GRAPH_RESULTS.md)에 범위를 기록했다. 다음은 plan 수명·다중 행 전체 graph·scheduler commit을 하나의 통합 batch로 연결하는 작업이다.

## 다중 요청 통합 batch 시작

격리 snapshot `0f419241b121e725a057cd1031db3df445afa1f8`에 native strided M1 plan 생성·수명·byte extent 검증과 기존 단일 요청 graph admission 분리를 구현했다. 기존 native execute/close 경로로 2,178개 case bitwise 비교 및 잘못된 config 8개 거부 검사가 통과했다. Rust 래퍼와 multi-row owner/scheduler 연결은 진행 전이다. [MULTISEQUENCE_INTEGRATION_STATUS.md](MULTISEQUENCE_INTEGRATION_STATUS.md)에 현재 코드 위치와 남은 통합 범위를 기록했다.

## Rust strided plan 소유권 연결

격리 snapshot `9737c7f98fd1220d4c3fccfceee60a52466b3344`에 별도 Rust config/plan 타입과 FFI를 연결했다. CPU library 86개, CUDA feature build/link, 실제 GPU Rust 경계 20개 조합을 통과했다. Multi-row graph ledger와 scheduler bulk commit은 아직 미연결이다. [통합 상태](MULTISEQUENCE_INTEGRATION_STATUS.md)에 증빙과 남은 범위를 기록했다.

## Strided GEMM graph 경계 연결

Snapshot `38fba1183142581301156d6fbbc5f55e713ae5da`에 별도 strided graph 상태·진입점과 Rust borrowed owner를 추가했다. GPU 20개 조합·60 replay, 기존 canonical/selected graph 회귀 2개, CPU library 86개가 통과했다. 전체 aggregate decode owner와 scheduler는 여전히 미연결이다. [통합 상태](MULTISEQUENCE_INTEGRATION_STATUS.md)에 고정 입력 replay 범위와 후속 연결을 기록했다.
