# Exact-order prefill FFN model and serving integration

Prefill FFN의 입력 공유·두-stage asynchronous copy와 shared-bank padding을 전체 모델 및 serving에 연결했다. C32에서 직전 paired 대비 output throughput **+2.57%**를 관측했지만, 같은 screen의 vLLM 대비 **−6.43%**다. E2E P99도 직전보다 나빠 기본값으로 승격하지 않는다. 2026-09-13 작업을 이어 09-14 KST에 검증했으며 최종 serving 목표는 미달성이다.

## 구현 범위

- Rust session → retained C ABI recorder → CUDA mixed/prefill model에 새 backend를 연결했다. `--ffn-backend prefill-pipeline-experimental-v1`은 loopback V7에서만 선택하며 `--decode-window paired-experimental-v1`과 함께 사용할 수 있다. 기존 decode FFN 옵션과 구분한다.
- Graph fingerprint에 backend ID와 kernel source를 포함한다. 기존 tiled weights·activation·KV owner를 사용하며 새 device/host workspace는 없다. 기본값은 기존 single이다.
- Mixed/prefill graph에 포함된 모든 packed row의 gate/up와 down에 적용한다. Pure decode kernel과 successor preparation overlap, pair settlement/publication 규약은 유지한다.
- 서버 runtime은 Rust→C ABI→CUDA다. Python은 외부 benchmark 및 artifact 분석에만 사용한다.

## C32 비교

RTX4090, SmolLM2-135M BF16, 고정된 동일 checkpoint/tokenizer/natural workload. 입력16/128/398·출력32/64/128, active32/waiting64, token budget512. 각 lane fresh server, warmup192+retained768. Single→직전 paired→후보 paired→vLLM 및 역순이며 GPU compute 작업을 겹치지 않았다. GUI 유지·Blender 종료, profiler/host phase timing 없음. Throughput과 P50은 두 반복 중앙값, P95/P99는 더 나쁜 반복이다.

| 경로 | Output tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| single | 10,443.75 | 10.727 | 2.855 | 395.154 | 468.863 |
| 직전 paired | 10,667.76 | 10.977 | 2.732 | 384.575 | 469.798 |
| prefill FFN paired | 10,941.56 | 10.671 | 2.660 | 378.606 | 478.745 |
| vLLM 0.27.1 | 11,693.60 | 20.162 | 2.300 | 350.431 | 464.973 |

[전체 percentile/집계](comparison-c32.json), [반복별 원본](evidence/prefill-ffn-c32-v1/completion.json). 후보/직전 throughput은 두 순서 +2.61%/+2.52%, TPOT P50 약−2.64%다. E2E P99는 약+1.90%로 안정적인 전 지표 개선은 아니다. 과거 screen의 vLLM 수치와 섞어 비율을 계산하지 않는다.

## Client C64 / active32

Client concurrency64, 양쪽 active capacity32. 나머지 workload·warmup·retained·순서는 C32와 같다.

| 경로 | Output tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| single | 10,589.01 | 224.817 | 2.870 | 631.404 | 768.035 |
| 직전 paired | 11,025.47 | 218.689 | 2.741 | 609.180 | 724.163 |
| prefill FFN paired | 11,211.43 | 212.935 | 2.684 | 605.318 | 729.648 |
| vLLM 0.27.1 | 11,914.66 | 197.766 | 2.485 | 576.957 | 722.035 |

[전체 percentile/집계](comparison-c64.json), [실행 조건](evidence/prefill-ffn-c64-v1/preparation.json). 후보/직전 처리량 +1.69%, 각 순서 +2.55%/+0.82%다. vLLM 대비 처리량 −5.90%이며 TTFT/TPOT도 더 높다. E2E P99는 직전 대비 약+0.76%다. 작은 이득을 고부하 전반의 안정적 개선으로 일반화하지 않는다.

## Correctness와 검증

- [GPU model test](evidence/prefill-ffn-model.log): synthetic boundary prompt32개, 자유 생성1,024토큰 차이0, 반복 prompt batch invariance 통과. 별도 자연어8문장×32target×49,152 vocabulary의 **12,582,912 BF16 logits bitwise 일치**, 양쪽 allocation zero. Logit dump SHA는 두 파일 모두 `492a46578581d131ab67c8c1cdb2d70f36a6539c868530e36269f1484f19f939`다.
- [CUDA release build](evidence/cuda-build.log)24.73초 통과. CLI33개 및 config6개 테스트 통과. Native kernel memcheck/racecheck 및 SM90a/SM100a compile은 [직전 native gate](../20260913-prefill-ffn-pipeline-native/README.md)에 있다. 이번 batch에서 kernel 본체는 변경하지 않았다. Hopper/Blackwell runtime은 장비 부재 미검증이며 multi-GPU 검증도 남아 있다.
- [Serving smoke](evidence/prefill-ffn-smoke-v1/completion.json): single/후보 각 warmup192+retained768 reference exact. 각 stop96개 token/text/finish/usage 동일, disconnect32개 후 retained 요청 정상. HTTP/SSE 오류0.
- 후보 binary SHA `e8b7f59bf95b6bab49c33db18988af3f600e9fd47fa08171fabdba8ed1ec156a`, 직전은 `a5036ff769a960d48dbd96fc971b6a5a3899c32c267c86da72137df76c0f0eaf`. [실행 metadata](evidence/prefill-ffn-c32-v1/preparation.json), [동기화한 소스 SHA](evidence/source-hashes.json). 빌드 이후 추가한 내용은 CPU 테스트와 설명 주석이며 production 동작은 동일하다.
- [Raw manifest](evidence/raw-manifest.json)는 원격 원본 경로·크기·SHA를 보존한다. Request gzip은 frames만 제외하고 token/text/usage/checks/timestamps를 유지한다. 전체 SSE frames와 model dumps는 `ai-assistant:/data/riley-serving-260913-recovery`에 남아 있다.

이 screen은 shared-host closed-loop 탐색 측정이다. CPU isolation/clock lock, open-loop SLO, 장시간 soak 및 다양한 모델의 qualification은 아니다. Riley/vLLM의 완전한 연산·KV capacity 동일성이나 cross-engine bitwise 일치를 주장하지 않는다.

## 판정과 다음 범위

C32/C64 각각 retained6,144개 HTTP/SSE 오류0, Riley warmup+retained5,760개 reference exact, 모든 lane 정상 종료다. vLLM retained의 Riley reference exact 수는 C32 1,138/1,536이며 C64 1,125/1,536이다. Cross-engine token 차이를 숨기지 않으며 이번 변경의 strict parity는 Riley 기존 경로를 기준으로 검증했다.

Native M32/128에서 관측한 큰 상대 이득이 전체 serving에서는 약1.7–2.6%로 줄었다. Target FFN은 전체 graph의 일부이고 큰 M에서는 native 이득 자체가 작았다. 이는 기존 profile/native gate와 일관되는 해석이며, 새 HBM counter 측정이나 정확한 인과 분해 결과는 아니다.

후보는 명시적 실험 옵션으로 유지한다. PR05의 일반 shape/모델 및 장기 안정성은 미완료다. 같은 FFN family의 미세 variant를 계속 추가하기보다 PR04 mixed batching·시간 예산/자원 계획의 지원 계약과 평가 workload를 다음 영역으로 검토한다. PR03의 실패한 수치 backend를 그대로 승격하거나, runtime 미검증을 완료로 취급하지 않는다.
