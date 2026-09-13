# Paired decode: recovered serving screen

두 decode graph를 먼저 제출하고 양쪽 drain 뒤 결과를 공개하는 opt-in 경로를 실제 HTTP serving에 연결했다. Native release build와 single 대비 HTTP correctness 검사는 통과했다. C16/C32 교차 screen에서는 처리량과 TPOT가 개선되지 않아 기본값은 `single`로 유지한다. 최종 vLLM 초과 성능 목표는 미달성이다.

## 조건과 측정 범위

- RTX 4090, driver 580.178.04, SmolLM2-135M BF16, 단일 GPU. GUI 유지, Blender 중지. 다른 CPU 서비스가 남은 공유 호스트의 exploratory screen이며 release qualification은 아니다.
- 동일 새 Riley binary의 `single`과 `paired-experimental-v1`, vLLM 0.27.1을 비교했다. 모델 revision은 `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. checkpoint와 tokenizer 파일 hash는 환경 자료에 보존한다.
- Natural prompt 16/128/398 tokens, output budget 32/64/128, 동일 corpus 순환. C16/C32 각각 순서 `single → paired → vLLM → vLLM → paired → single`. lane마다 새 server, warmup 192개와 retained 768개. warmup/준비 시간은 처리량에서 제외한다.
- max model length 1024, batch token budget 512, greedy temperature 0, seed 0, prefix cache 비활성. Riley KV 2048 blocks, vLLM GPU memory utilization 0.3으로 용량은 동일하지 않지만 이 workload를 수용하며 용량 포화 성능을 비교하는 실험은 아니다. 세 경로의 정확한 argv/env를 보존한다.
- Offline Python stdlib client가 C개의 동시 HTTP SSE 요청을 유지한다. Python은 Riley serving runtime에 포함되지 않는다. 처리량은 실제 반환 token 합계 / 첫 retained 요청 시작부터 마지막 완료까지의 시간이다. TTFT는 첫 token의 client 도착, TPOT는 첫/마지막 token 도착 간 평균, ITL은 인접 token 도착 간격이다. 여러 token이 같은 frame에 있으면 실제 관측 시각을 공유하며 보간하지 않는다.
- 요청 768개씩의 짧은 screen과 3개 prompt로 긴 soak·open-loop SLO·다양한 모델 품질을 증명하지 않는다. CPU 격리, GPU clock 고정, client 포화 검증은 하지 않았다. 재부팅으로 driver/toolkit/client가 복구됐으므로 과거 보고서 숫자와 직접 증감률을 계산하지 않는다.

## Serving 비교 결과

Throughput과 P50은 두 반복의 중앙값이다. P95/P99는 두 반복 중 더 큰 값으로 표시하여 tail을 평균으로 숨기지 않는다. 모든 latency 단위는 ms다. 개별 반복의 모든 값은 [C16](evidence/serving-c16-v1/completion.json), [C32](evidence/serving-c32-v2/completion.json), [집계 JSON](comparison.json)에 보존한다.

| C | 경로 | Output tok/s | TTFT P50 | TPOT P50 | E2E P95 | E2E P99 |
|---:|---|---:|---:|---:|---:|---:|
| 16 | single | 7,677.89 | 7.407 | 1.923 | 264.334 | 372.081 |
| 16 | paired | 7,327.30 | 8.526 | 2.003 | 273.816 | 378.403 |
| 16 | vllm | 7,681.09 | 13.722 | 1.804 | 270.339 | 352.380 |
| 32 | single | 10,399.27 | 10.691 | 2.868 | 401.886 | 493.318 |
| 32 | paired | 9,948.44 | 10.972 | 2.966 | 411.760 | 519.562 |
| 32 | vllm | 10,797.33 | 24.498 | 2.413 | 421.238 | 494.695 |

| C | 경로 | TTFT P95 / P99 | TPOT P95 / P99 | ITL P95 / P99 |
|---:|---|---:|---:|---:|
| 16 | single | 22.311 / 36.812 | 2.174 / 5.227 | 5.902 / 6.869 |
| 16 | paired | 21.013 / 139.783 | 2.128 / 4.502 | 5.809 / 6.839 |
| 16 | vllm | 31.882 / 48.469 | 2.166 / 4.465 | 5.095 / 6.841 |
| 32 | single | 56.084 / 77.327 | 3.247 / 5.993 | 7.087 / 7.885 |
| 32 | paired | 56.152 / 75.564 | 3.563 / 7.719 | 7.019 / 7.936 |
| 32 | vllm | 68.923 / 119.441 | 3.493 / 6.432 | 6.364 / 13.517 |

- C16: paired throughput는 single 대비 -4.57%, vLLM 대비 -4.61%. TPOT P50은 single 대비 +4.13%로 악화됐다.
- C32: paired throughput는 single 대비 -4.34%, vLLM 대비 -7.86%. TPOT P50은 single 대비 +3.44%로 악화됐다.

최종 12개 lane의 retained 9,216개 요청은 HTTP/SSE 검사 오류 0건이다. Riley 8개 lane의 warmup+retained 7,680개는 기존 reference와 전부 일치했다. vLLM의 prompt ID·출력 길이·finish는 모두 동일하지만 token/text는 C16 1,155/1,536, C32 1,088/1,536만 Riley reference와 exact였다. 따라서 cross-engine bitwise 동등성이나 품질 qualification 통과로 해석하지 않는다. [Fixture별 검사](correctness-by-fixture.json)에 차이를 보존한다.

Paired 종료 counter는 C16 1,943/1,938회(최대 폭 16), C32 886/888회(최대 폭 32)다. 준비·warmup·retained를 포함하는 server 수명 전체 count이며 retained pair 적용 비율을 뜻하지 않는다. 모든 최종 lane의 shutdown exit는 0이다. vLLM 종료 로그에 resource_tracker semaphore 경고가 남았으며 장시간 안정성 검증은 하지 않았다. 종료 후 GPU compute process는 없고 swap 사용량은 0이었다.

## Correctness와 실행 확인

`http-smoke-v3`: single/paired 각각 warmup 96개와 retained 96개가 기존 reference의 prompt ID·출력 token ID·text hash·finish와 일치했다. 각 경로에서 stop-string 요청 96개를 실행해 token/text/finish/usage가 single과 paired 간 일치함을 확인했다. 각 32개 연결을 최소 4 token 뒤 출력 예산 전에 닫고, 이후 retained 요청이 reference와 일치함을 확인했다. Paired window는 251회, 최대 폭 32로 실제 실행됐다.

이 검사에서 생성 모델의 자연 EOS 발생이나 모든 실패·race 경로를 검증했다고 주장하지 않는다. 선행 native model의 memcheck 0 errors 증거는 유지하며, 재부팅으로 유실된 full-model racecheck 결과는 미확인이다.

## 복구와 재현

`/tmp`의 이전 toolkit/source가 재부팅으로 사라져 persistent root `/data/riley-serving-260913-recovery`를 사용했다. CUDA nvcc/runtime/CRT/NVVM 13.0.88, CCCL 13.0.85, cuBLAS 13.0.2.14를 task 전용 디렉터리에 설치했다. 기존 vLLM 환경은 수정하지 않았고 vLLM은 자체 cu13 library 경로를 사용한다. toolkit의 `lib64 → lib`, libcudart/cuBLAS linker 이름과 현재 driver library 연결을 복구했다. [빌드 명령](build.sh)으로 실제 CUDA release 빌드가 통과했다. 복구 도중 CUDA 13.1 cuBLAS header와 13.0 compiler의 불일치를 발견해 같은 13.0 계열로 고정했다.

첫 C32 실행 `serving-c32-v1`은 vLLM 컴파일에서 `ninja` PATH 누락으로 readiness 이전 실패했다. 해당 로그와 부분 Riley 결과를 보존하며 최종 표에서 제외한다. 설치된 venv/bin을 vLLM PATH에 추가한 뒤 C32 전체 순서를 다시 시작했다.

이전 boot의 2026-09-13 20:00:24 커널 로그에는 `decode_window_g` RSS 약 37.3 GiB, swap 잔여 0과 global OOM이 기록됐다. 별도 devtron 프로세스가 OOM 종료됐으며 재부팅의 직접 원인은 확정하지 않는다. 전체 모델 racecheck는 다시 실행하지 않았다. 향후 sanitizer는 좁은 kernel 범위와 명시적 메모리·시간 제한을 먼저 적용한다.

## 성능 판단과 다음 범위

현재 경로는 같은 CUDA stream에서 두 graph를 순차 실행한다. GPU 연산 자체를 줄이거나 두 graph를 하나의 persistent kernel로 합치는 변경이 아니다. 다음 token을 GPU에서 전달해 중간 host wait를 줄이는 대신 두 계획·packet·결과 검증을 유지하고, 두 번째 drain까지 첫 token 공개를 늦춘다. 이 구조는 코드로 확인되지만 어느 비용이 회귀를 지배했는지는 이번 wall-clock screen만으로 단정하지 않는다.

추가 micro-optimization이나 기본값 변경은 하지 않는다. 다음 batch 전에 같은 serving trace에서 pair 적용 비율, plan/packet 준비·검증/commit 비용, graph 사이 GPU idle, SSE 공개 지연을 함께 계측해 가설을 구분한다. CPU 준비·검증이 지배하면 예약/metadata 재사용과 제출·완료 분리를 묶어서 재설계하고, GPU 연산이 지배하면 기존 연구 계획의 attention/FFN backend 영역으로 돌아간다. 후행 drain 때문에 tail이 악화되는 경우 첫 결과 공개와 후행 KV 수명 분리를 correctness 계약과 함께 설계해야 한다. 기존 두 graph 방식만으로 성능 개선을 확인했다고 표시하지 않는다.

Hopper/Blackwell/multi-GPU 실행은 장비 부재로 skip이며 지원 검증 완료로 계산하지 않는다.

## Evidence index

- [Controller](../../analysis/paired_decode_serving_screen.py), [frozen workload](../20260913-serving-decode-window/workload.json)
- [환경·compiler·library/model hashes](evidence/environment.json), [C32 binary/client/workload hashes](evidence/serving-c32-v2/preparation.json), [C16 동일성](evidence/serving-c16-v1/preparation.json)
- [소스 458개 snapshot](source-hashes.json), [새 persistent source readback 전체 일치](remote-source-readback.json). Riley/vLLM의 model.safetensors·tokenizer.json·config.json SHA256도 서로 일치한다.
- [실제 release build](evidence/logs/server-build.log), [exit 0](evidence/logs/server-build.exit)
- [HTTP smoke 결과](evidence/http-smoke-v3/completion.json), [paired 실행 counter](evidence/http-smoke-v3/c32-p0-paired.log)
- [초기 vLLM PATH 실패](evidence/serving-c32-v1/c32-p0-vllm.log), [이전 boot OOM](evidence/logs/previous-boot-oom.txt)
- 각 lane 디렉터리의 `*-retained.json.gz`, `*-warmup.json.gz`, `*-stop.json.gz`는 full SSE frame·token/text·client timestamp 원본을 압축 보존한다. [Manifest](evidence/raw-manifest.json)의 압축/원본 hash를 모두 대조했다.

재실행은 별도 빈 output 경로로 수행한다. `python3 paired_decode_serving_screen.py ROOT workload.json OUT --warmup 192 --retained 768 --concurrency 32`이며 C16은 마지막 값만 16으로 바꾼다. `--smoke --warmup 96 --retained 96 --concurrency 32`는 추가 stop/disconnect 검사를 포함한다. 이 환경 전용 경로가 들어 있으므로 다른 장비에서는 경로·hardware 조건을 명시적으로 새 manifest에 기록한다.
