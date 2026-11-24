# W3. 긴 프롬프트와 혼합 부하 성능 개선

작성일: 2026-11-13\
선행 조건: [W1 측정·판정 복구](01-validation-and-release.md), [W4 장문 정확성](04-correctness-and-schema.md)\
연계: [W2 비동기 스텝](02-async-step.md), [종합 계획](README.md)

Qwen2.5-0.5B의 burst 제한과 prompt 묶기 실측을 진행했다. 두 정책의 비용과 채택 판단은 결과 문서에 있다. [W3 실측 결과](03-long-context-result.md)

## 1. 남아 있는 문제

[0.1.0 RAG 재측정](../../benchmarks/evidence/20261104-goal/stage1t16-table.md)의 비율은 Riley/vLLM이며 지연은 낮을수록 좋다.

| 현상 | 대표 근거 | 해결할 질문 |
|---|---|---|
| 6K prompt 첫 토큰 지연 | Qwen3-4B rag6k-c1 1.19배, Qwen3-8B 1.17배 | prefill·attention·호스트 중 무엇이 문맥과 함께 늘어나는가? |
| 부하에서 tail 악화 | Qwen2.5-14B rag-c8 TTFT p95 1.44배, Coder-14B 1.43배 | 긴 스텝·대기열·KV·동시 도착 중 p95의 원인은? |
| prefill 개선이 decode 비용으로 이동 | Qwen2.5-0.5B rag-c8 TPOT 1.15배 | burst 범위·기동 조건을 어떻게 제한할 것인가? |
| 짧은 혼합 부하 잔여 지연 | Coder-14B closed-c32 TPOT 반복 합산 1.039배 | 혼합 256행 스텝의 추가 비용을 줄일 수 있는가? |

1,024행 burst·GEMM 동등 테이블·고정 KV 분할·element-wise kernel 최적화·불필요 LM head 생략이 이미 적용됐다. 큰 모델 rag-c1과 일부 p95는 구조 한계로 닫혔다. 같은 budget·GEMM 탐색은 새 근거 없이 반복하지 않는다.

## 2. 기존 목표와 범위

[목표 문서](../goal.md)를 유지한다.

- rag-c1/rag6k-c1 TTFT p50: 1단계 1.15배 이내, 최종 1.05배 이내.
- rag-c8/rag-r3 TTFT p95: 1단계 1.4배 이내, 최종 1.15배 이내.
- 짧은 워크로드·decode의 기존 회귀 제한을 유지한다. ±3% 동률과 회귀 허용치를 혼동하지 않는다.

구조 한계로 수용한 기록을 보존한다. 새 후보가 빨라져도 최종 기준에 못 미치면 부분 개선이다. 출력이 달라지는 split-K 재기준·attention 분할은 별도 승인·모델 재검증 대상이며 이 계획으로 실행 승인을 대신하지 않는다.

## 3. 조사 지도와 비교 조건

- [scheduler](../../crates/riley-scheduler/src/scheduler.rs), [계획](../../crates/riley-scheduler/src/plan.rs): prompt/decode 예산·KV 대기·공정성.
- [CLI](../../crates/riley-server/src/main.rs), [설정 문서](../configuration.md): burst·plateau·버킷·context 조합.
- [engine](../../crates/riley-server/src/engine.rs), [host trace](../../crates/riley-runtime/src/host_trace.rs): 입력→대기→GPU→응답의 임계 경로.
- [forward](../../crates/riley-runtime/src/llama/forward.rs), [batch executor](../../crates/riley-runtime/src/llama/batch_executor.rs): wide GEMM·attention·LM head.
- [비교 실행기](../../benchmarks/live/compare.py), [최신 성능표](../../benchmarks/evidence/20261104-goal/full16-table.md): 셀·설정·근거.

단어 수가 같아도 모델별 토큰 수가 다르므로 실제 prompt/output 길이를 기록한다. 14B는 KV 몫 차이로 reason 결과가 달랐던 이력이 있으므로 양쪽 KV 블록 수뿐 아니라 실제 바이트·수용량을 맞춘다.

## 4. 원인을 찾는 실험

### W3-1. 대기 시간 분해

요청별 `수신→토큰화 완료→admission→첫 prefill→마지막 prefill→첫 토큰 발행`을 측정한다. 각 GPU 스텝에 prompt/decode 행 수, 버킷, attention work, graph/eager, 시간, 남은 KV를 연결한다.

계측은 제안이며 현재 이 timestamp가 모두 공개돼 있다는 뜻은 아니다. 재현 ID를 사용하고 사용자 prompt를 운영 로그에 직접 쓰지 않는다. W5와 계측 비용을 확인한다.

### W3-2. 단독 길이 곡선으로 계산 비용 분리

짧은 입력부터 2K·4K·6K와 지원 상한 부근까지 C1에서 측정한다. 토큰화·GEMM·attention·기타 kernel·호스트 비용을 분리하고 실제 causal context·row 수를 기록한다.

kernel probe의 빠른 알고리즘은 실제 모델 activation과 서버 프로파일로 확인한다. 기존 문서에도 무작위 데이터의 GEMM 순위와 서버 결과가 달랐던 사례가 있다. 같은 바이트의 동등성 검증 전에는 row별 알고리즘 변경을 채택하지 않는다.

### W3-3. 혼합 부하로 tail 원인 분리

| 부하 | 바꾸는 축 | 보는 지표 |
|---|---|---|
| 짧은 decode 중 긴 prompt 삽입 | 삽입 시점·길이·decode 수 | 기존 inter-token gap p95/p99, 신규 TTFT |
| 긴 prompt 동시 도착 | C1/C8, 동기/지연 도착 | queue wait·최장 스텝·TTFT p95 |
| open-loop RAG | rate·burst·seed | 완료율·오류율·큐 길이·p95 |
| prefix cache | off/on, cold/warm, prefix 개수 | hit·tokenization·KV 압력 |
| KV 부족 | 동일 총 바이트에서 문맥·동시성 변화 | block 대기·공정성·starvation |

같은 입력·도착 순서를 기준/후보에 재생한다. closed-loop는 느려질수록 새 요청도 늦어지므로 open-loop와 구분한다. 예정 도착 시각과 실제 전송 지연을 기록해 부하 생성기 병목도 확인한다.

14B 후속 후보의 1% decode TPOT 회귀 판정에는 8개 동시 전송과 50ms 간격 전송을 모두 포함한다. 두 RAG seed의 TTFT 꼬리를 실행 순서를 뒤집어 반복하고, 가능한 진단 실행에서는 토큰 간격을 요청 ID와 스텝에 연결해 어느 구간에서 ITL 꼬리가 늘었는지 찾는다. [완료 가능한 prefill 후보의 요청별 결과](03-long-context-result.md#완료-가능한-prefill-우선-후보)

## 5. 개선 후보와 채택 순서

### W3-4. 바이트를 보존하는 스케줄 후보

1. **decode가 있는 스텝의 prompt 예산 제한:** 행 수만 보지 않고 측정된 모델별 스텝 비용으로 burst를 제한한다. 타깃 예산·비용표는 실험 전에 고정한다.
2. **대기 시간·공정성:** 짧은 요청 우선으로 긴 요청이 굶지 않게 대기 시간과 진행 보장을 검사한다.
3. **모델군별 정책:** 작은 2-KV-head와 14B의 비용이 다르므로 범용 기본값보다 명시적 프로파일·opt-in을 먼저 평가한다.
4. **넓은 prefill 선택:** decode가 없거나 여유가 있을 때 wide step을 사용한다. 기존 burst와 실제로 다른 조건·효과인지 검증한다.

스케줄도 chunk 구성을 바꾸므로 출력 불변성을 확인한다. 실행 중 kernel 선점을 가정하지 말고 다음 스텝 예산에서 조절한다.

### W3-5. 계산 병목이 확인된 경우

- **GEMM:** activation 기반 동등 테이블 선택·workspace·wide shape 사용을 확인한다. 이미 최적인 범위는 새 증거가 있을 때만 재탐색한다.
- **Attention:** 고정 token 경계·동일 fold/reduction을 지키며 work 분배·merge를 조사한다. reduction이 달라지면 수치 변경 트랙이다.
- **불필요 계산:** 무출력 행 LM head, padding, wide shape가 decode graph 예산을 소모하는지 확인한다.
- **호스트:** W2와 연결한다. 아직 검증되지 않은 비동기 이득을 전제로 정책을 채택하지 않는다.

disaggregated prefill·다중 프로세스·다중 호스트는 별도 설계다. 단일 GPU 후보의 근거와 한계가 나온 뒤 평가한다.

## 6. 실행 템플릿과 도구 한계

기존 캠페인 구조의 Linux GPU 호스트에서만 사용한다. 다음 변수는 실제 값으로 사전 지정하고 manifest에 기록한다.

```bash
python3 benchmarks/live/compare.py \
  "$CAMPAIGN_ROOT" "$RUN_LABEL" "$MODEL_NAME" "$GPU_ID" "$RILEY_BUILD_SUFFIX" \
  --order rvvr \
  --scenarios closed-c8,closed-c32,rag-c1,rag6k-c1,rag-c8,rag-r3,reason-c8,reason-c32
```

현재 실행기는 `ROOT/src/benchmarks/live/load.py`, `ROOT/models/<MODEL_NAME>`, `ROOT/bin/riley-<suffix>`, 전용 vLLM venv·CUDA 경로를 가정한다. 일반 clone에서 바로 실행된다고 보장하지 않는다. `--riley-short`, `--riley-long`, `--vllm-memory`, KV는 승인 설정으로 명시하고 기본값에 맡기지 않는다. W1의 실패 전파 보완 전에는 원본 JSON·종료 코드를 직접 확인한다.

## 7. 검증 확대와 종료 조건

초기 진단은 Qwen2.5-0.5B·3B·7B, Qwen3-4B·8B, Qwen2.5-14B·Coder-14B로 시작한다. 전역 scheduler 기본값을 바꾸면 최종적으로 24개 qualified 모델과 TP=2 영향 경로를 검사한다.

최종 p95는 사전 지정한 다중 seed·반복과 충분한 표본으로 판정한다. 셀당 완료 1,000건 이상·3회 반복을 신규 제안으로 검토하되 실행 시간·분산을 확인해 사전 확정한다. 기존 qualification 표본을 소급 변경하는 기준은 아니다.

- [x] Qwen2.5-14B의 RAG C8 seed 2와 50ms 유입에서 요청 대기와 호스트·GEMM·attention·KV 쓰기 trace를 같은 실행에 결속했다. [동시 계측](03-long-context-result.md#같은-실행의-요청-대기와-cuda-커널-대조)
- [x] 동시 계측의 두 부하에서 워밍업을 포함한 실제 입력으로 계산한 최대 KV 예약과 활성 요청 수가 설정 한도보다 작았다. [예약 상한 대조](03-long-context-result.md#같은-실행의-요청-대기와-cuda-커널-대조)
- [x] Qwen2.5-14B의 두 부하에서 KV 한도를 768블록으로 제한해 68건의 직접 부족 표식, 반환 뒤 입장, 104건 출력 일치와 종료 전 KV 회수를 확인했다. [예약 압력과 회수 대조](03-long-context-result.md#실제-kv-예약-압력과-회수)
- [x] 장문 경계 C32에서 768블록·512토큰 chunk와 기본 30초 입장 제한의 HTTP 408을 KV 부족 기록에 연결했다. [경계 부하 확인](03-long-context-result.md#장문-chunk와-kv-한도의-교차-대조)
- [x] Qwen2.5-14B의 요청별 prefill 대기와 다른 요청의 prefill 구간이 겹치는 시간을 실측했다. [요청별 계측](03-long-context-result.md#riley-14b의-요청별-대기와-prefill-계측)
- [x] 14B에서 진행 중인 prefill을 우선한 후보의 배치 불변성과 회귀를 확인했다. Decode TPOT 기준을 넘긴 설정은 미채택으로 판정했다. [후보 결과](03-long-context-result.md#riley-14b의-요청별-대기와-prefill-계측)
- [x] 14B에서 완료 가능한 prefill을 우선한 후보를 비교했다. 두 번째 RAG seed의 TTFT 꼬리와 두 동시 유입 조건의 decode 지연이 늘어 미채택했다. [후보 결과](03-long-context-result.md#완료-가능한-prefill-우선-후보)
- [x] 14B에서 혼합 스텝의 512행 제한을 두 실행 순서로 비교했다. RAG 첫 토큰 p95와 50ms 유입의 decode TPOT 회귀로 미채택했다. [후보와 검증 결과](03-long-context-result.md#14b-혼합-스텝의-행-예산)
- [x] Qwen2.5-14B의 1,024행 QKV 후보를 두 실행 순서로 비교했다. 출력과 decode·짧은 요청 기준을 충족했다. RAG 두 번째 seed의 첫 토큰 p95 방향이 뒤집히고 50ms 간격 유입의 p95가 두 번 모두 늘어 기본값 적용을 보류했다. [1,024행 스텝 결과](03-long-context-result.md#1024행-스텝의-계산-비용)
- [x] 같은 A100 바이너리에서 45번 요청의 도착 시각을 바꿔 두 실행 순서로 입장 경계의 영향을 확인했다. 앞뒤 시각에서는 기준과 후보의 prefill 순서가 같았고 p95가 줄었다. 기존 시각에서는 순서가 갈리고 p95가 늘었다. [실제 요청의 도착 시각 대조](03-long-context-result.md#실제-요청의-도착-시각-대조)
- [x] 두 seed에서 얻은 네 도착 시각표를 기준군과 QKV 후보에 그대로 재생했다. 두 실행 순서에서 TTFT p95와 TPOT p50이 감소했고 측정 1,024건의 출력이 재현됐다. [고정 도착 대조](03-long-context-result.md#두-rag-seed의-고정-도착-재생)
- [x] 형상이 같은 Coder-14B에서 1,024행 QKV 후보를 두 실행 순서로 대조했다. 출력은 일치했고 동시 유입의 decode TPOT이 1% 기준을 넘었으며 50ms 유입의 첫 토큰 p95가 두 번 모두 늘어 형상 공통 적용을 기각했다. [Coder-14B 결과](03-long-context-result.md#coder-14b의-1024행-후보-검증)
- [x] 시험한 burst 후보의 득실과 decode·짧은 요청 비용을 함께 기록했다. [W3 결과](03-long-context-result.md)
- [x] Qwen2.5-14B의 고정 입력 11개에서 chunk 512·1,024와 KV 768·2,576의 네 조합을 비교했다. 90초 진단으로 완료한 C1·C8·C32 308건에서 입력·생성 토큰 ID가 일치했다. [경계 정확성 결과](03-long-context-result.md#장문-chunk와-kv-한도의-교차-대조)
- [ ] 새 기본값 후보의 영향 모델과 TP=2에서 W4 장문·chunk 경계·배치 불변성 검사를 통과한다.
- [x] 같은 CUDA 후보와 Qwen2.5-0.5B에서 네 조합을 두 seed로 교차 측정했다. [조합 결과](03-long-context-result.md#w2-이벤트-발행과-묶음-정책의-상호작용)
- [x] 기존 달성 범위와 실험 후보의 부분 개선·기각·구조 한계를 기록했다. [현재 적용 판정](#현재-적용-판정)

가드레일을 지키는 레버 두 개가 각각 2% 미만이면 기존 규칙에 따라 닫는다. 특정 워크로드에만 유효하면 비용을 명시해 opt-in으로 둘 수 있다. 회수할 때 기존 정책·비용표·설정을 함께 복구한다.

### 현재 적용 판정

| 판정 | 적용 범위 | 근거 |
|---|---|---|
| 1단계 목표 충족 | 0.1.0의 기존 burst 측정에서 24개 모델이 `rag-c1`의 1단계 TTFT p50 기준을 충족했다. | [성능 목표와 재측정](../goal.md#31-성능-a100-24개-모델-전부에-적용) |
| 기존 opt-in | `--prefill-burst-steady-budget`으로 기존 burst를 켤 수 있다. 이번 비교에서 `after-submit-v1`과 기존 32행 burst 설정을 기준으로 사용했다. | [서버 설정](../configuration.md#execution-options) · [W2·W3 네 조합](03-long-context-result.md#w2-이벤트-발행과-묶음-정책의-상호작용) |
| 부분 개선 | Qwen2.5-14B의 1,024행 QKV 후보는 두 seed의 고정 도착 재생에서 지연을 줄였다. 일반 유입에서는 첫 토큰 꼬리 지연이 도착 순서에 따라 달라졌다. | [고정 도착 재생](03-long-context-result.md#두-rag-seed의-고정-도착-재생) |
| 기각 | 같은 형상의 Coder-14B에서 동시 유입 decode TPOT이 1.99% 늘어 1% 회귀 한도를 넘었다. 1,024행 QKV 계획의 형상 공통 적용을 기각했다. | [Coder-14B 교차 대조](03-long-context-result.md#coder-14b의-1024행-후보-검증) |
| 미채택 | 1,024행 QKV를 켠 Coder-14B의 첫 prefill 우선순위 후보는 50ms 유입의 두 실행 순서에서 decode TPOT p50을 5.57–5.68% 늘렸다. | [우선순위 교차 대조](03-long-context-result.md#coder-14b-서빙의-우선순위-교차-대조) |
| 부분 개선 | Coder-14B에서 반복 문서 네 개를 미리 적재한 prefix cache는 고정 도착 RAG의 TTFT p95를 96.01–96.17% 줄였다. 역순의 짧은 C32 TTFT p50은 3.03% 늘어 혼합 서비스 채택 기준을 넘었다. | [반복 문서 대조](03-long-context-result.md#coder-14b-반복-문서의-prefix-cache) |
| 미채택 | Coder-14B에서 128토큰 미만 요청을 캐시 조회·등록에서 제외해도 짧은 C32의 decode TPOT p50이 1.11–1.41% 늘었다. 반복 문서 RAG의 TTFT p95는 96.16–96.19% 줄었다. | [요청 길이 기준 대조](03-long-context-result.md#128토큰-미만-요청의-캐시-제외) |
| 워밍업 뒤 통과 | 같은 C32 요청을 한 번 더 실행하면 두 GPU에서 짧은 요청의 지연·처리량 기준을 충족했다. 첫 실행은 GPU 7번에서 TTFT p50이 8.80% 늘어 기본값 채택을 보류한다. | [첫 실행과 반복 실행](03-long-context-result.md#짧은-c32의-첫-실행과-반복-실행) |
| 시작 워밍업 미채택 | `step-shapes-v1`을 켠 Coder-14B 필터 실행은 준비 상태 전에 192행을 실행하지 않았고, 같은 워밍업의 캐시 비활성 실행보다 첫 C32 TTFT p50이 9.62–9.97% 늘었다. | [시작 워밍업 대조](03-long-context-result.md#시작-워밍업과-첫-c32-지연) |
| 형상 복원·꼬리 지연 미달 | Coder-14B에서 시작 워밍업만 공유 캐시에서 격리하면 192행이 다시 실행됐다. 일반 RAG 재사용과 C32 중앙값은 기준을 충족했고, 첫 C32 TTFT p95는 캐시 비활성·워밍업 켬 대비 16.96–27.19% 늘었다. | [캐시 격리 대조](03-long-context-result.md#시작-워밍업의-prefix-cache-격리) |
| 고정 도착 대조 | 같은 클라이언트 write 시각표에서는 격리 워밍업의 첫 C32 p95 증가가 0.4% 미만이었다. 캐시 비활성·워밍업 켬 대비 p50은 3.48–9.20% 늘어 기본값 채택을 보류한다. | [첫 C32 고정 도착 재생](03-long-context-result.md#첫-c32의-고정-도착-재생) |
| 구간 계측 | 계측을 끈 고정 도착 대조에서는 C32 p50이 14.55–18.29% 늘었다. 계측 실행에서 후속 64건의 prefill 배치와 겹친 엔진 큐 대기 차이는 GPU 6번 +2.25ms, 7번 −10.22ms였다. 계측을 끈 회귀의 원인은 미확정이며 기본값 채택을 보류한다. | [서버 구간별 지연](03-long-context-result.md#고정-c32의-서버-구간별-지연) |
| 기존 요청 관측 대조 | 새 타이머 없이 두 GPU에서 각각 두 번 대조했다. C32 전체 p50은 두 번 늘고 두 번 줄었다. 후속 64건에서 백엔드 제출부터 입장까지의 차이가 클라이언트 TTFT 차이와 같은 방향이었고 입장 뒤 첫 텍스트 차이는 0.27ms 이하였다. | [C32 입장 지연](03-long-context-result.md#기존-요청-관측값으로-대조한-c32-입장-지연) |
| 일반 prefill 예산 미채택 | 필터·격리 워밍업에서 192행과 128행은 256행보다 배치 한 개의 평균 GPU 시간을 줄였다. 두 예산 모두 C32의 스텝 수와 첫 토큰 p95를 늘려 채택하지 않았다. | [고정 C32 행 예산 대조](03-long-context-result.md#고정-c32의-일반-prefill-행-예산) |
| HTTP 워커 대조 | 128행에서 64워커는 연결 큐 대기를 줄였고 입장 뒤 지연이 늘어 전체 C32 p95는 비슷했다. 256행에서는 GPU 6번의 연결 큐와 첫 토큰 지연을 줄였고 GPU 7번의 필터 p95는 회귀 한도를 넘었다. | [HTTP 워커와 연결 큐](03-long-context-result.md#고정-c32에서-http-워커-수와-연결-큐-대기) |
| 활성화 대기 확인 | 128행에서 64워커는 두 GPU 모두 연결 큐 대기를 줄였고 활성화 대기를 늘렸다. 활성화 뒤 첫 prefill 계획까지의 p95는 1ms 미만이었으며 클라이언트 TTFT p95의 변화 방향은 GPU별로 달랐다. | [활성화와 prefill 구간](03-long-context-result.md#고정-c32의-활성-슬롯-대기와-첫-prefill) |
| 짧은 입력 burst 미채택 | 128행에서 짧은 입력도 burst에 포함하자 TTFT p95가 128행 기준 대비 14.54–36.99% 개선됐다. 256행 기준 대비 TTFT p50과 출력 처리량은 회귀 한도를 넘었다. | [짧은 입력 문턱 대조](03-long-context-result.md#고정-c32에서-짧은-입력의-burst-문턱-조정) |
| Page 48개 미채택 | A100 40GiB의 1,024행 조건은 page 작업 공간을 할당하지 못했다. GPU 7번의 192행 조건은 활성 48개에 도달했으며 같은 행 예산의 고정 타일 대비 출력 일치와 지연 기준을 충족하지 못했다. | [메모리 한계와 192행 대조](03-long-context-result.md#page-attention의-활성-슬롯-32개와-48개) |
| 고정 타일 48슬롯 미채택 | 두 GPU에서 1,312건의 출력을 보존했다. 128행에서는 실제 활성 43–45개로 늘었으나 C32 TTFT p95는 최대 0.20% 줄었고 p50이 두 GPU 모두 2% 회귀 한도를 넘었다. 256행에서는 실제 활성 수가 32개에 머물렀다. | [고정 타일 슬롯 대조](03-long-context-result.md#고정-타일의-활성-슬롯-32개와-48개) |
| 256행 버스트 미채택 | 984건의 출력을 보존했다. 16행 후보는 C32 TTFT p95를 6.84–21.84% 줄였으나 p50이 150.92–151.91% 늘었다. 8행 후보도 p50이 13.38–35.18% 늘었다. | [256행 예산의 버스트 대조](03-long-context-result.md#256행-예산을-고정한-짧은-입력-burst) |
| 짧은 입력 전용 512행 버스트 보류 | 8행 후보는 두 GPU에서 C32 TTFT p95를 5.15–9.78% 줄이고 출력을 보존했다. GPU 7의 비교 기준은 모두 통과했으며 GPU 6의 C32 TTFT p50은 9.45% 늘었다. | [512행 버스트 대조](03-long-context-result.md#짧은-입력-전용-512행-버스트) |
| 512행 버스트 입장 계측 | 새 교차 시험에서 계측 없이 두 GPU의 C32 지연 기준을 충족했다. 계측 실행에서는 GPU 6의 p50이 2.52% 늘었으며 후속 64건은 256행 초과 실행과 겹친 사례가 0건이었다. | [입장 구간과 반복 대조](03-long-context-result.md#512행-버스트의-엔진-입장-구간) |
| 512행 버스트 도착 민감도 | 후속 64건을 12ms 늦추자 GPU 6에서 계측한 큐 대기 차이 중앙값이 +9.38ms에서 −2.56ms로 바뀌었다. 계측을 끈 GPU 7은 두 시각표에서 TTFT p50이 각각 10.63%, 26.76% 늘어 기본 적용을 보류했다. | [도착 시각 교차 실행](03-long-context-result.md#후속-64건의-도착을-12ms-늦춘-교차-실행) |
| 기존 구조 한계 | 0.1.0의 Coder-14B `closed-c32`와 일부 RAG의 최종 TTFT 목표 미달은 기존 판정과 측정값을 유지한다. | [구조 한계 근거](../goal.md#31-성능-a100-24개-모델-전부에-적용) |

단일 GPU Qwen2.5-14B의 네 chunk·KV 조합과 배포 이미지 Qwen2.5-32B의 TP=2 기준 설정에서 C32 장문 증거를 확보했다. 새 전역 기본값에는 후보 설정으로 영향 모델 전체와 TP=2의 회귀 검증이 필요하다. [14B 네 조합](03-long-context-result.md#장문-chunk와-kv-한도의-교차-대조) · [32B TP=2 기준](05-mixed-soak-result.md#배포-이미지의-32b-tp2-장문-검증)

고정 C32의 두 도착 시각표와 두 GPU에서 p95는 처음 32건으로 결정됐다. 512행 후보의 가장 큰 p95 악화에서는 엔진 입장 및 입장 뒤 첫 텍스트 구간이 짧아졌고, 같은 요청의 write 이후 나머지 시간이 206.76ms 늘었다. 이 차이를 조사하는 데에는 본문 처리와 응답 전송 구간의 추가 관측이 필요하다. [C32 p95의 요청별 구간 확인](03-long-context-result.md#c32-첫-토큰-p95를-결정한-요청)

`benchmarks/live/load.py`의 `--records`와 `--transport-timing`으로 요청 write, 상태 줄 수신, 헤더 끝, 첫 SSE `data:` 줄, 첫 텍스트 사이의 시간을 기록한다. 고정 시각표 실행기에 기존 연결을 넘길 때에는 `one(..., capture_transport_timing=True)`를 쓴다. CPU 루프백에서 네 구간의 합을 확인한 뒤, A100의 같은 요청에서 서버 파싱·입장·소켓 쓰기 시각을 대조했다. 512행 후보의 p95는 네 비교에서 6.13–11.21% 줄었고 서버 전송 계측을 끈 두 비교에서 p50이 2% 한도를 넘어 기본값 적용을 보류했다. [부하 실행기](../../benchmarks/live/load.py) · [실제 시간 대조](03-long-context-result.md#c32의-httpsse-전송-시각-대조)

512행 버스트를 decode가 없는 스텝으로 제한하고 각 GPU에서 실행 순서를 바꿨다. 두 회차의 출력 656쌍은 일치했다. C32 TTFT p50은 GPU 6번에서 13.70–15.03% 늘고 GPU 7번에서 17.97–23.81% 줄었다. 역순 비교의 RAG p95는 두 GPU에서 늘었다. 현행 256행 기본값을 유지한다. [decode 0행 역순 대조](03-long-context-result.md#decode-0행-후보의-역순-대조)

`c5ec7b391` 배포 이미지에서 앞선 1,024행 옵션을 다시 적용해 A100 6·7번에서 vLLM과 교차 비교했다. Coder-14B 짧은 C32의 TPOT p50은 vLLM의 1.043–1.047배, RAG C8 TTFT p95는 1.434–1.672배였다. RAG C8 첫 토큰 지연의 추가 개선을 두 seed에서 확인하고, 짧은 C32 TPOT과 처리량을 함께 대조한다. [패키지 교차 비교](03-long-context-result.md#배포-이미지의-coder-14b-교차-비교)

짧은 C32는 vLLM 도착 시각표의 유효한 두 비교에서 TPOT p50 1.03배 목표를 충족했다. 콘텐츠 간격 p99는 Riley 119–125ms, vLLM 34–41ms였다. Riley 도착 시각표의 두 비교는 전송 오차 초과로 보류했다. [C32 고정 도착 결과](03-long-context-result.md#짧은-c32의-고정-도착-재생-결과)

같은 패키지에서 배치 한도는 고정하고 요청당 prefill 조각을 512행으로 낮췄다. 832쌍의 출력은 일치했으나 두 GPU와 두 RAG C8 seed 모두에서 첫 토큰 p95가 2.74–14.39% 늘었다. 네 비교 모두 커밋된 GPU 시간도 증가해 이 설정을 미채택했다. [요청당 prefill 조각 대조](03-long-context-result.md#coder-14b-요청별-prefill-조각-크기)

Coder-14B RAG의 첫 고정 도착 재생에서 유효한 GPU 6번의 두 비교는 첫 토큰 p95가 vLLM의 1.441배·1.450배였고 나머지 여섯 비교는 3ms 전송 오차 기준을 넘었다. GPU 상태 조회와 CPU 배치를 분리하자 GPU 7번은 RAG와 짧은 C32 6개 비교 모두 전송 오차를 통과했다. RAG 첫 토큰 p95는 0.991–1.529배였고 4개 중 1개에서 최종 1.15배 목표를 충족했다. [첫 고정 도착 결과](03-long-context-result.md#coder-14b-rag-고정-도착-재생-결과) · [CPU 분리 재생](03-long-context-result.md#gpu-상태-확인을-분리한-재생)

동일한 두 RAG 시각표에 서버 관측값을 연결했다. 추적한 두 요청의 첫 토큰 대기 중 약 94%가 입장 뒤에 발생했다. 별도의 스텝 계측에서는 두 요청이 다른 prefill 실행을 기다린 시간이 각각 1,215ms·780ms였다. 두 요청의 자체 prefill은 각각 4스텝 약 546ms·545ms였다. 기존 decode 선택을 보존하면서 오래 기다린 prefill의 진행 간격을 줄이는 후보를 우선 비교한다. [입장 구간 검증](03-long-context-result.md#같은-rag-요청의-입장과-첫-콘텐츠) · [prefill 스텝 대조](03-long-context-result.md#rag-prefill-스텝과-요청별-대기)

혼합 스텝에서 남은 예산을 접수 순서로 배정한 후보는 RAG seed 1의 TTFT p95를 두 GPU에서 11.84%·15.31% 줄였다. Seed 2 감소는 1% 미만이었고, 12개 중 7개 비교에서 decode TPOT가 1% 한도를 넘었다. C32 Riley 시각표의 TTFT p50도 두 GPU에서 2%를 넘겨 미채택했다. [스케줄 순서 교차 비교](03-long-context-result.md#혼합-스텝의-prefill-접수-순서-후보)

같은 1,024행을 채울 장문 두 요청의 순서만 바꾼 후속 후보는 두 seed의 RAG TTFT p95를 두 GPU에서 1.93–4.07% 줄였다. GPU 7번의 vLLM 대비 비율은 1.367–1.399배로 1단계를 충족했다. GPU 6번의 유효한 두 비교에서 decode TPOT p50이 1% 한도를 넘고 짧은 C32 한 비교의 TTFT p50이 2% 한도를 넘어 미채택했다. [장문 prefill 순서 후보](03-long-context-result.md#같은-행-수를-유지하는-장문-prefill-순서-후보)

이전 스텝 기록을 재계산하니 행 수를 그대로 채우는 교체 가능 스텝 37개 중 33개에서 causal attention 쌍이 증가했다. 같은 바이너리의 기준 반복에서 짧은 C32 TTFT 중앙값이 두 GPU에서 11–12% 달랐고, GPU 7번의 RAG seed 1과 동시 유입은 후보가 두 기준 모두에 대해 decode 회귀 한도를 넘었다. 교체 스텝의 실행 시간과 동시 decode 요청 ID를 연결하는 계측을 작성해 CPU에서 검증했다. [attention 계산량](03-long-context-result.md#같은-행-수에서의-attention-계산량과-기준-실행-차이) · [동일 바이너리 반복](03-long-context-result.md#동일-바이너리의-기준-반복과-prefill-후보) · [계측 준비](03-long-context-result.md#prefill-교체-스텝과-동시-decode의-계측-준비)

일반 CUDA의 행 예약 반복에서 RAG 토큰 간격 p99는 45.75–46.70% 줄었다. 짧은 C32의 이전 TTFT 중앙값 회귀는 재현되지 않았고 같은 설정끼리도 중앙값이 6.56–11.32% 달랐다. C32 p95는 설정을 바꾼 네 비교에서 −1.97%~+12.47%, GPU 7번 RAG p95는 저장 vLLM의 1.429–1.550배였다. C32 꼬리와 실제 RAG 목표를 충족할 때까지 행 예약의 기본 적용을 보류한다. [일반 CUDA 반복 비교](03-long-context-result.md#일반-cuda에서-행-예약의-반복-비교)

요청별 prefill을 연결한 진단에서 GPU 6번의 짧은 요청은 승인 뒤 약 75ms를 기다린 뒤 세 번째 스텝에서 전부 처리됐고, 나뉜 별도 요청은 두 조각 사이 약 76ms를 기다렸다. 짧은 혼합 스텝에서 이미 시작한 prefill을 먼저 마치는 CPU 후보는 기존 행 예산과 decode를 보존했다. [요청별 prefill 대조](03-long-context-result.md#짧은-c32-꼬리-요청의-prefill-배정)

RAG C8 두 시각표의 p95 요청 여덟 건이 다른 prefill 스텝을 기다린 시간은 923–1,237ms였다. GPU 7번의 네 대조에서 최종 첫 토큰 기준까지 311–455ms를 줄여야 했다. 가장 긴 대기 스텝 하나의 전체 시간은 각 사례에서 141–149ms였다. 여러 스텝에 걸친 prefill 배정과 decode 지연을 함께 대조한다. [RAG p95의 목표 격차](03-long-context-result.md#rag-p95-요청의-스텝별-시간과-목표-격차)

보존된 RAG 시각표의 최대 요청 KV 길이는 3,655토큰이다. 4,096토큰 문맥에서 2,048행에 필요한 고정 타일 작업 공간은 1,024행·8,192토큰 대비 24KiB 증가하며 행×문맥은 같다. 현행 CUDA 플래너는 스레드 하나당 행 하나를 읽어 1,024행에서 막힌다. 행 예산 확대에는 플래너가 요청 경계를 보존하며 1,024번째 이후의 행도 처리하도록 변경하고, 같은 문맥과 도착 시각표에서 출력·메모리·지연을 대조하는 과정이 필요하다. [2,048행 작업 공간과 실행 한도](03-long-context-result.md#고정-타일의-2048행-작업-공간과-실행-한도)

4,096토큰 문맥의 2,048행 후보가 A100에서 출력 1,920건과 별도 장문 토큰 ID 96건을 보존했다. RAG의 첫 토큰 p95는 시드 1에서 5.87–7.01% 줄었고 시드 2에서 4.60–5.72% 늘었다. 네 대조에서 두 시드의 TPOT p50이 21.02–24.37% 늘어 혼합 스텝에 2,048행을 허용하는 설정을 미채택했다. [실제 CUDA 교차 대조](03-long-context-result.md#4096토큰-문맥의-2048행-cuda-교차-시험)

보존된 RAG 꼬리 요청 56건에서 제출부터 마지막 prefill까지 decode 없는 prefill 스텝과 겹친 경우가 없었다. p95를 결정한 여덟 요청의 입장 이후 스텝 96개는 모두 혼합이었다. 같은 조건에서 decode 없는 스텝의 폭만 넓히는 시험의 우선순위를 낮추고, 실제 혼합 스텝의 실행 비용과 decode 지연을 함께 확인한다. [꼬리 구간 검산](03-long-context-result.md#rag-꼬리-요청과-decode-없는-prefill-스텝)

과거 1,024행 프로파일의 492개 스텝에서 gate/up·down을 A100 dense BF16 공시 최대치까지 줄이는 계산의 중앙값은 스텝당 10.04ms였다. 같은 RAG C8 시드에서 네 행렬곱에 모두 이 가정을 적용해도 첫 토큰 p95는 1,514.64ms로 당시 기준 1,407.89ms를 넘었다. 이후에는 긴 요청이 다른 prefill을 기다리는 스텝 수를 줄이면서 decode 지연을 지키는 배정을 우선 검증한다. [행렬곱별 시간과 계산 범위](03-long-context-result.md#1024행-행렬곱의-스텝당-시간과-개선-여지)

최근 RAG 기록에서 prefill에 쓰인 행 수와 스텝 시간을 그대로 두고 접수 순서로 재배정하니 첫 토큰 p95 계산값이 21.38–32.86% 줄었다. 기존 decode 행부터 배정하고 긴 prefill의 접수 순서를 여러 스텝에 걸쳐 유지하는 진단 후보를 A100 6·7번에서 교차 검증한다. 실제 첫 토큰 및 생성 지연과 출력 바이트를 같은 시각표에서 판정한다. [고정 스텝 재생과 검사 조건](03-long-context-result.md#고정-gpu-스텝에서-장문-prefill의-접수-순서-재생)

실제 교차 시험에서 두 RAG 시드의 첫 토큰 p95가 20.39–31.10% 줄고 GPU 7번에서는 vLLM의 1.019배·1.127배를 기록했다. 동시·50ms 유입에서 TPOT p50이 10.07–19.62% 늘어 적용을 보류했다. 같은 스텝에서 함께 대기하기 시작한 요청의 분포를 기준으로 다음 정책을 시험한다. [교차 시험과 회귀](03-long-context-result.md#장문-접수-순서의-cuda-교차-결과)

같은 계획 전에 입장한 장문 요청이 함께 대기할 때 순환을 유지한 후보는 50ms 유입의 TPOT 기준을 두 GPU에서 통과했다. RAG의 첫 토큰 p95는 11.28–18.61% 줄었으나 GPU 7번은 vLLM의 1.301배·1.161배에 머물렀고 C32 시각표 하나는 중앙값 회귀 한도를 넘었다. [입장 묶음 보호 결과](03-long-context-result.md#같은-계획-전에-들어온-장문-요청의-순환-유지)

짧은 prefill 이어서 처리하기 후보의 일반 CUDA 교차 시험에서 출력 1,920건을 보존했다. GPU 6번의 C32 첫 토큰 p95는 두 비교에서 9.34–17.64% 줄었고 GPU 7번에서는 1.15–2.54% 늘었다. GPU 7번 세 RAG 비교가 decode TPOT p50의 1% 회귀 한도를 넘겨 기본 적용은 보류한다. [일반 CUDA 교차 결과](03-long-context-result.md#짧은-prefill-이어서-처리하기의-교차-시험)

## 8. 다른 작업에 넘길 결과

W1에는 새 측정·판정 필드를, W4에는 chunk·schedule의 정확성 경계를, W5에는 권장 프로파일·지원 범위·트래픽 시나리오를 넘긴다. 최종 설명은 모델·prompt 길이·동시성·부하·설정·기준 후보를 연결한다.
