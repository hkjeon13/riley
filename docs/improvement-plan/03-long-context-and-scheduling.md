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
| 기존 구조 한계 | 0.1.0의 Coder-14B `closed-c32`와 일부 RAG의 최종 TTFT 목표 미달은 기존 판정과 측정값을 유지한다. | [구조 한계 근거](../goal.md#31-성능-a100-24개-모델-전부에-적용) |

단일 GPU Qwen2.5-14B의 네 chunk·KV 조합과 배포 이미지 Qwen2.5-32B의 TP=2 기준 설정에서 C32 장문 증거를 확보했다. 새 전역 기본값에는 후보 설정으로 영향 모델 전체와 TP=2의 회귀 검증이 필요하다. [14B 네 조합](03-long-context-result.md#장문-chunk와-kv-한도의-교차-대조) · [32B TP=2 기준](05-mixed-soak-result.md#배포-이미지의-32b-tp2-장문-검증)

## 8. 다른 작업에 넘길 결과

W1에는 새 측정·판정 필드를, W4에는 chunk·schedule의 정확성 경계를, W5에는 권장 프로파일·지원 범위·트래픽 시나리오를 넘긴다. 최종 설명은 모델·prompt 길이·동시성·부하·설정·기준 후보를 연결한다.
