# A100 에이전트 워크플로 검증

계획은 [종합 이정표](../../../docs/agentic-serving/README.md)에 있다. 초기 단일 호스트 시험은 aws-a100의 별도 프로세스와 루프백 HTTP 연결을 사용했다. 05단계의 두 호스트 기능 시험은 aws-a100과 ai-assistant의 전용 시험 연결을 사용했다. 서버 이미지와 모델 경로, GPU 예약, 입력과 출력 지문은 각 원본에 들어 있다.

| 파일 | 재현하는 결과 |
|---|---|
| [baseline-raw.tar.gz](baseline-raw.tar.gz) | GPU 7, 캐시 끔·켬·켬·끔. 당시 실행한 Python 드라이버, 이미지 실행 명령, 20개 작업과 종료 시 예약 |
| [profile-raw.tar.gz](profile-raw.tar.gz) | GPU 6, 진단 빌드. 두 이력 길이에서 요청 ID·GPU 이벤트·클라이언트 수신 시각 연결 |
| [baseline-holdout.tar.gz](baseline-holdout.tar.gz) | GPU 6 배포 이미지의 대응 결과와 GPU 7 보류 시드의 기준 결과 |
| [prefill-raw.tar.gz](prefill-raw.tar.gz) | GPU 7, 같은 배포 바이너리의 256행 기준·넓은 prefill 후보 교차 실행 |
| [prefill-verified.json](prefill-verified.json) | 원본에서 재집계한 완료 지연·스트리밍 간격·토큰당 시간·넓은 스텝 실행 횟수 |
| [phase05-multinode-routing-fault-20261126-raw.tar.gz](phase05-multinode-routing-fault-20261126-raw.tar.gz) · [검산](phase05-multinode-routing-fault-20261126-verified.json) | 서로 다른 GPU 호스트의 순차 분배 28개 작업과 시험 데이터 연결 장애 3개, 라우팅·SQLite·GPU 반환 원본 |

저장소 루트에서 아래 명령으로 prefill 후보 결과를 독립 재계산할 수 있다.

```sh
python3 benchmarks/evidence/20261124-agentic-serving/verify_prefill.py   benchmarks/evidence/20261124-agentic-serving/prefill-raw.tar.gz
```

요청별 캐시 검사에는 `cargo build --release --offline -p riley-server --features server --example qwen_prompt_profile`로 빌드한 진단 도구와 같은 Qwen 체크포인트의 토크나이저 메타데이터가 필요하다. [audit_agent_prefix.py](../../live/audit_agent_prefix.py)는 각 요청의 재구성 해시·실제 입력 길이를 대조한다. 오프라인 인코딩 시간은 진단 도구를 실행한 기기의 시간이다. GPU 호스트에서 관측한 토큰화와 스텝별 CUDA 시간은 [join_trace.py](join_trace.py)로 계산한다. GPU 시간은 동일 스텝에 함께 참여한 요청끼리 중복될 수 있다.

원격 시험 디렉터리는 `/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/agent-workflow-20261124.5qtoVRPa`이다. `reference/`는 GPU 6의 패키지와 vLLM 대조에, `cohort/`는 GPU 7의 네 세션 혼합 비용 계측에 사용한다. 각 경로의 로그와 실제 프로세스 종료·결과를 함께 확인한 뒤 판정한다.

[aligned-raw.tar.gz](aligned-raw.tar.gz)의 동일 GPU 교차 실행을 풀고 `python3 verify_decode.py <압축을 푼 경로>/aligned`로 기준 구성의 decode 결과를 재계산한다. 입력 토큰 ID·응답 길이·도착 중첩 검사 후 요청별 TPOT 중앙값과 가시 콘텐츠 간격 p95를 계산한다. [aligned-verified.json](aligned-verified.json)에 각 물리 기동 쌍의 값을 남겼다.

[cohort-raw.tar.gz](cohort-raw.tar.gz)을 풀고 `python3 replay_cohort.py <압축을 푼 경로>/cohort`로 혼합 스텝 리플레이를 재생한다. [replica-first.tar.gz](replica-first.tar.gz)을 풀고 `python3 verify_replicas.py <압축을 푼 경로>/replicas`로 두 복제본의 순차 분배, 인증·캐시 격리, 취소 후 자원 회수를 검산한다.

[replicas-raw.tar.gz](replicas-raw.tar.gz)의 여섯 기동을 풀고 `python3 verify_replicas.py <경로>/replicas`를 실행하면 제출 분모와 목표 충족 용량을 재집계한다. [operations-raw.tar.gz](operations-raw.tar.gz)을 풀고 `python3 verify_operations.py <경로>/operations/first`를 실행하면 두 정상 종료의 자원 해제, 비정상 종료 코드, SQLite 도구 효과와 끊긴 SSE를 교차 검사한다. [단일 호스트 운영 판정](operations-verified.json)

공개 산업 트레이스는 Liu 외(2026)의 *Agentic Coding in the Wild* 자료를 사용했다. CC BY 원본의 식별자·본문을 저장소에 옮기지 않고 통계와 토큰 길이·소요시간만 재구성했다. `python3 audit_public_trace.py date.2026-06-01.tar.gz`가 공개 릴리스 SHA-256을 확인한 뒤 등록한 8개 샤드를 집계한다. [공개 데이터·출처](https://github.com/Azure/AzurePublicDataset/blob/master/GitHubCopilotCodingAgentDataset2026.md) · [파생 통계](public-trace-verified.json)

[다섯 기동 쌍과 정책 시험 원본](promotion-raw.tar.gz)을 한 디렉터리에 풀고 `python3 verify_promotion.py decode <경로>/promotion`으로 동일 길이 decode를 검산한다. `python3 verify_workflow_policies.py wait <경로>/promotion`과 같은 명령의 `citations` 모드는 대기 예측과 출처 지시를 각각 검산한다. [decode](promotion-decode-verified.json) · [대기 정책](wait-policy-verified.json) · [출처 판정](citation-verified.json)

[32K 시험 전체 원본](public-shapes-raw.tar.gz)을 풀고 `python3 verify_serving_limits.py shapes <경로>/public-shapes`를 실행하면 8K 거절, fixed-tile 기동 실패, page 경로의 토큰 길이와 종료 이유를 모두 재집계한다. [32K 검산](public-shapes-verified.json) · [45K 실험 검색 원본](long-page-probe-raw.tar.gz)

[30분 부하 원본](soak-raw.tar.gz)의 1,200개 행은 `python3 verify_serving_limits.py soak <경로>/soak/run`으로 도착 시각표·취소·지연 목표·종료 자원을 재계산한다. [부하 검산](soak-verified.json)

[포화 시험 원본](crossover-raw.tar.gz)의 6개 동시 라우팅 비교와 240개 제출은 `python3 verify_serving_limits.py crossover <경로>/crossover/first`로 재생 결과를 집계한다. [복구 재시험 원본](operations-repeat-raw.tar.gz)은 `python3 verify_operations.py <경로>/operations-repeat/retry`로 저장된 도구 결과, 실제 목적지와 정상 종료를 검증한다.

[공개 토큰 길이 기반 캐시 시험 원본](public-cache-raw.tar.gz)은 `python3 verify_public_cache.py <경로>/public-cache`로 동일 프롬프트 지문, 실제 적중량, 준비 포함·제외 시간을 계산한다. [32K 재사용 결과](public-cache-verified.json)

[인벤토리·슬롯 배치 원본](inventory-raw.tar.gz)에 대해 `python3 verify_capacity_route.py <경로>/inventory-repeat`가 실제 요청 목적지와 진행 스트림 중첩을 검증한다. 같은 아카이브의 `sample/natural-lookup.json`을 `verify_lookup_decode.py natural`에 로컬 `qwen_prompt_profile` 실행 파일·토크나이저 경로와 함께 주면 자연 종료 토큰의 제안 가능량을 다시 계산한다. [라우팅](inventory-verified.json) · [보류 시드 decode](lookup-natural-verified.json)

[장문 실험·생성 마감 원본](long-resume-raw.tar.gz)은 `python3 verify_long_page.py <경로>/long-resume native`로 4개 공개 길이 표본, 2개 장거리 검색, 종료 자원 해제를 검산한다. `deadline` 모드는 1초 마감에서 받은 실제 SSE 오류를 기록한다. [장문 결과](long-resume-verified.json)

[32K 프로파일 기동 원본](named-prefix32-raw.tar.gz)을 풀고 `python3 verify_named_profile.py <경로>/named-prefix32 public-cache-verified.json`을 실행하면 배포 진입점 인자·바이너리·두 개의 답변 해시·재사용 토큰 수·GPU 반환을 대조한다. [결과](named-prefix32-verified.json)

Coder-14B 두 시드의 고정 스텝 민감도는 `python3 estimate_rag_step_cost.py ../20261124-w3-rag-prefill-steps/verified.json`으로 다시 계산한다. 요청 도착과 배치 구성을 바꾸지 않았을 때의 p95를 [스텝 길이별 계산](rag-step-sensitivity.json)에 기록했다.

[장문 attention 원본](long-tile-raw.tar.gz)과 [이전 Llama page 원본](long-resume-raw.tar.gz)을 같은 디렉터리에 해제한 후 `python3 verify_long_tile.py <경로>/long-tile <경로>/long-resume`으로 요청·서버 설정·실물 GPU·배치 경로·종료 상태를 대조한다. 공개 길이를 재현한 요청은 합성 문장으로 구성했다. [검산 결과](long-tile-verified.json)

[역순 기동과 스트리밍 간섭 원본](long-tile-holdout-raw.tar.gz)에 `python3 verify_tile_holdout.py <경로>/long-tile-holdout`을 사용한다. [별도 GPU의 사업 기록 검증 원본](long-tool-records-raw.tar.gz)은 `PYTHONPATH=../../live python3 verify_long_records.py <경로>/long-tool-records`로 재집계한다. [간섭 재집계](long-tile-holdout-verified.json) · [출처별 정확성](long-tool-records-verified.json)

명시적 ID 조회 시험에서 첫 기동은 캐시를 끄려고 `--prefix-cache-blocks 0`을 전달해 두 서버 모두 준비 전에 거절됐다. 요청은 전송되지 않았다. [초기 기동·GPU 반환 기록](exact-records-startup-raw.tar.gz). 수정한 실행기는 해당 옵션을 생략하고, 실제 응답별 캐시 적중 토큰이 0인지 확인한다.

[명시적인 키 조회 원본](exact-records-raw.tar.gz)을 해제해 `PYTHONPATH=../../live python3 verify_projection.py <경로>/exact-records-cold`로 두 A100의 교차 기동을 비교한다. 검산기는 최종 호출에 실린 레코드를 원래 도구 응답에서 독립적으로 다시 선택하고 전체 요청 해시·모델 정답·캐시 0·기동 설정·장치 반환을 확인한다. [6쌍의 정확성·종단 시간](exact-records-verified.json)

Coder-14B의 1,024행 행렬곱 회수 상한은 `python3 audit_coder_roofline.py ../20261124-w3-wide-projection-profile rag-step-sensitivity.json`으로 계산한다. 서로 다른 두 계측의 차이를 유지한 [492개 스텝 검산 결과](coder-roofline-verified.json)다.

[Coder-14B 혼합 트래픽 원본](coder-mixed-fleet-raw.tar.gz)을 해제해 `PYTHONPATH=../../live python3 verify_coder_fleet.py <경로>/coder-mixed-fleet protected-raw.tar.gz`으로 확인한다. [도착 시각표·실험 조건](coder-fleet-registration.json)과 기존 단일 엔진 응답을 대조한다. 순차 분배 구간에서 다른 복제본을 선택할 수 있었던 조건은 `python3 audit_fleet_opportunities.py <경로>/coder-mixed-fleet`으로 재생한다. [응답·지연](coder-mixed-fleet-verified.json) · [선택 시점별 상황](coder-fleet-opportunities.json)

[동일 GPU 두 장의 vLLM 원본](coder-fleet-reference-raw.tar.gz)은 `PYTHONPATH=../../live python3 verify_coder_fleet.py <경로>/coder-fleet-reference ../20261124-w3-separated-client/initial-results.tar.gz --vllm`으로 독립 검산한다. 두 엔진 원본을 해제한 뒤 `PYTHONPATH=../../live python3 compare_coder_engines.py <경로>/coder-mixed-fleet <경로>/coder-fleet-reference`에서 실제 입력·출력 길이와 본문 일치 집합의 TPOT를 확인한다. [엔진별 지연](coder-fleet-comparison.json)

[단일 호스트 이미지 롤백 원본](image-rollback-raw.tar.gz)을 해제해 `python3 verify_image_rollback.py <경로>/image-rollback`으로 이전·신규·복귀 이미지 ID, 실제 실행 파일 바이트 지문, 컨테이너 경로, 장애·복구·도구 효과를 함께 검산한다. [사전 조건](image-rollback-registration.json) · [인수 결과](image-rollback-verified.json)

[512행 혼합·단일 GPU 시험 원본](coder-gap-budget-raw.tar.gz)을 앞선 [256행 대조 원본](coder-mixed-fleet-raw.tar.gz)과 같은 위치에 해제한다. `PYTHONPATH=../../live python3 verify_coder_gap.py <경로>/coder-gap-budget <경로>/coder-mixed-fleet ../20261124-w3-separated-client/initial-results.tar.gz`이 실제 넓은 스텝 증가량, CPU 제출 오차, 엔진별 응답 일치, 두 GPU 혼합 및 한 GPU 보호 시험 결과를 대조한다. [검산 결과](coder-gap-budget-verified.json)

### 장문 생성 토큰과 점수 계산 재현

아래 명령은 이 디렉터리에서 실행한다. 각 원본 묶음은 이름이 겹치므로 서로 다른 빈 디렉터리에 해제한다. Python에 NumPy가 필요하다.

```sh
work=$(mktemp -d)
mkdir "$work/layer" "$work/score"
tar -xzf register-layer-raw.tar.gz -C "$work/layer"
tar -xzf register-score-raw.tar.gz -C "$work/score"
PYTHONPATH=../../live python3 verify_register_layers.py "$work/layer"
PYTHONPATH=../../live python3 verify_register_score.py "$work/score" "$work/layer"
PYTHONPATH=../../live python3 verify_router_prefix.py
```

첫 검산은 독립 Transformers 실행과 이전 page·fixed-tile 원본에 연결한다. 두 번째 검산은 같은 모델 파일 9개의 지문, 실행 파일, 캐시 0, 교차 기동 순서, 입력·생성 토큰, 가시 콘텐츠 시각, 각 GPU 반환을 확인한다. 점수 후보의 연산 오차는 원래 입력으로 복원한 네 위치의 기준 벡터와 비교한다. 세 번째 명령은 복제본 재생 클라이언트의 CPU 시험으로 CUDA를 사용하지 않는다.

두 경로의 첫 레이어 K/V 파일 전체는 시험 호스트에 보관했다. 파일 지문·페이지 인덱스와 계측 결과는 `register-kv-raw.tar.gz`에 들어 있다. 전체 K/V가 있는 호스트에서는 `OPENBLAS_NUM_THREADS=1 python3 replay_register_kv.py <KV-실행-경로> <레이어-실행-경로> --oracle-dir <출력-경로>`로 점수를 다시 계산할 수 있다. [실험 개요와 결과](../../../docs/agentic-serving/03-mixed-scheduling.md#최종-답변-토큰이-달라진-위치에서-attention-수치-재현)

### 자연 종료 지연과 도구 설명 표시 시험 재현

`python3 decompose_natural_final.py`는 기존 다섯 기동 쌍의 50개 자연 종료 작업을 비교한다. `PYTHONPATH=../../live python3 verify_fleet_breakdown.py`는 복제본 원본의 90개 제출과 운영 보고서가 일치하는지 검사한다.

[도구 설명 표시 교차 원본](final-layout-raw.tar.gz)과 [선택 필드의 교차 원본](final-layout-api-raw.tar.gz)을 별도 디렉터리에 해제한다. 검산 명령의 `probe`는 Riley의 `qwen_prompt_profile` 실행 파일, `tokenizer-dir`은 원래 Qwen2.5-7B-Instruct의 `tokenizer.json`과 `tokenizer_config.json`이 있는 디렉터리다. 같은 버전의 `agent_workflow.py`는 각 원본에도 들어 있다.

```sh
PYTHONPATH=../../live python3 verify_final_layout.py <교차-원본> <probe> <tokenizer-dir> > crossover.json
PYTHONPATH=../../live python3 audit_final_layout_answers.py crossover.json
PYTHONPATH=../../live python3 verify_final_layout_api.py <필드-원본> crossover.json <probe> <tokenizer-dir>
```

후속 시험은 같은 원본 레코드와 도구 설명을 유지하되 4,096바이트 이상의 마지막 도구 결과에만 이 방식을 적용했다. 다른 조건에서는 해당 결과에 p95의 정의와 해석 범위를 함께 담았다. [조건·GPU 순서](final-layout-selective-registration.json)와 [출력 지문별 해석 검토](final-selective-reviewed-answers.json)를 보존했다. 후속 원본을 별도 디렉터리에 해제한 뒤 `PYTHONPATH=../../live python3 verify_final_selective.py <후속-원본> crossover.json <교차-원본> <probe> <tokenizer-dir>`로 두 GPU에서 검사한다.

### 채팅 요청 준비 비용 검산

`chat-owned-render.patch`는 `18cd37cef`의 `crates/riley-server/src/chat.rs`에 적용한다. 도구 설명을 별도로 표시하는 `final-tool-layout.patch`도 같은 기준 파일에서 수행한 별도 실험이다. 두 패치의 원본 서버 동작을 재현할 때는 각각 독립된 소스에서 시작한다.

`chat_render_trial.py cpu`는 같은 요청에 두 실행 파일을 교대로 사용해 MiniJinja 렌더링을 포함한 요청 정규화와 토큰화를 각각 측정한다. 원본의 `recorded.json`은 앞선 GPU 실험에 실제 전송한 60개 요청의 복원에 쓰인다. `chat_render_trial.py serve`는 실제 HTTP의 첫 요청, 5개 순차 재요청, 8개 동시 재요청 및 잘못된 요청의 400 응답을 검사한다. `qwen_prompt_profile` 실행 파일은 CPU 검산에서만 사용한다.

[두 CPU 호스트와 네 GPU 기동의 원본](chat-render-raw.tar.gz)을 빈 디렉터리에 해제한 뒤 `PYTHONPATH=../../live python3 verify_chat_render.py <원본> <원본>/darwin.json`으로 다시 검산한다. Qwen과 Llama의 34개 요청·오류 조합은 묶음의 `contract.json`에 있다. [CPU 구간과 HTTP 결과](chat-render-verified.json)

[다른 프로세스의 76개 진행 스트림](cross-router-raw.tar.gz)과 [두 라우터의 에이전트 작업 32건](multi-router-raw.tar.gz)은 각각 별도 디렉터리에 푼다. 저장소 루트에서 `PYTHONPATH=benchmarks/live python3 benchmarks/evidence/20261124-agentic-serving/verify_cross_router.py <첫 디렉터리>/qualified`와 같은 경로의 `verify_multi_router.py <둘째 디렉터리>`로 검산한다. 첫 디렉터리에는 최초 이미지 기동 실패와 네이티브 로더의 샤드 상한 오류도 보관했다. 두 정책의 비교는 같은 네이티브 바이너리에서 이루어졌다. [단일 배치](cross-router-verified.json) · [에이전트 혼합 부하](multi-router-verified.json)

[HTTP 워커 9개 시험 원본](headroom-raw.tar.gz)은 위의 `verify_cross_router.py <디렉터리>/trial`로 검산한다. 빌드에서 보존한 해시와 실행한 두 바이너리를 대조할 수 있다. 고정 모델·부하에서 배치와 보호 스트림의 여섯 비교를 모두 통과했다. [판정](headroom-verified.json)

[HTTP 워커 9개의 두 라우터 혼합 부하](headroom-mixed-raw.tar.gz)는 `verify_multi_router.py <디렉터리>`로 같은 사전 기준을 검산한다. 32개 작업·96개 HTTP 호출의 정답, 두 입력군별 시작·완료 지연과 GPU 시간당 목표 달성 건수가 포함된다. [재집계](headroom-mixed-verified.json)

[HTTP 부하 조회와 외부 도착 원본](readiness-burst-raw.tar.gz)은 별도 디렉터리에 풀고 `PYTHONPATH=benchmarks/live python3 benchmarks/evidence/20261124-agentic-serving/verify_readiness_burst.py <디렉터리>/first`로 검산한다. 원본에는 모든 GET 결과와 시각, 960개 외부 요청, 진행 중이던 128개 스트림, 16개 보호 에이전트의 답변이 들어 있다. [등록](readiness-burst-registration.json) · [재집계](readiness-burst-verified.json)

[짧은 보류를 적용한 독립 재시험](protected-burst-raw.tar.gz)은 `verify_readiness_burst.py <디렉터리>/repeat --protect-reader`로 재계산한다. `inclusive_ttft_ms`는 라우터에 들어간 시각부터 첫 콘텐츠까지이며 `ttft_ms`는 HTTP 전송 이후 구간이다. `dispatch_after_protected_last_ms`의 양수는 상대가 마지막 콘텐츠를 받은 다음 새 요청을 전송했다는 뜻이다. [판정](protected-burst-verified.json)

[두 번 기동해 실제 도구 반환을 재생한 원본](natural-burst-raw.tar.gz)은 `PYTHONPATH=benchmarks/live python3 benchmarks/evidence/20261124-agentic-serving/verify_natural_burst.py <디렉터리>`로 검산한다. 도구 호출·도구 대기 중 미리 계산한 prefix·최종 입력을 복원하고, 자연 종료한 본문 해시·출처·세 수치·보호 에이전트·자원 반환을 확인한다. [시드별 결과](natural-burst-verified.json)

[초기 429 비교 원본](early-rejection-raw.tar.gz)을 풀고 `python3 benchmarks/evidence/20261124-agentic-serving/verify_early_rejection.py <디렉터리>/test`로 1,280개 제출과 정상 스트림 종료, 상태가 확인된 응답, 클라이언트 해시, GPU 반환을 검산한다. [재집계](early-rejection-verified.json)

[두 라우터의 세 정책 혼합 부하](defer-mixed-raw.tar.gz)를 풀고 저장소 루트에서 `PYTHONPATH=benchmarks/live python3 benchmarks/evidence/20261124-agentic-serving/verify_defer_mixed.py <원본 디렉터리>`로 정답·GPU 시간·짝 비교를 검산한다. [결과](defer-mixed-verified.json)

[캐시 순서 갱신의 첫 후보](prefix-recency-first-raw.tar.gz)는 동일 경로의 중복 방문만 생략했다. 원본에 시험 당시의 패치·빌드 조건·CPU 표본·GPU별 시작과 종료 기록이 들어 있다. [검산](prefix-recency-first-verified.json) · [초기 등록](prefix-recency-registration.json) · [교대 부하 후속 등록](prefix-recency-range-registration.json)

[최종 캐시 갱신 CPU·GPU 원본](prefix-recency-raw.tar.gz)을 푼 뒤 저장소 루트에서 `PYTHONPATH=benchmarks/live python3 benchmarks/evidence/20261124-agentic-serving/verify_prefix_recency.py <원본 디렉터리>`로 검산한다. 기존·수정·수정·기존 순 CPU 표본과 두 A100의 교차 기동, 52개 스트리밍 응답, 동일 입력·출력과 캐시 퇴출 대조를 포함한다. [검산값](prefix-recency-verified.json) · [스케줄러 테스트](prefix-recency-local-checks.json)

[첫 혼합 부하 출처 시험](citation-mixed-raw.tar.gz)은 정책당 64개 작업의 실제 모델 요청·출력과 종료 시 자원 상태를 담는다. 원본을 풀고 저장소 루트에서 `PYTHONPATH=benchmarks/live python3 benchmarks/evidence/20261124-agentic-serving/verify_citation_mixed.py <원본 디렉터리>`로 기본 정답·실제 첫 문장·측정값 뒤 출처·공통 정답의 지연을 검산한다. [첫 결과](citation-mixed-verified.json)

[문장과 인용 문단을 분리한 후속 원본](citation-layout-raw.tar.gz)은 같은 검산 도구로 재집계한다. 같은 입력 64개에서 기존과 후보를 바꿔 기동한 결과, 전체 384회 HTTP 호출, 실제 문장 형식·90단어·출처 위치와 시간당 목표 충족 건수를 확인할 수 있다. [후속 집계](citation-layout-verified.json)

[Qwen 3B·Coder 7B decode 교차 원본](model-decode-raw.tar.gz)에는 두 모델 각각의 Riley·vLLM 교차 기동, BF16 Transformers 독립 생성·첫 분기 점수, Riley page 모드 대조와 GPU 반환 기록이 있다. 원본을 푼 뒤 저장소 루트에서 `PYTHONPATH=benchmarks/live python3 benchmarks/evidence/20261124-agentic-serving/verify_model_decode.py <디렉터리>`를 실행한다. 같은 방식으로 `verify_model_decode_reference.py`와 `verify_model_decode_page.py`를 실행하면 독립 생성과 page 대조를 검산한다. [교차 집계](model-decode-verified.json) · [독립 생성 집계](model-decode-reference-verified.json) · [page 집계](model-decode-page-verified.json) · [등록 설명 정정](model-decode-page-correction.json)

### Coder-7B 최종 호출 prefill과 운영 GPU 시간 계수

두 A100의 page·fixed-tile 교차와 page 직접 집계 실험은 [같은 원본](coder-prefill-trace-raw.tar.gz)을 사용한다. [시간 분해](coder-prefill-trace-verified.json), [직접 집계 후보](coder-prefill-direct-verified.json), [도구 결과의 토큰 경계](coder-tool-boundary-verified.json)에 독립 검산 값을 보존했다. [운영 계수 원본](iteration-phase-raw.tar.gz)은 제품 변경과 진단 기록을 함께 빌드해 새로 실행한 별도 서버이며 [검산 결과](iteration-phase-verified.json)를 담았다. 저장소 루트에서 아래와 같이 재집계한다.

```sh
e=benchmarks/evidence/20261124-agentic-serving
trace_dir=$(mktemp -d)
phase_dir=$(mktemp -d)
tar -xzf "$e/coder-prefill-trace-raw.tar.gz" -C "$trace_dir"
tar -xzf "$e/iteration-phase-raw.tar.gz" -C "$phase_dir"
PYTHONPATH="$trace_dir:benchmarks/live" python3 -B "$e/verify_coder_prefill_trace.py" "$trace_dir"
PYTHONPATH="$trace_dir:benchmarks/live" python3 -B "$e/verify_coder_prefill_direct.py" "$trace_dir"
PYTHONPATH="$phase_dir:benchmarks/live" python3 -B "$e/verify_iteration_phase.py" "$phase_dir"
```

### Coder-7B의 도구 대기 중 이력과 확정된 호출 준비

[6개 기동 원본](coder-wait-raw.tar.gz)에는 실제 요청·캐시 적중·GPU 스텝·최종 스트림과 독립 체크포인트 토큰 경계 검사가 있다. [교차 검산](coder-wait-verified.json)의 `candidate_gates`에서 같은 GPU의 준비 끔 대비 차이를 확인할 수 있다. [정책 클라이언트 시험 등록](coder-wait-client-registration.json)은 보류 시드·200개 이력·도구 대기 예측을 사용하는 별도 조건이다.

```sh
e=benchmarks/evidence/20261124-agentic-serving
trace_dir=$(mktemp -d)
tar -xzf "$e/coder-wait-raw.tar.gz" -C "$trace_dir"
PYTHONPATH="$trace_dir:benchmarks/live" python3 -B "$e/verify_coder_wait.py" "$trace_dir"
```

실제 클라이언트 명령을 사용한 별도 두 기동의 [원본](coder-wait-client-raw.tar.gz)과 [판정](coder-wait-client-verified.json)은 아래처럼 재집계한다.

```sh
e=benchmarks/evidence/20261124-agentic-serving
client_dir=$(mktemp -d)
tar -xzf "$e/coder-wait-client-raw.tar.gz" -C "$client_dir"
PYTHONPATH="$client_dir:benchmarks/live" python3 -B "$e/verify_coder_wait_client.py" "$client_dir"
```

### Coder-7B에서 여러 에이전트가 동시에 도착한 부하

[준비 요청 부하 시험](coder-pressure-registration.json)은 각 GPU에 4·8·16·16개의 워크플로를 간격 30ms로 도착시킨다. 서버 로그의 실제 배치 참여자 수와 `/metrics` 차이로 GPU 슬롯 포화를 확인한다. [준비 예약 반환 시험](coder-release-registration.json)은 일반 서버 바이너리와 이전·수정 클라이언트를 사용한다. [워크플로 입장 시험](coder-window-registration.json)은 이력을 준비하지 않은 두 정책에서 긴 작업의 동시 입장 수만 바꾼다. 세 시험은 서로 다른 시드·원본을 사용한다. [두 복제본 시험](coder-scale-registration.json)은 준비를 끈 단일 복제본과 같은 시드·도착률을 사용한다. [공개 옵션의 보류 입력 재시험](coder-release-option-registration.json)은 기본 동작과 빠른 예약 반환을 동일 소스로 비교한다.

저장소 루트에서 다음 명령으로 원본의 요청과 최종 답변, GPU당 목표 충족 처리량을 다시 검산한다.

```sh
e=benchmarks/evidence/20261124-agentic-serving
for trial in pressure release window scale release-option; do
    data=$(mktemp -d)
    tar -xzf "$e/coder-$trial-raw.tar.gz" -C "$data"
    verifier=${trial%-option}
    PYTHONPATH="$data:benchmarks/live" python3 -B "$e/verify_coder_$verifier.py" "$data"
done
PYTHONPATH=benchmarks/live python3 -B "$e/test_wait_reservation.py"
```

[이력 길이별 복제본 배정 시험](coder-placement-registration.json)은 두 복제본을 새로 기동한 두 조건을 대조한다. 규칙적인 도착, 순서를 섞은 도착, 이력 길이가 실제 도구 결과 크기와 엇갈리는 도착을 사용한다. 정상 종료·목표 충족 처리량·기존 독자의 지연을 다음과 같이 다시 계산한다.

```sh
e=benchmarks/evidence/20261124-agentic-serving
data=$(mktemp -d)
tar -xzf "$e/coder-placement-raw.tar.gz" -C "$data"
PYTHONPATH="$data/rr:benchmarks/live" python3 -B "$e/verify_coder_placement.py" "$data"
```

[HTTP 워커 교차 시험 원본](http-worker-queue-raw.tar.gz)을 빈 디렉터리에 풀고 저장소 루트에서 `PYTHONPATH=<압축을 푼 경로>:benchmarks/live python3 -B benchmarks/evidence/20261124-agentic-serving/verify_http_worker_queue.py <압축을 푼 경로> <출력.json>`을 실행한다. 서버 종료 계수·구간 증분을 모든 연결의 로그와 맞춘 뒤 세 번의 모델 호출과 도구 입력을 원본 시드로 복원하고 지연·기한 충족 여부를 계산한다. 동일 CUDA 바이너리의 SHA-256은 `5262f4450c62de25c68d979978627ca6715b8cd4d642cb0abcbe6dacbc6a291c`다. 원본에는 빌드 로그, 기존 phase trace 소스에 적용한 제품 변경 패치, 추가 진단 스크립트와 기동 인자가 포함된다. [검산 결과](http-worker-queue-verified.json)

[GPU 슬롯 교차 시험 원본](engine-slots-raw.tar.gz)에는 같은 바이너리에서 HTTP 워커 16개로 고정해 새 시드에 실행한 네 기동이 있다. 위와 같은 검산 명령을 해당 압축 해제 경로에 적용하면 슬롯 8·16의 커밋된 GPU 배치 참여 수, 턴별 입력과 실제 계산 행, 긴 최종 답변 완료와 독자 스트리밍 간격을 재계산한다. [GPU 슬롯 판정](engine-slots-verified.json) · [로컬 Rust 테스트·Clippy·소스 계약 결과](http-worker-cpu-tests.tar.gz)

같은 작업의 이전 완료 프롬프트에서 보존할 수 있었던 KV 상한은 `PYTHONPATH=benchmarks/live python3 -B benchmarks/evidence/20261124-agentic-serving/verify_coder_session_prefix.py <시험 압축 해제 경로> <Coder 체크포인트 디렉터리> target/release/examples/qwen_prompt_profile <출력.json>`으로 검산한다. 앞서 설명한 Rust 예제 빌드를 사용한다. 스크립트가 원래 토크나이저 두 파일의 해시를 확인하고 각 기동에서 실제로 전송한 요청을 별도로 복원한다. [워커 시험](http-worker-session-prefix.json) · [슬롯 시험](engine-slots-session-prefix.json)

[Coder-7B 카탈로그 부하 원본](coder-catalog-raw.tar.gz)을 빈 디렉터리에 푼 뒤 저장소 루트에서 `PYTHONPATH=<해제 경로>:benchmarks/live python3 -B benchmarks/evidence/20261124-agentic-serving/verify_coder_catalog.py <해제 경로> <출력.json>`을 실행한다. 시드로 원본 64개와 두 도구 호출을 복원하고 별도 ID 필터로 최종 입력 해시를 확인한다. 작업별 정답·출처와 GPU 계산량·기한 충족 처리량을 검산한다. [등록](coder-catalog-registration.json) · [검산 결과](coder-catalog-verified.json)

[Coder 카탈로그의 첫 엔진 비교 시도](coder-catalog-comparison-attempt.tar.gz)를 별도 디렉터리에 푼 뒤 `PYTHONPATH=<해제 경로>:benchmarks/live python3 -B benchmarks/evidence/20261124-agentic-serving/verify_catalog_engines.py <해제 경로> <출력.json> --initial-attempt`로 일반 Riley 결과와 vLLM 기동 실패를 검증한다. [부분 검산](coder-catalog-comparison-attempt.json). 두 엔진의 비교 판정은 모두 끝난 원본에서만 같은 프로그램의 기본 모드로 수행한다.

[카탈로그 진단 원본](coder-catalog-raw.tar.gz)의 선택 경로에 대해 `PYTHONPATH=benchmarks/live python3 -B benchmarks/evidence/20261124-agentic-serving/verify_coder_named_steps.py <해제 경로> <Coder 토크나이저 디렉터리> target/release/examples/qwen_prompt_profile <출력.json>`를 실행하면 앱이 실제 도구를 실행해 구성한 최종 요청을 원본과 토큰 단위로 비교한다. 새 모델 응답 생성과 지연 측정은 이 오프라인 검사 범위 밖이다. [검산](coder-named-steps-verified.json)

### 앱이 확정한 도구 뒤의 실제 Coder 최종 답변

[기본 지시 원본](named-cpu-raw.tar.gz)과 [인용 지시 원본](named-cpu-citations-raw.tar.gz)은 각각 등록한 두 CPU 세션의 소스, 체크포인트 파일 해시 목록, 모든 생성 토큰 ID, 실제 HTTP 응답을 받은 클라이언트의 기록을 포함한다. 실행은 두 도구 호출의 인자와 결과, Riley 네이티브 템플릿으로 렌더링한 최종 입력, 체크포인트가 생성한 최종 답변을 기록한다. 아래 검산에는 Python의 `tokenizers` 패키지, 보존된 체크포인트의 `tokenizer.json`과 `tokenizer_config.json`, 같은 프로젝트에서 빌드한 Rust 예제가 필요하다. 검산기는 SSE로 조립한 전체 답변을 토큰 ID에서 독립적으로 복원하고 종료 토큰·보고서별 수치·출처·문단 형식을 대조한다. CPU 시간은 GPU 속도 판정에서 제외한다.

```sh
E=benchmarks/evidence/20261124-agentic-serving
raw_dir=$(mktemp -d)
tar -xzf "$E/named-cpu-raw.tar.gz" -C "$raw_dir"
PYTHONPATH="$raw_dir:benchmarks/live" python3 -B "$E/verify_named_cpu_live.py" \
  "$raw_dir" target/release/examples/qwen_prompt_profile "$CODER_TOKENIZER_DIR" "$raw_dir/checked.json"
cmp "$raw_dir/checked.json" "$E/named-cpu-verified.json"
```

[문단 분리 지시 원본](named-cpu-layout-raw.tar.gz)도 같은 절차로 재검산했다. 원본과 대응하는 검산 JSON을 바꿔 반복할 수 있다. [문단 분리 지시 검산](named-cpu-layout-verified.json) [기본 지시 검산](named-cpu-verified.json) · [인용 지시 검산](named-cpu-citations-verified.json)

`PYTHONPATH=benchmarks/live python3 -B "$E/test_coder_named_steps.py"`와 `PYTHONPATH=benchmarks/live python3 -B "$E/test_named_fleet.py"`는 모델 선택이 필요한 요청의 폴백, 확정된 도구의 직접 실행, 실제 CLI의 HTTP 요청 수, 반복 부하의 고유 시드, 예정 도착 시각에 따른 SLO 판정을 시험한다. GPU 혼합 부하에는 [별도 사전 등록](named-burst-registration.json)과 [기동 전 사용 상태](named-burst-preflight.json)가 있다.

### Qwen 토크나이저의 반복 조각 재사용

[원본](piece-cache-raw.tar.gz)에는 Linux·Darwin 두 기동 순서군의 개별 시간, 실제 요청을 복원할 클라이언트 기록, 사용한 Rust 소스 본문, 워크스페이스 소스 스냅샷, 비교할 토크나이저 구현과 해시, 빌드 스크립트와 테스트 로그가 있다. 커널이나 GPU 모델 가중치를 올리지 않고 Rust의 실제 채팅 정규화·토큰화 경로를 실행했다. Linux 측정은 한 코어에 고정했다. [입력별 검산](piece-cache-verified.json)

재생하려면 원본에서 워크스페이스 `src.tar.gz`를 풀고 `baseline.rs`와 `candidate.rs`를 각각 `crates/riley-model/src/tokenizer.rs`에 복사한 뒤 `cargo build --locked --release -p riley-server --features server --example qwen_prompt_profile`로 두 실행 파일을 보존한다. 아래는 저장소 루트에서의 후속 CPU 시험과 별도 A100 제출 원본 대조다. `RAW`와 `CATALOG`는 각각 이번 원본과 `coder-catalog-raw.tar.gz`를 해제한 디렉터리다. `BASE`와 `CAND`는 두 실행 파일의 절대 경로다.

```sh
E=benchmarks/evidence/20261124-agentic-serving
PYTHONPATH=benchmarks/live python3 -B "$E/coder_piece_cost.py" "$BASE" "$CAND" \
  "$CODER_TOKENIZER_DIR" "$RAW/client.json" "$RAW/source.txt" "$RAW/retest.json" --followup
PYTHONPATH=benchmarks/live python3 -B "$E/verify_piece_catalog.py" "$CATALOG" "$BASE" "$CAND" \
  "$CODER_TOKENIZER_DIR" "$RAW/catalog-retest.json"
cmp "$RAW/catalog.json" "$RAW/catalog-retest.json"
```

### 첫 번째 도구 대기부터 준비하는 최종 prefix

[공통 토큰과 캐시 수용량](first-tool-capacity-verified.json)은 기존 A100 카탈로그의 앱 도구 실행을 재구성해 두 시점의 프롬프트와 실제 최종 입력을 토큰 단위로 비교한 결과다. [스케줄러 재생](first-tool-schedules-verified.json)은 네 부하의 정상 순서·역순 완료와 C16의 서로 다른 도구 반환 시점에서 실제 스케줄러가 배정한 행을 기록한다. `CATALOG`는 기존 [카탈로그 원본](coder-catalog-raw.tar.gz)을 해제한 디렉터리다. `CONFIG`는 같은 Coder-7B 체크포인트의 config.json이다.

```sh
E=benchmarks/evidence/20261124-agentic-serving
cargo build --locked --release -p riley-server --features server --example qwen_prompt_profile
cargo build --locked --release -p riley-scheduler --example prefix_work_replay
PYTHONPATH=benchmarks/live python3 -B "$E/first_tool_capacity.py" "$CATALOG" \
  target/release/examples/qwen_prompt_profile "$CODER_TOKENIZER_DIR" "$CONFIG" "$OUT/capacity.json"
PYTHONPATH=benchmarks/live python3 -B "$E/first_tool_schedules.py" "$CATALOG" \
  target/release/examples/qwen_prompt_profile "$CODER_TOKENIZER_DIR" target/release/examples/prefix_work_replay \
  "$OUT/capacity.json" "$OUT/schedules.json" --overlap
```

실제 CPU KV 재사용은 [1초 대기 실행](named-first-kv-raw.tar.gz)과 [5초 대기 실행](named-first-kv-deadline-raw.tar.gz)에 남겼다. 각각 해제한 디렉터리를 `RUN`으로, 같은 체크포인트의 [준비 없는 실행](named-cpu-raw.tar.gz)을 해제한 디렉터리를 `CONTROL`로 지정한다. [최초 실패 로그](named-first-kv-initial-failure.tar.gz)에는 참조 라이브러리가 양수 길이의 캐시 자르기를 거절한 원인이 남아 있다. 최종 검산은 수정한 실행을 사용한다.

```sh
PYTHONPATH="$RUN:benchmarks/live" python3 -B "$E/verify_first_tool_kv.py" "$RUN" "$CONTROL" \
  target/release/examples/qwen_prompt_profile "$CODER_TOKENIZER_DIR" "$OUT/checked.json"
PYTHONPATH=benchmarks/live python3 -B "$E/test_first_tool_cancel.py"
cargo test --locked -p riley-server --features server --lib disconnected_chat_preview_is_cancelled_before_first_token_and_final_still_serves
```

실제 채팅 엔드포인트에서는 `python3 benchmarks/live/tool_wait_trial.py HOST:PORT on trial.json --first-fixed-tool --include-completed-call --cancel-on-final`로 전체 prefix 준비와 취소를 재생할 수 있다. `off`로 같은 직접 도구 경로의 대조군을 실행한다. 이 명령은 긴 도구 반환값을 포함하는 별도 재생 세트를 사용한다. 위 C16 표는 등록된 카탈로그 입력과 모델 없는 스케줄러 재생의 결과다.

### HTTP 준비 요청의 토큰 상한과 복제본 예약

[긴 이력의 실제 CPU 원본](bounded-http-raw.tar.gz), [빈 이력 원본](bounded-http-short-raw.tar.gz), [호스트 분리 원본](bounded-http-cross-host-raw.tar.gz)에는 실제 준비 및 최종 모델 요청, 두 도구 호출과 전체 답변, 네이티브 프로파일러·모델 해시가 들어 있다. 호스트 분리 원본의 `external-client-preview.json`과 `external-preview.json`은 각각 Darwin 앱과 Linux 워커의 관측이다. `http-preview.json`은 두 원본을 결합한 파일이다. 검산은 체크포인트 토크나이저로 최종 출력 ID를 별도 디코딩하고, 준비·최종 입력·자연 종료·최초 최종 forward의 입력 길이를 대조한다. `RUN`, `CONTROL`, `CATALOG`는 각각 시험 원본, [준비 없는 Coder CPU 원본](named-cpu-raw.tar.gz), [A100 카탈로그 원본](coder-catalog-raw.tar.gz)을 해제한 디렉터리다.

```sh
E=benchmarks/evidence/20261124-agentic-serving
cargo build --locked --release -p riley-server --features server --example qwen_prompt_profile
PROBE=target/release/examples/qwen_prompt_profile
PYTHONPATH=benchmarks/live python3 -B "$E/verify_bounded_http.py" "$RUN" "$CONTROL" "$CATALOG" \
  "$PROBE" "$CODER_TOKENIZER_DIR" "$OUT/checked.json"
PYTHONPATH=benchmarks/live python3 -B "$E/bounded_fleet_trial.py" \
  "$PROBE" "$CODER_TOKENIZER_DIR" "$OUT/new-fleet-run"
PYTHONPATH=benchmarks/live python3 -B "$E/bounded_planner_edges.py" \
  "$PROBE" "$CODER_TOKENIZER_DIR" "$LLAMA3_CHECKPOINT_DIR" "$OUT/edges.json"
```

[최종 토큰·카탈로그 선택 검산](bounded-http-verified.json) · [짧은 요청 검산](bounded-http-short-verified.json) · [원격 모델 워커 검산](bounded-http-cross-host-verified.json). [두 인증 리스너의 HTTP 관측](bounded-fleet-raw.tar.gz)과 [요약](bounded-fleet-verified.json)은 CLI의 복제본 고정·예약 총량·인증 경로를 검증한다. 추가 실패 주입에서는 한 리스너가 길이를 잘못 보고하고 다른 리스너가 동일 길이의 다른 ID를 보고했다. 다음 워크플로의 준비만 생략되고 양쪽 최종 요청이 계속 동작했다. [A100 사용권 기록](bounded-http-lease-window.jsonl)에는 6분간 각 장치 24회 조회와 종료 상태가 있다. 이 시간 동안 C16 교차 시험을 시작할 수 있는 장치는 없었다.

## 확정한 답변 앞부분과 모델이 생성한 설명을 따로 검산하기

`answer-prefix-raw.tar.gz`의 `plain/`과 `guided/`에는 두 도구 실행, 전송한 최종 요청의 해시, 모델이 생성한 토큰 ID, SSE로 받은 접미부, 조립된 전체 답변이 있다. `build-evidence.tar.gz`에는 Linux에서 사용한 Rust 렌더러 소스·빌드 기록과 모델 파일 해시 검사가 있다. 기본 생성 기준선은 `named-cpu-raw.tar.gz`에 있다. 각 후보는 같은 CPU FP32 Coder-7B, 0.5초씩 두 번의 도구 대기, 긴 이력·빈 이력 두 건이다. 앱이 제공한 문자열은 입력 사용량에 포함하며, 모델이 생성한 부분만 출력 길이와 스트리밍 시간에 반영한다.

아래에서 `RAW`는 후보 원본을 해제한 경로이고 `TOKENIZER`는 해당 Coder 체크포인트의 토크나이저 디렉터리다. 검산기는 최종 채팅을 원본 도구 호출로 재구성하고, Riley로 다시 인코딩하고, 생성 토큰을 독립적으로 디코딩해 SSE에서 받은 내용과 대조한다.

```sh
cargo build --locked -p riley-server --features server --example qwen_prompt_profile
E=benchmarks/evidence/20261124-agentic-serving
for arm in plain guided; do
  PYTHONPATH="benchmarks/live:$E" python3 "$E/verify_named_cpu_live.py" \
    "$RAW/$arm" "$PWD/target/debug/examples/qwen_prompt_profile" "$TOKENIZER" "$RAW/$arm-verified.json"
done
PYTHONPATH=benchmarks/live python3 -m unittest discover -s "$E" -p test_report_answer_prefix.py
```

## 최종 답변의 채팅 스트림 직렬화 비용과 TCP 전달 시간

`chat-stream-raw.tar.gz`의 `src.tar.gz`는 Linux에서 빌드한 소스다. `cases.json`의 첫 네 항목은 Coder-7B A100의 고정 길이 응답으로부터 복원한 출력 조각이다. 4개 원문 해시와 일반 HTTP 요청 12건의 가시 토큰 간격은 `meta.json`에 있다. 나머지 세 항목은 도구 호출 구분자·유니코드 경계 검사다. 다음 명령은 모델 가중치 없이 직렬화의 두 경로를 비교한다.

```sh
cargo build --locked --release -p riley-server --features server --example chat_stream_replay
target/release/examples/chat_stream_replay 300 < "$RAW/cases.json" > "$RAW/local-repeat.json"
```

TCP 재생은 시험용 소스 복사본에서 `chat_stream_http_overlay.py`와 `chat_stream_http_trial.rs`를 적용한다. 같은 release 바이너리에서 일반 HTTP 서버와 기존 테스트 백엔드를 기동하고, 클라이언트가 실제로 읽은 64개 텍스트 조각, 사용량, 종료 이벤트를 확인한다. 원본의 `http.sh`에는 코어 범위와 기존→직접→직접→기존 실행 순서가 들어 있다. 요청별 시간의 재집계는 다음 명령으로 재현할 수 있다.

```sh
python3 benchmarks/evidence/20261124-agentic-serving/verify_chat_stream.py "$RAW" "$RAW/recheck.json"
```
