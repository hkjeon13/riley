# Riley / vLLM competitive contract

이 디렉터리는 C01의 **계획 계약**이다. 여기의 lane template, GPU 요구사항, 모델
slot은 실행 가능성이나 production 지원을 주장하지 않는다. 현재 Riley lane은
contract-only이고 vllm-current lane은 campaign-pin-required이다. 따라서 이
디렉터리만으로는 원격 GPU에서 benchmark를 실행하거나 Riley의 M4/M5 통과를 주장할
수 없다.

계약은 vLLM의 historical lane과도 분리되어 있다. vllm-current는 campaign 시작
전에 version/tag, wheel 또는 source hash, dependency lock, runtime option dump를
고정한 경쟁 분모만을 뜻한다. 그 pin이 바뀌면 새 campaign ID를 만들어야 한다.

## 구성

| 파일 | 역할 |
| --- | --- |
| contract-v1.json | 동일 campaign identity, AB/BA 순서, M4/M5 기준, admission 규칙 |
| contract-v1.schema.json | contract, matrix, lane의 Draft 2020-12 closed schema |
| matrices/diagnostic-sm89-bf16-v1.json | 기존 SmolLM2-135M hash를 쓴 Tier D 진단 cell; 승리 판정 제외 |
| matrices/latency-sm89-bf16-v1.json | Tier C engine-only / HTTP streaming latency template |
| matrices/serving-sm89-bf16-v1.json | Tier S open-loop SLO, cancellation, disconnect, backpressure template |
| lanes/riley.json | Riley candidate command template |
| lanes/vllm-current.json | campaign에 pin해야 하는 vLLM baseline command template |

Tier C/S matrix에는 의도적으로 구체적 model revision, tokenizer/weight hash, SLO
숫자가 없다. 각 model slot은 campaign manifest에서 dense Llama/Qwen-compatible
BF16 model로 먼저 pin되어야 한다. hash가 없는 placeholder를 실제 hash처럼
사용하면 runner는 preflight에서 거부해야 한다. SLO threshold도 model identity와
함께 execution 전 manifest에 immutable하게 기록해야 하며, 결과를 본 뒤 추가하거나
바꿀 수 없다.

Tier D의 SmolLM2-135M revision, weights SHA-256, prompt artifact SHA-256만
저장소의 기존 diagnostic asset에서 검증된 값을 사용한다. Tier D는 병목 분석과
dry-run용이며 M4/M5의 증거가 아니다.

## Closed manifest rules

contract-v1.schema.json의 root는 다음 세 manifest type만 허용한다.

- riley.competitive.contract.v1
- riley.competitive.matrix.v1
- riley.competitive.lane.v1

모든 object는 closed (additionalProperties: false)다. 새 필드는 v1 manifest에
조용히 넣지 않고 schema version을 올려야 한다. template cell은 template: true와
작은 axes의 Cartesian expansion으로 표현한다. 확장한 cell ID는 template의
{model_slot}, {concurrency}, {prompt_tokens}, {requested_output_tokens},
{arrival_rate_class}, {traffic_profile_id} placeholder를 해당 axis value로 바꾼
값이다.

정본성도 admission gate다. `contract-v1.json` 및 정본 matrix/lane은 코드에 고정된
경로와 SHA-256이 정확히 일치해야 한다. concrete campaign matrix/lane은 대신
`parent_contract` (정본 contract path, hash, schema, ID)와 `parent_asset` (정본
matrix/lane path, hash, schema, ID)를 함께 담아야 한다. request manifest도
`parent_contract`를 담는다. 이 receipt가 없거나 현재 checkout의 정본 asset과
불일치하면 self-authored threshold, 쉬운 matrix, 이름만 같은 lane은 claim에 쓸 수
없다.

각 실행은 다음을 만족해야 한다.

- independent fresh process run은 최소 5개다.
- 각 비교 대상은 AB와 BA 모두 실행한다. A=Riley, B=vllm-current이다.
- warm-up은 측정 표본에서 제외하고, 실패 표본은 성공 percentile에 섞지 않는다.
- engine-only와 HTTP streaming metric은 별도 cell/series로 유지한다.
- 한 paired comparison 안에서는 request identity, model/tokenizer, GPU UUID,
  driver/CUDA, lane options가 같아야 한다.

required_environment_keys는 raw receipt에 모두 있어야 한다. 특히 remote GPU
target의 UUID, driver/CUDA, clock/power policy, clean Git/source archive, exact
binary/wheel/dependency lock과 runtime option hash를 두 lane에 대해 보존한다.

## Campaign admission

실행 전에 runner는 다음을 모두 확인한다.

1. contract, 모든 matrix, 모든 lane이 schema를 통과한다.
2. candidate/source, current vLLM, model, tokenizer, request set, SLO profile,
   runtime options가 모두 campaign ID에 pin되어 있다.
3. lane availability가 실제 실행을 허용한다. contract-only와
   campaign-pin-required는 plan 생성에는 쓸 수 있지만 execute preflight는
   fail-closed해야 한다.
4. remote GPU host가 matrix의 one-GPU sm89 / RTX 4090 / BF16 requirement와
   preflight 조건을 충족한다.

이 check가 하나라도 빠지면 checker는 passed를 낼 수 없다. 기본 outcome은
incomparable 또는 실행 전 fail-closed다.

available lane은 `engine.pin_status: pinned`, immutable version/revision,
dependency lock 및 executable/source/runtime/model/tokenizer hash receipt를 모두
가져야 한다. plan에 저장된 `readiness: ready`는 증거가 아니다. checker는 claim
시점에 lane file, preflight file과 script hash, current Git HEAD와 clean state를
다시 읽어 readiness를 재계산한다.

## Runner input/output interface

`scripts/run_campaign.py`는 remote engine을 직접 시작하지 않는 **plan-only**
도구다. 즉, 현재 구현은 실행 전 immutable input, AB/BA 순서, source/lane/matrix
hash를 `riley.competitive.execution-plan.v1` JSON으로 동결한다. remote executor는
후속 adapter가 이 plan만 소비하도록 별도 구현해야 하며, plan을 실행 결과로
오인하면 안 된다.

```bash
python3 benchmarks/competitive/scripts/run_campaign.py \
  --repo-root . \
  --contract <contract-v1.json> \
  --matrix <concrete-campaign-matrix.json> \
  --riley-lane <riley-lane.json> \
  --competitor-lane <vllm-current-lane.json> \
  --request-manifest <campaign-request-manifest.json> \
  --preflight-receipt <successful-preflight.stdout.txt> \
  --campaign-id <immutable-id> \
  --output <new-execution-plan.json>
```

기본값은 source tree가 clean이어야 하며, output은 create-only다. preflight receipt는
reviewed `benchmarks/scripts/preflight.sh`의 성공한 `key=value` stdout이어야 하며,
Git SHA, RTX 4090/sm89, driver, idle VRAM, 온도, clock/governor, staging 용량을
다시 검증해 plan hash에 묶는다. `--allow-dirty-source`는 테스트/개발용 blocked plan만
만들며 checker가 경쟁 evidence로 거부한다.
`--require-executable-lanes`는 contract-only 또는 campaign-pin-required lane으로
blocked plan을 쓰는 대신 즉시 fail-closed한다. Tier C/S template은 concrete model,
SLO, request identity가 pin된 campaign matrix로 materialize되기 전에는 실행 plan에
넣을 수 없다.

request manifest의 expected minimum shape는 아래와 같다. request의 prompt 원문은
필요하지 않으며, token IDs와 identity hash로 비교한다.

~~~json
{
  "schema_version": "riley.competitive.requests.v1",
  "manifest_id": "…",
  "parent_contract": {
    "path": "benchmarks/competitive/contract-v1.json",
    "sha256": "<canonical-contract-sha256>",
    "schema_version": "riley.competitive.contract.v1",
    "contract_id": "riley-vllm-competitive-v1"
  },
  "model_identity": {
    "model_id": "…",
    "model_revision": "…",
    "model_weights_sha256": "…",
    "tokenizer_revision": "…",
    "tokenizer_files_sha256": "…"
  },
  "request_sets": [
    {
      "cell_id": "expanded-cell-id",
      "requests": [
        {
          "request_id": "…",
          "prompt_token_ids_sha256": "…",
          "prompt_tokens": 128,
          "requested_output_tokens": 32,
          "sampling": "greedy",
          "seed": null,
          "eos_policy": "ignore-eos",
          "cache_policy": "cache-off",
          "arrival_schedule_id": "closed-loop-v1"
        }
      ]
    }
  ]
}
~~~

`model_weights_sha256`은 `weights_sha256`의 동의어가 아니라 campaign에서 하나만
사용한다. `tokenizer_aggregate_sha256`은 선택적인 추가 binding이다. Seeded sampling은
contract profile을 sampling에 넣고 seed가 null이 아니어야 한다. model/request
manifest는 Tier S에 필요한 predeclared per-cell SLO profile도 담아야 하며, raw
result가 그 policy를 공급하거나 바꾸면 안 된다.

runner는 schema_version: riley.competitive.raw.v1을 가진 append-only raw JSONL
record와 별도의 provenance, plan, preflight receipt를 쓴다. 각 raw record는
campaign, lane, expanded cell, independent run, order, request identity,
environment receipt, success/failure, latency/resource metric, token mismatch
count, terminal-event count를 식별한다.

plan의 `workloads`는 각 concrete cell의 model identity hash, warm state,
arrival mode/schedule, client behavior, cancellation rate, SLO profile과 모든
request의 prompt token hash/길이, output 길이, sampling profile, seed, EOS/cache
policy를 SHA-256으로 함께 고정한다. raw record는 그 전체 `workload_sha256`, 실행
projection (`workload`) 및 request별 동일 필드를 모두 제출해야 한다. checker는
plan 값을 신뢰하지 않고 pinned matrix/request manifest에서 workload를 다시 만든 뒤
각 raw record와 대조한다. 따라서 prompt를 짧게 하거나 sampling/seed/arrival/SLO를
바꾼 뒤 기존 campaign hash를 재사용할 수 없다.

## Checker interface

checker는 closed plan과 그 raw evidence만 소비한다.

~~~
check_campaign.py
  --repo-root <clean-repository-root>
  --plan <execution-plan.json>
  --raw <raw.jsonl-or-directory> [--raw <...>]
  --output <new-closed-report.json>
~~~

checker는 unknown field, plan drift, independent run 부족, AB/BA arm 누락,
cross-campaign 또는 cross-environment pairing, warm-up-as-measured record,
non-finite value, failure record를 포함한 success percentile을 거부해야 한다.
각 independent run이 immutable request manifest를 정확히 모두 커버할 때만 그
run 안에서 contract의 nearest-rank p95(TTFT/TPOT)를 구한다. M4/M5의 cell p95는
pooled request percentile이 아니라 이 per-run p95 summary들의 median이다. 이후
ratio를 계산하고 다음 중 하나만 쓴다.

특히 checker는 plan의 `readiness` 또는 parsed preflight value를 신뢰하지 않는다.
receipt file의 SHA-256과 closed key set을 다시 검증하고, reviewed
`benchmarks/scripts/preflight.sh` hash와 current clean Git HEAD가 campaign source와
일치하는지 확인한다. 따라서 preflight 없이 `ready`로 고친 plan, stale receipt,
unpinned lane, plan 생성 뒤 source drift는 `passed`가 될 수 없다.

~~~
passed | partial-win | failed | incomparable
~~~

M4는 모든 required_for: m4 expanded cell에서 Riley/vLLM p95 TTFT 및 TPOT ratio
<= 1.03, Riley failure/token mismatch 0, Tier S SLO goodput ratio >= 0.97을
요구한다. M5는 더 엄격한 사전 고정 limit (<= 0.90, >= 1.10, VRAM <= 1.05)와
모든 M5-required cell의 geometric mean을 쓴다. model/SLO/lane pin 누락은 waiver가
아니며 결과를 incomparable로 만든다.

## Remote evidence policy

GPU는 승인된 remote SSH target을 통해 접근할 것으로 예상하지만, 이 C01 plan
tool은 SSH 연결이나 deployment를 수행하지 않는다. 후속 executor는 clean immutable
checkout에서 strict preflight를 통과한 approved plan만 실행해야 한다. 실행 evidence는
receipt, checksum, summary만
benchmarks/competitive/results/<campaign>/에 저장한다. 대형 raw trace, model
data, credential은 승인된 외부 append-only evidence store에 둔다. secret, private
prompt, checkpoint, mutable latest baseline을 이 디렉터리에 넣으면 안 된다.
