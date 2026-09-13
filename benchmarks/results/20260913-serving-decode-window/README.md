# Serving decode window: opt-in integration, GPU/HTTP validation pending

`--decode-window paired-experimental-v1`으로 V7 serving의 두-step 경로를 선택하도록 구현했다. 기본값 `single`은 기존 단일 iteration 경로다. paired 옵션은 독립 `variable-smol-v7`, GPU greedy, loopback diagnostic 조건을 요구한다. FFN/FlashInfer 실험 profile과 혼합하지 않는다.

## Connected behavior

- 실제 scheduler가 두 decode plan을 제공하면 pair authority로 두 GPU graph를 제출·검증한다. 선택된 요청이 GPU greedy 부적격이면 미제출 reservation을 abort하고 일반 계획으로 진행한다. prefill·admission·남은 출력 1개 등은 scheduler의 기존 fallback을 사용한다.
- 두 단계의 token staging을 재사용하고 `(request ID, generated index)`로 audit/SSE publication을 대조한다. 한 요청에서 두 token이 같은 값이어도 순번으로 구분한다. 일반 경로에 추가 event Vec allocation을 넣지 않는다. 추가 token staging capacity는 paired 옵션에서만 확보한다.
- 첫 token을 기존 GenerationState에 수용하여 EOS/stop-token/stop-string을 판정한다. 종료된 요청의 후행 GPU token은 runtime 검증을 거치되 GenerationState에는 수용하지 않고 scheduler settlement에서 폐기한다. 따라서 후행 token이 tokenizer/text/usage 상태를 진행시키지 않는다.
- 두 결과 settlement 및 runtime commit 확인 후에만 기존 audit·SSE publication helper를 호출한다. audit capacity를 요청당 이번에 공개할 token 수만큼 검사한다. runtime configuration의 completion mode는 paired event drain으로 식별한다.

## Verified scope

- 실제 checkout의 non-CUDA server 검사: library 69 passed / 1 ignored, CLI 32 passed. 새 CLI 검사는 V7/다른 graph profile, CPU/GPU greedy, loopback/public bind 조합을 검증한다.
- runtime config의 normalization 및 다른 graph profile로 변경할 때 paired flag 초기화 검사: 1 passed.
- CUDA feature Rust typecheck (library + binary + tests): passed. **이 검사는 CUDA 빌드·링크·GPU 실행 증거가 아니다.** 원격 SSH가 응답하지 않아 `/tmp/riley-serving-window-typecheck`의 별도 복사본에서만 CUDA build script의 native 빌드를 생략하고 `cargo check --features cuda,server --tests`를 실행했다. 실제 checkout의 build script는 변경하지 않았다. 동일 소스와 schema fixture를 복사했으며 GPU 검사를 통과한 것으로 처리하지 않는다.
- 선행 [native model window](../20260913-native-decode-window/README.md)는 실제 4090 generation exact 및 memcheck 0 errors를 확인했지만, 이번 HTTP integration의 증거를 대신하지 않는다.

## Pending validation and measurement

원격 racecheck handle `42095`의 완료 상태와 SSH 연결을 먼저 확인한다. 그 전에는 GPU 검사를 중복 실행하지 않는다. 원격에는 이번 server 소스를 아직 반영하지 않았다.

복구 후 실제 native CUDA build를 수행하고 paired HTTP smoke와 token-ID reference를 검증한다. 직렬/paired의 ragged prompt, 반복 decode, page 경계, EOS/stop-string, cancellation, terminal reclaim, GPU-greedy 부적격 fallback, audit 순번·SSE 순번을 확인한다. 오류 경로에서는 graph close/drain 전에 KV를 재사용하지 않는지 확인한다.

이 통합이 통과한 뒤 frozen workload의 C16/C32 natural serving을 직전 Riley·paired Riley·vLLM의 반복 교차 순서로 측정한다. binary/checkpoint/workload hash, correctness, 오류율, throughput, TTFT/TPOT 및 P95/P99를 비교표로 기록한다. 두-token drain 뒤 공개하는 방식의 전달 간격과 tail 악화를 포함해 판단한다. 현재 새 serving 성능은 **미측정**이며 기본 경로로 승격하지 않는다. 전체 목표는 미완료다.
