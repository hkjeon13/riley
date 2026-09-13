# Serving decode window: opt-in integration and recovered GPU/HTTP validation

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

## Recovered native build and HTTP validation

원격 재부팅으로 `/tmp`의 소스·toolkit·실행 결과가 유실되어 `/data/riley-serving-260913-recovery`에 빌드 환경을 복구했다. CUDA 13.0.88, cuBLAS 13.0.2.14와 현재 드라이버로 실제 `cargo build --release -p riley-server --features cuda,server`가 통과했다. 기존 typecheck의 한계를 대신하는 native 빌드·링크 증거다.

`http-smoke-v3`에서 single/paired 각각 C32 warmup 96개와 retained 96개의 prompt ID·출력 token ID·text hash·finish를 기존 natural reference와 대조했다. 3종 prompt마다 32개씩 stop-string 요청을 보내 두 경로의 token·text·finish·usage가 일치함을 확인했다. 각 경로에서 32개 연결을 출력 완료 전에 닫고, 이후 retained 요청이 reference와 일치함을 확인했다. Paired 종료 로그의 완료 window는 251회, 최대 폭은 32다. 이 검사는 실제 모델의 자연 EOS 발생이나 모든 장애 경로를 포괄하지 않는다.

실제 serving 성능은 새 환경의 single·paired·vLLM을 같은 client와 workload로 재측정한다. 원본 자료와 비교표는 [복구 후 serving 보고서](../20260913-paired-serving-recovery/README.md)에 기록한다. 기본값은 single이며, 전체 성능 목표는 미완료다.
