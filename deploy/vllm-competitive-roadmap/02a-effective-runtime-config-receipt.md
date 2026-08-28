# C02-P0 — Effective Runtime Configuration Receipt

**상태:** In progress — P0 raw contract/CPU test, pinned CUDA Dockerfile build/strict Clippy, two-profile capture harness 정적 검증 및 RTX 4090의 two-profile `riley-0.1.0-rc99` container raw smoke capture와 artifact binding 검증은 통과했다. frozen-candidate semantic replay와 C02 qualification은 의도적으로 다음 C02 단계에 남긴다.<br>
**의미 등급:** `E0`<br>
**한 가지 목적:** cold prepare가 끝난 release runtime의 실제 configuration을 canonical `/v1/config` 응답과 create-only startup artifact로 고정해 C02 `startup_configuration` semantic gate가 재현 가능한 raw evidence를 받을 수 있게 한다.

[이전: C01](01-vllm-win-contract.md) | [목차](README.md) | [다음: C02](02-rc3-candidate-qualification.md)

## 1. 배경과 상태

기존 C02의 schema와 checker는 endpoint/startup receipt를 replay할 수 있었지만 server에는
`GET /v1/config`가 없었다. 현재 변경은 이 route, prepared-runtime fact snapshot, canonical
receipt, create-only startup artifact를 구현한다. 그래도 static contract, fixture, CPU test,
또는 GPU device를 쓰지 않는 build 검증은 실제 release runtime이 노출한 raw evidence가 아니다.

따라서 이 문서는 C02-P0를 C02 이전의 production corrective prerequisite로 둔다. 아직 frozen candidate, candidate-bound C02 semantic replay, 또는 actual C02 qualification decision은 존재하지 않는다. 단, RTX 4090에서 GPU used-memory `<=256MiB` preflight 뒤 `riley-0.1.0-rc99`의 two-profile container raw smoke capture를 수행했고, 두 arm의 canonical endpoint/startup-artifact binding은 raw validator로 확인했다.

### 현재 검증 경계

- 소스 구현 범위에서는 `/v1/config` exact-byte `200`, evidence 없는 `503`, 유효 framing 뒤
  non-`GET` `405`, all-or-none launch identity, canonical/create-only artifact를 unit test로
  검증했다.
- C02 static checker suite와 pinned CUDA Dockerfile의 release build, CUDA AOT, strict Clippy,
  ABI/dependency 검증은 통과했다.
- two-profile raw-capture harness는 explicit `env -i` maps, loopback-only listener,
  create-only external evidence, GPU UUID/`<=256MiB` preflight, raw-byte/artifact binding을
  정적 테스트로 검증했다. harness는 freeze/Gate E/C02 finalizer를 호출하지 않는다.
- RTX 4090에서 GPU used-memory `<=256MiB` preflight 뒤 `riley-0.1.0-rc99`를
  `stable-default`와 `max-performance-exact`로 독립 cold-prepare container launch했다.
  각 arm의 canonical endpoint/startup-artifact raw bytes와 mutual binding은 runner의 raw
  validator로 검증해 외부 evidence에 보존했다. 이 capture는 nonqualifying smoke이며
  frozen candidate, Gate E, C02 semantic replay, RC3 finalizer 또는 qualification decision은
  생성하지 않았다.

## 2. Runtime contract

C02-P0는 release launcher가 제공한 candidate identity와 configuration arm을 사용해 cold prepare가 성공한 뒤에만 하나의 immutable canonical body를 materialize한다. 이 PR은 C02 candidate를 freeze하거나 qualification을 판정하지 않는다.

### Runtime endpoint

- `GET /v1/config`는 cold prepare 뒤에만 canonical JSON bytes를 반환한다. body를 만들 수 없으면 guessed default를 반환하지 않고 `503`으로 fail closed한다.
- top-level body는 정확히 `schema_version`, `candidate_id`, `runtime_identity`, `effective_config`, `effective_config_sha256`만 포함한다.
- `runtime_identity`는 launch-time `configuration_profile`과 `configuration_sha256`만 포함한다. freeze manifest/Gate E hash는 future semantic binding이므로 raw body에 넣지 않는다.
- `effective_config`의 값은 CLI echo가 아니라 prepared runtime의 실제 attention/fallback 선택, buckets, KV geometry, GEMM policy에서 온다.
- service startup은 이 immutable body를 명시적으로 받아 worker에 공유하며, request path에서 parse하거나 reserialize하지 않는다.

`effective_config`는 다음 열 개 dimension을 빠짐없이 canonicalize한다.

```text
execution_completion_mode
batch_shape
metadata_transport
sampling_backend
attention_backend
gemm_reduction_policy
experimental_flags
fallback_policy
batch_token_budget
kv_geometry
```

### Create-only startup artifact

각 arm은 successful engine start 뒤 HTTP listener를 열기 전에 동일 canonical endpoint payload를 input으로 한 create-only startup artifact를 남긴다. endpoint response와 artifact는 서로 다른 raw path에 저장되며, artifact의 payload digest는 endpoint bytes를 정확히 bind해야 한다. artifact가 없거나 body가 canonical하지 않으면 C02-P0 contract는 fail closed한다.

## 3. Raw evidence와 semantic report

### Raw capture

`stable-default`와 `max-performance-exact` 각각에 대해 endpoint bytes와 startup-artifact bytes를 append-only 외부 evidence root에 수집한다. 이 네 raw input은 별개의 path여야 하며 raw file 자체가 gate pass/fail을 주장하지 않는다.

원격 producer는 `ci/release/run_remote_c02_runtime_config_capture.sh`다. 이 runner는
immutable binary/model digest와 두 explicit argv/environment map을 검증하고, `env -i`로
각 arm을 loopback에서 cold launch한다. GPU UUID와 `<=256MiB` used-memory preflight,
공유 evidence lock, new external evidence root, `O_EXCL` raw copy를 강제한다. GPU host의
Python 3.10에서도 raw evidence를 검증할 수 있게 stdlib-only
`validate_raw_c02_runtime_config.py`를 사용하며, 이 helper는 frozen candidate, Gate E,
또는 full C02 semantic checker를 대체하지 않는다. 실제 GPU capture가 끝나도 이 runner
자체는 candidate freeze, Gate E, C02 semantic report를 만들지 않는다.

### Semantic check

C02는 frozen candidate와 replayed Gate E report가 생긴 뒤 `check_effective_runtime_config_receipt.py`로 네 raw input을 재생한다. 이때 생성되는 `riley.effective-runtime-config-check.v1` report만이 `startup_configuration` semantic receipt이며, outer RC3 finalizer는 이를 다시 replay해 제출본과 byte-identical한지 확인한다.

`stable-default`만 promotion profile이지만, `max-performance-exact`도 독립적으로 capture/replay되어야 한다. 서로 다른 arm이 같은 resolved effective configuration으로 위장되면 fail closed한다.

## 4. C02와의 경계

C02-P0는 endpoint와 artifact를 구현·unit/integration 검증하는 production corrective PR이다. C02는 그 구현을 변경하지 않고 C02-P0가 제공한 raw mechanism으로 단일 frozen candidate의 Gate E, correctness, soak, rollback을 qualification한다.

C02-P0의 merge나 static checker test 통과는 release promotion, candidate qualification, 또는 vLLM 경쟁 승리를 뜻하지 않는다. C02의 모든 required receipt가 같은 candidate에 bind된 closed report로 replay될 때에만 C02가 결론을 낼 수 있다.

## 5. 구현 변경 경계

```text
crates/riley-server/src/main.rs
crates/riley-server/src/service.rs
crates/riley-server/src/engine.rs
crates/riley-runtime/src/llama/batch_executor.rs
crates/riley-server/Cargo.toml, Cargo.lock
benchmarks/release/candidates/effective-runtime-config-evidence-v1.schema.json
benchmarks/release/candidates/c02-p0-runtime-config-receipt.md
ci/release/effective_runtime_config_contract.py
ci/release/write_effective_runtime_config_startup_artifact.py
ci/release/test_effective_runtime_config_contract.py
ci/release/validate_raw_c02_runtime_config.py
ci/release/test_validate_raw_c02_runtime_config.py
ci/release/run_remote_c02_runtime_config_capture.sh
ci/release/test_run_remote_c02_runtime_config_capture.py
ci/check_workspace_boundaries.py
ci/approved_cargo_dependencies.toml
```

외부 raw evidence root와 C02 candidate result는 이 PR의 source-tree 산출물이 아니다. C02-P0는 이 파일들을 create-only/C02 replay 가능한 contract에 맞추되, C02 actual evidence를 만들어 채우지 않는다.

## 6. 승인 기준

- cold prepare 이후 두 profile에서 canonical endpoint와 create-only artifact를 생성한다.
- 준비 전 또는 runtime fact를 resolve할 수 없는 경우 endpoint는 `503`이고 invented default가 없다.
- endpoint와 artifact는 schema, canonical-byte, payload digest, profile/configuration hash 검사에 통과한다.
- 모든 열 개 effective-config dimension이 prepared runtime fact에서 계산되고 hidden mode를 추가할 수 없다.
- P0 raw contract/validator와 remote capture runner의 positive/negative fixture가 raw 경계를 검증한다.
- C02 semantic checker의 frozen-candidate/Gate-E positive·negative integration fixture는 다음 C02 commit에서 이 raw contract를 소비한다.
- 실제 C02 candidate를 freeze하거나 qualification 완료를 주장하지 않는다.

## 7. 롤백

endpoint/artifact 구현을 revert해야 하면 release route는 configuration을 추측하지 않고 `503`으로 fail closed한다. 이미 capture된 append-only evidence는 삭제하거나 다른 candidate의 evidence로 재사용하지 않는다.

## 8. 완료 정의

release runtime이 cold prepare 뒤 두 profile의 canonical configuration body와 create-only artifact를 제공하고, C02 checker가 이 raw inputs를 semantic receipt로 replay할 수 있을 때 완료다. 이 정의는 C02의 actual qualification 완료 정의와 별개다.
