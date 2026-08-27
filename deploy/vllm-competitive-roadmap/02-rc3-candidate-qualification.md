# C02 — RC3 Candidate-bound Qualification

**상태:** Planned  
**의미 등급:** `reference` + 기존 승인 `E0` 검증  
**한 가지 목적:** 최신 단일 Riley revision과 exact release binary를 대상으로 Gate E, Python-free, correctness, performance regression, soak를 모두 다시 실행해 정식 candidate를 판정한다.

[이전: C01](01-vllm-win-contract.md) | [목차](README.md) | [다음: C03](03-scheduler-output-routing-fuzz.md)

## 1. 배경

`0.1.0-rc2`는 mixed prefill/decode output routing 결함을 수정했지만, 최종 revision과 binary에 결합된 전체 soak를 다시 수행하지 않은 owner-approved prerelease다. 이후 decode fast path도 추가됐으므로 이전 evidence만으로 최신 main의 안정성을 주장할 수 없다.

이 PR은 qualification 동안 production 동작을 바꾸지 않는다. gate 실패가 코드 결함을 발견하면 qualification을 중단하고 별도 corrective PR을 만든 뒤 새 candidate SHA로 처음부터 재실행한다.

## 2. Candidate freeze

qualification 시작 전에 다음을 create-only로 고정한다.

```text
full Git commit SHA
clean source archive SHA-256
release ELF SHA-256
container image ID/digest
Cargo.lock SHA-256
CUDA C ABI version
Rust, nvcc, driver, CUDA runtime/toolkit, cuBLAS versions
model/tokenizer/config/weights hashes
exact CLI and environment configuration
extension registry bytes/hash
correctness contract/report hashes
```

`main`이 이후 이동해도 candidate는 바뀌지 않는다. 하나의 gate라도 다른 revision/binary를 사용하면 final report는 `incomparable`이다.

## 3. 범위

### 포함

- canonical E0 correctness 31-case
- SmolLM2와 Qwen multi-step generation
- cache on/off 또는 연속/paged KV parity
- fixed/active-row, synchronous/packed, CPU/GPU greedy 등 candidate가 제공하는 exact backend 조합
- mixed prefill/decode와 output slot ordering
- Python 없는 clean release environment
- API streaming, cancellation, client disconnect, overload
- long-running soak와 memory accounting
- reproducible build와 dependency manifest
- release rollback drill

### 비범위

- 새로운 CUDA Graph 구현
- 새로운 kernel fusion
- 신규 model family
- threshold 완화
- failing test의 ignored 처리

## 4. Candidate configuration 명시

현재 문서들 사이의 historical default 표현이 다를 수 있으므로 qualification은 추론으로 default를 결정하지 않는다. release binary가 다음을 machine-readable startup receipt와 `/metrics` 또는 config endpoint에 출력해야 한다.

```text
execution completion mode
batch shape policy and buckets
metadata transport
sampling backend
attention backend
GEMM reduction policy
experimental flags
fallback policy
batch token budget
KV block/page geometry
```

qualification arm은 `stable-default`와 `max-performance-exact` 두 개로 분리한다. stable release 승격은 stable-default만 판정하며, max-performance-exact는 opt-in evidence로 별도 보고한다.

## 5. Correctness matrix

### Request shape

- concurrency `1,2,5,8`
- prefill-only, decode-only, completing-prefill+decode mixed iteration
- prompt `empty, short, 128, 1024, 4096, near-context-limit`
- output `1,2,32,128`

### KV boundary

- token positions `15→16→17`
- multiple physical page permutations
- reserve/commit/abort/cancel paths
- capacity near-OOM admission

### Sampling

- greedy exact token sequence
- deterministic seed sampling
- request order permutation
- cancelled/rejected branch RNG consumption
- non-finite logits and invalid status handling

### Output routing

- dense slot permutation
- prefill output at last slot, decode outputs at preceding slots
- mid-iteration cancellation
- commit failure after GPU result
- duplicate/unknown request rejection

## 6. Python-free release gate

Python이 설치되지 않고 `PATH`에도 없는 clean image에서 다음을 수행한다.

1. release artifact 설치 또는 copy
2. model/config/tokenizer/safetensors load
3. prefill/decode
4. greedy 및 승인된 sampling
5. streaming HTTP
6. cancellation/client disconnect
7. model shutdown
8. golden token 검증

추가 검사는 다음과 같다.

- `ldd` 또는 loader manifest
- process tree에 Python child 0
- image filesystem에 production dependency로 PyTorch/Transformers/Triton JIT 없음
- network disabled 상태에서 startup/generation 성공
- runtime fallback이 Python subprocess가 아님

## 7. Soak 시나리오

기존 release contract의 전체 duration을 그대로 사용한다. 최소 시나리오는 다음을 순환한다.

- steady short-chat load
- burst 후 idle
- short/long prompt 혼합
- cancellation 0%, 10%, 50%
- client disconnect
- malformed request 반복
- KV utilization 70%, 90%, capacity boundary
- exact backend fallback 유발
- graceful shutdown/restart
- model load/unload 반복

매 interval에서 RSS, pinned bytes, VRAM, free/reserved/active KV blocks, request state totals, terminal event counts를 기록한다.

## 8. Fault injection

- host allocation failure
- device/pinned allocation rollback ambiguity
- H2D/D2H deferred error
- synchronize/query failure
- post-KV-write runtime error
- output status corruption test double
- scheduler commit failure
- worker/channel close race

실제 device-loss를 안전하게 재현할 수 없는 경우 injectable backend와 subprocess isolation을 사용하되, synthetic 결과와 실제 GPU sanitizer 결과를 구분해 보고한다.

## 9. Reproducible build

동일 source archive를 독립 clean environment 두 곳에서 build한다.

- native library와 release binary hash 비교
- build metadata와 embedded source revision 비교
- 허용된 비결정 요소가 있으면 binary section별 원인을 문서화
- 최소한 executable behavior, dependency manifest, ABI hash는 exact해야 함

binary hash exact를 목표로 하며 불가능한 toolchain metadata가 있으면 결과를 본 뒤가 아니라 PR 시작 전에 normalization 정책을 선언한다.

## 10. 예상 파일 변경

```text
benchmarks/release/candidates/rc3-candidate.json
ci/release/run_rc3_qualification.sh
ci/release/check_rc3_qualification.py
ci/release/tests/*
.github/workflows/rc3-qualification.yml
CHANGELOG.md
```

실제 raw evidence는 append-only result directory와 승인된 외부 evidence root에 저장한다.

## 11. Final report 조건

final report는 다음 모든 receipt hash를 포함한다.

- candidate manifest
- canonical correctness
- Qwen regression
- Python-free E2E
- performance regression
- soak
- fault injection
- reproducible build
- dependency manifest
- rollback drill

하나라도 없거나 다른 candidate에 묶이면 `passed`가 될 수 없다.

## 12. 승인 기준

- canonical failure/token mismatch 0
- cancellation/overload에서 request 또는 KV leak 0
- duplicate terminal/output routing mismatch 0
- soak 종료 후 RSS/VRAM/KV 추세가 contract bound 내
- Python-free gate 통과
- stable-default performance가 승인 baseline threshold를 통과
- rollback binary로 전환 후 health/generation 정상
- 모든 evidence가 같은 candidate SHA/source/binary에 결합

## 13. 실패 처리

qualification PR 안에서 production 코드를 고치지 않는다.

1. failing candidate를 immutable evidence로 보존한다.
2. failure class와 최소 재현을 issue/별도 corrective PR로 분리한다.
3. 수정 PR이 병합되면 새 candidate ID를 만든다.
4. 모든 gate를 처음부터 재실행한다.

부분 통과 결과를 이전 candidate와 조합하지 않는다.

## 14. 롤백

정식 tag 생성 전에는 기존 prerelease/tag를 유지한다. RC3 배포 후 이상이 발견되면 문서에 고정한 이전 stable/prerelease artifact로 atomic rollback하고, candidate worker/model을 drain한 뒤 재사용하지 않는다.

## 15. 완료 정의

최신 단일 Riley revision에 대해 release checker가 예외나 waiver 없이 `Gate E passed`, `Python-free passed`, `candidate qualified`를 반환할 때 완료다.
