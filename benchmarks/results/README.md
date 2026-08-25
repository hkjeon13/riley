# Version-controlled benchmark evidence

## PR 08 online prefill

[`20260825T053620Z-rustinfer-online-prefill-pr08-run001`](20260825T053620Z-rustinfer-online-prefill-pr08-run001/)
은 `73cbdad9e0f6b04dd46a9e719be33ae050aa4836`의 materialized reference 대
online-softmax CUDA prefill 결과다. S128/1K/4K raw latency, S128 SmolLM2 prepared
prefill proxy, pinned S7 correctness, Nsight Compute DRAM counters와 Compute Sanitizer
결과를 묶는다. 원문 대형 로그는 artifact가 고정한 checksum과 함께 `server-4096`의
append-only evidence root에 보존한다.

## PR 01 baseline

이 디렉터리의 두 `run003` bundle이 PR 01 Gate A의 canonical, version-controlled
증거다. 모델 실행은 primary RTX 4090 원격 host에서만 수행했고, source checkout은
`09911ba2630845e9d4094b7c33c3ff65931a919c`에 고정했다. 이 revision은 gzip cache
inventory 계약까지 포함하며, 결과와 이 README를 반입한 후속 commit과 구분한다.

고정한 contract는 report `rustinfer.repeatability.v2`, runner
`rustinfer.repeatability-runner.v2`, preparation
`rustinfer.repeatability-preparation.v2`, cache artifact
`rustinfer.cache-inventory-artifact.v1`, result metadata
`rustinfer.benchmark-result.v1`, finalizer `rustinfer.benchmark-finalize.v1`이다.
표의 bundle 크기는 모든 regular file의 logical byte 합이다.

## 채택한 canonical bundle

| lane | bundle | Gate | 크기 | 파일 | measured raw / combined rows |
|---|---|---:|---:|---:|---:|
| HF Transformers eager | [`20260824T185633Z-hf-transformers-eager-repeatability-pr01-v2-run003`](20260824T185633Z-hf-transformers-eager-repeatability-pr01-v2-run003/) | v2 pass | 20,804,958 bytes | 320 | 20 / 455 |
| vLLM | [`20260824T192344Z-vllm-repeatability-pr01-v2-run003`](20260824T192344Z-vllm-repeatability-pr01-v2-run003/) | v2 pass | 21,904,965 bytes | 281 | 20 / 455 |

두 bundle 모두 다음을 만족한다.

- `rustinfer.repeatability.v2`, `status=passed`, report error 0건
- 5개 독립 run × 4개 exact cell, measured failure 0건, token identity 일치
- warm throughput CV ≤ 5%, warm latency p50 CV ≤ 5%, p95 CV ≤ 10%
- cold model-load p50 CV ≤ 10%, peak VRAM relative range ≤ 1%
- 공통 contract validator: lane별 1,000 file-row trials, 두 bundle 반입 후 2,000 trials
- finalize manifest의 모든 파일 크기와 SHA-256 일치
- cache inventory JSON round-trip, canonical compact bytes, gzip level 9와 `mtime=0` 일치
- Git hosting 상한보다 충분히 작은 최대 단일 파일: HF 6,571,495 bytes, vLLM 7,199,118 bytes

### 반복성 통계

| lane | state / workload `(c,p,o)` | throughput p50 CV | latency p50 CV | latency p95 CV | cold model-load p50 CV | peak VRAM range | failures |
|---|---|---:|---:|---:|---:|---:|---:|
| HF | warm `(1,128,32)` | 2.8357% | 2.8012% | 3.7966% | — | 0% | 0 |
| HF | warm `(1,4096,128)` | 2.6668% | 2.7016% | 6.7986% | — | 0% | 0 |
| HF | warm `(8,128,32)` | 2.4024% | 2.4329% | 4.5804% | — | 0% | 0 |
| HF | cold `(1,128,32)` | 3.9887% (진단) | — | — | 0.9848% | 0% | 0 |
| vLLM | warm `(1,128,32)` | 0.5818% | 0.5844% | 3.4467% | — | 0% | 0 |
| vLLM | warm `(1,4096,128)` | 0.1611% | 0.1611% | 0.3804% | — | 0% | 0 |
| vLLM | warm `(8,128,32)` | 0.1142% | 0.1022% | 5.0099% | — | 0% | 0 |
| vLLM | cold `(1,128,32)` | 1.7696% (진단) | — | — | 0.4536% | 0% | 0 |

Cold throughput은 run마다 warmup 없는 첫 request 한 개뿐이므로 v2에서는 통계로
보존하되 gate로 사용하지 않는다. Cold의 pass/fail은 model-load, VRAM, failure와
token identity가 결정한다.

Validator의 lane별 1,000 trials는 서로 다른 measured 관측 수가 아니라 발견한
JSONL의 file-row 검증 수다. 455개 nested measured row, 그 455개를 결정적으로
결합한 top-level `raw.jsonl`, 그리고 90개 prime row를 각각 검증한다. 성능 통계에
사용한 distinct measured row는 lane별 455개이며 prime과 combined 복제본은 제외한다.

### Top-level SHA-256

| lane | `repeatability-report.json` | `raw.jsonl` | `finalize-manifest.json` | `completion.json` |
|---|---|---|---|---|
| HF | `6c76da4f0539ee117a87f53ed2cc63ceaf52914e80f7f1e82be8cbd54da1b5f3` | `5e02c537e1a155942e88d2086585f2b82a34660aeeba220fbaa5c111b93c1bef` | `bb42f56d8b3972981d4c423468893e9a5b2d16f7814c0399929f76987b664e32` | `d92473476133e660b2cf40f76b213cef68bc6a9d3564d4e96cc01bbcd695e62f` |
| vLLM | `a10699fd6200a85e6df65efac2fa2ef9368e17706f4023dc3e21cf7e7d12682d` | `97830a7fc574d7a30b88a4027f374d4de9ff5c47e08c51dc4139c32252ca82b8` | `04c38c94535a07f6842ae65677c11cf92bc5870b8f32862db98142fdc611191b` | `4fcaaaa4acff4fc1f96f2c174b5dda7d27145805b9bb2b8c09135a7be4775a84` |

## Calibration history와 제외한 실행

실패하거나 저장 계약을 만족하지 못한 실행을 새 결과로 덮어쓰거나 사후에
재판정하지 않았다. 원격 staging은 다음 경로에 그대로 보존한다.

### Checker v1 calibration

Source `adafd218c4d5410b550c6cfd0ff8f65269bb161c`의 첫 canonical 실행에서 HF는
통과했지만 vLLM은 cold throughput CV 14.1242% 때문에 실패했다. 같은 cold
cell의 model-load CV는 0.2487%, VRAM range와 failure는 0이었다. 단일 first-request
throughput을 warm steady-state threshold로 판정한 의미 오류를 수정해 checker v2를
도입했고, 기존 raw를 v2로 소급 통과시키지 않고 새 commit에서 전 셀을 다시 실행했다.

- HF staging: `server-4096:/home/psyche/rustinfer-artifacts/pr01/adafd218c4d5410b550c6cfd0ff8f65269bb161c/repeatability-hf-staging`
  - report `0032af9ab3fd5e7301cf64c5f2d8c3b93128ac380e5f404711ad387808ae214c`
  - raw `9b406a7974c01bc429843f14741fc2cc4e1018fa50aed2aaa8ce0aff3de0630a`
- vLLM staging: `server-4096:/home/psyche/rustinfer-artifacts/pr01/adafd218c4d5410b550c6cfd0ff8f65269bb161c/repeatability-vllm-staging`
  - failed report `42c70162c809fc5e5ad74b5b50d40222b250cbdcb078ecc6c9292c489d8a9350`
  - failure `3a7dfc85fe0190bbbaaa73557392cb78b1fbccd76b64ba6f3f3f888eeb432fd6`

### Passing run002, Git artifact로는 제외

Source `3472921189ae1d3115e0eb87e4d26ffce3cf75f8`의 run002는 두 lane 모두 v2를
통과했다. 그러나 전체 cache inventory를 평문 JSON으로 보존해 HF bundle이
272,763,951 bytes, vLLM bundle이 294,782,034 bytes였고 단일 파일이 각각 최대 135,560,727 bytes와
151,087,106 bytes였다. 내용은 버리지 않고 원격에 보존하되 version control에는
반입하지 않았다. 그 뒤 inventory 내용과 summary/fingerprint를 유지한 deterministic
gzip 계약을 추가하고 run003을 새로 수행했다.

| lane | remote staging | report | raw | finalize manifest | completion |
|---|---|---|---|---|---|
| HF | `server-4096:/home/psyche/rustinfer-artifacts/pr01/3472921189ae1d3115e0eb87e4d26ffce3cf75f8/repeatability-hf-run002-staging` | `6a983eb7ad8455694afa096a834ea3f42d7cca0257571cb2543f44b8e2e0a1ce` | `effe27e14515bc83ad9993140d951ac1471703da32062119891a79ea04823462` | `60c13575e4dbc4e6e08155f49ce8eb73010f7fa3c102d7c9d5294f34a84030d9` | `063560b21489751ae069bc23f198a0c2d524de6800f1e49a4e34d89fe748c2e6` |
| vLLM | `server-4096:/home/psyche/rustinfer-artifacts/pr01/3472921189ae1d3115e0eb87e4d26ffce3cf75f8/repeatability-vllm-run002-staging` | `a23512a69e3bcbe3c9088d87157b909fef9f16053283280f39db7ded0863fea3` | `1f9063971c288b0902361af4c42ba0ace1154d89cb67ebf8b04afa6368b3b5e4` | `e7a0fe135ad34b179581f1d7b078d44a706f08ab38e4eeabff49b29ece5c9ba6` | `91eb9d95fa312ddb08a53fb1dfd133a8de2ec842ce99b4f349f7e9d694825e44` |

Run003의 외부 원본 staging도 append-only로 다음 위치에 남긴다.

- `server-4096:/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-hf-run003-staging`
- `server-4096:/home/psyche/rustinfer-artifacts/pr01/09911ba2630845e9d4094b7c33c3ff65931a919c/repeatability-vllm-run003-staging`

위 원격 staging은 PR 01 release evidence를 durable artifact store로 이관하고
별도의 deprecation 결정을 기록할 때까지 보존한다.

## Model-free 재검증 명령

다음 명령은 로컬에서 모델이나 GPU를 사용하지 않고 실행한다. Canonical model
runner의 exact argv와 sanitized environment는 각 bundle의 `execution-plan.json`과
`metadata.json`에 보존돼 있다.

```bash
PYTHONPATH=tools/python/reference \
  python3 -m unittest discover \
    -s tools/python/reference/tests -t tools/python/reference -p 'test_*.py'
python3 -m unittest discover -s benchmarks/lanes/vllm/tests -p 'test_*.py'
python3 -m unittest discover -s benchmarks/scripts/tests -p 'test_*.py'
python3 benchmarks/scripts/validate_contract.py
git diff --check
```

검증 결과는 각각 reference 57개, vLLM adapter 17개, benchmark scripts 57개가
통과했고 contract validator는 3 lanes, 31 prompts, 31 golden cases와 2,000
file-row trials를 확인했다.
