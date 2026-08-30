# Riley vLLM 경쟁력 로드맵

**상태:** In progress — C01과 C02-P0는 clean remote branch에 별도 커밋됐고, C02-P1 provenance closure와 C02 source pre-freeze check가 candidate freeze 전에 진행 중이다. initial one-scenario lifecycle supervisor/receipt, native sampling fallback source leaf/marker, source pair raw capture v2, fresh native-fallback lifecycle-evidence preparer v5, fixed native-fallback lifecycle bind-request writer v5, 별도 terminal binder v5의 private held-lock core와 fixed-name private raw compositor, authenticated native-fallback raw-v5 runner, pre-existing binary-bound RC2 baseline v2 held-FD admission과 정적 v3 binding-input snapshot preparer, strict held-FD candidate-source/config/shutdown join과 complete consumed-path inventory·static identity/descriptor TOCTOU closure를 가진 fixed-name rollback v3 bind-request writer 및 same-stack private v3/v4 finalizer, 그 정상 반환만 같은 FD stack에서 이어 쓰는 private rollback finalizer receipt v1, same-invocation rollback terminal-provenance v4, **authenticated RC3 rollback raw runner and private held-FD raw operational-semantics replay**, held-FD source pre-freeze checker, RC3 freeze-input structural admission, create-only RC3 frozen-candidate input-identity manifest/FD-safe replayer, fresh opaque 14-leaf Gate E input snapshot preparer와 closed four-gate input-inventory held-FD replayer, native·optimizer canonical-E0, Python-free, performance, soak held-FD semantic component adapters와 aggregate Gate E held-FD semantic replayer는 CPU/static hostile-path 범위로 구현됐다. 각 component adapter의 `passed`와 aggregate replay는 exact input consistency만 재검증하며, optimizer와 Python-free의 caller-supplied image/golden 값은 exact equality anchor일 뿐 독립 review 승인 자체는 주장하지 않는다. actual Gate E producer normal return, durable semantic receipt, actual GPU capture·actual candidate freeze·lifecycle-v5 receipt·qualification은 아직 수행하지 않았다.
**작성 기준:** 초기 성능 비교 기준은 `main@1195cf20eef0bd6c3d72ac90437d308265e6f951`이며,
현재 source-defaults release/pre-freeze contract pin은
`main.rs@21f445f4870a140346509144c36c7294f2f677f3`이다.
**목표:** Riley가 제한된 우선 지원 범위에서 vLLM보다 더 낮은 지연, 더 높은 SLO goodput, 더 예측 가능한 오류 격리와 복구를 제공하도록 후속 작업을 독립 PR로 분해한다.

이 폴더는 기존 [`deploy/00-pr-contract.md`](../00-pr-contract.md), [`deploy/15-profiling-and-optimization.md`](../15-profiling-and-optimization.md), [`deploy/16-reliability-and-release.md`](../16-reliability-and-release.md), [`deploy/17-extension-gates.md`](../17-extension-gates.md)의 하위 실행 계획이다. 기존 gate와 충돌하는 경우 기존 문서의 fail-closed 규칙이 우선한다.

## 1. 현재 출발점

현재 Riley는 SmolLM2-135M BF16, RTX 4090, `c1/p128/o32/greedy`에서 active-row bucket, packed asynchronous metadata H2D, GPU greedy, cooperative shared-KV GQA attention을 결합하여 내부 baseline 대비 다음 결과를 확보했다.

| 지표 | 기존 경로 | M1 후보 | 변화 |
|---|---:|---:|---:|
| TTFT p50 | 5.450 ms | 4.033 ms | -25.98% |
| TPOT p50 | 7.166 ms | 4.109 ms | -42.66% |
| E2E p50 | 227.595 ms | 131.404 ms | -42.26% |
| 처리량 | 140.535 tok/s | 243.536 tok/s | +73.29% |

다만 이 결과는 최신 Riley와 vLLM을 같은 시점·같은 candidate·같은 campaign에서 직접 비교한 승리 증거가 아니다. 또한 `0.1.0-rc2`는 candidate-bound 전체 soak를 생략한 prerelease이므로 정식 Gate E qualification도 남아 있다.

## 2. 최초 경쟁 범위

Riley는 vLLM의 전체 기능을 복제하지 않고 다음 범위부터 이긴다.

| 축 | 최초 범위 |
|---|---|
| GPU | RTX 4090 우선, H100 확장 |
| 모델 | Llama/Qwen 호환 dense decoder |
| 크기 | 진단용 135M, 경쟁 판정용 0.5B·1~3B·7~8B |
| dtype | BF16 우선 |
| context | 128~8K, 후속 32K |
| concurrency | 1, 2, 4, 8, 16, 32 |
| output | 32, 128, 512 tokens |
| API | OpenAI 호환 streaming |
| 초기 비범위 | MoE, multimodal, multi-GPU, multi-LoRA |

## 3. 경쟁 milestone

| milestone | 판정 |
|---|---|
| M1 | 기존 Riley 대비 TPOT 30% 이상 개선 — 달성 |
| M2 | graph-ready decode 경로로 기존 Riley 대비 TPOT 2배 이상 개선 |
| M3 | 대표 c1 decode TPOT p50 2.0 ms 이하 |
| M4 — parity | 동일 campaign에서 Riley/vLLM TTFT·TPOT p95 ratio 각각 `<= 1.03` |
| M5 — win | TTFT·TPOT p95 ratio 각각 `<= 0.90`, SLO goodput ratio `>= 1.10`, peak VRAM ratio `<= 1.05` |
| S1 — stable win | candidate-bound soak와 오류 주입에서 잘못된 token·누수·hang·중복 종료 0 |

절대 지연 목표와 경쟁 상대 대비 비율을 함께 판정한다. vLLM 버전이 바뀌면 과거 절대값으로 승리를 선언하지 않고 같은 campaign에서 M4/M5를 다시 계산한다.

## 4. PR 순서

| ID | 문서 | 한 가지 목적 | 선행 조건 |
|---|---|---|---|
| C01 | [vLLM 승리 계약](01-vllm-win-contract.md) | 공정 비교와 M4/M5 판정 계약 고정 | 없음 |
| C02-P0 | [effective runtime configuration receipt](02a-effective-runtime-config-receipt.md) | cold-prepared `/v1/config`와 startup artifact raw evidence 구현 | C01 |
| C02-P1 | [provenance v2와 reconstructed rollback baseline](02b-c02-p1-provenance-v2.md) | v3 config bridge와 v4 serial-session raw provenance 위에 one-scenario lifecycle/receipt raw chain, source-owned native fallback leaf, 그리고 raw capture v2를 닫고 self-authored soak/rollback summary를 raw process/evidence provenance로 교체 | C02-P0 |
| C02 | [RC3 candidate qualification](02-rc3-candidate-qualification.md) | 최신 단일 revision의 정식 release gate 종료 | C01, C02-P0, C02-P1 |
| C03 | [Scheduler routing property fuzz](03-scheduler-output-routing-fuzz.md) | request-token routing invariant를 생성형 테스트로 고정 | C02 |
| C04 | [Llama executor 분리](04-llama-executor-refactor.md) | graph/fusion 작업 전에 거대 executor를 동작 보존 분리 | C03 권장 |
| C05 | [CUDA Graph ownership ABI](05-cuda-graph-ownership-abi.md) | capture/instantiate/replay/close의 native ownership 경계 구현 | C04 |
| C06 | [Graph signature dispatcher](06-graph-signature-dispatcher.md) | full/piecewise/eager 선택과 exact fallback 구현 | C05 |
| C07 | [Decode graph buckets](07-decode-graph-buckets.md) | `M=1..32` pure-decode graph fast path로 M2 판정 | C06 |
| C08 | [Executable pattern registry](08-executable-pattern-registry.md) | semantic IR과 kernel implementation 선택 분리 | C04, C07 결과 |
| C09 | [Packed QKV/Gate-Up weights](09-packed-projection-weights.md) | 중복 projection과 weight ownership 정리 | C08 |
| C10 | [Transformer subgraph fusion](10-transformer-subgraph-fusion.md) | QKV-RoPE-KV 및 MLP 반복 subgraph의 E0 fusion | C09 |
| C11 | [LM-head/sampling fusion](11-lm-head-sampling-fusion.md) | greedy 경로의 full logits materialization 제거 | C08, C10 권장 |
| C12 | [Tenant-safe prefix cache](12-tenant-safe-prefix-cache.md) | exact prefix reuse와 보안/lifetime 계약 구현 | C02, KV 안정화 |
| C13 | [Restartable GPU worker](13-restartable-gpu-worker.md) | CUDA/panic 실패를 API process에서 격리 | C02, C03 |
| C14 | [Multi-model/hardware matrix](14-multi-model-hardware-matrix.md) | 0.5B~8B·RTX4090/H100에서 M4/M5 최종 판정 | C07~C13 선택 결과 |

C12는 기존 PR 17 규칙 때문에 하나의 merge PR로 끝나지 않는다. 실제 병합은 `C12-A admission-only`와 `C12-B implementation`, 필요 시 schema v2 이후 `C12-C stable promotion`으로 분리한다.

## 5. 공통 PR 계약

모든 PR은 다음을 지킨다.

1. 한 PR은 한 가지 성능 또는 안정성 가설만 다룬다.
2. `reference`, `E0`, `E1`, `A1`, `M1` 중 의미 등급을 명시한다.
3. correctness 수정과 aggressive optimization을 같은 PR에 섞지 않는다.
4. production default 변경 전 exact fallback과 명시적 rollback flag를 유지한다.
5. threshold는 결과를 본 뒤 완화하지 않는다.
6. 성능 evidence는 clean Git commit, source archive, executable hash, model/tokenizer hash, GPU UUID, image, driver/CUDA 버전에 결합한다.
7. hot loop의 host/device allocation 증가는 허용하지 않는다.
8. 오류가 난 arm의 결과를 성공 표본에 포함하지 않는다.
9. unsupported shape, sampling, backend는 암묵적 근사가 아니라 exact fallback 또는 fail-closed로 처리한다.
10. Python/Triton은 reference와 prototype에만 사용하며 production dependency graph에 들어가지 않는다.

## 6. 공통 promotion gate

별도 문서가 더 강한 기준을 두지 않는 한 다음을 최소 기준으로 사용한다.

- failure, dropped trace, token routing mismatch: 모두 0
- canonical numeric/token correctness: 통과
- independent process 5회 이상, AB/BA 교차 실행
- primary metric arm median 및 paired median: 사전 목표 통과
- TTFT p95 candidate/current ratio: `<= 1.05`
- 비대상 필수 workload TPOT/E2E p95 ratio: `<= 1.05`
- c8 이상 대표 throughput candidate/current ratio: `>= 0.95`
- peak VRAM 증가: `<= 5%`
- usable KV block capacity 감소: 0 또는 사전 승인된 명시적 trade-off
- hot-loop allocation delta: 0
- owner close 후 Riley-owned live allocation: 0
- cancellation, client disconnect, overload에서 duplicate terminal event: 0

M4/M5 비교 PR에서는 current baseline 대신 같은 campaign의 vLLM arm을 분모로 사용한다.

## 7. 공통 산출물

각 PR은 최소 다음을 남긴다.

```text
설계 문서
변경 전/후 source boundary
정확성 report
성능 또는 신뢰성 raw evidence
기계 판독 가능한 closed report
환경·artifact provenance
운영 flag와 default 상태
rollback 절차
known limitations
```

승격하지 않은 후보도 append-only evidence에 보존한다. 실패한 실험을 삭제하거나 성공 결과로 덮어쓰지 않는다.

### C02 read-only structural admissions

check_soak_v2_receipt.py와 check_rc3_rollback_structural_precheck.py는 각각 completed raw soak v4/v5와 completed rollback terminal provenance v4를 held private-FD replay로 읽는 admission diagnostic이다. 두 output 모두 정확히 bound/not-run, authority raw-structural-only이며 producer/lifecycle/rollback success, semantic receipt, candidate freeze, Gate E 또는 qualification을 주장하지 않는다. rollback precheck CLI/API는 python3 -B 또는 PYTHONDONTWRITEBYTECODE=1 없이 evidence를 읽지 않으며, embedding caller가 import 직전 bytecode-write flag를 바꾼 경우도 거부한다. post-link fsync ambiguity 뒤에도 visible completion pair가 남을 수 있으므로, future semantic checker와 outer C02/RC3 finalizer는 이 두 schema/version을 semantic input으로 수용해서는 안 된다.

### RC3 rollback finalizer normal-return receipt

`write_rc3_rollback_finalizer_receipt_v1.py`와 `rc3-rollback-finalizer-receipt-v1.schema.json`은 public path replayer가 아니라 caller-held root EX/switch EX 안의 private continuation이다. fixed v3/v4 finalizer가 **이번 invocation에서** 정상 반환한 typed closure만 받아 static preparation, candidate/source join의 complete consumed-path inventory, candidate/rollback phase, transaction, fixed request와 v3/v4 descriptor를 다시 비교한다. 첫 receipt leaf 전에는 private operational replay를 failure-only veto로 실행해 typed v3/v4/transaction closure와 cross-bind한 뒤 fixed receipt와 paired marker를 create-only로 낸다. veto diagnostic은 disk나 receipt JSON에 남지 않으며, status는 `completed/not-run`, authority는 `raw-finalizer-normal-return-only`뿐이다.

모든 closure/receipt replay와 ephemeral veto는 receipt leaf와 marker를 만들기 **전에** 끝나며, terminal hardlink helper의 paired-link validation·directory sync가 성공하면 함수는 그 뒤 즉시 반환한다. 따라서 visible receipt pair는 독립된 rollback/lifecycle success 또는 later semantic input이 아니고, terminal hardlink 자체가 post-link 오류를 `ambiguous-terminal-publication`으로 보고할 때만 pair가 남은 채 성공 반환이 없을 수 있다. 정상 완료된 v4 뒤 veto가 실패하면 v4 pair만 남고 receipt pair는 없으며, fixed output collision 때문에 그 root는 재개하지 않고 보존·retire한 뒤 새 private root에서만 새 시도를 시작한다. authenticated rollback runner의 같은 normal-return stack만 함수 반환값을 소비할 수 있으며, path reopen/resume/CLI 또는 structural precheck로 그 edge를 복구할 수 없다. GPU, deployment rollback, candidate freeze, Gate E, semantic receipt와 qualification은 계속 후속 단계다.

check_rc3_freeze_input_admission.py는 실제 freeze 전에 source pre-freeze checker를 입력 읽기 전후로 재실행하고, source checkout 밖의 exact-0700 evidence root에서 하나의 canonical request와 모든 declared external leaf를 held-FD shared lock 아래 재해시하는 별도 admission이다. external Cargo.lock과 extension registry는 새 source pre-freeze report의 SHA-256과 byte length에 일치해야 하며, workspace manifests와 reviewed server-defaults source는 caller input이 아니라 그 report에서만 derive한다. 두 launch arm의 line-delimited args/env는 runner-owned C02 identity option과 freeze/Gate E/configuration self-reference를 거부한다. reconstructed baseline은 canonical manifest bytes를 hash-bind한 뒤 binary-bound reconstructed-v2 vocabulary만 검사하며 binary-unbound v1은 명시적으로 거부한다; nested baseline graph나 semantic result는 replay하지 않는다. 다만 baseline tag는 candidate와 같은 semver의 바로 앞 RC여야 하며, report는 그 declared tag와 baseline ID를 별도 structural binding으로 기록한다. 이 비교는 manifest의 declared vocabulary만 대상으로 하며 Git tag object/target, archive 또는 nested baseline leaves의 history proof는 후속 full replay가 담당한다.

hostile request가 rehash I/O를 무제한 증폭하지 않도록 admission request는 최대 8,192개 external descriptor와 총 1 TiB declared byte budget으로 제한한다. 이는 nested baseline raw graph를 아직 replay하지 않는 이 read-only admission의 resource boundary이며, semantic evidence의 충분성 판정이 아니다.

성공 output은 riley.rc3-freeze-input-admission.v1의 bound/not-frozen/not-run 및 freeze-input-structural-only authority뿐이다. 이 checker는 artifact, marker, freeze hash, Gate E report, semantic receipt 또는 qualification을 쓰거나 만들지 않는다. frozen-candidate writer와 outer finalizer는 이 diagnostic을 semantic input으로 승격하지 말고 original request와 raw leaves를 다시 replay해야 한다.

### RC3 frozen-candidate input identity

`write_rc3_frozen_candidate_v1.py`는 새 source/input tree 밖의 create-only `0700` root에 단 하나의 `frozen-candidate.json`을 기록한다. writer와 `replay_rc3_frozen_candidate_v1.py`는 admission report나 `freeze.raw`를 받지 않고, caller-held source/input/root FD에서 original canonical request와 모든 declared leaf를 다시 해시한다. reconstructed baseline은 vocabulary 확인에 그치지 않고 v2 raw graph 전체를 held-FD replay하며, candidate request와 baseline raw leaf의 path reuse도 거부한다. 두 CLI는 original admission bytecode guard를 유지하므로 `python3 -B` 또는 `PYTHONDONTWRITEBYTECODE=1`로만 호출한다. 결과는 `frozen/not-run`, `frozen-candidate-input-identity-only`뿐이다.

이 manifest는 대용량 모델/weight를 복사하지 않는 recheckable input pin이며, source archive→revision, ELF/container/toolchain/model/correctness의 의미적 provenance, Gate E, semantic receipt, deployment, rollback result, qualification을 증명하지 않는다. manifest가 보인다는 사실도 writer normal-return lineage를 증명하지 않는다. 후속 same-stack producer는 exact v1 manifest와 original raw input을 다시 replay해야 한다.

### RC3 Gate E closed input inventory

`replay_rc3_gate_e_input_inventory_v1.py`는 별도 private `0700` Gate E root의 canonical `gate-e-inputs.json`과 정확히 14개의 direct regular leaf만 수용한다. v1은 Gate E의 네 공통 증거군인 canonical E0(native 31-case report/raw와 optimizer E0 report/raw 및 각각의 candidate/profile binary), Python-free(report/raw/golden), performance(report/raw), soak(report/raw)을 닫고 release bundle을 함께 bind한다. source/archive/release ELF/model은 새 self-authored scalar로 중복하지 않고 caller-held source/input/frozen roots의 frozen-candidate replay에서 다시 읽는다. inventory의 frozen-manifest descriptor는 held-FD replay 결과와 일치해야 한다.

public wrapper는 source/input/frozen/Gate-E root의 lexical·held-FD ancestry·Linux mount backing alias를 거부하고 세 private evidence root에 shared lock을 잡는다. inventory의 direct-path uniqueness와 14-leaf/1 TiB byte budget을 artifact stream 전에 확인하며, exact root entry set과 inventory를 전후 재검증한다. frozen candidate는 Gate E artifact replay 전후에 다시 replay된다. 성공 output은 오직 `riley.rc3-gate-e-input-inventory-replay.v1`, `bound/frozen/not-run`, `gate-e-input-inventory-replay-only`이다.

`write_rc3_gate_e_input_snapshot_v1.py`는 후속 authenticated producer가 사용할 수 있는
준비 단계다. fixed fourteen role의 caller-supplied opaque regular file을 새 private `0700`
root의 fixed direct `0600` snapshot leaf로 create-only copy하고, 모든 leaf의 private
rehash가 끝난 뒤에만 `gate-e-inputs.json`을 마지막으로 쓴다. source checkout FD는 held
상태로 유지하고 input/frozen root에만 shared lock을 잡아 frozen closure를 먼저 replay한 뒤,
output root는 exclusive lock으로 self-replay한다. source absolute path나 source descriptor는 inventory에 기록하지
않으며, failure 뒤 partial root는 보존되고 resume할 수 없다.

`run_remote_rc3_gate_e_session_v1.sh --supervisor-smoke-test`는 future
authenticated Gate E runner에 필요한 shared host lock와 clean parent/child
handoff만 fail-closed로 점검하는 probe다. 이 command는 GPU를 조회하거나 선택하지
않고, evidence/session root·artifact·marker·receipt를 만들지 않으며, subproducer,
semantic replay, qualification을 호출하지 않는다. stderr의 성공 진단은 오직
authenticated supervisor handoff가 완료됐다는 뜻이지 Gate E action 또는 producer
normal-return이 아니다. script path가 relative이면 ambient shell hook이 child identity를
바꾸지 못하도록 lock 전에 거부한다. 실제 producer는 raw capture를 private
under-parent-lock child core에 직접 연결할 때에만 이 probe를 확장할 수 있다.

이는 local snapshot/preparer일 뿐 source raw-capture origin, cross-leaf atomic coherence,
snapshot writer 또는 actual producer normal-return, evidence-root immutability, actual GPU
capture, Gate E semantic replay/pass, durable receipt, deployment, qualification을 확립하지
않는다. 특히 public performance/soak runner의 완료 output을 나중에 복사한 사실은
same-invocation producer authority가 아니며, future authenticated session은 private
under-parent-lock child core를 통해 raw producer를 직접 연결해야 한다.

이 단계는 report body의 self-authored `passed`/threshold를 해석하지 않으며 Gate E semantic replay, Gate E pass, promotion 또는 qualification을 만들지 않는다. CUDA fault·reproducible-build는 legacy final checker의 추가 superset이므로 four-gate v1에 넣지 않았고, Qwen/C02 runtime-config/rollback도 후속 versioned semantic boundary다. legacy path-based `check_release_candidate.py`는 호출하거나 same-stack semantic input으로 수용하지 않는다.

`replay_rc3_gate_e_python_free_v1.py`는 Python-free raw tar와 release bundle 두 개만 held FD에서 fixed external `/var/tmp` 아래 private `0700` scratch로 옮기며, legacy raw loader/validator에는 그 scratch 경로만 준다. report·golden·canonical native report는 Gate E held descriptor로 strict JSON/rehash하고, frozen manifest/original input replay에서 나온 source archive, release ELF, immutable release image digest 및 one selected model의 revision/tree/config/tokenizer/weight descriptor와 raw model을 다시 묶는다. scratch bundle의 embedded `bin/riley`는 frozen release ELF와 같아야 한다. component는 raw loader의 보관 payload를 768 MiB, bundle verifier의 uncompressed 보관 payload를 640 MiB로 별도 제한한다. 대형 model body는 여전히 stream-hash하므로 raw tar 전체 admission 한계는 유지하며, 이 수치는 process RSS 전체가 아니라 replayer가 보관하는 payload의 상한이다.

호출자는 `--expected-release-image-id`와 `--expected-correctness-golden-sha256`를 제공해야 하며 각각 frozen image digest와 Gate E golden descriptor에 exact-match해야 한다. 이는 raw/report가 자기 image 또는 golden을 anchor로 재사용하지 못하게 하는 target binding일 뿐 review approval provenance는 아니다. output은 오직 `bound/frozen/not-run`, `python_free_status: passed`, `gate-e-python-free-semantic-replay-only`이다. full legacy `evaluate()`가 다시 여는 external model mount/producer sidecar/source archive content, native/optimizer E0 semantic pass, performance, soak, aggregate Gate E, receipt, qualification, deployment은 모두 `not-established`로 남는다.

Frozen-candidate full replay에는 별도 조기 합산 한계가 있다. bounded control-plane JSON을 발견한 뒤 original request와 정확히 23개의 physical baseline leaf 합이 8,192 descriptor 및 1 TiB 이하여야 recipe나 artifact streaming을 시작한다. wrapper는 lexical path만 보지 않고 held-directory ancestry와 Linux mount backing coordinate를 검사하며, writer는 create 직후와 self-replay 직후 visible output root를 다시 열어 held FD와 묶는다. private Python layer는 caller FD를 직접 변경하지 않지만 source pre-freeze가 pinned FD를 통해 trusted read-only Git source oracle을 호출하므로 configured Git executable/PATH와 그 read-only behavior는 별도 trusted boundary다. manifest의 machine-readable not-established field에는 writer normal return과 input-root immutability도 포함된다.

## 8. 권장 착수 순서

우선 `C01 → C02-P0 → C02-P1/v3 config bridge → C02-P1/v4 serial-session binder → C02-P1 initial lifecycle raw closure → C02-P1 native fallback source leaf → native fallback capture v2/terminal binder v5 → rollback normal-return receipt → rollback·semantic/freeze closure → C02 → C03 → C04`로 비교 기준, raw evidence surface, provenance closure, release 안정성, routing invariant, 코드 경계를 닫는다. initial C02-P1 lifecycle runner는 authenticated no-follow host lock, clean environment, 새 private evidence root, host binary/model의 launch 전·후 input revalidation 아래 contract 1 scenario·즉시 observation 1회·v4 manifest 1개만 허용한다. same-process receipt finalizer만 successful v4 bind 뒤 shutdown artifact/marker를 다시 bind하고 `completed`/`not-run` raw receipt를 publish할 수 있다. native fallback source leaf는 max-performance-exact의 committed request-local sampling transition만 audit hash와 함께 기록하며, 모든 audited selection이 one-output-slot plan에서 온 경우에만 발행한다. multi-output plan은 peer-caused CPU 선택을 request-local fallback으로 오인할 수 있어 일반 audit만 남기고 leaf를 만들지 않는다. raw capture v2는 Rust `f32` decoder에서도 0으로 반올림되지 않는 public `temperature: 1` request 하나와 source audit/fallback marker 두 쌍을 보존한다. 별도 binder v5는 그 네 source leaf와 config bridge의 validated `effective_config.sampling_backend == gpu-greedy` 및 observation tuple을 다시 bind한 뒤에만 raw terminal manifest를 발행한다. 별도 `run_remote_c02_soak_v5.sh`는 authenticated no-follow GPU lock과 clean `env -i` 아래 release host binary만 `127.0.0.1`에 기동해, 새 v5 evidence root의 frozen fallback-v2 contract 1개, validated `gpu-greedy` config bridge, raw source-pair capture 및 즉시 observation 1회를 같은 invocation에서 private v5 raw finalizer로 연결한다. 최대 하나의 v5 raw manifest만 만들며 `qualification_status: not-run`이다. 이는 raw-producer 메커니즘 코드일 뿐 아직 실제 GPU에서 실행되지 않았고 candidate freeze, lifecycle-v5 receipt, Gate E/semantic qualification, Docker/SSH/system-service/privileged action을 수행하지 않는다. 기존 v4/lifecycle은 계속 fallback scenario를 거부한다. rollback terminal-provenance v4는 v3 nonterminal manifest와 fixed preparation/atomic/transaction closure를 exact descriptor map으로 join한다. 공개 raw producer는 **새 preparation부터** 시작하고 같은 held root/switch FD stack의 nested normal-return chain에서만 preparation → transaction → v3 → v4 → ephemeral operational veto → finalizer receipt를 연결한다. 어느 단계의 post-link `fsync` ambiguity나 veto failure도 새 path/FD invocation으로 재개할 수 없으며, 나중의 path-only replay는 structural `bound/not-run` 진단일 뿐 이전 completion pair나 receipt pair를 producer success로 되살리지 않는다. 새 `check_soak_v2_receipt.py`는 source checkout 밖의 private `0700` evidence root를 shared held-FD lock으로 한 번만 열어 completed raw v4/v5 manifest pair를 replay하고 `bound`/`not-run`, `authority: raw-structural-only` diagnostic만 stdout으로 낸다. visible completion pair가 post-link `fsync` ambiguity 뒤에 남아도 producer/lifecycle success로 해석하지 않으며, 이 precheck는 semantic receipt·candidate qualification이 아니므로 outer C02 finalizer가 semantic input으로 수용해서는 안 된다. 이 구현은 CPU/static hostile-path만 검증됐으며, 실제 GPU capture, candidate freeze, Gate E/semantic qualification, lifecycle-v5 receipt나 fallback·rollback 판정을 뜻하지 않는다. multi-scenario timing/집계는 후속 versioned semantic contract로 미룬다. 이후 `C05 → C06 → C07`로 CUDA Graph M2를 판정한다. C07 이후 새 profile이 실제 병목을 선택할 때만 C09~C11의 fusion 순서를 확정한다. Prefix cache와 process 격리는 성능 숫자를 만들기 위한 우회가 아니라 별도의 serving 효율·안정성 축으로 판정한다.
