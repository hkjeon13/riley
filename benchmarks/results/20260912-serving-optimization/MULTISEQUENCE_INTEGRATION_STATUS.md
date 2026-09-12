> Current instruction: Blender remains stopped by explicit user request; do not run historical restoration controllers. See V35 below.

# 다중 요청 통합 batch 진행 상태

통합 소스는 로컬 `/tmp/riley-multisequence-integration-20260912`, 원격 `/tmp/riley-opt-260912/multisequence-integration-source-v1`이다. 원격 commit `0f419241b121e725a057cd1031db3df445afa1f8`은 C1 후보 `4c5bcff43d1942fdd3c396b2b9bcd9df3bf63593`에서 분기한 격리 snapshot이다. 주 작업 트리의 기존 변경과 accepted baseline은 건드리지 않았다.

현재 native API `riley_cuda_gemm_plan_create_strided_m1`을 구현했다. 검증된 다섯 BF16 shape, M1 topology, batch2/4, compact 또는 padded stride만 허용한다. 내부 plan이 descriptor와 batch 전체 byte extent를 소유하고 기존 실행·해제 및 context-child 경로를 사용한다. Algorithm metadata의 M은 per-matrix 1을 유지한다. 기존 single-matrix graph admission은 `batch_count != 1`을 거부하므로 추후 별도 다중 행 binder가 필요하다.

실제 GPU에서 새 생성 API → 기존 span 검증/실행 API → 기존 close 경로를 사용한 2,178개 case가 BF16 불일치 0으로 통과했다. 25개 plan(다섯 M1 oracle + 20 strided)을 준비했고, 잘못된 batch/stride/M/reduction flag 8개를 출력 handle 공개 전에 거부했다. 단일 요청 graph admission의 분리도 검사했다. 메모리 가드·입력·가중치·allocation 통계 및 전체 자원 정리가 통과했다.

이 검사는 새 gemm.cu를 독립 translation unit으로 컴파일하고 나머지는 기존 native archive와 연결했다. 전체 Rust/CUDA serving build나 runtime 통합이 완료됐다는 뜻은 아니다. 아직 Rust FFI/소유권 래퍼, multi-row graph binder/owner, descriptor packet staging, scheduler reservation과 bulk commit을 연결해야 한다. 통합 batch가 끝나면 전체 모델 정확성과 실제 serving benchmark를 실행한다. 구성요소 검사만으로 높은 concurrency 성능을 주장하지 않는다.

[변경 patch](multisequence-owned-plan-v1.patch), [GPU 증빙](raw/multisequence-strided-owned-plan-v2/receipt.json), [실행 harness](multisequence_strided_owned_plan_v2.cpp).


## Rust 소유권 연결 완료

현재 원격 snapshot은 `9737c7f98fd1220d4c3fccfceee60a52466b3344`다. `CudaStridedGemmConfig`는 M1 알고리즘 형상과 batch 전체 storage extent를 구분한다. `CudaPreparedStridedGemm`는 별도 공개 타입으로 native/context 수명, buffer/context/alias 검증, 실패 후 poison, close를 소유한다. 기존 단일 요청 그래프 API가 받는 `CudaPreparedGemm`로 변환하는 경로는 제공하지 않는다.

CUDA 비활성 library 테스트 86개와 rustfmt가 통과했다. 원격 CUDA-feature library test build/link도 통과했고, 이후 실제 Rust GPU 통합 테스트에서 20개 projection/batch/stride 조합의 impulse 출력·padding·입력 보존, 잘못된 span 거부 후 정상 재사용을 확인했다. GPU 테스트 log를 로컬로 복사하고 receipt SHA와 대조했다. 이는 이전 2,178개 실제 가중치 native 검사와 구분되는 Rust 경계 검사다.

[추가 변경 patch](multisequence-rust-owner-v1.patch), [GPU 테스트 receipt](raw/multisequence-rust-strided-gpu-v1.json), [CUDA build receipt](raw/multisequence-integration-build-v1.json). Build receipt는 Rust 구현 commit `50e91bd3a233a3277d6d5eef372ddcc96e1bb808`, GPU test receipt는 테스트 파일을 추가한 위 최신 commit을 가리킨다.

남은 통합은 multi-row graph 자원 ledger/binder 및 enqueue, packet staging, scheduler reservation과 bulk result commit이다. Rust wrapper 자체는 동기식 eager 실행 경로를 먼저 연결했다. 이를 serving의 최종 실행 방식이나 높은 concurrency 성능 달성으로 간주하지 않는다.


## Strided GEMM 그래프 수명 경계 연결

현재 원격 snapshot은 `38fba1183142581301156d6fbbc5f55e713ae5da`다. Native에 명시적인 strided graph entry와 상태 구분을 추가했고, Rust `BorrowedStridedGemmGraph`/resources 타입이 plan·stream·입출력을 close/drop까지 빌린다. 기존 lease 획득·해제 및 replay 절차를 재사용한다. 단일 요청 graph entry는 여전히 strided plan을 승인하지 않는다.

GPU Rust 경계 검사 20개 조합에서 60 replay를 통과했다. 출력·padding 및 명시적 close/drop 후 정리를 확인했다. 기존 canonical borrowed GEMM 및 selected GEMM graph GPU 테스트 각 1개도 통과했다. CUDA 비활성 CPU library 86개도 통과했다.

이 경계는 standalone GEMM graph이며 전체 aggregate decode owner가 아니다. Borrowed owner가 살아 있는 동안 Rust 입출력을 외부에서 다시 빌릴 수 없으므로 이번 Rust 검사는 고정 입력의 반복 실행이다. 입력을 바꾸는 raw native graph 6,534회 검사는 이전 별도 증빙이다. 이 둘을 합쳐 실제 scheduler 전이를 검증했다고 주장하지 않는다.

[변경 patch](multisequence-strided-graph-owner-v1.patch), [GPU receipt](raw/multisequence-rust-strided-graph-v1.json), [기존 graph 회귀 로그](raw/multisequence-legacy-gemm-graph-regression-v1.log). 다음은 aggregate ledger의 strided plan 예약과 명시적 binder, multi-row KV/attention·pointwise 및 packet staging 연결이다.


## Aggregate 자원 예약 연결

원격 snapshot `617bbd5497e43ee934dee24384f1d85e2d35cb78`에서 native aggregate ledger가 strided plan을 예약하고, Rust `reserve_with_strided`가 해당 부모의 독점 borrow를 예약 종료까지 유지한다. 기존 single-row record 메서드는 기존 plans 목록만 참조한다. 새 내부 binder는 expected batch/N/K/stride 및 기존 부모/lease 검사를 사용하고 실패 시 출력 state를 비운다. 이 binder는 컴파일됐지만 실제 aggregate decode recording을 통한 실행 검증은 아직 하지 않았다.

Native GPU 자원 테스트 3개가 통과했다. 새 테스트는 20개 shape/batch/stride 조합에서 중복 plan 예약, 사용 중인 plan close 거부, busy plan 이전에 획득한 stream/buffer의 rollback, Drop 후 재예약을 검사한다. Rust GPU 통합 테스트도 20개 조합에서 safe aggregate 예약 close/Drop 후 기존 eager·60회 graph replay를 통과했다. 로컬 CPU library 86개가 통과했다. 두 원격 로그를 복사하고 SHA256을 대조했다.

[검증 기록](multisequence-strided-ledger-v1-validation.json), [patch](multisequence-strided-ledger-v1.patch). Native 검사 snapshot은 `fd223c97cc82c5df0178d74a2d355e2f9482925b`, Rust 검사 snapshot은 위 최신 commit이다. 원격 격리 checkout은 검사 후 clean이다.

이번에는 timed serving benchmark를 실행하지 않았고 Blender를 종료하지 않았다. 다음 필수 통합은 새 binder를 사용하는 전체 다중 행 decode DAG, descriptor packet staging, KV/attention·pointwise 연결과 scheduler bulk commit이다. 이 영역들을 연결한 뒤 모델 correctness와 matched serving benchmark를 수행한다. 목표는 계속 진행 중이다.


## 전체 30-layer 다중 행 numerical DAG 연결

현재 원격 snapshot `4c3be11bc5caf8954c0a9971a486f80b937bc4b3`에 고정 SmolLM2 decode bucket2/4의 전체 graph recorder를 추가했다. 검증된 prototype의 attention/precise 커널을 별도 translation unit으로 옮겼고, attention만 기존과 같은 fast-math를 사용한다. Packed QKV → 행별 RoPE/KV write/attention → output projection → residual/norm → packed gate/up/SwiGLU → down/norm을 30회 연결하고 strided LM head와 일괄 completion을 기록한다. GEMM binder가 이번 DAG에서 실제 호출된다. 입력은 1280-byte packet, 출력은 greedy640/full-logits(640+bucket*98304), 기존 transfer API는 max(input,result) 길이로 staging한다.

Native에서 부모 membership·alias·shape를 확인하고 replay 전에 packet 구조, 행별 위치/블록/valid count, cross-row 중복 KV ID, 태그/cookie/slot 중복 및 zero padding을 검사한다. **이 구조 검사는 scheduler reservation 권한을 증명하지 않는다.** Scheduler의 실제 예약과 owner 세대/서명/replay identity를 대조하는 runtime adapter는 아직 연결되지 않았다. 현 단계는 low-level numerical recorder이며 서버에 활성화하지 않았다.

CUDA 전체 build/link가 통과했다. 최종 GPU 검사에서 zero-weight 모델 DAG의 4개 bucket/output-mode 조합, 12 replay, 12 malformed-packet 거부, 4→3→4 전이, 결과 slot/활성 행/padding 검사가 통과했다. 실제 모델 크기의 RoPE 테이블을 받을 수 있도록 최소160 positions와 cos/sin 동일 extent를 검사하며, smoke에서는161 positions를 사용했다. 기존 strided GPU 20개 조합/60 replay도 통과했다. 이 zero-weight 결과는 실제 모델 logits/KV parity를 증명하지 않는다. GPU 테스트에는 새 allocation-stats 최종 assertion이 없으므로 allocation accounting 검증으로 확대 해석하지 않는다.

[검증 기록](multisequence-full-dag-v1-validation.json), [patch](multisequence-full-dag-v1.patch), [GPU 로그](raw/multisequence-full-dag-smoke-v2.log). 최종 원격 checkout은 clean이고 실행 중인 검사 세션은 없다. Blender 종료 및 timed serving 측정은 하지 않았다.

다음 연결: 실제 executor의 가중치/packed weights·KV pool·RoPE 부모로 이 recorder를 준비하고 기존 M1 graph와 full-logits/KV를 비교한다. 이후 P128 및 bucket1/2/4 capture catalog를 공통 owner로 묶고 descriptor codec의 실제 reservation authority, scheduler bulk commit, server publication을 연결한다. 이 통합 batch 전체를 실제 serving에서 비교하기 전에는 성능 개선 채택이나 목표 달성을 선언하지 않는다.


## 실제 모델 logits/KV parity 통과 및 수치 경로 수정

최종 원격 snapshot `7fb25b7171144a840dfed8ade0c988ad66e2db19`에서 실제 SmolLM2 가중치를 연결한 다중 행 decode 검사가 통과했다. 각 요청은 기존 P128 그래프로 prefill하고, 단일 요청 M1 oracle과 새 bucket2/4를 각각31단계 실행했다. Bucket4는4→3→4 활성 행 전이를 반복한다. 합계62 replay,176 active output rows의 모든 BF16 logits(8,650,752 words)와 greedy ID가 일치했다. 두 bucket의 최종 전체 물리 key/value pool도 byte-identical이고, 종료 시 context allocation stats가0이었다. 명시적 non-contiguous mapping을 사용하며, 프롬프트는 요청별로 다른 deterministic vocabulary token 배열이다.

첫 시도는 bucket2/step15/row1/position143에서48,918 logits words가 달랐다. 진단 버전에서 key/value의 첫 불일치 layer는6이었다. 분리된 RoPE+attention prototype을 사용한 경로를 기존 M1의 결합 RoPE/attention 수치 경로로 교체해 이를 해소했다. 행별 입력/출력 offset과 packet row view, inactive CTA guard를 추가하고 기존 연산 본문과 helper를 재사용했다. 어느 개별 부동소수점 연산이 차이를 만들었는지까지 특정하지는 않았다. 실패 로그v1/v2를 보존했다. 수정v3와 사용하지 않는 분리형 attention/RoPE를 제거한 최종v4 모두 통과했다.

최종 소스의 zero-weight greedy/full-logits DAG 회귀도4구성/12replay/12잘못된packet거부를 통과했다. CPU riley-cuda library86개 및 rustfmt도 통과했다. 원격 model.safetensors와 tokenizer.json의 SHA256을 직접 확인했고, 복사한5개 로그의 SHA256을 대조했다. [검증 기록](multisequence-real-model-v1-validation.json), [patch](multisequence-real-model-v1.patch), [최종 모델 로그](raw/multisequence-real-model-v4.log).

이 결과는 같은 소스의 M1 profile2 oracle과의 범위가 정해진 parity다. 직접 vLLM 비교, scheduler 취소/예약 검증, production serving 결과가 아니다. 현재 테스트는 prefill owner를 정상 close한 뒤 같은 실제 부모를 다중 행 owner에 연결한다. Hot serving에서 이런 재기록을 수행하는 경로로 채택하지 않는다. 다음 작업은 P128/M1/N2/N4 capture catalog를 한 owner로 묶고 실제 reservation/codec/scheduler bulk commit을 연결하는 것이다. timed benchmark와 Blender 종료는 이번에도 하지 않았다. 목표는 active다.


## 공통 ledger의 cold decode catalog 및 실제 bucket 전환

현재 원격 snapshot `16b71a5950a3ddfb18dc383c043c97e0bcf20e23`다. Native owner에 추가 N2/N4 graph/exec를 보관하는 두 catalog slot을 추가했다. index0은 기존 graph,1은 appended N2,2는 appended N4다. 각 entry의 staging과 전송 길이/출력 mode는 cold 고정이다. 공통 자원 ledger의 모든 부모는 모든 graph가 파괴될 때까지 유지된다. Capture 실패의 부분 graph도 slot에 남겨 close가 소유하고, terminal/completion-unknown 상태는 binding 복원으로 지워지지 않는다. 첫 실제 replay 제출부터 catalog가 sealed되어 이후 capture 추가를 거부한다.

Replay/read는 선택한 entry의 binding을 동기 호출 동안 사용하고 원래 binding을 복원한다. 완료 상태는 마지막 성공한 entry에만 해당한다. 일반 read_transfer로 다른 entry의 stale 결과를 읽는 경우도 거부한다. Rust borrowed owner와 retained owned owner 양쪽에 catalog replay/read 경로를 연결했다.

실제 모델 검사 snapshot은 `44c626aa448aa162dd166b526488378867ff074f`다. 같은 owner 안에서 N4와 N2를 준비한 뒤 active4→3→2→4를 반복했다. N2 단독 case와 합계62replay/162행에서 전체 BF16 logits와 greedy ID가 M1 oracle과 같았고, 전체 physical KV bytes 및 최종 allocation stats0도 통과했다. 다른 index의 completion read, 일반 stale read, 중복 cold append 거부를 확인했다. 이후 최신 commit은 owned forwarding 메서드14줄만 추가했으며, 그 상태의 native 자원 GPU 회귀3개도 통과했다. CPU library86개는 catalog 구현 과정에서 통과했다.

[검증 기록](multisequence-catalog-v1-validation.json), [patch](multisequence-catalog-v1.patch), [모델 로그](raw/multisequence-catalog-model-v1.log). 두 원격 로그의 로컬 SHA256을 대조했고 remote checkout은 clean이다. 실행 중 검사 세션은 없다.

아직 실제 runtime의 P128/M1과 N2/N4를 한 번의 cold prepare로 묶은 serving owner는 구현되지 않았다. 이번 모델 검사는 prefill owner close 이후 N2/N4 공통 owner를 준비한다. 다음은 기존 prepare_full_decode_with_prefill의 부모 수집을 확장하여 P128/M1과 추가 N2/N4를 동시에 예약·기록하고, 실제 scheduler reservation/codec/bulk commit과 연결하는 작업이다. 성능 측정, Blender 종료, 목표 완료 선언은 하지 않았다.


## P128/M1/N2/N4 공통 cold owner 통합

최종 snapshot `fb22505950dab470fb0090e9525f040219f9214c`에서 기존 runtime prepare_full_decode_with_prefill의 부모 수집을 확장했다. MultiDecodeParents가 N2/N4 scratch·10개 strided plan·두 pinned staging을 cold 할당하고, 원래 executor 가중치/packed weights/KV/RoPE와 함께 하나의 aggregate ledger로 예약한다. 기존 P128/M1을 기록한 직후 N2/N4를 append하며, 첫 request 처리 전에 전체 catalog가 준비된다. 기존 공개 into_owned_decode_graph는 추가 catalog 없이 그대로 준비하고, catalog 생성 entry는 아직 crate 내부에서만 사용한다.

실제 모델 검사에서 oracle은 기존 owner, candidate는 새 공통 owner다. Candidate는 초기 P128부터 같은 owner를 유지한다. 4→3→2→1 및2→1 decode 전이 중 step16에서 마지막 요청의 새 P128을 실행하고 decode를 재개한다. 합계62 decode replay/126 decode rows,8 P128 replay가 full-logits parity를 통과했고 최종 전체 physical KV와 allocation stats0도 통과했다. 로그의 replays=62는 decode만 센 값이며 전체 candidate graph launch는70회다. 요청별 prompt/block mapping은 테스트 코드의 결정적 fixture다. Scheduler cancellation/새 reservation 권한을 증명하는 검사는 아니다.

추가 scratch/staging의 retained footprint를 기존 M1/P128 registry 보고량에 포함했다. 이 과정의 v2는 잘못된 생성 경로에 적용된 변수 참조 때문에 컴파일 단계에서 실패했고 GPU 검사는 실행되지 않았다. 범위를 수정한 v3에서 전체 검사를 다시 통과했다. [검증 기록](multisequence-common-owner-v1-validation.json), [patch](multisequence-common-owner-v1.patch), [최종 GPU 로그](raw/multisequence-common-owner-v3.log). 로컬로 복사한3개 로그의 SHA256을 대조했고 remote checkout은 clean이다.

다음은 실제 PreparedLlamaIteration/예약 cookie·block 소유권을 descriptor codec의 기대값과 연결하고, 다중 결과 전체 검증 후 scheduler의 bulk commit과 서버 publication으로 전달하는 작업이다. 현재 검사에서는 crate-private owner.graph로 catalog를 호출한다. Production adapter에서는 이 경로의 output readiness/poison, 전체 catalog signature와 replay identity를 별도로 관리해야 하며, 기존 M1 signature가 N2/N4의 scheduler 권한을 나타내는 것으로 취급하면 안 된다. Serving 활성화 및 benchmark는 아직 수행하지 않았다. Blender는 종료하지 않았고 goal은 active다.


## Scheduler 정책 적용 및 live execution authority

최종 snapshot `e63e976b0bc806a06258e67b0a89593fb0f130da`에 기존 batch9 prototype의 CompletePrefill128DecodeN 정책을 적용했다. 기존 Scheduler::new는 General 정책을 유지하며 새 정책은 명시적으로 선택해야 한다. 이번에는 server 설정을 전환하지 않았다.

새 Scheduler::authorize_execution은 실제 in-flight 예약을 검증한 뒤 전달된 plan의 iteration/request/kind/target/slot, 입력 토큰, physical KV ID와 valid count를 live scheduler 상태와 대조한다. AuthorizedExecution이 scheduler의 immutable borrow를 보유하므로 그동안 mutable settlement/cancel/새 plan 발급을 할 수 없다. 반환 rows에는 실제 committed length, generated index, output limit, 예약 table이 있고, block_owners에는 off-batch resident 요청도 포함된다. 객체를 drop해도 commit/rollback하지 않는다. Owner-issued cookie와 catalog generation/digest는 아직 runtime adapter에서 연결해야 한다.

로컬 전체 scheduler129개+doctest1개가 통과했다. 변조 토큰/바뀐 KV table/종료된 plan 거부 및 off-batch 소유권 포함을 새로 검사했다. 이후 정책 테스트의 모든 발급 plan과 지연 취소된4행 plan에도 authority 검증을 적용했고14개 정책 테스트가 통과했다. 원격은 구현 snapshot의 library31개+policy14개, 최신 test 추가 snapshot의 policy14개가 통과했다. 개발 중 pool getter와 테스트 생성자 인자 컴파일 오류를 수정했으며 최종 local/remote 명령은 성공했다. [검증 기록](multisequence-scheduler-authority-v1-validation.json), [patch](multisequence-scheduler-authority-v1.patch).

아직 execute_llama_iteration_graph와 server가 이 authority를 사용하지 않는다. 다음은 scoped authority를 받는 GPU adapter가 실제 reservation 데이터로 descriptor를 구성하고, catalog owner의 readiness/poison과 전체 결과 검증을 거쳐 기존 DownloadedLlamaIteration/complete_iteration으로 전달하도록 연결하는 작업이다. 이번 결과는 CPU scheduler 계약이며 native GPU correctness나 serving 성능 결과가 아니다. Remote checkout은 clean, 실행 중 테스트 세션은 없고 Blender 종료/성능 측정은 하지 않았다.


## Live authority → descriptor 기대값 연결

Snapshot `3a2607138c8a306268fdcf92cf1a232239192a4e`에서 기존 descriptor prototype을 scheduler 내부 module로 옮기고 golden binary fixtures 및22개 codec 검사를 유지했다. AuthorizedExecution::descriptor_expectation은 actual scheduler row의 token/target/committed length/generated index/output limit/slot/table 및 off-batch block owners를 기대값으로 구성한다. Owner generation/digest/replay/cookies는 retained runtime owner가 제공해야 할 입력이며, 아직 runtime owner와 직접 연결하지 않았다. 이 메서드는 crate 내부에서만 호출할 수 있다.

새 CPU 통합 검사는 실제 Scheduler가 발급한 예약을 authorize한 뒤 P128 packet을 encode/decode하고, 다른 pool geometry와 누락된 physical block ownership을 거부한다. 이 검사의 owner identity/cookie는 명시적인 CPU fixture 값이며 GPU owner identity 증거가 아니다. Local/remote library54개(기존31+codec22+연결1)가 통과했다. [검증 기록](multisequence-descriptor-authority-v1-validation.json), [patch](multisequence-descriptor-authority-v1.patch). Remote log를 복사하고 SHA256을 대조했다. Checkout은 clean이고 실행 중인 검사 세션은 없다.

다음은 retained runtime owner의 process-unique generation, catalog digest, replay/cookie counter와 별도 output readiness를 마련하고 scoped-authority GPU adapter가 이 descriptor를 제출하도록 연결하는 것이다. M1/P128 compatibility view와 multi result 형식 변환, 전체 결과 검증 후 실제 scheduler complete_iteration 및 codec settle_success 연결이 남아 있다. 이번에는 GPU 실행이나 serving benchmark, Blender 종료를 하지 않았다.

## Live scheduler authority integration (807b615)

Moved the shared CPU descriptor codec into riley-runtime and bound generation, catalog digest, replay IDs and cookies to the retained GPU owner. Added scheduler adapter accepting only live AuthorizedExecution. Real model test passes 110 full BF16 logit rows against retained M1, with decode active sizes 1/2/3/4 and four actual scheduler completions. Runtime CUDA and scheduler CUDA builds passed; local scheduler library tests: 32 passed. Receipt: multisequence-live-authority-v1-validation.json. This is correctness evidence, not a serving performance result. Server activation, failure/cancellation integration and matched serving benchmarks remain outstanding.

## Server integration (86258e3)

Connected multi owner selection, scheduler policy, output staging capacity, live authority, post-commit confirmation and owner-first shutdown to the HTTP worker. CLI supports capacity 2/4 with P128, context160 and output <=32. First C2 HTTP test failed because owner capacity4 required40 KV blocks; fixed owner capacity/catalog restriction from physical coverage. Failure log retained in raw/multisequence-server-smoke-v1. Final cuda,server release build passed. Smoke v2: capacities2/4, each16 requests at client concurrency8 (8 SSE and8 nonstreaming); all32 exact reference texts and expected token counts/finish contracts, both child servers graceful exit0. This is correctness-only, not timing or diverse-corpus qualification. Receipt: multisequence-server-integration-v1-validation.json.

## Cancellation and strict HTTP token qualification (1f39e35)

Actual-model scheduler tests: two passed. Cancel after GPU execution and before commit, then submit replacement; assert physical KV page reuse. All119 full-logit rows match M1 exactly; normal110-row scenario also passes. Added strict HTTP token client qualification:64 responses total across capacities2/4, half streaming; input/output token IDs, text, finish and usage all match pinned Hello P128/O32 reference. Each phase observed actual overlap8 and both servers exit0. No exclusive GPU timing performed. Candidate build/source freeze and next measurement session preparation remain.

## Exclusive serving screen round17

Frozen candidate33c530ed, two reversed-order repeats at client concurrency1/4/8,128 retained streaming requests per lane plus16 warmup, pinned P128/O32 token reference. C1 candidate990.8token/s vs baseline918.9/vLLM890.6 (+11.25% vs vLLM), TTFT5.157ms vs7.259ms, TPOT0.868ms vs0.898ms. C4 candidate1632.8token/s vs baseline934.4; C8 candidate1667.4 vs933.9. However candidate TPOT2.160/2.323ms vs baseline0.955ms: throughput gains do not meet joint latency target. vLLM C4/C8 failed exact token-reference warmup and have no comparable retained phase. Results are screening only, fixed single prompt, two repeats and128 samples; no high-concurrency stability/winner claim. Next: profile multi decode execution/transfer/host overhead and diagnose vLLM batch-dependent token differences. Blender17 restore verified live PID/start/ports3772828/85261398/9876,3772970/85261447/9911,3773079/85261499/9887 with command and private vendor runtime checks.

## Completion batch profiling and implementation (0aed6e2)

Nsight Systems2024.6.2 CUDA graph node trace successfully captured frozen server with8 strict-token C4 responses. This probe ran with Blender alive and is diagnostic, not exclusive serving timing. Kernel aggregate: GEMV46.8%; multi_completion117us average; serial per-row argmax34.9us average. New worktree multisequence-completion-source-v2 preserves frozen candidate source. Implemented parallel full-logit copy/zero fill with independent metadata CTA and single batched launch of unchanged exact argmax row kernel. Initial reuse of old target failed CMake source-cache identity; rebuild is using independent multisequence-completion-target-v2. Runtime correctness, updated trace and matched serving benchmark remain required before accepting.

Completion batch v2 validation: independent target rebuild and server build passed. Actual model229 total full-logit rows passed (110 normal +119 cancellation/reuse). Nsight repeated8 strict C4 HTTP responses passed. multi_completion average117.03us ->3.846us; argmax launch count258 ->73 for same workload, aggregate8.998ms ->2.510ms, per-launch~34.9us unchanged. These traces ran with Blender alive and include prefill/startup; they establish mechanism improvement, not serving acceptance. Candidate v2 binary c4cf44f9d9e437d47646183aefb6c6c8ad3695109487b13f2bcd20347b9bd2b5 frozen separately. Next required work: exclusive paired serving v1/v2 and baseline/vLLM comparison. No Blender pause occurred this turn; round17 restored successors remain current.

## Completion batch exclusive serving screen round18

Two reversed-order repeats, client concurrency1/4/8,128 retained requests per process plus16 warmup, four lanes: baseline/previous multi/new completion/vLLM. New vs previous multi: C4 throughput1781.5 vs1629.1token/s (+9.35%), TPOT1.953 vs2.164ms (-9.77%); C8 throughput1810.0 vs1661.4 (+8.94%), TPOT2.119 vs2.330ms (-9.05%). C1 unchanged. Keep completion optimization in working candidate; broader serving acceptance remains pending. C1 vLLM throughput887.8 vs new990.6 (+11.57%); TTFT7.578 vs5.150ms, TPOT0.896 vs0.868ms. vLLM C4/C8 strict token warmups fail again, so no comparable retained phases. P99 samples are descriptive only; no stability claim. Next: multi GEMV/LM-head profiling and batch-dependent vLLM reference divergence analysis.

Round18 restoration revalidated live: PID/start/port3835127/85327210/9876,3835188/85327260/9911,3835256/85327311/9887. Original commands, GUI/runtime environment and pinned vendor maps verified. No measurement process remains owned/running.

## vLLM batch invariance diagnostic

Installed vLLM0.27.1 source has VLLM_BATCH_INVARIANT selecting linear_batch_invariant and attention num_splits1. Four fresh servers,8 observed responses each: default C1 has1 distinct sequence and8/8 match pinned reference; default C4 has4 sequences and2/8 match. Invariant C1/C4 both produce exactly the same single sequence, diverging from pinned default-C1 at output29 (638 instead of253). Input token arrays identical. This is concrete evidence of batch-sensitive vLLM arithmetic, not proof identifying a single faulty operator. Invariant option changes multiple kernels and is not substituted for default vLLM performance. Existing HF teacher-forced common-prefix BF16 and FP32 receipts also choose638 at output29, but do not isolate cached vLLM arithmetic. No correctness acceptance gate relaxed. Probe ran with Blender alive, no timing claim; all4 servers closed. Evidence raw/vllm-batch-invariance-probe-v1/analysis.json.

## LM-head alternatives (diagnostic; no production change)

Actual head weights and18 existing guarded cases per setting: custom options74/75/76/88/90/91 admit but mismatch M1 BF16 logits; only existing89 exact. Keep89. Splitting N49152 into1/2/4/8/16 chunks using same89 maintains exact logits for all18 cases each. Nsight standalone batched GEMV aggregate B4 (12 base cases) grows1.249/1.310/1.386/1.568/1.786ms respectively, so no evidence to adopt splitting. This standalone fixture trace is nonexclusive and not representative of whole-model cache state; per-head times differ materially from serving trace. No serving claim or engine change from these probes. Current accepted working candidate remains completion-v2 (0aed6e2). Next investigate whole-model memory traffic/cache behavior and GEMV reduction-preserving reuse opportunities. Raw head-options-v1, head-chunks-v1-receipt.json, head-chunks-profile-v1-receipt.json.

## Head memory-history pressure experiment

Nsight Compute counters unavailable (ERR_NVGPUCTRPERM); no host changes made. Created guarded native probe with128MiB volatile GPU read before each head. This changes memory-access history but does not directly measure cache hit rate. All18 cases for each1/2/4/8/16 split keep exact logits and close resources. With pressure B4 total GPU head kernel time over12 cases:2.890/2.955/2.931/2.093/2.031ms; B2 over6 cases:.742/.765/.786/.680/.620ms. Unsplit B4 average241us reproduces serving trace~240us, compared with standalone no-pressure104us. Thus prior standalone rejection of all chunking was premature for the serving memory-history condition: 8/16 splitting reduces pressure-case B4 kernel sum~28/30%. Candidate direction:16 head partitions retaining custom89 and original dot-product arithmetic; must integrate cold layouts into retained plan, validate actual full-model logits/KV/cancellation and measure paired serving. These probes are nonexclusive, kernel-time-only diagnostic; no serving improvement yet claimed. Frozen production candidate remains completion-v2.

## Retained head partition integration and round19

Source189fc3df7b31e2641c913a2eedcdb920e9f5ba7e, frozen binaryb2eb22727065e492c3f92ec5165766fb2c6c23d90ddae759cdfe4e377d1bf0c5. Cold strided head layouts now cover16 N slices; logical full buffer extents/strides and exact89 reduction preserved. Graph and eager execution both traverse prepared slices. Actual full-model normal/cancellation229 rows pass; strided20configs/60graphreplays pass (initial missing-feature invocation ran0 tests, corrected cuda-feature run passes);8 strict HTTP responses pass. Exclusive round19 two reversed repeats per C1/4/8, four lanes128retained+16warmup each. V3 vs V2 C4throughput1844.9/1782.3 (+3.51%), TPOT1.874/1.953ms (-4.01%); C8throughput1878.1/1809.7 (+3.78%), TPOT2.042/2.124ms (-3.84%). Keep V3 as working candidate. C1 unchanged,~11%throughput over vLLM; vLLM C4/C8 token failures still prevent retained comparison. Goal remains incomplete; stability/diverse workloads and latency targets unproven.

Round19 live restoration verified:3974835/85480054/9876,3974945/85480103/9911,3975020/85480153/9887 (PID/start/port), original commands and pinned vendor runtime maps. No measurement server remains running.

Next profiled candidate: argmax remains~35us per batch in node trace (also present on M1). Explore parallel vocabulary partitions with exact ordered-value/token tie reduction, explicit nonfinite propagation and signed-zero tie handling, reusing owned8-byte result storage if feasible. Require meaningful nonfinite/tie/padding tests, full model logits/token parity, and fresh serving baseline comparisons before acceptance. No implementation yet.

## Partitioned argmax candidate V4

Source64cbb36d3f4588fade7e0a40739be6504f97e47c, binaryf43a6f92f0a31f2d5699b4eeede0c17a0458cb4e419565670dfbc6877cb2591a frozen asmultisequence-candidate-v4. Fixed49152 vocabulary now uses memset of existing8-byte/row result,32 CTA partitions per row with exact BF16 ordered key and reversed token ID, then result finalization. Nonfinite dominates with error sentinel; signed zeros share key. No scratch GPU allocation introduced. Shared primitive code included in owner catalog digest. Independent CPU reference240 cases cover ties/zeros/nonfinite/boundaries/random finite values for1/2/4rows in eager and graph replay; allpass. Actual model229 logit rows and cancellation/reuse pass;8strict HTTP responses pass. Nsight partition kernel2.024us+finalize0.923us average vs prior~35us per argmax launch; reset and inter-kernel gaps excluded, no serving improvement claim yet. Next: exclusive round20 paired serving measurement against V3/baseline/vLLM. Blender remained untouched this turn; round19 successors remain latest.

## Argmax V4 exclusive screen round20

Two reversed-order repeats atC1/4/8,128retained+16warmup requests per lane; baseline,V3,V4,vLLM. V4 C1 throughput1021.0 vs V3 989.0 (+3.24%), TPOT0.8374 vs0.8683ms (-3.56%). Relative to current vLLM875.2token/s,7.415ms TTFT,0.8973ms TPOT: throughput+16.66%,TTFT-31.05%,TPOT-6.67%. Throughput target exceeded in this short fixed-prompt screen only; no final success claim. V4 C4/C8 throughput1870.1/1913.3 vsV3 1842.5/1879.4 (+1.50/+1.80%); TPOT1.848/2.007ms vs1.881/2.043 (-1.71/-1.79%). Keep V4 working candidate. vLLM C4/C8 warmup still differs from exact C1 reference, no retained comparison. Need expanded representative corpus, longer repeatability/tail and numerical qualification, plus furtherTPOT work.

Round20 restoration verified live PID/start/port4042549/85552763/9876,4042626/85552815/9911,4042695/85552869/9887, including original command and pinned vendor runtime checks. All measurement processes terminal.

## Diverse synthetic P128 numerical qualification

Twelve synthetic prompts cover summary/code/arithmetic/planning/comparison/extraction/science/editing/logic/SQL/story/classification. Inputs are repeated/truncated to exactly128 tokens, outputs8/16/24/32; this is not a production corpus or variable-length prefill qualification. Original baseline C1 supplied raw references. Frozen V4 matches12/12 at capacity1 and48/48 at capacity4/client8, alternating streaming/nonstreaming; actual overlap8 and all server exits0. Default vLLM0.27.1 matches12/12 at C1 and46/48 at C4; two SQL responses differ at output index4, while prompt tokens match. These are nonexclusive correctness/observation runs with Blender alive; no timing claim or relaxed correctness gate. Raw diverse-p128-correctness-v1 and diverse-vllm-observation-v1 retained.

Round21 mixed synthetic screen in progress: identical balanced corpus,24warmup+240retained, two reversed lane orders atC1/4/8. Per-request corpus identity, token count, common wall interval, actual overlap and numerical match are accounted separately. vLLM numerical divergences are retained only as observations; incomplete transport/protocol phases stop refill. No performance acceptance follows from observations.

## Mixed P128 exclusive screen round21 completed

All18 retained phases completed (4320 responses),24warmup+240retained each,2 reversed lane orders. Candidate1440/1440 retained references match; baseline1440/1440. vLLM C1 480/480, C4 464/480, C8 458/480 match. All responses passed protocol/transport checks; divergent vLLM high-concurrency phases remain numerical observations, not correctness-qualified performance comparisons. Medians of two run summaries: C1 candidate940.8token/s vs vLLM803.6 (+17.08%), TTFT5.248/7.337ms (-28.48%), TPOT0.8296/0.8870ms (-6.46%). C4 candidate1560.3 vs observed vLLM1643.0 (-5.03%), TPOT2.272/1.740ms (+30.57%). C8 candidate1606.8 vs observed vLLM2375.6 (-32.36%), TTFT53.332/19.046ms. Candidate active capacity4, vLLM capacity8 at C8; this configuration limitation is explicitly retained. Candidate C8 E2E P99 empirical132.29ms vs observed122.13ms; only480samples/lane, not stability proof. Goal remains unmet.

Round21 restoration revalidated live PID/start/port4127250/85646254/9876,4127309/85646304/9911,4127375/85646356/9887. Same commands, GUI/runtime and pinned vendor maps checked. All benchmark servers terminated.

Existing V4 graph-node copy trace separates startup from replay:61 N4 full-logit D2H copies393856bytes average17.76us;2 N2 copies197248bytes average10.08us. Full-logit transfer is avoidable for GPU greedy but this magnitude alone cannot explain the mixed C4 TPOT gap. Next diagnostic is a48-response mixed-corpus Nsight trace, preserving strict tokens; investigate decode GEMV cost, prefill interruptions and row occupancy/capacity before choosing the next meaningful optimization batch. No source changes or performance acceptance in this measurement expansion.

Mixed Nsight diagnostic completed:48/48 strict streaming references, serverexit0, frozen V4 SHA unchanged. Graph packet segmentation identifies48prefill and245decode submissions. Prefill GPU span sum232.484ms,median4.800ms; decode354.024ms,median1.465ms. Decode kernel sum317.594ms, GEMV226.575ms (71.34%); graph copies4.074ms total. This nonexclusive instrumented trace includes profiler overhead and does not replace serving measurements. It supports prioritizing prefill computation and batched GEMV/occupancy over small transfer savings; scheduler fairness changes alone cannot remove computation. Raw mixed-p128-nsys-v4 has receipt, per-request rows, stats and graph-stage-analysis; full nsys-rep/sqlite retained remotely. Next optimization batch should preserve full-model/cancellation correctness and remeasure both fixed and mixed workloads.

## Prefill projection load/loop batch V5

Previous goal turn made progress:4320-response mixed screen and48-response graph trace established a high-concurrency gap and substantial prefill/GEMV cost. Explored vector BF16 loads, weight reuse across32/64 rows, and loop unroll controls in25 shape/variant native cases. Every output matches the original primitive, but larger row tiles mostly regress. Isolated synthetic hot-cache timings are not serving acceptance. Working source V5 b9a0db847ebaf53a18d3af09eaf284f0f4010d55 adds packed32-bit loads and unroll1 depth loops to four fixedP128 projections, preserves original N192K/V path, exact MMA/reduction/BF16 round order, and includes precise source in catalog digest. Frozen binary7b9e3bed69b468638ba789e6e8ec3a39bef11d6bd4b0b62aa4ab07e1409ca615; build65038aaa03b1a261201d509dbbd48b444b5221e588b985afef3839e71e4d808d.

Actual model normal/cancellation229 logit rows pass. P128 parity:3prompts,96whole-KV snapshots pass; retained6requests/899slow+137fast replays,42invalid-shape/stage cases, cancellation and zero allocation checks pass. The runtime filter ran2targeted unit tests; unrelated integration binaries had0selected tests and are not counted. Mixed strict48HTTP responses pass, serverexit0. Whole-model trace benefits are smaller than synthetic probe: gate/up17.55->13.79us, down31.19->30.81us, but Q10.94->12.48us and O10.51->12.30us regress. Current V5 is an unaccepted experimental candidate pending exclusive round22; evidence suggests revisiting per-shape selection after serving results.

## V5 exclusive mixed screen round22 completed

24retained phases complete,5760responses total; V5,V4,baseline each1440/1440 exact. vLLM C1 matches480/480,C4 459/480,C8 455/480; high-concurrency rates remain observations only. V5 vs V4 throughput C1+0.779%,C4+0.999%,C8+1.435%; TTFT-2.643%,-2.182%,-0.887%; TPOT-0.170%,-0.782%,-0.246%. Two reversed repetitions agree on small throughput improvement. V5 is retained as an interim experimental candidate, not a release/stability qualification. C4 empirical TTFT P99 rose11.953->12.681ms despite median improving;480samples do not settle tail stability. C1 vs vLLM: throughput945.99/804.61 (+17.57%),TTFT5.127/7.438ms(-31.07%),TPOT0.8297/0.8858ms(-6.33%). C4 throughput1575.8 vs observed1650.6;C8 1627.4 vs2410.1. Full objective remains unmet.

Round22 live restoration verified PID/start/port29154/85746309/9876,29213/85746362/9911,29351/85746413/9887, same command and GUI/private runtime maps. All benchmark and profiling child servers are terminal.

Next meaningful prefill batch: retain benefiting projection choices and revisit Q/O regressions together with attention computation reuse. Source shows P128 attention duplicates each query across all16MMA rows, while48-prefill trace spends~54ms in attention kernels. Explore multiple independent causal queries per MMA while retaining depth accumulation, MODE6 denominator order, BF16 probability conversion, and exact causal-tail behavior. This is a candidate direction, not an implemented or accepted optimization. High-concurrency capacity4 limitation and batched decode GEMV remain separate major gaps.

## P128 attention computation-reuse batch V6 in progress

Prior turn qualifies as progress: V5 implementation, full-model checks,5760-response exclusive screen and restored jobs. Current V5 source was revalidated clean. Native attention prototypes compare2/4/8 query rows per MMA with original P128 operator. Shared-memory padding alone gives little gain; fused score maximum and preserved-order probability/denominator work reach only~5% eager primitive improvement. Sharing softmax and splitting output dimensions across4warp groups per head reduces finite-case primitive timing~39.7->21.6us for4queries. Added per-CTA input magnitude guard and unchanged row oracle fallback for NaN/Inf/extreme finite values.96 comparisons (32patterns times3query tiles), including future-token NaN and signed zeros, match all BF16 output bytes. Latest timing~39.6->18.0us uses a changed finite timing fixture; do not attribute this additional gain to the guard or treat these nonexclusive primitive observations as serving acceptance.

Integrated attention_queries<4,4> only for P128/MODE6 into isolated multisequence-attention-source-v6, with invalid-position fallback and exact exceptional-value row oracle. Preserves causal-tail MMA depth count since4-row groups align with16-token depth boundaries. Restores original Q/O projection choices based on V5 whole-model regressions; retains benefiting gate/up/down load/loop variants. Actual-model scheduler and P128 whole-KV/reuse checks are running. No serving acceptance, no Blender pause yet.

V6 integration validation: source79831bec9508ddb22142e03c18b5d1020a48e43a; binary1b0c1114ecf85d0da61eeadc2787281ab3394bd6396fd7b2b88eba23f0266d99; buildf01f37ad1e4f8c2c944ef6bb3e0ada070f9054ad463995f31cc827f38cd1a166. Scheduler actual-model229 full-logit rows pass. P128 tests pass96full-KV snapshots plus6retained requests/899slow+137fast replays and42invalid stage/shape cases, with cancellation/reuse and zero hot allocation checks. Mixed48strict HTTP responses pass; serverexit0. Whole-model Nsight attention kernel average37.422us(V5)->17.595us(V6),1440calls each; Q/O restored to original kernels. This is a nonexclusive diagnostic, not serving acceptance. Exclusive mixed screen round23 comparing baseline/V5/V6/vLLM now running under authorized Blender lifecycle helper4a3f127a901e420b1fa2c4e7c1430916860fa56cab47cfda9eb3b7cbe0af56b1.

## V6 exclusive mixed screen round23 completed

24retained phases,5760responses,2reversed lane orders and240retained+24warmup per process. V6/V5/baseline each1440/1440 exact responses; vLLM C1 480/480,C4 466/480,C8 460/480, so high-C vLLM timings remain numerical observations only. V6 vs V5 throughput C1+3.346%,C4+5.869%,C8+5.776%; TTFT-13.568%,-11.086%,-4.178%; TPOT+0.136%,-4.813%,-3.648%. C1 TPOT effectively unchanged; no decode-only improvement claimed. Keep V6 as the current working candidate, not final performance/stability acceptance. C1 vs vLLM throughput980.44/787.72 (+24.47%), TTFT4.420/7.548ms (-41.45%), TPOT0.8291/0.8846ms (-6.27%). C4 candidate1673.3 vs observedvLLM1651.7 but TPOT2.152/1.711ms remains worse. C8 candidate1721.7 vs2397.4 (-28.18%), TTFT50.398/18.604ms. Candidate cap4 versus vLLM cap8 at C8 remains explicit. Sample C8 E2E P99 V5 130.887->V6 122.806ms, observed vLLM116.729ms;480samples per lane are not long-run tail/stability proof. Only synthetic P128 inputs and mixed O8/16/24/32 have been screened. Full goal remains unmet.

Round23 live restoration verified PID/start/port157976/85885249/9876,158026/85885298/9911,158102/85885350/9887. Original command, GUI/private runtime and pinned vendor maps match. All benchmark/profiler child servers terminated. Next prioritize batched decode GEMV cost and capacity4 queueing limit; representative variable prompt lengths, stronger numerical qualification against vLLM and long-run high-concurrency stability are still required.

## N8 native execution prerequisite qualified

Previous turn was progress (V6 implementation, exact full-model validation, matched serving screen). Revalidated frozen V6 source clean. New isolated source multisequence-n8-source-v7 commit520a5dfe90d61eac5af659c5ac19640b15257fb9 admits batch8 in Rust configuration and native strided-plan creation/readiness, retaining the same SM89/cuBLASLt13.1 algorithm metadata, workspace0 and head partitioning. Ownership tests cover30compact/padded N2/N4/N8 cases and90graph replays; dense tests independently compare80N8+40N4 rows with M1 and60graph replays. All exact BF16 outputs, padding/input preservation and close/invalid-span reuse checks pass. Two CPU boundary tests were updated and passed for N8 and adjacent unsupported counts.

N4/N8 synthetic dense Nsight graph spans (us): QKV5.008/8.208, gate-up11.808/22.432, O3.616/5.408, down4.704/7.153, head150.945/271.282. Six observations each, graph-node correlation checked against expected batch gridZ and launch count. Larger batches amortize some work, but gate/up and head scale weakly; no whole-model or serving improvement is claimed. Profiling predates only the subsequent CPU-test-list update, with production code unchanged. Native execution test binary is not a serving candidate.

Current working server remains frozen V6 with max-active4. N8 scheduler/wire/catalog integration is not yet implemented. N8_EXECUTION_EXPANSION.md records the coupled V2 wire geometry (request1792/result-prefix1152), catalog cleanup/identity, authority/KV/output mapping and end-to-end qualification work required next. No Blender lifecycle change or exclusive benchmark occurred this turn. All native test/profiling processes completed. Full goal stays active and unmet.

## N8 full integration V7 validated, round24 in progress

Integrated request/result wire V2, eight-row ownership/codec checks, third native catalog entry and cleanup, N8 buffer/plan preparation, and scheduler/server capacity8. Isolated source b5da54d6d2f2f6e66b0b08e75c460a752547cc09; binary9144515b203cf14ca78d6d08692cdfb507cf70cc54ee8dc234840069054cc082; frozen builda0043f22c02f84ef0991143a0c2a9e03124e061282a14debc0fd5ab1139a4672. Initial codec run had22pass/2fail due obsolete V1 offsets in tests, corrected with preserved failure log. Final24 codec tests and scheduler authority test pass. Scheduler GPU tests cover586 full-logit rows (4-row normal110/cancel119;8-row normal172/cancel185), all decode sizes1..8 and cancelled KV reuse exactly match independent M1 execution. Whole-model manual catalog test covers93replays/269rows, prefill reentry and entire physical K/V pool exactness, zero live allocations at close. Actual N8 streaming48/48 reference responses pass, serverexit0, diagnostic Nsight trace retained.

Exclusive round24 compares original baseline, frozen V6 and V7 plus vLLM under the same corpus and reversed run order; V7 capacity1/4/8 vs V6 capacity1/4/4 at C1/4/8. User-authorized three-job lifecycle helper0907b201459c37e59c05b3dfaa3d3d20ea045871a04f9bc86ceeb7376cd576fc checked prior live successors before pause. Results and restoration remain pending; do not claim V7 serving improvement yet.

## V7 exclusive round24 completed; N8 latency regression blocks adoption

All24 retained phases completed,5760responses. Baseline/V6/V7 each1440/1440 exact; vLLM C1 480/480,C4 463/480,C8 462/480, so high-C vLLM comparison remains observational. V7 vs V6 C1 throughput+0.101%,C4-0.093% (essentially unchanged). C8 throughput1720.869->1918.314 tokens/s (+11.474%), TTFT49.792->10.182ms (-79.551%), but TPOT2.233->3.831ms (+71.585%). C8 E2E P95 122.187->138.143ms andP99 125.955->139.355ms regress. Therefore retain frozen V6 as working candidate and keep V7 as experimental capacity expansion; do not call N8 a qualified optimization or final success. At C8 observedvLLM2426.138 tokens/s,TTFT19.004ms,TPOT2.361ms,P99E2E111.185ms: V7throughput-20.931%,TPOT+62.236%. C1vs vLLM+22.053%throughput,-38.750%TTFT,-6.687%TPOT; this narrow screen still lacks the full target qualification.

Blender restoration live-verified PID/start/port350722/86103656/9876,350796/86103706/9911,350882/86103757/9887. Same command, listening ports, GUI/private runtime and pinned vendor maps pass. All benchmark/profiler children terminal.

V7 diagnostic actual-serving Nsight trace (48exact responses, nonexclusive):48prefill/141decode submissions; prefill medianGPUspan4031.688us,decode2137.589us. Decode kernel total259.061ms, GEMV203.809ms (78.67%); graph copies5.238ms (~1.83%decode span). Capacity8 reduces queueing but current batched GEMV cost makes decode intervals and tails worse. Next meaningful optimization batch should target batched decode projection/head computation: measure shape-specific full-model kernel costs and explore weight reuse across rows plus head partition/launch geometry while preserving exact M1 arithmetic. Compact greedy transfers alone are unlikely to remove the measured gap. Variable prompt lengths, larger concurrency, long stability and independent vLLM numerical qualification remain outstanding.

## Decode GEMV geometry/algorithm investigation after V7

Previous goal turn classified as progress: N8 integration, exact correctness tests, exclusive round24 and live restoration changed the adoption decision. Frozen V7 git source revalidated clean at the start of this investigation. No Blender pause or serving source change this turn.

Actual V7 graph-node trace grouping shows N8 gate/up grid768 consumes79.868ms across3480calls (22.951us average), QKV grid24029.117ms/3480calls, O+down grid14444.175ms/6960calls, head grid19232.225ms/1856calls. Existing head uses16parts, while all other projections are unpartitioned.

Standalone cuBLASLt geometry probe covers420 cases:5shapes x N4/N8 x2finite patterns x column partitions1/2/4/8/16/32 and row groups1/2/4/8 (bounded by batch). All exact BF16 outputs match independently executed M1 with the original shape-specific custom option. Current geometry is fastest for QKV/gate-up/O/down. Head smaller partition counts look faster (N4~97us unpartitioned vs140us16parts;N8~189us2parts vs251us16parts), but this is nonexclusive repeated primitive timing with hot weights, not actual-serving acceptance. Two subsequent pattern runs varied the best head choice between1and2parts; benchmark whole-model cache history before selecting.

Algorithm sweep180cases initially all matched for custom74/75/89, suggesting89 might improve QKV/gate-up. Stronger360case sweep uses four independently hashed BF16 sign/mantissa/exponent patterns (exponents118..131). It finds165mismatching cases, demonstrating the initial dyadic fixtures were insufficient. QKV/gate-up/O custom89 fails N8 full-output parity; reject these substitutions under the current exact arithmetic contract. Original custom74/75/89 per shape stays exact across all tested partitions. Head custom89 partition1/2/16 stays exact. Sources and all960 rawcase rows retained under campaign gemv_*_probe.cu and raw/gemv-*.jsonl; summary raw/gemv-algorithm-dynamic-v1-summary.json. These probes do not cover exceptional values, model/cache history, padding or serving, so no integration or performance adoption follows from them alone.

Next output-path optimization batch: (1) reduce head column partitions with unchanged custom89 arithmetic, selected using real-model trace, (2) carry validated GPU-greedy metadata without unnecessary full-logit transfer, retaining full-logit mode for correctness oracle. This targets measured head launch/work cost and D2H cost together; it is not expected by itself to prove the full vLLM goal. Large batched projection cost still needs an exact-reduction weight-reuse solution or a separately justified numerical correctness standard, not an untested algorithm swap. Working candidate remains V6; V7 remains experimental.

## Output-path V8 integrated; exclusive round25 in progress

Previous goal turn was progress:960 GEMV cases ruled out faster but arithmetically different projection algorithms. Isolated V7 source was revalidated clean before creating codex/output-path-v8. Coupled batch reduces strided head custom89 from16column slices to1 and adds greedy N2/N4/N8 graph entries sharing calculation buffers with retained full-logit entries. Native catalog now6entries; all completion selection/close paths cover them. Greedy graph D2H is1152bytes, staging/request transfer envelope1792bytes; full-logit mode remains available for all buckets. M1/P128 still uses its retained full-result D2H and synthesizes mode-specific validated metadata; no M1 transfer improvement claimed. Scheduler chooses descriptor mode from requested output backend with existing authoritative ownership validation.

First GPU test found a stale full-logit payload-count header in M1 greedy adaptation; fixed to mode value, failure logs retained. Final24codec tests pass. Four existing scheduler tests preserve586full-logit rows exactly. Added alternating full/greedy N8 cancellation/reuse test validates185mixed-mode rows against independent M1 tokens/logits (its old generic console label says full-logit rows, but these185are mixed). Entire physical KV test93replays/269rows passes with zero live allocations at close. Real streaming48/48exact passes, serverexit0. Sourcebd699ea5aa455e19dbb45561918b74173e338dff; binaryd2bfbf586194e7eaabbdfe53783ebe31b759c6aa00ef2110c64a3e01f7f3d736; build782cf3f977176c13bb25633f4b396dafdae41816b5e0fd5f0b333f2f3da40507.

Round25 uses5lanes baseline/V6/V7/V8/vLLM, C1/4/8, two reversed orders and240retained+24warmup each. V6 max4, V7/V8 max8; allsame model/corpus/settings. Explicit V7 lane separates output-path gains from N8 admission change. Lifecycle helper17a19339bacd1c5713c1a86ad551673ef37b5ff04a8d31705500e435269ac3d6 checked round24 successors before the user-authorized stop. Measurements/restoration pending; no V8 performance adoption yet.

## V8 round25 completed; compact output benefits, unpartitioned head regresses in model

30retained phases complete,7200responses total. Baseline/V6/V7/V8 each1440/1440exact. vLLM C1 480/480,C4 471/480,C8 458/480: high-C performance remains observational, not matched-correctness qualification. V8 vs V7 throughput C1-0.108%(unchanged),C4+6.910%,C8+7.724%; TPOT C4-7.014%,C8-9.083%; TTFT C4-3.679%,C8+1.231%. C8 E2EP95 137.869->128.029ms,P99 139.808->132.161ms improve relativeV7 but remain worse thanV6P99124.043ms. V8 vs V6 C4throughput+6.638%,TPOT-7.074%;C8throughput+20.120%,TTFT-82.556%,TPOT+52.292% (capacity4vs8). V8C8throughput2066.802 vsobservedvLLM2387.121(-13.419%);TPOT3.403 vs2.374ms(+43.358%). Full goal unmet; V8 is an experimental improvement basis, not a qualified replacement across all latency/stability conditions.

Round25 restored jobs live-verified PID/start/port465129/86228584/9876,465195/86228634/9911,465310/86228686/9887. Same command/listening port/runtime/vendor maps verified; benchmark/profiler child processes terminated.

Actual V8 diagnostic Nsight reveals an important batch attribution limit.130multi-decode D2H copies are now1152bytes with mean0.835us, compared to full-output transfers before;58M1/P128 still copy98304bytes. But unpartitioned N8 head averages476.475us across116calls vsV7's16parts x17.363us~277.804us per head, while gate/up22.951->23.026us, O/down6.347->6.352us andQKV8.367->8.452us are similar. V8decode kernel sum283.079ms (GEMV227.864ms), medianGPUspan2308.415us; trace is nonexclusive and batches differ slightly140vs141. The hot standalone head timing was not predictive of whole-model cache history. Do not attribute observed serving improvement to head unpartitioning. Compact output also avoids host full-logit validation/copy/allocation by code inspection, but no isolated CPU speedup is claimed.

Next action: retain compact output, re-evaluate head1/2/4/8/16 partitions using actual-model trace, then remeasure selected coupled output path against frozenV8 andV6/vLLM. Head16 is previously full-model qualified; reverting the head part is justified if no other actual-model geometry wins. After output-path selection, exact batched projection weight reuse remains the major compute target. No algorithm swap that failed dynamic BF16 parity is authorized by these results.

## Output-path correction V9; round26 running

Previous turn classified as progress: coupled V8 implementation, exact mixed-mode validation,7200-response screen and Nsight identified a head regression hidden by net output-path improvement. Revalidated frozenV8 clean; new isolated codex/output-path-v9 retains compact output catalog and restores previously real-model-qualified16head partitions. Source d589dba24ddb23ca37d66b6338fa9999a10abf29; binary e870a315b1b44bdab3ca14f2d79d43adc9f6c18125bcd9951226c92bde2df34a; build53d35ec6bbed4bf15533123a80c7afe124b7a8c80f619ca0b4a3cb2e5ffafd3e.24codec tests,586full-logit+185mixed-mode scheduler rows with cancellation/reuse,93replay269row whole-KV test and48strictstreaming responses pass.

Round26 compares baseline/V6/V8/V9/vLLM under identical mixed fixed-P128 screen:5lanes,C1/4/8,two reversedorders,240retained+24warmup each. Lifecycle helperb07da9aafd0999d0bb355d706067403512601713542f092843e08fea1e86928d checked live round25 successors before authorized pause. Final result/restoration pending. This correction belongs to the existing output-path batch; it is not a new performance qualification or final success.

## V9 round26 completed; retain corrected compact output path

30retained phases/7200responses complete; baseline/V6/V8/V9 each1440/1440exact. vLLM C1 480/480,C4 461/480,C8 455/480, so high-C comparisons remain observations. V9 vsV8 C4throughput+3.713%,TPOT-3.961%;C8throughput+5.840%,TTFT-4.740%,TPOT-12.046%. C8E2EP95 128.331->121.792ms,P99 130.057->124.290ms. C1throughput+0.128% is unchanged. V9 vsV6 C4throughput+10.672%,TPOT-10.756%;C8throughput+26.535% butTPOT+42.539% (capacity4vs8) andP99124.290vs121.844ms. V9C8throughput2179.546 vsobservedvLLM2326.324(-6.309%);TTFT8.653vs19.685ms(-56.044%);TPOT3.184vs2.429ms(+31.085%). C4vsobservedvLLMthroughput+5.698%,TPOT+13.607%. This remains fixed syntheticP128/mixedoutput short screen, not representative workload/stability or full numerical qualification. Keep V9 as the corrected experimental optimization base; preserve V6 as the older reference and all frozen candidates. Full goal active/unmet.

Nsight confirms head correction: N8head16 x17.536us~280.581us vsV8single476.475us. Other N8 GEMV costs unchanged (gate/up23.009us,QKV8.451us,O/down6.377us combined).140decode submissions medianGPUspan2131.169us,sum279.027ms; kernels259.556ms,GEMV204.239ms (~78.7%).48prefill submissions median4052.912us,total193.141ms; prefill is still a substantial part of mixed execution. Nonexclusive diagnostic, not an additional serving speedup claim. Compact graph copies remain low, sum0.231ms acrossdecode; no reason to continue tuning tiny transfers as the primary target.

Live round26 restoration verified PID/start/port533187/86302874/9876,533282/86302924/9911,533448/86302975/9887. Same command, listening ports, GUI/private runtime and pinned maps verified. All benchmark/profiler/build jobs terminated.

Next scope should move beyond output-path tuning: batched projection computation (weight reuse across rows) and representative variable prompt/output lengths/high concurrency remain required. Existing exact M1 profile must keep its tests. Before evaluating arithmetic-changing kernels, establish an independent FP32/teacher-forced numerical accuracy validation with explicit tolerances and held-out inputs; do not silently weaken exact comparisons or treat the known vLLM batch-dependent token divergence as a passing gate. The user's correctness requirement is broader than one exact implementation, but a new numerical contract requires evidence, not a faster kernel passing simplistic fixtures. This investigation should accompany a meaningful prefill/decode compute batch, not another repetition of unchanged P128 measurements.

## Shared-weight compute investigation with independent FP64 reference

Previous turn was progress (V9 correction,7200-response benchmark, verified restore). Revalidated frozen V9 source clean. Prepared121actual-weight projection fixtures across30layers+head usingCPU FP32model activation inputs from8synthetic held-out sequences oflength16..512. Native shared-row GEMM candidate was compared with original strided GEMV and independent FP64 dot products,1,637,376outputs. Default heuristic had1447predeclared rounding-bound violations; split-K1/reduction0 variant had0violations and590BF16differences.537differences are farther fromFP64,53closer; no claim of exact or full-model equivalent accuracy. Both head implementations matchFP64argmax7/8; counts alone do not prove same chosen tokens.

Hot graph probe suggests QKV8.16->4.27us,gate/up21.61->4.71us,head262.3->21.6us; down7.25->7.54us does not benefit, O5.53->4.23us is deferred. No production source change or serving benchmark in this turn, no Blender lifecycle modification. All fixture/probe processes completed. ROW_SHARED_COMPUTE_BATCH.md defines three coupled candidate integrations and separate numerical/ownership/full-model/serving gates; do not confuse primitive accuracy screen with a new accepted model correctness policy. Frozen V9 remains base; full goal unmet.

## Shared-row V10 integrated experimentally; model-level numerical flags remain

Previous turn was progress: actual-weight121operator/1.64Moutput FP64 screen selected QKV/gate-up/head and rejected default split reduction. Frozen V9 source was revalidated clean. New isolated /tmp/riley-opt-260912/multisequence-shared-source-v10 on codex/shared-compute-v10 is dirty atopd589dba24ddb23ca37d66b6338fa9999a10abf29; local source /tmp/riley-shared-integration-v10. Not frozen or serving-adopted.

New named native record/append_shared_multisequence_decode APIs preserve original ABI. Separate shared binder requires M=bucket2/4/8,N/Kexact,algorithm21,splitK1,workspace0 with existing ledger/parent checks. QKV/gate-up/head use ordinary M=B prepared GEMM plans; O/down retain stridedM1. Runtime experimental into_owned_shared_multi_decode_graph constructor includes the actual shared algorithm metadata in catalog identity. Old constructor/CLI remain unchanged; do not describe the built default server as exercising shared rows.

Existing24codec,586full-logit+185mixed-mode scheduler,93replay269row whole-KV regression tests and server build pass. New shared_model_diagnostic_gpu records172full-logit rows across decode1..8 from8requests, feeding both owners the original selected tokens to keep a common history. It checks finite outputs/execution/settlement, not numerical equivalence. Original and shared raw logits are retained for each row. Initial diagnostic compile errors (missing reexport and bool counter conversion) were corrected; failure log retained. No GPU completion/runtime failure in the final diagnostic. Shared-mode cancellation/greedy parity and malformed shared-plan bindings still need targeted qualification.

Independent CPU FP32 HuggingFace eager cached replay uses precisely the captured per-request input history. Diagnostic flag policy written before evaluation: sharedRMSE >1.1*originalRMSE+1e-4 or sharedKL >1.1*originalKL+1e-5; these are diagnostic flags, not a release acceptance policy. Shared meanRMSE0.111351vsoriginal0.117115; meanKL0.00101422vs0.00114997; maxKL0.00604036vs0.0126742. Nevertheless65RMSE and62KL rows flag. Top1againstFP32 original167/172,shared166/172; shared matchesoriginal167/172. Of5divergences,FP32supportsoriginal3/shared2. FP32top2margins0.0329..0.1771; do not dismiss all differences as negligible ties. The mean improvement alone is insufficient to declare correctness maintained.

V10 remains experimental and has no serving performance receipt or adopted model-quality contract. V9 remains the validated optimization base. No Blender pause this turn; all build/model/probe jobs completed. Evidence and working patch (plus untracked diagnostic source) exported with SHA256manifest. Next: qualify shared-mode ownership/mode switching, then evaluate a predeclared model-level numerical policy on a larger independent natural-text teacher-forced set with next-token loss and probability divergence, alongside unchanged V9 and FP32 reference. Preserve the existing exact profile; do not silently convert exact failures into passing performance cases. Representative variable-length serving, high-concurrency stability and full vLLM goal remain outstanding.

## V10 shared-mode reliability and natural-text numerical screen passed; serving screen running

Previous turn was progress: shared-row implementation and common-prefix FP32 diagnosis exposed unresolved numerical flags. This turn revalidated isolated source state. Added dual-shared-owner tests: full logits versus alternating full/greedy produce identical outputs/tokens across304rows (N4cancel119,N8cancel185), all shapes1..8, cancelled-page reuse and zero allocations at close. Added low-level shared record contract test acrossN2/N4/N8 xfull/greedy:6zero-weight cases,18replays,18malformed packet rejections and12wrong projection-plan binding rejections; rejected cold bindings do not prevent subsequent valid capture; all resources released.

Natural corpus: Salesforce/wikitext wikitext-2-raw-v1 validation, revisionb08601e04326c79dfdd32d625aee71d232d685c3, official https://huggingface.co/datasets/Salesforce/wikitext .704eligible rows; deterministic hash selection64windows,160tokens each. CorpusSHA6967f6407973283f5224e71bbba03418a0fe9c213828eda3d72874b599f1f10d. P128then32dataset ground-truth teacher-forced next tokens;2048scored outputs. This is a natural-text finite-corpus screen, not variable-P serving nor independent-article statistical proof. PyArrow21wheel installed only into /tmp/riley-opt-260912/natural-eval-deps via separate PYTHONPATH; existing vLLM environment unchanged. Downloaded wheel/source/corpus hashes retained.

Policy was persisted before running the candidate on this corpus: upper one-sided95% bootstrap meanNLL delta<=0.01nats; maxpassage mean delta<=0.05; shared meanKL<=original+0.0001;P99KL<=original+0.001; zero nonfinite.64passage-unit bootstrap10000repetitions with seed20260912; article independence/generalization not established. All screen checks passed: shared meanNLL3.1313107 vsoriginal3.1325617 vsFP323.1320420; meanpaired delta-0.0012510,upper95bound0.0003563,maxpassagedelta0.0183489. Shared meanKL0.00056587 vs0.00058122;P99KL0.00256113vs0.00267663. Both original/shared matchFP32top1in1990/2048;shared matchesoriginal2012/2048. Exact equality is not claimed, and this does not supersede unchanged exact-profile tests.

Added explicit --graph-numerics shared-smol-p128-v1 option; default and existing exact option remain unchanged. It requires graph policyrequire, uses boundedP128 geometry and M1 atC1. Builder normalization preserves shared flag; explicitly selecting exact graph clears it. Profile/CLI tests and server build pass. ActualsharedHTTP48responses complete, token identity differs from original references as expected and is recorded as numerical observation; no exact-reference claim.

Frozen source d073b1e8665ec1f5eb4fc56b6ac0a75e8cd3ce65; binary103960ccdfe539cb8ce30686d554a477afd9754d61171863abc99efef06346da; buildc4c4f802272132cd18258c4cf61306197234df8e18b7ac84c4ddc104cf9f451b. Round27 compares originalbaseline/V9/V10/vLLM,cap1/4/8 matched forV9/V10/vLLM, two reversedorders,240retained+24warmup. V10andvLLM use observation mode for exact-reference differences, with V10finite-corpus screen receipt separately pinned; neither becomes a final qualified performance claim. Lifecyclehelper0fca0f609602a6e67985421b7aa8c1d434c487237742b8005deda935ef8b3913 verified round26 successors before authorized pause. Results/restoration pending.


## V10 round27 completed: shared projections improve throughput; full goal remains unmet

24 retained phases and 5760 responses completed without transport/protocol failures; two reversed run orders, 240 requests per phase, fixed P128 and mixed O8/16/24/32. V10 is an explicit experimental numerical profile, not an exact replacement for V9. Source and binary remain frozen as recorded above.

| Concurrency | V9 tokens/s | V10 tokens/s | vLLM tokens/s | V10 vs V9 throughput | V10 vs vLLM throughput | V10 TPOT ms | vLLM TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 983.863 | 979.172 | 801.755 | -0.477% | +22.129% | 0.828 | 0.887 |
| 4 | 1856.591 | 2051.772 | 1726.418 | +10.513% | +18.846% | 1.704 | 1.699 |
| 8 | 2183.381 | 2711.437 | 2332.481 | +24.185% | +16.247% | 2.512 | 2.419 |

These are medians of two run summaries, not confidence intervals. C8 V10 throughput was 2707.141 and 2715.734 tokens/s; V9 2183.973 and 2182.789; vLLM 2344.017 and 2320.944, so the throughput direction agrees in both orders. C8 V10 vs V9 TPOT -22.646%, TTFT -4.726%. Compared with observed vLLM, C8 TTFT 8.854 vs 19.822ms (-55.334%) but TPOT +3.823%, which fails the equal-or-lower latency goal. E2E P95 98.600 vs 106.284ms and P99 101.143 vs 129.108ms; these short-run tails do not establish long-run stability. C1 throughput is effectively unchanged versus V9, as expected for the unchanged M1 path.

Exact reference matches: V9 480/480 at each concurrency; V10 C1 480/480, C4 440/480, C8 440/480; vLLM 480/460/453 of480. Natural-text finite-corpus numerical policy passed separately, but 40 divergent serving requests at each multi-row concurrency remain visible and neither numerical profile has broad quality qualification. Do not relabel these observations as a correctness-qualified vLLM win. Preserve V9 as the exact-profile reference and use V10 only as an opt-in experimental compute base.

Nsight diagnostic (nonexclusive): 140 decode submissions, median GPU span 1354.206us vs V9 2131.169us; aggregate decode span 186.250 vs 279.027ms. V10 decode kernels total167.219ms, including53.075ms strided GEMV and51.026ms dense GEMM (name-based groups). 48 prefill submissions total193.345ms, median3970.313us; V9 total193.141ms. Prefill now consumes approximately as much aggregate GPU time as decode in this trace. The next batch should address prefill execution and flexible geometry, rather than further small output-copy changes. Trace timing is diagnostic and is not the exclusive serving timing.

Live round27 restore verified: PID/start/port750171/86545229/9876,750244/86545279/9911,750312/86545331/9887. All three are live with original commands and listening ports; private runtime environment and mapped libraries checked without exporting them. completion.json records restored=true. Export contains231 SHA256-verified files (round27 raw responses/summaries, build identity, source patch, numerical policy/corpus/results, test logs, profiler analysis). 4104 natural-evaluation dump files remain remote with a SHA256 manifest; they were not all copied locally.

Next meaningful batch: first profile prefill projection/attention costs and inspect the fixed P128/context160 contracts; then implement related prefill/shape support changes with bounded ownership and numerical tests, and benchmark identical variable-length workloads against frozen V9/V10 and vLLM where supported. Unsupported shapes must be explicit rather than silently filtered from a comparison. C16+ admission behavior, longer outputs, repeated sustained load, tail latency and general model-quality validation remain outstanding. The full goal is active and not achieved.


## Prefill source and trace audit

Fresh per-kernel analysis: custom prefill projections141.549/181.742ms (77.884%); down44.393ms and gate/up39.638ms are the largest targets. The prefill dense_gemm_ns name group includes custom MMA and is not a cuBLAS attribution. Fixed geometry is jointly enforced by native packet/scratch checks, stage routing and server context/output limits. See PREFILL_COMPUTE_AND_SHAPE_BATCH.md for coupled scope and verification. No candidate change or new serving performance claim in this audit.


## V11 variable-row primitive and frozen natural serving inputs

Previous turn was progress: per-kernel attribution and source-contract audit changed the next batch scope. Created isolated codex/prefill-shapes-v11 from frozen V10, leaving V10 unchanged. Current intermediate commit217473bcdb3300c5dd3e8403b8b3938535396137 contains a variable-row projection primitive and standalone GPU validation. It is not connected to serving yet and is not a completed optimization batch.

The new projection accepts1..1024 live rows, launches ceil(rows/16) MMA tiles and masks inactive tail loads/stores. It preserves K16 recurrence and intermediate BF16 rounding. Original fixed projection kernels remain unchanged. Five projection geometries across22row counts (including1/7/15/16/17,127/128/129,255/256/257,398,511/512,1023/1024) and2input patterns pass220cases,34,311,168 exact BF16 comparisons,220tail-canary checks and invalid-row rejection. Pattern2 spans powers of two from2^-8 through2^8; this does not cover every floating-point value or full-model arithmetic. CUDA compute-sanitizer memcheck exits0 with ERROR SUMMARY:0errors over the same220cases. No performance timing was taken; Blender processes were not paused.

Before implementation, froze256 natural prose requests from the already pinned validation source: deterministic lowest SHA256 keys among1761full rows with16..1024tokens; no truncation. Selected prompt lengths16..398, median137; bucket counts32:16,64:32,128:69,256:106,512:33. Output lengths32:93,64:75,128:88. CorpusSHA c721b50e944fe0e0d3942a47f253fbde8556ef2f0e1dc55e6f64aa26a7d38b8a. This is prose completion, not a representative chat distribution; it expands geometry coverage but does not replace the full workload objective. V10 cannot serve most of these shapes. No requests were served or silently excluded.

Exported8files with verified SHA256, including corpus/manifest, new native source/probe, build/test/memcheck logs. Remaining same-batch implementation: RoPE/KV/attention variable shape and context, native metadata/scratch/catalog identity, scheduler and server admission; then full-model/cancellation/mixed-mode checks and serving benchmark. Existing primitive tests alone do not authorize relaxing server guards.


## V11 coupled prefill batch: fused variable RoPE/KV primitive verified

Previous turn was progress: variable projection implementation, frozen natural inputs and GPU/memcheck evidence. Continued the same unfinished batch in isolated source; intermediate commit47cf42dedea803e4915d438082d29e2fd5603d05. New prefill_shape_rope_kv.cuh combines RoPE and paged KV publication for1..1024rows and context<=4096, preserves original BF16 table rounding and output rounding, supports nonzero chunk starts and omits the temporary K output. This is an internal primitive, not a serving entry point. Callers must retain parents and validate complete map/extent/alias contracts before enqueue; per-request publication still requires owner completion. Existing serving kernels and V10 are unchanged.

GPU memcheck suite passes27cases: rows1/15/16/17/127/128/129/398/1024 at starts0/13/128; context1152,72physical pages, permuted logical map. Compared against unchanged M1 compiled_rope GPU output with independently assembled host page scatter:3,205,440exact Q elements and11,943,936full-cache elements, including untouched bytes. Invalid context overflow is rejected at launch. Compute-sanitizer exits0 and reports0errors. Tests do not cover aliasing, malicious page maps, full-owner cancellation, arbitrary FP values or model quality. No serving/performance test and no Blender pause occurred.

Fresh attention source inspection finds generic attention also checks n<=160; its128-token softmax tiles already iterate over the causal prefix, but context extension and query-tail behavior require explicit validation. Next same-batch work is attention/context and full owner/staging/catalog/scheduler integration, followed by full-model and serving qualification. Exported4files plus SHA256manifest, all hashes verified locally.


## V11 same batch: variable-context attention primitive

Previous turn was progress: fused variable RoPE/KV implementation and whole-cache GPU evidence. Intermediate isolated commitb2b11aa64e93620a9fc8bbf9d1e90fcaed582632 adds a private MODE6 attention primitive with1..1024rows, context<=4096 and causal start offsets. Existing accumulation and reverse128-token tile order remain; existing production kernels are unchanged. Host launcher validates scalar geometry; map/parent extents remain a prerequisite for the future owner. Parameters are currently captured as host values; a reusable bucket graph still requires validated per-replay metadata or correctly keyed exact-shape captures, not arbitrary replay length changes.

22 GPU cases pass: starts0/13/128/1024 with rows1/17/127/129/398, plus position4095 M1 and start3072+1024rows. Ten cases within the old160-token extent exactly match the original kernel. Independent CPU FP64 softmax/dot sampled first/middle/last rows across9heads:38,016 comparisons (includes repeated row samples for M1), maximum absolute error0.000997302 against a predeclared0.005 bound for bounded inputs[-0.5,0.5]. This finite primitive tolerance is not a full-model quality policy. Per-sampled-row future K/V replaced by NaN leaves output exactly unchanged when evaluated as M1, checking causality and row-offset parity. Compute-sanitizer exits0 with0errors. Initial test generator failed on an unhandled #elif; generator corrected, compile and all GPU checks rerun successfully; both diagnostic logs retained.

Projection, fused RoPE/KV and generic attention foundations now exist in the same unfinished batch. They are not linked into serving, and no batch benchmark or improvement claim exists yet. Next: build the owning execution/staging path, preserve complete page-map and cancellation lifetime validation, wire scheduler/server admission, then perform whole-model numerical/ownership tests and identical baseline/vLLM workload measurements. No Blender pause or host environment change in this turn.


## V11 retained graph integration regression passed

Previous turn was progress: variable-context attention primitive and causal/FP64 diagnostics. Intermediate isolated commitfb17aaad1e518149e6e2194a5f3c24aba8a3834c connects shape projection and fused prefill RoPE/KV to the existing validated P128 packed graph. Preserves the established P128 scalar Q/K/V/O load choice because prior profiling rejected replacing it wholesale. Fused path replaces separate RoPE/KV enqueue; ordinary unbatched paths remain. Multi-catalog identity now includes all three new header contents, not only their include filenames. Existing complete stage/parent validation and server admission remain unchanged. Wider primitive geometry does not yet imply wider request support.

Fresh release CUDA build in separate prefill-shapes-target-v11 completed. Two actual checkpoint owner tests passed in43.19s:3prompts full logits/argmax/status exact,96full initialized KV snapshots after every decode, zero allocations at close;6request retained reuse suite899slow/137fast replays, cancellation after prefill and decode,42invalid shape/stage cases, raw outputs and final full KV exact. The test command filters batched_prefill_; numerous other integration binaries report0tests and are not counted as validation. No performance benchmark was run. Build and tests are terminal; no Blender pause.

Remaining same-batch work is variable per-replay geometry and request admission. Current codec enforces fresh complete128prompt, target<=160 and generated_index==position-127; native scratch/logit extraction and scheduler must change consistently. Generic attention foundation is not yet connected to the serving owner. Do not loosen CLI guards alone or interpret these P128 regressions as variable-serving proof. Full objective remains unmet. Exported integration patch/test log with SHA256 verified locally.


## V11 scheduler-owned prompt length and request progress

Previous turn was progress: native prefill integration and full checkpoint regression. Intermediate commitbc31f6c7df6ffa72e0dc04e398d8125e8d67b2c8 adds request-progress validation used by the existing descriptor path, and carries prompt length directly from the live scheduler request into AuthorizedExecutionRow. The scheduler adapter validates that value before forming a descriptor. Input chunks calculate target position/page count/final-page occupancy and expose a logits row only once the prompt is complete; decode requires committed length==prompt+generated_index-1. Context admission accounts for the final emitted token not requiring another forward pass. Checked arithmetic rejects overflow and exhausted generation.

The logical helper supports chunked progress but the v2 wire still requires P128/context160. The adapter explicitly rejects a different prompt length and incomplete prompt publication until native metadata and output routing support them. No variable request support or speedup is claimed. Generic primitive argument limits and logical progress alone are insufficient to change graph replay shape.

26descriptor CPU tests pass (including chunked trajectories over11prompt lengths and4output limits, invalid stage/overlap/context/overflow);32scheduler library tests pass. Fresh CUDA scheduler authority suite5tests passes:771compared rows across1..8decode shapes,35completed requests, cancellation/KV reuse and alternating output modes. These GPU tests are still P128 regressions. Source patch and3test logs exported and SHA256 verified. No Blender pause; GPU test job terminal. Remaining same batch: new variable wire/staging/catalog and native replay geometry, partial-prefill output suppression and variable decode context, server admission, full-model numerical qualification and serving baseline/vLLM benchmarks.


## V11 variable request wire implementation

Previous turn was progress: scheduler-owned prompt length and real GPU authority regression. Intermediate isolated commitffbff8132fbf53e6862a18475b79e1f214f95ab1 adds separate V3 request encoding and native structural validation. Extent17536bytes:128header,8x1664row records,1024prefill tokens. Each row holds prompt/context/progress/output bounds, explicit logits input row (UINT32_MAX for unfinished prefill),256page IDs and valid counts, request/cookie identities. Prefill is one request; decode supports1..8rows. Rust encoder validates external block ownership and unique reservations before mutating a retained output buffer; canonical comparison with caller-owned scratch binds every byte, including padding, to the external expectation. Native structural validation does not replace ownership/catalog/replay authority.

28descriptor tests pass, including every-byte mutation on partial-prefill and N8decode packets, stale replay, foreign/aliased pages and no writes on rejected expectations. Canonical Rust packets are accepted by C++ validator;24known-invalid structural mutations rejected. ASan/UBSan pass while exercising35072single-byte mutations for memory safety (not all are structurally invalid; changes to legitimate identity values require Rust authority binding). Two fixtures, patch and logs exported with5SHA256-verified files. No GPU execution or Blender pause in this turn.

V3 is not yet wired into the native recorder, retained completion owner or scheduler dispatch. Existing V2 and server bounds remain intact. Next same-batch work must connect V3 page/token offsets and validated per-replay progress to the new kernels, add partial-prefill no-output completion and variable decode, then qualify whole-model behavior and serving performance. This is not a completed optimization batch or goal success.


## V11 V3 geometry consumed by a retained CUDA graph

Previous turn was progress: V3 encoder and native structural validation. New private kernel arguments consume live row count and start/position from fresh V3 device metadata. Projection masks row tails; fused RoPE/KV uses committed start; attention rejects inactive launch rows before reading Q and uses the replay position. Existing fixed host-shape calls use null metadata and preserve their semantics. Header includes are now self-contained; the first standalone build exposed missing uint32_t includes, corrected before successful rebuild and tests.

One graph captures H2D metadata plus projection->RoPE/KV->attention. Replayed16times with rows17/73/128/398/1024/1/129/16 and starts0/128 in the same capture, demonstrating shrinking and growing shapes. Full Q/rotated-Q/attention scratch including padding and full KV pools match separate exact-shape launches:53,477,376BF16comparisons. Compute-sanitizer exits0 with0errors. This is an operator-chain replay test with synthetic bounded input and identity RoPE tables, not all30model layers or representative serving. Host structural V3 validation occurs before each replay; it is not a replacement for live Rust ownership.

Fresh actual-model P128 owner regression rerun also passes2tests: full logits/status,96KV snapshots, cancellation/reuse and invalid shape/stage tests. Results preserved under a new log name so the previous exported receipt remains unchanged. Native dispatch of a complete variable-shape model, norm/embedding/live final-row selection, partial-prefill output suppression and variable decode remain incomplete; server admission is unchanged. No serving benchmark or Blender pause this turn. Patch and4logs exported with SHA256 verification.

Intermediate source commit: 5caa8110f04927b26abe3674100edf11900ef7b8


## V11 dynamic embedding, norm, SwiGLU and final hidden selection

Previous turn was progress: dynamic V3 projection/RoPE/KV/attention single-capture replay and P128 whole-owner regression. Intermediate commit36f4dafb9236f8c99cd9ebf0b8eae20344cc6b4c adds private live-row embedding, all3original norm/residual modes, SwiGLU, and final hidden selection. Invalid token records a device status bit; hidden selection requires clean status and a valid logits row, otherwise zeroes the576element result and clears publish. Partial prefill UINT32_MAX row therefore cannot leave a previously generated hidden vector marked publishable. This GPU flag is not yet wired to scheduler completion or HTTP publication.

Single CUDA graph14valid replays: rows1/17/73/128/398/1024/16, each final/partial. Byte-exact comparisons against original fixed row operators plus host embedding lookup cover159,645,696bytes, including FP32 residual and inactive padding. Selected final live row matches reference; partial output zero/publish0 and64element output canary checked. One additional invalid-token replay explicitly requests publication and still produces status error, zero selected hidden andpublish0. Compute-sanitizer exits0 with0errors. Synthetic bounded inputs and64token embedding fixture, not the full checkpoint. No timing or Blender pause.

All basic variable-prefill operators now have dynamic replay scaffolding, but full model recorder/parent ownership, completion routing and variable decode remain unfinished. Need integrate these operators across30layers and a dynamically selected head input, preserve page/cancellation transaction lifetime, and wire V3 completion/dispatch before accepting general serving shapes. Patch/build/memcheck logs exported with3SHA256-verified files. Goal active; no new serving-performance claim.


## V11 full checkpoint prefill composition

Previous turn was progress: dynamic embedding/norm/SwiGLU and publishable hidden selection. Intermediate commite08f5e435ff564dac82645ea619d43f9a1f5f2ba composes the private operators across all30SmolLM2 transformer layers, including final normalization and selected hidden. Uses supplied borrowed scratch/weights/KV and V3 metadata, so complete pointer extent/alias/ledger validation is still a prerequisite for a retained production recorder. Head GEMM, result envelope, Rust completion owner and server dispatch are not included. Native CUDA library build passes with the new header included.

Actual pinned BF16 checkpoint weights and real RoPE theta100000 tables used. Natural input lengths16/128/398 from the frozen variable corpus run through one captured graph. Full prompt vs73-token chunks:12replays, selected hidden and full30-layer KV pools byte-exact,70,782,336bytes compared. Intermediate partial chunks have selected hidden zero and publish0; completion chunk publish1. Compute-sanitizer exits0 with0errors. Export includes reproducible weight order/source hash; derived large weights.bin remains remote with its SHA256.

Independent CPU FP32 HF diagnostic on the three selected final hidden states: finite; RMSE0.022964/0.024020/0.027793, maximum absolute0.387852/0.179727/0.199813. Projecting both hidden states through the same CPU FP32 head gives KL0.00048955/0.00018196/0.00025265. Top1matches all3but is token198 in all cases, so this is weak quality evidence and not a model-quality pass. It does not validate the future GPU head. Full-vs-chunk parity alone would not establish independent numerical correctness.

Remaining same batch: bind borrowed sequence to the native retained-resource ledger and V3 replay authority, connect GPU head/conditional output envelope, implement variable decode and scheduler settlement, then whole-model generation and exclusive serving benchmark against baseline/vLLM. No Blender pause or serving speedup claim this turn. Exported source patch, logs, small fixtures, hidden outputs and manifests with verified SHA256.


## V11 retained V3 recorder and canonical GPU head

Previous turn was progress: borrowed30-layer checkpoint prefill, chunk/full KV parity and independent final-hidden diagnostics. Intermediate commitc170f60470ab0e886906334d465f3ba2a5e40c82 adds a public native recorder and Rust BorrowedGraphResourceReservation wrapper. Reuses original resource ledger and capture lifecycle. Preflights273weights,22required mutable/scratch parents plus optional workspace, sizes/context identity/aliases and retained canonical head plan. Captures V3 H2D->30layers->selected hidden->canonical head GEMM->argmax->D2H. Input17536bytes is separated from result98432bytes. Result status/publish at0/4, argmax at8 and full logits at128; prefix padding initialized once. Replay invalidates earlier completion before V3 structural/shape checking. Existing V2 sizes and dispatch remain unchanged.

Rust GPU recorder test passes with zero weights:3replays including partial/final prefill,2cold alias/weight-extent rejections followed by valid capture,3malformed-page rejections followed by blocked stale result reads, head zero logits and all allocations released. Build and ABI C checks pass. Initial implementation used an incorrect canonical state type name and the test had an extra angle bracket; both corrected and the passing build/test rerun. This is lifecycle/zero-weight proof, not actual-model head parity. The previous checkpoint test used the borrowed standalone sequence, so those two proofs must not be merged into a claim that actual checkpoint output through the Rust recorder is verified.

Next same-batch work: prepare actual checkpoint parents and head plan through the Rust recorder, validate logits and chunk/whole parity, add the retained runtime V3 result/settlement wrapper and variable decode, then connect server admission and perform baseline/vLLM serving measurements. No Blender pause or performance claim. Exported patch and2logs with3SHA256-verified files; previous full-model native-build receipt preserved locally under its original hash.


## V11 actual checkpoint recorder and device completion identity

Previous turn was progress: retained native ledger plus canonical head, zero-weight lifecycle proof. Intermediate commit99672ec844bad355b96927a6b5b69fa97ea75609 runs actual exported checkpoint weights through the public Rust reservation/recorder, not only the standalone borrowed sequence. P16/P128/P398 full vs73-token chunks:12replays, GPU BF16 logits/argmax/status exact and complete KV pools exact, partial logits zero with publish0, final logits finite, all allocations released. Ordinary and compute-sanitizer runs pass; memcheck0errors. The fixture loader is diagnostic; a production model owner still needs to prepare these parents.

Added device-produced completion identities: result16..128 echoes owner generation,replay,iteration,request,cookie,target,prompt,generated index,input count,context,stage,digest,last position,slot,mode and magic. A128byte device status/header replaces the former4byte status parent; one header D2H consolidates status/argmax/identity copies. Full result remains98432bytes. Runtime validator binds these bytes to the outstanding expectation, rejects GPU/argmax errors, checks finite logits and deterministic host argmax, and returns no token for unfinished prefill. This validation function is not yet coupled to a runtime transaction or scheduler settlement.29descriptor CPU tests and updated zero-weight recorder rejection tests pass after the change.

Independent CPU FP32 diagnostic using actual GPU logits (not only CPU-projected hidden): KL0.00110251/0.00031366/0.00026597 atP16/P128/P398; finite and top1matches3/3. All top1s are198, making the small diagnostic weak for quality equivalence; no quality acceptance claim. Remaining same batch: retained runtime transaction/result routing, variable decode and request admission, broader generation/numerical/cancellation qualification, then actual serving baseline/vLLM benchmark. No Blender pause or speedup claim. Source patch, logs, actual GPU logits and FP32 diagnostic exported with verified SHA256.


## V11 variable-context single-request decode

Previous turn was progress: actual checkpoint Rust-recorder logits and device completion identity. Intermediate commit82cfd2e8466d8f100eadc41c9c819657b4939695 supports one-row decode using V3's input token and preserved KV. Native structural/graph guards still reject more than one active request. Prepared separate prefill-capacity and1row decode graphs under the same ledger, selected by validated stage. Added uniform inactive-tile return in projection to avoid weight/MMA work outside live rows. Completion validator now accepts a single decode row and binds its generated index. This remains the experimental prefill arithmetic for decode, not the frozen V10 shared-row profile.

Checkpoint validation covers P16/O32,P128/O64,P398/O128. Three paths per request: full-prefill cached generation, chunked-prefill cached generation, and independent full-prefix recomputation on the same generated history.678replays pass; every generated-step GPU logits/argmax and final full KV pools exact, finite, output-bound rejection invalidates reads, all allocations released. Thirty descriptor CPU tests pass. Memcheck passes all678replays with0errors (270.45s instrumented); ordinary run about6s is a correctness harness duration, NOT serving throughput/latency. Final rerun adds token/logit export only; no further computational change after sanitizer validation.

Independent CPU FP32 on224common-history steps:222top1matches,59distinct generated tokens, meanKL0.000486323,P99KL0.002470926,maxKL0.002846303. Divergences: P128 step21 native30vsFP321673, FP32top2margin0.047859;P398 step11 native1977vsFP324771,margin0.021478. These differences remain reported; no general model-quality acceptance or independent FP32 greedy trajectory equivalence is claimed.

Source patch, generated token IDs, all224GPU logit vectors, diagnostics and logs exported with SHA256verification. No Blender pause or performance claim. Remaining same batch: multi-request V3 execution, live scheduler authority/settlement, server admission and representative serving comparison against frozen baseline and vLLM. Full goal remains active and unmet.


## V11 live scheduler V3 expectation adapter

Previous turn completed variable single-request decode evidence; this turn adds authoritative source commit5e4b3b6abdfb44160bcc9b49308cdba940fb600f. A crate-private V3 adapter derives prompt/progress/input/page fields from AuthorizedExecution and the full live block ledger, and binds runtime-supplied owner identity/context/replay/cookies. Rejects mixed-stage plans, foreign pool geometry, inconsistent target/output publication, missing cookies and stale replay. Partial prefill uses protocol row slot0 but must not produce IterationOutput; actual output-slot presence must exactly match progress publication. Existing V2 adapter remains restricted. This method is not yet called by the production execution adapter and does not prove a retained V3 catalog or GPU completion.

New live scheduler CPU test follows P16/O32,P128/O64,P398/O128 through73token prefill chunks and every decode step, encoding/validating V3 against real reservations, testing stale replay and missing ownership, then settling synthetic CPU outputs and checking final logical length. This is host authority/settlement evidence, not GPU scheduler integration. Complete non-CUDA scheduler suite passes132tests including1doctest (33unit plus98integration plus1doc); CUDA-gated tests did not run. Patch and2logs copied and verified against remote SHA256. No GPU changes, Blender pause, or serving performance measurement. Next same-batch work remains retained runtime transaction/result routing, multi-request V3 execution and server admission before baseline/vLLM qualification.


## V11 retained GPU transaction through scheduler completion

Previous turn was progress: live V3 authority adapter. Current isolated commit5a2bb1fa133b64b25570d842dc7ac2c722032c23 adds BorrowedVariableSession owning the prepared native reservation. Issues unique generation/cookies and sequential replay identity, retains one expectation through native synchronous replay/read and GPU result validation, blocks further issues until explicit matching scheduler commit, and poisons on native/completion errors. Native close remains required before releasing GPU parents. Constructor requires already prepared V3 graph and caller-bound digest; it does not prepare the actual production model or independently authenticate the catalog. Single active request only. Wire max_active_rows remains8 because native V3 recorder structurally requires that existing capacity; first integration run using1 was rejected before dispatch, then corrected.

Scheduler execution adapter now derives the V3 expectation from live authority and returns DownloadedLlamaIteration only after validated native completion; partial-prefill output is empty, generated logits are retained for normal sampling/settlement. GPU integration test uses actual checkpoint parents, real scheduler page reservations,73token prefill chunks, P16/O32,P128/O64,P398/O128.230iterations pass through GPU completion and scheduler settlement; every published logit vector exactly matches earlier recorder generation references. Busy-session issue and wrong-iteration commit rejected at every step. All allocations released. Compute-sanitizer passes230iterations with0errors (44.52s instrumented). Full non-CUDA scheduler suite132tests passes. No cancellation/GPU failure injection or multi-request validation in this new bridge yet.

This is actual single-request scheduler/GPU integration, but test fixture still prepares borrowed parents with diagnostic weights.bin and fixed test digest. Production model-owner/catalog preparation, server admission and multi-request V3 execution remain incomplete. No Blender pause or serving performance claim. Patch and4logs exported and verified against remote SHA256. Next work must connect production preparation and batched decode before matched baseline/vLLM benchmark and workload/quality qualification.


## V11 actual model loader to retained V3 scheduler execution

Previous turn was progress: borrowed transaction and actual-checkpoint scheduler completion using diagnostic prepared parents. Commit5272d2aab20966a9bb4efafa3d0c908acb042ae9 adds PreparedLlamaBatchExecutor::prepare_variable_session using existing loaded weights, immutable RoPE and actual KV parents, plus cold VariableGraphBuffers for18scratch/result parents and canonical head plan. Rejects unsupported geometry, bias/epsilon/profile, poisoned owner and excessive context; only fixed30layerSmolLM2 is supported. Cold identity hashes actual uploaded weights/RoPE, selected head algorithm/runtime fields, V3 operator/packet/recorder source and logical parent weight ordering. Exclusive borrows prevent eager mutation while retained session lives. Native recorder accepts8192position RoPE storage but caps V3 execution context4096. This does not enable8192context execution.

Initial direct old-fixture comparison passedP16/O32 then failedP128 first output (43045BF16logits differed). Investigation downloaded first1024RoPE positions from the loader:36365of65536FP32values differ from CPU fixture, maxabs5.738437e-5. Temporary diagnostic source writes were removed before final build. With ONLY the loader RoPE substituted into an isolated fixture, existing independent recorder passes678full/chunk/cached/full-prefix-recompute replays with exact logits/fullKV. The final actual-loader scheduler run matches all those reference logits across230iterations atP16/O32,P128/O64,P398/O128;0allocation remainder. This isolates the old-fixture discrepancy to RoPE generation rather than relaxing equality. Original fixture and receipts preserved.

Compute-sanitizer final actual-loader run passes230iterations,0errors,61.17s instrumented. CPUFP32 comparison on the new common generated histories:223/224top1matches,68distincttokens,meanKL0.000508804,P99KL0.002832017,maxKL0.003347447. One divergenceP128step21:GPU30vsFP321673,KL0.001031269,FP32margin0.047859. No independent FP32 greedy or general quality acceptance claim. Native loader preparation and scheduler path now exist, but owned server lifecycle/admission and multi-request V3 still absent. No Blender pause or serving benchmark this turn.18files exported with remote/local SHA256 verification, including new RoPE/reference logits and numerical diagnostics; large weight fixture identified by hash, not duplicated.


## V11 owned model/session lifetime for server preparation

Previous turn was progress: actual model loader to borrowed V3 execution. Commit7f5f1361955c5a8ad5638d6bb365613668e947e8 adds into_owned_variable_session: moves prepared model, stream and scratch into existing OwnedGraphResourceReservation. A sealed native-operation trait lets borrowed and owned sessions share the same issue/execute/commit state machine; external fake completion implementations are disallowed. Native close succeeds before explicit scratch/head, executor and stream close. Scheduler V3 adapter now accepts either reservation form. This is an owned resource path, not yet a server request routing option.

Actual loaded-model owned test passes P16/O32,P128/O64,P398/O128 with230normal iterations and all logits exact against loader-RoPE references. One extra P398 partial prefill remains uncommitted: another issue is rejected, native session closes, then scheduler aborts DeviceQuiescedMutationUnknown and closes. Total231GPUreplays;all allocations released. Final compute-sanitizer run passes with0errors (61.46s instrumented). CUDA riley-server cargo check passes; this proves linkage/type compatibility, not serving behavior. No native failure injection or multi-request stability claim. Patch and3logs exported with remote/local SHA256 verification. Server V3 routing/admission and multi-request decode remain needed before serving benchmark. No Blender pause, no performance claim; goal active.


## V11 actual HTTP serving through variable graph session

Previous turn was progress: owned model/resource lifetime and pending-result shutdown. Commit13bd8fcb263f9980447fa3a4d8a2f908f6017299 introduces explicit --graph-numerics variable-smol-v3. Requires graph policy require, one active sequence, CPU sampling, fixed-max shape, scheduler token/chunk budgets<=1024 and context<=4096; native model preparation still bounds supported Smol geometry. Existing profiles remain defaults and retain their gates. Prepared eager metadata is1row while dedicated V3 prefill scratch uses scheduler chunk capacity. Backend now checks this separate geometry, prepares the owned session, dispatches authorized V3 plans, confirms successful scheduler commit, closes V3 before reclaiming scheduler reservations, and includes the session in shutdown liveness.

First startup was correctly rejected by old eager budget-equality guard (73scheduler vs1prepared row). Replaced only for validated V3 geometry; subsequent actual cuda,server binary built and served HTTP. Final dedicated localhost run:8complete responses,608generated tokens, P16/O32,P128/O64,P398/O128, including3concurrent submissions queued behind one active request and one full SSE completion. Returned prompt IDs and generation token IDs exactly match loader-RoPE references. Output request129 rejectedHTTP400. Client disconnected another token stream, then a subsequentP16/O32 completed correctly. This demonstrates recovery after disconnect, not measured cancellation latency or a specific in-flight cancellation instant. Graceful stdin shutdown exit0; authoritative final metrics show0active/waiting requests,0KVblocks,0device/pinned allocations/bytes. V3 shape/timing statistics remain degraded/unavailable; no fake GPU timing reported.

Server CPU tests:68library pass1ignored;24CLI pass. Initial4C02tests failed because default umask created their own outer directories775; private TMPDIR alone did not fix it. Re-run with process-local umask077 passed all92tests, without changing C02 validation. Debug HTTP binary identity saved; final source adds a CLI unit test and corrected method comments only. Actual serving workload correctness exists now, but only one active sequence, not batched V3 decode or a matched performance qualification. No Blender pause or performance run. Evidence exported with SHA256, including HTTP bodies and verified shutdown metrics. Next substantive work is multi-request execution/scheduling and profiling/benchmarking the completed batch against frozen baseline and vLLM.


## Round28 V3 matched single-active serving screen and regression diagnosis

Previous turn was progress: actual V3 HTTP serving with variable prompts and lifetime verification. Current source remains13bd8fcb263f9980447fa3a4d8a2f908f6017299; release binary frozen at /tmp/riley-opt-260912/variable-candidate-v11/riley SHA7e571cc36c04a324553eccf23308c4a3bb02b0aced9ffb948b49ed619a620e37, build log SHAe3ebfe6f2579525306334c501a9ecb6afd7a908ead08da65dc41ebec0f7ad655. Matched model/tokenizer hashes, GPU, HTTP token-aware client and request sets; C1,128token batch/chunk,2reversed orders,12warmups+120retained per lane,1200retainedresponses total,0transport/protocol failures. FixedP128 mixedO8/16/24/32 includes V10,V3,vLLM. NaturalP16/O32,P128/O64,P398/O128 includes V3,vLLM only; V10 cannot accept these variable shapes. CPU sampling V3 vs GPU greedy V10 is each current implementation, so this measures present serving paths rather than isolating sampling alone. No broad quality acceptance.

Two-run medians: fixed throughputV10981.111,V3184.475,vLLM798.779tok/s;TTFT4.409/5.573/7.464ms;TPOT0.827/5.384/0.886ms. V3 fixed throughput-81.20%vsV10,-76.91%vsvLLM;TPOT+550.80%vsV10,+507.76%vsvLLM. Natural throughputV3131.542,vLLM840.364tok/s(-84.35%);TTFT5.609vs8.098ms;TPOT5.641vs1.016ms(+455.09%). These are strong regressions, not an optimization success. Fixed all references match240/240perlane; naturalV3240/240, vLLM160/240 vsV3 loader-reference. Tail statistics are short-run empirical samples, not stability proof.

Previously authorized exact3Blender pause/restoration performed with round28 helperSHAb1483a14987826350b9f6e40f9b49dc23908d66df228eeb2b1e0568a25c0e506, successor of round27SHA0fca0f609602a6e67985421b7aa8c1d434c487237742b8005deda935ef8b3913. Verified immediately before pause and after restoration via exact command/cwd/GUI environment/tag/start/private driver mappings and ports. Restored PIDs1505364/start87373189/port9876,1505421/start87373239/port9911,1505496/start87373291/port9887. Public minimal restoration proof exported; private snapshots/journals and session logs excluded. No host driver/global configuration changes.

After restoration, nonexclusive Nsight diagnostic on12fixed requests captured12prefill+228decode graphs; all references exact. Decode GPU kernel span median5214.744us;60.500%kernel time in generic prefill projection and34.831%in generic attention. Prefill span median4608.192us,projection69.252%,attention24.520%. The trace is not matched serving timing but localizes GPU work: the generic prefill path used for M1decode is the dominant regression. Next optimization batch should replace M1projection and decode attention together, preserve/qualify numerical behavior, then repeat serving comparison before expanding V3 concurrency. Source unchanged this turn;96evidence files exported and SHA256verified. Full goal remains active and unmet.


## V12 decode projection/attention batch and Round29 serving comparison

Commit 00820aed7e13e49e0815882774105c8dcb89110f in the isolated remote source implements the measured M1 bottleneck batch: parallel projection CTAs across original BF16 rounding chunks with ordered merge; token/head-parallel attention scores and dimension-parallel values preserving original recurrence; shared bounded scratch allocation and digest/extent validation. Multirow prefill remains unchanged. Frozen release SHA256 666b4c2c1ad13acbf6be2510ddc8603868d52649bc25e4f8d19e2ce8952f5ae3. Source patch and build receipt are in raw/decode-batch-v12.patch and raw/variable-candidate-v12/build.json.

Correctness: loaded owned scheduler test 231 GPU replays with all logits exact against immutable loader-RoPE references, including pending-result close/abort and zero allocations. Same test under compute-sanitizer reports zero errors. Primitive validation covers 15 projection cases and 11 attention contexts through4096 with permuted physical pages, all BF16 outputs exact; primitive memcheck zero errors. Recorder ledger validation passes cold rejection, malformed packet rejection, partial no-output and allocation cleanup. These preserve the existing numerical contract, not independent model quality acceptance.

Round29: identical model/tokenizer/GPU/client and request sets, C1/max-active1, batch/chunk128, two reversed orders, 12 warmups+120 retained per phase, 1440 retained responses across twelve phases. Two-run medians:

| Workload/lane | Throughput tok/s | TTFT ms | TPOT ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| Fixed / V11 |184.187|5.585|5.387|174.762|175.268|
| Fixed / V12 |391.330|5.539|2.381|79.999|80.625|
| Fixed / vLLM |804.253|7.420|0.887|36.372|37.882|
| Variable / V11 |131.779|5.582|5.627|1208.742|1210.714|
| Variable / V12 |358.921|5.618|2.417|392.479|393.577|
| Variable / vLLM |850.201|7.938|1.012|153.789|156.610|

V12 throughput improves112.46% fixed and172.37% variable versus V11; TPOT decreases55.80%/57.04%. Against vLLM, throughput remains51.34%/57.78% lower and TPOT168.43%/138.82% higher. Keep the batch as a measured V3 improvement; goal is unmet. All V12/V11 responses match their reference IDs/text; vLLM fixed240/240 and variable160/240. Tail samples are short C1 runs, not high-concurrency or stability qualification.

Authorized three Blender processes restored and freshly verified using exact command/cwd/GUI environment/tag/start/private driver mapping and listening ports:1632103/start87515897/9876,1632256/start87515948/9911,1632354/start87516001/9887. Minimal public proof exported, private restoration journals/session logs excluded.

Post-restoration nonexclusive Nsight captured12prefill+228decode graphs with exact references. Median decode GPU span falls from5214.744us to2228.473us; prefill remains4610.016us. Decode projection is72.974% of kernel time and attention15.845%. Gate/up projection alone consumes170.194ms of503.472ms decode kernel time,12.441us per invocation; attention values62.590ms,9.151us per invocation. Next batch should target projection loading/warp utilization, particularly gate/up, while preserving chunk arithmetic; assess fusion alongside this only with numerical proof. Trace timings are diagnostic, not exclusive serving results. V3 multiactive execution remains unimplemented and must follow before goal qualification.

120 evidence files exported with per-file SHA256 verification (raw/serving-round29-manifest.json); eight reproduction sources copied alongside this report. Full goal remains active.


## V13 projection layout diagnostic: no promotion

Previous turn completed V12 implementation, matched Round29 serving evidence and export (progress). Revalidated isolated remote source clean at00820aed7e13e49e0815882774105c8dcb89110f. Added standalone diagnostic comparing1/2/4/8warps per CTA and paired64-bit weight loads with warp shuffles, preserving the original MMA and BF16 chunk recurrence. All24 tested configurations produce exact BF16 output versus V12 for gate/up, down and K/V projection shapes. Production source remains unchanged.

Nonexclusive CUDA-event diagnostic,50warmups+500repetitions per configuration: gate/up one-warp baseline6.304us, multiwarp6.414/6.412/6.509us; paired-load variants6.840–6.980us. Down baseline5.487us, best5.460us (too small to promote); paired loads5.925–6.056us. K/V baseline4.428us, alternatives4.454–4.600us. The extra shuffles do not repay their cost in this test; CTA packing alone does not materially improve performance. No serving optimization or speedup claim follows from these measurements. Diagnostic repeatedly reuses one weight matrix, unlike the full-model trace (gate/up12.441us), so cache behavior and whole-model weight traffic need investigation before choosing the next batch. Candidate directions are capture-time weight packing and reducing duplicate loads, with exact arithmetic validation and a full-model graph diagnostic before matched serving. Exported source/log hashes in decode-projection-probe-v13-manifest.json. No Blender pause or production change; goal active.


## V14 tiled-weight graph diagnostic and next integration batch

Previous turn produced evidence rejecting warp packing/vector-shuffle candidates (progress). Remote source revalidated clean at00820aed7e13e49e0815882774105c8dcb89110f. New standalone projection uses offline N8/K16 weight tiles with two contiguous64-BF16 planes, so each MMA weight instruction accesses a contiguous warp segment without shuffles. Arithmetic recurrence and BF16 rounding intervals remain unchanged. Diagnostic CUDA Graphs alternate original/tiled lanes in two reversed orders,5warmups+30timed launches, with1 or60distinct weight matrices per graph. All305matrix comparisons across five projection geometries match V12 exactly; compute-sanitizer terminal exit0,zero errors. Managed allocation and CPU packing are setup only; all matrices execute before timing.

At60matrices, gate/up original12.881us vs tiled10.789us(-16.24%); down8.728us vs7.170us(-17.86%). Each of these matrix sets is106.168MB. K/V9.263vs9.281us,Q3.175vs3.257us,O2.817vs2.800us offer no compelling improvement. Smaller shapes have smaller aggregate working sets and are not equivalent cache-pressure experiments. Single-matrix gate/up6.112vs6.400us and down5.188vs5.359us regress. Thus cached microbenchmarks masked a useful weight-layout candidate for the larger projections. Timings are nonexclusive diagnostics after Blender restoration and do not prove serving improvement.

Next coherent integration batch: (1) owned capture-time packing for gate/up/down only, preserving original tensors for multirow prefill; (2) tile-aware M1 kernels selected only with validated packed parents; (3) allocation lifetime/extent/digest and partial-failure cleanup bound to the owned graph session. This adds159252480bytes for30layers x3large matrices unless reuse is proven safe. Avoid untracked native allocations and preserve ledger/zero-allocation shutdown proof. Existing recorder currently accepts273original weight parents and uses one weights table for both captures; the packed parents must be explicit in reservation ownership and only substitute the selected pointers for M1 capture. Validate loaded full-model logits, partial abort/close and sanitizer, then compare frozen V12/new/vLLM in matched serving. No production source change or serving gain claimed yet. Sources/logs and manifest saved alongside report; goal active.


## V14 owned tiled weights and Round30 matched serving

Previous turn established tiled-layout diagnostic evidence (progress). Implemented three related changes: owned cold-packed gate/up/down parents, tile-aware M1 capture while preserving original prefill weights, and resource extent/alias/lifetime plus source dependency/digest coverage. Native optional363-parent table includes90unique packed tensors with exact1769472byte extents. Extra persistent GPU allocation159252480bytes. CPU packing and upload happen before capture. Main commit31ecb63aa9a605ee341592f85bc1a1cc9d62d2da; frozen binary e46efaf2c2836562ba589d70bccc811a1e5f7aec9efd6d12744da3647448ceda. Follow-up test-only commit91d5117 adds four packed-parent rejection cases and serializes in-process native graph capture.

Loaded model231replays: every reference logit exact, pending result close/abort passes, zero final allocations. Compute-sanitizer231replays passes with0errors. Original and tiled native recorder tests both pass, including malformed packets, partial output suppression, parent count/extent/alias rejections and zero allocations. Initial parallel test execution hit native MLP instantiate-domain busy; added a test-process mutex and reran default test execution successfully. Initial cargo invocation selected wrong package, and first proper invocation omitted --ignored; neither is counted as GPU validation. Final direct GPU execution and sanitizer receipts are authoritative. Release rebuilt explicitly with cuda,server and --bin riley.

Round30 uses identical model/tokenizer/GPU/client and both prior workloads, C1/max-active1,128token batch/chunk, two reversed orders,12warmups+120retained per phase,1440retained responses. Zero failed phases, V12/V14 references exact480/480perlane, vLLM fixed240/240 and variable160/240. These are numerical observations, not broad model-quality or high-concurrency acceptance.

| Workload/lane | Throughput tok/s | TTFT ms | TPOT ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| fixed / V12 | 392.456 | 5.540 | 2.374 | 79.897 | 80.230 |
| fixed / V14 | 416.482 | 5.545 | 2.219 | 74.992 | 75.320 |
| fixed / vLLM | 808.017 | 7.128 | 0.888 | 36.023 | 42.979 |
| natural / V12 | 358.796 | 5.573 | 2.420 | 392.535 | 394.098 |
| natural / V14 | 381.496 | 5.568 | 2.249 | 371.773 | 373.079 |
| natural / vLLM | 849.743 | 7.770 | 1.011 | 154.067 | 159.949 |

Throughput improves6.12% fixed/6.33% variable and TPOT falls6.54%/7.06% versus V12. Keep the batch as a measured improvement with159MB memory cost. Against vLLM, throughput remains48.46%/55.10% lower and TPOT149.90%/122.33% higher. Tail samples do not prove stability. Full goal remains active and unmet.

Blender restoration freshly verified at exact command/cwd/GUI environment/tag/start/private driver mappings and ports:1743798/start87638573/9876,1743877/start87638625/9911,1743989/start87638678/9887. Helper SHA1a86ef44e987b9f4ee8aacee0b504c482a58dd9cec510bd34e561ee1ec9a7eed derives from round29 helperdc9e92115d01288ab6fdcf5ae8cac78e87658dd76bd639656e0612dd3b0afde0. Private restoration journals/session logs excluded from export.

Post-restoration nonexclusive Nsight:12prefill+228decode, exact references. Decode median GPU kernel span2081.948us vs V122228.473us; projection70.889%, attention17.041% of kernel time. Prefill4612.747us unchanged. This confirms the intended GPU region improved but is not matched performance evidence. Projection remains dominant; subsequent work should examine reducing repeated projection work/launches and multi-request row sharing, with the variable-shape scheduler contract retained. V3 is still single-active and requires multiactive execution before high-concurrency goal qualification.

120evidence files SHA256verified against raw/serving-round30-manifest.json; nine reproduction scripts saved. Production runtime source fixed at31ecb63, test-only successor91d5117.


## V15 variable multirequest scheduling foundation

Previous turn completed V14 implementation and matched serving comparison (progress). Revalidated clean remote source91d5117d4a6a34c9be3cc4996cf1db877662142b. Current27c02d14223dc4a941bee690768c601a29dc8fcf adds explicit VariablePrefillDecodeN host policy: one bounded variable prefill chunk OR homogeneous independent decode rows, alternating ready classes only after dispatch. NotDispatched abort does not advance fairness state. Configuration requires capacity1/2/4/8, budget covering decode capacity and<=1024, chunk<=1024, context<=4096, fixed Smol KV geometry. Promises use each request's actual maximum length, unlike fixedP128 envelope. Vocabulary bounds retained for request/output validation. General and fixedP128 policies remain supported; no server opt-in or GPU capacity gate changed.

Live scheduler/authority test submits P16/P128/P398/P73 with O32, verifies isolated partial prefill, at least3simultaneous decode rows, canonical V3 packet encoding from live page ownership, dense output slots, eventual completion and NotDispatched retry reproducing same request/input/target work. Runs both all-complete and active-request cancellation variants. Synthetic results prove scheduler contracts only. Entire riley-scheduler CPU suite133passed0failed, including prior fixedP128 policy cases and doctest. CUDA-gated test binaries execute0tests in this run and are not counted as GPU proof. Three evidence files verified with SHA256; manifest and reproduction sources saved.

Remaining batch work: native V3 currently validates only one active row, result wire is a single98432byte payload, session issues one cookie, executor adapter assumes one output. Implement retained per-row completion/identity and actual shared-row GPU projection/attention before exposing this policy in serving. Do not simulate concurrency with repeated M1 launches and claim batched compute. Frozen V14 remains the serving baseline. No new performance claim or Blender pause; full goal active.


## V16 shared GPU projection and request-specific attention primitives

Previous turn implemented and tested V15 scheduler foundation (progress). Source revalidated clean at27c02d14223dc4a941bee690768c601a29dc8fcf. Current4220668f20e21b6e171659fcb3cfe8dcd3678547 adds actual shared-weight MMA projection for1..8independent decode rows, with parallel original BF16 rounding chunks and ordered per-row merge. Supports original Q/K/V/O layouts and tiled gate/up/down layouts. A single launch shares each weight tile across rows rather than looping M1 launches. Scratch uses chunk-major [chunk,8,N] FP32. Separate attention scores/values kernels launch across request/head pairs, using416u32 row descriptor stride and each request's own256-entry physical page map. Score scratch8*9*4096FP32; input/output rows576BF16. These primitives require upstream validated geometry/authority and are not yet connected to the native model recorder.

GPU validation five projection geometries x3seeds x active1..8 =120valid batch cases (540reference row comparisons), plus active0/9 no-write rejection cases. Attention1..8active rows compares36reference rows with contexts1,16,17,127,128,129,2049,4096 and disjoint permuted page maps across2048physical blocks. Every BF16 output exactly matches corresponding V14 M1 execution. Inactive output rows and16element trailing guard remain unchanged; malformed row counts0/9 cause no writes. Final compute-sanitizer memcheck terminalexit0,0errors. Earlier smaller-context pass also passed, superseded by retained full4096 run. All tests use synthetic tensors, not full-model or serving evidence.

Next integration must bind embedding/RoPE/KV scatter to per-request descriptors, emit per-row identity+logit records, retain and settle every cookie, and capture shared projection/attention/head under ledger ownership. The result remains single-row in current serving. No performance claim, no server activation, no Blender pause. FrozenV14 remains serving baseline. Three evidence files exported and SHA256verified with manifest; primitive test is tracked kernels/tests/shared_decode_v16.cu. Full goal remains active.


## V17 shared thirty-layer model composition

Previous turn implemented and validated shared projection/attention primitives (progress). Source revalidated clean; current16d4e159d7a1a797fac1701dcaa19668d36bebdb composes shared projection, per-request embedding and RoPE/KV scatter, independent-context attention, norm/residual and SwiGLU across30layers. Final normalized hidden rows remain in scratch1. Pointwise kernels reuse original arithmetic with metadata active-count binding; each request's last position and416u32-stride page map govern KV writes. Scratch7 requires8*9*4096*4bytes. All projections currently use original untiled weights; packed MLP integration can follow when native ownership is bound. Header is an internal primitive requiring validated descriptors/extents, not a public trusted execution boundary.

Loaded immutable fixture validation prefills8real requests using P16/P128/P398 prompts on disjoint permuted64-page reservations (512physical pages,1024context). For active1/2/4/8,8consecutive decode steps each compare one shared CUDA Graph replay against independent M1 execution on identical full KV state.32shared graph replays,120row-step comparisons; every final hidden byte and entire key/value allocation match exactly. Decode tokens are identical predetermined17+request+step inputs, not model-selected greedy generations. Full-model arithmetic/cache parity is proven for this scope; final head/logits, completion identity and scheduler settlement are still absent. Ordinary run and compute-sanitizer terminalexit0,0errors. Fixture weights/RoPE/request hashes and test binary hash recorded. No timing/serving claim; standalone harness owns and explicitly frees CUDA resources, while native reservation-ledger linkage remains next work.

Next: connect per-row final head and result records to retained cookies/owner/replay identity, native validation and resource ownership, then scheduler/HTTP routing under the V15 policy. Full serving benchmark and high-concurrency qualification must follow the whole batch. No Blender pause. Three evidence files SHA256verified and exported with manifest; tracked kernels/tests/full_shared_model_v17.cu is the final reproduction source. Full goal active.


## V18 per-row GPU completion and Rust validation

Previous turn produced full shared hidden/KV parity (progress). Current50e7dbfbfff2153ba2535024915fa8f44519427e introduces fixed eight-record readback, each98432bytes (128identity/status+49152BF16logits), total787456bytes. Records follow descriptor order and bind explicit output slots, owner/replay/iteration, request/cookie, progress and digest. Rust validates every active row before returning any RowResult and rejects all nonzero inactive bytes. Existing single-result validator reuses the same indexed identity/logit validation. No live session or native recorder calls the new batch API yet.

GPU completion kernel independently copies each row's logits, computes finite-checked argmax with lowest-token tie policy, stamps descriptor identity and clears inactive records. Synthetic1/2/4/8row GPU output matches Rust-generated reference bytes exactly; actual GPU result files are then reread by Rust validate_batch_result. Four GPU cases pass, injected NaN produces error status; compute-sanitizer0errors. All five variable_wire tests pass, including single decode, partial-prefill suppression, stale replay/foreign KV rejection, every request-byte mutation and every active result-header-byte mutation. Initial compile lacked math_constants.h; fixed include and final compile/run passed. This proves completion serialization/validation, not logits from a shared model head or serving performance.

Next coherent integration: prepare retained multirow head/resources, connect shared model output through head to these GPU result records, reserve exact787456byte output and input staging extents, bind multi-cookie session/scheduler settlement and V15 server routing. FrozenV14 remains serving baseline. No Blender pause or performance claim.18evidence/fixture files SHA256verified against raw/batch-result-v18-manifest.json. Goal active.


## V19 loaded shared model through head and GPU completion

Previous turn implemented GPU/Rust result contract parity (progress). Current661bdd3d5731468e49b2d76fcc821ad577d9ef6d composes the loaded shared model, cuBLASLt eight-row head and V18 result kernel in one standalone CUDA Graph. Added explicit inactive hidden zeroing before fixed-capacity head reads. Test selects first successful workspace0/no split reduction cuBLASLt heuristic under BF16/FP32 accumulation; both M1 and M8 select algorithm21. Native reservation binding and production plan selection are not exercised by this standalone harness.

Real P16/P128/P398 prefills seed disjoint per-request KV; active1/2/4/8 each execute8consecutive steps.32shared graph replays and120row-step comparisons match independent M1 full hidden, all KV bytes and every49152BF16logit. GPU result argmax matches independent host argmax; every inactive completion record is zero. Same predetermined tokens in both lanes (common-history arithmetic validation, not greedy generation or broad quality qualification). Final ordinary run passes; compute-sanitizer terminalexit0,0errors after explicit inactive-row initialization. Fixture hashes and binary hash recorded. Standalone graph includes output serialization but uses minimal descriptor identities; V18 separately proves identity serialization and Rust validation. These do not substitute for live native/session authority.

Remaining integration is native owned graph/resource binding, exact transfer extents and multi-cookie Rust/session/scheduler/HTTP routing. Preserve model/head arithmetic and use the V15 homogeneous-stage policy. FrozenV14 remains matched serving baseline; no new performance claim or Blender pause. Three artifacts exported and SHA256verified with manifest. Full goal active.


## V20 native reservation and shared transfer binding

Previous turn validated standalone loaded shared model/head/results (progress). Currentf9e23f5c316538b83c3cb994ea2d37b747044ffe adds record_v3_shared to C ABI, Rust FFI and borrowed reservations. Existing V3 single-row entry remains. Shared recorder leases26device parents (optional workspace index22),273or363weight parents and bothM1/M8plans. New native devices23/24/25 are exact9216byte hidden,786432byte logits and787456byte result. Scratch7 minimum1179648bytes; capacity8..1024. Shared head is bound with existing algorithm21/splitK1/workspace0 checks. Pinned staging>=1574912; requestH2D17536 and resultD2H787456 use retained offsets. Prefill records preserve original model/head and zero trailing outputs; decode records run shared model, copy initialized8hidden rows, execute shared head and produce all result records. Current shared model uses original untiled weights. All row contexts are checked against retained RoPE context, not only row0.

Real native reservation GPU test uses zero weights, three prefills and decode2/4/8, exact row slots/tags, zero inactive output and zero final allocations. Invalid mutable alias/M1-as-shared-head rejected before capture; oversized context in last row rejected and prior output hidden. Initial test decode packet retained prefill token area and was correctly rejected; fixed canonical zero padding, final test passes. Compute-sanitizer terminalexit0,0errors. Legacy original/tiled recorder tests both pass. This is native ownership/lifetime proof on zero weights, separate from V19 loaded-model numerical proof. No loaded owned shared Rust session or HTTP execution yet.

Next: extend VariableGraphBuffers with shared parents/head, bind loaded executor and digest to new recorder, retain multiple cookies/results through scheduler completion, activate V15 policy only once GPU/CPU contracts are complete. FrozenV14 remains serving baseline. No Blender pause or performance claim. Six logs/patch artifacts SHA256verified and saved with manifest. Goal active.


## V21 loaded owned multirow session and scheduler

Previous turn implemented native shared reservation and transfer (progress). Current4dd3d7f3b514f6663126f639d0073146f085033d connects loaded model to shared recorder, retained shared head/scratch/output/staging parents, multi-cookie issuance, full batch validation and slot-based scheduler logits. Shared buffers omit unused packed MLP weights, avoiding159MB persistent duplication. Capacity is at least8 for shared scratch but zero/oversized preparation remains invalid. Digest binds shared code, mode and both head algorithm metadata. The owned wrapper closes native reservation before graph parents/model; results remain outstanding until exact scheduler iteration commit.

GPU test runs P16/O32,P128/O64,P398/O128 concurrently through V15 policy with73token chunks. Maximum3decode rows,224generated-token full logits exactly match immutable loaded-RoPE references,144settled iterations plus one uncommitted partial prefill. Extra issue/wrong commit rejected; close native session then abort pending scheduler reservation succeeds; allocations zero. Compute-sanitizer passes0errors. Existing single-owned session regression231replays retains all exact logits, pending close/abort and zero allocations. Scheduler CPU suite133passed. Final cargo CUDA check passes after comment/zero-capacity guard refinement; accepted-capacity numerical path unchanged. Tests are not serving timing.

Remaining: server must select shared owned preparation and V15 policy for multi-active variable profile and relax its currentmax-active1 gate coherently; then validate real HTTP concurrent streams/cancellation and repeat matched V14/new/vLLM serving across concurrency. Current server still routes single-row V3. Eight evidence files SHA256verified and exported with manifest. No Blender pause/performance claim. Goal active.


## V22 concurrent HTTP serving and Round32 C4 comparison

Previous turn completed loaded shared session/scheduler verification (progress). Current52be6c4091afdd1bfaa44350a1ee5e3753d9185a wires capacity2/4/8 to shared owned preparation and VariablePrefillDecodeN; capacity1 retains original V3. Fixed eager metadata rows/output slots are1 for V3, separate from shared graph scratch. Initial startup exposed max-input1 vs eager rows4; corrected this before final HTTP validation. CLI parser test now checks parsing supported capacities, while runtime configuration performs rejection. CPU93passed1ignored with process-localumask077.

Release SHAffcd95f0ce2975a3787ce860f602bee5319b91676b1bed83591b214f56d9ccaf frozen at variable-candidate-v22. Actual HTTP:13completedresponses,928tokens, including8barrier-concurrent mixed ordinary/SSE requests at active4 and256KVblocks. Every prompt/output token reference and requested output count matched. Output129 rejected400; live stream disconnect followed by correct response. Shutdownexit0, active/waiting/KV/device/pinned counts/bytes0, cancellation1/disconnect1. GPU timing/shape metrics still degraded; no fabricated batch metrics.

Round31 was interrupted as invalid for sharedC4: template Riley argv hardcoded active1 despite clientC4. No Round31 performance is promoted. ControllerSIGINT triggered finally restoration and fresh exact verification. Round32 explicitly sets and asserts launch active capacity before every phase. SameC4 external workload/model/tokenizer/GPU/client; V14 baseline necessarily active1, V22/vLLM active4. This baseline capacity limitation is explicit, so relative improvement includes concurrency support. Two reversed orders,12warmups+120retained per lane,1440retained total,zero failed phases.

| Workload/lane | Throughput tok/s | TTFT ms | TPOT ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| fixed / V14 active1 | 422.814 | 139.442 | 2.234 | 189.490 | 190.252 |
| fixed / V22 active4 | 828.299 | 8.688 | 4.468 | 150.657 | 151.548 |
| fixed / vLLM active4 | 1703.429 | 12.712 | 1.697 | 72.701 | 82.138 |
| natural / V14 active1 | 383.033 | 589.439 | 2.254 | 955.921 | 956.542 |
| natural / V22 active4 | 795.327 | 9.271 | 4.789 | 656.422 | 659.188 |
| natural / vLLM active4 | 2321.909 | 16.328 | 1.475 | 224.136 | 231.475 |

Throughput improves95.90%/107.64% vs baseline, but TPOT worsens100.02%/112.48%. Against vLLM throughput remains51.37%/65.75% lower and TPOT163.25%/224.68% higher. Goal clearly unmet. V14/V22 references480/480each, vLLM fixed229/240 and natural132/240; no broad numerical-quality acceptance. Short120sample tails are not stability evidence.

Round32 authorized Blender pause/restoration freshly verified at PIDs2040030/start87960064/9876,2040117/start87960116/9911,2040210/start87960169/9887. Helper628ef742e555a231e1e04dd25a1617185ec941d8a35936a93a92d16fd1da6831 derives from round31helper9f5a1fac2e0bdbe3693d06a42115eba06685a50caab41d729e7b011dad7124b1. Private snapshots/journals/session logs excluded.

Post-restore nonexclusive Nsight initially used client1/active4 (228decode,median2835.907us); retained as such, not C4. Corrected separate C4-client run confirms12prefill+73decode graph replays, exact references. Decode median2933.394us; projection61.018%,attention29.614%. Prefill4762.958us. Next batch should address shared projection/attention and expensive prefill interleaving; this profile localizes GPU costs but cannot by itself quantify end-to-end scheduling attribution. Full high-concurrency/stability qualification remains.

187evidence files SHA256verified against raw/serving-round32-manifest.json;14reproduction scripts saved. Goal active.

## V23 shared decode optimization and Round33 matched C4 comparison

Source a399f04cbe853f836567a66e0689de0c3a51e09f; release SHA7c879c24f8af08c536ec33f58d005ce51c76565ca1e927561ce4f9f109c016b7. One batch combines bounded score dispatch (at most32 token-tile CTAs/head with runtime grid-stride traversal) and packed gate/up/down weights for shared projection. It preserves each QK MMA recurrence and projection chunk rounding/order. Packing adds90 retained weight buffers,159252480device bytes, plus startup packing work; hot-path weight reads become contiguous. Original weights remain for prefill. Shared recorder accepts validated273 or363 weight parents; loaded shared preparation now supplies363.

Primitive GPU tests: projection geometry/seeds/rows1..8 and invalid0/9; attention contexts1,16,17,127,128,129,2049,4096 with permuted/disjoint pages; exact results/inactive guards and compute-sanitizer0errors. Actual loaded scheduler224full-logit comparisons match immutable references,144settled iterations, pending close/abort and zero allocations. HTTP13completed responses/928tokens including8barrier-concurrent ordinary/SSE requests, invalid bound400 and post-disconnect recovery; all token references exact. Shutdownexit0 and active/waiting/KV/device/pinned counts/bytes0.

Round33 uses same model/tokenizer/GPU/client/workloads and active4 for V22,V23,vLLM, two reversed orders,12warmups+120retained per lane. Unlike Round32, previous and new active capacity and KV budgets are identical.1440retained requests,0failures. Medians of two runs:

| Workload/lane | Throughput tok/s | TTFT ms | TPOT ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| fixed / V22 | 829.277 | 8.633 | 4.440 | 154.664 | 156.917 |
| fixed / V23 | 958.628 | 7.998 | 3.810 | 131.732 | 132.133 |
| fixed / vLLM | 1721.981 | 12.591 | 1.700 | 69.815 | 78.457 |
| natural / V22 | 796.839 | 9.200 | 4.781 | 655.832 | 658.211 |
| natural / V23 | 906.241 | 8.600 | 4.196 | 579.647 | 582.232 |
| natural / vLLM | 2272.249 | 16.687 | 1.521 | 231.689 | 242.371 |

V23 throughput improves15.60%/13.73% and TPOT improves14.18%/12.23% vs V22 (fixed/natural). E2E P99 improves15.79%/11.54%. Relative to vLLM, throughput remains44.33%/60.12% lower and TPOT124.08%/175.86% higher. TTFT is lower but does not meet the combined goal. V22/V23 references480/480 each; vLLM fixed230/240 and natural130/240. Numerical differences are observed, not a broad quality qualification; short runs cannot establish high-concurrency stability.

Authorized Blender restoration freshly verified:2114072/start88039578/9876,2114144/start88039629/9911,2114282/start88039684/9887. Helper3d783bd1cd2f86b698ceb6ad88b2b4b01845a6995f0c82ac39daa159fc850fa9 derives from Round32helper628ef742e555a231e1e04dd25a1617185ec941d8a35936a93a92d16fd1da6831. Exact command/cwd/GUI environment/private vendor libraries/listening ports restored. Private snapshots/journals/session-stop/restore logs excluded.

Post-restoration nonexclusive C4-client Nsight:12prefill+73decode; exact tokens. Decode GPU kernel span median2933.394us(V22) to2296.430us(V23), prefill4750.941us essentially unchanged. Gate/up mean13.231us to11.189us; down8.367us to6.909us; scores18.373us to2.818us. Shared projection now70.64% and attention17.24% of decode kernel time. These traces support the bottleneck explanation, but matched performance comes from Round33. Next batch should reduce projection cost and remaining repeated attention-value work, while tracking unchanged prefill cost.

122evidence files SHA256verified against raw/serving-round33-manifest.json;10reproduction scripts saved. Goal remains active.

## V24 grouped projection dispatch and Round34 matched C4 comparison

Previous turn is progress: V23 measured a serving improvement. Current source190dec94b99af52ba7c17fb34c628506c4302841 groups QKV into one parts kernel and one ordered merge, and gate/up into one packed projection launch. Each original projection retains its MMA recurrence and BF16 chunk rounding. Shared compute helper is reused by separate and grouped launchers. No additional retained model buffers; QKV parts use92160bytes within existing shared scratch. This removes5kernel launches/layer and increases independent work available per projection dispatch.

Release SHA0a67d1df8a16b10c51d7bd25f0d0372bc593d250d3bd600c39b7771aa3ff796f. Dedicated grouped primitive test compares QKV/gate-up to original M1 projections for active1..8 and invalid0/9, with inactive guards; compute-sanitizer0errors. Existing primitive suite passes. Actual loaded shared scheduler224full-logit comparisons/144settled iterations pass with zero allocations and pending close/abort. HTTP13responses/928tokens including concurrent ordinary/SSE, invalid-bound rejection and live-disconnect recovery pass; serverexit0 and shutdown resource counts0. An auxiliary test command unintentionally repeated the existing build/test suite before the dedicated grouped test; all runs completed, and these nonexclusive tests are not performance evidence.

Round34: V23/V24/vLLM all active4, same model/tokenizer/GPU/workloads/client and Riley KV budgets. Two reversed orders,12warmups+120retained/lane,1440retained requests,zero failures. Two-run medians:

| Workload/lane | Throughput tok/s | TTFT ms | TPOT ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| fixed / V23 | 961.017 | 8.011 | 3.801 | 131.524 | 131.842 |
| fixed / V24 | 1199.332 | 7.185 | 2.991 | 102.989 | 107.145 |
| fixed / vLLM | 1728.194 | 13.414 | 1.690 | 69.832 | 79.922 |
| natural / V23 | 905.844 | 8.831 | 4.142 | 582.638 | 585.279 |
| natural / V24 | 1122.068 | 7.847 | 3.377 | 473.853 | 478.106 |
| natural / vLLM | 2300.908 | 16.635 | 1.491 | 226.341 | 234.893 |

V24 vs V23 throughput+24.80%/+23.87%, TTFT-10.31%/-11.15%, TPOT-21.31%/-18.47%, E2E P99-18.73%/-18.31% (fixed/natural). V24 vs vLLM throughput remains30.60%/51.23% lower and TPOT76.98%/126.46% higher. V23/V24reference480/480each; vLLMfixed233/240,natural131/240. Goal unmet; high concurrency, long-run stability and broad quality remain unqualified.

Round34 authorized Blender pause and restoration freshly verified exact command/cwd/GUI environment/private libraries/ports at2158635/start88086531/9876,2158760/start88086583/9911,2158877/start88086636/9887. Helper8bee136910c6633156bcb1108758666d1fd477cc0e98d3954c7ba8b22832cf35 derives from Round33helper3d783bd1cd2f86b698ceb6ad88b2b4b01845a6995f0c82ac39daa159fc850fa9. Private journals/session logs excluded.

Post-restoration nonexclusive NsightC4:12prefill+72decode, all12streaming references exact. Decode GPU kernel span median1421.278us vs V23 2296.430us; prefill4753.886us unchanged. Analyzer updated to count new grouped projection names: projection53.36%,attention27.49% of decode kernel time. Between D2H completion and next H2D start median822.184us (V23 822.181us), p951073.834us. This gap includes sampling/scheduler/transport/dispatch and is not isolated CPU sampling proof. Next step should profile these host stages alongside remaining GPU attention/projection before selecting the next batch; unchanged prefill remains relevant.

125evidence files SHA256verified,12reproduction/analysis scripts saved. Goal active.

## V25 CPU sampling/validation batch and Round35 matched C4 comparison

Previous turn is progress: V24 measured grouped-projection improvement. Before selecting this batch, temporary per-stage wall timers on V24 actual HTTP measured active4 completion validation225us and sampling517us median. This is nonexclusive diagnostic instrumentation, not matched serving timing. V24 diagnostic binary SHA10249999fa24722ac1846e4435ad5ab651034497859983ac7f5d6d124e652e5c and source patch retained.

Source a12e94cd5df75963d8af3bd93420eb553465dfda adds two changes: vectorizable OR reduction over all inactive/partial result bytes; deterministic temperature0/repetition1 sampling fast path that selects the winner directly and materializes the same full public distribution without candidate/history work. Input/history/mask/parameter and nonfinite-logit validation remains, as do masked-token behavior, ascending-ID tie breaking, signed zero and public probabilities/log-probabilities/processed-logits. Other sampling parameters retain the existing path.

Runtime CPU309passed/23ignored. Differential test uses the previous general processing path as a test-only reference over vocab sizes1,2,17,257,49152, eight cases each, masks/all-masked, signed zeros and duplicate history; complete distributions compared. Existing nonfinite/error/sampling tests pass. HTTP13responses/928tokens with eight concurrent ordinary/SSE requests, invalid-bound rejection and disconnect recovery all match token references; shutdownexit0/resources0. No CUDA kernels changed in this batch. Production release SHAf9ab84f34f155b3378e4c2adedf91850c190d2d73f28d8131720ed9e4bbf7783.

Round35: V24/V25/vLLM active4, same workload/model/tokenizer/GPU/client and Riley KV budgets. Two reversed orders,12warmups+120retained/lane,1440retained requests,zero failures. Two-run medians:

| Workload/lane | Throughput tok/s | TTFT ms | TPOT ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| fixed / V24 | 1198.086 | 7.193 | 2.997 | 105.458 | 110.875 |
| fixed / V25 | 1354.487 | 6.696 | 2.620 | 93.533 | 96.324 |
| fixed / vLLM | 1619.718 | 13.588 | 1.757 | 74.181 | 96.748 |
| natural / V24 | 1122.858 | 7.826 | 3.377 | 473.643 | 476.447 |
| natural / V25 | 1249.598 | 7.344 | 3.028 | 427.665 | 428.644 |
| natural / vLLM | 2291.396 | 17.795 | 1.479 | 226.584 | 233.452 |

V25 vs V24 throughput+13.05%/+11.29%, TPOT-12.58%/-10.32%, E2E P99-13.12%/-10.03% (fixed/natural). V25 vs vLLM throughput remains16.38%/45.47% lower and TPOT49.14%/104.67% higher. V24/V25reference480/480each; vLLMfixed234/240,natural131/240. FixedP99near-equality in this short screen is not stability qualification. Goal unmet.

Round35 authorized Blender restoration freshly verified exact command/cwd/GUI environment/private libraries/ports:2203734/start88135316/9876,2203801/start88135369/9911,2203905/start88135422/9887. Helper3fbe3804dd59c282b4bb472c38de3a5ce5497b3521dc87909c55566b99514306 derives from Round34helper8bee136910c6633156bcb1108758666d1fd477cc0e98d3954c7ba8b22832cf35. Private restoration journals/session logs excluded.

Post-restore instrumented V25 HTTP confirms active4 validation154us and sampling240us; active1 validation43us and sampling66us (before168us/133us). Stage-conditioned medians only: iterations differ586before/621after because scheduling changes. Diagnostic after binary SHA12f8fef23fb2cf3b4b6bf8f88014085be14e59cedcdc227f55ae1a4d97366ad8. Instrumentation patches archived and removed; source clean; release rebuilt and SHA exactly equals frozen production V25. No instrumentation remains in measured candidate.

Next batch should return to remaining GPU projection/attention and prefill costs, while using the reduced host baseline. High concurrency and sustained stability still require qualification after throughput/latency gaps close; no microbenchmark is substituted for the serving goal.126evidence files SHA256verified,16scripts saved. Goal active.

## V26 attention reuse and Round36 C4/C8 serving comparison

Previous turn is progress: V25 measured CPU improvement. Current source6818d7264a4aa845c15813ba5bdcfd419daeb14d changes the shared attention value kernel in one batch: reuse unrounded FP32 exponentials from probability construction for the original lane-local denominator order, and reuse the physical KV page base across each aligned K16 tile. Adds512bytes shared memory per CTA; no persistent allocation change. Warp barriers protect producer/consumer accesses; tile traversal, MMA order and BF16 probabilities/output rounding remain unchanged.

Primitive full projection/attention suite passes with active1..8, invalid0/9, disjoint/permuted pages and contexts1,16,17,127,128,129,2049,4096; compute-sanitizer0errors. Actual loaded shared scheduler224full-logit comparisons/144settled iterations with zero allocations and pending-close cleanup pass. Actual HTTP suites at active4 and active8 each complete13responses/928tokens with concurrent ordinary/SSE, output-bound rejection and live-disconnect recovery; all prompt/generated token references match and shutdown resources0. Release SHAc41f7e114dc37f02af26b64ecc37d69e52c35c4218b5e138c6d2af915034f2b4.

Round36 expands the same serving screen to C4 and C8. PreviousV25/newV26/vLLM use matching active capacity and external workload; Riley KV budgets10blocks/request fixed or64blocks/request natural. Two reversed orders,12warmups+120retained/lane,2880retained total,zero failures. Two-run medians:

| Workload/lane | Throughput tok/s | TTFT ms | TPOT ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| c4-fixed / V25 | 1357.713 | 6.681 | 2.618 | 93.452 | 93.848 |
| c4-fixed / V26 | 1438.151 | 6.527 | 2.460 | 85.773 | 86.520 |
| c4-fixed / vLLM | 1705.664 | 13.388 | 1.703 | 72.586 | 79.360 |
| c4-natural / V25 | 1252.483 | 7.327 | 2.987 | 427.140 | 428.108 |
| c4-natural / V26 | 1515.903 | 6.778 | 2.490 | 356.093 | 356.990 |
| c4-natural / vLLM | 2305.495 | 17.475 | 1.478 | 224.805 | 233.826 |
| c8-fixed / V25 | 1780.122 | 10.869 | 3.979 | 147.326 | 162.422 |
| c8-fixed / V26 | 1856.459 | 10.815 | 3.720 | 141.676 | 153.658 |
| c8-fixed / vLLM | 2423.188 | 18.662 | 2.331 | 101.481 | 108.355 |
| c8-natural / V25 | 1818.316 | 11.571 | 4.087 | 563.263 | 606.898 |
| c8-natural / V26 | 2098.643 | 11.031 | 3.536 | 491.366 | 542.328 |
| c8-natural / vLLM | 3665.454 | 16.924 | 1.868 | 275.607 | 294.926 |

V26 throughput vs V25: C4fixed+5.92%,C4natural+21.03%,C8fixed+4.29%,C8natural+15.42%. TPOT decreases6.04%,16.65%,6.52%,13.48% respectively. All corresponding P95/P99 improve, but120samples/run remains insufficient for sustained stability. Against vLLM throughput remains15.68%,34.25%,23.39%,42.75% lower and TPOT44.45%,68.48%,59.58%,89.27% higher. Thus C8 adds evidence but does not satisfy the serving goal. V25/V26reference960/960each; vLLM C4fixed232/240,C4natural128/240,C8fixed229/240,C8natural137/240; broad numerical quality remains separate.

Round36 authorized Blender restoration freshly verified exact command/cwd/GUI environment/private libraries/ports:2271185/start88207263/9876,2271241/start88207315/9911,2271312/start88207368/9887. Helper1c795763acba75fc7de761591d12db5c138e295177736c39017ad6ca3c210557 derives from Round35helper3fbe3804dd59c282b4bb472c38de3a5ce5497b3521dc87909c55566b99514306. Private journals/session logs excluded.

Post-restoration nonexclusive C4 Nsight:12prefill+74decode, all12streaming references exact. Decode median GPU kernel span1266.729us (V24GPUbaseline1421.278us; V25 changed CPU only). Attention values mean4.344us. Decode projection60.57%,attention17.45%; prefill4736.066us remains essentially unchanged, with projection69.01% and attention24.72%. Next optimization batch should address prefill projection/attention and interleaving costs; the current kernel gain is confirmed in matched serving above, not inferred solely from this diagnostic trace.

222evidence files SHA256verified,11reproduction scripts saved. Goal active.

## V27 prefill projection batch and Round37 C4/C8 serving comparison

Previous turn is progress: V26 measured attention gains. Current source7f0865f07433f9af53931df1e308f9e8b94692f7 routes already-retained packed gate/up/down weights into M16 prefill projection and unrolls depth by4. Each MMA/BF16 chunk recurrence remains ordered. Native shared recorder holds a separate packed prefill weight-pointer catalog, while shared decode retains its original/packed catalog. Single-row recorder likewise uses packed weights for both prefill and decode. No additional retained weight buffers.

Primitive compares old unroll1/original weights against current unroll4 and packed large projections at rows0,1,7,15,16,17,73,128,398,1024 over all five geometries; exact arrays match, including inactive sentinel regions. Compute-sanitizer0errors. Actual shared loaded scheduler224full-logit comparisons and existing single-owned231replay regression pass with zero allocations/pending-close cleanup. Active4 and active8 actual HTTP suites each13responses/928tokens pass, including eight concurrent ordinary/SSE, invalid bounds and disconnect recovery; token references exact and shutdownresources0. Release SHAebd099a5f4d201c904c24b055835178e4907afed2a4e74add530ff89dbcd7599.

Round37: V26/V27/vLLM same model/tokenizer/GPU/client/workloads with matching C4/C8 active capacities and Riley KV budgets. Two reversed orders,12warmups+120retained/lane,2880retained total,zero failures. Two-run medians:

| Workload/lane | Throughput tok/s | TTFT ms | TPOT ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| c4-fixed / V26 | 1437.214 | 6.540 | 2.457 | 90.536 | 91.424 |
| c4-fixed / V27 | 1591.437 | 5.221 | 2.265 | 79.634 | 79.879 |
| c4-fixed / vLLM | 1666.329 | 13.378 | 1.732 | 75.069 | 82.977 |
| c4-natural / V26 | 1514.075 | 6.794 | 2.457 | 356.917 | 360.925 |
| c4-natural / V27 | 1602.768 | 5.483 | 2.364 | 332.262 | 332.696 |
| c4-natural / vLLM | 2248.182 | 16.762 | 1.508 | 233.767 | 255.332 |
| c8-fixed / V26 | 1855.023 | 9.976 | 3.852 | 141.579 | 156.601 |
| c8-fixed / V27 | 2123.402 | 7.653 | 3.406 | 122.504 | 137.227 |
| c8-fixed / vLLM | 2256.064 | 19.518 | 2.363 | 120.037 | 140.951 |
| c8-natural / V26 | 2092.241 | 11.054 | 3.542 | 493.348 | 540.071 |
| c8-natural / V27 | 2271.542 | 8.320 | 3.272 | 450.411 | 495.186 |
| c8-natural / vLLM | 3663.334 | 19.518 | 1.858 | 277.432 | 298.252 |

V27 throughput vs V26: C4fixed+10.73%,C4natural+5.86%,C8fixed+14.47%,C8natural+8.57%. TTFT decreases20.17%,19.29%,23.29%,24.73%; TPOT decreases7.83%,3.79%,11.57%,7.63%. Against vLLM throughput remains4.49%,28.71%,5.88%,37.99% lower; TPOT30.76%,56.80%,44.14%,76.11% higher. Short fixed-workload P99 below vLLM does not establish sustained stability or the combined goal. V26/V27reference960/960each; vLLM C4fixed231/240,C4natural130/240,C8fixed231/240,C8natural146/240.

Round37 authorized Blender restoration freshly verified exact command/cwd/GUI environment/private libraries/ports:2326232/start88265355/9876,2326308/start88265407/9911,2326471/start88265462/9887. Helperf403a51a119a348275e14fa1c54990059fa9b4d2fa8a7b27627e44c7328091bc derives from Round36helper1c795763acba75fc7de761591d12db5c138e295177736c39017ad6ca3c210557. Private journals/session logs excluded.

Post-restoration nonexclusive fixedC4 Nsight:12prefill+73decode, all12references exact. Prefill median3440.153us vs V26 4736.066us; decode1268.041us essentially unchanged. Prefill projection57.19%,attention34.15%. Additional naturalC4 trace:24prefill+295decode,12requests/896tokens exact; prefill3941.853us median, attention50.23%,projection43.41%; decode1437.162us median, projection52.35%,attention28.84%. Natural attention-values mean9.377us and attention-scores4.247us; prefill attention73.118us. Stage totals differ from fixed because work/context differs, and nonexclusive diagnostics are not matched serving evidence.

Next batch should target remaining long-context attention work (prefill and shared decode), retaining exact recurrence and measuring serving again. Decode projection remains a major cost; high concurrency/sustained stability and broad quality are still unqualified.233evidence files SHA256verified,14scripts/probes saved. Goal active.

## V28 prefill attention reuse and Round38 C4/C8 serving comparison

Previous turn is progress: V27 measured prefill projection gain and localized natural attention costs. Current sourcee181ae97e1202050654e887753893b609dd5ff44 hoists Q pairs into registers, reuses unrounded softmax exponentials in the original lane-local denominator sum, and interchanges PV token/output-block loops to reuse probabilities and one aligned page base. Each output accumulator retains its token/MMA sequence and BF16 rounding. Adds1536bytes shared memory per prefill CTA and eight query registers/lane in source; no retained allocation change. Contiguous and paged KV address formulas are both preserved.

Primitive compares previous kernel against current for rows0,1,7,16,17,73,128,398,1024 with contexts1..4096 under contiguous and permuted paged KV; every output exact and inactive guards preserved. Compute-sanitizer0errors. Actual loaded shared224full-logit/144iteration and single-owned231replay regressions pass with zero allocations/close cleanup. C4/C8 actual HTTP suites each13responses/928tokens pass including concurrent ordinary/SSE, invalid bounds and disconnect recovery; all token references exact, shutdownresources0. Release SHA2c7ca6d03b1b779830a1c0de15839195daf809e5051ebfce7dd2157b6f73c28b.

Round38: V27/V28/vLLM same model/tokenizer/GPU/client/workloads, matching C4/C8 active capacity and Riley KV budgets. Two reversed orders,12warmups+120retained/lane,2880retained requests,zero failures. Two-run medians:

| Workload/lane | Throughput tok/s | TTFT ms | TPOT ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| c4-fixed / V27 | 1588.938 | 5.226 | 2.267 | 81.367 | 82.378 |
| c4-fixed / V28 | 1657.978 | 4.684 | 2.191 | 77.452 | 78.734 |
| c4-fixed / vLLM | 1670.749 | 13.427 | 1.709 | 75.007 | 95.825 |
| c4-natural / V27 | 1602.640 | 5.490 | 2.363 | 332.665 | 333.279 |
| c4-natural / V28 | 1676.237 | 4.918 | 2.271 | 313.092 | 313.622 |
| c4-natural / vLLM | 2269.316 | 17.596 | 1.488 | 232.309 | 240.271 |
| c8-fixed / V27 | 2089.257 | 8.466 | 3.364 | 125.435 | 130.221 |
| c8-fixed / V28 | 2231.072 | 7.497 | 3.203 | 116.131 | 126.735 |
| c8-fixed / vLLM | 2174.338 | 19.773 | 2.422 | 125.593 | 156.384 |
| c8-natural / V27 | 2273.919 | 8.316 | 3.270 | 451.969 | 493.079 |
| c8-natural / V28 | 2425.213 | 7.599 | 3.083 | 419.714 | 454.672 |
| c8-natural / vLLM | 3731.267 | 16.481 | 1.831 | 270.952 | 284.587 |

V28 throughput vs V27: C4fixed+4.35%,C4natural+4.59%,C8fixed+6.79%,C8natural+6.65%. TPOT decreases3.35%,3.89%,4.77%,5.72%; TTFT decreases10.36%,10.41%,11.44%,8.63%. Against vLLM throughput is-0.76%,-26.13%,+2.61%,-35.00%; TPOT remains28.17%,52.64%,32.29%,68.38% higher. The narrow C8fixed throughput win and lower short-run tails do not meet the combined goal; vLLM fixed runs also vary. V27/V28reference960/960each; vLLM C4fixed234/240,C4natural127/240,C8fixed228/240,C8natural137/240.

Round38 authorized Blender restoration freshly verified exact command/cwd/GUI environment/private libraries/ports:2389841/start88331103/9876,2389888/start88331155/9911,2389993/start88331209/9887. Helperd9742d5cdc708cdceea6e4dc0327fba9661b07c583b7391543ad24c449be7901 derives from Round37helperf403a51a119a348275e14fa1c54990059fa9b4d2fa8a7b27627e44c7328091bc. Private journals/session logs excluded.

Post-restoration nonexclusive fixedC4 Nsight:12prefill+74decode; prefill median2866.716us vs V27 3440.153us, attention21.14% of prefill kernel time; decode1257.994us unchanged. NaturalC4:24prefill+296decode; prefill3096.864us vs3941.853us, attention35.41%/projection56.38%; decode1430.222us, attention28.91%/projection52.38%. Both12-request diagnostic streams match all token references. Natural decode kernel total417.880ms vs prefill80.427ms now makes decode the priority. These short traces include tail/batch-fill effects and are not full steady-state utilization proof.

Next: profile sustained decode batch fill and stage costs before choosing shared decode attention/projection fusion or scheduler work; avoid assuming short-trace average rows proves a scheduler bottleneck. High-concurrency sustained stability and broad quality remain unqualified.233evidence files SHA256verified,14scripts/probes saved. Goal active.

## V29 sustained batch-fill diagnostic, decode fusion and Round39

Previous turn is progress: V28 measured prefill attention gain. A temporary V28 diagnostic recorded actual stage, row count and execution wall time for132natural requests at each capacity. Initial C8 run accidentally retained client4; it is explicitly invalid as C8 evidence in fill-v29-analysis.json. Corrected run sets client concurrency to capacity and freezes diagnostic binaryc27be2fcfa69b7bb33a2730c2afd999e7e19c38a858693b49779d5c8e1f55712. All264corrected requests/19712tokens exact. Excluding first/last10% iterations, C4 mean decode rows3.894/full89.44%, C8 mean7.778/full78.38%. Prefill execution-wall fractions17.90%/28.61%. These are request-row fill/host-inclusive times under nonexclusive conditions, not GPU utilization or matched performance. Instrumentation archived and removed before production build. Evidence favors reducing per-iteration costs over assuming poor batch fill.

Source0526d4f3d341a3aa815f806c6f2e6cbab8d86d07 fuses QKV ordered BF16 merge with RoPE/KV write, and uses two warps per gate/up tile with rounded shared results immediately consumed by SwiGLU. Removes2kernel launches/layer plus intermediate reads/writes; persistent buffers unchanged, gate/up fusion uses256bytes shared memory/CTA. Both packed and original-weight paths supported. No relaxation of original rounding, masks or settlement contracts.

Existing primitive and actual shared scheduler224full-logit tests pass; HTTP C4/C8 each13responses/928tokens including cancellation/invalid bounds recover correctly, all references exact and shutdownresources0. Additional full-model reference comparison at active1,2,4,8 over8steps each, both unpacked and packed, verifies hidden/full KV/all logits/argmax/inactive records (120row-steps per mode); packed run compute-sanitizer0errors. Release SHAe860b320b3eced69d0fb168d0b435b81b9c652df5ca803df037f95588b688f8d.

Round39: V28/V29/vLLM matching C4/C8 workload/model/tokenizer/GPU/client/active capacity and Riley KV budgets; two reversed orders,12warmups+120retained/lane,2880retained,zero failures. Two-run medians:

| Workload/lane | Throughput tok/s | TTFT ms | TPOT ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| c4-fixed / V28 | 1656.225 | 4.686 | 2.183 | 77.609 | 78.217 |
| c4-fixed / V29 | 1689.545 | 4.631 | 2.151 | 73.248 | 75.113 |
| c4-fixed / vLLM | 1680.165 | 13.256 | 1.723 | 72.579 | 83.526 |
| c4-natural / V28 | 1675.694 | 4.923 | 2.265 | 315.577 | 316.845 |
| c4-natural / V29 | 1703.485 | 4.897 | 2.211 | 311.008 | 313.207 |
| c4-natural / vLLM | 2284.803 | 17.744 | 1.490 | 225.418 | 246.582 |
| c8-fixed / V28 | 2239.441 | 7.862 | 3.232 | 116.044 | 130.251 |
| c8-fixed / V29 | 2252.234 | 6.947 | 3.223 | 115.787 | 124.534 |
| c8-fixed / vLLM | 2327.057 | 18.988 | 2.396 | 105.067 | 124.050 |
| c8-natural / V28 | 2412.046 | 7.660 | 3.087 | 425.319 | 455.621 |
| c8-natural / V29 | 2455.997 | 7.612 | 3.044 | 415.058 | 447.765 |
| c8-natural / vLLM | 3738.879 | 17.695 | 1.832 | 269.422 | 277.393 |

V29 vs V28 throughput+2.01%,+1.66%,+0.57%,+1.82% (C4fixed,C4natural,C8fixed,C8natural); TPOT-1.48%,-2.35%,-0.25%,-1.40%. Gains are small in two short repetitions and not strong stability evidence. Against vLLM throughput+0.56%,-25.44%,-3.22%,-34.31%; TPOT24.81%,48.44%,34.53%,66.12% higher. Goal unmet. V28/V29reference960/960each; vLLM C4fixed232/240,C4natural131/240,C8fixed227/240,C8natural146/240.

Round39 authorized Blender restoration freshly verified exact command/cwd/GUI environment/private libraries/ports:2464844/start88412495/9876,2464937/start88412547/9911,2465098/start88412603/9887. Helper8b08bfbd92cc8e8eeb34fc73f92e817e8084aa5c534370e909b155eba158c9c6 derives from Round38helperd9742d5cdc708cdceea6e4dc0327fba9661b07c583b7391543ad24c449be7901. Private journals/session logs excluded.

Post-restoration nonexclusive Nsight fixed:12prefill+72decode, decode median1225.839us vs1257.994us, prefill2878.435us unchanged. Natural:24prefill+296decode, decode1390.929us vs1430.222us, prefill3107.190us unchanged. Both12request traces exact. Projection percentages now include fused RoPE/SwiGLU epilogues (natural56.63%), so not directly comparable as pure GEMM percentages. Kernel reduction explains a small gain; it does not close the serving gap.

Next priority: CPU completion/sampling duplicated work. V25 timers measured validation154us and sampling240us at4rows even after fast-path optimization; current completion validation recomputes argmax and sampling scans the same logits again. Investigate retaining the already-validated winner for eligible greedy requests and reducing validation scan cost without weakening nonfinite, tie, mask, penalty or identity contracts. Mixed prefill/decode execution remains a larger architectural option, not established as necessary by these fill measurements. High-concurrency sustained stability and broad quality remain unqualified.253evidence files SHA256verified,21scripts/probes saved. Goal active.


## V30 validated argmax reuse and Round40

Source `00c9da9c9c50d214c1f2f111c2ce8ab0d2e04d1e` retains the independently CPU-validated V3 argmax by dense output slot, reuses it only for eligible temperature-zero / unit-penalty requests with valid parameters/history and an allowed winner, and scans finite BF16 logits using eight independent maxima. Other sampling settings and masked winners use the existing full path. Identity/status/nonfinite checks, lowest-token ties (including signed zero), RNG lifecycle checks and full-logit publication remain intact.

Runtime tests310 passed/23ignored, scheduler35passed, server69passed/1ignored; real shared GPU scheduler test passed. Fixed a new test constructor requiring nonzero IterationId unwrap before these results. C4/C8 HTTP each13responses/928tokens exact, cancellation/disconnect recovery and invalid bound rejection pass, shutdown active/waiting/KV/device/pinned allocations zero. Nine additional HTTP requests per version (temperature0 with top_p0.1, stochastic temperature0.7/top_p0.9/seed7, temperature1.2/seed42 across three prompts) give exactly equal choices and usage. This does not prove broad model quality or all sampling distributions.

Frozen V30 SHA256 `6f7d248187d9b614fd0d27a85a6863d38b70464c2b69270551dd6c249bd43582`. Round40 compares V29/V30/vLLM on matching C4/C8 client/active capacity, model/tokenizer/GPU, two reversed orders,12warmups+120retained per lane;2880retained,zero failures. Two-run medians:

| Workload/lane | Throughput tok/s | TTFT ms | TPOT ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| c4-fixed / V29 | 1690.193 | 4.623 | 2.149 | 74.621 | 75.296 |
| c4-fixed / V30 | 1957.236 | 4.295 | 1.825 | 64.439 | 65.233 |
| c4-fixed / vLLM | 1716.606 | 13.052 | 1.699 | 70.665 | 76.873 |
| c4-natural / V29 | 1703.186 | 4.865 | 2.232 | 309.588 | 310.987 |
| c4-natural / V30 | 1969.510 | 4.597 | 1.899 | 270.193 | 271.127 |
| c4-natural / vLLM | 2271.719 | 16.529 | 1.504 | 231.756 | 239.683 |
| c8-fixed / V29 | 2262.442 | 6.854 | 3.159 | 114.633 | 123.404 |
| c8-fixed / V30 | 2769.927 | 7.310 | 2.527 | 93.668 | 106.467 |
| c8-fixed / vLLM | 2364.959 | 18.708 | 2.344 | 104.510 | 128.002 |
| c8-natural / V29 | 2456.962 | 7.601 | 3.039 | 415.062 | 449.916 |
| c8-natural / V30 | 3037.197 | 6.933 | 2.427 | 335.169 | 366.784 |
| c8-natural / vLLM | 3722.249 | 17.333 | 1.831 | 270.049 | 282.960 |

V30 vs V29 throughput gains15.80%,15.64%,22.43%,23.62% (C4fixed,C4natural,C8fixed,C8natural), TPOT reductions15.04%,14.95%,20.03%,20.13%. C8fixed TTFT regresses6.65% vs previous; other TTFT improves. Against vLLM throughput+14.02%,-13.30%,+17.12%,-18.40%; TPOT remains7.42%,26.22%,7.77%,32.56% higher. Thus combined goal unmet despite substantial measured CPU gain. These short repeated runs do not qualify sustained high-concurrency P95/P99 stability. V29/V30 references960/960each; vLLM C4fixed234/240,C4natural135/240,C8fixed231/240,C8natural147/240; numerical parity and broad quality are separate qualification gaps.

Post-restoration nonexclusive CPU instrumentation: four-row validation median153→91us and sampling246→2us; single-row43→27us and66→0us (integer microsecond measurement, zero means below timer resolution). Native median1425→1418us broadly unchanged. Diagnostic HTTP references pass, instrumentation patches/build hashes archived, all touched source restored and production binary hash verified. Before patch is recorded relative to V30 HEAD and includes reversal to V29, with explicit base commit in diagnostic receipts. No instrumentation in candidate or matched benchmark.

Authorized Blender3 restoration freshly verified exact process/cwd/GUI environment/private libraries and listening ports:2562417/start88517485/9876,2562528/start88517538/9911,2562670/start88517592/9887. Round40 helper SHA256 b96d3944e8acdb39a39efae2de9eabfe220f85d20f791443dd20e20490187cc0 derives from Round39 helper8b08bfbd92cc8e8eeb34fc73f92e817e8084aa5c534370e909b155eba158c9c6. Private journals and session logs excluded.233 evidence files SHA256 verified locally,18 reproducibility scripts saved.

Next: use the V29 GPU traces (kernels unchanged in V30) and fresh CPU evidence to target long-context shared decode attention/projection costs in a related optimization batch. CPU sampling is now small; GPU/native execution dominates. Preserve matched serving checks and investigate C8fixed TTFT regression; do not claim high-concurrency qualification from C8 or120-request tails. Goal active.


## V31 grouped decode attention experiment — rejected after Round41

Previous goal turn V30 was progress: measured CPU savings and matched serving gains. Current remote source began clean at00c9da9. V31 candidate3bc31075ba70c95ed58ad5372ed7ad005730c4e6 groups four output warps per CTA, sharing one softmax exponential/probability tile, and retains QK query fragments across score tiles. Maintains original MMA recurrence, lane denominator order, online scaling and BF16 rounding. Candidate SHA256392d0401fadc8582e5910cecf2f3224f7e34e8c81230f9c69e4285bc23365441.

Projection/attention primitive outputs exact, including attention active1..8, invalid0/9, context1,16,17,127,128,129,2049,4096 and disjoint permuted KV pages. Memcheck zero errors; racecheck zero errors/warnings. Real shared scheduler full-logit test passes. HTTP C4/C8 each13responses/928tokens exact, with invalid-bound rejection, disconnect recovery and clean shutdown. No correctness relaxation to obtain speed.

Round41 matched V30/V31/vLLM with C4/C8 client and active capacity, same model/tokenizer/workloads, two reversed orders,12warmups+120retained per lane,2880retained and zero failures. Two-run medians:

| Workload/lane | Throughput tok/s | TTFT ms | TPOT ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| c4-fixed / V30 | 1955.929 | 4.294 | 1.831 | 62.978 | 64.654 |
| c4-fixed / V31 | 1921.635 | 4.340 | 1.865 | 65.559 | 67.358 |
| c4-fixed / vLLM | 1690.092 | 13.432 | 1.696 | 74.032 | 87.454 |
| c4-natural / V30 | 1968.907 | 4.566 | 1.922 | 269.003 | 270.149 |
| c4-natural / V31 | 1874.067 | 4.669 | 2.018 | 281.936 | 283.502 |
| c4-natural / vLLM | 2305.450 | 17.560 | 1.468 | 226.637 | 241.242 |
| c8-fixed / V30 | 2765.162 | 7.237 | 2.502 | 93.888 | 101.304 |
| c8-fixed / V31 | 2732.156 | 6.942 | 2.558 | 94.915 | 102.742 |
| c8-fixed / vLLM | 2406.975 | 18.561 | 2.336 | 101.411 | 111.557 |
| c8-natural / V30 | 3022.069 | 6.894 | 2.438 | 337.440 | 369.784 |
| c8-natural / V31 | 2905.457 | 6.974 | 2.535 | 350.799 | 381.160 |
| c8-natural / vLLM | 3653.028 | 20.124 | 1.849 | 287.736 | 293.528 |

V31 vs V30 throughput-1.75%,-4.82%,-1.19%,-3.86%; TPOT+1.84%,+5.05%,+2.24%,+3.97% (C4fixed,C4natural,C8fixed,C8natural). Reject this batch: every serving case regresses. Short120-request tails remain screening evidence only, not high-concurrency stability qualification.

Nonexclusive post-restoration natural C4 Nsight trace:24prefill/296decode, all12requests reference exact. Decode kernel span1492.324us vs unchanged-GPU V29 1390.929us; prefill3104.167us essentially unchanged. Attention values mean12.595us vs9.419us; scores4.175us vs4.236us. The values stage explains the regression; query hoisting brings no material overall benefit. Fewer CTAs and added cross-warp barriers are plausible mechanisms, not separately proven hardware-counter causes. At active4 the values stage has72 active CTAs instead of288, and introduces block barriers; this tradeoff is not adopted.

Reverted with b5072806119663baa425fb8e6e06844db30e7789; entire source tree diff against V30 is empty. Production target restored to frozen V30 binary SHA2566f7d248187d9b614fd0d27a85a6863d38b70464c2b69270551dd6c249bd43582. Frozen rejected V31, patch, logs and reversal receipt retained for reproduction.

Authorized Blender3 restored and freshly verified:2612365/start88569542/9876,2612490/start88569595/9911,2612606/start88569649/9887. Round41 helperf0d1a65c48fd3c218f85eb97e9448ca95080f38d6560800afdccd5402e061ea1, predecessor Round40b96d3944e8acdb39a39efae2de9eabfe220f85d20f791443dd20e20490187cc0. Private journals/session logs excluded.226 evidence files SHA256 verified and13 scripts/probes saved locally.

Next direction: preserve CTA parallelism when reducing attention work. Evaluate separate probability preparation with existing output CTA geometry, or projection data movement/launch cost based on remaining GPU trace. Do not repeat four-warp grouping or infer that fewer arithmetic operations guarantees serving gains. Current accepted baseline remains V30; long natural workload throughput/TPOT and sustained high-concurrency quality/stability remain below the full goal. Goal active; this turn yielded a measured rejection and restored baseline, not a blocker.


## V32 separate probability preparation — rejected after Round42

Previous turn V31 was progress through measured rejection and restored baseline. Starting source b507280 was clean and tree-equivalent to V30. Candidate fa06f6c31ceee4591313ad36c56ebfe70fda1f43 prepares online softmax once per row/head, packs128BF16 probabilities into64float slots within each existing128float score tile, stores FP32 alpha at64 and final lane-specific denominators at65..68 of tile0, and preserves original8 output CTAs/head. No extra device allocation or scratch-size change. Descending tile packing never overwrites unread lower score tiles. Values reads packed probabilities directly and retains original MMA/scaling order. One new kernel/layer is the tradeoff.

Initial tests passed, but source review identified that the final denominator should preserve all four lane sum orders. Final candidate stores four denominators; all tests repeated after this correction. Initial logs/HTTP retained separately; final HTTP directories contain shared-r2-final. Primitive exact outputs, memcheck zero errors, racecheck zero errors/warnings, real shared scheduler full-logit test passes. C4/C8 final HTTP each13responses/928tokens exact with cancellation and invalid-bound recovery. Candidate SHA25676ef9fa9e30d92377884a2b00e517921e0b39fb1fc5208e7827639e47119e03c.

Round42 V30/V32/vLLM matched model/tokenizer/GPU, C4/C8 client and active capacity, workload and KV settings; two reversed orders,12warmups+120retained per lane,2880retained/zero failures. Two-run medians:

| Workload/lane | Throughput tok/s | TTFT ms | TPOT ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| c4-fixed / V30 | 1947.384 | 4.299 | 1.835 | 63.086 | 66.205 |
| c4-fixed / V32 | 1916.371 | 4.335 | 1.878 | 64.405 | 65.050 |
| c4-fixed / vLLM | 1777.326 | 12.145 | 1.669 | 68.098 | 73.505 |
| c4-natural / V30 | 1971.342 | 4.573 | 1.918 | 269.677 | 270.625 |
| c4-natural / V32 | 1913.767 | 4.631 | 1.971 | 277.428 | 278.571 |
| c4-natural / vLLM | 2286.985 | 17.959 | 1.484 | 227.983 | 235.660 |
| c8-fixed / V30 | 2739.033 | 5.972 | 2.573 | 94.648 | 107.464 |
| c8-fixed / V32 | 2709.911 | 7.424 | 2.573 | 95.733 | 106.546 |
| c8-fixed / vLLM | 2438.569 | 18.807 | 2.333 | 100.967 | 102.521 |
| c8-natural / V30 | 3031.519 | 6.863 | 2.432 | 336.508 | 366.849 |
| c8-natural / V32 | 2961.769 | 6.975 | 2.491 | 343.755 | 374.130 |
| c8-natural / vLLM | 3656.551 | 21.039 | 1.850 | 275.606 | 283.717 |

V32 vs V30 throughput-1.59%,-2.92%,-1.06%,-2.30%; TPOT+2.37%,+2.77%,+0.02%,+2.44% (C4fixed,C4natural,C8fixed,C8natural). C8fixed TTFT+24.32% also regresses. Candidate rejected. These short runs do not qualify sustained tails or broad correctness against vLLM.

Nonexclusive post-restoration natural C4 Nsight:24prefill/296decode, all12request references exact. Decode kernel span1443.593us vs unchanged-GPU V29 1390.929us; prefill3104.338us unchanged. Values9.419→7.445us, scores4.236→4.235us, added prepare_probabilities3.729us. Measured preparation cost outweighs values savings; no speculative claim needed to reject this architecture. Projection kernels including fused epilogues remain54.51% of decode kernel total in this candidate trace.

Reverted with8b6ed46dc8fe4cbe48844ad0162285b5952bfe0a; full tree diff against V30 empty, production target binary restored to SHA2566f7d248187d9b614fd0d27a85a6863d38b70464c2b69270551dd6c249bd43582. Frozen V32 and exact patch/reversal retained. A broader native stress probe was prepared but not run after rejection; it is not validation evidence.

Authorized Blender3 restored and freshly verified:2665849/start88624702/9876,2665918/start88624754/9911,2666011/start88624809/9887. Round42 helper2ef0f891b0fa155836dd79eb1957905e53521f9eafa8a913f2409a9d773d047d derives from Round41f0d1a65c48fd3c218f85eb97e9448ca95080f38d6560800afdccd5402e061ea1. Private journals/session logs excluded.

Next: move to shared decode projection scheduling/data movement, which remains roughly half of GPU kernel time. Both cross-warp softmax sharing and separate preparation have measured serving regressions; avoid repeating those approaches. Inspect packed gate/up/down load scheduling and instruction dependency costs while preserving ordered BF16 chunk recurrence, then implement related projection changes as one batch. Current accepted baseline V30; full goal active and unmet.

240 evidence files SHA256 verified locally;12 scripts saved.


## V33 projection load scheduling and Round43 — accepted baseline

Previous turn V32 made progress through measured rejection. Starting source8b6ed46 was clean/tree-equivalent to V30. Candidate214c8ed7309d00ea908c7e29e3c2ed962f8f945d hoists input/weight bases, groups four independent K16 loads ahead of their ordered MMA calls, and bounds outer-loop unrolling instead of fully expanding K. Packed and original weight paths supported, tail chunk bounds guarded, original chunk BF16 rounding and accumulation order retained. No new allocations or graph launches.

Primitive packed/unpacked projections and attention exact, active1..8/invalid0,9 and inactive guards; memcheck zero errors, racecheck zero errors/warnings. Real shared scheduler full-logit test passes. C4/C8 HTTP each13responses/928tokens reference exact, invalid-bound rejection and disconnect recovery pass. Frozen release SHA2565ffceea0d2494f951cf3b92115ad04063bbadb8e5c45c3c5ffdf67a3cd8ec7a3; source and production target verified clean/current at export.

Round43 V30/V33/vLLM matched C4/C8 client/active capacity, same model/tokenizer/GPU/workloads and KV settings, two reversed orders,12warmups+120retained/lane,2880retained/zero failures. Two-run medians:

| Workload/lane | Throughput tok/s | TTFT ms | TPOT ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| c4-fixed / V30 | 1949.480 | 4.304 | 1.835 | 63.102 | 66.694 |
| c4-fixed / V33 | 2227.679 | 4.044 | 1.598 | 54.734 | 58.984 |
| c4-fixed / vLLM | 1758.942 | 11.775 | 1.692 | 68.170 | 74.797 |
| c4-natural / V30 | 1969.416 | 4.581 | 1.918 | 269.946 | 271.012 |
| c4-natural / V33 | 2265.219 | 4.323 | 1.665 | 235.354 | 236.131 |
| c4-natural / vLLM | 2255.811 | 15.496 | 1.516 | 238.201 | 249.605 |
| c8-fixed / V30 | 2764.120 | 6.730 | 2.545 | 93.994 | 107.543 |
| c8-fixed / V33 | 3040.371 | 6.658 | 2.341 | 86.178 | 98.456 |
| c8-fixed / vLLM | 2317.297 | 19.290 | 2.399 | 108.700 | 124.656 |
| c8-natural / V30 | 3025.475 | 6.920 | 2.437 | 336.863 | 368.737 |
| c8-natural / V33 | 3388.290 | 6.663 | 2.177 | 301.297 | 330.733 |
| c8-natural / vLLM | 3635.727 | 18.440 | 1.870 | 281.979 | 300.567 |

V33 vs V30 throughput+14.27%,+15.02%,+9.99%,+11.99%; TPOT-12.88%,-13.20%,-8.02%,-10.65% (C4fixed,C4natural,C8fixed,C8natural). All TTFT and E2E tail medians also improve. Accept V33 as current baseline. Against vLLM throughput+26.65%,+0.42%,+31.20%,-6.81%; TPOT-5.55%,+9.82%,-2.45%,+16.40%. Fixed workloads now pass minimum throughput/latency screen, but natural C8 still loses throughput and natural C4/C8 lose TPOT. Two short repetitions and C8 cannot prove sustained high-concurrency P95/P99 stability or broad quality. Full goal unmet.

Nonexclusive post-restoration natural C4 Nsight:24prefill/296decode, all12requests strict reference exact. Decode median1133.320us vs unchanged-GPU V29 1390.929us, prefill3105.413us unchanged. Down projection6.732→3.414us, gate/up/SwiGLU6.267→5.396us, QKV parts5.657→2.953us, O projection3.827→2.173us. Projection including fused epilogues46.80% and attention36.49% of decode kernel time. This attributes gains to projection kernels, without claiming hardware-counter proof of overlap or instruction-cache mechanism.

Authorized Blender3 restored and freshly verified:2707546/start88666820/9876,2707612/start88666872/9911,2707765/start88666926/9887. Round43 helper56b66e191fed43f1d0dc7396897ab121229fab080aec9218eec44ba79b512078 derives from Round422ef0f891b0fa155836dd79eb1957905e53521f9eafa8a913f2409a9d773d047d. Private journals/session logs excluded.

Next: refresh C8 natural stage/batch-fill evidence under the faster baseline to choose between remaining projection/gate-up cost and prefill blocking decode. Assess broader concurrency/longer sustained workloads before any overall success claim. Prior attention normalization-sharing alternatives V31/V32 are rejected; preserve the accepted V30 CPU changes and V33 projection gains. Goal active.

223 evidence files SHA256 verified locally;12 scripts saved.


## V34 V33 batch-fill and queued-concurrency diagnostic

Previous turn V33 made measured serving progress. Current source214c8ed and clean tree verified. Temporary stage/row/token/elapsed instrumentation was compiled, frozen as fill-diagnostic-v34 with binary/patch hashes, then source and production target restored to V33 before diagnostic runs. No production optimization change this turn.

Natural workload384requests each at client4,8,16,32; active capacity=min(client,8). Source engine.rs gates V3 active sequences to1/2/4/8. Client16/32 therefore test queued requests, not active16/32. Initial run completed C4/C8 then stopped at the existing mixed client assertion concurrency<=8 before issuing C16 work. Existing C4/C8 results preserved. A separate mixed_phase_v34 copy changes only that client bound to32; original worker count, overlap/identity/corpus/transport accounting retained. C16/C32 resumed in natural-fill-v34-high. All1536requests reference exact and observed peak in-flight equals requested client concurrency. Nonexclusive instrumented diagnostic, not matched vLLM performance or long sustained stability proof.

Excluding first/last10%iterations:

| Client / active | Mean decode rows | Full decode fraction | Prefill fraction of execution time | Decode execution median us |
|---|---:|---:|---:|---:|
| 4 / 4 | 3.895 | 89.45% | 21.73% | 1285 |
| 8 / 8 | 7.786 | 78.84% | 33.65% | 1423 |
| 16 / 8 | 7.787 | 78.85% | 33.64% | 1419 |
| 32 / 8 | 7.786 | 78.86% | 33.63% | 1421 |

Execution time includes transfer and validation. Row fill is not GPU occupancy. Each384request run has768prefill submissions (average2/request). Extra waiting clients do not improve batch fill or throughput beyond active8.

Diagnostic external measurements (not competitive claims):

| Client / active | tok/s | TTFT median ms | TPOT median ms | E2E P95 / P99 ms |
|---|---:|---:|---:|---:|
| 4 / 4 | 2295.37 | 4.34 | 1.668 | 234.07 / 235.51 |
| 8 / 8 | 3443.27 | 6.72 | 2.186 | 301.46 / 303.44 |
| 16 / 8 | 3450.15 | 175.69 | 2.182 | 465.23 / 484.33 |
| 32 / 8 | 3436.49 | 521.98 | 2.185 | 820.09 / 861.38 |

This exposes queue-latency growth under the active8 ceiling; successful completion alone does not meet high-concurrency latency goals. No Blender stop was needed for this explicitly nonexclusive diagnostic. Last restored identities remain Round43; revalidate before any next exclusive measurement.

Next: compare a coordinated increase in prefill chunk and token budget (128→512) against V33-128 and appropriately configured vLLM on identical external workloads, recording internal settings explicitly. It may reduce768prefill launches toward384 and the measured prefill share, but larger chunks may block decode longer; measure both throughput and latency before adopting. Separately, active8 is an architectural limit to address for high-concurrency serving; do not treat queued C32 as solving it. Current accepted source/binary remains V33, full goal active.

22 diagnostic evidence files SHA256 verified locally;8 scripts saved.


## V35 prefill configuration comparison, Round44 failure and Round45

**New user instruction: leave the three Blender processes stopped; do not restore them in future work.** Round45 restoration completed just before this instruction was applied. Exact successors2778255/start88743410/9876,2778413/start88743463/9911,2778536/start88743519/9887 were identity/runtime/pidfd verified and sent SIGTERM. The immediate port check raced socket teardown; fresh verification confirms all three stopped, ports closed and no Round45 watchdog. Receipt blender-keep-stopped-public.json is current. Earlier restoration receipts are historical. Do not run old stop/restore watchdog controllers for future measurements; check GPU ownership directly while respecting unrelated processes.

Starting source214c8ed clean. Same frozen V33 binary for previous/new lanes: configuration-only experiment, no source performance changes. Previous budget/chunk128. Candidate budget512, natural chunk512, fixed chunk128. vLLM budget512 (previous screens used128), so older vLLM comparisons are not evidence against this better-tuned lane.

Round44 initially set fixed chunk512 with max_sequence160 and failed startup validation before completing comparison. Exact error: max_prefill_chunk_tokens must not exceed max_sequence_tokens. Complete finally restoration verified, failed evidence retained. Round45 uses fixed chunk128 and natural chunk512 with original workload sequence bounds. Matching external workload/model/tokenizer/GPU/client/active capacity/KV settings; two reversed orders,12warmups+120retained/lane;2880retained/zero failures. Both Riley configurations1920/1920 reference matches. Two-run medians:

| Workload/lane | Throughput tok/s | TTFT ms | TPOT ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| c4-fixed / V33 budget128 | 2242.450 | 4.037 | 1.574 | 54.674 | 55.779 |
| c4-fixed / V33 budget512 | 2245.546 | 4.038 | 1.572 | 54.766 | 57.796 |
| c4-fixed / vLLM budget512 | 1951.210 | 9.374 | 1.586 | 63.079 | 66.617 |
| c4-natural / V33 budget128 | 2271.265 | 4.330 | 1.638 | 236.292 | 237.112 |
| c4-natural / V33 budget512 | 2492.685 | 4.426 | 1.489 | 208.765 | 212.633 |
| c4-natural / vLLM budget512 | 2581.730 | 9.387 | 1.377 | 195.591 | 207.253 |
| c8-fixed / V33 budget128 | 3050.513 | 6.064 | 2.300 | 85.636 | 96.611 |
| c8-fixed / V33 budget512 | 3029.498 | 7.071 | 2.267 | 86.651 | 94.550 |
| c8-fixed / vLLM budget512 | 3045.251 | 13.544 | 1.869 | 80.739 | 91.948 |
| c8-natural / V33 budget128 | 3385.426 | 6.636 | 2.178 | 302.340 | 332.201 |
| c8-natural / V33 budget512 | 3888.702 | 6.896 | 1.926 | 255.721 | 260.980 |
| c8-natural / vLLM budget512 | 4548.419 | 11.904 | 1.534 | 215.731 | 222.504 |

Natural C4/C8 Riley throughput+9.75%/+14.87%, TPOT-9.09%/-11.57%; TTFT+2.22%/+3.91% and E2E tails improve. Fixed throughput+0.14%/-0.69%, no established benefit; C8fixed TTFT+16.60% warrants caution. Accept budget/chunk512 as natural workload configuration for further optimization, not a globally proven default. Fixed workload retains chunk128; compare settings fairly and consider vLLM tuning part of baseline maintenance.

Against vLLM budget512, natural C4/C8 throughput-3.45%/-14.50%, TPOT+8.13%/+25.58%. Fixed C4 throughput+15.08%/TPOT-0.87%, C8 throughput-0.52%/TPOT+21.25%. Earlier fixed wins against vLLM128 do not establish victory against vLLM512. Goal unmet, broad quality and sustained high-concurrency stability unqualified.

Frozen V34 instrumented V33 reused for budget/chunk512 diagnostic after Blender was stopped. C4/C8 each384requests exact. Prefill submissions halve from768 to384 per run. Middle80 prefill execution fraction C4 21.73%→14.20%, C8 33.65%→23.13%; mean decode rows3.946/7.893, full fraction94.64%/89.30%, decode execution median1294/1434us. This supports fewer repeated prefill submissions as the mechanism. Diagnostic and matched timing remain separate; native execution includes transfer/validation. Production source and binary unchanged.

Next: retain improved natural prefill settings and use vLLM512 as a competitive baseline. Active8 ceiling still causes queued C16/C32 TTFT growth; investigate widening active scheduling beyond8 while preserving eight-row execution descriptors and per-request KV ownership, or increasing native batch capacity with full correctness validation. Do not treat the current queued concurrency test as completing high-concurrency goal. Goal active.

226 evidence files SHA256 verified locally;12 scripts saved.


## V36/V37 admission and HTTP concurrency separation, Rounds46/47

Previous turn made prefill-configuration progress. Current source began clean at214c8ed. V36 commit562fa77305f09f2ccfbf1c0fc0c459e3d7b963e7 admits1/2/4/8/16/32 active V3 requests, caps each decode plan at8, preserves ready-time fairness/NotDispatched rollback, and updates server/CLI guards. GPU wire, completion, head and scratch remain eight-row. Extended owned GPU test asserts peak active32, maximum dispatch8 and compares2336full logits for32independent KV owners; original224logit test also passes. All requests finish, pending mutation close/abort and zero allocations pass. Runtime/scheduler/server/CLI tests pass. Initial test u32/usize mismatch corrected; first HTTP startup exposed and then corrected a separate CLI capacity guard.

Round46 measured V36 against V33 and vLLM at client16/32, but found V36 indistinguishable from V33. **Correction to prior high-concurrency attribution: HTTP ServerConfig defaults to8 blocking workers; each SSE connection occupies its worker until completion. V34 queued client16/32 was constrained by HTTP workers as well as active8. Raising scheduler capacity alone does not expose higher engine concurrency.** Round46 is retained as evidence of this discovered restriction, not proof of active16/32 serving.

V37 commit424b6edec524fac0c67d7126f272f5945e6bf52a scales HTTP workers to max(active,8) only for V3; other profiles preserve default behavior. CLI tests cover16/32. C32 HTTP test includes32barrier requests plus sequential/streaming/invalid-bound/disconnect recovery cases (37complete responses), all exact and clean shutdown. Initial worker change was scoped to V3 before final build and tests repeated. Source patches are authoritative reproduction artifacts; exploratory edit scripts include intermediate corrections.

Frozen V36 SHAcd126db641ee018ea66e3d61f1ad7bf0cc53ec03906702e2fa5db0ce8ccb046b; V37 SHA8343eb096a025f18f86cbdd5cfa17882576326071e9e5b5d8c062f0b59d1fb49.

Round47: same model/tokenizer/GPU/external workloads, client16/32; V33 active8/HTTP8 versus V37 active16/32/HTTP16/32 versus vLLM active16/32. Both Riley lanes and vLLM budget512; natural chunk512/fixed128; Riley physical KV pages equal per compared client condition. The baseline's lower internal admission capacity is explicit. Two reversed orders,96warmups+384retained/lane;9216retained,zero failures; Riley6144/6144 references exact. Two-run medians:

| Workload/lane | Throughput tok/s | TTFT ms | TPOT ms | E2E P95 ms | E2E P99 ms |
|---|---:|---:|---:|---:|---:|
| c16-fixed / V33 active8 | 3104.689 | 56.161 | 2.334 | 136.999 | 138.049 |
| c16-fixed / V37 wider admission | 3141.290 | 4.559 | 4.978 | 163.683 | 181.097 |
| c16-fixed / vLLM | 5098.090 | 16.860 | 2.260 | 99.415 | 109.696 |
| c16-natural / V33 active8 | 3983.851 | 143.152 | 1.926 | 404.695 | 424.208 |
| c16-natural / V37 wider admission | 3973.684 | 4.643 | 3.933 | 518.310 | 522.813 |
| c16-natural / vLLM | 7727.348 | 12.876 | 1.814 | 264.897 | 299.214 |
| c32-fixed / V33 active8 | 3080.517 | 160.454 | 2.304 | 242.259 | 244.049 |
| c32-fixed / V37 wider admission | 3108.865 | 4.469 | 10.433 | 335.042 | 417.480 |
| c32-fixed / vLLM | 7256.614 | 29.803 | 2.940 | 141.561 | 153.036 |
| c32-natural / V33 active8 | 3994.223 | 449.525 | 1.920 | 699.741 | 702.715 |
| c32-natural / V37 wider admission | 3854.644 | 4.785 | 8.268 | 1074.896 | 1177.074 |
| c32-natural / vLLM | 12003.222 | 19.833 | 2.277 | 335.763 | 357.199 |

Wider admission drastically lowers TTFT but approximately doubles TPOT atC16 and quadruples it atC32; throughput barely changes and tails worsen. C32natural V37 3854.64tok/s versus vLLM12003.22, TPOT8.268ms versus2.277ms. Do not adopt active16/32 as a performance default or claim goal progress as a serving win. Development source retains explicitly selected higher-capacity support as groundwork for wider GPU execution; CLI default remains8. Last measured strong low-concurrency configuration remains V33 plus natural budget/chunk512. Goal unmet, high-concurrency tail and throughput gaps now measured directly.

Next execution batch must widen real GPU work, not merely admission: current shared projection invokes m16n8k16 with second row-half input registers zero and publishes only first8rows. Expand to16 genuine rows (and evaluate32 afterward), covering projection stores/partial offsets, gate-up shared epilogue, QKV/RoPE/KV bindings, attention row grids, final head and completion publication, Rust wire/staging/slot bounds and graph resource sizing together. Preserve8-row baseline/frozen comparison; never just raise guard constants over eight-row buffers. Validate inactive rows, divergent contexts, dense output routing, full logits, page ownership, cancellation and resource cleanup before matched serving. No subagents started.

Blender remains stopped per user instruction. Both controllers contain no restoration helper or watchdog. Fresh process/port/watchdog verification passed after Round47; no Blender restoration scheduled. Current source clean at424b6ed.

421 evidence files SHA256 verified locally;15 scripts saved.


## V38 native sixteen-row decode building blocks

Previous turn made progress by exposing the eight-worker ingress limit and measuring the admission/dispatch tradeoff. Current native commit4ee3d8667c304ea019322f45e9291a80d74c4ac1 adds four separate decode_shared16 headers and two committed CUDA probes, without changing the production eight-row implementation or claiming serving gains.

Projection fills both row halves of the existing m16n8k16 MMA, retains grouped loads and per-chunk BF16 rounding, and writes all16rows using16-row partial strides. Gate/up epilogue shares128 BF16 elements/warp instead of64 and publishes both row halves. Model updates QKV offsets, RoPE/KV grids, normalization row limits and inactive hidden clearing. Attention retains original per-output CTA geometry with144 row/head groups. Completion allocates16records and clears unused rows; M16 head uses algorithm21, workspace0, matching the M1 reference in this environment. Exploratory symbol renaming accidentally changed __shared__ and was fixed before validation; unused standalone QKV merge launch also corrected to cover16rows.

Native graph replay test uses real Smol weights/loaded RoPE, disjoint permuted pages, mixed contexts16/128/398, active1,2,4,8,9,15,16 for8steps each. Both original and packed weights compare hidden, entire K/V allocation, full49152 logits, argmax/status and zero inactive records against the single-request path:440row-steps per mode,880total. Packed full-model memcheck0errors. Expanded primitive tests cover original/packed projections, active1..16, invalid0/17, inactive guards and divergent attention contexts through4096; memcheck0errors and racecheck0errors/0warnings.

This is **native-only correctness evidence**. No sixteen-row Rust wire, graph resource recorder, compiled serving wrapper, scheduler execution selection or HTTP path exists yet. Production target stays at V37 SHA8343eb096a025f18f86cbdd5cfa17882576326071e9e5b5d8c062f0b59d1fb49. No sixteen-row serving benchmark or adoption claim. Blender remains stopped; no restoration controller invoked.

Next integration contracts are explicit in kernels/tests/README_shared16.md (also exported raw/README_shared16.md): canonical16-row request30848bytes/token offset26752; result1574912bytes; distinct/versioned or parameterized wire with unused-row/identity/ownership validation; pinned offsets, larger attention scratch and hidden/logits/head leases; prefill reader and native wrappers; Rust adapter's argmax/output-slot arrays; execution width selection. The standalone native probe uses old13440 token offset solely in its single-row prefill reference setup; that overlapping layout must not become the serving wire. Merely changing max rows over eight-row buffers is invalid.

After integration, repeat Rust-owned32-request full-logit/cancellation/cleanup validation and matched serving at C4/C8/C16/C32 against frozen V33/V37 and tuned vLLM. The full throughput/TTFT/TPOT/stability objective remains active and unproven.

15 evidence files SHA256 verified locally;6 scripts saved.


## V39: distinct sixteen-row wire and native structural validation

Source `e530ec7c01c96de2dc1db02c321da72e07b24a1c` retains the V3 default `Expectation<8>` and adds V4 `Expectation<16>` through a shared const-generic validator. Request magic/version and completion magic distinguish the paths, including single-record results with equal byte lengths. V4 token offset26752, request30848, result1574912, output offset1574912 and pinned staging3149824 are explicit constants. The native structural parser has separate V3/V4 entry points with shared checks.

Rust11/11 passed: canonical packet byte mutations, upper-row identity/slot/page aliases, stale replay, unsupported capacity and reject-before-write behavior, inactive result bytes, finite logits/argmax, cross-version results and maximum1024-token prefill without descriptor overlap. Eleven Rust fixtures passed native ASan/UBSan checks;156588 invalid mutations/extent/version/unused checks rejected. CPU scheduler/server and CUDA-enabled server cargo check passed. Native parser remains structural; live Rust ownership is still required.

This is a contract integration checkpoint within the ongoing sixteen-row optimization batch, not a serving improvement. Production frozen binary remains V37. Native16 standalone model still uses its historical reference setup token offset/result magic and is not wired to V4. Next work must connect retained session/buffers, native graph record/replay, canonical prefill token offset and completion magic, M16 head, scheduler selection and output slots together. Then whole Rust-owned GPU32-owner parity, malformed/cleanup/cancellation and matched C4/C8/C16/C32 serving are required before any adoption claim.

Fresh verification confirms the designated Blender processes remain absent and ports closed; no restoration invoked.21 evidence files SHA256 verified in `raw/wire16-v39-manifest.json`; six scripts retained locally. No new GPU or serving performance claim.


## V40 / Round48: sixteen-row serving integration and matched comparison

Code commit `f09ef11281fbbd9a4f341c3d57c9848ecfac4e1b`; documentation-only successor `19566b8f46bb958d64917731b8f88546f109d750`. Frozen V40 binary SHA256 `2a8261364a31ebe6f9ab0fee2507f1effe385bccc2d4df694ad5261c3c55e762`. Profile `variable-smol-v4` now selects actual sixteen-row GPU decode and scheduler selection, with V4 exact extents/offsets/magic, retained ownership, canonical prefill token location and M16 head. V3 remains available; default options unchanged. Work resides in the isolated remote source; unrelated dirty local files preserved.

Whole retained Rust GPU tests: V4 32 owners /2336 full-logit outputs, V3 32 owners /2336 and original3 owners /224. All matched; maximum decode widths16/8/3 and pending-close abort/allocation cleanup passed. Final-code owned16 memcheck0; native V4 recorder memcheck0, partial prefill and active1/2/4/8/9/15/16, alias/wrong-head/context/version rejection and stale-output hiding passed. Scheduler CPU35 and CLI3 tests passed. HTTP37 completed responses,32 simultaneous mixed streaming/nonstreaming, invalid bound and disconnect recovery passed; shutdown active/waiting/KV/device/pinned allzero and one cancellation/disconnect.

Round48: same model/tokenizer/GPU and external workloads, all client/admission capacities4/8/16/32, V37 GPU8 versus V40 GPU16 versus vLLM0.27.1. Both Riley budget512, chunk128 fixed/chunk512 natural; vLLM budget512. Same Riley KV budget within each case. Two reversed orders,96 warmups+384 retained per lane.48 lanes /18432 retained requests, zero transport failures; all12288 Riley references exact. vLLM reference matches are observations, not an equivalence/quality claim.384-request tails and two repeats are a screen, not long-term stability qualification.

Values below are median across two runs: output tok/s / median TTFT ms / median TPOT ms.

| Case | V37 GPU8 | V40 GPU16 | vLLM |
|---|---:|---:|---:|
| c4-fixed | 2246.678 / 4.065 / 1.600 | 1989.788 / 4.392 / 1.814 | 1932.321 / 9.052 / 1.573 |
| c4-natural | 2525.320 / 4.404 / 1.500 | 2235.998 / 4.718 / 1.699 | 2599.523 / 9.249 / 1.377 |
| c8-fixed | 3097.154 / 6.929 / 2.256 | 2817.273 / 7.170 / 2.619 | 3235.499 / 12.854 / 1.822 |
| c8-natural | 3968.714 / 6.906 / 1.931 | 3585.359 / 7.356 / 2.141 | 4669.409 / 10.876 / 1.542 |
| c16-fixed | 3140.398 / 4.705 / 4.978 | 3484.906 / 7.655 / 4.310 | 4995.551 / 16.898 / 2.311 |
| c16-natural | 3955.332 / 4.880 / 3.950 | 4989.116 / 9.378 / 3.055 | 7738.160 / 12.966 / 1.800 |
| c32-fixed | 3109.801 / 4.564 / 10.433 | 3509.854 / 10.154 / 9.069 | 6967.914 / 30.684 / 2.994 |
| c32-natural | 3823.125 / 4.808 / 8.330 | 4908.230 / 9.372 / 6.390 | 11723.246 / 20.854 / 2.323 |

E2E P95 / P99 in milliseconds, median across two runs.

| Case | V37 GPU8 | V40 GPU16 | vLLM |
|---|---:|---:|---:|
| c4-fixed | 55.460 / 56.150 | 64.218 / 66.157 | 65.119 / 83.015 |
| c4-natural | 209.364 / 210.503 | 235.321 / 236.343 | 198.649 / 214.901 |
| c8-fixed | 86.162 / 87.579 | 94.048 / 95.729 | 78.344 / 95.303 |
| c8-natural | 257.003 / 261.470 | 284.347 / 289.750 | 216.126 / 228.244 |
| c16-fixed | 163.235 / 176.787 | 147.433 / 162.174 | 102.600 / 116.464 |
| c16-natural | 520.482 / 535.240 | 406.664 / 432.586 | 269.848 / 302.777 |
| c32-fixed | 335.179 / 425.094 | 281.385 / 314.324 | 149.350 / 176.175 |
| c32-natural | 1086.892 / 1185.408 | 833.587 / 946.277 | 340.570 / 357.929 |

**Decision:** V40 is an explicit high-concurrency candidate, not a general default replacement. C4/C8 throughput regresses9.04–11.46% and TPOT worsens10.83–16.12%. C16/C32 fixed throughput improves10.97/12.86%, TPOT improves13.41/13.08%; natural throughput improves26.14/28.38%, TPOT improves22.67/23.29%. However TTFT versus V37 worsens62.71–122.48% at high concurrency, while remaining below vLLM. High-concurrency V40 throughput is still30.24–58.13% below vLLM and TPOT69.71–202.89% higher. Goal remains unmet.

Post-benchmark diagnostic uses an instrumented V40 binary (`a7504ec59443b4655149b3dbc470dbfb3bd0a4d334f18729ce3ab9dcae2e9cce`) with its own V3/V4 profiles, not original frozen V37. Each C16/C32 lane384 requests;1536/1536 exact. Production file and source restored afterward. Middle80 decode mean rows V3=8, V4=15.787/16; C16/C32 V4 median total1938/2000us, encode24/42us, native1551/1570us, validation365/385us. V3 native1256/1254us and validation183/198us. V4 prefill execution share31.306/31.237% versus V3 approximately23%. Native interval includes replay, synchronization and host readback; it is not GPU-only timing. Batch underfill does not explain C32 saturation. Next profiling/optimization targets are completion transfer/validation and prefill execution, with GPU trace needed to separate native kernel and transfer costs. No next optimization is yet implemented.

431 evidence files SHA256 verified via `raw/serving-round48-manifest.json`;16 preparation/benchmark/analysis scripts saved. Fresh public Blender verification confirms designated processes absent and ports closed; no restoration helper/watchdog used. Frozen V40 server restored after profiling; no GPU compute process remained at final check.


## V41 / Round49: complete host validation accelerated with AVX2

Source `4941e7318ae5cd6a6857a9743946854922e04b51`, frozen binary SHA256 `3d30e9dc14d998f9e3a7af27658614fd0d3ee006084555180974c91895810c14`. Four independent AVX2 BF16 reductions preserve complete finite-logit validation and lowest-ID ties; a vector OR scan checks inactive bytes. Runtime feature detection retains the prior scalar fallback. Unsafe intrinsics are confined to a private checked module with explicit feature/bounds proofs and unsafe-op lint enforcement; kernel, wire identity, ownership and sampling rules are unchanged. The new module is included in retained catalog hashing.

CPU40 tests passed, including all65536 BF16 patterns in vector/tail positions, nonfinite rejection, signed-zero ties, unaligned inputs, every relevant tail and inaccessible guard-page boundaries. Final isolated CPU diagnostic median per full49152-logit row approximately22.07us scalar versus5.49us SIMD; this is not serving qualification. Whole GPU3 tests matched4896 full-logit rows across eight/sixteen-row sessions with zero cleanup allocations. HTTP37 responses,32 simultaneous mixed streaming, invalid bounds and disconnect recovery matched references and cleaned up. No CUDA kernel changed in this batch.

Round49 compares frozen V40 versus V41 versus vLLM0.27.1. Both Riley use GPU8 (`variable-smol-v3`) at C4/C8 and GPU16 (`variable-smol-v4`) at C16/C32, so execution width is matched within every pair. Same model/tokenizer/hardware/workload and admission capacity, budget512; fixed chunk128, natural chunk512.48 lanes, two reversed orders,96 warmups+384 retained each:18432 retained completions, zero failures, all12288 Riley references exact. vLLM reference matches remain observations, not model-quality equivalence. Short-run tail measurements do not establish long-term stability.

Median across two runs: output tok/s / median TTFT ms / median TPOT ms.

| Case | V40 | V41 | vLLM |
|---|---:|---:|---:|
| c4-fixed | 2267.680 / 4.055 / 1.566 | 2359.517 / 3.974 / 1.511 | 1976.232 / 9.075 / 1.559 |
| c4-natural | 2537.042 / 4.398 / 1.488 | 2645.643 / 4.347 / 1.432 | 2597.490 / 9.296 / 1.386 |
| c8-fixed | 3093.190 / 6.783 / 2.369 | 3244.084 / 6.258 / 2.210 | 3211.499 / 12.632 / 1.842 |
| c8-natural | 3972.516 / 6.910 / 1.928 | 4242.446 / 6.772 / 1.801 | 4576.823 / 11.290 / 1.546 |
| c16-fixed | 3480.816 / 7.688 / 4.313 | 3672.877 / 7.442 / 4.077 | 4858.407 / 17.156 / 2.327 |
| c16-natural | 4959.775 / 9.373 / 3.066 | 5439.216 / 9.119 / 2.791 | 7877.638 / 12.952 / 1.792 |
| c32-fixed | 3480.802 / 9.817 / 9.139 | 3678.425 / 8.915 / 8.617 | 7429.050 / 30.317 / 2.698 |
| c32-natural | 4877.769 / 8.615 / 6.434 | 5337.417 / 7.849 / 5.867 | 11336.253 / 22.480 / 2.377 |

E2E P95 / P99 milliseconds, median across two runs.

| Case | V40 | V41 | vLLM |
|---|---:|---:|---:|
| c4-fixed | 56.168 / 57.008 | 52.659 / 52.948 | 62.944 / 78.953 |
| c4-natural | 208.463 / 209.195 | 199.938 / 201.297 | 198.495 / 213.657 |
| c8-fixed | 86.202 / 89.018 | 82.313 / 85.512 | 79.126 / 92.885 |
| c8-natural | 256.700 / 263.850 | 239.978 / 246.858 | 226.924 / 238.904 |
| c16-fixed | 149.394 / 163.551 | 141.095 / 152.060 | 108.656 / 129.864 |
| c16-natural | 412.926 / 431.891 | 373.149 / 393.593 | 259.523 / 271.016 |
| c32-fixed | 285.489 / 314.696 | 270.142 / 304.725 | 142.531 / 168.200 |
| c32-natural | 839.922 / 936.875 | 763.838 / 878.667 | 358.964 / 393.016 |

**Decision:** retain V41 SIMD validation for both execution widths. Across these cases throughput improves4.05–9.67%, TPOT improves3.52–8.97%, and TTFT plus sampled E2E P95/P99 also improve versus V40. C4-fixed exceeds vLLM throughput19.39% with TPOT3.13% lower, but this is only one workload: high-concurrency throughput remains24.40–52.92% below vLLM and TPOT55.72–219.38% higher. The full goal remains unmet; do not extrapolate the C4 result to all serving.

Post-benchmark instrumented V41 (`09905f0dbcf37089187de56fb3ca3edb27d975f29e311825002958ee10881153`) runs each C16/C32 and V3/V4 profile384 requests;1536/1536 references exact. Middle80 V4 mean decode rows15.787/16. Median validation drops from prior V40 diagnostic365/385us to106/125us; V4 native interval remains1551/1563us versus1551/1570us previously. Total V4 session decode falls1938/2000us to1681/1734us. V3 validation54/67us, native1255/1256us. V4 prefill share rises34.096/34.259%; native prefill median3014us in both cases. These intervals include synchronization/readback and omit some outer scheduler/sampling/HTTP work. Native CUDA trace is the next step to separate kernels, transfers and synchronization rather than treating the interval as GPU-only time.

425 evidence files SHA256 verified via `raw/serving-round49-manifest.json`;13 scripts saved. Profiling source edits restored, source clean at4941e73, frozen V41 restored to the serving target. Final GPU compute query empty. Designated Blender processes remain absent and ports closed; no restoration invoked.

## V42 — attention fragment loading and matched serving Round50

Source `fca8a7204cfeae342a7837fb43a686021d8cbf13`, frozen binary SHA256 `69b9cd5ad2dd9dd2c46643fa9308fbe54301026ee8c19157e8d7098b4ff3549d`. Three related changes hoist shared16 query/page addressing, load four key fragments before ordered score MMAs, and load four probability/value fragment pairs before ordered value MMAs; prefill applies the key-loading change. Arithmetic order, launch geometry and synchronization stay fixed. Independent old-oracle attention probes remain exact through context4096; both memchecks report zero errors. Owned GPU tests validate4896 full-logit rows; HTTP37 including32 mixed concurrent requests and disconnect recovery pass.

Round50:48 lanes, two reversed orders,96 warmup plus384 retained each; C4/C8 use8 GPU rows and C16/C32 use16 for both V41 and V42. 18432 retained requests finish without failures; all12288 Riley references match. vLLM token sequences are not all identical, so this does not qualify model-quality equivalence.

| Case | V41 tok/s | V42 tok/s | vLLM tok/s | V42 vs V41 | V42/vLLM TPOT ms |
|---|---:|---:|---:|---:|---:|
| c16-fixed | 3677.7 | 3768.6 | 5164.8 | +2.47% | 3.976 / 2.141 |
| c16-natural | 5414.2 | 5566.8 | 7491.0 | +2.82% | 2.730 / 1.809 |
| c32-fixed | 3701.2 | 3804.8 | 6819.7 | +2.80% | 8.291 / 3.061 |
| c32-natural | 5260.0 | 5435.1 | 11577.9 | +3.33% | 5.765 / 2.271 |
| c4-fixed | 2339.0 | 2384.9 | 1917.9 | +1.96% | 1.500 / 1.595 |
| c4-natural | 2644.8 | 2638.9 | 2586.6 | -0.22% | 1.429 / 1.387 |
| c8-fixed | 3236.6 | 3324.4 | 3170.7 | +2.72% | 2.181 / 1.824 |
| c8-natural | 4234.7 | 4230.0 | 4479.1 | -0.11% | 1.806 / 1.572 |

Retain V42 as next high-concurrency baseline: throughput gains 2.47-3.33 percent at C16/C32, TPOT and E2E tails improve, some TTFT regresses. Low-concurrency natural throughput is flat/slightly worse. vLLM high-concurrency gap remains. Shape diagnostic motivates evaluating bucketed prefill graphs; not a serving win claim. P95/P99 are sampled observations, not long-run stability qualification. Full metrics are in `raw/serving-round50-analysis.json`.

Node-level Nsight natural-workload diagnostics each complete192 exact requests. V41→V42 median decode graph span: C16 1605.366→1549.523us, C32 1609.398→1540.227us. Prefill3108.394→3084.390us and3119.322→3098.903us. Profiler overhead and varying mixed-workload replay composition prevent treating these differences as standalone production speedups.

Separate V42 fixedP128/O64 diagnostic atC16 compares prepared prefill capacity128 versus512 without changing actual token counts, budget, or decode capacity.192 requests exact. Prefill graph median2900.405 versus3084.582us (5.97% reduction); kernel sum2608.964 versus2788.581us; decode1363.394 versus1368.163us. This motivates the next prefill graph bucket batch, but requires unprofiled serving verification.

492 exported evidence files SHA256 verified (`raw/serving-round50-manifest.json`), including patch, traces, build identities, probes and measurements. Remote source clean. Designated Blender processes and ports remain absent; user instruction is to leave them stopped, and no restoration was invoked.

## V43 — retained prefill graph buckets; serving validation running

Source `d3809d36a930b9c53098be0428a2f58602cf45cf`; frozen binary SHA256 `4d79277efb3dc665a2c98f4dec2d6a9f5814b7ad7cc0ac4dadf01f1418ee8f09`. Capture additional capacity16 and128 prefill graphs when smaller than the maximum, select the smallest fitting graph after packet validation, and destroy all additional graph resources under the existing parent ledger and ambiguity rules. Scratch remains allocated at the validated maximum size. Decode and scheduler policy unchanged.

Five owned GPU tests pass:5344 full-logit output rows match immutable references, including capacity512/chunk129 exercising the16/128/max buckets for both8/16 execution widths. Pending-result close/abort and zero final allocations pass. HTTP37 including32 concurrent mixed streaming/nonstreaming requests, invalid request handling and disconnect recovery passes with capacity512. Source patch and checks are in `raw/buckets-v43.patch`, `raw/buckets-v43-owned.log`, and `raw/v4-http-v43-c32-final`.

Round51 is running on the remote GPU, comparing V42, V43 and vLLM at C4/8/16/32 with two reversed orders,96 warmup and384 retained requests per lane. Controller `/tmp/riley-opt-260912/serving_screen_round51.py`, output directory `/tmp/riley-opt-260912/variable-serving-screen-round51`, log `/tmp/riley-opt-260912/serving-screen-round51.log`; initial live local SSH session9614. Inspect live state before resuming; never restart on a polling timeout. Performance acceptance pending. Blender stays stopped; no restoration requested or invoked.

## Round51 completed — V43 not accepted as a performance improvement

All48 lanes completed,18432 retained requests with zero failures and12288 exact Riley references. Two reversed orders per configuration; these remain short serving screens, not long-run stability qualification.

| Case | V42 tok/s | V43 tok/s | vLLM tok/s | Throughput change | TPOT change |
|---|---:|---:|---:|---:|---:|
| c16-fixed | 3744.4 | 3754.7 | 5019.9 | +0.27% | -0.37% |
| c16-natural | 5563.6 | 5554.2 | 7718.0 | -0.17% | -0.24% |
| c32-fixed | 3808.9 | 3786.3 | 6680.8 | -0.59% | +0.64% |
| c32-natural | 5393.0 | 5418.4 | 11419.0 | +0.47% | -0.57% |
| c4-fixed | 2382.1 | 2384.0 | 1934.6 | +0.08% | -0.44% |
| c4-natural | 2645.2 | 2649.4 | 2600.0 | +0.16% | -0.20% |
| c8-fixed | 3315.1 | 3306.9 | 3216.8 | -0.25% | +0.68% |
| c8-natural | 4206.9 | 4225.5 | 4559.5 | +0.44% | -0.49% |

V43 throughput changes−0.59% to+0.47%; TPOT changes−0.57% to+0.68%. C32-natural TTFT improves35.65%, but without consistent throughput or E2E tail gains. Do not adopt the extra retained graph complexity on this evidence. Use frozen V42 as the next performance baseline. Remote source still contains the V43 commit pending removal of its graph changes during the next integration; no source rollback or target rebuild is claimed here. The expanded full-logit tests remain useful coverage.

The shape microdiagnostic isolatedP128 capacity128 versus512 and found~6% lower prefill span. Serving mixes prompt lengths, includes decode/scheduling/CPU work, and changes only some prefills; that result did not establish an end-to-end gain. This batch demonstrates why profiling improvements require matched serving validation.

409 evidence files SHA256 verified via `raw/serving-round51-manifest.json`. Authoritative raw analysis retains the generic analyzer decision string; this section records the reviewed acceptance decision. Next experiment is the query-tiled attention prototype, still outside the authoritative source. Its8-row memcheck passed46 cases throughcontext4096 including full-tile causal poisoning; racecheck and16-row checks are pending in live session41524 (`/tmp/riley-opt-260912/check_query_tile_v44.py`). Do not restart on a polling timeout. Blender remains stopped.

## V44 query tile primitive results; serving integration pending

Both8/16-row prototypes pass46 cases under memcheck and racecheck, throughcontext4096. Finite outputs match the independent old oracle exactly. Causal suffix poisoning with all query rows active preserves exact finite prefixes, with NaN classification checked against the oracle for contaminated outputs. No serving speedup is established.

| Rows / start | Existing us | Tile8 us | Tile16 us |
|---|---:|---:|---:|
| 16 / 0 | 3.62 | 6.15 | 6.55 |
| 128 / 0 | 15.62 | 13.79 | 14.58 |
| 398 / 0 | 109.54 | 45.51 | 44.52 |
| 16 / 128 | 14.73 | 17.74 | 19.14 |
| 128 / 128 | 30.35 | 25.61 | 25.94 |
| 398 / 128 | 150.94 | 56.32 | 54.44 |
| 16 / 1024 | 90.09 | 95.59 | 102.55 |
| 128 / 1024 | 124.45 | 102.38 | 109.38 |
| 398 / 1024 | 450.95 | 145.94 | 139.08 |
| 1024 / 3072 | 2712.22 | 1077.10 | 758.05 |

Each value is the median of4 paired orders,20 warmup launches and100 measured launches, CUDA-event timing without profiler. GPU cooldown≤48C precedes each width; no other compute process is allowed. These isolated kernel measurements screen candidates and do not include serving costs. Tile8 is2.4–3.1x faster for398 query rows but regresses16-row cases. Tile16 helps the largest cases more but is slightly slower at128 rows.

The next8-row variant selects original per-query arithmetic below32 live rows and query tiling above, within one kernel. Extra tests cover actual live rows against a retained capacity of512 or1024. Its memcheck passed; racecheck remains live under session34743 (`/tmp/riley-opt-260912/check_query_dispatch_v44.py`). Timing and full-model integration are pending.

V43 graph additions were removed in remote commit1979ab3; expanded tests remain. The serving target binary has not been rebuilt after that removal and is still V43; frozen V42 is the accepted next comparison baseline.17 primitive evidence files SHA256 verified via `raw/query-tile-v44-manifest.json`.

## V44 integrated and frozen — Round52 running

Source `eea543563ebf1e71daffc2d8fe4298e219e43650`; binary SHA256 `3a661dd0b1a1e47d897886b54881fea2325cafad76e9c783d945e1c83db234c5`. Prefill uses8-query MMA tiles for32 or more live rows, original per-query arithmetic below32, and uniform original-arithmetic fallback for nonfinite V. Kernel grid covers both short queries and retained-capacity tiles without requiring extra captured graphs. New header participates in native rebuild dependencies and graph catalog identity. V43 graph additions have been removed; decode is unchanged fromV42.

The dispatch variant passes46 independent-oracle cases under memcheck/racecheck, including live input smaller than retained512/1024 capacity and mixed-query causal poisoning. Fixed-capacity kernel timing improves all sampled cases: at start0 existing/new16 rows4.58/3.68us,128 rows21.10/14.37us,398 rows97.89/44.79us. This remains an isolated diagnostic. Integrated owned GPU5 tests pass5344 full-logit rows exactly, with pending-close/abort and zero final allocations. HTTP37 passes including32 simultaneous mixed requests and disconnect recovery. The repo-local independent probe builds and runs successfully after integration.

Precheck archive SHA25625858fb38c6426c10f2c1e4915a767422e293b9b46c5d8eb616fee118c3138a3 verified on download; proof lives under `raw/query-v44*`, `raw/query-dispatch-v44`, and `raw/v4-http-v44-c32-final`. Remote source clean, target and frozenV44 agree.

Round52 now compares frozenV42/V44/vLLM across C4/8/16/32 fixed/natural, two reversed orders,96 warmup plus384 retained requests per lane. Live remote PID3763900, local session26262; controller `/tmp/riley-opt-260912/serving_screen_round52.py`, log `/tmp/riley-opt-260912/serving-screen-round52.log`, outputs `/tmp/riley-opt-260912/variable-serving-screen-round52`. Inspect current process state before further GPU work; never restart on a polling timeout. No serving performance claim yet; full goal remains unmet. Blender remains stopped.

## Round52 completed — retain V44, full serving goal still unmet

All48 lanes finish:18432 retained requests, zero failures,12288 exact Riley reference matches. Two reversed orders per case, with96 warmup and384 retained requests per lane. No long-run stability qualification; vLLM output sequences are not all identical to the Riley reference.

| Case | V42 tok/s | V44 tok/s | vLLM tok/s | Throughput change | TPOT change |
|---|---:|---:|---:|---:|---:|
| c16-fixed | 3775.6 | 3821.9 | 4975.8 | +1.23% | -1.27% |
| c16-natural | 5550.2 | 5829.7 | 7734.2 | +5.04% | -4.87% |
| c32-fixed | 3805.6 | 3812.9 | 7101.2 | +0.19% | -0.27% |
| c32-natural | 5417.5 | 5676.3 | 11699.7 | +4.78% | -4.12% |
| c4-fixed | 2366.6 | 2386.2 | 1916.2 | +0.83% | -1.64% |
| c4-natural | 2639.2 | 2698.1 | 2574.7 | +2.23% | -1.91% |
| c8-fixed | 3320.3 | 3354.9 | 3147.3 | +1.04% | +1.29% |
| c8-natural | 4231.2 | 4391.8 | 4603.2 | +3.80% | -3.48% |

Retain V44 as next baseline: mixed-serving throughput improves2.23-5.04 percent, with TTFT, TPOT and sampled E2E tails improving in each mixed case. Fixed cases show small gains; C8-fixed TPOT regresses1.29 percent and C32-fixed P95 regresses0.43 percent. High-concurrency vLLM gap and long-run stability remain unqualified. Next target unchanged decode value attention.

C16/C32 natural TTFT decreases19.07%/27.28%, E2E P95 decreases4.37%/4.61% and P99 decreases7.42%/6.49%. These improvements are relative toV42, not vLLM. V44 high-concurrency throughput remains well below vLLM.

Post-benchmark V44 node traces complete192 exact requests. C16/C32 median prefill graph spans2908.170/2904.587us, decode1539.797/1552.084us. Prefill attention mean19.289/19.831us versus V42 37.947/38.695us. This is profiling evidence from mixed-query runs; median replay composition and profiler overhead prevent translating the kernel ratio directly into serving speedup. Decode remains essentially unchanged, supporting the next value-attention experiment.

438 exported evidence files SHA256 verified via `raw/serving-round52-manifest.json`. Remote source remains clean at eea5435; frozen V44 and target retain the verified build. Blender remains stopped. Round52 controller and trace runners terminated successfully.

The grouped decode values V45 prototype now passes the independent row reference, memcheck and racecheck for1..16 active rows, invalid0/17, context boundaries through4096 and disjoint permuted pages. Checks are in `raw/grouped-values-v45`. It has not been integrated or timed. Next: paired existing/grouped value-kernel timing and full-model validation before a serving benchmark; do not assume fewer CTAs is a win. No GPU experiment is left running by this turn.

## V45 rejected; fresh V44 result-path profile

Grouped3-head and paired2-head decode value prototypes both pass correctness, memcheck and racecheck. Paired kernel timing has4 reversed orders with40 warmup and200 launches per sample. Both regress longer contexts, so neither was integrated.

|16 active rows / context|Existing attention us|Grouped3 us|Paired2 us|
|---|---:|---:|---:|
| 16 | 7.45 | 7.30 | 7.24 |
| 128 | 8.92 | 8.36 | 8.52 |
| 398 | 15.33 | 16.10 | 16.74 |
| 1024 | 27.15 | 30.25 | 32.24 |
| 4096 | 92.64 | 107.18 | 118.58 |
| -1 | 59.25 | 77.27 | 90.92 |

Context−1 denotes the mixed per-row context corpus. Grouped sharing trades less repeated V/MMA work for more per-thread normalization and less parallelism; the measured regression rejects the presumed benefit. No serving comparison was claimed for these unintegrated candidates.21 V45 evidence files SHA256 verified.

Fresh V44 instrumentation b0946f86202616f55c410f113dabf2856934243a703216f95605b7001f80e530 completes1536 exact requests across V3/V4 and C16/C32. Middle80 V4 decode native1469/1484us, CPU validation109/125us, encode26/42us, total1604/1654us; active rows15.787/16. V4 prefill share31.60%/31.71%. Native includes synchronization/readback and is not GPU-only. Instrumented source restored; clean source and serving target matching frozenV44 verified.21 profile evidence files SHA256 verified.

Next batch rationale and preserved contracts are in `COMPACT_GREEDY_NEXT.md`: reuse existing GPU finite/lowest-ID argmax semantics and assess compact greedy result transfer. This remains unimplemented. V44 remains accepted baseline; full vLLM performance and long-run stability goals remain unmet.

## V46 GPU full-logit verification primitive: promising, not integrated

Existing one-block-per-row argmax and a new2048-element split/reduce variant each pass306 correctness cases with memcheck/racecheck. Tests cover every BF16 bit pattern in both two-column orders, tie ordering, signed zeros, NaN/Inf, real vocab49152 and output guards. Split reduction retains lowest finite maximum token and fails any row containing a nonfinite value.

At16 rows, graph-based argmax plus compact D2H takes14.11us with the split variant, versus40.73us for the existing GPU kernel and91.80us for full D2H alone in the split comparison. At8 rows compact13.98us/full30.56us; at4 rows13.94/15.64us; at1 row13.92/4.45us. CPU full-logit verification is excluded from the full-copy measurement, so these are component costs, not a serving speedup claim. Each measurement uses4 reversed-order pairs with30 warmup and100 graph replays.

18 evidence files SHA256 verified in `raw/greedy-v46-manifest.json`. Runtime source and accepted V44 serving binary are unchanged. One initial SSH file transfer timed out before the test process existed; the confirmed failed transfer was retried, then both runners completed. No GPU job remains running. Compact result transport, identity validation, sampling eligibility/fallback and end-to-end tests are still to implement, as detailed in `COMPACT_GREEDY_NEXT.md`. Full goal remains unmet.


## V46 compact results: integrated, measurement pending

V46 is frozen at `4668b77b2e8c5311fdd5abfb2b0b665775b7ef66`, binary `e64ea083114f8a56608e68d96be2aa2aa74ad09ddc3b50f19c32695e6cc5fabc`. See `COMPACT_GREEDY_NEXT.md` for exact GPU finite validation, compact identity protocol, full-logit fallback, token workspace reuse, correctness coverage and fixed integration failures. All 75 exported evidence files are SHA-verified. Round53 C16/C32 matched V44/V46/vLLM serving comparison is running; V44 remains the accepted baseline and the overall vLLM goal remains incomplete. Blender stays stopped.


## V46 measured result

Round53 is complete. V46 improves V44 throughput15.13–19.01% across C16/C32 fixed/natural conditions, with lower TTFT/TPOT/P95/P99 in all four. 9,216 retained requests succeed and6,144 Riley references match. It is adopted for these measured GPU-greedy conditions, but V46 throughput remains10.98–43.36% below vLLM; the overall goal is not complete. Exact ratios and next decode-computation/batching focus are in `V46_SERVING_RESULTS.md`. New192-request Nsight evidence and223 exported files are SHA-verified. All test/benchmark/profiler controllers are terminal; Blender stays stopped.


## V47 width32 prototype validated

The next calculation/batching batch has a verified32-row primitive prototype, with612 configurations passing exact BF16/guards under memcheck and racecheck. At32 active rows, fused QKV is39.3% shorter, gate/up32.5% shorter and attention18.5–38.5% shorter than two existing16-row groups. Smaller active counts regress in some cases; existing V46 configurations must remain available. No serving or full-model32-row claim yet. See `DECODE_WIDTH32_V47.md` for remaining integrated wire/session/scheduler/model work. All22 evidence files are SHA-verified; accepted source remains V46.


## V47 integrated; Round54 pending

V47 source `ead5c09f9cb7bd3d62ad39902e65801582ab6479`, binary `0da2d8f74f3fb1fa7d5692fdecd6fa271519a328c6574227e94ef01b7f991090`. V5 wires and executes32 rows with full/compact completion, scheduler ownership, bounded output slots and full sampling fallback. Ten model tests including32-live-row4,096-output full and alternating checks pass;14 wire tests,3 model memcheck tests,37 HTTP references and22 CPU/GPU fallback responses per backend pass. Round54 C16/C32 V46/V47/vLLM serving comparison is running. See `DECODE_WIDTH32_V47.md`; goal remains active and Blender stays stopped.


## V47 measured: do not adopt globally

Round54 is complete: C32 natural throughput+26.36%, TPOT−22.79%, P99−20.07%, but fixed P128 TTFT rises7.997→59.466ms with throughput+0.32%. General baseline remainsV46. All9,216 retained requests succeed and6,144 Riley references match. Four new profiler lanes/384 references reveal underfilled wide decode in short outputs and a client first-token wait backlog increase1→13. Next investigate packed multi-owner prefill plus scheduler class selection; no benchmark-specific success criterion. See `V47_SERVING_RESULTS.md`. All292 evidence files are SHA-verified; sourceHEAD remains experimentalV47 and frozenV46 is unchanged. Goal active, Blender stopped.


## V48 packed prefill prototype qualified

Corrected current-loader RoPE fixture passes32 full-model cases, memcheck and racecheck, including up to4 owners and1,024 aggregate tokens. Selected hidden, entire KV, partial/inactive zeros and guards match sequential execution. Component timing and qualification boundaries are in `PACKED_PREFILL_V48.md`; 15 corrected evidence files are SHA256 verified. Initial older-RoPE evidence remains explicitly preliminary. V48 serving integration is pending, general baseline remainsV46, goal active and Blender stays stopped without restoration.


## V48 integrated; Round55 measurement started

V6 packed prefill plus scheduler admission is integrated at `4195aa5dd08e5ebbc604c89a09ae8831fc339e0d`. Three new model tests and ten compatibility tests pass; V6 memcheck,37 HTTP references and22 ordered CPU/GPU fallback responses pass. All59 integration evidence files are SHA256 verified. Round55 compares V46/V48/vLLM at matched512-token budget on C16/C32 fixed/natural workloads. See `PACKED_PREFILL_V48.md`. General baseline remainsV46; no V48 serving adoption claim. Goal active, Blender stays stopped.


## V48 measured; remaining serving gap

Round55 completes9,216 retained requests with0 failures and6,144 exact Riley references. V48 improvesC32 throughput36–40% overV46, but remains14–20% behindvLLM with slowerTPOT. C16 fixed exceedsvLLM throughput12.10% but missesTPOT. V48 is the next optimization comparison candidate; frozenV46 remains a latency reference, with no general production/default promotion. All202 benchmark files are SHA256 verified. See `V48_SERVING_RESULTS.md`; freshV48 profiling is running. Goal active and Blender stays stopped.


## V48 profiling complete; V49 mixed-execution direction

All288 new diagnostic references match. C32 fixed prefill accounts for57.64% of the summed middle-window GPU graph spans; natural C16/C32 decode accounts for73.13%/67.54%. The next batch will test mixed prefill/decode execution and compact per-owner query-tile mapping, with V7 ownership/result contracts and full serving validation. See `MIXED_PREFILL_DECODE_V49.md`. All35 profile files are SHA256 verified; all benchmark/profile controllers are terminal. Source remains frozenV48, general promotion and full goal unproven. Blender stays stopped.


## V49 geometry prototypes; mapped qualification running

242 attention cases and30 selected-hidden/entire-KV model cases pass for the first32-owner prototype. Timing rejects its persistent tile loop for many small chunks. A direct tile map removes that regression in the tested32-owner attention patterns, while small-owner regressions remain explicit. The mapped full-model correctness run passes; mapped memcheck/racecheck remain live in session21123. All40 completed evidence files are SHA256 verified. See `MIXED_PREFILL_DECODE_V49.md`. No V7 serving integration or V49 performance claim yet; frozenV48 source is unchanged and Blender stays stopped.


## V49 integrated; Round56 running

V7 mixed serving is frozen at `c63a395cdbaea050b8281aa58bfb0f9ac681bb14`. Mapped full-model memcheck and attention racecheck pass;18 wire,37 scheduler,16 owned model tests, V7 memcheck,37 HTTP references and22 ordered CPU/GPU fallback responses pass. All64 integration evidence files are SHA256 verified. Round56 is comparing V48/V49/vLLM under matched512-token budgets. See `MIXED_PREFILL_DECODE_V49.md`. Goal active; Blender stays stopped.


## Round56 및 V49 trace 완료

24 lane,9,216 요청 실패0, Riley6,144 기준 일치. V48 대비 처리량 +2.43~10.05%이나 vLLM 대비 TPOT 목표 미달이다. 별도288 요청 Nsight trace도 기준 일치로 완료했다. [최신 결과와 다음 후보](V49_SERVING_RESULTS.md)를 따른다. Blender는 종료 상태를 유지한다.


## V50 decode GQA batch 검증 중

Grouped QK와 같은CTA의 독립3warp values를 선택했다. 840개 primitive 정확성 검사가 통과했고 커널 비교20조건에서 attention시간4.86~38.51% 감소했다. 원격 mixed profile decode에 통합했고 sanitizer→전체 build→실제 모델 GPU 회귀 controller가 실행 중이다. Serving 성능은 아직 미검증이다. [진행 기록](GQA_ATTENTION_V50.md).


## V50 frozen 및 Round57 시작

실제 모델16회귀·3memcheck·HTTP37·fallback22/backend가 통과했다. V50 commit44ccb8d를 고정하고 V49/V50/vLLM의24lane matched serving을 실행 중이다. [최신 V50 기록](GQA_ATTENTION_V50.md).


## Round57 완료

9,216 요청 실패0, Riley6,144 기준 일치. V50의V49대비 처리량 개선은C16+3.45~3.94%,C32+0.24~1.19%이며 vLLM 대비TPOT는 아직8.44~53.05% 느리다. C32 fixed P99도2.62% 악화해 전체 성능 개선을 단정하지 않는다. [최신 결과](V50_SERVING_RESULTS.md). 별도 V50 trace 실행 중이다.


## V51 fusion batch frozen 및 Round58 실행

Projection 재사용과 gate/up/SwiGLU fusion을비교해 M16fusion을선택했다. Primitive1728case·memcheck·실제모델16회귀·V7modelmemcheck·HTTP/fallback통과. Commita0536a5를고정하고 V50/V51/vLLM24lane matched serving실행중. [구현및증빙](PREFILL_FUSION_V51.md).
