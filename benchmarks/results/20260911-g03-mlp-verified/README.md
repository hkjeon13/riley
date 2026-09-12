# G03 MLP chain — authorized remote validation

사용자가 소스 11개 전송 및 해당 파일 내 수정·재전송과 CUDA 검증을 승인한 뒤 기존 scratch `ai-assistant:/tmp/riley-g01-native-260910`에서 검증했다. 전송 전 10개 기존 파일은 예상 해시와 일치했고 신규 audit 파일은 없었다. 최초 전송 11개 및 최종 수정 후 11개 모두 로컬/원격 SHA-256 일치를 확인했다.

## 결과

- CUDA native 및 CUDA 활성화 Rust 빌드 성공. 최초 빌드에서 `graph_resources.rs`의 closure 인덱스 타입 추론 E0282를 발견해 `usize`를 명시하고 해당 승인 파일만 재전송했다. 최초 실패와 수정 후 성공 로그를 모두 보존한다.
- **GPU 20개 테스트 통과**: MLP native 1, C07 14, ledger/transfer 회귀 3, SwiGLU 회귀 2.
- 실제 모델 5개 case × 4회 × 32회 = **MLP 640 replay**. zero/original 입력 교대, down 및 residual 출력의 eager byte parity, 기존 선택 GEMM plan 불변성, pinned tail 보존 및 scratch 복원 통과.
- SmolLM2 L30, canonical L2/L3 logits·initialized KV 해시와 continuation은 직전 SwiGLU 검증 결과와 동일. SmolLM2 `[808,2775,288,536]`, L2 `[2,2,2,2]`, L3 `[5,5,5,0]`.
- native fixture는 잘못된 geometry/alias/부모 미등록/중복 capture/완료 전 read를 거절하고, workspace 없음/제공, close/Drop, capture domain 해제를 확인한다. 이 fixture 자체는 초기화되지 않은 weight를 실행하지 않으며 실제 실행 정확성은 모델 audit에서 확인한다.
- 원격 C11 ABI 검사와 로컬 fmt/diff 통과. 앞 단계 CPU 391개 통과 기록을 유지한다. 이번 타입 수정은 CUDA 조건부 코드이며 CUDA 활성화 빌드/GPU 테스트로 확인했다. 기존 컴파일러 경고는 남아 있다.
- 무관한 model-loader 6파일 전후 해시 일치. commit/push/deploy 없음.

## 범위와 남은 작업

검증한 graph는 staged input/residual → gate/up GEMM → SiLU/multiply → down GEMM → residual add → staged output이다. 완료된 iteration의 실제 scratch snapshot과 마지막 layer weight를 사용한 MLP subgraph 검증이다. 모든 layer의 실제 실행 중 입력 연결, 전체 decode graph, G02H/G03 qualification 또는 vLLM 대비 속도 향상을 증명하지 않는다.

다음은 post-attention norm과 MLP를 실제 layer 입력에 연결하고, attention 경로와 함께 동일한 retained model DAG로 통합하는 작업이다. 전체 decode replay·오류/완료 계약을 검증한 뒤 동일 조건의 SmolLM2 vLLM 비교를 진행한다. capture 종료·CUDA 실행 실패 주입과 필수 nonzero workspace 알고리즘 검증은 이번 통과 범위에 포함되지 않는다.

## 증거

- [승인 및 최종 소스 해시](authorized-transfer.json), [최종 원격 바이너리·checkpoint·소스 식별](remote-final-identities.txt)
- [최초 빌드 오류](build.log), [수정 후 빌드](build-fixed.log)
- [MLP native](gpu-mlp.log), [실제 모델](gpu-model.log), [ledger/transfer](gpu-aggregate.log), [SwiGLU](gpu-swiglu.log)
- [모델 parity 비교](model-parity.json), [무관한 변경 보존](model-preservation.json), [로컬 검사](local-checks.log)

이전 단계의 11개 파일 전송 승인 차단은 이번 승인과 실행으로 해소됐다.
