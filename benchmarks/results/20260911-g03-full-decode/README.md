# SmolLM2 M=1 전체 decode 통합 — 2026-09-11

## 구현 결과와 범위

`generate_greedy_with_decode_graph`는 실제 prepared executor를 소비하여 prompt를 기존 eager 경로로 prefill하고, 마지막 prompt 토큰부터 연속 생성까지 하나의 retained CUDA graph로 실행한다. 지원 조건을 만족하지 않으면 graph 작업 전에 기존 eager 생성 경로를 선택한다. 실행 중 오류 후에는 변경됐을 수 있는 KV로 재시도하지 않으며 부분 토큰을 반환하지 않는다.

Graph는 fresh token/metadata H2D → embedding → 모든 layer의 norm/QKV/RoPE/KV write/attention/output projection/residual/norm/MLP/residual → final norm → LM head → greedy argmax → token/status D2H를 포함한다. 생성 모드는 전체 logits를 D2H하지 않는다. 진단 모드만 전체 logits를 반환하여 독립 eager와 비교한다.

실제 weight/selected GEMM plan/scratch/KV/RoPE 부모를 하나의 native ledger가 유지한다. 전용 metadata/result/staging 부모도 같은 ledger에 등록한다. layer 내 residual scratch 역할을 고정하고 다음 layer hidden으로 연결하므로 capture 중 Rust callback이나 host-side buffer 교대가 없다. capture 전에 모든 부모·크기·alias·GEMM binding을 검사한다.

## 검증된 다섯 단계

| 단계 | 이번 검증 |
| --- | --- |
| 단일 layer 연결 | attention 앞부분과 projection/residual/norm/MLP 뒷부분을 한 native sequence로 연결 |
| 모든 layer 연결 | SmolLM2 30-layer 및 canonical 2/3-layer, 모든 logits와 전체 K/V parent byte 일치 |
| decode 입출력 연결 | embedding부터 argmax와 token/status D2H까지; 진단 모드 logits 비교 |
| 연속 decode | 모델별 64토큰, 4개 block, 16/32/48 경계, 비순차 physical mapping `[2,0,3,1]`; 17-token prompt 뒤 48-token 생성 일치 |
| 지원 조건·오류·수명 | 사전 eager fallback, invalid token/position/offset/prefix/padding/duplicate block 거절, stale output 차단, 비정상 GPU 출력 거절, 성공/실패 cleanup 및 완료 불명 보존 |

이 표는 **새 명시적 M=1 생성 API의 기능 통합**을 뜻한다. 기존 C06/C07 registry/default server dispatch는 전환하지 않았고, 기존 14-slot aggregate qualification을 자동으로 Supported로 승격하지 않았다. 새 전용 metadata layout의 증거를 기존 exact-slab evidence로 대체하지 않는다.

## 실행 증거

- 일반 CUDA 빌드: `build-production-final.log`, exit 0. 기존 dead-code 경고 존재.
- 일반 GPU 검증: `gpu-qualified-all.log`, 총 25 passed(전체 decode 3 + 기존 C07 모델 14 + native 8).
- 오류 관측 주입: `gpu-lifecycle-fault.log`, 1 passed, 두 격리 child에서 각각 launch-error/완료불명 관측 정책 검증.
- CPU: 377 passed(84 cuda lib + 259 runtime lib + 32 graph + 2 inventory/binding integration). `cpu-cuda-final.log`, `cpu-runtime.log`, `cpu-inventory.log`.
- fmt/diff 및 C ABI 구문 검사 통과. 일반 CPU Clippy 완료, 경고 존재. CUDA feature의 실제 Rust/C++ 빌드와 GPU 실행이 별도 증거다.
- `verification.json`: 최종 로그의 테스트 수, 전체 decode hash, entry/prefix 결과, 기존 5개 binding receipt와 7개 경로 replay 수.
- `remote-identities-final.txt`: 9개 소스, 일반/오류주입 바이너리, checkpoint manifest, GPU identity. 일반 바이너리에 새 오류주입 심볼이 없음을 확인.
- `transfer.json`, `sources.tar.gz`: 최종 source identity와 전체 payload. 사용자에게 구체적 payload/대상 승인을 받은 뒤 전송했으며 자동 승인 차단은 해소됐다.
- `model-preservation.json`: 별도 model-loader 변경 6파일의 hash 보존.

오류주입은 **실제 CUDA 작업을 먼저 동기화한 뒤 관측된 반환값만 바꾸는 테스트 전용 기능**이다. 완료 불명 상태에서 graph/부모/stream/context 해제를 차단하는 정책을 검증하며, 실제 GPU reset/device-loss 실험을 의미하지 않는다. 의도적으로 유지한 자원은 격리된 child process 종료 시 정리된다. 일반 CUDA feature에는 이 hook이 없다.

중간 build/typecheck 로그와 40-token/이전 64-token 로그는 작업 이력이며, 최종 판정은 위 지정 로그와 manifest를 사용한다. 임시 복사본의 Rust-only typecheck는 누락된 header 때문에 중단됐으며 CUDA 검증 근거로 사용하지 않았다. 이후 실제 원격 CUDA 빌드가 통과했다.

## 사용과 남은 확대 범위

API: `PreparedLlamaBatchExecutor::generate_greedy_with_decode_graph(self, context, stream, prompt, steps)`.

- 별도 sampling penalty/EOS/stop 정책 없이 정확히 steps개의 greedy token을 반환한다.
- 현재 검증은 M=1, D64, canonical attention, canonical/HF norm, zero-workspace selected no-split GEMM, bias 없는 projection에 한정한다.
- prompt prefill은 한 토큰씩 기존 eager executor로 처리한다. 외부 scheduler가 관리하던 KV prefix를 인수하거나 기존 서버의 기본 경로를 교체하는 API가 아니다.
- 실제 검증 context는 최대 64토큰/4 physical blocks다. 더 긴 context, 다중 요청·M>1 bucket, Fixed37 및 nonzero-workspace graph, 일반 C06 registry 연결은 이번 증거로 완료 처리하지 않는다.
- 기존 기본 경로로의 전환·배포·커밋은 수행하지 않았다.

**성능 측정과 vLLM 비교는 시작하지 않았다. 속도 향상이나 일반 workload의 성능 측정 준비 완료를 주장하지 않는다.**
