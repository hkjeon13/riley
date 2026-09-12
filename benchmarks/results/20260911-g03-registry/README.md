# Full decode 생성 API의 C06 registry 연결 — 2026-09-11

## 변경

`graph_decode_full.rs`의 실제 전체 decode 생성 경로를 기존 `GraphRegistry<1>` 및 `select_registered_execution_graph`에 연결했다. native capture/instantiate 성공 후에만 Prepared 항목을 만들며, registry는 exclusive native graph owner와 함께 유지된다. 선택된 FullGraph slot이 이 owner의 slot 0과 일치해야 replay할 수 있다. replay/read 실패 후에는 owner를 poisoned로 처리하여 추가 선택·실행을 차단한다.

기존 `generate_greedy_with_decode_graph`는 Auto로 위임한다. 새 `generate_greedy_with_decode_graph_policy(..., ExecutionGraphPolicy)`는 다음 정책을 제공한다.

- Disabled: signature hash와 graph 준비 없이 기존 eager 실행.
- Auto: 지원 조건을 만족하면 cold graph를 준비·등록하여 C06로 선택하고, 미지원 조건은 C06 정책으로 eager 선택.
- Require: 미지원 조건을 prefill 전에 거절하고, 등록 후 exact key/slot 선택 실패도 거절.

실행 중 실패한 graph를 eager로 재시도하거나 부분 토큰을 반환하지 않는다.

## 키와 owner 결합

키는 실제 업로드된 weight/RoPE table bytes, weight-role index, norm epsilon/profile, RoPE theta, 모든 selected GEMM algorithm 옵션·shape·workspace·수치 flag, GPU compute capability/runtime/cuBLASLt/native ABI, 모델 geometry 및 전용 metadata layout으로 만든다. 파일 경로나 포인터, Debug 문자열을 fingerprint로 사용하지 않는다. 모델/config revision digest에 실제 content와 binding/plan 정보를 포함하고, C06에서는 전체 `GraphSignature` equality를 사용한다.

weight/table hash는 **cold 준비 시 D2H로 읽어 계산**한다. replay 중에는 키를 다시 만들거나 weight를 읽지 않는다. 동일 모델이라도 plan이나 내용이 다르면 다른 키가 된다. registry는 generation owner 밖으로 공유하지 않으므로 동일한 논리 키가 다른 native graph의 slot을 참조하지 않는다.

registry footprint는 이 graph가 유지하는 device/pinned buffer 부모의 byte 수다. 불투명한 CUDA graph/driver 내부 할당이나 native host bookkeeping까지 계산한 총메모리 수치는 아니다.

## 검증

- 실제 원격 CUDA/native/Rust 빌드 통과: `build-final.log`.
- GPU 17 passed: 새 registry/full-decode 3 + 기존 C07 모델 회귀 14. `gpu-final.log`.
- CPU 281 passed: runtime lib 259 + 기존 dispatch/registry/registry-dispatch integration 22.
- SmolLM2 L30 및 canonical L2/L3: 64-token logits/전체 KV hash 유지, 17-token prompt 이후 48-token 생성 유지.
- 서로 독립적으로 준비한 같은 모델 owner의 키·buffer footprint 일치.
- weight byte 및 metadata capacity 변경 시 키 변경, 복원 후 원래 키 회복.
- 정확한 slot 선택, 키 miss 시 Auto eager/Require 거절, 잘못된 slot 거절, poisoned owner 재선택 거절.
- Auto/Require/Disabled 생성 결과 일치, 미지원 Require 거절, 기존 eager fallback/비정상 output status/cleanup 검증 유지.
- 최종 원격 source/binary/checkpoint/GPU identity와 별도 model-loader 6파일 보존 확인.
- fmt 및 diff 검사 통과. 빌드 경고는 존재한다.

`verification.json`과 `manifest.json`에 최종 로그의 parsed receipt 및 hash를 기록한다. 이전 whole-decode receipt와 동일한 logits/KV hash인지 비교하며 성능 수치로 해석하지 않는다.

## 남은 범위

이번 연결은 명시적 단일 시퀀스 생성 API 안의 C06 registry/정책 연결이다. `serve` 기본 요청 경로와 scheduler-owned KV lifecycle, 요청 간 graph cache 재사용은 아직 연결하지 않았다. 기존 C07 exact-slab의 별도 inventory를 전체 Supported로 일괄 승격하지 않는다.

검증 범위는 기존 M=1/D64/canonical/SmolLM2 64-token bucket이다. 다중 요청·더 긴 context·다른 profile 확대, 서버 기본 경로 전환 및 성능 측정은 수행하지 않았다. 다음 단계는 서버의 요청/완료/취소 수명에 이 owner를 연결하는 것이다.
