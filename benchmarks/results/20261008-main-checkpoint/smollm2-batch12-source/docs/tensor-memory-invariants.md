# Tensor와 CUDA memory invariant

이 문서는 PR 04에서 도입하는 tensor metadata, CUDA allocation, host/device copy의
안전 계약을 정의한다. 이 계층은 shape와 byte range를 검증하고 storage 수명을
표현할 뿐이다. dtype 변환, contiguous materialization, model-specific packing,
caching allocator는 수행하지 않는다.

## Tensor metadata

- `Shape`의 dimension과 `Layout`의 stride는 element 단위다. byte offset과 byte
  range는 dtype의 고정 byte width를 마지막에 곱해 계산한다.
- rank-0 shape `[]`는 scalar이며 element count가 `1`이다. dimension 하나라도
  `0`이면 empty tensor이고 element count와 logical byte length는 `0`이다.
- element count, 최대 reachable element offset, exclusive storage span, byte
  offset과 byte length는 모두 checked arithmetic으로 계산한다. overflow는
  wraparound나 allocation 시도로 바꾸지 않고 오류로 반환한다.
- non-empty layout의 storage span은
  `offset + sum((extent - 1) * stride) + 1` element다. empty layout은 data를
  참조하지 않으며 backing storage 끝의 one-past offset도 허용한다.
- canonical hidden activation layout은 `[batch, sequence, hidden]`, Q/K/V layout은
  `[batch, heads, sequence, head_dim]`이다. checkpoint 원본 weight layout과 향후
  execution packed layout은 서로 다른 layout으로 취급한다.
- transpose는 shape와 stride만 바꾸는 metadata operation이다. reshape는 logical
  element order가 보존되는 row-major contiguous view에만 zero-copy로 허용한다.
  지원하지 않는 layout을 암묵적으로 복사하지 않는다.
- dtype는 metadata operation에서 절대 암묵적으로 바뀌지 않는다. cast와
  contiguous copy는 후속 planner가 명시적인 operation으로 요청해야 한다.

## View와 alias

- `TensorView<'a, B>`와 `TensorViewMut<'a, B>`는 marker만 보유하지 않고 실제
  `&'a B` 또는 `&'a mut B`를 보유한다. 따라서 view는 backing storage보다 오래
  살 수 없다.
- immutable view는 복제할 수 있지만 mutable view는 `Clone`이 아니다. mutable
  metadata transform은 view를 소비하고 새 view 하나만 반환한다.
- mutable arbitrary-stride view는 distinct logical index가 같은 storage element를
  가리키지 않음을 보수적으로 증명할 수 있을 때만 생성한다. broadcast stride와
  겹치는 slice는 거부한다.
- `TensorStorage`는 crate가 sealing한 byte-addressable storage 계약이다. 사용자가
  잘못된 길이나 수명을 보고하는 구현을 추가할 수 없다.
- `Workspace<B>`는 호출자가 명시적으로 제공한 backing storage를 소유한다.
  자동 allocation, resize, clone 또는 dtype/layout 변환을 하지 않는다.

## CUDA allocation ownership

- device buffer와 pinned host buffer는 untyped byte allocation이다. tensor view는
  dtype와 element/byte span을 검증하지만 raw pointer를 만들거나 base-address
  alignment를 주장하지 않는다. 실제 typed kernel boundary가 생기면 그 operation의
  alignment capability를 별도로 검증해야 한다.
- buffer는 `Clone`이 아니며 safe API에서 raw pointer를 노출하지 않는다. 초기
  계약은 host thread 사이로 ownership을 이동할 수 있는 `Send + !Sync`다.
  공유 immutable weight는 이 타입에 임의의 `Sync`를 추가하지 않고 별도 타입과
  동기화 계약으로 도입한다.
- allocation 통계는 context owner별 device/pinned live byte 수와 allocation 수를
  하나의 coherent snapshot으로 조회한다. 같은 device ordinal의 별도 retained
  context owner는 통계를 공유하거나 resource ownership으로 간주하지 않는다.
- zero-byte allocation은 native CUDA pointer 없이도 소유권과 accounting을 갖는
  logical handle이다. allocation count는 증가하지만 live bytes는 `0`이다. close
  후 count와 bytes가 모두 원래 값으로 돌아온다.
- 명시적 `close`가 정상 오류 관측 경로다. native free를 시도한 결과가 모호하면
  handle 재사용이나 false zero-accounting을 허용하지 않는다. resource와
  accounting을 unresolved 상태로 보존하는 fail-closed leak이 double free나
  use-after-free보다 우선한다.
- allocation create 실패 뒤 rollback free까지 확정하지 못한 경우에는 반환할
  handle이 없어도 unresolved bytes/count와 context child lease를 영구 보존한다.
  따라서 실패한 create 경로도 통계 0이나 context release로 위장되지 않는다.

## Async copy lifetime

- H2D/D2H copy는 생성에 사용한 stream, device buffer, pinned host buffer를
  completion value가 mutable borrow한다. pending value가 살아 있는 동안 safe
  Rust 코드로 해당 resource를 재사용하거나 close할 수 없다.
- native handle에도 persistent active-use state를 둔다. copy 하나는 stream과
  두 buffer를 각각 exclusive하게 예약하며, 같은 resource를 쓰는 두 번째 copy,
  buffer close, stream close는 완료 처리 전까지 거부된다.
- Rust value에만 의존하지 않는다. `mem::forget`으로 pending value의 `Drop`을
  건너뛰어도 native reservation은 남는다. 그 resource는 영구적으로 busy한
  fail-closed 상태가 되고 context child ownership도 해제되지 않는다.
- reservation은 originating stream의 완료가 확인되고 호출 thread의 CUDA context
  복원까지 성공한 뒤에만 해제한다. query/synchronize 또는 context 복원 결과가
  모호하면 buffer를 free하거나 다른 stream으로 넘기지 않는다.
- copy enqueue에서 발견한 deferred CUDA 오류는 query, synchronize와 completion
  close에서 소실되지 않고 호출자에게 다시 보고한다. `Drop`은 unwind하지 않는
  best-effort fallback일 뿐 오류 보고를 대신하지 않는다.
- 한 buffer에는 동시에 copy 하나만 허용한다. 첫 copy를 명시적으로 완료한 뒤
  Rust borrow가 끝나야 두 번째 stream으로 handoff할 수 있다. 범용 multi-stream
  deferred-free queue는 이 PR의 범위가 아니다.
- zero-byte copy는 CUDA memcpy를 enqueue하지 않는 logical no-op이지만 동일한
  range, context-owner, reservation 규칙을 통과하며 명시적 완료 경로를 유지한다.

이 계약으로 host borrow와 device execution 사이의 간극을 Rust lifetime과 native
reservation 양쪽에서 닫는다. 성능 최적화 allocator나 더 느슨한 concurrency는
동일한 실패 경계와 accounting을 유지한다는 별도 증거가 있을 때만 후속 PR에서
추가한다.
