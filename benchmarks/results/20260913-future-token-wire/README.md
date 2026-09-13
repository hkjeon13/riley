# Future-token Rust wire 및 제출별 cookie 연결

Runtime의 `issue`는 매 제출마다 새 cookie를 발급한다. 이전 native v1 prototype은 선행/후행 cookie가 같다고 가정해 실제 session과 맞지 않았다. 새 ABI는 이전 결과의 canonical identity와 다음 제출 cookie를 각각 보관한다. 이 수정 전 v1 검사는 native fixture 내 검증이며 실제 runtime 호환성 증거가 아니다.

## 구현

`multi_descriptor::future_token::prepare`는 두 expectation과 `Host` / `PreviousRow` 선택에서 tentative packet·sidecar를 만든다. V7 mixed32 greedy, 동일 owner/catalog, 연속 replay/iteration, 새 cookie, source publication, request identity, decode progress, 이전 KV page prefix와 전체 ownership 보존을 검사한다. 결과를 출력한 선행 요청의 후행 입력을 임의 Host token으로 대체할 수 없다. 미완료 prompt는 기존 page의 끝에서 Host prefill로 이어갈 수 있다.

후행 expectation의 `last_accepted_replay`는 실제 committed 값 그대로다. 구조 검사에만 내부 clone을 사용하며 clone이나 임의 commit authority를 반환하지 않는다. Prepared batch는 기존 단일 in-flight session에 제출할 수 있는 API가 아니다. 이번 변경은 scheduler authorization/두 예약을 만드는 구현이 아니라, 그 adapter가 사용할 checked wire 준비 계층이다.

Native sidecar는 row당 140 bytes, 32-row 총 4,480 bytes다. source row + canonical result 128 bytes + destination cookie 8 bytes를 little-endian으로 인코딩한다. source result cookie를 next packet cookie와 비교하던 조건을 제거하고, 각각 자기 기대값에 묶으면서 새 cookie가 이전보다 큰지 검사한다. ABI v1의 132-byte row와 호환되지 않는다. 현재 두 ABI 모두 외부 serving에 노출하지 않은 prototype이다.

## 검증 범위

- Rust targeted 4개 검사: page16→17 경계·역순 row mapping, committed replay 유지·서로 다른 cookie, 새 prefill admission, partial prefill continuation, stale identity/order/cookie·page remapping·없는/중복 source 거절.
- Rust canonical encoder가 실제로 구조 검증한 pure-decode packet과 새 prefill+future-decode mixed packet을 export했다. 해당 packet·reference·canonical 이전 result를 GPU probe가 직접 읽었다. 양쪽에서 같은 4,480-byte ABI를 확인했고, GPU producer→resolver 사이 host token readback 없이 두 경로 모두 packet 전체가 예상값과 일치했다.
- 수정 native의 288-case mutation suite를 재실행했다. 두 Rust fixture 각각 memcheck 0 errors, racecheck 0 errors/0 warnings.
- SM90a / SM100a compile 재통과, 실제 runtime은 장비 부재 skip.

전체 runtime 회귀는 319 passed / 1 ignored(기존 timing diagnostic)로 통과했다. 이 실행은 partial-prefill 조건 보강 전 snapshot이며, 보강 후 최종 future-token 4개 검사를 재실행했다. [전체 회귀](runtime-tests.log), [GPU manifest](gpu/manifest.json), [최종 소스·fixture hash](source-hashes.json).

[최종 targeted 검사](future-tests.log), [decode GPU](gpu/run.log), [mixed GPU](gpu/mixed.log), [decode memcheck](gpu/memcheck.log), [mixed memcheck](gpu/mixed-memcheck.log), [mixed racecheck](gpu/mixed-racecheck.log).

GPU producer는 값 전달 검사용 kernel이다. 이전 결과의 실제 모델 생성, retained resource lifetime, 두 scheduler reservation·순차 settlement, EOS/cancel 이후 drain·추가 결과 폐기는 아직 연결하지 않았다. Canonical wire validation은 GPU 완료나 scheduler 소유권 증거를 대신하지 않는다. 기본 backend 변경과 새 serving 측정은 없고, vLLM 비교표도 갱신하지 않는다.

다음 연결은 두 계획을 함께 보유하는 scheduler authority와 runtime expectation queue다. Model status 초기화 뒤·embedding 앞의 resolver 삽입, source result 재사용 시점, KV prefix commit과 suffix drain을 함께 구현해야 한다. 전체 serving 목표는 미완료다.
