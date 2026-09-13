# GPU future-token 전달 경계

FP16 prefill의 모델 품질 실패 후 PR02의 남은 overlap 실행 경로로 돌아왔다. 기존 C32 trace의 graph 간 시간 비중 17.47%는 구현 우선순위 근거이며 실제 serving 가속 예측이 아니다. 이 결과는 GPU 전달 primitive 검증이고 두 iteration ahead serving 완료가 아니다.

`kernels/optional/future_token.cuh`는 다음 기능을 하나의 전달 경계로 구현한다.

- 선행 compact V7 greedy 결과의 status·publication·argmax·vocabulary 검사 및 token을 제외한 31개 word의 canonical identity 비교.
- generation, 연속 replay/iteration, catalog, sequence/cookie, progress를 다음 descriptor에 연결하고 source row 재배치를 허용한다.
- 한 CTA에서 모든 reference를 먼저 검증한다. 한 row라도 실패하면 packet 전체를 변경하지 않고 status64를 기록한다. 기존 status도 보존한다.
- Future row의 token을 descriptor 및 packed token slab에 전달한다. Host-token row는 그대로 보존한다.

최대 32-row sidecar는 4,224 bytes, compact 이전 결과는 4,096 bytes다. 이는 이 두 데이터 구조의 크기이며 전체 실행 owner 메모리나 serving 메모리 증가 실측이 아니다. 미지원 ABI/layout을 자동 추정하지 않는다.

## GPU 검증

동일 graph에서 device producer가 매 replay token을 바꾼 뒤 consumer가 읽도록 했다. 중간 token host readback 없이 1/16/32-row, descriptor/token-slab 경로, 역순 source row 매핑, identity/status 32-word 변조, 범위·generation·replay·progress 오류, invalid suffix, 기존 오류 보존과 host-token bypass를 조합한 288개 검사가 통과했다. 거부 case는 모든 packet word가 원래 값 그대로인지 비교했다.

RTX 4090 / CUDA 13.0.88에서 memcheck 0 errors, racecheck 0 errors/0 warnings. SM90a/SM100a compile 통과, 해당 GPU runtime은 장비 부재로 skip했다. [manifest와 소스 hash](manifest.json), [실행](run.log), [memcheck](memcheck.log), [racecheck](racecheck.log).

Fixture는 전달 필드와 buffer 범위를 검증한다. 완전한 scheduler-authorized V7 packet이나 실제 KV 예약을 생성하는 검사가 아니다. 특히 packed slab 경로 검사를 실제 mixed serving 검사로 해석하지 않는다. Probe producer도 모델 sampler가 아니라 GPU 값 전달을 검증하는 작은 kernel이다.

## 통합 시 반드시 유지할 경계

호출자는 canonical sidecar를 retained authority에서 만들고, V7 전체 구조·KV page ownership을 별도로 검증해야 한다. 원본 결과와 sidecar·목적 packet은 같은 stream 순서 또는 명시적인 event dependency로 보호하고 마지막 읽기가 끝날 때까지 보유한다. HostToken은 값의 유효성을 증명하지 않으므로 기존 host-token 검증도 유지한다.

`decode_shared32_model.cuh`는 enqueue 시작에서 status를 0으로 초기화한다. Resolver를 그 앞에 단순 추가하면 실패가 지워진다. 실제 연결은 **status 초기화 → resolver → embedding → model → result** 순서를 보장해야 한다. 원본 compact 결과의 재사용·덮어쓰기 시점도 함께 연결해야 한다.

다음 batch는 Rust future-token 표현과 canonical result identity 생성, 최대 두 expectation/ticket 및 slot lifetime, scheduler tentative KV reservation·순차 commit을 연결한다. EOS/cancel 이후 후행 결과 공개 금지와 GPU drain 전 page 반환 금지도 포함한다. 기존 단일 in-flight authority를 우회하지 않는다.

현재는 runtime/FFI/모델/서버에서 선택하지 않는 native prototype이다. 실제 모델 correctness, EOS/cancel, allocation-zero, overlap trace, 동일 조건 vLLM 표가 남아 있다. 새 serving 성능을 측정하지 않았고 기본값을 변경하지 않았다.


후속 Rust 연결에서 실제 runtime은 매 제출 cookie를 새로 발급한다는 차이를 확인했다. 이 v1의 same-cookie 가정은 실제 session과 호환되지 않아 [v2 wire/ABI](../20260913-future-token-wire/README.md)로 수정했다. 위 수치와 manifest는 역사적 native v1 검사이며 현재 통합 계약으로 사용하지 않는다.
