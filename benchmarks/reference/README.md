# Golden reference artifacts

Python reference lane이 생성한 작은 golden fixture는 다음 구조로 저장한다.

```text
benchmarks/reference/<fixture-id>.json
```

단일 JSON fixture는 token IDs, 선택 hidden state의 checksum·통계, 최종 logits와 FP32 log-softmax 통계·top-k, cache on/off greedy token IDs, request별 RNG 초기 snapshot을 기록한다. Tensor 값 전체 대신 canonical FP32 byte checksum을 사용하므로 sidecar tensor 파일이 필요하지 않다.

Checkpoint, 전체 hidden state/logit vector, pickle, Python object serialization은 커밋하지 않는다. 동일 model bytes, dependency lock, hardware/driver에서 fixture를 다시 만들 수 있어야 하며 기존 fixture를 덮어쓰지 않는다. Checksum은 reference artifact의 byte provenance이지 서로 다른 backend의 tolerance 기준이 아니다.

PR 01의 versioned artifact는 `smollm2-135m-bf16.json`이다. Primary RTX 4090
환경의 clean commit `2d22ca061f601389fad7f45708497daad14d9297`에서 active
`smollm2-fp32-bf16-native-e0-v2` gate를 사용해 offline으로 생성했다. 31개
case와 cache on/off 16-token exact window를 포함하며, 파일 SHA-256은
`87333a1859be45a2f8e7563d898dde5e64256ccc03ca4da3cab90def07dd3c95`, 입력
`prompts.jsonl` SHA-256은
`709612e45d735888b240951d51b979b7ded1e87ef6cae9296f0b1250647255d2`이다.
두 checksum은 파일을 이동하거나 복사한 뒤에도 반드시 다시 검증한다.

이 fixture와 그 v2 gate/matrix binding은 역사적 oracle trust anchor로 동결한다.
PR16의 `smollm2-fp32-bf16-native-e0-v3` candidate gate는 임계값이나 oracle
bytes를 다시 이름 붙이지 않고, v2 lineage를 명시한 채 canonical candidate만
비교한다. 따라서 이 파일은 v3 candidate evidence가 아니며 수정·재바인딩하지
않는다.

PR 12의 Qwen2 호환성 artifact는
`qwen2.5-0.5b-instruct-bf16.json`이다. 고정 revision
`7ae557604adf67be50417f59c2c2f167def9a775`의
`Qwen/Qwen2.5-0.5B-Instruct`를 primary RTX 4090 원격 환경에서
Transformers eager/BF16으로 실행한 결과다. 영어·한국어·Rust 코드 3개 chat
case에 대해 rendered chat, prompt token IDs, masking 전 raw last logits의 top-10과
probe 값, addressable tokenizer domain으로 masking한 cache on/off 8-token greedy
parity를 기록한다. 파일 SHA-256은
`42cc7f3fd04098bc4d70836ee9d18dbf919f158a010da3da6fdaa3d9deeceab7`이다.
생성과 standard-library-only 정적 검증은
`riley_reference.qwen2_compat` 모듈을 사용하며, 모델/CUDA 생성 명령은
반드시 `server-4096`에서만 실행한다.

생성과 검증 명령은 `tools/python/reference/README.md`를 따른다.
