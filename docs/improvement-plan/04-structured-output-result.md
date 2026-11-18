# W4-B 구조화 출력 계약 검증

2026-11-19 KST에 스키마 컴파일러의 분기와 종료 조건을 수정하고 CPU 및 실제 CUDA 서빙에서 검증했다. [현행 API 계약](../api.md) · [W4 계획](04-correctness-and-schema.md)

`oneOf`는 400 `invalid_schema`를 반환한다. `anyOf`는 여러 분기를 만족할 수 있는 합집합으로 유지한다. `const`와 `enum`은 `type`을 함께 적용하며, `anyOf`·한 항목의 `allOf`·`$ref` 및 리터럴의 강제할 수 없는 형제 제약은 이름을 명시해 반환한다. 일반 문자열 생성은 유효한 유니코드 스칼라만 받아들이며, 서로게이트 쌍은 길이 한 글자로 센다.

`stop` 문자열로 JSON 값이나 강제 도구 호출이 잘리면 성공 조건을 보장할 수 없어 해당 조합은 400으로 거부한다. CUDA 서버는 문법이 완료되지 않은 상태에서 EOS로 끝나는 경우 성공 `stop` 대신 해당 요청의 실패를 전송한다. `max_tokens`가 소진된 경우의 `length`는 부분 출력으로 유지한다.

- CPU 서버 라이브러리 101건과 Python 부하 도구 13건이 통과했다. Rust 문법과 독립 `jsonschema 4.23.0` Draft 2020-12 판정기가 공유한 fixture 13개, 인스턴스 28개를 검증했다.
- `f7c956d3c`를 Linux에서 Rust 1.85.0, CUDA 12.4, SM80, `server,hf-exact-probe`로 빌드했다. 첫 빌드에서 probe 전용 shape 준비의 가변성 누락을 발견하고 수정했다. [빌드 결과와 바이너리 해시](../../benchmarks/evidence/20261119-w4-schema/binary.sha256)
- A100 GPU 7에서 Qwen2.5-3B-Instruct로 계약 검증 11건과 기본 API 검사 40건이 통과했다. 범위는 스트리밍·비스트리밍, `logprobs` 혼합, `required`·named 도구, 토큰 예산 종료, 거부 응답이다. 실제 tokenizer가 진행 불가능한 문법으로 들어가는 경우의 오류 경로는 이번 GPU 표본에서 의도적으로 만들지 않았다. [계약 결과](../../benchmarks/evidence/20261119-w4-schema/live-schema-result.json) · [API 결과](../../benchmarks/evidence/20261119-w4-schema/general-api-result.json)
- Postman 1.73.0에서 파일 인자로 보낸 첫 요청은 multipart로 전달돼 415였다. 같은 요청을 raw JSON으로 재전송하자 예상한 400과 오류 필드가 일치해 assertion 두 건이 통과했다. CLI의 기대 400 응답 종료 코드는 1이었다. [Postman 결과](../../benchmarks/evidence/20261119-w4-schema/postman-observations.json)
- 전용 서버 종료 이후 `aws-a100` GPU 6·7의 동일 홀더가 각각 20GiB 대기 예약을 유지했고 별도 compute process는 없었다. [예약 기록](../../benchmarks/evidence/20261119-w4-schema/gpu-reservation.json)

이번 검증 범위는 문서화한 스키마 부분집합과 한 모델의 GPU API다. W4-A의 HF 장문 분기 추적과 다른 모델군, 운영 soak는 이후 단계에서 확인한다.
