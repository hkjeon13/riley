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

## Pattern 점 연산자의 줄바꿈 처리

`^.$`가 JSON 이스케이프로 전달된 줄바꿈을 한 글자로 받아들이던 결함을 수정했다. 점 연산자는 LF, CR, U+2028, U+2029를 거부한다. 일반 문자열과 unanchored 패턴에서 매치 앞뒤의 문자열은 기존 JSON 문자열 규칙을 사용한다. 응답 스키마와 강제 도구 호출이 같은 제약을 적용하는지 CPU에서 확인했다. [API 계약](../api.md#structured-output) · [ECMAScript Atom](https://tc39.es/ecma262/multipage/text-processing.html#sec-compileatom) · [줄바꿈 코드 포인트](https://tc39.es/ecma262/multipage/ecmascript-language-lexical-grammar.html#sec-line-terminators)

Unicode 스칼라 1,112,064개를 각각 JSON 직렬화와 UTF-16 이스케이프로 검사했다. 표현 2,224,128건 중 네 줄바꿈 코드 포인트에 해당하는 8건만 거부됐다. 서버 라이브러리 테스트 105건이 통과했다. 테스트에는 두 가지 요청 경로와 짧은 이스케이프, 대소문자 16진수, UTF-8, 서로게이트 쌍이 포함된다. [컴파일러 회귀 검사](../../crates/riley-server/src/structured.rs) · [요청 경로 회귀 검사](../../crates/riley-server/src/chat.rs)

`9373ea412`의 CUDA 12.4 배포 이미지를 A100에서 재빌드해 GitHub 빌드와 런타임 파일 21개의 해시를 대조했다. GPU 6·7번에서 Qwen2.5-7B-Instruct의 API 40건씩 통과했다. GPU 6번의 장문 스키마·지정 도구는 C1 16건을 기준으로 C8 16건과 C32 32건이 모두 일치했다. GPU 7번의 45초 혼합 부하는 정상 완료 338건, 의도적 연결 종료 21건을 기록했고 캐시·배치 메트릭 및 종료 검사를 통과했다. 두 GPU에서 종료 자원을 회수하고 기존 20GiB 예약을 확인했다. [배포 이미지 CI](https://github.com/hkjeon13/riley/actions/runs/45126944223) · [A100 검산](../../benchmarks/evidence/20261123-w4-dot-package/verified.json)

## Pattern 문자 클래스의 JSON 인코딩

기존 `^[!-#]$` 문법은 따옴표가 범위 안에 있을 때 `"""`를 완성된 출력으로 받아들였다. `^[A-z]$`에서는 범위에 포함된 역슬래시가 원시 바이트로 들어갔고, `[\n]`과 `[\u0061]`처럼 지원하지 않는 클래스 이스케이프는 일반 문자로 해석됐다. JavaScript `RegExp`의 Unicode 모드에서 클래스가 매치하는 값을 확인하고, JSON의 따옴표·역슬래시 이스케이프와 대조했다.

지원 범위는 디코딩한 문자에 적용한다. 범위에 든 따옴표·역슬래시는 올바른 JSON 이스케이프로 생성한다. 지원하지 않는 제어·Unicode 패턴 이스케이프와 내림차순 범위·클래스 집합이 들어간 범위 끝점은 이름이 있는 오류로 거부한다. `\s`는 API의 단일 공백 범위를 유지한다. 부정 클래스의 `[^\s]`는 ECMAScript 공백 전체를 제외해야 하므로 거부한다. 응답 스키마와 지정 도구 인자 경로의 오류를 포함해 CPU 서버 테스트 107건을 통과했다. [API 계약](../api.md#structured-output) · [문법 회귀 검사](../../crates/riley-server/src/structured.rs) · [요청 경로 회귀 검사](../../crates/riley-server/src/chat.rs)

`6d618f64e`의 CUDA 12.4 이미지는 GitHub와 A100 빌드에서 런타임 파일 21개가 일치했다. A100의 새 HTTP 검사 10건에서 응답 스키마와 지정 도구가 따옴표·역슬래시를 유효한 JSON으로 반환한 4건과, 지원하지 않는 패턴에 `400 invalid_schema`를 반환한 6건을 확인했다. GPU 6·7번에서 기본 API를 각각 40건씩 검증했다. GPU 6번의 장문 스키마·도구 호출 64건에서 C8·C32 결과는 C1과 일치했다. GPU 7번의 45초 혼합 부하는 정상 완료 338건, 의도적 연결 종료 21건이었다. 양쪽 종료 자원은 모두 회수됐고 각 GPU의 기존 20GiB 예약이 복귀했다. [A100 검산](../../benchmarks/evidence/20261123-w4-pattern-classes/verified.json) · [CUDA 이미지 CI](https://github.com/hkjeon13/riley/actions/runs/45148470780)

## 중첩 스키마 ID와 로컬 참조

문서 최상위와 하위 `$id`에 같은 이름의 정의를 각각 1과 2로 지정했다. Draft 2020-12 독립 검증기 `jsonschema 4.23.0`은 하위 스키마의 `$ref`를 해당 리소스에서 해석해 `{"id":2}`만 허용했다. 수정 전 컴파일러는 최상위 정의를 읽었고 이 스키마를 그대로 받아들였다. [JSON Schema 참조 규칙](https://json-schema.org/draft/2020-12/json-schema-core#section-8.2)

컴파일하는 하위 스키마에 `$id`가 있거나 로컬 `$ref` 포인터가 그런 리소스를 가로지르면 `invalid_schema`로 거부한다. 최상위 `$id`, `#` 재귀, `$id`라는 속성·정의 이름을 사용하는 같은 문서 내 참조는 회귀 검사에서 통과했다. 응답 스키마와 지정 도구 요청이 각각 해당 요청 필드의 스키마 오류를 반환했고 전체 CPU 서버 테스트는 109건 통과했다. [스키마 회귀 검사](../../crates/riley-server/src/structured.rs) · [응답·도구 진입점 검사](../../crates/riley-server/src/chat.rs)

`c5ec7b391`의 CUDA 12.4 배포 이미지를 A100에서 재빌드했다. GitHub와 A100의 런타임 파일 21개는 경로와 SHA-256이 일치했다. Qwen2.5-7B-Instruct의 HTTP 요청 6건에서 응답 스키마와 지정 도구 모두 중첩 `$id` 및 리소스를 가로지르는 포인터에 `400 invalid_schema`를 반환했고, 문서 최상위 `$id`와 내부 포인터는 `200`과 `{"$id":3}`을 반환했다. 기존 문자 클래스 요청 10건, API 검사 40건씩, 장문 스키마·도구 64건과 45초 혼합 부하를 함께 검증했다. GPU 6·7번의 기존 20GiB 예약을 확인했다. 재현할 때 `schema.patch.gz`를 `schema.patch`로 풀어 `dispatch.sha256`으로 확인한다. [HTTP 및 이미지 증거](../../benchmarks/evidence/20261123-w4-schema-ids/verified.json) · [GitHub 배포 이미지 CI](https://github.com/hkjeon13/riley/actions/runs/45177901703)

## 로컬 참조의 URI 인코딩

로컬 `$ref`의 URI fragment를 UTF-8로 디코딩한 뒤 JSON Pointer를 해석한다. 예를 들어 `#/$defs/part%20name`은 공백이 든 `part name` 정의를 가리킨다. 수정 전 컴파일러는 이 사례에서 `part%20name`이라는 별도 정의를 선택했다. 잘못된 percent escape와 유효하지 않은 UTF-8은 `invalid_schema`로 거부하고, 디코딩한 포인터에도 기존 하위 `$id` 검사를 적용한다. [URI fragment 규칙](https://www.rfc-editor.org/rfc/rfc6901.html#section-6) · [컴파일러](../../crates/riley-server/src/structured.rs)

응답 스키마와 이름을 지정한 도구 호출 양쪽의 요청 정규화와 생성 문법을 확인했다. 독립 Draft 2020-12 검증기 `jsonschema 4.23.0`과 Riley가 같은 fixture 15개, 인스턴스 32개를 통과했다. CPU 서버 라이브러리 검사는 111건 통과, 기존 비활성 검사 1건이었다. [응답·도구 경로 회귀 검사](../../crates/riley-server/src/chat.rs) · [공통 fixture](../../crates/riley-server/tests/fixtures/structured_schema_oracle.json)

`8a7c618c1`의 CUDA 12.4 배포 이미지는 GitHub와 A100의 Riley 바이너리를 포함한 런타임 파일 21개의 SHA-256이 일치했다. Qwen2.5-7B-Instruct에서 로컬 참조 HTTP 16건을 두 경로로 확인했다. 조건을 충족한 6건은 공백·Unicode·JSON Pointer 이스케이프를 올바르게 해석한 값을 포함해 HTTP 200을 반환했다. 10건은 인코딩 오류와 하위 `$id` 참조에 해당하며, 각 요청 필드에서 HTTP 400 `invalid_schema`를 반환했다. [GitHub 이미지 검증](https://github.com/hkjeon13/riley/actions/runs/45274813611) · [HTTP 원본](../../benchmarks/evidence/20261124-w4-uri-reference-fix/a100/structured/schema-references.json)

GPU 6·7번의 기본 API 검사는 각각 40건을 통과했다. 장문 스키마·도구 64건에서 C8·C32의 출력이 C1과 일치했다. 45초 혼합 부하는 정상 완료 338건과 의도적 연결 종료 21건을 기록했다. 두 서버가 모든 자원을 회수하고 각 GPU가 기존 홀더의 20GiB 예약으로 돌아왔다. 재현 시 `schema.patch.gz`를 `schema.patch`로 풀어 `dispatch.sha256`으로 확인한다. [같은 이미지의 대조와 종료](../../benchmarks/evidence/20261124-w4-uri-reference-fix/verified.json)

## Pattern 선택 분기의 시작과 끝 조건

`^cat|dog$`를 이전 컴파일러에서 평가하면 `catfish`와 `sheepdog`를 거부했다. 전체 패턴의 처음과 끝 기호를 분리한 뒤 두 분기에 함께 적용했기 때문이다. 각 최상위 분기에서 자기 시작·끝 조건을 적용하도록 수정했다. `^cat$|dog|^eel$`처럼 가운데 분기만 부분 문자열과 일치하는 경우와, 그룹·이스케이프된 `$`·문자 클래스 속 `|`도 대조했다. [JSON Schema의 pattern 규칙](https://json-schema.org/draft/2020-12/json-schema-validation#name-pattern) · [ECMAScript의 분기 규칙](https://tc39.es/ecma262/multipage/text-processing.html#sec-compile-subpattern)

추가한 13개 입력은 Node.js `RegExp`의 Unicode 모드와 일치했다. 독립 검증기 `jsonschema 4.23.0`은 공유 fixture 18개, 인스턴스 45개의 판정과 일치했다. 응답 스키마와 이름을 지정한 도구의 생성 문법을 포함한 CPU 서버 테스트는 111건 통과, 기존 비활성 검사 1건이었다. [공유 fixture](../../crates/riley-server/tests/fixtures/structured_schema_oracle.json) · [응답·도구 회귀 검사](../../crates/riley-server/src/chat.rs)

`2b58655f3`의 CUDA 12.4 이미지를 GitHub와 A100에서 빌드해 Riley 바이너리를 포함한 런타임 파일 21개가 일치함을 확인했다. Qwen2.5-7B-Instruct의 정규식 HTTP 검사 10건에서 두 요청 경로의 유효한 8건은 지정된 값을 반환했고, 분기 중간에 `$`를 넣은 2건은 `400 invalid_schema`를 반환했다. 과거 문자 클래스·로컬 참조 검사도 다시 통과했다. [GitHub 이미지 검사](https://github.com/hkjeon13/riley/actions/runs/45307835331) · [정규식 HTTP 기록](../../benchmarks/evidence/20261124-w4-pattern-anchors/a100/structured/pattern-anchors.json)

A100 6·7번의 API 검사는 각각 40건을 통과했다. 장문 스키마·도구 64건에서 C8·C32가 C1과 일치했다. 45초 혼합 부하는 정상 완료 338건과 의도적 연결 종료 21건을 기록했고, 두 GPU는 종료 자원을 모두 회수하고 기존 20GiB 예약으로 돌아왔다. [패키지 검산](../../benchmarks/evidence/20261124-w4-pattern-anchors/verified.json)

## 로컬 참조의 JSON Pointer 이스케이프 검증

JSON Pointer에서 `~0`과 `~1`은 각각 물결표와 슬래시를 나타낸다. 기존 컴파일러는 같은 이름의 필드가 있으면 `~2`처럼 유효하지 않은 이스케이프도 참조로 받아들였다. URI 프래그먼트를 디코딩한 다음 포인터를 검사해 이 경우를 거부하도록 수정했다. `~01`이 필드 이름의 `~1`을 가리키는 동작은 유지했다. [RFC 6901 문법과 평가](https://www.rfc-editor.org/rfc/rfc6901.html#section-3)

응답 스키마와 지정 도구 양쪽에서 잘못된 이스케이프를 거부하는 CPU 회귀 검사를 추가했다. 서버 라이브러리 전체 테스트에서 113건이 통과했고 기존 비활성 검사 1건이 남았다. [참조 컴파일러](../../crates/riley-server/src/structured.rs) · [응답·도구 정규화 검사](../../crates/riley-server/src/chat.rs)

`b7ca110fc`의 GitHub와 A100 CUDA 12.4 이미지에서 런타임 파일 21개의 SHA-256이 일치했다. Qwen2.5-7B-Instruct의 실제 HTTP 검사에서 응답 스키마와 지정 도구는 유효한 참조 두 건에 정해진 값을 반환했고, 잘못된 이스케이프 여섯 건에 `400 invalid_schema`를 반환했다. A100 6·7번의 API는 각각 40건을 통과했고 장문 스키마·도구 64건은 동시성 1·8·32에서 대응 출력이 같았다. 45초 혼합 부하에서 338건을 정상 완료하고 21건의 의도적 연결 종료를 처리했다. 종료 뒤 자원과 A100 예약이 복구됐다. [실제 HTTP 결과](../../benchmarks/evidence/20261124-w4-pointer-syntax/a100/structured/pointer-syntax.json) · [패키지 검산](../../benchmarks/evidence/20261124-w4-pointer-syntax/verified.json) · [GitHub 이미지 검사](https://github.com/hkjeon13/riley/actions/runs/45329963969)

## 현행 배포 이미지의 장문 채팅 검증

Qwen2.5-7B-Instruct의 고정 입력에서 255·1,024·7,836·7,917토큰 사례를 선택했다. 각 입력을 채팅 메시지로 보내 `type`과 `enum`의 교집합 및 정수 `const`를 포함한 JSON 스키마를 검사했다. 같은 스키마를 이름이 지정된 도구의 인자로도 적용했다. 서버가 붙인 채팅 템플릿까지 포함한 입력 ID와 생성 ID·텍스트·도구 인자·종료 사유·사용량을 대조했다. 스키마와 도구 인자는 독립 Draft 2020-12 검증기로 검사했다. [고정 입력](../../benchmarks/evidence/20261119-w4-holdout/holdout.json.gz) · [검사기](../../benchmarks/live/structured_holdout_check.py)

가장 긴 도구 요청의 렌더링 입력은 8,116토큰이었다. 출력 상한 64토큰은 정상 완료했고, 같은 요청에 96토큰을 지정하면 HTTP 400 `invalid_request`를 반환했다. 요청 예산 합계는 8,212토큰이다. 스케줄러의 최대 KV 길이는 마지막 생성 토큰을 제외한 8,211토큰으로, 설정 상한 8,192를 초과했다. [스케줄러의 길이 계산](../../crates/riley-scheduler/src/scheduler.rs)

2026-11-21 KST에 `aws-a100`의 A100 6번에서 기준 이미지 `b731f48cc6ec`와 jemalloc 파생 이미지 `c3fae872c532`를 실행했다. 같은 코드 리비전 `36aa1880`, 실행 파일, 체크포인트를 확인했다. 각 이미지의 단독 요청 16건에서 스트리밍과 비스트리밍 여덟 쌍이 일치했다. C8 16건과 C32 32건은 해당 단독 응답과 같았다. 두 이미지에서 관측한 최대 활성 요청은 C8에서 8건, C32에서 31건이었다. 이미지 간 단독 응답 해시도 16/16건 일치했다. [검산 결과와 원본 보고서 해시](../../benchmarks/evidence/20261121-w4-structured-holdout/verified-holdout.json)

각 이미지의 일반 API 40건도 통과했다. 종료 코드 0과 활성·대기 요청, KV, device·pinned 할당 0을 확인했다. 각 실행 전후 A100 6·7번의 동일 홀더와 20GiB 예약을 검증했다. 장문 결과의 적용 범위는 이 체크포인트의 고정 입력과 지정 도구 호출이다. [실행 순서](../../benchmarks/evidence/20261121-w4-structured-holdout/campaign.sh) · [독립 검산](../../benchmarks/evidence/20261121-w4-structured-holdout/fetch-summary.sh)
