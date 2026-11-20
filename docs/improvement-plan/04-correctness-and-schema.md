# W4. 장문 정확성과 structured output 계약 개선

작성일: 2026-11-13\
종합 계획: [의존관계·가드레일](README.md)\
구성: W4-A는 장문 수치 정확성, W4-B는 스키마 계약이다. 조사·수정·완료 여부를 각각 기록한다.

2026-11-19에 `oneOf`와 강제할 수 없는 형제 제약을 거부하고, `type`과 `enum`·`const`의 교집합을 강제했다. Draft 2020-12 독립 검증기와 공유 fixture 13개·인스턴스 28개를 대조하고 W4-B CUDA API 실행을 검증했다. W4-A는 Riley와 Hugging Face에 같은 입력 ID를 공급해 다섯 사례의 첫 분기를 재현했다. [W4-A 장문 결과](04-long-accuracy-result.md) · [W4-B 검증 결과](04-structured-output-result.md) · [현행 API 계약](../api.md)

## 1. 현재 근거와 해석

| 항목 | 확인 내용 | 해석 |
|---|---|---|
| 장문 배치 불변성 | Qwen2.5-7B C8/C32 각각 54/54 | 18개 prompt를 세 번 반복한 비교다. 54개 독립 prompt의 품질 평가가 아니다. |
| 장문 HF 일치 | 원래 비교에서 Riley 텍스트 12/18, vLLM 16/18. 후속 동일 입력 ID를 사용한 HF SDPA 비교에서 전체 생성 ID와 텍스트 각각 Riley 13/18, vLLM 16/18 | 다섯 첫 분기의 점수 간격과 두 prefill 사례의 첫 BF16 차이를 확인했다. HF의 bias 순서와 prefill 분할 대조에서 두 사례의 첫 토큰은 유지됐다. |
| `oneOf` | 초기 컴파일러가 `anyOf`처럼 처리 | 수정 후 400을 반환한다. 이중 일치 반례를 독립 검증기에 확인했다. |
| 스키마 조기 반환 | 초기 구현이 `const`·`enum`·조합 키워드 처리 뒤 일부 형제 제약을 무시 | 지원하는 `type` 교집합을 강제하고 나머지 형제 제약은 명시적으로 거부한다. |
| HF 기준 경로 | `long_check.py`는 HF BF16 `sdpa` 사용 | owned HF-eager 프로파일 기록과 섞어 같은 기준이라고 주장하면 안 된다. |

근거: [장문 정확성 기록](../serving-validation.md), [스키마 구현](../../crates/riley-server/src/structured.rs), [API 제한](../api.md), [long_check](../../benchmarks/live/long_check.py), [owned 프로파일 조건](../model-support.md).

## 2. W4-A: 장문 차이의 원인을 찾는 절차

### W4-A1. 입력과 참조 환경을 먼저 일치시킨다

모델 revision·파일 해시·tokenizer·chat template·BOS/EOS·special token·stop·generation 설정을 고정한다. Riley가 받은 prompt token ID를 HF에도 그대로 넣고, vLLM에도 실제로 같은 ID가 전달됐는지 확인한다. tokenizer 차이를 GEMM 문제로 오인하지 않는다.

CUDA·cuBLAS·PyTorch·Transformers와 attention backend를 기록한다. GPU 라이브러리 정렬은 원인 분리 실험이며 프로덕션 요구사항으로 자동 승격하지 않는다. HF exact를 주장할 때는 어떤 backend·버전·모델·문맥·출력 길이에 대한 주장인지 명시한다.

최초 재현은 기존 18개 prompt·32 output token을 보존했다. [`long_accuracy.py`](../../benchmarks/live/long_accuracy.py)는 raw ID를 기록하고 stop으로 종료된 마지막 EOS만 정규화한다. 길이가 다른 출력은 공통 prefix 길이가 같아도 별도로 판정한다.

### W4-A2. 최초 분기 token에서 teacher forcing으로 좁힌다

1. 요청별 prompt ID, Riley/HF/vLLM raw generated ID, 종료 사유, 최초 분기 위치를 저장한다.
2. 분기 전까지 동일한 token prefix를 세 엔진에 강제로 공급한다. 이미 다른 경로를 생성한 logits끼리 비교하지 않는다.
3. 첫 차이의 top-1/top-2 margin, 선택 token, finite 여부, 전체 vocabulary logits의 오차·분포 지표를 진단한다.
4. 같은 prefix에서 norm→QKV/bias→RoPE→attention→o_proj→MLP→최종 norm/LM head 순으로 중간 activation을 비교한다.
5. 처음 차이가 생기는 층·연산·prefill/decode 경계를 찾고 pinned checkpoint 실제 입력의 독립 GPU 재현으로 줄인다.

작은 margin 때문에 다른 token이 나왔다는 설명만으로 종료하지 않는다. 원인이 허용된 BF16 rounding인지, 잘못된 mask·position·bias·cache write인지 구분한다. 오차 임계값은 관측 결과에 맞춰 사후 조정하지 않는다.

### W4-A3. 원인별 대조 실험

| 가설 | 분리 실험 | 해결 방향 |
|---|---|---|
| 토큰화·BOS/EOS | 같은 prompt ID·template·종료 규칙 | tokenizer 또는 benchmark 입력 수정, 기존 결과 영향 명시 |
| KV position·mask·chunk | 길이 경계 앞뒤, cache on/off, C1/C8/C32 | cache 주소·row kind·position/mask 수정 후 정확성 재검증 |
| norm·RoPE·bias | 동일 activation, library 정렬·프로파일별 비교 | 잘못된 선택·epsilon·bias epilogue 교정 |
| GEMM·attention reduction | 같은 입력·같은 알고리즘·chunk, 강제 prefix | 수치 계약 차이인지 구현 결함인지 분리 |
| 참조 경로 차이 | HF sdpa/eager를 별개 baseline으로 기록 | 기준을 고정하고 주장 범위 제한 |

기존 `owned_teacher_logits_gpu` 진단과 component oracle을 재사용할 수 있는지 [model-support](../model-support.md)에서 확인한다. 특정 pinned geometry를 지원하는 진단을 임의 모델에 그대로 적용하지 않는다.

### W4-A4. 수정과 재평가

결함이면 최소 재현을 회귀 검사로 남기고 affected 모델·프로파일·경계 길이를 모두 재평가한다. 수치 구조 차이면 선택지를 분리한다.

- 기존 바이트를 보존한 구현 수정: W2·W3의 공통 정확성 기준에 적용한다.
- 출력이 달라지는 새 수치 기준: 별도 변경으로 검토하고 기존 사전 승인·전체 affected 모델 품질 검증을 거친다.
- 원인은 확인됐지만 채택할 수정이 없음: HF 일치 범위·한계·미해결 이유를 공개하고 통과로 바꾸지 않는다.

기존 18개 표본에만 맞추지 않도록 고정된 추가 holdout을 둔다. 다국어·코드·숫자·반복 토큰·schema/tool·긴 출력과 256/512/1024 경계 앞뒤를 포함한다. prompt+output은 실제 지원 문맥 상한 안으로 제한한다. GSM8K는 같은 1,319문항·4-shot·generation 설정을 사용하며 축약 smoke로 최종 품질을 대신하지 않는다.

Qwen2.5-7B의 검증된 tokenizer로 별도 greedy holdout 11개를 고정했다. 입력은 255/256/257, 511/512/513, 1,023/1,024/1,025, 7,836, 7,917토큰이다. 한국어·일본어·코드·숫자·반복 작업을 포함하며, 출력 상한은 64·128·256토큰이다. 입력과 출력 상한의 합은 최대 8,173토큰이다. 도구는 Riley가 반환하는 입력 ID와 Hugging Face tokenizer가 만드는 ID를 각각 fixture에 맞춰 검사한다. 같은 raw ID를 vLLM에도 보내고 응답에서 입력 ID를 대조한다. Riley는 C1을 기준으로 C8·C32에서 각 입력을 세 번 반복한다. [고정 입력과 토큰 ID (gzip)](../../benchmarks/evidence/20261119-w4-holdout/holdout.json.gz) · [GPU 실행 명령](../../benchmarks/evidence/20261119-w4-holdout/tools/run_accuracy.py)

이번 fixture는 greedy 생성과 배치에 따른 출력 차이를 확인한다. 실행은 W5-B의 두 혼합 부하 조건이 종료된 뒤 같은 GPU 7에서 순서대로 시작한다. 장문 schema/tool과 GSM8K 전수 비교는 별도의 실행 항목이다. [GPU 예약 및 실행 순서](../../benchmarks/evidence/20261119-w4-holdout/campaign.sh)

GSM8K는 기존 train 7,473행과 test 1,319행의 파일 해시를 고정했다. 기존 세 모델인 Qwen2.5-7B, Llama-3.2-3B, Qwen3-4B와 두 KV-head 모델 Qwen2.5-3B를 Riley·vLLM에서 전수 비교한다. 두 엔진에서 4-shot, greedy, 최대 256토큰, `Question:` 종료 문자열, 문맥 4,096과 클라이언트 동시성 32를 사용한다. 요청 본문과 서버가 받은 토큰 ID를 모두 대조하고, 응답 오류가 하나라도 있으면 점수 비교는 무효다. [고정 데이터](../../benchmarks/evidence/20261119-w4-gsm8k/dataset.json) · [비교 검사기](../../benchmarks/evidence/20261119-w4-gsm8k/tools/compare_gsm8k.py)

CPU에서 과거 세 Riley 결과의 1,319문항별 정오를 각각 재계산해 원래 결과와 일치함을 확인했다. 현재 네이티브 GPU 후보는 네 모델 모두에서 vLLM과 1,319개 동일 입력으로 검증해 −1.2%p 품질 기준을 통과했다. [과거 결과 재채점](../../benchmarks/evidence/20261119-w4-gsm8k/preflight.json) · [GPU 실행 순서](../../benchmarks/evidence/20261119-w4-gsm8k/campaign.sh) · [전수 비교 결과](04-long-accuracy-result.md#gsm8k-전수-품질-대조)

## 3. W4-B: 스키마 의미를 일치시키는 절차

### W4-B1. 계약표와 최소 반례 작성

각 keyword를 `완전 강제`, `제한된 형태만 강제`, `명시적 거부`, `annotation`으로 표기한다. `schema()`의 분기 순서와 typed keyword 검사를 연결하고 형제 제약도 포함한다.

대표 반례는 다음과 같다. 값 `1`은 두 분기를 모두 만족하므로 `oneOf`에는 부적합하고 `anyOf`에는 적합하다. `-1`은 첫 분기만 만족한다.

```json
{
  "oneOf": [
    {"type": "integer"},
    {"type": "integer", "minimum": 0}
  ]
}
```

추가로 `const`/`enum`과 범위·길이·type·미지원 keyword의 조합, `anyOf`와 형제 제약, 중복 `oneOf` branch, 중첩·재귀 `$ref`, optional property, unicode·escape·빈 배열을 검사한다. 조기 반환 위험은 이 재현 전까지 확정된 HTTP 오출력으로 보고하지 않는다.

### W4-B2. 단기 수정은 보장하지 못하는 입력을 거부

일반 `oneOf`의 정확한 의미를 구현하기 전에는 이를 명시적 400 `invalid_schema`로 거부하는 방향을 우선 검토한다. 정적 분석으로 분기 배타성을 증명하는 부분 지원은 후속 후보이며 얕은 휴리스틱으로 판정하지 않는다.

기존 문서가 `oneOf` 제한을 알렸더라도, 지원 입력을 거부하게 바꾸면 API 호환성 변경이다. 영향 요청·오류 메시지·이행 방법을 기록한다. 사용자가 합집합을 원할 때에만 `anyOf`로 바꾸도록 안내하며 의미가 같은 대체라고 자동 치환하지 않는다.

다른 keyword도 "모두 강제하거나 명시적으로 거부" 원칙으로 정리한다. 성공 HTTP 응답 뒤에 독립 validator를 붙이는 것만으로 token 단위 보장을 대체하지 않는다. 특히 streaming에서 이미 발행한 바이트는 회수할 수 없다.

### W4-B3. 독립 판정으로 검증

지원 JSON Schema draft·validator 버전을 테스트 환경에 고정한다. validator는 개발·검증 의존성으로 두고 Rust serving에 Python을 추가하지 않는다. 지원 subset과 선택 draft의 의미가 맞는지도 명시한다.

검사 층은 다음처럼 분리한다.

1. CPU compiler/grammar: 유효한 값은 받아들이고 무효한 값은 거부하는지 독립 validator와 비교한다. 현재 compiler와 동일한 로직을 복제한 oracle은 쓰지 않는다.
2. 오류 계약: 미지원 keyword·조합은 구체적 이름과 400을 반환하며 요청을 부분 수락하지 않는다.
3. 실제 GPU API: nonstream/stream, `tool_choice` required·forced, logprobs 혼합에서 완성된 output을 독립 검증한다.
4. 종료 경계: `finish_reason=stop` 성공 출력은 schema-valid여야 한다. 토큰 예산으로 끝난 `length`는 불완전할 수 있으며 성공 보장과 구분한다. 부분 tool call 처리도 확인한다.

유효한 문법이 있다는 사실과 모델이 토큰 제한 안에 완성한다는 사실을 분리한다. 모델이 tool을 반복하는 현상과 서버 parser 결함도 같은 rendered prompt의 HF 비교로 구분한다.

## 4. 의미 있는 회귀 검사

| 트랙 | 최소 검사 | 확대 조건 |
|---|---|---|
| W4-A 입력 | BOS/EOS·template·prompt ID·종료 경계 | tokenizer·generation 입력 변경 시 전체 관련 모델 |
| W4-A 수치 | 최초 분기 연산·실제 activation·경계 길이 | kernel·chunk·reduction 변경 시 affected geometry 전부 |
| W4-A 배치 | C1 대비 C8/C32, 혼합 stage·취소·cache | W2·W3 변경 시 필수 |
| W4-B 의미 | overlapping oneOf·형제 제약·미지원 조합 | compiler 변경 시 지원 subset 전체 |
| W4-B API | API 기본 검사와 독립 schema/tool 검사 | 실제 serving 경로 변경 시 GPU 재실행 |

새 테스트는 의도적으로 잘못된 후보가 실패하는지 확인한다. 기존 known limitation을 무조건 통과 처리하는 snapshot으로 고정하지 않는다.

## 5. 실행 시 주의할 실제 도구 동작

- `benchmarks/live/long_check.py ADDRESS MODEL_ID MODEL_DIR OUT [SERVER_PID]`는 HF 모델을 GPU에 올린다. 선택적 PID를 전달하면 서버에 SIGTERM을 보낸다. 전용 실험 서버만 대상으로 하며 공유 서비스 PID를 전달하지 않는다. 한 GPU에 양쪽 모델이 안 맞으면 Riley 결과를 보존한 뒤 전용 서버를 정리하고 HF를 순차 실행하는 형태로 분리한다.
- `benchmarks/live/api_check.py ADDRESS MODEL --tools --out FILE`은 실제 API를 호출한다. schema 의미 검증은 기존 40개 검사 외에 별도로 추가한다.
- `benchmarks/live/gsm8k.py ADDRESS MODEL DATA_DIR OUT`은 로컬 `train.jsonl`·`test.jsonl`을 기대한다. 입력 파일 해시·문항 수·shot 구성을 근거에 남긴다.

## 6. 완료 조건과 다른 작업의 사용법

- [x] W4-A: 같은 입력에서 최초 분기 토큰과 첫 BF16 차이를 재현했고 남은 불확실성을 기록했다. [장문 정확성 결과](04-long-accuracy-result.md#첫-prefill-분기-추적)
- [x] W4-A: 두 고정 decode 사례에서 첫 레이어 QKV 행을 cuBLASLt 대체 계획과 HF 개별 투영의 값으로 각각 교체했다. 최종 로짓이 달라졌고 두 사례 모두 분기 위치의 선택 ID와 전체 생성 결과가 유지됐다. [행 교체와 독립 검산](04-long-accuracy-result.md#대체-qkv-행이-최종-decode에-미친-영향)
- [x] W4-A: 같은 두 decode 위치에서 28개 레이어의 residual 경계를 하나씩 교체해 전체 남은 순전파를 실행했다. 두 사례에서 Riley 선택을 재현한 첫 경계는 각각 레이어 0의 MLP 이후와 attention 직후였다. [경계별 토큰 대조](04-long-accuracy-result.md#residual-경계-교체로-decode-토큰-선택을-재현)
- [x] W4-A: 레이어 0의 동일 피연산자 대조에서 사례 12의 게이트 곱·down projection과 `numbers-1023`의 attention 출력 투영 차이를 확인했다. 두 사례에서 출력 행을 교체해 Riley ID가 선택되는 조건을 검증했다. [같은 피연산자의 연산별 대조](04-long-accuracy-result.md#같은-입력으로-레이어-0의-mlp와-attention-출력-투영-대조)
- [x] W4-A: FP64 내적과 BF16 최근접 기준으로 두 사례의 투영을 검산했다. HF down의 축소 정밀도 reduction 설정을 끄면 최근접 기준과 다른 값이 두 행에서 각각 4개와 3개로 줄었다. [독립 투영 정밀도 대조](04-long-accuracy-result.md#투영의-bf16-반올림과-축소-정밀도-대조)
- [x] W4-A: 실제 Riley down 계획의 Split-K 3과 BF16 출력형 reduction을 재현했다. 동일한 계획에서 부분합만 FP32 계산형으로 바꾸자 BF16 최근접 기준과 다른 값이 1,248→15개, 1,263→21개로 줄었다. [Riley 계획과 reduction 대조](04-long-accuracy-result.md#실제-riley-계획과-split-k-부분합-정밀도)
- [ ] W4-A: 결함 수정 또는 수치 기준 변경의 영향 모델·품질 결과가 명확하다. 모든 HF 문자열을 무조건 같게 만들었다고 과장하지 않는다.
- [x] W4-B: 지원 제약의 독립 판정, 미지원 응답, `oneOf`·형제 제약 반례를 검증했다. [구조화 출력 계약 결과](04-structured-output-result.md)
- [ ] 공통: 배치 불변성·기존 품질 범위·W2/W3 성능 회귀 제한을 함께 만족한다.

W1은 이 검사를 필수 후보 판정에 연결한다. W2·W3는 같은 corpus·raw token·oracle을 사용하고 수치 변경 여부를 선언한다. W5는 API 지원표·known limitation·모델별 품질 주장을 갱신한다. 스키마 수정의 회수 시에도 약한 의미를 숨겨 성공시키지 말고 이전 계약·문서·버전을 함께 복구한다.
