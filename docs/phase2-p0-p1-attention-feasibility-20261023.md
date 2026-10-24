# Phase 2 P0/P1: attention 커널 하네스와 FlashInfer 타당성

2026-10-23 KST, aws-a100 GPU7(holder 유지). 계획은 [plan-fixed-split-varlen-attention-20261023.md](plan-fixed-split-varlen-attention-20261023.md)를 따른다. 진단용 측정이며 서버 성능 결과가 아니다.
근거: [`benchmarks/evidence/20261023-phase2/`](../benchmarks/evidence/20261023-phase2/).

## 측정 조건

- Qwen3-4B 형상: query head 32, KV head 8, head 크기 128, 16토큰 페이지.
- 두 가지 step 형상을 natural 워크로드에서 정상 상태 context 길이로 만들었다.
  - **decode32:** decode 행 32개(context 17–525토큰).
  - **mixed:** 224토큰 prompt 청크(위치 0–223) + decode 행 31개.
- 레이어 1회 실행당 시간이다(워밍업 20회, 측정 200회).

### 하네스

- **Riley:** `crates/riley-cuda/tests/attention_bench_gpu.rs` → `riley-bench.log`
- **FlashInfer 0.6.16 / vLLM FA2:** Python으로 측정 → `flashinfer-probe*.json`
  - 같은 형상으로 만든다. KV는 무작위 BF16, 물리 페이지는 섞어서 배치한다.

GPU6에는 승인되지 않은 외부 프로세스(PID 903988)가 있어 holder 컨트롤러가 실행을 거부한다. 모든 측정은 GPU7에서 했다.

## P0: 시간 (µs / 레이어 1회)

| 경로 | decode32 | mixed |
|---|---:|---:|
| Riley 현재 (페이지 상태 + prompt segment + merge) | 59.4 | 183.9 |
| Riley direct prompt (13절) | 57.9 | 177.6 |
| Riley 하이브리드 12페이지 (14절) | 63.4 | 231.4 |
| FlashInfer 기본 휴리스틱 | 39.1 | 45.4 |
| **FlashInfer, query 타일 64 + split 256 고정** | **38.0** | **45.3** |
| FlashInfer, query 타일 16 고정 | 37.7 | 51.1 |
| FlashInfer, query 타일 128 고정 | 53.6 | 59.5 |
| vLLM FA2 varlen (`vllm_flash_attn`, 참고) | 105.6 | 90.6 |

- FA2 측정은 split-KV 설정 없이 varlen 함수를 직접 호출한 값이다.
- vLLM 서빙의 decode-only step은 그래프 안의 split-KV 경로를 쓰므로 이 숫자와 다르다.
- vLLM 서빙 트레이스에서 mixed step attention은 레이어당 약 93µs였다(Phase 1 §11). 위 FA2 mixed 값과 맞는다.

## P1: 배치 불변성 (요청별 출력 바이트가 달라진 요청 수)

기준 배치는 mixed 32개 요청이다. 같은 요청을 아래 네 가지 방식으로 다시 실행해 바이트를 비교했다.
- **순서 반전:** 요청 순서를 거꾸로
- **decode만:** decode 행 31개만
- **절반:** 앞 12개 요청만
- **1개씩:** 요청을 하나씩 따로

| 설정 | 순서 반전 | decode만 | 절반 | 1개씩 |
|---|---:|---:|---:|---:|
| 기본 (타일·split 자동) | 0 | 28/31 | 9/12 | 7/8 |
| split만 고정 (128/256/512) | 0 | 28/31 | 9/12 | 7/8 |
| 타일 16 고정, split 자동 | 0 | 0 | 0 | 5/8 |
| 타일 64 고정, split 자동 | 0 | 0 | 2/12 | 5/8 |
| 타일 128 고정, split 자동 | 0 | 0 | 4/12 | 6/8 |
| **타일 16/64/128 고정 + split 256 고정** | **0** | **0** | **0** | **0** |

### 원인

FlashInfer는 query 타일 크기(`cta_tile_q`)를 배치의 평균·최대 query 길이로 고른다. 그런데 타일 크기는 KV 리듀션 구조도 함께 바꾼다.
- 타일 16이면 warp 4개가 KV를 나눠 처리한 뒤 CTA 안에서 합친다.
- 타일 64/128이면 warp 하나가 KV를 순차로 처리한다.

여기에 split 수까지 배치에 따라 바뀐다. 둘 다 고정해야 행의 리듀션 순서가 그 행 자신의 context로만 정해진다.

### 확인 방법

FlashInfer 패키지를 복사해(`fi-patched`, venv 원본과 vLLM 기준선은 건드리지 않음) `FA2DetermineCtaTileQ`가 환경 변수 `RILEY_FI_FORCE_CTA_TILE_Q`를 따르도록 패치했다. 패치 내용은 `flashinfer-utils-patch.txt`에 있다.

## 판정과 다음 단계

- **P0 기준 통과.** 기준은 attention −30% 이상이다. 타일 64 + split 256 고정 시 decode −36%, mixed −75%다. vLLM FA2보다도 빠르다.
- **P1 결정: A안(FlashInfer 커널 바인딩)으로 간다.** 고정 설정은 query 타일 64, split 256토큰이다.
  - CUDA 12.4 nvcc로 FlashInfer 헤더가 컴파일되는 것을 확인했다. FlashInfer에 포함된 CCCL 헤더를 쓴다.
- **서버 수준 기대 효과 (거친 추정):** natural C32에서 attention이 커널 합의 약 20%(527ms)다. mixed와 decode에서 각각 −75%, −36%면 attention이 약 200ms로 준다. span은 약 −10~12% 줄고, 처리량은 약 +10%로 vLLM 대비 약 1.0배 이상이 된다. 서버 측정 전의 추정이다.

### P2 설계 요점 (다음 작업)

1. **템플릿 인스턴스화:** FlashInfer 헤더(Apache-2.0)를 고정 버전으로 가져와 `BatchPrefillWithPagedKVCacheDispatched<CTA_TILE_Q=64, 128, 128, kNone, false, kCausal, DefaultAttention, PagedParams>` 하나만 인스턴스화한다.
2. **KV 레이아웃 연결:** Riley의 페이지 레이아웃(페이지 × KV head × 16토큰 × 128)을 FlashInfer HND paged KV로 연결한다.
3. **요청 구성:** 행을 요청 단위로 묶는다. prompt run은 같은 시퀀스의 연속 위치 행들이고, decode 행은 하나가 요청 하나다. 각 요청의 `qo_indptr`, `kv_indptr`, 페이지 목록, 마지막 페이지 길이를 만든다.
4. **plan을 GPU에서 만든다.** 타일과 split이 고정이므로 작업 목록이 (요청별 query 길이, kv 길이)의 단순 함수다. 작은 디바이스 커널로 Riley의 디바이스 메타데이터에서 작업 목록을 만들고, 최대 크기 그리드와 `block_valid_mask`로 실행하면 step graph 캡처와 호환된다. host plan과 업로드가 필요 없다.
5. **새 numerics mode로 다룬다.** 정확성(G1~G3)과 qualification(G4~G5)을 사전 등록 후 검증한다.
   - prefill 청크 분할에 대한 불변성(같은 prompt를 다른 크기의 청크로 나눔)은 이번 probe에서 다루지 않았다. P3 GPU 테스트에 반드시 넣는다.
