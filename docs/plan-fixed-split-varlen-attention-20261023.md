# 개선 계획: 배치 불변 고정 split varlen attention

작성: 2026-10-23 KST. 상태: 계획이며 구현 전이다.
전제는 [Phase 0](phase0-contract-cost-and-aa-20261022.md)과 [Phase 1](phase1-native-attention-and-burst-20261022.md)의 측정과 결정이다. native(배치 불변)가 제품 기본이고, HF-literal은 검증 모드다.

## 1. 왜 이 계획인가

### 1.1 남은 격차는 attention에 있다

Qwen3-4B, A100 GPU7, natural C32에서 같은 64요청 구간을 비교했다(Phase 1 §1, §11).

| 항목 | Riley native | vLLM 0.27.1 | 차이 |
|---|---:|---:|---:|
| GEMM | 1857 ms | 1908 ms | −51 |
| attention (decode + prompt) | 527 ms | 338 ms | **+189** |
| RMSNorm | 113 ms | 60 ms | +53 |
| step 사이 host 유휴 (profiler 없이 측정) | step당 약 0.3 ms | 거의 0 | 처리량 약 2% |

- GEMM은 동률이고, Rust host 경로도 step당 0.3 ms 수준이다. 남은 격차의 가장 큰 단일 원인은 attention이다.
- mixed step의 attention이 특히 크다. Riley는 step당 4.56 ms, vLLM은 3.34 ms다. 게다가 vLLM은 step당 prompt 토큰이 약 2배(budget 576 대 256)다.

### 1.2 구조 차이

| | vLLM (FA2) | Riley (현재) |
|---|---|---|
| mixed step | varlen 커널 **1회**: prompt 행과 decode 행을 함께 처리, 중간 상태 없이 바로 출력 | prompt MMA + decode producer + merge **3회**, FP32 부분 상태를 메모리에 썼다가 다시 읽음 |
| decode-only step | split-KV + combine (그래프 안) | 페이지 단위(16토큰) producer + merge |
| split 결정 | 배치 크기와 SM 수로 자동 결정 → **배치 불변이 아님** | 행 자신의 페이지 수로 결정 → 배치 불변 |

### 1.3 이미 확인한 것: 두 극단은 모두 실패했다

| 시도 | split 단위 | 결과 |
|---|---|---|
| 기존 decode producer | 16토큰 | 상태 쓰기와 merge 비용이 큼 |
| fused decode (§2) | 행당 순차 | −4 ~ −11% |
| segment decode (§10) | 64토큰 | C1/C8 −1.9 ~ −2.7%, natural C32 −0.1% |
| direct prompt (§13) | 나누지 않음 | fixed C32 +2.2%, natural C32 −0.5% |
| 하이브리드 (§14 예정) | 짧은 행만 나누지 않음 | qualification 진행 중 |

**결론:** 배치 불변성과 varlen의 이점은 상충하지 않는다. 충돌하는 지점은 "배치에 따라 split 수를 바꾸는 자동 선택" 하나뿐이다. 따라서 split을 **행 자신의 context 길이로만 정해지는 고정 크기**로 두고, prompt 행과 decode 행을 **한 커널 실행**에 담으면 두 가지를 함께 얻을 수 있다. 지금까지 이 중간 지점은 시도하지 않았다.

## 2. 목표와 비목표

**목표**
- native natural C32 처리량을 vLLM 이상으로 만든다. 현재 0.914배(Phase 0)다.
- fixed C1–C32의 동률 이상은 유지한다.
- mixed step과 decode-only step의 attention을 한 가지 수치 규칙으로 통일한다.
- 배치 불변성을 유지한다. 출력이 배치 구성, 순서, prefill 청크 분할, step 종류(mixed / decode-only)와 무관해야 한다.

**비목표**
- HF 바이트 일치는 목표가 아니다. strict는 검증 모드로 유지한다.
- FP8 KV, 양자화, prefix cache, 비동기 step 제출은 별도 과제다(§8).

## 3. 수치 계약 (새 numerics mode: `fixed-split-varlen-v1`)

한 행(query 토큰 하나 × query head 하나)의 출력은 아래 규칙으로만 정해진다.

1. **context:** 위치 0부터 자기 위치까지의 KV(causal).
2. **split:** context를 **S 토큰 고정 크기**로 나눈다(S는 서버 시작 시 정하는 상수, 후보 128/256/512). 조각 수는 `ceil((pos + 1) / S)`로, 행 자신의 위치로만 정해진다.
3. **조각 내부:** 16토큰 페이지를 오름차순으로 online softmax(max 갱신, 재스케일) 처리한다.
   - 점수는 BF16 Q·K MMA를 FP32로 누산한다.
   - 확률을 V와 곱하는 MMA 입력 형식(BF16 1조각인지, 기존 3조각 정확 분해인지)은 Phase 2에서 정확도와 성능을 비교해 하나로 고정한다.
4. **조각 결합:** 조각 상태 `[m, l, n]`를 조각 번호 오름차순으로, 전역 max 기준 고정 순서로 결합한다. 조각이 1개면 결합 없이 바로 출력한다.
5. **출력:** `BF16(n × (1 / l))`.
6. **같은 규칙의 적용 범위:** prompt 행과 decode 행, mixed step과 decode-only step, 그래프 캡처 여부에 관계없이 같은 규칙을 쓴다.

이 계약에서 배치에 따라 바뀌는 것은 **어느 CTA가 어떤 조각을 계산하는지(스케줄)뿐이다.** 리듀션 순서는 바뀌지 않는다.

## 4. 커널 설계

### 4.1 단일 varlen 실행

- **입력:**
  - 이어 붙인 query `[total_rows, QH, D]`
  - 행별 `(sequence slot, position)`과 block table: 기존 PackedBatch를 그대로 쓴다.
  - 행 종류: 기존 row kinds를 그대로 쓴다.
- **작업 단위:** (query tile, KV head, split).
  - query tile은 같은 시퀀스의 연속 위치 최대 16행 또는 64행이다. prompt run은 여러 행, decode 행은 1행이다.
  - 작업 목록은 host가 만든다. 그래프 캡처 시에는 최대 개수로 고정하고, 실제 개수는 디바이스에서 읽는다(현재 dynamic counts 방식).
- **조각이 1개인 행:** 바로 출력한다. 짧은 prompt와 짧은 decode가 여기에 해당하며, direct 모드의 이점을 그대로 얻는다.
- **조각이 2개 이상인 행:** 조각 상태를 작은 workspace에 쓴다. 크기는 `rows × QH × ceil(T/S) × (D + 2) × 4 B`로, 현재 페이지 단위 workspace의 S/16분의 1이다.
  - 결합은 같은 실행 안의 마지막 CTA(원자 카운터)가 하거나, 별도의 작은 결합 커널이 한다. 두 방식 모두 결합 순서는 고정한다.

### 4.2 GQA 처리

KV head 하나를 공유하는 query head(Qwen3-4B는 4개)를 한 CTA에서 함께 처리해 K/V 읽기를 공유한다. 기존 tensor-core producer의 MMA 열 배치를 재사용한다.

### 4.3 구현 경로: 두 가지 중 선택 (Phase 2 시작 시 결정)

| 경로 | 장점 | 위험 |
|---|---|---|
| **A. FlashInfer 바인딩** (C++ 헤더 템플릿, paged KV, split 크기 고정 옵션) | 검증된 고성능 커널, 개발 기간 짧음 | split 고정이 prefill / decode / mixed 전 경로에 일관되게 적용되는지 확인 필요. SM80 지원, ABI와 workspace 수명을 Riley CUDA 런타임(lease 원장)에 맞춰야 함 |
| **B. 직접 구현** (기존 prompt MMA / tensor-core producer 코드를 기반으로 확장) | 수치 규칙을 완전히 통제, 기존 테스트 하네스 재사용 | 성능 튜닝 비용. Phase 1에서 이미 여러 번 이득이 작았음 |

**권장:** A를 먼저 2일 안에 타당성만 검증한다. 구체적으로는 고정 split 상태에서 배치 불변성이 유지되는지 확인하고, 커널 단위 하네스로 vLLM FA2 대비 시간을 잰다. 안 되면 B로 간다.

## 5. 단계별 계획

| 단계 | 내용 | 산출물 | 예상 |
|---|---|---|---|
| **P0 하네스** | 커널 단위 벤치 하네스: 실제 Qwen3-4B 형상, natural/fixed step 구성을 재현한 행 목록으로 현재 3커널 경로, 후보, (참고) FA2 시간을 측정 | 2분 이내 반복 루프, 기준 시간표 | 1일 |
| **P1 경로 선택** | A(FlashInfer)의 고정 split 불변성과 속도 확인. 실패 시 B | 선택 근거 문서 | 1–2일 |
| **P2 커널** | §3 계약 구현, S ∈ {128, 256, 512} 하네스 비교, 확률 MMA 형식 결정 | 커널 + 선택된 S | 3–5일 |
| **P3 정확성** | GPU 단위 테스트 (§6 G1), HF 일치 검사 | 테스트 통과 | 1–2일 |
| **P4 통합** | 실행 경로 연결(그래프 캡처, workspace, dynamic counts), 서버 플래그 `--attention fixed-split-varlen-v1 --attention-split-tokens S` | 빌드 + CPU/GPU 테스트 | 2–3일 |
| **P5 qualification** | 사전 등록 후 6개 셀 × 5쌍 + vLLM 대조 | 판정 문서 | 1일 |

각 단계 끝에서 진행 여부를 결정한다. P0 하네스에서 후보 커널의 attention 시간이 현재 경로 대비 −30% 미만이면 서버 통합(P4)으로 넘어가지 않는다. 서버 이득은 "attention 비중(약 20%) × 감소율"로 미리 추정하고, 노이즈 범위의 2배를 넘을 때만 진행한다.

## 6. 검증 게이트 (P5에서 사전 등록할 초안)

| 게이트 | 기준 |
|---|---|
| G1 GPU 단위 테스트 | 8개 head 구성 × 짧은/긴 context에서, 행별 출력 바이트가 배치 순서·타일 정렬·prefill 청크 분할·mixed/decode-only 구분·그래프 캡처 여부와 무관. 패딩 행 0. 기존 페이지 모드 대비 최대 차이가 반올림 범위 이내 |
| G2 서빙 불변성 | 후보의 모든 arm에서 측정 192/192가 자기 canonical과 일치, 교차 동시성 불일치 0 |
| G3 HF 일치 | correctness 세트에서 pass마다 11/15 이상 (현재 native와 vLLM은 12/15) |
| G4 주 목표 | natural C32 처리량 ≥ vLLM의 1.00배(같은 캠페인 대조), 기준선 native 대비 +5% 이상 |
| G5 비열등성 | 모든 셀에서 처리량 ≥ −1.3%, TPOT p50 ≤ +1.5%, TPOT p99 ≤ +10% (Phase 0 A/A 기반) |

## 7. 위험과 대응

| 위험 | 대응 |
|---|---|
| 고정 split 때문에 C1의 긴 context 병렬성이 부족함 | S를 작게 잡으면 조각 수가 늘어남(현재 16토큰보다는 크게). 하네스에서 C1 natural을 반드시 포함 |
| 결합 단계 비용이 다시 커짐 | 조각 수가 현재보다 S/16배 적음. 조각이 1개인 행은 결합을 생략 |
| 그래프 캡처와 동적 작업 목록의 호환 | 현재 dynamic counts 방식(최대 크기로 캡처, 디바이스에서 개수 읽기)을 그대로 씀 |
| FlashInfer 통합 시 메모리와 lease 규칙 충돌 | workspace를 Riley 소유 버퍼로 전달하고, 커널 호출만 위임 |
| HF 일치 수 변동 (반올림 변화로 근소한 동률 지점의 토큰이 바뀜) | G3는 correctness 세트 기준. 측정 workload의 일치 수는 참고 지표로 보고 |
| 기존 실험 모드 누적 | 채택 시 fused / segment / direct / 하이브리드 실험 경로는 제거하거나 검증 전용으로 격리 |

## 8. 이 계획 이후 (우선순위 순)

1. **RMSNorm 경로 정리 (+53 ms):** hidden 크기가 256의 배수가 아닌 경우를 포함해 범용 커널을 정리한다.
2. **native를 실제 기본값으로:** 29개 플래그를 모델·GPU별 자동 프로파일로 접는다. 모델 특화 상수를 일반화한다(`qwen2.5-3b.d128.qh16.kvh2` 같은 백엔드 ID, 256행 상한의 근거 등).
3. **비동기 step 제출:** step당 약 0.3 ms, 처리량 약 2%. attention 격차가 줄어든 뒤 착수한다.
4. **제품 기능:**
   - 결정적 prefix cache
   - `/v1/chat/completions`와 샘플링 옵션
   - speculative decoding (C1에서 이길 유일한 수단)
   - weight-only 양자화
5. **다른 모델 재측정:** A100에서 Qwen3-8B, Llama-3.x. 10/14에 c32-natural TPOT +5–12%로 같은 attention 원인이 나온 모델들이다.

## 9. 진행 중인 항목과의 관계

- **하이브리드 direct prompt(§14):** 5쌍 qualification이 진행 중이다. 결과와 무관하게 이 계획의 3절 규칙이 상위 호환이다. 조각이 1개인 행은 direct와 같고, 긴 행은 고정 split으로 병렬화한다.
- **버스트 정책(§8):** 사전 등록 기준대로 opt-in으로 유지한다. attention 비용이 줄면 큰 step의 TPOT 비용도 줄어 재평가할 가치가 생긴다.
