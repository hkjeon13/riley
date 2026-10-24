# Phase 1 중간 결과: native 긴 문맥 attention과 버스트 TTFT

2026-10-22 KST, aws-a100 GPU7, Qwen3-4B BF16 greedy.
전제는 [Phase 0](phase0-contract-cost-and-aa-20261022.md) 이후의 결정이다. native(배치 불변)가 제품 기본이고, HF-literal은 검증 모드다.
아래 수치는 모두 스크린 결과다(엔진·셀마다 새 프로세스 3쌍, 64 warm + 192 측정, 중앙값). 정식 qualification이 아니다.
근거는 [`benchmarks/evidence/20261022-phase1/`](../benchmarks/evidence/20261022-phase1/)에 있다.

## 1. 남은 격차의 위치 (natural C32 프로파일)

같은 64요청 구간을 Nsight로 측정했다. 두 엔진 모두 CUDA Graph 노드 단위로 트레이스했다.

| 항목 | native ms | vLLM 0.27.1 ms |
|---|---:|---:|
| GEMM 합 | ~1856 | ~1906 |
| decode attention (producer + merge) | 263 + 121 | 317 + 21 (FA2 split-KV + combine) |
| prompt attention (dense-v1 MMA) | 143 | (split-KV에 포함) |
| 커널 합 / span | 2647 / 2805 | 2497 / 2557 |

GEMM은 동률이다. span 차이 약 10%는 거의 attention에서 나온다. 특히 페이지별 FP32 상태를 global memory에 썼다가 다시 읽는 merge와 prompt-row 상태가 크다.

## 2. 실험 A: fused decode attention (미채택)

producer와 global-max merge를 (행, KV head)당 CTA 하나로 합쳤다. 기존 경로와 연산 순서를 그대로 재현한다.
- **정확성:** GPU 테스트 `fused_decode_attention_gpu`에서 8개 head 구성 × 2개 문맥 길이(최대 128페이지) × 5개 배치 구성 모두 **기존 경로와 출력 바이트가 같았다**. 서빙에서도 측정 요청 192/192가 canonical 출력과 일치했다.
- **성능:**

| 셀 | native | fused | 변화 |
|---|---:|---:|---:|
| natural C8 | 818.4 | 729.2 | −10.9% |
| natural C32 | 2113.1 | 1892.4 | −10.4% |
| fixed C1 | 123.7 | 118.3 | −4.4% |
| fixed C32 | 1693.0 | 1626.0 | −4.0% |

- **원인:** CTA 수가 (행 × KV head)로 줄고, 페이지를 warp 안에서 순차 처리하면서 메모리 수준 병렬성이 줄었다. 상태 왕복을 없앤 이득보다 손실이 컸다.
- **처리:** 코드는 남기되 기본값은 꺼짐이다(`RILEY_NATIVE_D128_FUSED_DECODE_MIN_ROWS` 미설정 또는 0).
- **교훈:** 페이지 단위로 병렬성이 넓은 현재 producer가 강점이다. 다음 시도는 merge 자체(행당 순차 fold)와 prompt-row 상태 경로를 줄이는 방향으로 한다.

## 3. 실험 B: budget 512 항상 적용 (미채택)

dense-row 상한을 1024로 올렸다. 대신 budget × 시퀀스 토큰이 기존에 검증된 256 × 32768을 넘지 않도록 묶었다.

| 셀 | 처리량 | TTFT p95 | TPOT p50 | ITL p99 |
|---|---:|---:|---:|---:|
| fixed C16 | −2.9% | −18% | +2.3% | +77% |
| fixed C32 | −8.8% | −20% | +16% | +63% |
| natural C32 | −2.2% | −3% | +2.6% | +65% |

TTFT 꼬리는 줄지만, 큰 step이 decode를 지연시켜 처리량과 TPOT가 나빠진다.

## 4. 실험 C: 버스트 정책 (`--prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows K`, budget 512)

decode 행이 K개 이하일 때만 전체 512를 열고, 그 외에는 256을 쓴다. 출력 불변성은 유지됐다(교차 동시성 불일치 0, 측정 192/192).

| K | 셀 | 처리량 | TTFT p95 (ms) | TPOT p50 | TPOT p99 | ITL p99 (ms) |
|---|---|---:|---:|---:|---:|---:|
| 8 | fixed C16 | +3.4% | 68.8 → 64.8 | −3.6% | +3.3% | 16.5 → 16.8 |
| 8 | fixed C32 | +1.1% | 256.1 → 253.5 | −0.3% | +9.7% | 17.8 → 17.9 |
| 8 | natural C32 | 0.0% | 288.9 → 282.9 | +0.2% | −2.4% | 20.3 → 20.7 |
| 16 | fixed C16 | −2.6% | 69.9 → 63.4 | +2.1% | +7.6% | 16.5 → 29.1 |
| 16 | fixed C32 | +1.0% | 263.8 → **217.4** | −0.5% | +24.7% | 18.3 → 28.7 |
| 16 | natural C32 | +1.0% | 269.2 → **244.2** | −1.1% | +2.5% | 20.4 → 20.7 |

참고로 vLLM의 fixed C32 값은 TTFT p95 163.3ms, TPOT p50 15.59ms, ITL p99 27.0ms다(Phase 0).

- **K=8:** 처리량과 TPOT 중앙값을 지키지만, TTFT 개선이 작다.
- **K=16:** fixed C32 TTFT p95를 18% 줄인다. 대신 ITL p99가 vLLM과 비슷한 수준(28.7 대 27.0ms)으로 올라간다.
- 아직 vLLM의 TTFT p95(163ms)에는 못 미친다.
- 3쌍 스크린이라 ±1% 안팎의 처리량 변화는 노이즈 범위로 본다(Phase 0 A/A 기준).

## 5. 바뀐 코드

- `kernels/src/batch_primitives.cu`: fused decode 커널(기본 꺼짐), merge의 prompt 전용 모드
- `crates/riley-cuda`: `set_native_d128_fused_decode_min_rows`, GPU 테스트 `fused_decode_attention_gpu.rs`
- `crates/riley-runtime/.../owner.rs`, `crates/riley-server/src/main.rs`: dense-row 상한 256 → 1024, budget × 시퀀스 토큰 ≤ 256 × 32768
- `crates/riley-scheduler`: `PrefillBurstPolicy`와 테스트 `prefill_burst.rs`
- 서버 플래그: `--prefill-burst-steady-budget`, `--prefill-burst-max-decode-rows`

테스트:
- 로컬: 스케줄러 4개, CLI 1개 통과
- 호스트(v3 바이너리 sha `bfe682f8…`): CPU 테스트는 원래부터 있던 C02 EFS 소유권 문제 4개와, 이후 수정한 CLI 테스트 1개를 빼고 모두 통과했다. `batch_model_exact_gpu` 통합 테스트는 원래부터 컴파일되지 않는다.

## 6. 다음 단계

1. **버스트 정책:** K=12 근처를 확인하고, 노이즈 기준 게이트로 5쌍 qualification을 진행한다.
2. **긴 문맥 attention:**
   - merge kernel: 행당 순차 fold를 depth별 다중 load로 바꾸거나, 여러 페이지를 묶어 producer 단계에서 부분 결합한다.
   - prompt-row 상태 경로: 143ms를 줄인다.
   - 모두 바이트 동일성 테스트를 먼저 통과해야 한다.
3. **유지할 조건:** 모든 후보는 native 출력 불변성(측정 192/192, 교차 동시성 불일치 0)을 지켜야 한다.

## 7. 버스트 정책 K 선택과 5쌍 qualification (사전 등록)

사전 등록: [burst-qualification-preregistration.json](../benchmarks/evidence/20261022-phase1/burst-qualification-preregistration.json). 해시는 측정 전에 호스트의 `burst-preregistration.sha256`에 기록했다.

### K 선택 (fixed C16/C32, 3쌍)

규칙은 세 가지다. 처리량 변화 ≥ −1.0%, TPOT p50 변화 ≤ +1.5%, C32 ITL p99 ≤ 27.0ms를 모두 만족하는 K 중에서 C32 TTFT p95가 가장 낮은 값을 고른다.
- K=8, 10, 12는 통과했다.
- K=14는 C16 처리량 −2.8%와 TPOT p50 +2.0%로 탈락했다.
- K=16은 C16 처리량 −2.5%, TPOT p50 +2.1%, C32 ITL p99 28.7ms로 탈락했다.
- C32 TTFT p95는 K=10이 242.70ms, K=12가 242.75ms로 사실상 같았다. 규칙상 **K=10**을 골랐다.

### Qualification 결과 (6개 셀 × 5쌍, 중앙값 paired 변화)

| 셀 | 처리량 | TPOT p50 | TPOT p99 | TTFT p95 | ITL p99 |
|---|---:|---:|---:|---:|---:|
| fixed C1 | +0.50% | −0.03% | +0.79% | −0.70% | −0.10% |
| fixed C8 | +0.40% | −0.31% | +1.15% | **+8.36%** | **+39.7%** |
| fixed C16 | +3.07% | −3.66% | +2.43% | −7.59% | +0.27% |
| fixed C32 | +0.73% | −0.75% | +9.26% | **−8.97%** (262.9 → 239.3ms) | −0.35% (17.7ms) |
| natural C8 | −0.58% | +1.29% | +0.29% | −7.80% | **+76.3%** |
| natural C32 | +0.86% | −0.82% | −3.92% | −7.91% | +0.46% |

출력 불변성은 모든 arm에서 유지됐다(측정 192/192, 교차 동시성 불일치 0).

### 판정: 게이트 미통과 → 미채택

- **주 목표 미달:** fixed C32 TTFT p95 −8.97%로 기준(−10% 이하)에 못 미쳤다.
- **나머지 조건은 모두 통과:**
  - 처리량: 최저 −0.58%, 기준 −1.3%
  - TPOT p50: 최대 +1.29%, 기준 +1.5%
  - TPOT p99: 최대 +9.26%, 기준 +10%
  - C32 ITL p99: 17.7ms, 기준 27.0ms
  - 출력 불변성
- **게이트 밖의 악화:** fixed C8 TTFT p95 +8.4%, fixed C8 ITL p99 +40%, natural C8 ITL p99 +76%. 동시성이 낮은 셀에서는 decode 행이 늘 K 이하라 큰 step이 자주 섞이기 때문이다.
- **처리:** 사전 등록대로 기본은 budget 256으로 유지한다. 버스트 플래그는 opt-in으로 남긴다.
- **다음 개선 방향:** K를 고정값 대신 `max_active_sequences`에 비례시키거나, 대기 prompt 토큰 수를 조건에 더한다. 그러면 C8에서는 큰 step이 열리지 않는다.

## 8. 동시성 비례 bound (P=31) 5쌍 qualification (사전 등록)

사전 등록: [burst-percent-preregistration.json](../benchmarks/evidence/20261022-phase1/burst-percent-preregistration.json). 해시는 측정 전에 호스트의 `burst-percent-preregistration.sha256`에 기록했다.

- **후보:** `riley-phase1-v4`(sha `5c9d71fc…`). 새 플래그 `--prefill-burst-max-decode-percent 31`을 쓴다. 전체 budget을 여는 decode 행 수 상한 K를 `ceil(31% × max_active_sequences)`로 정하므로 C1/C8/C16/C32에서 K는 1/3/5/10이다.
- **기준선:** v15C, budget 256.
- **조건:** 6개 셀 × 5쌍, 제외 0. 출력 불변성은 모든 arm에서 유지됐다(측정 192/192, 교차 동시성 불일치 0).

| 셀 | 처리량 | TPOT p50 | TPOT p99 | TTFT p95 | TTFT p99 | ITL p99 |
|---|---:|---:|---:|---:|---:|---:|
| fixed C1 | +0.45% | 0.00% | −0.19% | −0.59% | +2.34% | +0.02% |
| fixed C8 | +1.66% | −1.43% | −1.79% | −1.22% | −3.52% | −0.10% |
| fixed C16 | +2.71% | −3.32% | −0.46% | −2.01% | −4.41% | +0.59% |
| fixed C32 | +0.98% | −0.56% | +9.47% | **−5.28%** (260.8 → 242.7ms) | −3.48% | −0.97% (17.7ms) |
| natural C8 | +0.55% | −0.65% | −1.19% | −2.66% | +9.56% | +0.17% |
| natural C32 | +0.81% | −0.75% | −3.01% | **−13.88%** (308.4 → 270.9ms) | −4.61% | +0.41% |

30개 쌍 모두 처리량 변화가 양수였다(+0.05% ~ +3.59%).

### 판정

- **질문 1 (C8 악화 해결): 통과.**
  - fixed C8: TTFT p95 −1.22%, ITL p99 −0.10%
  - natural C8: TTFT p95 −2.66%, ITL p99 +0.17%
  - K=10 고정 버전에서 생긴 C8의 TTFT +8.4%, ITL +40~76% 악화가 사라졌다.
- **질문 2 (채택 게이트, 변경 없음): 미통과.**
  - 주 목표인 fixed C32 TTFT p95가 −5.28%로 기준(−10% 이하)에 못 미쳤다.
  - 나머지 조건은 모두 통과했다: 처리량 최저 +0.45%, TPOT p50 최대 0.00%, TPOT p99 최대 +9.47%, C32 ITL p99 17.7ms, 출력 불변성.
- **처리:** 사전 등록대로 기본 설정으로는 채택하지 않는다. 플래그는 opt-in으로 남긴다.
- **참고:** 정책 자체는 K=10 고정판과 같은데도 C32 TTFT p95 감소가 −8.97%에서 −5.28%로 줄었다. C32 TTFT p95 효과는 실행 간 변동이 크다는 뜻이다. 반면 다른 지표는 악화가 없거나 노이즈 수준이다. **채택 여부는 TTFT 목표를 어떻게 볼지에 달린 판단이며, 사용자의 결정이 필요하다.**

## 9. 긴 문맥 attention: 바이트 동일 최적화 2차 (미채택)

사용자 결정(2026-10-23): 버스트 정책은 사전 등록 기준대로 opt-in으로만 두고, 긴 문맥 attention 개선으로 넘어간다.

| 후보 | 바이트 동일성 검증 | 처리량 변화 (3쌍 중앙값) | 판정 |
|---|---|---|---|
| merge prefetch (v5, sha `7f0de1b6…`): 앞 16페이지 분자를 리듀션 전에 미리 읽고, 이후 8페이지씩 load | fused 커널(독립 구현한 merge 규칙)과 8개 head 구성 × 2개 문맥 길이 모두 일치 | natural C8 +0.03%, natural C32 −0.06%, fixed C1 +0.31%, fixed C32 −1.21% | 기각 |
| prompt-row 커널의 한 페이지 앞 K/V 레지스터 prefetch (v7, sha `d68326b5…`) | 변경 전후 테스트 덤프 80개 파일 바이트 일치 | natural C8 −0.10%, natural C32 −0.58%, fixed C1 ±0%, fixed C32 −0.45% | 기각 |

- 모든 arm에서 출력 불변성을 유지했다(측정 192/192, 불일치 0).
- prompt prefetch의 fixed 스크린 중 `c1-r2-native-after` arm이 측정 도중 중단됐다. 모니터가 holder(PID 2027915)의 heartbeat 지연을 외부 프로세스로 오인했기 때문이다.
  - 근거: `screen-prompt-fixed/c1-r2-native-after.monitor-errors.json`
  - holder는 직후 정상(idle) 상태였다.
  - 중단된 arm은 보존했다. 빠진 부분은 새 label `screen-prompt-fixed-c32`(3쌍)와 `screen-prompt-fixed-c1-cont`(1쌍)로 측정했다.
- 두 변경 모두 소스에서 되돌렸다. fused 커널(기본 꺼짐)과 merge의 prompt 전용 모드는 남아 있다.

### 결론

현재 수치 계약(페이지 단위 상태 + global-max merge)을 바이트 단위로 유지하는 저비용 재구성 세 가지는 모두 이득이 없었다(fused, merge prefetch, prompt prefetch). 커널을 합치거나 메모리 지연을 겹치는 것만으로는 natural C32에 남은 약 9% 격차가 줄지 않는다.

남은 레버는 수치 순서가 바뀌는 새 numerics mode다. 예를 들어 decode 행도 prompt 행처럼 4페이지 segment 단위 상태를 쓰면 상태 수와 merge 작업이 1/4로 준다. 배치 불변성은 유지되지만 출력 바이트가 바뀐다. 따라서 이 방향으로 가려면 HF 토큰 일치와 동시성 불변성을 새로 검증해야 하고, 사용자의 결정이 필요하다.

## 10. segment decode numerics mode (사전 등록, 미채택)

사용자 결정(2026-10-23): 배치 불변성은 유지하는 새 numerics mode로 진행한다. decode 행도 4페이지 segment 상태를 쓰고, HF 토큰 일치와 동시성 불변성을 재검증한다.

사전 등록: [segment-decode-preregistration.json](../benchmarks/evidence/20261022-phase1/segment-decode-preregistration.json). 해시는 측정 전에 `segment-preregistration.sha256`에 기록했다.

- **후보:** `riley-phase1-v8`(sha `0f5a2f86…`)에 `RILEY_NATIVE_D128_DECODE_SEGMENT_MODE=1`을 준다.
- **방식:** CTA 하나가 (4페이지 segment, KV head, 행)을 맡는다. 페이지마다 tensor-core producer와 같은 방식으로 점수와 가중치를 계산하고, segment 안에서는 online rescale로 오름차순 결합한다. merge는 segment 상태를 접는다.
- **러너 변경:** 러너는 모드별 canonical 출력을 따로 둔다(`--per-mode-canonical`). 새 모드의 불변성은 자기 출력 기준으로 검사한다.

### 게이트 판정

| 게이트 | 결과 | 판정 |
|---|---|---|
| G1 GPU 단위 테스트 | 8개 head 구성 × 20/120페이지: 배치 순서·호출 분할·prompt 이웃에 대해 행별 바이트 동일(구성당 124행), 패딩 행 0, 페이지 모드와 최대 차이 0.00024, 원소의 99.98% 이상이 페이지 모드와 비트 동일 | 통과 |
| G2 서빙 불변성 | 후보의 모든 arm에서 측정 192/192가 자기 canonical과 일치, 교차 동시성 불일치 0 | 통과 |
| G3 HF 토큰 일치 | correctness 세트에서 C1 12/15, C>1 36/45 (기준선 native와 같음) | 통과 |
| G4 natural C32 처리량 ≥ +2.0% | −0.10% | **미통과** |
| G5 비열등성 | 처리량: fixed C1 −2.71%, C8 −1.90%. TPOT p50: C1 +2.95%, C8 +2.61% | **미통과** |

나머지 셀: fixed C16 −1.09%, fixed C32 −0.93%, natural C8 −0.47%. TPOT p99는 모든 셀에서 +3.1% 이하였다.

참고(게이트 밖): 측정 workload의 HF 토큰 일치가 달라졌다.
- fixed: 192개 중 160 → 176으로 올랐다.
- natural: 128 → 64로 내려갔다.
- 반올림 차이로 근소한 동률 지점에서 다른 토큰을 고른 결과다. 같은 프롬프트가 반복되므로 프롬프트 단위로 묶여 바뀐다.

GPU 테스트의 첫 시도는 테스트 데이터 오류로 실패했다. 혼합 배치에서 prompt run과 decode 행이 같은 (시퀀스, 위치)를 썼다. 커널과 바이너리는 바꾸지 않고 테스트 데이터만 고쳐 다시 실행했다. 첫 시도 로그는 `segment-gpu-test-attempt1/`에 보존했다.

### 결론

segment 모드는 정확성 게이트(G1~G3)를 모두 통과했지만, 성능 게이트(G4, G5)를 통과하지 못해 미채택이다. 기본값은 꺼짐이다.
- 동시성이 낮으면 CTA 수가 1/4로 줄어 C1/C8이 느려졌다.
- natural C32에서도 이득이 없었다.

따라서 natural C32의 약 9% 격차는 페이지 상태 수나 merge 작업량에서 오는 것이 아니다. 지금까지 시도한 네 가지(fused, merge prefetch, prompt prefetch, segment)가 모두 이 가설 위에 있었다.

다음에 할 일은 원인 재조사다. 커널 시간 합이 아니라 step 단위 타임라인으로 다시 본다.
- prefill이 섞인 step과 순수 decode step을 나눠 vLLM과 비교한다.
- prompt-row attention 143ms가 vLLM의 어느 커널에 해당하는지 확인한다.

## 11. natural C32 격차: step 단위 타임라인 재조사

1절의 Nsight 노드 단위 트레이스를 step 단위로 다시 나눴다. 64요청 구간, Riley native(v15C)와 vLLM 0.27.1을 비교했다.
- **step 경계:** 각 엔진의 샘플링 커널. Riley는 `bf16_argmax_kernel`, vLLM은 `_gumbel_sample_kernel`이다.
- **mixed(prompt 토큰 포함) 판정:** Riley는 prompt-row MMA 커널이 있는 step이다. vLLM은 그래프 밖(eager, piecewise)에서 실행된 `flash_fwd_splitkv_kernel`이 있는 step이다.
- **근거:** 스크립트는 [steps.py](../benchmarks/evidence/20261022-phase1/steps.py), 원자료는 `steps-natural-c32.json`이다.
- **주의:** 노드 단위 트레이스는 계측 비용이 있다. 두 엔진을 같은 방식으로 측정했으므로 비율 비교에만 쓴다.

| | Riley mixed | Riley decode | vLLM mixed | vLLM decode |
|---|---:|---:|---:|---:|
| step 수 | 61 | 175 | 33 | 201 |
| span 합 (ms) | 1016 | 1623 | 741 | 1816 |
| step 중앙값 span (ms) | 17.3 | 9.51 | 26.2 | 9.17 |
| **step 사이 GPU 유휴 합 (ms)** | **54** | **111** | 0.04 | 0.22 |
| step 사이 유휴 중앙값 (µs) | 896 | 637 | 1 | 1 |
| step당 attention 중앙값 (ms) | 4.56 | 1.41 | 3.34 | 1.15 |

커널 범주별 합 (ms, Riley / vLLM):

| 범주 | Riley | vLLM | 차이 |
|---|---:|---:|---:|
| GEMM | 1857 | 1908 | −51 |
| attention (decode + prompt) | 527 | 338 | **+189** |
| RMSNorm | 113 | 60 | +53 |
| act + KV 기록 + 기타 + 샘플링 | 150 | 191 | −41 |
| **step 사이 유휴** | **165** | **0.3** | **+165** |

### 결론

1. **가장 큰 단일 원인은 step 사이 GPU 유휴다.**
   - Riley는 step마다 0.64ms(decode)~0.90ms(mixed)씩 GPU가 쉰다. 합치면 165ms로, Riley span의 약 6%다.
   - vLLM은 다음 step을 미리 제출해 유휴가 거의 0(약 1µs)이다.
   - 앞서 시도한 attention 최적화 네 가지로는 이 부분을 줄일 수 없었다.
2. **prompt-row attention이 대응하는 vLLM 커널:** mixed step에서 그래프 밖(eager)으로 실행되는 `flash_fwd_splitkv_kernel` 하나다(1188회 = 33 step × 36 layer, grid (1–7, 1–32, 32)).
   - 이 커널은 varlen FlashAttention-2로 prompt 토큰과 decode 토큰을 함께 처리한다. 순수 decode step에서는 그래프 안의 split-KV 경로(7236회)와 combine 커널을 쓴다.
   - Riley는 같은 일을 prompt MMA 143ms + 페이지 상태 + merge로 처리한다. vLLM의 mixed step attention 전체는 108ms다.
   - vLLM mixed step은 step당 prompt 토큰이 약 2배(budget 576 대 256)인데도 그렇다. 따라서 prompt 토큰당 attention 비용은 Riley가 크게 높다.
3. **decode 전용 step:** attention이 step당 1.41ms 대 1.15ms(+23%)다. step span은 +3.7%(9.51 대 9.17ms)다.
4. **RMSNorm:** +53ms. 10/14 C1 진단(일반 hidden 폭용 norm 커널)과 같은 계열이다.

### 다음 레버 (기대 효과 순)

1. **비동기 step 제출 (CPU/GPU overlap):**
   - native 경로에서 step N의 샘플 결과를 기다리지 않고 step N+1을 GPU에 제출해 유휴 165ms를 없앤다.
   - strict 경로의 v18B4는 GPU 계산이 병목이라 효과가 묻혔다. native에서는 유휴가 span의 6%다.
   - EOS, 취소, KV commit 계약을 지켜야 한다.
2. **prompt-row attention:**
   - mixed step 전용 varlen 커널을 만든다. 세 가지가 필요하다: 128행 query 블록, 전체 context online softmax, 페이지 상태 없이 직접 출력.
   - 새 numerics mode이므로 G1~G3 같은 정확성 검증이 필요하다.
3. **RMSNorm 경로 정리:** 53ms.

## 12. step 사이 유휴의 정정: profiler 없이 측정한 host 시간

사용자 지시(2026-10-23): native 경로에 비동기 step 제출을 구현해 step 사이 유휴를 없앤다. 착수 전에 유휴의 실제 크기를 다시 측정했다.

### 노드 단위 트레이스가 유휴를 부풀렸다

11절 트레이스의 CUDA runtime API 기록으로 step 사이 구간(GPU 끝 → 다음 step GPU 시작, 중앙값 616µs)을 분해했다.

| 구간 | 중앙값 |
|---|---:|
| 동기화 복귀 | 41µs |
| host 계산 | 11µs |
| API 호출 | 486µs |

API 호출 중 거의 전부는 `cudaGraphLaunch`(약 506µs/회)였다. 그런데 계측 비용이 작은 그래프 단위 트레이스(1절의 기본 트레이스)에서는 같은 호출이 중앙값 110µs(p90 220µs)였다. 즉 노드 단위 트레이스가 launch 비용을 4~5배 부풀렸다. **11절의 "유휴 165ms, span의 약 6%"는 과대 추정이다.**

### profiler 없는 측정 (`RILEY_HOST_STEP_TRACE=1`)

기존 host 단계 계측(Instant 기반, 128 step마다 평균 출력)을 v15C native에 켜고 C32를 1회씩 측정했다. 근거는 [`hosttrace/`](../benchmarks/evidence/20261022-phase1/hosttrace/)에 있다. 아래는 natural C32 정상 상태의 step당 평균(µs)이다.

| plan | pack | prepare+upload | submit (graph launch 포함) | download | sample | commit+events+publish | cancel+commands | wait (GPU 실행 대기) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42–44 | 42–45 | 5 | 131–135 | 26 | 24 | 25–27 | 5 | 12,755–13,090 |

- **natural C32:** GPU가 기다리는 host 구간(wait 밖)은 step당 약 0.30~0.33ms다. wait(약 13ms)의 **약 2.4%**다.
- **fixed C32:** 정상 상태 구간(세 번째 출력)에서 약 0.24ms로, wait의 약 1.8%다.
- **fixed C32의 앞 두 출력은 제외했다.** 시작 직후 그래프 캡처와 warm-up이 섞여 있다(`captured=27`, `cancel+commands` 2.8~5.2ms).

### 결론과 판단 필요 사항

- 비동기 step 제출로 줄일 수 있는 상한은 step당 약 0.25~0.33ms, 처리량 기준으로 **약 2~2.5%**다. 11절에서 추정한 6%가 아니다.
- 그중 가장 큰 단일 항목은 graph launch(`submit`, 약 130µs)다.
- 필요한 변경 범위는 크다. CUDA command batch의 제출·대기 분리(lease 원장), staging 버퍼 2벌, GPU 측 토큰 전달 커널, 스케줄러의 이중 in-flight와 연쇄 KV 예약, 종료된 행의 투기적 결과 폐기가 모두 필요하다.
- 따라서 투입 대비 효과를 다시 판단해야 한다. 같은 범위의 변경으로 얻을 이득이 앞선 추정의 절반 이하다.
- natural C32에 남은 약 9% 격차의 대부분은 GPU 커널 시간과 step 구성이다.
  - attention: 특히 mixed step의 prompt attention
  - RMSNorm
  - budget 256으로 인한 mixed step 수(61개 대 33개)

## 13. direct prompt-row attention numerics mode (사전 등록, 미채택)

사용자 결정(2026-10-23): 비동기 제출은 보류하고, mixed step용 varlen prompt attention을 새 numerics mode로 만들어 검증한다.

사전 등록: [prompt-direct-preregistration.json](../benchmarks/evidence/20261022-phase1/prompt-direct-preregistration.json). 해시는 측정 전에 `direct-preregistration.sha256`에 기록했다.

- **후보:** `riley-phase1-v9`(sha `bd47e1fe…`)에 `RILEY_NATIVE_D128_PROMPT_DIRECT=1`을 준다.
- **방식:** prompt 행은 (16행 타일, KV head)마다 CTA 하나가 전체 context를 online softmax로 접고 BF16 출력을 직접 쓴다. 페이지 상태와 merge가 없다.
- **decode 행:** 변경 없음.

### 게이트 판정

| 게이트 | 결과 | 판정 |
|---|---|---|
| G1 GPU 단위 테스트 | 8개 head 구성 × 20/120페이지에서 prompt 행 바이트가 실행 순서·타일 정렬·generated 목록 유무·단독 실행·청크 분할과 무관(구성당 549행). decode 행은 바이트 동일, 패딩 행 0. segment 경로 대비 최대 차이 0.00195, 원소의 약 90~97%는 비트까지 동일 | 통과 |
| G2 서빙 불변성 | 후보의 모든 arm에서 측정 192/192가 자기 canonical과 일치, 교차 동시성 불일치 0 | 통과 |
| G3 HF 토큰 일치 | correctness 세트에서 C1 12/15, C>1 36/45 (기준선과 같음) | 통과 |
| G4 natural C32 처리량 ≥ +2.0% | **−0.51%** | **미통과** |
| G5 비열등성 | 처리량 최저 −0.53%, TPOT p50 최대 +0.64%, TPOT p99 최대 +1.02% | 통과 |

셀별 결과(5쌍 중앙값 변화):

| 셀 | 처리량 | TPOT p50 | TTFT p95 |
|---|---:|---:|---:|
| fixed C1 | +0.22% | +0.05% | −3.27% |
| fixed C8 | +0.30% | −0.04% | −0.77% |
| fixed C16 | +0.58% | −0.77% | −4.39% |
| fixed C32 | **+2.21%** (5쌍 모두 +2.17% ~ +2.59%) | **−2.30%** | −4.40% |
| natural C8 | −0.53% | +0.64% | +2.86% |
| natural C32 | −0.51% | +0.39% | −5.86% |

### 해석

- **짧은 prompt(fixed, 115–130토큰)에서는 이득이 있다.**
  - 페이지 상태와 merge 왕복이 사라졌다.
  - C32에서 처리량 +2.2%, TPOT −2.3%를 얻었다.
- **긴 prompt(natural, 최대 398토큰)에서는 이득이 없다.**
  - 기존 경로는 4페이지 segment마다 CTA를 나눠 병렬로 처리했다.
  - direct 경로는 CTA 하나가 최대 25페이지를 순차로 처리한다. segment 병렬성을 잃은 손실이 상쇄한 것으로 보인다.
  - 사전 등록한 주 목표(natural C32)를 통과하지 못했으므로 기본 설정으로는 채택하지 않는다.
- **HF 일치(게이트 밖 참고):** 측정 workload에서 fixed 160 → 176, natural 128 → 64로 바뀌었다.
  - segment 모드(10절)와 수치가 같다.
  - 반올림이 조금만 바뀌어도 가장 근소한 동률에 있는 같은 프롬프트(fixed 1개, natural 1개)가 바뀌기 때문이다.

### 다음 후보 (사용자 판단 필요)

- **(a) 하이브리드:** prompt 행의 context가 짧을 때(예: 8페이지 이하)만 direct를 쓰고, 긴 행은 segment 경로를 쓴다.
  - 선택 기준이 행 자신의 context 길이뿐이라 배치 불변성은 유지된다.
  - fixed C32의 +2.2%를 얻으면서 natural의 손실을 피할 수 있다.
- **(b) split-KV direct:** 긴 context는 일정한 페이지 수(예: 8페이지) 단위로 나눠 병렬로 처리하고, 나눈 조각을 고정 순서로 결합한다.

## 14. 하이브리드 direct prompt (≤12페이지만 direct, 사전 등록, 미채택)

사전 등록: [prompt-hybrid-preregistration.json](../benchmarks/evidence/20261022-phase1/prompt-hybrid-preregistration.json). 해시는 측정 전에 `hybrid-preregistration.sha256`에 기록했다.
- **후보:** `riley-phase1-v10`(sha `f162bd78…`)에 `RILEY_NATIVE_D128_PROMPT_DIRECT=1`, `RILEY_NATIVE_D128_PROMPT_DIRECT_MAX_PAGES=12`를 준다.

### 정확성 게이트: 모두 통과

- **G1:** 4페이지 기준 하이브리드 검사에서 짧은 prompt 행 44개는 direct와, 긴 행 78개는 segment 경로와 바이트가 같았다. 구성마다 불변성 검사 행은 846개다.
- **G2:** 측정 192/192, 교차 동시성 불일치 0.
- **G3:** correctness 세트에서 12/15, 36/45.

### 성능: 미통과 (G4, G5)

| 셀 | 처리량 | TPOT p50 | TPOT p99 | TTFT p95 |
|---|---:|---:|---:|---:|
| fixed C1 | −0.09% | +0.03% | −0.04% | +0.28% |
| fixed C8 | −1.96% | +2.04% | +2.97% | −0.75% |
| fixed C16 | −2.01% | +2.24% | +1.73% | −1.04% |
| fixed C32 | **−2.08%** | +2.66% | +4.26% | +5.23% |
| natural C8 | −1.84% | +1.85% | +2.64% | +8.69% |
| natural C32 | **−3.20%** | +4.11% | +4.31% | +1.78% |

30쌍 모두 처리량이 음수였다(C1 제외).

### 원인 (추정)

- 이 구현은 prompt 행이 있는 step마다 direct 커널과 segment 커널을 모두 띄운다.
- segment 커널은 짧은 run을 커널 안에서 건너뛸 뿐이고, 그리드는 (타일 × KV head × 16 segment) 전체가 뜬다.
- merge 커널도 계속 돈다.
- 그래서 full direct(13절, fixed C32 +2.2%)의 이득이 사라지고, 커널 실행이 하나 더 붙은 비용만 남았다.
- 결국 "경로 선택은 행 단위로 하되 커널은 하나"여야 한다는 결론이다. 이는 개선 계획(plan-fixed-split-varlen-attention-20261023.md)의 단일 varlen 실행 설계와 일치한다.

판정: 미채택. 기본값은 꺼짐이다.
