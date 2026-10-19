# Exact small-row batching and actual-row CUDA Graph reuse (SM89)

검증한 통합 소스에서 `--owned-decode-batch exact-smallrows-v5`를 선택할 수 있습니다. Qwen3-4B의 3행 Down/Attention-output과 Mistral-7B의 2행 Query/Attention-output에서 가중치 로드를 재사용하고, 실제 행 수마다 별도의 레이어 전후 CUDA Graph를 보관합니다. QKV·Gate/Up·Down의 기존 연산 순서와 원래 요청별 prefill 행 수를 유지합니다. Q/K/V의 무바이어스 경로도 새 모드를 전달하도록 수정했습니다.

옵션 5는 시험한 SM89/128-SM/CUDA12.8/cuBLAS120804와 두 모델의 검증된 geometry에서 최대 32개 시퀀스로 제한합니다. 기존 2/3/4 모드의 계산 경로와 Graph 적용 범위를 별도 검증했습니다. 새 옵션은 처리량과 첫 토큰 지연 사이의 실측 차이가 있어 명시적으로 선택합니다. Qwen3-4B C2 전용의 기존 `exact-r2-global-v4` 설정도 별도로 유지합니다. 새 옵션의 성능 검증 범위는 아래 네 셀입니다.

아래 값은 같은 RTX 4090에서 실행한 모델·배치별 새 프로세스 1회씩의 비교입니다. 기존 실행은 실제 통합151 바이너리, 개선 실행은 현재 소스 935개와 해시가 일치하는 통합164 바이너리입니다. stock vLLM 0.27.1을 같은 GPU에서 따로 실행했습니다. 22GiB 예산, context1024, 토큰 예산512, KV block16/블록 수64×동시성, prefix cache off, 원래 EOS와 greedy 설정을 맞췄습니다. Riley는 CUDA12.8, stock 설치본은 CUDA13입니다. Graph 워밍 뒤의 측정이며, 측정 구간의 새 Graph 캡처는 0건입니다.

| 모델 / 동시성 | 개선 전 tok/s | 개선 후 tok/s | 변화 | stock vLLM tok/s | 개선 후 / vLLM |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B / C3 | 153.0 | 209.2 | +36.7% | 268.8 | 0.78× |
| Qwen3-4B / C8 | 370.5 | 383.9 | +3.6% | 561.0 | 0.68× |
| Mistral-7B / C2 | 88.9 | 96.7 | +8.7% | 115.4 | 0.84× |
| Mistral-7B / C8 | 256.1 | 256.9 | +0.3% | 349.0 | 0.74× |

첫 토큰과 스트리밍 지연의 차이는 다음과 같습니다. TPOT는 요청별 평균 토큰 간격의 p99이며 개별 토큰 ITL p99와 구분합니다.

| 모델 / 동시성 | TTFT p95 전→후 ms | stock TTFT p95 ms | TTFT p99 전→후 ms | TPOT p99 전→후 ms | stock TPOT p99 ms | E2E p99 전→후 ms |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B / C3 | 55.70 → 56.47 | 47.32 | 71.43 → 69.51 | 19.20 → 13.78 | 10.53 | 2476.60 → 1799.57 |
| Qwen3-4B / C8 | 134.31 → 150.21 | 86.14 | 149.72 → 150.35 | 17.52 → 17.54 | 11.43 | 2300.15 → 2285.13 |
| Mistral-7B / C2 | 86.10 → 86.31 | 76.45 | 86.16 → 86.41 | 21.77 → 19.90 | 16.51 | 2838.19 → 2601.66 |
| Mistral-7B / C8 | 216.21 → 216.31 | 140.72 | 271.40 → 271.67 | 27.15 → 27.16 | 17.73 | 3373.19 → 3372.49 |

수치 검증은 허용오차 없이 수행했습니다. 실제 가중치 136개의 독립 native eager/CAPI 조건 9,792개, 입력을 바꾸는 Graph 재실행, 두 모델의 HF 기준 전체 어휘 출력 104행씩, 실제 1~32행의 Graph 재사용을 검사했습니다. Driver API에서 Qwen의 새 커널 72개, Mistral의 Query·출력 새 커널 64개를 확인했고, 모든 행 수에서의 캡처 2304개/2048개도 확인했습니다. 추가로 두 모델의 128-step 장문 실행에서 512-token 경계를 넘는 모든 전체 어휘 BF16 출력을 기존151 eager와 비트 단위로 비교했습니다. 이 추가 장문 비교는 native 전후 기준이며 새로운 HF logits를 생성한 검사는 아닙니다.

CPU release 테스트 517개가 통과했습니다. 별도 GPU 실행과 서버 검증에서 전체 응답 값 834개를 확인했고, 연결 중단 2건 뒤 재요청과 KV/할당 복구, 소유 PID, 관측 메모리 예산을 검사했습니다. SmolLM2와 Qwen2.5의 기존 모드도 전후 일치했습니다. Qwen2.5에 원래 존재하던 HF 차이는 별도로 보존했으며 이번 변경으로 HF 일치를 새로 달성했다는 뜻은 아닙니다.

vLLM 출력의 HF 일치 횟수와 출력 토큰 수는 원본 측정 결과에 별도로 기록했습니다. 속도 표는 동일 입력·설정의 실행 비교이며 일반적인 모델 품질 순위를 뜻하지 않습니다. 일부 첫 토큰/스트리밍 지연 지표의 증가를 원본 그대로 유지했습니다. 이 범위의 1회 비교로 5회 반복 전체 행렬이나 모든 tail의 무회귀를 주장하지 않습니다.

Attention의 QK·Softmax·PV 수학은 유지했습니다. 요청 길이에 따라 바뀌는 부분은 eager로 실행하고 전후 레이어 작업을 Graph로 재사용합니다. 직접 paged KV 읽기와 더 큰 배치의 linear 개선은 후속 가설로 남아 있습니다. 다양한 모델·동시성에서 stock vLLM 이상이라는 전체 목표는 아직 미달입니다.

실행 시 기존 필수 설정과 함께 선택합니다:

```text
--numerics-profile owned-hf-eager-bf16-v1
--execution-completion iteration-batch
--owned-decode-graph segments-v1
--owned-decode-batch exact-smallrows-v5
```

실측 바이너리: `/data/riley-multimodel-parity-20261017/riley-candidate-v164` (ssh ai-assistant), SHA256 `c034c7c8d36e6a3ed0cdc9e0c5dd0eee1239e19dd6129618dca3f860690306e7`. 배포·커밋·푸시는 수행하지 않았습니다.

검증 근거: [최종 영수증](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v164-actual-integrated-final-receipt.json), [전체 값·지연·자원 감사](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v164-pilot-independent-audit.json), [실제 Graph 커널](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v164-driver-controller-result.json), [장문 전체 어휘 비교](/Users/psyche/PycharmProjects/riley/artifacts/multimodel-parity-20261017/v164-long-context-controller-result.json).
