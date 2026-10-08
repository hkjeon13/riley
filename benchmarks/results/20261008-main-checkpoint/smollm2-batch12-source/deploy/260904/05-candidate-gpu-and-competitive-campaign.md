# RUN-B — candidate GPU qualification과 competitive campaign cards

**목적:** implementation 결과를 candidate-bound GPU evidence로 바꾸고, Tier D 진단과 Tier C/S
경쟁 판정을 서로 다른 실행 카드로 수행한다. 이 문서는 운영 실행 계획이며 코드 PR과 섞지 않는다.

## 모든 실행의 공통 조건

- Q05 no-GPU acceptance와 Q06 explicit GPU/Docker/capture authorization
- 별도 clean worktree와 immutable evidence root

하나라도 없으면 `blocked: missing-authority-or-input`이다.

## RUN-B01 — candidate freeze와 GPU regression qualification

**한 가지 목적:** graph/runtime 변경을 포함한 한 candidate를 freeze하고 C03/C04 및 Gate E를 실제 실행한다.

1. source archive/revision, build image/ELF/toolchain, model weights/tokenizer, runtime options,
   GPU UUID/driver/CUDA/clock/power를 one candidate manifest로 묶는다.
2. actual Gate E producer/semantic/lifecycle receipts를 create-only evidence root에 기록한다.
3. C03-B GPU routing corpus와 C04 GPU parity/allocation/5-pair non-regression을 실행한다.
4. fixed-37 SHA와 stale cosine gate는 별도 corrective PR에서 baseline/candidate truth를 재수집한다.
   threshold 완화는 금지한다.

**완료:** checker가 same candidate closure를 재검증하고 qualification verdict를 낸다.

## RUN-B02 — Tier D post-graph diagnostic campaign

**선행:** B01 qualified candidate.
**목적:** SmolLM2 `c1/p128/o32`에서 current Riley/vLLM의 matched directional gap과 post-graph 병목을 수집한다.

추가 입력으로 current vLLM competitor version/tag, wheel/source hash, dependency lock과 Tier D
model/tokenizer/request manifest를 immutable pin한다.

- fresh process 5회 이상, AB/BA, warmup 5 제외, 30 measured request
- identical model/tokenizer/token IDs, greedy/EOS/cache policy, GPU identity
- engine-only와 HTTP streaming을 별도 series로 기록
- thermal/contention/process/environment receipt와 실패 sample을 append-only journal에 보존

Tier D는 K00 candidate 선택용이며 M4/M5 승리 근거가 아니다. B02 뒤 [06](06-profile-selected-kernel-prs.md)의
K00을 수행한다.

## candidate mutation rule

K01~K03 중 어떤 코드라도 채택하면 B01 candidate는 폐기하지 않고 historical diagnostic으로 보존하되,
새 revision을 **B01에서 다시 freeze/qualify**한다. 새 candidate는 B02 sanity cell도 다시 통과해야 한다.

## RUN-B03 — Tier C competitive latency campaign

**선행:** K-track decision 후 final candidate의 B01/B02 재실행.

실행 전에 Tier C model/tokenizer/request slot과 vLLM lane을 concrete immutable input으로 materialize한다.

- 최소 0.5B, 1~3B, 7~8B dense Llama/Qwen-compatible BF16 model slot
- concurrency 1/2/4/8, prompt 128/1024/4096, output 32/128/512의 contract-required cells
- request identity/token correctness, TTFT/TPOT/ITL/E2E/throughput/VRAM/KV capacity

**M4:** required cells TTFT/TPOT p95 ratio 각각 `<=1.03`, failure/token mismatch 0.
**M5:** primary/required geometric mean TTFT/TPOT p95 `<=0.90`, peak VRAM `<=1.05`.

이 실행이 C14의 첫 RTX 4090 multi-model lane이다. 같은 model-size matrix를 S05에서 중복 실행하지 않는다.

## RUN-B04 — Tier S service SLO와 closed report

**선행:** B03 correctness.
**목적:** open-loop arrival rate, c8/c16/c32, cancellation/disconnect/backpressure에서 SLO goodput과
reliability를 측정하고 최종 M4/M5 report를 close한다.

arrival schedule과 model별 SLO threshold는 실행 전에 pin하며 결과를 본 뒤 변경하지 않는다.

M4 Tier S goodput ratio는 `>=0.97`, M5는 `>=1.10`이다. raw journal, materialized lanes,
candidate/request/model/environment identity, failure records를 checker가 재검증해야 한다.

## 완료 판정

historical vLLM 0.27.1, separate campaign 또는 Tier D 결과는 directional comparison이다.
B03/B04가 같은 final candidate에서 closed report를 만들었을 때만 M4/M5를 표현한다.
