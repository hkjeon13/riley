# Prefill Gate/Up 통합 + decode attention v16C 검증

2026-10-21 KST. v15C 대비 독립 빌드·원래 정확도 gate·C16/C32 정상 비교를 수행했다. **판정: 미채택.** attention만 바꾼 v16B와 서로 다른 캠페인의 수치를 차감해 Gate/Up만의 이득이라고 주장하지 않는다. 이번 after는 두 변경을 합친 후보이고 before는 동일 v15C다.

## 큰 구조 변경과 지원 범위

기존 separate Gate/Up 두 GEMM을, 원래 native one-chain family와 일치하는 prefill M65..130에서 N19456 merged GEMM 하나로 계산한다. cold M130 별도 plan, BF16 input/output·FP32 accumulate, cuBLASLt120405의21/20/11/0/0/0/0 recipe다. decode plan31/15/17/0/0/0/0과 workspace를 분리하며, 기존 merged weight payload를 재사용한다. 실행 중 CUDA 할당이나 추가 checkpoint dtype 변환을 도입하지 않는다.

mode10·fused SiLU·원래 merged weights/plan이 있고, decode prefix 이후 정확히 한 prefill group65..130행일 때만 선택한다. prefix MLP Graph와 Down·residual의 원래 순서를 유지하며 새 merged raw를 원래 BF16 경계에서 SiLU/multiply한다. 입력·가중치·출력 view256-byte 정렬은 제출 전에 검사한다. 여러 prefill group이 합쳐진 큰 M이나 미검증 shape는 기존 경로다. 별도 큰 batch 연구 없이 범위를 넓히지 않았다.

v16B의 QK native-order warp reduction과 PV4-warps/CTA도 포함한다. 원래 BF16 경계·softmax·KV page layout·supported context는 그대로다. attention만으로는 v16B 정상 처리량 +1.25%/+1.35%와 tail 경고가 남았으므로 combined 후보는 새로 검증했다.

## 실제 가중치 구성요소 선별

Qwen3-4B layer0 safetensors에서 추출한 BF16 Gate/Up과 Down weight를 해시로 고정했다. 원래 finite seed의6변형, 모든 M65..130에서 Gate/Up396개와 Down396개, 합계792 raw byte 비교가 일치했다. 이 검사는 checkpoint의 전체 hidden activation을 재현한 whole-model 증명이 아니다. Down 입력은 hidden seed를 반복해 만든9728차원 finite 입력이다.

Gate/Up GEMM-only 시간은 M65 103.435→100.687µs, M115106.609→103.978µs, M128107.625→104.535µs, M129168.134→113.328µs, M130168.277→113.317µs였다. 기존 separate pack이나 activation의 추가 이득을 이 숫자에 포함했다고 주장하지 않는다. M129/130 개선을 모든 행·서버 처리량의32% 개선으로 바꾸지 않는다.

Down의 기존 pinned Lt6/24/15/5/4/1/0은65..128에서 느렸고129/130에서 거의 같았다. 추가 tile/stage/splitK/reduction/swizzle 탐색의 exact shortlist3개도, 대표 tile18의 M65 54.344→53.432µs, M11557.333→56.287µs, M12857.651→56.964µs 정도이며12965.110→72.791µs/13065.172→73.014µs로 악화했다. **Down runtime 변경은 채택하지 않았다.** 전부 raw 결과를 남긴다.

## 같은 캠페인 정상 C16/C32 비교

아래는 각 엔진·concurrency별5개 새 프로세스 run metric의 중앙값이다. 정상30 arms에서 profile/preload를 사용하지 않았고 완료 표본 제외0개다. 첫 C16 before/after2개는 완료됐지만 첫 stock 시작이 누락된EFS ipc7 디렉터리로 실패했다. 디렉터리만 만들고 원래2개를 보존한 채 stock r0, 나머지 C16과 C32를 이어 실행했다. runner 종료 인덱스 의미 때문에 마지막 C16 r4는 C32 이후 별도3개 arms로 채웠다. source/binary/runner/원래 argv hash는 그대로이고 continuation argv를 따로 해시했다. 첫 triple 사이에 이 중단 시간이 있다는 한계를 남긴다.

| C | v15C tok/s | combined 후보 tok/s | 변화 | vLLM tok/s | 후보/vLLM |
|---:|---:|---:|---:|---:|---:|
| 16 | 738.804 | 760.250 | +2.90% | 1146.853 | 66.29% |
| 32 | 975.260 | 991.783 | +1.69% | 1676.802 | 59.15% |

| C | TTFT p95 기존→후보 | TPOT p95 기존→후보 | TPOT p99 기존→후보 | E2E p95 기존→후보 |
|---:|---:|---:|---:|---:|
| 16 | 80.973→81.350 | 20.239→19.843 | 21.403→20.760 | 671.225→649.325 |
| 32 | 252.533→254.537 | 31.761→31.358 | 36.809→36.907 | 1016.215→995.724 |

ms. TPOT은 요청별 평균 output-token 지연의 percentile이며 token-level ITL이 아니다. paired TTFT/TPOT p95/p99 +2% 경고:

- C16: r1 ttft_ms_p95 +3.48%.
- C32: r0 tpot_ms_p99 +2.63%; r3 tpot_ms_p99 +2.71%; r4 ttft_ms_p95 +3.41%; r4 ttft_ms_p99 +2.90%; r4 tpot_ms_p99 +4.38%.

## 정확도·자원·재현

독립 source937/빈 target, CPU524 pass/40ignored, 원래 native/HF full-vocabulary BF16 4,480파일 literal equality를 검증했다. 실제 GPU 계약6개를 실행했다. 기존 alignment120경우에 추가로 merged Gate/Up6행수×3finite입력×per-op/command-batch=36 raw cases, output suffix sentinel·alignment preflight 이후 owner 재사용·다른 stream 조기 read 거절·hot allocation0을 검사했다. eager104HFrows와 all-R Graph104rows, natural128 body1024rows, mixed output1024rows/no-output992rows 범위이며 전체104×32 Cartesian이나 새 early-EOS coverage로 확장해 주장하지 않는다.

정확도 검사와64warm 뒤64keep 요청 구간만 Nsight start/stop으로 포착한 별도 before/after C16/C32 profile로 실제 qualified merged dimensions의 NVTX range와 Graph attribution을 확인한다. 정상 처리량과 분리하고 broad trace의 도착·batch work 차이를 고정 연산량 speedup으로 환산하지 않는다. 실제 binary/source/argv/runner hash와 stock FA2/FULL_AND_PIECEWISE startup을 감사한다.

Qwen3-4B BF16 greedy, budget576, maxseq1024, KV blocks=C×64, prefixoff,12fixed prompts115..130tokens, 균형8/16/24/32output limits,64warm192keep/arm=3840outputtokens. stock vLLM0.27.1+cu129/Torch2.13+cu129 FA2는 Riley correctness reference가 아니다. 서버와 부하 CPU pinning을 분리했지만 전체 노드 CPU·I/O 격리의 증거는 아니다.

GPU6 새 외부 작업은 중단하거나 허용 정책을 넓히지 않았다. finite GPU 작업은 GPU7에서 직렬 실행했고 원래 GPU6/7 holder PID2027914/2027915, start ticks6083215119/6083215121은 유지했다. 모델·스크립트·빌드·scratch는 EFS이며 사전 허용된4개 C02 private fixture만 `/tmp`에서 실행 후 삭제했다. GPU6 거절과 controller 동시 진입 거절 등 pre-child 실패·수정도 raw 로그에 보존한다.

[원자료 ZIP](../benchmarks/evidence/20261021-mlp-prefill-v16c.zip) · [검증 목록](../benchmarks/evidence/20261021-mlp-prefill-v16c/README.md) · [전체5회 결과](../benchmarks/evidence/20261021-mlp-prefill-v16c/final-statistics.json) · [source/binary lineage](../benchmarks/evidence/20261021-mlp-prefill-v16c/source-lineage.json).
