# V51 prefill projection 및 gate/up fusion batch

V50 profile에서 fixedC32 graph span의77.33%가 prefill/mixed이고, gate/up 각17.509µs, down19.147µs가30layer마다 반복된다. 이에 두 연관 개선을 함께 구현/비교했다: gate/up 입력을 공유하고 SwiGLU까지 fusion하여 중간BF16buffer traffic/launch를 줄이는 방식, 그리고 두/네M16 tile에 동일 weight fragment를 재사용하는 방식이다. 각 K16 누적 및 BF16 chunk-round 순서를 유지한다.

첫 capacity1024 prototype은864case 정확성/memcheck 통과했지만 실제 serving capture512와 다르므로 예비 자료로만 보존한다. 수정 검사는 capacity512/1024, rows0/1/7/15/16/17/31/32/33/73/128/256/398/512/1024/1025, seeds1/17/99, 다섯projection geometry 및 fusion을 포함한다. 1,728개 비교에서 모든출력/padding byte 일치, 입력/가중치 불변, memcheck 오류0이다. Timing은 rows<=512에서capacity512, rows1024에서capacity1024,4역순×100replay+10warmup,672records다.

| Gate/up+SwiGLU rows | 기존 µs | Fusion M16 µs | 변화 |
|---|---:|---:|---:|
| 16 | 17.567 | 8.688 | -50.55% |
| 32 | 17.618 | 8.688 | -50.68% |
| 128 | 18.948 | 9.472 | -50.01% |
| 256 | 29.781 | 14.897 | -49.98% |
| 398 | 48.148 | 22.326 | -53.63% |
| 512 | 53.949 | 30.576 | -43.32% |
| 1024 | 111.416 | 56.457 | -49.33% |

32/64행 weight reuse는 작은rows에서 성능이 크게 악화하여 채택하지 않았다. 예를들어 down128행은M32+54.0%,M64+355.9% 느리다. 선택한 구현은M16 gate/up fusion과SwiGLU fusion이다. 모든커널측정결과를 보존하며 이 결과를 serving 성능 주장으로 사용하지 않는다.

원격 격리 checkout의 `prefill_fused_gate_v51.cuh`를 `mixed_model_v49.cuh`의 capacity>1 && tiled 경로에만 연결했다. 기존M1/nonpacked 및V5/V6 모델은 유지한다. Build dependency와graph source hash에 새헤더를 포함했다. 전체CUDA build와실제모델GPU회귀16개가통과했다. Controller `qualify_prefill_v51.py`(관찰session35737)는 V7 model memcheck 및 HTTP/fallback을 순차실행 중이다. 통합source는 아직uncommitted이며 frozen/serving비교는 검증후진행한다. Blender는종료상태를유지한다.

[Matched 분석](raw/prefill-v51-prototype/prefill-projection-v51-matched/analysis.json), [증빙 manifest](raw/prefill-v51-prototype-manifest.json). 21파일 SHA256 검증 완료; archive `24bde1d34ca117c4bec4cddd4a1c04b3fe5965607417715e09e08085fe7967e5`.


## V51 frozen, Round58 실행 중

전체modelGPU16회귀, V7 full/compact/partial3memcheck 오류0, HTTP37 기준일치 및 invalid-bound/disconnect회복, CPU/GPU-greedy22응답씩 ordered fallback완전일치가통과했다. 원격commit `a0536a5a3d7aebf721b7a41d9bd39897c60ddde5`, binary SHA256 `03262d945a3c2f31589d5525318b175c43e38001a45be7dcc9cceea22288018d`, build log SHA256 `b07dc38b613864484737db17364b184fb6140d21c326f7f8dedea9858db397c7`. 원격checkout clean 후freeze했으며 이전V50 바이너리는 보존했다.

[통합 patch](raw/integration-v51/prefill-v51.patch), [통합manifest](raw/integration-v51-manifest.json). 33파일 SHA256 검증 완료, archive `d182cf3575a9044d72a6db927fb4e8620dab8c35d61d9c8784409afdc9e9a718`.

Round58은 V50/V51/vLLM ×C16/C32 ×fixed/natural ×2역순의24lane, 각각96warmup+384retained다. 두 Riley V7/GPU-greedy/budget512/fixedchunk128/naturalchunk512 동일. Controller `serving_screen_round58.py`, log `serving-round58-controller.log`, 관찰session20981. 측정 중다른GPU작업이나무거운build/export를시작하지않는다. 완료후분석/export, 별도V51trace로다음병목선정. 목표달성은아직미검증이다.
