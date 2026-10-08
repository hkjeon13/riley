# V55 decode mask / loop prototype

V54 serving은 V52보다 C16/C32 natural throughput −6.21%/−2.35%로 손해여서 거절했다. V54 trace의 C32 natural mixed attention은 52.307→45.015µs로 빨라졌지만 decode values는 9.730→15.267µs로 느려졌다. V54 binary는 비교용으로만 보존한다.

다음 세 prototype을 V52 token-major와 V54 packed 기준에 함께 비교한다. 아직 application source는 V54 frozen commit `636376e67fdcc508552371628808fc21d2378c44` 그대로다.

1. Valid page 안의 aligned BF16 pair를 먼저 읽고 integer mask로 future words를 0으로 만드는 branchless mask.
2. 완전한 K16 tile은 무조건 pair load, 마지막 부분 tile만 branchless mask.
3. Branchless mask와 K16 순차 loop를 결합해 네 부분의 준비·조건 검사 상태를 줄임. MMA/softmax의 연산 순서는 보존.

원격 `/tmp/riley-opt-260912/decode-mask-v55`, 준비 스크립트 `prepare_decode_mask_v55.py` 및 `expand_mask_v55.py`. 세 seed, 일곱 context 모드, 여덟 active-row 조건(0/33 invalid 포함), finite/visible NaN/visible Inf/future NaN 네 모드, 네 후보 비교로 2,688 cases다. Score/output/padding byte 비교 및 memcheck 오류 0을 확인했다. Future load는 물리 page 경계 안의 정렬된 pair로 제한되며 integer mask가 future NaN/Inf word를 정확한 0으로 지운다.

진행 중인 후속 controller는 local exec session31738 (`after_round60.py`)이다. Racecheck PID1187006, probe PID1187015이며 완료 후 400 timing records(20조건×5variant×4역순, 10 warmup+100 replay)를 수집한다. `analyze_decode_mask_v55.py`를 준비했다. 분석 전 선택/통합/성능 채택은 없다. GPU 작업은 순차 실행하며 Blender는 계속 종료 상태다.

Round60 controller session97540은 정상 종료했다. 첫 export는 저장하지 않은 stdout log를 요구하여 실패했고, 실제 종료를 담은 execution receipt를 사용하도록 수정했다. 후속 controller의 첫 session51867은 이 export 오류로 종료했으며 재개 session31738이 export와 V54 profiling을 완료했다. Benchmark를 다시 실행하지 않았다. V54 profile 288개 요청의 기준 출력이 모두 일치했다.

## 완료 및 통합 갱신

Racecheck도 오류·경고 0으로 완료했고 400개 timing record를 분석했다. Variant3(full K16 fast path와 partial-tail branchless mask)이 최선이다. V54 대비 모든 20조건에서 개선했다. V52 대비 C32 context128/398/1024는 각각 −5.53%/−9.58%/−13.04%, context4096은 +3.28%다. 긴 문맥의 작은 row 수는 여전히 V52보다 느리므로 일반적인 우위를 주장하지 않는다. 단순 branchless mask 또는 K16 loop만 적용한 다른 후보는 선택하지 않았다.

선택한 두 연관 개선을 V7 decode 값 kernel에 함께 통합했다. Mixed attention과 V 저장 코드는 V54와 동일하며 별도 pool을 추가하지 않았다. V55 실제 모델 16개 회귀, V7 memcheck 3개, HTTP 37개 응답, CPU/GPU fallback 22개 응답 일치가 통과했다.

Frozen commit `a33cdc8488122eebefad090cc493dec370193e49`, binary SHA256 `897c6ec9118feecb4121aef82e9aeb117a50a09a18f9b54ac7076ab10bab0a81`, build log SHA256 `2a4d96d238938cd9a0e2b3980fd9c52770a1b85e647e3c66b2248737b7f84eae`. 이는 비교용 freeze이며 성능 채택이 아니다.

Prototype 51파일을 `raw/mask-v55`에서 검증했다(archive SHA256 `052569307142c58242c018f492778530a2d759c33eec3656ce207f4ebecdefd2`). 통합 31파일도 `raw/integration-v55`에서 검증했다(archive SHA256 `b297b4354250f63cf6606d0158918ab7e5762c23e64d562615ca15ba33005584`).

Round61은 V51 anchor/V52 previous/V54 packed/V55 new/vLLM의 40개 lanes, warmup96/retained384, C16/C32 fixed/natural 두 역순으로 진행 중이다. Controller local session81317, remote PID1240127(parent1240104), stdout `serving-round61-controller.log`. 이 작업 완료 전 GPU 실험을 겹치지 않는다. V55 trace 스크립트는 준비만 했다. 이전 session31738과 qualification session39179는 정상 종료했다.
