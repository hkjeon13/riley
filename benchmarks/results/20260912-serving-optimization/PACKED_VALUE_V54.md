# V54 packed V cache 및 vector load prototype

V53출력열분할/warp배치의네후보는모든30timing조건에서손해가있어거절했다. 다음batch는attention의QK/softmax작업량을증가시키지않고V읽기배치를변경한다. V52naturalC32 mapped attention52.307µs가여전히주요비용이다. Application source는V52clean상태이고V51일반기준을보존한다.

기존Vpage/head내16token×64dimension배치를MMA Bfragment의8dimension×16token타일배치로바꾼다. Packedoffset은 `(page*3+head)*1024+(dim/8)*128+(tokenWithinPage/8)*64+(dim%8)*8+tokenWithinPage%8`이다. Variant0은기존V,1은packedVscalarloads,2는packedV32-bitpairloads다. Partial마지막token의상위BF16word는zero mask하여기존causal/NaN/Inf처리를유지한다. Kcache및QK계산/orderedMMA/softmax/BF16round는변경하지않는다.

원격 `/tmp/riley-opt-260912/packed-value-v54`의prototype은기존V와CPUprepack한V를별도로보유한다. 원래V의roworacle과packed후보의모든output/padding을비교한다. 초기240scenarios×3variants및invalidowner2case에서byte일치했고memcheck오류0이다. Nonfinite값도두배치의동일논리위치에주입/복구한다. Build의unusedpage_base/base경고는packedaddress로교체한뒤남은기존변수이며build성공을확인했다.

Controller관찰session31095,remote racecheckPID1033304(parentbash1032213)가실행중이다. 이후360records(30patterns×3variants×4역순,10warmup+100replay)를수집한다. `analyze_packed_value_v54.py`,context4096/8192physicalpages의`boundary.cu`, `qualify_boundary_v54.py`를준비했다. 경계검사는아직실행전이다.

이primitive에는prepack/실제KVwrite비용이포함되지않는다. 기존V와packedV가같이있는실험이며whole-modelcache환경도다르다. 유망하면V7의prefill/decodeKVwrite및reader를함께통합하고cache배치/graphidentity/ownercontract,실제모델및serving을검증해야한다. 아직application통합/commit/frozen/성능채택은없다. 준비된Round60은폐기된V53binary를가리키므로실행하지않으며다음후보가선택되면V51/V52/new/vLLM구성으로갱신한다. Blender는종료상태유지.

## 2026-09-13 진행 갱신

초기 controller는 정상 종료했다. Racecheck 오류 0이며 360개 timing record가 완성됐다. Packed scalar는 22–163% 느려 거절했다. Vector pair는 30개 조건 모두 0.60–18.91% 빨랐다. P128+decode는 owners 4–32에서 약 8%, 네 P128 prefill+decode는 약 5–7.5%, P398+decode는 3.28–6.45% 개선됐다. 이는 실제 쓰기 비용이 제외된 커널 결과다.

Context 4096 경계 검사 36 scenarios×3 variants 및 invalid-owner 2 cases가 통과했고 memcheck 오류는 0이다. 원격 격리 checkout에 V54를 통합했다. Mixed prefill writer와 V7 전용 decode writer가 같은 packed V를 직접 기록한다. V7 mixed/decode reader는 vector pair를 읽으며 추가 V pool이나 hot-path 변환은 없다. V5/V6 decode writer는 기존 template specialization을 사용한다. 새 helper는 Cargo rebuild 의존성과 graph catalog source digest에 포함했다. V7 mixed buffer 선택은 owned-session 생성의 내부 설정이고, executor/stream/scratch 부모는 세션이 소유한다.

Server build, 실제 모델 GPU 회귀 16개, V7 모델 memcheck 3개, HTTP 37개 응답 및 CPU/GPU fallback 22개 응답의 일치가 확인됐다. Writer 추가 검사의 일반 실행 84개에서도 Q/K byte 일치, unpack한 V 전체 pool 일치, 비활성 영역과 다른 페이지의 보존을 확인했다. Writer sanitizer와 별도 decode primitive 검증이 진행 중이다. 아직 candidate freeze와 serving 비교 전이며 성능 채택하지 않았다. 위의 이전 '실행중/통합전' 문장은 이 갱신으로 대체한다.

Writer 84개와 decode 168개 추가 검사, 각각 memcheck/racecheck 오류 0으로 완료됐다. Decode 160개 timing record는 20조건×2variants×4역순이다. 새 packed decode는 모두 2.81–54.32% 느렸다. 두 decode kernel 모두 register 40, shared memory 2,304 bytes, stack/local 0으로 같았다. SASS의 분기·주소 연산은 증가했으나 실제 stall counter가 없어 인과관계는 미확정이다. NCU 권한은 변경하지 않는다.

비교용 frozen V54: commit `636376e67fdcc508552371628808fc21d2378c44`, binary SHA256 `14f772200199121fb05127948129a5be163b4c97f610638a5b6adf0ddcbb6c52`, build log SHA256 `27a1e769cbcd43bfe2d4b25331a8983765280b9ce4865b401a4cc7df3e303c93`. 이는 성능 채택이 아니다. 통합 evidence 31개를 `raw/integration-v54`에 가져와 각 파일 SHA256/size를 검증했다. Archive SHA256 `95edfbd83b9b8caa11527291bc2152312ce1f826fdfaffa04b71b35c3b3e7866`.

Round60은 V51 anchor/V52 previous/V54 new/vLLM의 32개 lane으로 시작했다. 기존 V53 참조를 V54로 수정했고 두 역순, C16/C32, fixed/natural 각각 warmup 96/retained 384를 사용한다. 시작 전 GPU가 48°C 이하가 되도록 기다렸다. 현재 controller session 97540, remote PID 1103056. 이 측정 완료 전 다른 GPU 작업은 실행하지 않는다. V55의 mask/loop 비교 prototype 세 개는 소스만 준비했으며 실행·통합·채택 전이다.
