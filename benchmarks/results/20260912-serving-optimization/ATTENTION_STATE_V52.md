# V52 mapped attention state batch

V51 Round58은V50대비네조건에서처리량/TTFT/TPOT/tails를개선했다. C16fixed는vLLM대비처리량+33.81%,TPOT−5.18%지만natural및C32TPOT격차가남아전체목표미달성이다. 다음비교기준은V51이다. [V51결과](V51_SERVING_RESULTS.md).

V51 C32natural trace에서mapped attention 평균55.261µs가가장큰단일kernel이며prefill/mixed graph span비중41.05%다. CUDA속성및occupancy진단(syntheticP398+31decode)에서127registers/staticshared20,480bytes/SM당최대4activeblocks와4warps를확인했다. Nsight Compute는ERR_NVGPUCTRPERM으로hardwarecounter를읽지못했다. 이결과는정적상한이며achieved occupancy/warp stall을측정한것은아니다. 시스템권한설정은변경하지않았다.

두연관개선을4variant로비교한다. Variant0은V51baseline,1은실제TileRows8에맞춘scores/exps/probs와상위미사용query연산제거,2는score가더이상필요없는시점에exponential저장공간으로재사용,3은둘다다. K16MMA/softmax누적/BF16round/causal및비유한값fallback연산순서는유지한다.

원격 `/tmp/riley-opt-260912/attention-state-v52`에prototype을분리했다. Application source는V51commit에그대로있다. 초기240scenarios(2seeds×6patterns×5ownercounts×4finite/NaN/Inf/causal예외)에서각각4variants를기존roworacle과비교했고,추가invalidowners0/33두case도통과했다. 모든output/padding byte일치,memcheck오류0이다. 기존harness에서계승된로그의four_geometries필드는이번에는동일grid의네variant를의미한다.

cuobjdump기준sharedmemory는baseline20,480bytes,variant1 10,240,variant2 12,288,variant3 6,144bytes이며모두127registers/stack0/local0이다. Racecheck진행중이며완료뒤480timingrecords(30patterns×4variants×4역순,10warmup+100replay)를수집한다. Controller관찰session64846,remote racecheckPID891054, parentbash889949. 관찰timeout만으로재시작하지않는다.

추가 `boundary.cu`는context4096,8192physicalpages,owner1/4/32,3patterns,finite/NaN/Inf/causalcase를준비했으나아직compile/실행전이다. 초기controller종료후boundary검증과memcheck를실행하고timing결과로후보를고른다. 이후실제model/HTTP/serving검증이필요하며sharedmemory축소만으로성능채택을주장하지않는다. Blender는종료상태유지.


## 초기검사·timing·4096경계 완료 및 통합

초기racecheck0errors/0warnings,480timingrecords완료. P398+decode패턴에서variant3은9.36~13.06%개선,작은prefill혼합다수에서는최대43.21%개선이다. 일부one-token패턴에서약0.2~1.5%손해가있으므로serving검증으로판단한다. Context4096의36scenarios×4variants및invalidowner2case도exact/guards/memcheck오류0으로통과했다. 새CUDAoccupancy API결과127registers/shared6144bytes/SM당최대14blocks/warps로정적상한이4→14다. 실제achieved occupancy는counter권한이없어측정하지못했다.

Variant3을기존mixed attention헤더에통합했다(단일파일26line변경). V51의decode/fusedprefill경로는유지한다. 전체servingbuild및실제모델GPU16회귀가통과했고, controller`qualify_attention_v52.py`(관찰session73249)가V7modelmemcheck→HTTP/fallback을실행중이다. 이전prototype/race/timing/boundarycontroller는모두terminal이다. 아직sourcecommit/frozen/serving검증전이다.

[Prototype분석](raw/attention-v52-prototype/attention-state-v52/analysis.json), [Manifest](raw/attention-v52-prototype-manifest.json). 63파일SHA256검증완료; archive `45760b4ad4c0dd883f61096c278f3a956a36b319387d5a049408585232a849ff`. 실패한NCUcounter시도로그와성공한CUDAoccupancy결과를모두보존했다.


## V52 frozen 및 Round59 시작

모든통합검증terminal: 실제모델GPU16회귀,V7full/compact/partial3modelmemcheck오류0,HTTP37기준일치/invalidbound/disconnect회복,CPU/GPU-greedy22응답씩orderedfallback일치. Commit `06302d8d8396b8f2f4996fec8595bbfa1dcd7450`, binary SHA256 `d24783e7aab9e5ffbf7489e720a54122b967c5274131b385c3c0f405b3fbe07d`, buildlogSHA256 `c64d74c704c10ced7c2a7e3d2505ba82780cb3e4a214451d29fe3b2441ae876b`. 원격checkoutclean확인후freeze했다.

[통합patch](raw/integration-v52/attention-v52.patch), [통합manifest](raw/integration-v52-manifest.json). 33파일SHA256검증완료; archive `f99ca27a299aa1a06421caf76c6a2e06124d62875c1abc02f41835bee699f509`.

Round59은V51/V52/vLLM×C16/C32×fixed/natural×2역순,각96warmup+384retained,24lane이다. 두Riley V7/GPU-greedy/budget512/fixedchunk128/natural512동일. Controller`serving_screen_round59.py`,log`serving-round59-controller.log`,관찰session11224. 측정중다른GPU작업/무거운build/export금지. 끝난뒤분석/export/V52trace를진행한다. 목표는여전히미달성이며이측정으로V52의serving효과를판단한다.


## Round59 및 V52 trace 완료

9,216요청실패0,Riley6,144기준일치,별도trace288기준일치. C32fixed개선과C16일부손해가함께관찰되어V52전반승격은하지않는다. V51을일반기준으로보존하며V52는후속실험후보로유지한다. [최종결과](V52_SERVING_RESULTS.md).
