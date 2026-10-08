# V53 attention column parallelism batch

V52 Round59은C32fixed개선과C16일부손해가함께관찰되어일반승격을하지않았다. V51일반기준과V52실험후보를모두보존한다. V52trace에서C32natural mapped attention52.307µs가남아있고,C32fixed도19.256µs다. 이전primitive에서single-query혼합은state축소만으로거의개선되지않았다. 다음가설은한warp가8outputcolumn group을직렬처리하는비용을분할하는것이다.

Batch는(1)V가중합의출력열을2/4그룹으로나누고각warp의accumulator크기를줄이는변경,(2)분할warp를같은CTA에배치하여CTA dispatch를줄이는변경을비교한다. QK/softmax중복증가가발생하므로성능을가정하지않는다. Warp별sharedstate를분리하여동일CTA에서도기존orderedMMA/softmax/BF16round를유지한다.

Variant0=V52baseline,1=Columns2/Warps1,2=Columns4/Warps1,3=Columns2/Warps2,4=Columns4/Warps4. Gridz=Columns/Warps,threads=32*Warps. 각warp는해당출력열만쓰며causal/nonfinitefallback도같은열영역을계산한다. 원격 `/tmp/riley-opt-260912/attention-columns-v53`에prototype을분리했고application source는V52clean상태다.

초기240scenarios×5variants(2seeds,6patterns,owners1/4/8/16/32,finite/NaN/Inf/causalexception)에서기준output/padding이byte일치했고추가invalidowner0/33도통과했다. Memcheck오류0이다. cuobjdump의registers/staticshared: baseline127/6144B,2/1=103/6144,4/1=95/6144,2/2=103/12288,4/4=95/24576; stack/local0이다. 이정적자원수치가serving이득을증명하지않는다.

Racecheck controller관찰session9543,remotePID976035(parentbash974554)가실행중이다. 이어600timingrecords(30patterns×5variants×4역순,10warmup+100replay)를수집한다. `boundary.cu`는context4096,8192physicalpages,owners1/4/32,3patterns,4exception조건으로준비했으나compile/실행전이다. `analyze_attention_columns_v53.py`도준비했다. 기존controller종료후분석/경계검증을수행하며GPUjob을중복시작하지않는다. 채택시실제모델/HTTP/serving검증이필요하다. Blender는종료상태유지.


## V53 거절 — 모든 timing 조건에서 손해

Racecheck0errors/0warnings,600records완료. 네후보모두30조건전체에서V52baseline보다느렸다. P398패턴에서는+18.27~87.88%느리고일부작은query다수조건은+180.45%까지악화했다. 따라서어떤후보도통합하지않는다. Register감소가중복QK/softmax/dispatch증가를상쇄하지못한것으로해석하며,정확한hardwarestall은counter권한이없어측정하지못했다.

4096boundary및serving은준비만했고실행하지않았다. 성능단계에서후보가탈락했으므로추가qualification을중단한다. Application source는V52그대로다. 준비된Round60스크립트는존재하지않는V53binary를가리키므로실행하지않는다. 이후후보가선택되면V51/V52/new/vLLM4lane구성으로갱신한다.

[분석](raw/columns-v53/attention-columns-v53/analysis.json), [Manifest](raw/columns-v53-manifest.json). 46파일SHA256검증완료,archive `320084bb1bbf8aa39f5175f4944b8985c97423996b244af57122a0ebb484205a`.

다음V54는Vcache배치를MMA Bfragment에맞춰packed화하고vectorpairload를비교한다. QK/softmax계산중복은추가하지않는다. Prototype는기존V와CPU로prepack한V를별도로보유해기준출력과비교한다. HotKVwrite비용은이primitive에없으며이후실제model/serving에서검증해야한다.
