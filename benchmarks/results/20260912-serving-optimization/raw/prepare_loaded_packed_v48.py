from pathlib import Path
import shutil
r=Path('/tmp/riley-opt-260912');out=r/'packed-prefill-loaded-v48';out.mkdir()
for n in ['packed_prefill_model_v48.cuh','packed_prefill_attention_v48.cuh','packed_prefill_rope_v48.cuh']:shutil.copy2(r/'packed-prefill-v48'/n,out/n)
s=(r/'packed_prefill_probe_v48.cu').read_text().replace('cap=512,physical=256','cap=1024,physical=256').replace('int lists[6][4]={{128,128,128,128},{16,128,129,239},{7,8,9,31},{1,17,127,128},{129,129,127,127},{1,1,1,1}}','int lists[8][4]={{128,128,128,128},{16,128,129,239},{7,8,9,31},{1,17,127,128},{129,129,127,127},{1,1,1,1},{256,256,256,256},{17,129,398,480}}').replace('pattern<6','pattern<8').replace('packed_cap=total<=128?128:512','packed_cap=total<=128?128:(total<=512?512:1024)').replace('pattern==0||pattern==1||pattern==5','pattern==0||pattern==1||pattern==5||pattern==6');(out/'probe.cu').write_text(s)
s=(r/'check_packed_prefill_v48.py').read_text().replace("out=r/'packed-prefill-v48'","out=r/'packed-prefill-loaded-v48'").replace(";shutil.copy2(r/'packed_prefill_probe_v48.cu',out/'probe.cu')",'').replace("r/'prefill-full-model-v11'","r/'loaded-rope-fixture-v11'");a=s.index('for name,args in jobs:');s=s[:a]+'''jobs += [(tool,['/data/cuda-12.8.1/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',str(out/'probe'),str(r/'loaded-rope-fixture-v11')]) for tool in ['memcheck','racecheck']]
jobs += [('timing',[str(out/'probe'),str(r/'loaded-rope-fixture-v11'),'timing'])]
'''+s[a:];(r/'check_loaded_packed_v48.py').write_text(s)
