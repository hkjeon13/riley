import json,hashlib,subprocess,time
from pathlib import Path
from materialize_qwen_m2_source import materialize
R=Path('/data/riley-serving-261007');O=R/'kernel-batch04-primitive-validation-attempt01';O.mkdir();failure=None
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
try:
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not gpu,'GPU occupied; leave all processes untouched'
 inputs=[R/n for n in ['kernel-batch04-source-attempt01.tar','kernel-batch04-commit-attempt01.pack','kernel-batch04-source-receipt-attempt01.json']]
 source=R/'kernel-batch04-source-attempt01';receipt=materialize(*inputs,source)
 (O/'source-git-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 tool=Path('/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13');binary=O/'native1088'
 argv=[str(tool/'bin/nvcc'),'-O3','-DNDEBUG','-std=c++17','-arch=sm_89','--use_fast_math','-I'+str(source/'kernels/src'),str(source/'kernels/tests/kernel_batch04_correctness.cu'),'-o',str(binary)]
 (O/'preparation.json').write_text(json.dumps({'source':receipt,'argv':argv,'GPU_execution':'unexecuted before compilation','serving_performance':'미실행'},indent=2)+'\n')
 with (O/'build.log').open('x') as f:p=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,'native compile failed; source/log preserved'
 h=sha(binary);(O/'native-launch.json').write_text(json.dumps({'argv':[str(binary)],'binary_sha256':h,'time_ns':time.time_ns()},indent=2)+'\n')
 with (O/'native.log').open('x') as f:p=subprocess.run([str(binary)],stdout=f,stderr=subprocess.STDOUT)
 assert sha(binary)==h,'binary changed during native execution'
 (O/'native-process.json').write_text(json.dumps({'exit':p.returncode,'log_sha256':sha(O/'native.log'),'binary_sha256_after':sha(binary)},indent=2)+'\n')
 assert p.returncode==0,'native exact byte comparison failed; original log preserved'
 result=json.loads((O/'native.log').read_text().splitlines()[-1]);assert result['passed'] is True and result['cases']==1088 and result['fusion_cases']==272
 assert all(sha(source/n)==h for n,h in json.loads(inputs[2].read_text())['files'].items()),'source changed'
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU resources not reclaimed'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'serving_performance':'미실행','adopted':False,'goal_achieved':False},indent=2)+'\n')
