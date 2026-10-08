import pathlib,subprocess,json,hashlib,time
src=pathlib.Path('/data/riley-serving-261007/kernel-batch08-source-attempt01')
out=pathlib.Path('/data/riley-serving-261007/kernel-batch08-primitive-validation-attempt01')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
failure=None
def idle():assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
try:
 idle()
 revision=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()
 files={str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file() and '.git' not in p.parts and p.name!='.git'}
 binary=out/'native3161'
 argv=['/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13/bin/nvcc','-O3','-DNDEBUG','-std=c++17','-arch=sm_89','--use_fast_math','--ptxas-options=-v','-I'+str(src/'kernels/src'),str(src/'kernels/tests/kernel_batch08_correctness.cu'),'-o',str(binary)]
 (out/'preparation.json').write_text(json.dumps({'revision':revision,'files':files,'argv':argv,'serving':'미실행'},indent=2))
 with (out/'build.log').open('x') as log:p=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
 assert p.returncode==0,'compile failed'
 h=sha(binary)
 idle()
 (out/'native-launch.json').write_text(json.dumps({'argv':[str(binary)],'sha256':h,'time_ns':time.time_ns()}))
 with (out/'native.log').open('x') as log:p=subprocess.run([str(binary)],stdout=log,stderr=subprocess.STDOUT)
 (out/'native-process.json').write_text(json.dumps({'exit':p.returncode,'binary_after':sha(binary),'log_sha256':sha(out/'native.log')}))
 assert p.returncode==0 and sha(binary)==h,'native failed'
 result=json.loads((out/'native.log').read_text().splitlines()[-1])
 assert result['passed'] is True and result['cases']==3161 and result['decode_value_cases']==360 and result['mapped_value_cases']==1593
 assert all(sha(src/n)==h for n,h in files.items()),'source changed'
 idle()
 (out/'verified-native.json').write_text(json.dumps(result,indent=2))
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)}
 raise
finally:
 (out/'completion.json').write_text(json.dumps({'failure':failure,'serving':'미실행','adopted':False,'goal_achieved':False},indent=2))
