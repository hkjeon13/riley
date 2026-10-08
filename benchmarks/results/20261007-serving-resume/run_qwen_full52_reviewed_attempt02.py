"""Full52 M1 stage gate after the already queued prefill candidate correctness job."""
import hashlib,json,subprocess,time
from pathlib import Path
R=Path('/data/riley-serving-261007');O=R/'qwen-m1-regular-full52-attempt02';O.mkdir()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(O/n).write_text(json.dumps(x,indent=2)+'\n')
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
source=R/'qwen-m1-regular-source';pack=R/'qwen-full52-reviewed-attempt02.pack'
base=Path('/data/riley-benchmarks/20260915T134348Z-n06a-shared-host');teacher=base/'qwen3b-hf-eager-teacher-forced-r1-20260916T002940Z'
hf=base/'qwen3b-p2048-cacheon-layer-stage-r7-20260917T141511Z';manifest=hf/'qwen3b-p2048-cache-on-layer-stage.json';sidecar=hf/'qwen3b-p2048-cache-on-layer-stage.safetensors'
pins={str(p):sha(p) for p in [pack,manifest,sidecar,R/'qwen-m1-regular-candidate-source-receipt.json',R/'qwen-m1-regular-source-provenance.json']}
write('preparation.json',{'pins':pins,'wait_pid':3697803,'scope':'all52 M1 BF16 stages plus same-owner reset/re-prefill repeat; separate from selected-layer17 stage diagnostic','prefill_KV_all36_gate':'previous exact72/72 plus prefill logits; reuses9eeeeae9 code','M2_full128_generation':'not proven by this M1 test','serving_performance':'미실행'})
reviewed=json.loads((R/'qwen-full52-reviewed-attempt02-receipt.json').read_text())
pins.update({str(p):sha(p) for p in [R/'qwen-full52-reviewed-attempt02.rs',R/'qwen-full52-reviewed-attempt02-receipt.json']})
write('reviewed-test-source-update.json',reviewed)
failure=None
try:
 assert (R/'qwen-m1-regular-full52-attempt01/completion.json').exists(),'prior terminal absent'
 assert not gpu(),'GPU occupied'
 assert all(sha(p)==h for p,h in pins.items()),'pinned evidence changed'
 assert sha(manifest)=='b2fc7301636ba618b904b554cd914ac35dd88c2f83e7c7736571353a4fd1fa4d'
 assert sha(sidecar)=='5c12fc6f34cddfc73ef0ebc33b4903550ef7f632c722e53e158f1c715ffd70c9'
 assert sha(pack)==reviewed['pack_sha256']
 test=source/reviewed['path'];assert sha(test)==reviewed['old_test_sha256'],'old own test source changed'
 assert sha(source/'crates/riley-runtime/src/llama/decode.rs')==reviewed['unchanged_rust_decode_sha256']
 assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==reviewed['old_commit']
 assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True),'own source dirty'
 native=json.loads((R/'qwen-m1-regular-candidate-source-receipt.json').read_text())
 for f,h in native['files'].items():assert sha(source/f)==h,'numerical CUDA source changed'
 with pack.open('rb') as inp:subprocess.run(['git','-C',str(source),'index-pack','--stdin','--promisor'],stdin=inp,check=True,stdout=subprocess.DEVNULL)
 (O/'old-quality-gate-source.rs').write_bytes(test.read_bytes())
 test.write_bytes((R/'qwen-full52-reviewed-attempt02.rs').read_bytes());assert sha(test)==reviewed['new_test_sha256']
 subprocess.run(['git','-C',str(source),'hash-object','-w',str(test)],check=True,stdout=subprocess.DEVNULL)
 subprocess.run(['git','-C',str(source),'update-ref','HEAD',reviewed['commit'],reviewed['old_commit']],check=True)
 subprocess.run(['git','-C',str(source),'read-tree','--reset','HEAD'],check=True)
 subprocess.run(['git','-C',str(source),'read-tree','-mu','HEAD'],check=True)
 assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True),'updated source not clean'
 native['commit']=reviewed['commit']
 write('source-git-receipt.json',{'revision':reviewed['commit'],'numerical_source_unchanged':True,'strict_test_source_sha256':sha(test),'pack_sha256':sha(pack)})
 result=O/'qwen3b-p2048-cache-on-m1-full52-rust-result.json'
 env={'HOME':'/home/psyche','LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PATH':'/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin','CUDA_HOME':'/data/cuda-12.8.1','CUDAToolkit_ROOT':'/data/cuda-12.8.1','CUDACXX':'/data/cuda-12.8.1/bin/nvcc','CMAKE':'/data/cmake-3.31.12/bin/cmake','CARGO_BUILD_JOBS':'1','CMAKE_BUILD_PARALLEL_LEVEL':'1','CARGO_TARGET_DIR':str(R/'qwen-m1-regular-target'),'RUSTUP_TOOLCHAIN':'1.85.0','LD_LIBRARY_PATH':'/data/cuda-12.8.1/lib64','CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','RILEY_QWEN_SERVING_WORKLOAD':str(base/'inputs/qwen3b-c8-p2048-o128.json'),'RILEY_QWEN_HF_TEACHER_FORCED_ORACLE':str(teacher/'teacher-forced-oracle.json'),'RILEY_QWEN_HF_CACHE_OFF_SIDECAR':str(teacher/'cache-off-logits.safetensors'),'RILEY_QWEN_HF_CACHE_ON_SIDECAR':str(teacher/'cache-on-logits.safetensors'),'RILEY_QWEN3B_CHECKPOINT':'/data/riley-models/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1','RILEY_QWEN3B_P2048_CACHE_ON_STAGE_MANIFEST':str(manifest),'RILEY_QWEN3B_P2048_CACHE_ON_STAGE_SIDECAR':str(sidecar),'RILEY_QWEN3B_P2048_CACHE_ON_M1_CUBLAS_QK_AV_STAGE_OUTPUT':str(result)}
 argv=['/home/psyche/.cargo/bin/cargo','test','--locked','-p','riley-runtime','--features','cuda,cuda-cublas-gemm-probe','--test','qwen3b_p2051_full_forward_gpu','--','--ignored','--exact','qwen3b_p2048_hf_compatible_cache_on_m1_cublas_qk_av_full_forward_candidate_trace','--nocapture']
 write('native-launch.json',{'argv':argv,'source_revision':native['commit'],'serving_performance':'미실행'})
 with (O/'native.log').open('x') as log:p=subprocess.run(argv,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
 write('native-process.json',{'exit':p.returncode,'log_sha256':sha(O/'native.log'),'result':json.loads(result.read_text()) if result.exists() else None})
 deadline=time.monotonic()+40
 while gpu() and time.monotonic()<deadline:time.sleep(1)
 assert not gpu(),'native resources not reclaimed'
 assert p.returncode==0,'full52 gate failed; preserved native result/log'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:write('completion.json',{'failure':failure,'serving_performance':'미실행','M2_full128_generation':'unverified','goal_achieved':False})
