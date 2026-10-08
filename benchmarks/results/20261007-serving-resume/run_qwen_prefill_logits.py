"""Source-bound correctness-only all-layer KV oracle and Rust comparison."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/data/riley-serving-261007')
repo=root/'qwen-prefill-logits-source'
out=root/'qwen-prefill-logits-attempt01';out.mkdir()
base=Path('/data/riley-benchmarks/20260915T134348Z-n06a-shared-host')
teacher=base/'qwen3b-hf-eager-teacher-forced-r1-20260916T002940Z'
checkpoint=Path('/data/riley-models/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1')
workload=base/'inputs/qwen3b-c8-p2048-o128.json'
python=root/'vllm0271-venv/bin/python'
module='riley_reference.qwen3b_cache_on_prefill_kv_trace'
manifest=out/'qwen3b-p2048-cache-on-prefill-kv.json'
sidecar=out/'qwen3b-p2048-cache-on-prefill-kv.safetensors'
result=out/'qwen3b-p2048-cache-on-prefill-kv-rust-result.json'
provenance=root/'qwen-prefill-logits-source-provenance.json'
env=os.environ.copy()
env.update({'PATH':str(root/'vllm0271-venv/bin')+':/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin',
 'PYTHONPATH':str(repo/'tools/python/reference'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
 'CUBLAS_WORKSPACE_CONFIG':':4096:8','CUDA_VISIBLE_DEVICES':'0',
 'LD_LIBRARY_PATH':'/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13/lib',
 'RILEY_QWEN3B_CACHE_ON_PREFILL_KV_SOURCE_PROVENANCE':str(provenance)})

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def write(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n')
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
def host():return {'time_ns':time.time_ns(),'gpu_compute':gpu(),'pressure':{k:Path('/proc/pressure',k).read_text() for k in ['cpu','io','memory']}}
steps=[]
def run(label,argv,child_env):
 assert not gpu(),'GPU compute overlap before '+label
 before=host();started=time.time_ns()
 with (out/(label+'.log')).open('w') as log:
  p=subprocess.run(argv,cwd=repo,env=child_env,stdout=log,stderr=subprocess.STDOUT)
 receipt={'label':label,'argv':argv,'exit':p.returncode,'started_ns':started,'ended_ns':time.time_ns(),
          'before':before,'after':host(),'log_sha256':sha(out/(label+'.log')),'serving_performance':'미실행'}
 steps.append(receipt);write('progress.json',steps);print(label+' exit '+str(p.returncode),flush=True)
 if p.returncode:raise RuntimeError(label+' failed with exit '+str(p.returncode))
 assert not gpu(),'owned GPU resource not reclaimed after '+label

failure=None
try:
 assert not gpu(),'foreign GPU compute process'
 write('preparation.json',{'scope':'Qwen cache-on correctness only; no serving benchmark',
  'source_provenance':json.loads(provenance.read_text()),'checkpoint_receipt_sha256':sha(checkpoint/'riley-checkpoint.json'),
  'workload_sha256':sha(workload),'teacher_manifest_sha256':sha(teacher/'teacher-forced-oracle.json'),
  'teacher_cache_on_sidecar_sha256':sha(teacher/'cache-on-logits.safetensors'),
  'controller_sha256':sha(__file__),'packages_sha256':sha(root/'vllm0271-packages.txt'),
  'required_contract':'immutable teacher prefill logits BF16 exact before emitting 72 K/V tensors; Rust K/V and native prefill logits byte equality and same-owner repeat; no tolerance changes',
  'historical_container_missing':True,'replacement_HF_environment':'torch2.13.0 transformers5.15.1; golden logit equality remains mandatory',
  'native_Rust_contract':'Rust1.85.0 CUDA12.8.1 matching prior Qwen diagnostic', 'host':host()})
 run('hf-produce',[str(python),'-m',module,'produce','--checkpoint',str(checkpoint),
     '--workload',str(workload),'--teacher-manifest',str(teacher/'teacher-forced-oracle.json'),
     '--teacher-cache-on-sidecar',str(teacher/'cache-on-logits.safetensors'),
     '--manifest',str(manifest),'--sidecar',str(sidecar),'--repo-root',str(repo),'--device','cuda:0'],env)
 run('hf-validate',[str(python),'-m',module,'validate',str(manifest),'--sidecar',str(sidecar),
     '--workload',str(workload),'--teacher-manifest',str(teacher/'teacher-forced-oracle.json'),
     '--teacher-cache-on-sidecar',str(teacher/'cache-on-logits.safetensors'),'--repo-root',str(repo)],env)
 native=env.copy();native.update({'CUDA_HOME':'/data/cuda-12.8.1','CUDAToolkit_ROOT':'/data/cuda-12.8.1',
  'CUDACXX':'/data/cuda-12.8.1/bin/nvcc','CMAKE':'/data/cmake-3.31.12/bin/cmake','CARGO_BUILD_JOBS':'1',
  'CMAKE_BUILD_PARALLEL_LEVEL':'1','CARGO_TARGET_DIR':str(root/'qwen-prefill-logits-target'),'RUSTUP_TOOLCHAIN':'1.85.0',
  'LD_LIBRARY_PATH':'/data/cuda-12.8.1/lib64','RILEY_QWEN_SERVING_WORKLOAD':str(workload),
  'RILEY_QWEN_HF_TEACHER_FORCED_ORACLE':str(teacher/'teacher-forced-oracle.json'),
  'RILEY_QWEN_HF_CACHE_OFF_SIDECAR':str(teacher/'cache-off-logits.safetensors'),
  'RILEY_QWEN_HF_CACHE_ON_SIDECAR':str(teacher/'cache-on-logits.safetensors'),
  'RILEY_QWEN3B_CHECKPOINT':str(checkpoint),
  'RILEY_QWEN3B_P2048_CACHE_ON_PREFILL_KV_MANIFEST':str(manifest),
  'RILEY_QWEN3B_P2048_CACHE_ON_PREFILL_KV_SIDECAR':str(sidecar),
  'RILEY_QWEN3B_P2048_CACHE_ON_PREFILL_KV_STAGE_OUTPUT':str(result)})
 run('rust-kv-gate',['/home/psyche/.cargo/bin/cargo','test','--locked','-p','riley-runtime',
     '--features','cuda,cuda-cublas-gemm-probe','--test','qwen3b_p2051_full_forward_gpu','--',
     '--ignored','--exact','qwen3b_p2048_hf_compatible_cache_on_prefill_kv_quality_gate','--nocapture'],native)
except BaseException as e:
 failure={'type':type(e).__name__,'message':str(e)};raise
finally:
 write('completion.json',{'steps':steps,'failure':failure,
       'rust_result':json.loads(result.read_text()) if result.exists() else None,
       'serving_performance':'미실행','goal_achieved':False,'performance_claim_eligible':False,
       'files':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='completion.json'}})
