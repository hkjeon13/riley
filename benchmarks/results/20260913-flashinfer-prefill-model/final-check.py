import os,subprocess,re,json,hashlib
from pathlib import Path
root=Path('/tmp/riley-opt-260912');source=root/'hardware-validation-source';out=root/'prefill-model-v1'
cu=root/'flashinfer-compatibility/toolchain130/nvidia/cu13'
env=dict(os.environ,PATH=f'/home/psyche/.cargo/bin:{cu}/bin:/usr/bin:/bin',CUDA_HOME=str(cu),CUDAToolkit_ROOT=str(cu),CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(root/'flashinfer-cargo-target'),RILEY_FLASHINFER_DATA='/data/riley-vllm-interim.CfrT9T/venv/lib/python3.13/site-packages/flashinfer/data',LD_LIBRARY_PATH=str(root/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib',RILEY_REAL_CHECKPOINT='/data/riley-benchmark/20260827T051948Z-d7ad713a/model')
args=['cargo','test','-p','riley-scheduler','--release','--features','cuda','--test','flashinfer_prefill_free_generation_gpu','--test','flashinfer_prefill_natural_logits_gpu','--no-run']
p=subprocess.run(args,cwd=source,env=env,capture_output=True,text=True);log=p.stdout+p.stderr;(out/'final-build.log').write_text(log);assert p.returncode==0,log[-3000:]
binaries=re.findall(r'Executable tests/(\w+)\.rs \(([^)]+)\)',log);assert len(binaries)==2,binaries
receipt={'binaries':{},'commands':[]}
for name,binary in binaries:
 receipt['binaries'][name]={'path':binary,'sha256':hashlib.sha256(Path(binary).read_bytes()).hexdigest()}
 if 'free_generation' in name:
  args=[binary,'--ignored','--nocapture']
  p=subprocess.run(args,cwd=source,env=env,capture_output=True,text=True);log=p.stdout+p.stderr
  (out/'final-free-generation.log').write_text(log);assert p.returncode==101 and 'differing_tokens=56' in log and 'repeated_prompt_invariance=true' in log
 else:
  folder=root/'prefill-natural-memcheck-v1';folder.mkdir(exist_ok=True);(folder/'cases.json').write_bytes((root/'prefill-natural-v1/cases.json').read_bytes())
  args=['/data/cuda-12.8.1/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',binary,'--ignored','--nocapture']
  p=subprocess.run(args,cwd=source,env=dict(env,RILEY_PREFILL_NATURAL_SCREEN_DIR=str(folder)),capture_output=True,text=True);log=p.stdout+p.stderr
  (out/'model-memcheck.log').write_text(log);assert p.returncode==0 and 'ERROR SUMMARY: 0 errors' in log,log[-2000:]
  for lane in ['baseline','candidate']:assert (folder/(lane+'.bf16')).read_bytes()==(root/'prefill-natural-v1'/(lane+'.bf16')).read_bytes()
 receipt['commands'].append({'command':args,'exit_code':p.returncode})
receipt['baseline_logits_equal_previous_v7']=(root/'prefill-natural-v1/baseline.bf16').read_bytes()==(root/'ffn-natural-v1/baseline.bf16').read_bytes()
assert receipt['baseline_logits_equal_previous_v7']
(out/'final-verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
