"""Batch14 source receipt, native primitive exactness, server build identity and HTTP60 exact screen. Correctness only; no serving timing claim."""
import json,hashlib,subprocess,time,os
from pathlib import Path
R=Path("/data/riley-serving-261007");S=R/"kernel-batch14-source-attempt01";T=R/"kernel-batch14-target-attempt02"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
tool=Path("/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13")
env={"HOME":"/home/psyche","LANG":"C.UTF-8","LC_ALL":"C.UTF-8","PATH":"/home/psyche/.cargo/bin:"+str(tool/"bin")+":/usr/bin:/bin","CUDA_HOME":str(tool),"CUDAToolkit_ROOT":str(tool),"CMAKE":"/data/cmake-3.31.12/bin/cmake","CMAKE_BUILD_PARALLEL_LEVEL":"4","CARGO_BUILD_JOBS":"4","CARGO_TARGET_DIR":str(T),"LD_LIBRARY_PATH":str(tool/"lib"),"CUDA_VISIBLE_DEVICES":"0"}
git=lambda *a:subprocess.check_output(["git","-C",str(S),*a],text=True)
gpu=lambda:subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
failure=None;O=R/"kernel-batch14-server-validation-attempt03";O.mkdir();P=R/"kernel-batch14-primitive-validation-attempt03";P.mkdir()
try:
 assert not git("status","--porcelain","--untracked-files=no"),"source dirty"
 commit=git("rev-parse","HEAD").strip();files={n:sha(S/n) for n in git("ls-files").splitlines()}
 receipt={"source_commit":commit,"parent_candidate":"9acd7ee6bb1574931a8390b79610683ef7eb11e0","files":files,"changed_vs_batch13":git("diff","--stat","9acd7ee6bb1574931a8390b79610683ef7eb11e0","HEAD").splitlines(),"device_kernel_sources_changed":[n for n in git("diff","--name-only","9acd7ee6bb1574931a8390b79610683ef7eb11e0","HEAD").splitlines() if n.endswith((".cu",".cuh"))]}
 (R/"kernel-batch14-source-receipt-attempt03.json").write_text(json.dumps(receipt,indent=1)+"\n")
 argv=[str(tool/"bin/nvcc"),"-O3","-DNDEBUG","-std=c++17","-arch=sm_89","--use_fast_math","-I"+str(S/"kernels/src"),str(S/"kernels/tests/kernel_batch14_correctness.cu"),"-o",str(P/"native3161")]
 (P/"preparation.json").write_text(json.dumps({"argv":argv,"source_commit":commit},indent=1))
 with (P/"build.log").open("x") as f:assert subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT).returncode==0,"native build failed"
 assert not gpu(),"GPU occupied";before=sha(P/"native3161")
 with (P/"native.log").open("x") as f:p=subprocess.run([str(P/"native3161")],env=env,stdout=f,stderr=subprocess.STDOUT)
 records=[json.loads(l) for l in (P/"native.log").read_text().splitlines() if l.startswith("{")];assert len(records)==7 and all(x["passed"] for x in records) and records[1]["cases"]==3161 and records[2]["merge_norm_prefetch_cases"]==24 and records[3]["prefill_qkv_cases"]==69 and records[4]["mixed_rope_cases"]==45 and records[5]["fused_qkv_cases"]==270 and records[6]["narrow_grid_attention_cases"]==4800,"batch14 native records";result=dict(records[1],merge_norm_prefetch_cases=24,prefill_qkv_cases=69,mixed_rope_cases=45,fused_qkv_cases=270,narrow_grid_attention_cases=4800)
 (P/"native-process.json").write_text(json.dumps({"exit":p.returncode,"binary_before":before,"binary_after":sha(P/"native3161"),"log_sha256":sha(P/"native.log"),"native_result":result,"GPU_compute_after":gpu()},indent=1))
 assert p.returncode==0 and result["passed"] and result["cases"]==3161 and result["BF16_and_FP32_exact"],"native exactness failed"
 (P/"completion.json").write_text(json.dumps({"failure":None},indent=1))
 argv=["cargo","build","--locked","--release","-p","riley-server","--features","cuda,server"]
 with (O/"build.log").open("x") as f:assert subprocess.run(argv,env=env,cwd=S,stdout=f,stderr=subprocess.STDOUT).returncode==0,"server build failed"
 binary=T/"release/riley";h=sha(binary);(O/"binary-receipt.json").write_text(json.dumps({"source_commit":commit,"sha256":h},indent=1)+"\n")
 assert not gpu(),"GPU occupied"
 with (O/"http-screen.log").open("x") as f:p=subprocess.run([str(R/"vllm0271-venv/bin/python"),str(R/"controller/kernel_batch14_http_screen_attempt02.py")],env=env,cwd=S,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,"HTTP correctness failed; no serving promotion"
 assert sha(binary)==h and all(sha(S/n)==x for n,x in files.items()),"binary/source changed during validation"
except BaseException as e:failure={"type":type(e).__name__,"message":str(e)};raise
finally:(O/"completion.json").write_text(json.dumps({"failure":failure,"serving_performance":"미실행","adopted":False,"goal_achieved":False},indent=2)+"\n")
