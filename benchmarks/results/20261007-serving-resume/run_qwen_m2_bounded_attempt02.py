"""Build and run source-bound M1/M2 correctness after both serving jobs end."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from materialize_qwen_m2_source import materialize
R=Path('/data/riley-serving-261007');O=R/'qwen-m2-bounded-validation-attempt02';O.mkdir()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(name,x):(O/name).write_text(json.dumps(x,indent=2)+'\n')
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
source=R/'qwen-m2-bounded-source-attempt02'
base=Path('/data/riley-benchmarks/20260915T134348Z-n06a-shared-host')
teacher=base/'qwen3b-hf-eager-teacher-forced-r1-20260916T002940Z'
hf=base/'qwen3b-p2048-cacheon-layer-stage-r7-20260917T141511Z'
manifest=hf/'qwen3b-p2048-cache-on-layer-stage.json';sidecar=hf/'qwen3b-p2048-cache-on-layer-stage.safetensors'
model=Path('/data/riley-models/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1')
failure=None
try:
    assert not gpu(),'GPU occupied; no native build/run started'
    for folder in ['kernel-batch03-quiet-attempt01','kernel-batch02-matched-queue-attempt01']:
        assert (R/folder/'completion.json').exists(),'prior benchmark terminal receipt missing'
    inputs=[R/n for n in ['qwen-m2-bounded-source-attempt02.tar','qwen-m2-bounded-commit-attempt02.pack','qwen-m2-bounded-source-receipt-attempt02.json']]
    pins={str(f):sha(f) for f in [*inputs,manifest,sidecar,base/'inputs/qwen3b-c8-p2048-o128.json',
        teacher/'teacher-forced-oracle.json',teacher/'cache-on-logits.safetensors',teacher/'cache-off-logits.safetensors',
        *sorted(f for f in model.iterdir() if f.is_file())]}
    write('preparation.json',{'pins':pins,'compute_pids':gpu(),'pressure':{k:Path('/proc/pressure',k).read_text() for k in ['cpu','io','memory']},
        'scope':'M1+M2 exact104 teacher-forced stages and lifecycle; no serving or full128 generation claim'})
    assert sha(manifest)=='b2fc7301636ba618b904b554cd914ac35dd88c2f83e7c7736571353a4fd1fa4d'
    assert sha(sidecar)=='5c12fc6f34cddfc73ef0ebc33b4903550ef7f632c722e53e158f1c715ffd70c9'
    receipt=materialize(*inputs,source);write('source-git-receipt.json',receipt)
    result=O/'qwen3b-p2048-cache-on-m1-m2-native-result.json'
    env={'HOME':'/home/psyche','LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PATH':'/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin',
        'CUDA_HOME':'/data/cuda-12.8.1','CUDAToolkit_ROOT':'/data/cuda-12.8.1','CUDACXX':'/data/cuda-12.8.1/bin/nvcc',
        'CMAKE':'/data/cmake-3.31.12/bin/cmake','CARGO_BUILD_JOBS':'1','CMAKE_BUILD_PARALLEL_LEVEL':'1',
        'CARGO_TARGET_DIR':str(R/'qwen-m2-bounded-target-attempt02'),'RUSTUP_TOOLCHAIN':'1.85.0',
        'LD_LIBRARY_PATH':'/data/cuda-12.8.1/lib64','CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8',
        'RILEY_QWEN_SERVING_WORKLOAD':str(base/'inputs/qwen3b-c8-p2048-o128.json'),
        'RILEY_QWEN_HF_TEACHER_FORCED_ORACLE':str(teacher/'teacher-forced-oracle.json'),
        'RILEY_QWEN_HF_CACHE_OFF_SIDECAR':str(teacher/'cache-off-logits.safetensors'),
        'RILEY_QWEN_HF_CACHE_ON_SIDECAR':str(teacher/'cache-on-logits.safetensors'),
        'RILEY_QWEN3B_CHECKPOINT':str(model),'RILEY_QWEN3B_P2048_CACHE_ON_STAGE_MANIFEST':str(manifest),
        'RILEY_QWEN3B_P2048_CACHE_ON_STAGE_SIDECAR':str(sidecar),'RILEY_QWEN3B_P2048_CACHE_ON_M1_M2_STAGE_OUTPUT':str(result)}
    build=['/home/psyche/.cargo/bin/cargo','test','--locked','-p','riley-runtime','--features','cuda,cuda-cublas-gemm-probe',
        '--test','qwen3b_p2051_full_forward_gpu','--no-run','--message-format=json']
    write('build-launch.json',{'argv':build,'env':env,'source_revision':receipt['source_commit'],'serving_performance':'미실행'})
    with (O/'build.log').open('x') as log:p=subprocess.run(build,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
    write('build-process.json',{'exit':p.returncode,'log_sha256':sha(O/'build.log')})
    assert p.returncode==0,'native build failed; source/log preserved'
    executables=[]
    for line in (O/'build.log').read_text().splitlines():
        try:entry=json.loads(line)
        except json.JSONDecodeError:continue
        if entry.get('reason')=='compiler-artifact' and entry.get('target',{}).get('name')=='qwen3b_p2051_full_forward_gpu' and entry.get('executable'):
            executables.append(Path(entry['executable']))
    assert len(executables)==1,'native test executable receipt ambiguous'
    binary=executables[0];argv=[str(binary),'--ignored','--exact','qwen3b_p2048_cache_on_m1_m2_bounded_full_forward_quality_gate','--nocapture']
    assert not gpu(),'foreign GPU process before actual test; no overlap allowed'
    write('native-launch.json',{'argv':argv,'source_revision':receipt['source_commit'],'binary_sha256':sha(binary),'time_ns':time.time_ns()})
    with (O/'native.log').open('x') as log:p=subprocess.run(argv,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
    write('native-process.json',{'exit':p.returncode,'log_sha256':sha(O/'native.log'),'binary_sha256_after':sha(binary),
        'result':json.loads(result.read_text()) if result.exists() else None})
    deadline=time.monotonic()+40
    while gpu() and time.monotonic()<deadline:time.sleep(1)
    assert not gpu(),'native resource reclamation failed'
    assert all(sha(path)==digest for path,digest in pins.items()),'pinned inputs changed during native test'
    assert p.returncode==0,'native qualifier nonzero; source/log preserved; inspect actual log before assigning a numerical failure'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:write('completion.json',{'failure':failure,'serving_performance':'미실행','full128_generation':'unverified','goal_achieved':False})
