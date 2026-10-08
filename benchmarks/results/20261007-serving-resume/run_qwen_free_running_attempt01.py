"""Fixed128 own-greedy Qwen correctness; launch only after Batch05 terminal collection/replay."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from materialize_qwen_m2_source import materialize
R=Path('/data/riley-serving-261007');O=R/'qwen-free-running-validation-attempt01';O.mkdir()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(name,x):(O/name).write_text(json.dumps(x,indent=2)+'\n')
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
source=R/'qwen-free-running-source-attempt01'
base=Path('/data/riley-benchmarks/20260915T134348Z-n06a-shared-host')
teacher=base/'qwen3b-hf-eager-teacher-forced-r1-20260916T002940Z'
hf=base/'qwen3b-p2048-cacheon-layer-stage-r7-20260917T141511Z'
manifest=hf/'qwen3b-p2048-cache-on-layer-stage.json';sidecar=hf/'qwen3b-p2048-cache-on-layer-stage.safetensors'
model=Path('/data/riley-models/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1')
failure=None
try:
    assert json.loads((R/'qwen-isolated-norm-native-validation-attempt02/completion.json').read_text())['failure'] is None, 'isolated norm correction prerequisite failed'
    batch05=json.loads((R/'kernel-batch05-quiet-attempt01/completion.json').read_text())
    assert batch05['failure'] is None and batch05['all_lanes_complete'] and len(batch05['records'])==96, 'Batch05 serving not successfully terminal; no overlap allowed'
    assert not Path('/proc/2231583').exists(), 'actual Batch05 controller still live'
    assert not gpu(),'GPU occupied; no native build/run started'
    for folder in ['kernel-batch03-quiet-attempt01','kernel-batch02-matched-queue-attempt01']:
        assert (R/folder/'completion.json').exists(),'prior benchmark terminal receipt missing'
    profile=json.loads((R/'kernel-batch03-profiling-attempt01/completion.json').read_text())
    assert profile['failure'] is None and len(profile['completed'])==32, 'profiling terminal incomplete'
    assert json.loads((R/'qwen-step109-layer3-hf-stage-validation-attempt02/controller-completion.json').read_text())['failure'] is None, 'HF stage observer terminal missing'
    inputs=[R/n for n in ['qwen-free-running-source-attempt01.tar','qwen-free-running-commit-attempt01.pack','qwen-free-running-source-receipt-attempt01.json']]
    assert sha(teacher/'teacher-forced-oracle.json')=='35d5d9b153d73b4badc67e5eedf2e8226ca2b55c7e30eb0970cfb741031edd35'
    assert sha(teacher/'cache-on-logits.safetensors')=='d1c649d126d494b5d902fa14ff22fbab46f13feeed46df380b9f55f4e5ab4c39'
    assert sha(model/'config.json')=='eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b'
    assert sha(model/'generation_config.json')=='ea35dfb6fc5051b01114f9b995820d55dab01ed33ee490f6378b442af82c09f9'
    pins={str(f):sha(f) for f in [*inputs,manifest,sidecar,base/'inputs/qwen3b-c8-p2048-o128.json',
        teacher/'teacher-forced-oracle.json',teacher/'cache-on-logits.safetensors',teacher/'cache-off-logits.safetensors',
        *sorted(f for f in model.iterdir() if f.is_file())]}
    write('preparation.json',{'pins':pins,'compute_pids':gpu(),'pressure':{k:Path('/proc/pressure',k).read_text() for k in ['cpu','io','memory']},
        'scope':'fixed128 own-greedy BF16 logits/token comparison plus same-owner lifecycle; no HTTP serving or production-selector claim'})
    assert sha(manifest)=='b2fc7301636ba618b904b554cd914ac35dd88c2f83e7c7736571353a4fd1fa4d'
    assert sha(sidecar)=='5c12fc6f34cddfc73ef0ebc33b4903550ef7f632c722e53e158f1c715ffd70c9'
    receipt=materialize(*inputs,source);write('source-git-receipt.json',receipt)
    result=O/'qwen3b-p2048-cache-on-free-running128-native-result.json'
    env={'HOME':'/home/psyche','LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PATH':'/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin',
        'CUDA_HOME':'/data/cuda-12.8.1','CUDAToolkit_ROOT':'/data/cuda-12.8.1','CUDACXX':'/data/cuda-12.8.1/bin/nvcc',
        'CMAKE':'/data/cmake-3.31.12/bin/cmake','CARGO_BUILD_JOBS':'1','CMAKE_BUILD_PARALLEL_LEVEL':'1',
        'CARGO_TARGET_DIR':str(R/'qwen-free-running-target-attempt01'),'RUSTUP_TOOLCHAIN':'1.85.0',
        'LD_LIBRARY_PATH':'/data/cuda-12.8.1/lib64','CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8',
        'RILEY_QWEN_SERVING_WORKLOAD':str(base/'inputs/qwen3b-c8-p2048-o128.json'),
        'RILEY_QWEN_HF_TEACHER_FORCED_ORACLE':str(teacher/'teacher-forced-oracle.json'),
        'RILEY_QWEN_HF_CACHE_OFF_SIDECAR':str(teacher/'cache-off-logits.safetensors'),
        'RILEY_QWEN_HF_CACHE_ON_SIDECAR':str(teacher/'cache-on-logits.safetensors'),
        'RILEY_QWEN3B_CHECKPOINT':str(model),'RILEY_QWEN3B_P2048_CACHE_ON_STAGE_MANIFEST':str(manifest),
        'RILEY_QWEN3B_P2048_CACHE_ON_STAGE_SIDECAR':str(sidecar),'RILEY_QWEN3B_P2048_CACHE_ON_FREE_RUNNING128_OUTPUT':str(result)}
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
    binary=executables[0]
    with (O/'selection-policy.log').open('x') as log:
        cpu=subprocess.run([str(binary),'--exact','qwen_addressable_greedy_selection_policy','--nocapture'],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
    write('selection-policy-process.json',{'exit':cpu.returncode,'log_sha256':sha(O/'selection-policy.log'),'binary_sha256':sha(binary)})
    assert cpu.returncode==0, 'actual native binary selection policy failed; no GPU qualifier launched'
    argv=[str(binary),'--ignored','--exact','qwen3b_p2048_cache_on_free_running128_logits_quality_gate','--nocapture']
    assert not gpu(),'foreign GPU process before actual test; no overlap allowed'
    write('native-launch.json',{'argv':argv,'source_revision':receipt['source_commit'],'binary_sha256':sha(binary),'time_ns':time.time_ns()})
    with (O/'native.log').open('x') as log:p=subprocess.run(argv,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
    write('native-process.json',{'exit':p.returncode,'log_sha256':sha(O/'native.log'),'binary_sha256_after':sha(binary),
        'result':json.loads(result.read_text()) if result.exists() else None})
    deadline=time.monotonic()+40
    while gpu() and time.monotonic()<deadline:time.sleep(1)
    assert not gpu(),'native resource reclamation failed'
    assert all(sha(path)==digest for path,digest in pins.items()),'pinned inputs changed during native test'
    expected_source=json.loads(inputs[2].read_text())['files']
    source_after={name:sha(source/name) for name in expected_source}
    source_status=subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True)
    write('source-integrity-after.json',{'source_commit':receipt['source_commit'],'files':source_after,'code_worktree_clean':not source_status})
    assert source_after==expected_source and not source_status,'materialized source changed during native build/test'
    assert p.returncode==0,'native qualifier nonzero; source/log preserved; inspect actual log before assigning a numerical failure'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:write('completion.json',{'failure':failure,'serving_performance':'미실행','fixed128_free_running_generation':'independent replay pending','goal_achieved':False})
