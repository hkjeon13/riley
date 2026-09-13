import hashlib,json,os,subprocess
from pathlib import Path
root=Path('/tmp/riley-opt-260912')
build=root/'flashinfer-model-build'; source=root/'hardware-validation-source'
out=root/'prefill-cmake-evidence-v1'; cmake='/data/cmake-3.31.12/bin/cmake'
data='/data/riley-vllm-interim.CfrT9T/venv/lib/python3.13/site-packages/flashinfer/data'
env=dict(os.environ,LD_LIBRARY_PATH=str(root/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib')
results={}
def run(name,args,ok=True):
    p=subprocess.run(list(map(str,args)),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (out/(name+'.log')).write_text(p.stdout)
    results[name]={'exit_code':p.returncode,'expected_success':ok}
    if (p.returncode==0)!=ok:raise RuntimeError(name+' unexpected result: '+p.stdout[-2000:])
    return p.stdout
configure=[cmake,'-S',source/'kernels','-B',build,'-DRILEY_FLASHINFER_DATA='+data]
run('reconfigure',configure)
header=build/'flashinfer-prefill-overlay-v1/include/flashinfer/attention/prefill.cuh'
original=header.read_bytes()
try:
    header.write_bytes(original+b'\n// deliberate generated-overlay drift\n')
    msg=run('changed-header-rejected',[cmake,'--build',build,'--target','riley_flashinfer_prefill_object'],False)
    assert 'existing generated FlashInfer overlay has changed' in ' '.join(msg.split())
finally:header.write_bytes(original)
receipt=build/'flashinfer-prefill-overlay-v1/overlay.json';receipt_bytes=receipt.read_bytes()
try:
    receipt.write_text('{}\n')
    msg=run('changed-receipt-rejected',configure,False)
    assert 'overlay receipt has changed' in ' '.join(msg.split())
finally:receipt.write_bytes(receipt_bytes)
run('restored-configure',configure)
run('restored-build',[cmake,'--build',build,'--target','riley_cuda_native','-j4'])
archive=build/'libriley_cuda_native.a'
symbols=run('enabled-symbols',['nm','-g','--defined-only',archive])
for s in ['prefill_workspace_bytes','prefill_prepare','prefill_run','decode_run']:assert 'riley_flashinfer_'+s in symbols
nvcc=root/'flashinfer-compatibility/toolchain130/nvidia/cu13/bin/nvcc'
run('probe-build',[nvcc,'-std=c++17','-O3','-arch=sm_89',source/'benchmarks/analysis/flashinfer_prefill_probe.cu',archive,'-o',out/'probe'])
run('archive-probe',[out/'probe',out/'output.bf16'])
new=(out/'output.bf16').read_bytes();old=(root/'flashinfer-prefill-sync-v1/output-v2.bf16').read_bytes()
assert new==old
results['output']={'values':len(new)//2,'sha256':hashlib.sha256(new).hexdigest(),'bitwise_equal_prior_probe':True}
run('archive-memcheck',['/data/cuda-12.8.1/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',out/'probe'])
msg=run('archive-racecheck',['/data/cuda-12.8.1/bin/compute-sanitizer','--tool','racecheck','--error-exitcode','99',out/'probe'])
assert '0 hazards' in msg and '0 warnings' in ' '.join(msg.split())
run('disabled-configure',configure[:-1]+['-DRILEY_FLASHINFER_DATA='])
run('disabled-build',[cmake,'--build',build,'--target','riley_cuda_native','-j4'])
symbols=run('disabled-symbols',['nm','-g','--defined-only',archive])
assert 'riley_flashinfer_' not in symbols
run('reenabled-configure',configure)
run('reenabled-build',[cmake,'--build',build,'--target','riley_cuda_native','-j4'])
run('original-lock-check',['python3',source/'kernels/optional/verify_flashinfer.py',data])
results['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
(out/'checks.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
