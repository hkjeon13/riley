"""Build and verify the experimental native plan ABI; Python is build-only.
Usage: python build_fa3_native_adapter.py FA3_CHECKOUT NVCC FRESH_OUTPUT
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from build_fa3_native_probe import prepare


def build(source, nvcc, output):
    source, nvcc, output = source.resolve(), nvcc.resolve(), output.absolute()
    receipt=prepare(source,output)
    header=output/'fa3/hopper/flash_fwd_launch_template.h'
    original=header.read_text()
    patched=original
    replacements=[
        ('void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {',
         'void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream, bool riley_prepare_only = false) {',1),
        ('if (Varlen && !params.skip_scheduler_metadata_computation) {',
         'if (Varlen && !params.skip_scheduler_metadata_computation && !riley_prepare_only) {',1),
        ('if (smem_size >= 48 * 1024) {','if (smem_size >= 48 * 1024 && riley_prepare_only) {',2),
        ('        dim3 cluster_dims(', '        if (riley_prepare_only) return;\n        dim3 cluster_dims(',1),
        ('        // kernel<<<grid_dims,', '        if (riley_prepare_only) return;\n        // kernel<<<grid_dims,',1)]
    for before,after,count in replacements:
        if patched.count(before)!=count: raise ValueError('native launch overlay anchor mismatch: '+before)
        patched=patched.replace(before,after)
    header.write_text(patched)
    root=Path(__file__).resolve().parents[2]
    native=root/'kernels/optional'
    probe=Path(__file__).with_name('fa3_adapter_contract_probe.cpp')
    command=[str(nvcc),'-std=c++17','-O3','-arch=sm_90a','--expt-relaxed-constexpr',
             '--expt-extended-lambda','-Xptxas=-v','-I'+str(output/'fa3/hopper'),
             '-I'+str(output/'cutlass/include'),'-I'+str(output/'cutlass/tools/util/include'),
             '-I'+str(native),str(native/'fa3_adapter.cu'),
             str(output/'fa3/hopper/flash_prepare_scheduler.cu'),str(probe),'-lcuda',
             '-o',str(output/'fa3-adapter-probe')]
    receipt.update({'command':command,'launch_original_sha256':hashlib.sha256(original.encode()).hexdigest(),
                    'launch_overlay_sha256':hashlib.sha256(patched.encode()).hexdigest(),
                    'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in [native/'fa3_adapter.cu',native/'fa3_api.h',native/'fa3_contract.hpp',probe]},
                    'nvcc':subprocess.check_output([str(nvcc),'--version'],text=True)})
    with (output/'build.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    receipt['build_exit']=result.returncode
    if not result.returncode:
        result=subprocess.run([str(output/'fa3-adapter-probe')],capture_output=True,text=True)
        receipt['contract_exit']=result.returncode
        (output/'contract.stdout').write_text(result.stdout)
        (output/'contract.stderr').write_text(result.stderr)
        receipt['binary_sha256']=hashlib.sha256((output/'fa3-adapter-probe').read_bytes()).hexdigest()
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return result.returncode

if __name__=='__main__': sys.exit(build(*map(Path,sys.argv[1:])))
