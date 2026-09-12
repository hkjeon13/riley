from pathlib import Path
import subprocess,os
r=Path('/tmp/riley-opt-260912');d=r/'mixed-attention-v49';s=(d/'probe.cu').read_text();old='for(unsigned blocks:{64,128,256}){riley_mixed_attention::compact_attention';new='for(unsigned blocks:{64,128,256}){for(int i=0;i<capacity*576+16;++i)b[i]=__ushort_as_bfloat16(0x4123);riley_mixed_attention::compact_attention';assert old in s;s=s.replace(old,new);a=s.index('if(timing){');b=s.index('v[at]=old',a);t=s[a:b].replace('if(timing){','if(timing){unsigned benchmark_capacity=total<=512?512:1024;').replace('dim3(128,9,32)','dim3((benchmark_capacity+7)/8,9,owners<=4?4:32)').replace('b,capacity,m','b,benchmark_capacity,m');s=s[:a]+t+s[b:];(d/'probe-guards.cu').write_text(s)
env=os.environ.copy();env.update(PATH='/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip()
jobs=[('guards-build',['nvcc','-std=c++17','-arch=sm_89','-O3','--fmad=false','-I'+str(d),'-I'+str(r/'prefill-shapes-source-v11/kernels/src'),str(d/'probe-guards.cu'),'-o',str(d/'probe-guards')]),('guards-correctness',[str(d/'probe-guards')]),('guards-memcheck',['/data/cuda-12.8.1/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',str(d/'probe-guards')])]
jobs += [('guards-timing',[str(d/'probe-guards'),'timing'])]
for name,args in jobs:
 with (d/(name+'.log')).open('w') as log:subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
