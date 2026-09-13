#!/usr/bin/env python3
"""Offline same-input precision experiment; never builds or selects a serving backend."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--nvcc',type=Path,required=True);p.add_argument('--headers',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
source=a.source.resolve();probe=(source/'benchmarks/analysis/flashinfer_prefill_probe.cu').read_text();adapter=(source/'kernels/optional/flashinfer_prefill.cu').read_text()
# Record the numerical inputs independently of their 16-bit storage format.
probe=probe.replace(' CHECK(cudaMemcpy(q,hq.data()', r''' if(argc>1){
  std::string path=std::string(argv[1])+".inputs.f32";FILE* f=std::fopen(path.c_str(),"wb");if(!f)return 13;
  for(const auto* values:{&hq,&hk,&hv})for(auto value:*values){float x=__bfloat162float(value);if(std::fwrite(&x,4,1,f)!=1)return 14;}
  if(std::fclose(f)!=0)return 15;
 }
 CHECK(cudaMemcpy(q,hq.data()''').replace('#include <cstring>','#include <cstring>\n#include <string>')
probe=probe.replace(' auto random=', ' float qk_scale=argc>2?std::atof(argv[2]):1.F;\n auto random=')
probe=probe.replace('for(auto& x:hq)x=__float2bfloat16_rn(random());for(auto& x:hk)x=__float2bfloat16_rn(random());', 'for(auto& x:hq)x=__float2bfloat16_rn(random()*qk_scale);for(auto& x:hk)x=__float2bfloat16_rn(random()*qk_scale);')
# Build only against the verified overlay, including pinned original provenance.
import sys
sys.path.insert(0,str(source/'kernels/optional'))
from prepare_flashinfer_prefill_overlay import prepare
receipt=prepare(a.headers,a.output/'headers')
includes=[a.output/'headers/include',a.headers/'cccl/libcudacxx/include',a.headers/'cccl/cub',a.headers/'cccl/thrust',source/'kernels/optional',source/'kernels/src']
wrapper=r'''
#include "mixed_attention_v49.cuh"
__global__ void existing_attention(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* o,const unsigned* packet,const unsigned* status){
 if(*status || blockIdx.z>=packet[5])return;
 const unsigned* shape=packet+32+blockIdx.z*416;unsigned offset=shape[16];
 riley_mixed_attention::attention_body<8>(q+offset*576,k,v,o+offset*576,1024,0,reinterpret_cast<const int*>(shape+1),shape+32,shape+2,blockIdx.x);
}
int existing_run(cudaStream_t stream,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* o,const unsigned* packet,const unsigned* status){
 existing_attention<<<dim3(1024,9,32),32,0,stream>>>(q,k,v,o,packet,status);return cudaGetLastError();
}
'''
# The baseline wrapper calls the actual shared attention body, using the same
# packet and FlashInfer planner validation to preserve the invalid-suffix test.
base=probe.replace('int main(',wrapper+'\nint main(').replace('riley_flashinfer_prefill_run(stream,q,k,v,o,ws,bytes)','existing_run(stream,q,k,v,o,packet,status)')
# FP16 receives exactly the BF16-rounded input values of the other two lanes.
# Round its outputs back to BF16 before scoring, matching model output storage.
half=probe.replace('__nv_bfloat16','__half').replace('__bfloat162float','__half2float').replace('__float2bfloat16_rn','input_round')
half=half.replace('int main(','''#include <cuda_fp16.h>
__half input_round(float x){return __float2half_rn(__bfloat162float(__float2bfloat16_rn(x)));}
int main(''')
needle='  if(test==3||(mixed&&'
assert needle in half
half=half.replace(needle,'  for(auto& x:ho)x=input_round(__half2float(x));\n'+needle)
half_adapter=adapter.replace('__nv_bfloat16','__half')
variants={'existing':(base,adapter),'flashinfer_bf16':(probe,adapter),'flashinfer_fp16_bf16_output':(half,half_adapter)}
report={'scope':'synthetic equal BF16 input values; not serving or model-quality evidence','overlay':receipt,'variants':{}}
for name,(harness,kernel) in variants.items():
 folder=a.output/name;folder.mkdir();h=folder/'probe.cu';k=folder/'adapter.cu';h.write_text(harness);k.write_text(kernel)
 cmd=[str(a.nvcc),'-std=c++17','-O3','-lineinfo','-arch=sm_89',*[f'-I{x}' for x in includes],str(h),str(k),'-o',str(folder/'probe')]
 run=subprocess.run(cmd,capture_output=True,text=True);(folder/'build.log').write_text(run.stdout+run.stderr)
 assert run.returncode==0,(name,run.stderr[-2000:])
 scales={}
 for scale in ['0.125','1','4']:
  prefix=folder/('scale-'+scale)
  run=subprocess.run([str(folder/'probe'),str(prefix)+'.bin',scale],capture_output=True,text=True);Path(str(prefix)+'.log').write_text(run.stdout+run.stderr)
  assert run.returncode==0,(name,scale,run.stdout,run.stderr)
  rows=[dict(zip(['case','query_rows','values','max_abs','rmse'],map(float,m))) for m in re.findall(r'case=(\d+) query_rows=(\d+) values=(\d+) max_abs=([\d.e+-]+) rmse=([\d.e+-]+)',run.stdout)]
  assert len(rows)==4
  scales[scale]={'input_float32_sha256':hashlib.sha256(Path(str(prefix)+'.bin.inputs.f32').read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(Path(str(prefix)+'.bin').read_bytes()).hexdigest(),'cases':rows,'weighted_rmse':(sum(r['rmse']**2*r['values'] for r in rows)/sum(r['values'] for r in rows))**.5}
 report['variants'][name]={'scales':scales,'output_storage':'fp16-rounded-to-bf16' if 'fp16' in name else 'bf16','command':cmd,'source_sha256':hashlib.sha256(harness.encode()).hexdigest(),'adapter_sha256':hashlib.sha256(kernel.encode()).hexdigest(),'binary_sha256':hashlib.sha256((folder/'probe').read_bytes()).hexdigest()}
for scale in ['0.125','1','4']:
 assert len({v['scales'][scale]['input_float32_sha256'] for v in report['variants'].values()})==1
report['all_input_values_bitwise_equal']=True
(a.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
