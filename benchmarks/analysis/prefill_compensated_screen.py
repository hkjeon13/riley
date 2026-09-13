#!/usr/bin/env python3
"""Offline same-input denominator and residual precision experiment; never builds or selects a serving backend."""
import argparse,hashlib,json,re,subprocess,shutil
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
from experimental_prefill_residual import transform
variants={'flashinfer_bf16':(probe,adapter),'fp32_denominator':(probe,adapter),'bf16_probability_residual':(probe,adapter)}
report={'scope':'synthetic equal BF16 input values; not serving or model-quality evidence','overlay':receipt,'variants':{}}
for name,(harness,kernel) in variants.items():
 folder=a.output/name;folder.mkdir();h=folder/'probe.cu';k=folder/'adapter.cu';h.write_text(harness);k.write_text(kernel)
 variant_includes=includes
 if name!='flashinfer_bf16':
  shutil.copytree(a.output/'headers/include',folder/'include')
  header=folder/'include/flashinfer/attention/prefill.cuh';text=header.read_text()
  text=transform(text,name=='bf16_probability_residual')
  header.write_text(text)
  variant_includes=[folder/'include',*includes]
 cmd=[str(a.nvcc),'-std=c++17','-O3','-lineinfo','-arch=sm_89',*[f'-I{x}' for x in variant_includes],str(h),str(k),'-o',str(folder/'probe')]
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
 report['variants'][name]={'scales':scales,'output_storage':'bf16','prefill_header_sha256':hashlib.sha256(((folder/'include') if name!='flashinfer_bf16' else a.output/'headers/include').joinpath('flashinfer/attention/prefill.cuh').read_bytes()).hexdigest(),'command':cmd,'source_sha256':hashlib.sha256(harness.encode()).hexdigest(),'adapter_sha256':hashlib.sha256(kernel.encode()).hexdigest(),'binary_sha256':hashlib.sha256((folder/'probe').read_bytes()).hexdigest()}
for scale in ['0.125','1','4']:
 assert len({v['scales'][scale]['input_float32_sha256'] for v in report['variants'].values()})==1
report['all_input_values_bitwise_equal']=True
(a.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
