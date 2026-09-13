"""Offline isolated full-model experiment; preserves the normal serving checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();source=a.source.resolve();target=a.output.resolve()
if source==target or source in target.parents:
    raise ValueError('output must be outside source')
shutil.copytree(source,target,ignore=shutil.ignore_patterns('.git','target','__pycache__','._*'))
f=target/'kernels/src/decode_shared32_model.cuh';s=f.read_text()
anchor=' if(grouped_attention)riley_gqa50_attention::enqueue'
assert s.count(anchor)==1
start=s.index(anchor)
end=s.index('\n }\n clear_inactive_hidden',start)
old=s[start:end]
s=s[:start]+''' if(ffn_pipeline){
  riley_persistent_layer::Args args{b(1),w(273+layer*3),w(274+layer*3),
    w(275+layer*3),w(layer+1<30?base+9:1),b(4),w(base+4),b(0),w(base+5),static_cast<float*>(scratch[10]),
    b(11),b(0),b(1),static_cast<float*>(scratch[7]),pointwise,b(3),lk,lv,shape,pages};
  err=riley_persistent_layer::enqueue(stream,persistent_plan,args);
  if(err!=cudaSuccess)return err;
 }else{
'''+old+ '\n }'+s[end:]
s=s.replace('#include "../optional/ffn_pipeline.cuh"',
            '#include "../optional/ffn_pipeline.cuh"\n#include "../optional/persistent_layer.cuh"')
anchor=' auto err=cudaMemsetAsync(status,0,4,stream);'
assert s.count(anchor)==1
s=s.replace(anchor,''' riley_persistent_layer::Plan persistent_plan;
 if(ffn_pipeline){
  if(!grouped_attention||!tiled||attention_workspace)return cudaErrorInvalidValue;
  auto prepared=riley_persistent_layer::prepare(&persistent_plan);
  if(prepared!=cudaSuccess)return prepared;
 }
'''+anchor)
f.write_text(s)
f=target/'crates/riley-runtime/src/llama/graph_decode_full.rs';s=f.read_text()
s=s.replace('riley.experimental.ffn-pipeline.exact-order.v1','riley.experimental.persistent-layer.exact-order.v1')
anchor='            hash.update(include_bytes!("../../../../kernels/optional/ffn_pipeline.cuh"));'
assert s.count(anchor)==1
s=s.replace(anchor,'\n'.join('            hash.update(include_bytes!("../../../../'+name+'"));' for name in [
    'kernels/optional/persistent_layer.cuh','kernels/optional/attention_tasks.cuh','kernels/src/decode_shared32.cuh','kernels/src/decode_gate_v56.cuh',
    'kernels/src/decode_merge_norm_v56.cuh','kernels/src/decode_shared32_model.cuh']))
f.write_text(s)
f=target/'crates/riley-server/src/main.rs';s=f.read_text()
assert 'pipeline-experimental-v1' in s
f.write_text(s.replace('pipeline-experimental-v1','persistent-layer-diagnostic-v1'))
files=['kernels/src/decode_shared32_model.cuh','kernels/optional/persistent_layer.cuh','kernels/optional/attention_tasks.cuh',
       'kernels/src/decode_gate_v56.cuh','kernels/src/decode_merge_norm_v56.cuh',
       'crates/riley-runtime/src/llama/graph_decode_full.rs','crates/riley-server/src/main.rs']
receipt={name:{'source_sha256':hashlib.sha256((source/name).read_bytes()).hexdigest(),
               'diagnostic_sha256':hashlib.sha256((target/name).read_bytes()).hexdigest()} for name in files}
(target/'persistent-experiment.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(target)
