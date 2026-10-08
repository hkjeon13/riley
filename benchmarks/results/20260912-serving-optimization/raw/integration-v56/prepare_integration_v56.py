from pathlib import Path
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11'
def edit(path,a,b):
 p=s/path;t=p.read_text();assert a in t,(path,a);p.write_text(t.replace(a,b))
t=(r/'decode-gate-layout-v56/gate.cuh').read_text();a=t.index('template<int Steps,bool Adaptive>');gate=t[:a]+'}\n';a=t.index('namespace riley_gate_v56 {',a);b=t.index('__global__ void hybrid',a);gate+=t[a:b]+'}\n';(s/'kernels/src/decode_gate_v56.cuh').write_text(gate)
(s/'kernels/src/decode_merge_norm_v56.cuh').write_text((r/'merge-norm-v56/merge.cuh').read_text())
edit('kernels/src/decode_shared32_model.cuh','#include "decode_gqa_attention_v50.cuh"','#include "decode_gqa_attention_v50.cuh"\n#include "decode_gate_v56.cuh"\n#include "decode_merge_norm_v56.cuh"')
old=''' enqueue_shared32_projection<576,576,128,false>(stream,b(4),w(base+4),b(2),static_cast<float*>(scratch[7]),active);
 riley_prefill_pointwise::norm_rows<<<32,256,0,stream>>>(b(2),b(0),w(base+5),scratch[10],b(1),1,pointwise,32);'''
new=''' if(grouped_attention){
  shared32_projection_parts<576,576,128,false><<<dim3(72,5),32,0,stream>>>(b(4),w(base+4),static_cast<float*>(scratch[7]),b(2),active);
  riley_merge_norm_v56::merge_norm<<<32,256,0,stream>>>(static_cast<const float*>(scratch[7]),b(0),w(base+5),scratch[10],b(1),1,pointwise,32);
 }else{
'''+old+'''\n }'''
edit('kernels/src/decode_shared32_model.cuh',old,new)
edit('kernels/src/decode_shared32_model.cuh',' if(tiled)shared32_gate_up_swiglu<true>', ' if(grouped_attention&&tiled)riley_gate_v56::split_rows<4><<<192,64,0,stream>>>(b(1),w(273+layer*3),w(274+layer*3),b(11),active);\n else if(tiled)shared32_gate_up_swiglu<true>')
old=''' if(tiled)enqueue_shared32_projection<576,1536,320,true>(stream,b(11),w(273+layer*3+2),b(4),static_cast<float*>(scratch[7]),active);
 else enqueue_shared32_projection<576,1536,320,false>(stream,b(11),w(base+8),b(4),static_cast<float*>(scratch[7]),active);
 riley_prefill_pointwise::norm_rows<<<32,256,0,stream>>>(b(4),scratch[10],w(layer+1<30?base+9:1),b(0),b(1),2,pointwise,32);'''
new=''' if(grouped_attention){
  if(tiled)shared32_projection_parts<576,1536,320,true><<<dim3(72,5),32,0,stream>>>(b(11),w(273+layer*3+2),static_cast<float*>(scratch[7]),b(4),active);
  else shared32_projection_parts<576,1536,320,false><<<dim3(72,5),32,0,stream>>>(b(11),w(base+8),static_cast<float*>(scratch[7]),b(4),active);
  riley_merge_norm_v56::merge_norm<<<32,256,0,stream>>>(static_cast<const float*>(scratch[7]),scratch[10],w(layer+1<30?base+9:1),b(0),b(1),2,pointwise,32);
 }else{
'''+old+'''\n }'''
edit('kernels/src/decode_shared32_model.cuh',old,new)
for name in ['decode_gate_v56.cuh','decode_merge_norm_v56.cuh']:
 edit('crates/riley-cuda/build.rs','kernels_dir.join("src/decode_gqa_attention_v50.cuh"),','kernels_dir.join("src/decode_gqa_attention_v50.cuh"),\n        kernels_dir.join("src/'+name+'"),')
 edit('crates/riley-runtime/src/llama/graph_decode_full.rs','include_bytes!("../../../../kernels/src/decode_gqa_attention_v50.cuh").as_slice(),','include_bytes!("../../../../kernels/src/decode_gqa_attention_v50.cuh").as_slice(),\ninclude_bytes!("../../../../kernels/src/'+name+'").as_slice(),')
for name in ['qualify_attention_v55.py','qualify_serving_v55.py','run_v7_http_v55.py','run_v7_fallback_v55.py']:
 dest=name.replace('v55','v56').replace('qualify_attention','qualify_ffn')
 (r/dest).write_text((r/name).read_text().replace('v55','v56').replace('attention-v56','ffn-v56'))
