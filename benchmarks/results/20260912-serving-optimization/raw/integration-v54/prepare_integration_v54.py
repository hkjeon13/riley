from pathlib import Path
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11'
def edit(p,a,b):
 p=s/p;t=p.read_text();assert a in t,(str(p),a);p.write_text(t.replace(a,b))
p=(r/'packed-value-v54/packed.cuh').read_text();a=p.index('__device__ __forceinline__ int packed_index');b=p.index('// Exceptional',a)
helper=p[a:b]
(s/'kernels/src/packed_value_v54.cuh').write_text('#pragma once\n#include <cuda_bf16.h>\n#include <stdint.h>\n// V7 owned sessions retain this page-local V layout across mixed and decode graphs.\n// Same pool extent as token-major BF16; K remains token-major.\nnamespace riley_packed_value_v54 {\n'+helper+'}\n')
p=p[:a]+'using riley_packed_value_v54::packed_index;\nusing riley_packed_value_v54::masked_pair;\n'+p[b:]
p=p.replace('#pragma once','#pragma once\n#include "packed_value_v54.cuh"').replace('riley_attention_v54','riley_mixed_attention')
(s/'kernels/src/mixed_attention_v49.cuh').write_text(p)
edit('kernels/src/mixed_model_v49.cuh','riley_mixed_attention::mapped_attention<<<','riley_mixed_attention::mapped_attention<true><<<')
edit('kernels/src/mixed_rope_v49.cuh','#pragma once','#pragma once\n#include "packed_value_v54.cuh"')
edit('kernels/src/mixed_rope_v49.cuh','values[destination]=v[row*192+base+dim];values[destination+32]=v[row*192+base+dim+32];','uint32_t vd=riley_packed_value_v54::packed_index(pos,head-9,dim,pages);\n  values[vd]=v[row*192+base+dim];values[vd+512]=v[row*192+base+dim+32];')
edit('kernels/src/decode_shared32_model.cuh','__global__ void qkv_merge_rope','template<bool PackedValue=false>\n__global__ void qkv_merge_rope')
edit('kernels/src/decode_shared32_model.cuh','values[dst]=qkv_rounded(v,192,row,base+dim);values[dst+32]=qkv_rounded(v,192,row,base+dim+32);','uint32_t vd=PackedValue?riley_packed_value_v54::packed_index(pos,head-9,dim,pages):dst;values[vd]=qkv_rounded(v,192,row,base+dim);values[vd+(PackedValue?512:32)]=qkv_rounded(v,192,row,base+dim+32);')
old='qkv_merge_rope<<<dim3(2,32),256,0,stream>>>(static_cast<float*>(scratch[7]),b(3),lk,lv,cos,sin,pages,shape,active);'
edit('kernels/src/decode_shared32_model.cuh',old,'if(grouped_attention)'+old.replace('qkv_merge_rope<<<','qkv_merge_rope<true><<<')+'\n else '+old.replace('qkv_merge_rope<<<','qkv_merge_rope<false><<<'))
edit('kernels/src/decode_gqa_attention_v50.cuh','#pragma once','#pragma once\n#include "packed_value_v54.cuh"')
old='''int dim=block*8+g;int value_base=live?((pages[at/16]*3+head/3)*16)*64+dim:0;
    auto val=[&](int pos){return pos<end&&live?v[value_base+(pos-at)*64]:zero;};
    vb[part]=riley_prefill_shape::pair(val(at+2*t),val(at+2*t+1));
    vbb[part]=riley_prefill_shape::pair(val(at+2*t+8),val(at+2*t+9));'''
new='''// Each lane reads an aligned adjacent-token pair from the packed V tile.
    int vi=live?(pages[at/16]*3+head/3)*1024+block*128:0;
    vb[part]=live?riley_packed_value_v54::masked_pair(v+vi+lane*2,at+2*t,end):0;
    vbb[part]=live?riley_packed_value_v54::masked_pair(v+vi+64+lane*2,at+2*t+8,end):0;'''
edit('kernels/src/decode_gqa_attention_v50.cuh',old,new)
edit('crates/riley-cuda/build.rs','kernels_dir.join("src/decode_gqa_attention_v50.cuh"),','kernels_dir.join("src/decode_gqa_attention_v50.cuh"),\n        kernels_dir.join("src/packed_value_v54.cuh"),')
edit('crates/riley-runtime/src/llama/graph_decode_full.rs','include_bytes!("../../../../kernels/src/decode_gqa_attention_v50.cuh").as_slice(),','include_bytes!("../../../../kernels/src/decode_gqa_attention_v50.cuh").as_slice(),\ninclude_bytes!("../../../../kernels/src/packed_value_v54.cuh").as_slice(),')
for name in ['qualify_attention_v52.py','qualify_serving_v52.py','run_v7_http_v52.py','run_v7_fallback_v52.py']:
 (r/name.replace('v52','v54')).write_text((r/name).read_text().replace('v52','v54'))
