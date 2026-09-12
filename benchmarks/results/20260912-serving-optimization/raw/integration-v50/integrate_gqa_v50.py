from pathlib import Path
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11';k=s/'kernels/src'
h=(r/'gqa-attention-v50-independent/gqa_attention_v50.cuh').read_text()
start=h.index('// Three query heads');end=h.index('__global__ void independent_values')
h=h[:start]+h[end:];h=h[:h.rfind('}')]+'''inline void enqueue(cudaStream_t stream,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,float* scratch,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows,uint32_t context){
 scores<<<dim3(min((context+7)/8,32u),96),32,0,stream>>>(q,k,scratch,shape,pages,live_rows);
 independent_values<<<dim3(8,96),96,0,stream>>>(scratch,v,out,shape,pages,live_rows);
}
}
''';(k/'decode_gqa_attention_v50.cuh').write_text(h)
p=k/'decode_shared32_model.cuh';x=p.read_text().replace('#include "decode_shared32_attention.cuh"','#include "decode_shared32_attention.cuh"\n#include "decode_gqa_attention_v50.cuh"').replace('bool tiled=false){','bool tiled=false,bool grouped_attention=false){')
x=x.replace(' riley_shared32_attention::enqueue(stream,b(3),lk,lv,b(4),static_cast<float*>(scratch[7]),shape,pages,active,context);',' if(grouped_attention)riley_gqa50_attention::enqueue(stream,b(3),lk,lv,b(4),static_cast<float*>(scratch[7]),shape,pages,active,context);\n else riley_shared32_attention::enqueue(stream,b(3),lk,lv,b(4),static_cast<float*>(scratch[7]),shape,pages,active,context);');p.write_text(x)
p=k/'graph_numerics_precise.cu';x=p.read_text();start=x.index('cudaError_t enqueue_compiled_v5_shared_model(');end=x.index('\n}',start)+2;f=x[start:end].replace('enqueue_compiled_v5_shared_model','enqueue_compiled_v7_gqa_shared_model').replace('context,tiled);','context,tiled,true);');x=x[:end]+'\n'+f+x[end:];p.write_text(x)
p=k/'ffi_internal.hpp';x=p.read_text();line=next(l for l in x.splitlines() if 'cudaError_t enqueue_compiled_v5_shared_model(' in l);x=x.replace(line,line+'\n'+line.replace('enqueue_compiled_v5_shared_model','enqueue_compiled_v7_gqa_shared_model'));p.write_text(x)
p=k/'graph_resources.cu';x=p.read_text();old='Rows==16?enqueue_compiled_v4_shared_model:enqueue_compiled_v5_shared_model';assert x.count(old)==1;x=x.replace(old,'Rows==16?enqueue_compiled_v4_shared_model:(Mixed?enqueue_compiled_v7_gqa_shared_model:enqueue_compiled_v5_shared_model)');p.write_text(x)
p=s/'crates/riley-cuda/build.rs';x=p.read_text();line='        kernels_dir.join("src/decode_shared32_attention.cuh"),';assert line in x;x=x.replace(line,line+'\n        kernels_dir.join("src/decode_gqa_attention_v50.cuh"),');p.write_text(x)
