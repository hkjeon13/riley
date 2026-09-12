from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'kernels/src/prefill_shape_projection.cuh';s=p.read_text().replace(' static_assert(N%', ' if(blockIdx.y*16>=rows)return;\n static_assert(N%',1);p.write_text(s)
p=r/'kernels/src/prefill_shape_pointwise.cuh';s=p.read_text().replace('uint32_t vocabulary,uint32_t* status){','uint32_t vocabulary,uint32_t* status,const uint32_t* stage=nullptr){',1).replace(' uint32_t token=tokens[row];',' if(stage&&*stage==1&&live!=1){if(threadIdx.x==0)atomicOr(status,2U);return;}\n uint32_t token=stage&&*stage==1?shape[0]:tokens[row];',1);p.write_text(s)
p=r/'kernels/src/prefill_shape_model.cuh';s=p.read_text().replace('shape,capacity,49152,status);','shape,capacity,49152,status,reinterpret_cast<const uint32_t*>(metadata)+4);',1);p.write_text(s)
p=r/'kernels/src/graph_resources.cu';s=p.read_text().replace('if(value(16)!=0||value(20)!=1||value(136)>r->v3_prefill_capacity','if(value(20)!=1||value(136)>r->v3_prefill_capacity',1);p.write_text(s)
p=r/'crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs';s=p.read_text().replace('e.stage==InputStage::Prefill && e.rows.len()==1 && bytes.len()==RESULT_BYTES','e.rows.len()==1 && bytes.len()==RESULT_BYTES',1);p.write_text(s)
