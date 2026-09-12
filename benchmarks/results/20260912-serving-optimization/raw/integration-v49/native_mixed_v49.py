from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');d=Path('/tmp/riley-opt-260912/mixed-model-map-v49')
s=(d/'mixed_attention_v49.cuh').read_text();a=s.index('// Each CTA');b=s.index('__global__ void mapped_attention',a);s=s[:a]+s[b:];(r/'kernels/src/mixed_attention_v49.cuh').write_text(s)
s=(d/'mixed_rope_v49.cuh').read_text().replace('packed_prefill_rope_kv','mixed_rope_kv_v7');(r/'kernels/src/mixed_rope_v49.cuh').write_text(s)
s=(d/'mixed_model_v49.cuh').read_text().replace('packed_prefill_rope_kv','mixed_rope_kv_v7').replace('packed_prefill_select_hidden','mixed_select_hidden_v7').replace('enqueue_packed_prefill_model','enqueue_mixed_model_v7').replace('if(threadIdx.x==0)publish[owner]','if(publish&&threadIdx.x==0)publish[owner]').replace('||!publish||','||').replace('validate V3 packet','validate V7 packet');(r/'kernels/src/mixed_model_v49.cuh').write_text(s)
p=r/'kernels/src/graph_numerics_precise.cu';s=p.read_text();s+='''
#include "mixed_model_v49.cuh"
namespace riley_cuda_internal {
cudaError_t enqueue_compiled_v7_prefill_model(cudaStream_t s,void*const* d,const void*const* w,const void* m,void* k,void* v,const void* c,const void* sn,void* selected,uint32_t* status,uint32_t* publish,uint32_t rows,uint32_t physical,bool tiled) noexcept {
 return enqueue_mixed_model_v7<32>(s,d,w,m,k,v,c,sn,selected,status,publish,rows,physical,tiled);
}
cudaError_t enqueue_compiled_v7_shared_result(cudaStream_t s,const void* m,const void* logits,const void* status,void* result) noexcept {
 return riley_shared32_result::enqueue<0x37524d52>(s,m,static_cast<const __nv_bfloat16*>(logits),static_cast<const uint32_t*>(status),result);
}
}
''';p.write_text(s)
p=r/'kernels/src/ffi_internal.hpp';s=p.read_text()
for name in ['prefill_model','shared_result']:
 line=next(x for x in s.splitlines() if x.startswith('cudaError_t enqueue_compiled_v6_'+name+'('));s=s.replace(line,line+'\n'+line.replace('v6_','v7_'))
p.write_text(s)
p=r/'kernels/src/decode_shared32_result.cuh';s=p.read_text().replace('result[19]=m[4];','result[19]=Magic==0x37524d52?shape[18]:m[4];');p.write_text(s)
p=r/'kernels/src/compact_shared_result.cuh';s=p.read_text().replace('bool Packed=false>','bool Packed=false,bool Mixed=false>').replace('result[19]=m[4];','result[19]=Mixed?shape[18]:m[4];').replace('(Packed?0x36524d52U:', '(Mixed?0x37524d52U:Packed?0x36524d52U:').replace('finish<Rows,Packed><<<','finish<Rows,Packed,Mixed><<<');p.write_text(s)
p=r/'kernels/src/graph_resources.cu';s=p.read_text().replace('bool packed_prefill=false;', 'bool packed_prefill=false;\n  bool mixed_execution=false;')
s=s.replace('(128+r->variable_rows*1664+4096)', '(128+r->variable_rows*1664+4096+(r->mixed_execution?4096:0))')
s=s.replace('r->packed_prefill?valid_v6_shape_packet', 'r->mixed_execution?valid_v7_shape_packet(source,bytes,r->v3_prefill_physical,32):r->packed_prefill?valid_v6_shape_packet')
s=s.replace('r->packed_prefill&&value(16)==0?', 'r->packed_prefill&&value(16)!=1?').replace('if(stage==0){if(!r->prefill_exec)', 'if(stage!=1){if(!r->prefill_exec)')
s=s.replace('if(stage>1||!r->compact_exec[stage])', 'if(stage>(r->mixed_execution?2u:1u)||!r->compact_exec[stage==1?1:0])').replace('selected_exec=r->compact_exec[stage];','selected_exec=r->compact_exec[stage==1?1:0];')
s=s.replace('bool Packed=false>\nstatic RileyCudaStatus record_variable_shared', 'bool Packed=false,bool Mixed=false>\nstatic RileyCudaStatus record_variable_shared')
s=s.replace('constexpr uint64_t request_bytes=128+Rows*1664+4096;', 'static_assert(!Mixed||(Packed&&Rows==32),"mixed V7 requires packed32");\n constexpr uint64_t request_bytes=128+Rows*1664+4096+(Mixed?4096:0);')
s=s.replace('riley_compact_result::enqueue<Rows,Packed>', 'riley_compact_result::enqueue<Rows,Packed,Mixed>')
s=s.replace('(Packed?enqueue_compiled_v6_shared_result:', '(Mixed?enqueue_compiled_v7_shared_result:Packed?enqueue_compiled_v6_shared_result:')
s=s.replace('runtime_error(enqueue_compiled_v6_prefill_model(', 'runtime_error((Mixed?enqueue_compiled_v7_prefill_model:enqueue_compiled_v6_prefill_model)(')
s=s.replace('runtime_error(enqueue_compiled_v6_shared_result(', 'runtime_error((Mixed?enqueue_compiled_v7_shared_result:enqueue_compiled_v6_shared_result)(')
s=s.replace('r->packed_prefill=Packed;r->compact_ready', 'r->mixed_execution=Mixed;r->packed_prefill=Packed;r->compact_ready')
for suffix in ['','_greedy']:
 a=s.index('extern "C" RileyCudaStatus riley_cuda_graph_resources_record_v6_shared'+suffix+'(');b=s.index('\n}',a)+2;t=s[a:b].replace('v6_shared','v7_shared').replace('record_variable_shared<32,false,true>', 'record_variable_shared<32,false,true,true>').replace('record_variable_shared<32,true,true>', 'record_variable_shared<32,true,true,true>');s+='\n'+t+'\n'
p.write_text(s)
for path,prefix in [('crates/riley-cuda/src/graph_resources.rs','pub fn '),('crates/riley-cuda/src/ffi.rs','pub(super) fn ')]:
 p=r/path;s=p.read_text()
 for suffix in ['','_greedy']:
  a=s.index(prefix+'record_v6_shared'+suffix+'(');brace=s.index('{',a);depth=1;b=brace+1
  while depth:depth+=(s[b]=='{')-(s[b]=='}');b+=1
  t=s[a:b].replace('v6_shared','v7_shared').replace('V6 packed','V7 mixed');s=s[:b]+'\n'+t+s[b:]
 if path.endswith('ffi.rs'):
  for suffix in ['','_greedy']:
   line=next(x for x in s.splitlines() if x.startswith('fn riley_cuda_graph_resources_record_v6_shared'+suffix+'('));s=s.replace(line,line+'\n'+line.replace('v6_shared','v7_shared'))
 p.write_text(s)
p=r/'kernels/include/riley_cuda.h';s=p.read_text()
for suffix in ['','_greedy']:
 a=s.index('RileyCudaStatus riley_cuda_graph_resources_record_v6_shared'+suffix+'(');b=s.index(';',a)+1;t=s[a:b].replace('v6_shared','v7_shared');s=s[:b]+'\n'+t+s[b:]
p.write_text(s)
p=r/'crates/riley-cuda/build.rs';s=p.read_text();marker='        kernels_dir.join("src/packed_prefill_model_v48.cuh"),';s=s.replace(marker,marker+'\n'+''.join('        kernels_dir.join("src/'+n+'"),\n' for n in ['mixed_attention_v49.cuh','mixed_rope_v49.cuh','mixed_model_v49.cuh']));p.write_text(s)
print('V7 native mixed capture and per-row stage result bindings added')
