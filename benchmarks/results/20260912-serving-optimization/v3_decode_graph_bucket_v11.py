from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'kernels/src/graph_resources.cu';s=p.read_text();needle='  auto selected_exec = r->exec;';s=s.replace(needle,needle+'''\n  if(r->v3_prefill_capacity){uint32_t stage;std::memcpy(&stage,source+16,4);if(stage==0){if(!r->prefill_exec)return reject(error,"V3 prefill capture missing");selected_exec=r->prefill_exec;}}''',1)
start=s.index('extern "C" RileyCudaStatus riley_cuda_graph_resources_record_v3_prefill(')
a=s[:start];b=s[start:];b=b.replace(' status=record_reserved_sequence(r,staging,transfer,[&]() noexcept {',' auto record_shape=[&](uint32_t row_capacity) noexcept {return record_reserved_sequence(r,staging,transfer,[&]() noexcept {',1)
b=b.replace('static_cast<uint32_t*>(d[19]->device_data),capacity,physical)','static_cast<uint32_t*>(d[19]->device_data),row_capacity,physical)',1)
b=b.replace(' },error);\n if(status==RILEY_CUDA_STATUS_SUCCESS){r->v3_prefill_capacity=', ''' },error);};
 status=record_shape(capacity);
 if(status==RILEY_CUDA_STATUS_SUCCESS){r->prefill_graph=r->graph;r->prefill_exec=r->exec;r->graph=nullptr;r->exec=nullptr;status=record_shape(1);}
 if(status==RILEY_CUDA_STATUS_SUCCESS){r->v3_prefill_capacity=''',1)
p.write_text(a+b)
p=r/'crates/riley-cuda/tests/v3_recorder_decode_gpu.rs';s=p.read_text()
s=s.replace('count:usize,replay:u64)->Vec<u8>','count:usize,replay:u64,limit:u32)->Vec<u8>').replace('(156,32)','(156,limit)')
s=s.replace('for prompt in requests {','for prompt in requests {let output_limit=match prompt.len(){16=>32,128=>64,_=>128};')
s=s.replace('1152*512','1152*1024').replace('384*512','384*1024').replace('3072*512','3072*1024').replace('2304*512','2304*1024').replace(',0,0,512,64)',',0,0,1024,64)')
s=s.replace('packet(&tokens,start,n,replay_count)','packet(&tokens,start,n,replay_count,output_limit)')
s=s.replace('for generated in 1..8 {','for generated in 1..output_limit as usize {')
s=s.replace('packet(&history,0,history.len(),replay_count)','packet(&history,0,history.len(),replay_count,output_limit)').replace('packet(&history,history.len()-1,1,replay_count)','packet(&history,history.len()-1,1,replay_count,output_limit)')
s=s.replace('p32(&mut p,152,32)','p32(&mut p,152,output_limit)')
s=s.replace('phase={} generated=8 logits_exact=true full_kv_exact=true",tokens.len(),phase','phase={} generated={} logits_exact=true full_kv_exact=true",tokens.len(),phase,output_limit')
p.write_text(s)
