from pathlib import Path
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11';p=src/'kernels/src/graph_resources.cu';s=p.read_text()
s=s.replace('#include "ffi_internal.hpp"','#include "ffi_internal.hpp"\n#include "compact_shared_result.cuh"',1)
s=s.replace('  cudaGraphExec_t prefill_exec = nullptr;','''  cudaGraphExec_t prefill_exec = nullptr;
  cudaGraph_t compact_graph[2]{};
  cudaGraphExec_t compact_exec[2]{};
  bool compact_ready=false,last_compact=false;''',1)
s=s.replace('      (*resources)->catalog[0].graph ||','''      (*resources)->compact_graph[0] || (*resources)->compact_exec[0] ||
      (*resources)->compact_graph[1] || (*resources)->compact_exec[1] ||
      (*resources)->catalog[0].graph ||''',1)
marker='    for(auto& entry:(*resources)->catalog){'
block='''    for(int i=0;i<2;++i){
      if(status==RILEY_CUDA_STATUS_SUCCESS&&(*resources)->compact_exec[i]){
        status=runtime_error(cudaGraphExecDestroy((*resources)->compact_exec[i]),error,RILEY_CUDA_ERROR_STAGE_CLOSE,kClose);
        if(status==RILEY_CUDA_STATUS_SUCCESS)(*resources)->compact_exec[i]=nullptr;else (*resources)->completion_unknown=true;
      }
      if(status==RILEY_CUDA_STATUS_SUCCESS&&(*resources)->compact_graph[i]){
        status=runtime_error(cudaGraphDestroy((*resources)->compact_graph[i]),error,RILEY_CUDA_ERROR_STAGE_CLOSE,kClose);
        if(status==RILEY_CUDA_STATUS_SUCCESS)(*resources)->compact_graph[i]=nullptr;else (*resources)->completion_unknown=true;
      }
    }
'''
assert marker in s;s=s.replace(marker,block+marker,1)
s=s.replace('  r->completion_visible = false;\n  r->last_catalog = 0;','  r->completion_visible = false;\n  r->last_compact = false;\n  r->last_catalog = 0;',1)
marker='  if(r->decode_capacity!=0){'
block='''  if(r->compact_ready){
    uint32_t mode,stage;std::memcpy(&mode,source+32,4);std::memcpy(&stage,source+16,4);
    if(mode==0){
      if(stage>1||!r->compact_exec[stage])return reject(error,"compact stage capture missing");
      selected_exec=r->compact_exec[stage];r->last_compact=true;
    }
  }
'''
assert marker in s;s=s.replace(marker,block+marker,1)
s=s.replace('bytes != r->transfer_bytes)\n    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,','bytes != (r->last_compact?r->variable_rows*128:r->transfer_bytes))\n    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,',1)
a=s.index('template<uint32_t Rows>\nstatic RileyCudaStatus record_variable_shared(');b=s.index('extern "C" RileyCudaStatus riley_cuda_graph_resources_record_v3_shared(',a)
part=s[a:b].replace('template<uint32_t Rows>','template<uint32_t Rows,bool Compact=false>',1)
part=part.replace(' static_assert(Rows==8||Rows==16,"wire capacity");',' static_assert(Rows==8||Rows==16,"wire capacity");\n static_assert(!Compact||Rows==16,"compact capture currently requires sixteen rows");',1)
part=part.replace('auto record_shape=[&](uint32_t row_capacity)','auto record_shape=[&](uint32_t row_capacity,bool compact=false)',1)
marker='   if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error((Rows==8?enqueue_compiled_v3_shared_result:'
idx=part.index(marker)
block='''   if(compact){
    if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(riley_compact_result::enqueue<Rows>(r->stream->stream,d[16]->device_data,d[24]->device_data,static_cast<uint32_t*>(d[18]->device_data),d[7]->device_data,d[25]->device_data),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"compact decode validation");
    if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(host+transfer,d[25]->device_data,Rows*128,cudaMemcpyDeviceToHost);
    return result;
   }
'''
part=part[:idx]+block+part[idx:]
part=part.replace('if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(cudaMemsetAsync(d[25]','if(!compact&&result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(cudaMemsetAsync(d[25]',1)
marker='  if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(enqueue_decode_argmax('
idx=part.index(marker)
block='''  if(compact){
   if(result==RILEY_CUDA_STATUS_SUCCESS)result=runtime_error(riley_compact_result::enqueue<Rows>(r->stream->stream,d[16]->device_data,d[20]->device_data,static_cast<uint32_t*>(d[18]->device_data),d[7]->device_data,d[25]->device_data),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,"compact prefill validation");
   if(result==RILEY_CUDA_STATUS_SUCCESS)result=copy(host+transfer,d[25]->device_data,Rows*128,cudaMemcpyDeviceToHost);
   return result;
  }
'''
part=part[:idx]+block+part[idx:]
old=' if(status==RILEY_CUDA_STATUS_SUCCESS){r->prefill_graph=r->graph;r->prefill_exec=r->exec;r->graph=nullptr;r->exec=nullptr;status=record_shape(1);}'
new=''' if(status==RILEY_CUDA_STATUS_SUCCESS){
  r->prefill_graph=r->graph;r->prefill_exec=r->exec;r->graph=nullptr;r->exec=nullptr;
  if constexpr(Compact){
   for(int stage=0;stage<2&&status==RILEY_CUDA_STATUS_SUCCESS;++stage){
    status=record_shape(stage==0?capacity:1,true);
    if(status==RILEY_CUDA_STATUS_SUCCESS){r->compact_graph[stage]=r->graph;r->compact_exec[stage]=r->exec;r->graph=nullptr;r->exec=nullptr;}
   }
  }
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=record_shape(1);
 }'''
assert old in part;part=part.replace(old,new,1).replace('{r->v3_shared=true;','{r->compact_ready=Compact;r->v3_shared=true;',1)
s=s[:a]+part+s[b:]
def extract(text,start):
 a=text.index(start);brace=text.index('{',a);n=1;i=brace+1
 while n:
  if text[i]=='{':n+=1
  elif text[i]=='}':n-=1
  i+=1
 return a,i,text[a:i]
a,b,fn=extract(s,'extern "C" RileyCudaStatus riley_cuda_graph_resources_record_v4_shared(')
s=s[:b]+'\n'+fn.replace('record_v4_shared(','record_v4_shared_greedy(').replace('record_variable_shared<16>','record_variable_shared<16,true>')+s[b:];p.write_text(s)
p=src/'kernels/include/riley_cuda.h';s=p.read_text();a=s.index('RileyCudaStatus riley_cuda_graph_resources_record_v4_shared(');b=s.index(';',a)+1;s=s[:b]+'\n// Retains full and compact greedy stage graphs under the same parent ledger.\n'+s[a:b].replace('record_v4_shared(','record_v4_shared_greedy(')+s[b:];p.write_text(s)
for path,marker in [('crates/riley-cuda/src/ffi.rs','    pub(super) fn record_v4_shared('),('crates/riley-cuda/src/graph_resources.rs','    pub fn record_v4_shared(')]:
 p=src/path;s=p.read_text();a,b,fn=extract(s,marker);s=s[:b]+'\n'+fn.replace('record_v4_shared','record_v4_shared_greedy')+s[b:]
 if path.endswith('ffi.rs'):
  a=s.index('    fn riley_cuda_graph_resources_record_v4_shared(');b=s.index(';',a)+1;s=s[:b]+'\n'+s[a:b].replace('record_v4_shared','record_v4_shared_greedy')+s[b:]
 p.write_text(s)
p=src/'crates/riley-cuda/build.rs';s=p.read_text();line='        kernels_dir.join("src/prefill_query_tile_attention.cuh"),';assert line in s;s=s.replace(line,line+'\n        kernels_dir.join("src/compact_shared_result.cuh"),');p.write_text(s)
