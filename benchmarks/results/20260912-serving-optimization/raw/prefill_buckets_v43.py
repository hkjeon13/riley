from pathlib import Path
r=Path('/tmp/riley-opt-260912');p=r/'prefill-shapes-source-v11/kernels/src/graph_resources.cu';s=p.read_text()
s=s.replace('  cudaGraphExec_t prefill_exec = nullptr;','''  cudaGraphExec_t prefill_exec = nullptr;
  // Smaller prefill DAGs share maximum-sized scratch and the same parent ledger.
  // They execute sequentially, so no additional device allocation is needed.
  struct PrefillBucket {
    uint32_t capacity = 0;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
  };
  PrefillBucket prefill_buckets[2]{};''',1)
s=s.replace('      (*resources)->catalog[0].graph ||', '''      (*resources)->prefill_buckets[0].graph || (*resources)->prefill_buckets[0].exec ||
      (*resources)->prefill_buckets[1].graph || (*resources)->prefill_buckets[1].exec ||
      (*resources)->catalog[0].graph ||''',1)
marker='    for(auto& entry:(*resources)->catalog){'
cleanup='''    for(auto& entry:(*resources)->prefill_buckets){
      if(status==RILEY_CUDA_STATUS_SUCCESS&&entry.exec){
        status=runtime_error(cudaGraphExecDestroy(entry.exec),error,RILEY_CUDA_ERROR_STAGE_CLOSE,kClose);
        if(status==RILEY_CUDA_STATUS_SUCCESS)entry.exec=nullptr;else (*resources)->completion_unknown=true;
      }
      if(status==RILEY_CUDA_STATUS_SUCCESS&&entry.graph){
        status=runtime_error(cudaGraphDestroy(entry.graph),error,RILEY_CUDA_ERROR_STAGE_CLOSE,kClose);
        if(status==RILEY_CUDA_STATUS_SUCCESS)entry.graph=nullptr;else (*resources)->completion_unknown=true;
      }
    }
'''
assert marker in s;s=s.replace(marker,cleanup+marker,1)
old='  if(r->v3_prefill_capacity){uint32_t stage;std::memcpy(&stage,source+16,4);if(stage==0){if(!r->prefill_exec)return reject(error,"V3 prefill capture missing");selected_exec=r->prefill_exec;}}'
new='''  if(r->v3_prefill_capacity){
    uint32_t stage;std::memcpy(&stage,source+16,4);
    if(stage==0){
      if(!r->prefill_exec)return reject(error,"V3 prefill capture missing");
      selected_exec=r->prefill_exec;
      uint32_t tokens;std::memcpy(&tokens,source+136,4);
      for(const auto& bucket:r->prefill_buckets){
        if(bucket.exec&&tokens<=bucket.capacity){selected_exec=bucket.exec;break;}
      }
    }
  }'''
assert old in s;s=s.replace(old,new,1)
start=s.index('static RileyCudaStatus record_variable_shared(');a=s[:start];b=s[start:]
old=' if(status==RILEY_CUDA_STATUS_SUCCESS){r->prefill_graph=r->graph;r->prefill_exec=r->exec;r->graph=nullptr;r->exec=nullptr;status=record_shape(1);}'
new=''' if(status==RILEY_CUDA_STATUS_SUCCESS){
  r->prefill_graph=r->graph;r->prefill_exec=r->exec;r->graph=nullptr;r->exec=nullptr;
  const uint32_t sizes[2]={16,128};
  for(size_t i=0;i<2&&status==RILEY_CUDA_STATUS_SUCCESS;++i){
   if(sizes[i]>=capacity)continue;
   status=record_shape(sizes[i]);
   if(status==RILEY_CUDA_STATUS_SUCCESS){
    auto& bucket=r->prefill_buckets[i];
    bucket.capacity=sizes[i];bucket.graph=r->graph;bucket.exec=r->exec;
    r->graph=nullptr;r->exec=nullptr;
   }
  }
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=record_shape(1);
 }'''
assert old in b;b=b.replace(old,new,1);p.write_text(a+b)
