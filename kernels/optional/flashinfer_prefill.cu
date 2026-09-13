// Experimental native paged-prefill adapter. Fixed SmolLM2 geometry; no Python runtime.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <cstdint>
#include "flashinfer_api.h"

namespace riley_flashinfer_prefill {
constexpr unsigned Requests=32, Tiles=64;
struct Metadata {
  int pages[8192], kv_indptr[33], last[32], q_indptr[33];
  int requests[Tiles], q_tiles[Tiles], kv_tiles[Tiles];
  bool valid[Tiles];
  int chunk;
};
// Single bounded planner; no host readback or Q/K/V repacking. Commit valid bits
// only after validating the complete packet, so an invalid suffix cannot publish.
__global__ void prepare(const unsigned* packet, Metadata* m, unsigned physical,
    unsigned capacity, unsigned* status) {
  if(threadIdx.x || blockIdx.x)return;
  for(unsigned i=0;i<Tiles;++i)m->valid[i]=false;
  m->chunk=4096;
  unsigned active=packet[5],total=packet[9],cursor=0,page_cursor=0,tile_cursor=0;
  if(!active || active>Requests || !total || total>capacity){atomicOr(status,1u);return;}
  for(unsigned r=0;r<active;++r){
    const unsigned* shape=packet+32+r*416;
    unsigned n=shape[1]+1,q=shape[2],offset=shape[16];
    if(!q || q>n || n>4096 || offset!=cursor || q>total-cursor){atomicOr(status,2u);return;}
    unsigned count=(n+15)/16;
    m->q_indptr[r]=cursor;m->kv_indptr[r]=page_cursor;m->last[r]=(n-1)%16+1;
    for(unsigned p=0;p<count;++p){unsigned page=shape[32+p];
      if(page>=physical){atomicOr(status,4u);return;}
      m->pages[page_cursor++]=page;
    }
    // GQA packs three query heads per KV head into the CTA Q axis.
    unsigned tiles=(q*3+127)/128;
    if(tile_cursor+tiles>Tiles){atomicOr(status,8u);return;}
    for(unsigned t=0;t<tiles;++t){m->requests[tile_cursor]=r;m->q_tiles[tile_cursor]=t;m->kv_tiles[tile_cursor]=0;++tile_cursor;}
    cursor+=q;
  }
  if(cursor!=total){atomicOr(status,16u);return;}
  for(unsigned r=active;r<=Requests;++r){m->q_indptr[r]=cursor;m->kv_indptr[r]=page_cursor;if(r<Requests)m->last[r]=0;}
  for(unsigned t=0;t<tile_cursor;++t)m->valid[t]=true;
}
}
extern "C" uint64_t riley_flashinfer_prefill_workspace_bytes() noexcept {
  return sizeof(riley_flashinfer_prefill::Metadata);
}
extern "C" int riley_flashinfer_prefill_prepare(void* stream,const void* packet,
    void* workspace,uint64_t bytes,unsigned physical,unsigned capacity,void* status) noexcept {
  if(!packet||!workspace||!status||bytes!=sizeof(riley_flashinfer_prefill::Metadata)||
      !physical||physical>4096||!capacity||capacity>1024)return cudaErrorInvalidValue;
  riley_flashinfer_prefill::prepare<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(
      static_cast<const unsigned*>(packet),static_cast<riley_flashinfer_prefill::Metadata*>(workspace),
      physical,capacity,static_cast<unsigned*>(status));
  return cudaGetLastError();
}
extern "C" int riley_flashinfer_prefill_run(void* stream,const void* q,const void* k,
    const void* v,void* out,void* workspace,uint64_t bytes) noexcept {
  if(!q||!k||!v||!out||!workspace||bytes!=sizeof(riley_flashinfer_prefill::Metadata))return cudaErrorInvalidValue;
  try {
    using namespace flashinfer;
    auto* m=static_cast<riley_flashinfer_prefill::Metadata*>(workspace);
    using Params=BatchPrefillPagedParams<__nv_bfloat16,__nv_bfloat16,__nv_bfloat16,int>;
    Params p;
    p.q=const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(q));p.o=static_cast<__nv_bfloat16*>(out);
    p.paged_kv=paged_kv_t<__nv_bfloat16,int>(3,16,64,32,QKVLayout::kHND,
      const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(k)),
      const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(v)),m->pages,m->kv_indptr,m->last);
    p.q_indptr=m->q_indptr;p.o_indptr=m->q_indptr;p.group_size=uint_fastdiv(3);p.num_qo_heads=9;
    p.q_stride_n=576;p.q_stride_h=64;p.sm_scale=.125F;p.window_left=-1;
    p.request_indices=m->requests;p.qo_tile_indices=m->q_tiles;p.kv_tile_indices=m->kv_tiles;
    p.block_valid_mask=m->valid;p.kv_chunk_size_ptr=&m->chunk;p.padded_batch_size=64;p.max_total_num_rows=1024;
    return BatchPrefillWithPagedKVCacheDispatched<128,64,64,PosEncodingMode::kNone,false,
      MaskMode::kCausal,DefaultAttention<false,false,false,false>,Params>(
        p,nullptr,nullptr,false,static_cast<cudaStream_t>(stream));
  }catch(...){return cudaErrorUnknown;}
}
