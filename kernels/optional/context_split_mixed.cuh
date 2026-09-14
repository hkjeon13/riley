#pragma once
#include "context_split_fp32.cuh"
#include "../src/decode_gqa_attention_v50.cuh"
namespace riley_split_fp32 {
// One retained allocation for pure and mixed decode. Partials remain first.
struct MixedWorkspace {
    Partial partial[partial_count];
    __nv_bfloat16 query[32*576], output[32*576];
    uint32_t shape[32*416], pages[32*416], active;
};
static_assert(sizeof(MixedWorkspace)==2613252,"Rust allocation contract");
__global__ void gather_decode(const __nv_bfloat16* q,const uint32_t* meta,
                              MixedWorkspace* workspace,unsigned capacity) {
    unsigned row=blockIdx.x,lane=threadIdx.x,active=meta[5],total=meta[9];
    if(row==0 && lane==0)workspace->active=active<=32?active:0;
    if(lane==0)workspace->shape[row*416+1]=4096; // Invalid for every supported context.
    if(active>32 || row>=active || total>capacity)return;
    const uint32_t* shape=meta+32+row*416;
    unsigned offset=shape[16];
    if(shape[18]!=1 || shape[2]!=1 || offset>=total)return;
    if(lane==0)workspace->shape[row*416+1]=shape[1];
    for(unsigned i=lane;i<256;i+=32)workspace->pages[row*416+i]=shape[32+i];
    for(unsigned i=lane;i<576;i+=32)workspace->query[row*576+i]=q[offset*576+i];
}
__global__ void scatter_decode(const MixedWorkspace* workspace,__nv_bfloat16* out,
                               const uint32_t* meta,unsigned capacity) {
    unsigned row=blockIdx.x,lane=threadIdx.x,active=meta[5],total=meta[9];
    if(active>32 || row>=active || total>capacity)return;
    const uint32_t* shape=meta+32+row*416;unsigned offset=shape[16];
    if(shape[18]!=1 || shape[2]!=1 || offset>=total || workspace->shape[row*416+1]>=4096)return;
    for(unsigned i=lane;i<576;i+=32)out[offset*576+i]=workspace->output[row*576+i];
}
inline cudaError_t enqueue_mixed(cudaStream_t stream,const __nv_bfloat16* q,
                                const __nv_bfloat16* k,const __nv_bfloat16* v,
                                __nv_bfloat16* out,float* scores,const uint32_t* meta,
                                MixedWorkspace* workspace,unsigned capacity,unsigned context) {
    if(!workspace || context<1 || context>4096)return cudaErrorInvalidValue;
    gather_decode<<<32,32,0,stream>>>(q,meta,workspace,capacity);
    riley_gqa50_attention::scores<<<dim3(min((context+7)/8,32u),96),32,0,stream>>>(
        workspace->query,k,scores,workspace->shape,workspace->pages,&workspace->active);
    auto error=enqueue(stream,scores,v,workspace->partial,workspace->output,
                       workspace->shape,workspace->pages,&workspace->active,32,context);
    if(error!=cudaSuccess)return error;
    scatter_decode<<<32,32,0,stream>>>(workspace,out,meta,capacity);
    return cudaGetLastError();
}
}
