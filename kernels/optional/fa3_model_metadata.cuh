#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace riley_fa3_model {
constexpr unsigned Requests=32, Columns=256, PacketWords=32+Requests*416;
struct alignas(256) Workspace {
    alignas(256) int pages[Requests*Columns];
    alignas(256) int q_indptr[Requests+1];
    alignas(256) int kv_lengths[Requests];
    alignas(256) int splits[Requests];
    alignas(256) int m_blocks[Requests];
    alignas(256) int batch_indices[Requests];
    alignas(256) int nheads[Requests];
    alignas(256) int semaphore;
    alignas(256) float lse[1024*9];
};
// One CTA validates and publishes a complete mixed packet. Only valid physical
// IDs may enter the table; q offsets/lengths stay zero until every request passes.
#ifdef RILEY_FA3_METADATA_IMPLEMENTATION
static __global__ void prepare(const unsigned *packet, Workspace *w,
    unsigned physical, unsigned capacity, unsigned context, unsigned *status) {
    __shared__ unsigned bad, inherited, active, total, offsets[33], lengths[32], counts[32];
    for(unsigned i=threadIdx.x;i<Requests*Columns;i+=blockDim.x) w->pages[i]=0;
    if(threadIdx.x<=Requests) w->q_indptr[threadIdx.x]=0;
    if(threadIdx.x<Requests) w->kv_lengths[threadIdx.x]=0;
    if(threadIdx.x==0) {
        bad=0; inherited=*status; active=packet[5]; total=packet[9];
        unsigned cursor=0;
        if(active>Requests || total>capacity || (!active && total) || (active && !total)) bad=1;
        if(!bad && !inherited) for(unsigned r=0;r<active;++r) {
            const unsigned *s=packet+32+r*416;
            unsigned n=s[1]+1, q=s[2];
            if(!q || q>n || n>context || s[16]!=cursor || q>total-cursor ||
               s[18]>1 || (s[18]==1 && q!=1)) {bad=2;break;}
            offsets[r]=cursor; lengths[r]=n; counts[r]=(n+15)/16; cursor+=q;
        }
        if(!bad && !inherited && cursor!=total) bad=16;
        if(!bad && !inherited) for(unsigned r=active;r<=Requests;++r) offsets[r]=cursor;
    }
    __syncthreads();
    bool valid_header=!bad && !inherited;
    __syncthreads(); // Complete all shared bad reads before any atomic page-error write.
    if(valid_header) for(unsigned i=threadIdx.x;i<active*Columns;i+=blockDim.x) {
        unsigned r=i/Columns, p=i%Columns;
        if(p<counts[r]) {
            unsigned page=packet[32+r*416+32+p];
            if(page>=physical) atomicOr(&bad,4u);
            else w->pages[i]=page;
        }
    }
    __syncthreads();
    if(bad || inherited) {if(threadIdx.x==0 && bad) atomicOr(status,bad);return;}
    if(threadIdx.x<=Requests) w->q_indptr[threadIdx.x]=offsets[threadIdx.x];
    if(threadIdx.x<active) w->kv_lengths[threadIdx.x]=lengths[threadIdx.x];
}
#endif
}
