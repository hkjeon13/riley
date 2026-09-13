#pragma once
#include <algorithm>
#include <cooperative_groups.h>
#include "attention_tasks.cuh"

// Finite diagnostic attention DAG. The native owner must validate all page
// extents, distinct output/scratch spans, and immutable metadata before launch.
namespace riley_persistent_attention {
struct Args {
 const __nv_bfloat16 *q,*k,*v;
 __nv_bfloat16* out;
 float* scores;
 const uint32_t *shape,*pages,*active;
};
__global__ void execute(Args a){
 const unsigned rows=*a.active;
 if(!rows||rows>32)return;
 auto grid=cooperative_groups::this_grid();
 __shared__ unsigned prefix[33];
 __shared__ __nv_bfloat16 probs[8][128];
 __shared__ float exps[8][128];
 if(threadIdx.x==0){
  prefix[0]=0;
  for(unsigned r=0;r<rows;++r){
   unsigned count=a.shape[r*416+1]+1;
   prefix[r+1]=prefix[r]+(count>=1&&count<=4096?3*((count+7)/8):0);
  }
 }
 __syncthreads();
 unsigned warp=threadIdx.x/32,worker=blockIdx.x*8+warp,workers=gridDim.x*8;
 for(unsigned task=worker;task<prefix[rows];task+=workers){
  unsigned row=0;while(row+1<rows&&task>=prefix[row+1])++row;
  unsigned tiles=(a.shape[row*416+1]+8)/8,local=task-prefix[row];
  riley_attention_tasks::score_task(a.q,a.k,a.scores,a.shape,a.pages,a.active,
      row,local/tiles,(local%tiles)*8,tiles*8);
 }
 grid.sync();
 // Interleave rows across resident warp workers to distribute ragged contexts.
 for(unsigned task=worker;task<rows*72;task+=workers){
  riley_attention_tasks::value_task(a.scores,a.v,a.out,a.shape,a.pages,a.active,
      task%rows,(task/rows)%9,task/(rows*9),probs[warp],exps[warp]);
  __syncwarp();
 }
}
struct Plan {int device=-1,blocks=0;};
inline cudaError_t prepare(Plan* p){
 if(!p)return cudaErrorInvalidValue;*p=Plan{};
 int device,supported,sms,resident;
 auto e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;
 e=cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,device);if(e!=cudaSuccess)return e;
 if(!supported)return cudaErrorNotSupported;
 e=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);if(e!=cudaSuccess)return e;
 e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,execute,256,0);if(e!=cudaSuccess)return e;
 if(resident<1)return cudaErrorNotSupported;
 p->device=device;p->blocks=sms;return cudaSuccess;
}
inline cudaError_t enqueue(cudaStream_t stream,Plan p,Args a){
 if(!a.q||!a.k||!a.v||!a.out||!a.scores||!a.shape||!a.pages||!a.active||p.blocks<1)return cudaErrorInvalidValue;
 int device,resident,sms;auto e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;
 if(device!=p.device)return cudaErrorInvalidDevice;
 e=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);if(e!=cudaSuccess)return e;
 e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,execute,256,0);if(e!=cudaSuccess)return e;
 if(p.blocks>resident*sms)return cudaErrorCooperativeLaunchTooLarge;
 void* params[]={&a};return cudaLaunchCooperativeKernel((const void*)execute,dim3(p.blocks),dim3(256),params,0,stream);
}
}
