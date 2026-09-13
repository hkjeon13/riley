#pragma once
#include <algorithm>
#include <cooperative_groups.h>
#include "attention_tasks.cuh"
#include "../src/decode_gate_v56.cuh"
#include "../src/decode_merge_norm_v56.cuh"

// Internal diagnostic: attention through FFN in one finite cooperative grid.
// QKV/RoPE precede this kernel; global intermediates remain allocated.
// Caller owns disjoint fixed-capacity buffers until stream completion. Metadata
// is immutable during a launch. No device queue, host polling, or Python runtime.
namespace riley_persistent_layer {
struct Args {
 __nv_bfloat16* x;
 const __nv_bfloat16 *gate, *up, *down, *norm;
 __nv_bfloat16* attention;
 const __nv_bfloat16 *projection, *hidden, *mid_norm;
 float* residual_in;
 __nv_bfloat16 *activation, *residual_out, *out;
 float* parts;
 const uint32_t* shape; // three words, shape[2] is the live row count
 const __nv_bfloat16 *q,*k,*v;
 const uint32_t *request_shape,*pages;
};

__global__ void execute(Args a) {
 auto grid=cooperative_groups::this_grid();
 const unsigned rows=a.shape[2];
 if(!rows||rows>32)return; // grid-uniform: no participant skips a barrier
 const unsigned warp=threadIdx.x/32;
 const unsigned worker=blockIdx.x*8+warp,workers=gridDim.x*8;
 __shared__ unsigned prefix[33];
 __shared__ __nv_bfloat16 probs[8][128];
 __shared__ float exps[8][128];
 if(threadIdx.x==0){
  prefix[0]=0;
  for(unsigned r=0;r<rows;++r){
   unsigned count=a.request_shape[r*416+1]+1;
   prefix[r+1]=prefix[r]+(count>=1&&count<=4096?3*((count+7)/8):0);
  }
 }
 __syncthreads();
 for(unsigned task=worker;task<prefix[rows];task+=workers){
  unsigned row=0;while(row+1<rows&&task>=prefix[row+1])++row;
  unsigned tiles=(a.request_shape[row*416+1]+8)/8,local=task-prefix[row];
  riley_attention_tasks::score_task(a.q,a.k,a.parts,a.request_shape,a.pages,(a.shape+2),
      row,local/tiles,(local%tiles)*8,tiles*8);
 }
 grid.sync();
 // Interleave rows across resident warp workers to distribute ragged contexts.
 for(unsigned task=worker;task<rows*72;task+=workers){
  riley_attention_tasks::value_task(a.parts,a.v,a.attention,a.request_shape,a.pages,(a.shape+2),
      task%rows,(task/rows)%9,task/(rows*9),probs[warp],exps[warp]);
  __syncwarp();
 }
 grid.sync(); // All attention score readers finish before parts are overwritten.
 __shared__ float sums[8];
 // Projection's five rounded partials share storage with the later down stage.
 for(unsigned task=worker;task<360;task+=workers)
  shared32_projection_compute<576,576,128,false>(a.attention,a.projection,
      a.parts,nullptr,a.shape+2,task%72,task/72);
 grid.sync();
 for(unsigned row=blockIdx.x;row<rows;row+=gridDim.x){
  riley_merge_norm_v56::merge_norm_row(a.parts,a.hidden,a.mid_norm,
      a.residual_in,a.x,1,a.shape,32,row,sums);
  __syncthreads();
 }
 // All projection-partial readers finish before the down stage reuses parts.
 grid.sync();

 for(unsigned task=worker;task<384;task+=workers){
  const unsigned row=(task%2)*16;
  if(row<rows)riley_gate_v56::compute<4,1>(a.x+row*576,a.gate,a.up,
      a.activation+row*1536,min(rows-row,16u),task/2);
 }
 grid.sync(); // every activation producer precedes every down consumer
 for(unsigned task=worker;task<360;task+=workers)
  shared32_projection_compute<576,1536,320,true>(a.activation,a.down,
      a.parts,nullptr,a.shape+2,task%72,task/72);
 grid.sync(); // all five partials are visible before each row reduction
 for(unsigned row=blockIdx.x;row<rows;row+=gridDim.x){
  riley_merge_norm_v56::merge_norm_row(a.parts,a.residual_in,a.norm,
      a.residual_out,a.out,2,a.shape,32,row,sums);
  __syncthreads(); // finish readers before reusing sums for another row
 }
}

struct Plan { int device=-1,blocks=0; };
inline cudaError_t prepare(Plan* plan){
 if(!plan)return cudaErrorInvalidValue;
 *plan=Plan{};
 int device,supported,sms,resident;
 auto e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;
 e=cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,device);if(e!=cudaSuccess)return e;
 if(!supported)return cudaErrorNotSupported;
 e=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);if(e!=cudaSuccess)return e;
 e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,execute,256,0);if(e!=cudaSuccess)return e;
 if(resident<1||sms<1)return cudaErrorNotSupported;
 plan->device=device;plan->blocks=sms;
 return cudaSuccess;
}
inline cudaError_t enqueue(cudaStream_t stream,const Plan& plan,Args a){
 if(!a.q||!a.k||!a.v||!a.request_shape||!a.pages||!a.attention||!a.projection||!a.hidden||!a.mid_norm||!a.x||!a.gate||!a.up||!a.down||!a.norm||!a.residual_in||
    !a.activation||!a.residual_out||!a.out||!a.parts||!a.shape||plan.blocks<1)return cudaErrorInvalidValue;
 int device;auto e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;
 if(device!=plan.device)return cudaErrorInvalidDevice;
 // Recheck residency so a forged/stale Plan cannot oversubscribe a grid barrier.
 int resident,sms;
 e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,execute,256,0);if(e!=cudaSuccess)return e;
 e=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);if(e!=cudaSuccess)return e;
 if(plan.blocks>resident*sms)return cudaErrorCooperativeLaunchTooLarge;
 void* args[]={&a};
 return cudaLaunchCooperativeKernel((const void*)execute,dim3(plan.blocks),dim3(256),args,0,stream);
}
} // namespace riley_persistent_layer
