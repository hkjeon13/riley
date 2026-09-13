#pragma once
#include <algorithm>
#include <cooperative_groups.h>
#include "../src/decode_gate_v56.cuh"
#include "../src/decode_merge_norm_v56.cuh"

// Internal diagnostic: projection/residual/norm plus FFN in one finite grid.
// Attention itself precedes this kernel; global intermediates remain allocated.
// Caller owns disjoint fixed-capacity buffers until stream completion. Metadata
// is immutable during a launch. No device queue, host polling, or Python runtime.
namespace riley_persistent_post_attention {
struct Args {
 __nv_bfloat16* x;
 const __nv_bfloat16 *gate, *up, *down, *norm;
 const __nv_bfloat16 *attention, *projection, *hidden, *mid_norm;
 float* residual_in;
 __nv_bfloat16 *activation, *residual_out, *out;
 float* parts;
 const uint32_t* shape; // three words, shape[2] is the live row count
};

__global__ void execute(Args a) {
 auto grid=cooperative_groups::this_grid();
 const unsigned rows=a.shape[2];
 if(!rows||rows>32)return; // grid-uniform: no participant skips a barrier
 const unsigned warp=threadIdx.x/32;
 const unsigned worker=blockIdx.x*8+warp,workers=gridDim.x*8;
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
 plan->device=device;plan->blocks=std::min(48,resident*sms);
 return cudaSuccess;
}
inline cudaError_t enqueue(cudaStream_t stream,const Plan& plan,Args a){
 if(!a.attention||!a.projection||!a.hidden||!a.mid_norm||!a.x||!a.gate||!a.up||!a.down||!a.norm||!a.residual_in||
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
} // namespace riley_persistent_post_attention
