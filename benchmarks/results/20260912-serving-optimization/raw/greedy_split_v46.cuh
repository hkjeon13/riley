struct GreedyPartial {float value;uint32_t token,invalid,reserved;};
__device__ void finish_partial(float value,uint32_t token,uint32_t bad,GreedyPartial* output,float* values,uint32_t* tokens,uint32_t* invalid){
 const int lane=threadIdx.x%32,warp=threadIdx.x/32;
 reduce_argmax_warp(&value,&token,&bad);
 if(lane==0){values[warp]=value;tokens[warp]=token;invalid[warp]=bad;}
 __syncthreads();
 if(warp==0){
  value=lane<8?values[lane]:-CUDART_INF_F;token=lane<8?tokens[lane]:UINT32_MAX;bad=lane<8?invalid[lane]:0;
  reduce_argmax_warp(&value,&token,&bad);
  if(lane==0)*output={value,token,bad,0};
 }
 __syncthreads();
}
__global__ void greedy_parts(const __nv_bfloat16* logits,GreedyPartial* partial,uint64_t rows,uint64_t vocab){
 __shared__ float values[8];__shared__ uint32_t tokens[8],invalid[8];
 const uint64_t parts=(vocab+2047)/2048;
 for(uint64_t task=blockIdx.x;task<rows*parts;task+=gridDim.x){
  const uint64_t row=task/parts,first=task%parts*2048,end=min(first+2048,vocab);
  float best=-CUDART_INF_F;uint32_t id=UINT32_MAX,bad=0;
  for(uint64_t column=first+threadIdx.x;column<end;column+=256){
   float x=__bfloat162float(logits[row*vocab+column]);
   if(!isfinite(x))bad=1;else select_argmax_candidate(x,column,&best,&id);
  }
  finish_partial(best,id,bad,partial+task,values,tokens,invalid);
 }
}
__global__ void greedy_finish(const GreedyPartial* partial,RileyCudaBf16ArgmaxResult* result,uint64_t rows,uint64_t vocab){
 __shared__ float values[8];__shared__ uint32_t tokens[8],invalid[8];__shared__ GreedyPartial combined;
 const uint64_t parts=(vocab+2047)/2048;
 for(uint64_t row=blockIdx.x;row<rows;row+=gridDim.x){
  float best=-CUDART_INF_F;uint32_t id=UINT32_MAX,bad=0;
  for(uint64_t p=threadIdx.x;p<parts;p+=256){auto x=partial[row*parts+p];bad|=x.invalid;select_argmax_candidate(x.value,x.token,&best,&id);}
  finish_partial(best,id,bad,&combined,values,tokens,invalid);
  if(threadIdx.x==0)result[row]=combined.invalid||combined.token==UINT32_MAX?RileyCudaBf16ArgmaxResult{UINT32_MAX,1}:RileyCudaBf16ArgmaxResult{combined.token,0};
  __syncthreads();
 }
}
inline void split_argmax(cudaStream_t stream,const __nv_bfloat16* logits,RileyCudaBf16ArgmaxResult* result,GreedyPartial* partial,uint64_t rows,uint64_t vocab){
 greedy_parts<<<static_cast<unsigned>(min(rows*((vocab+2047)/2048),uint64_t(65535))),256,0,stream>>>(logits,partial,rows,vocab);
 greedy_finish<<<static_cast<unsigned>(min(rows,uint64_t(65535))),256,0,stream>>>(partial,result,rows,vocab);
}
