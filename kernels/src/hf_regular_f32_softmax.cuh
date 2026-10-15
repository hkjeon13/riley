// Adapted from PyTorch cf30153c4c131c8164ee7798e5022d810682e2cb.
// See LICENSE-PyTorchSoftmax. CUDA float32 register softmax, bounded T<=9216.
#pragma once
#include "decode_hf_regular_softmax.cuh"
template<int Registers>
__global__ void hf_regular_f32_softmax(float*out,const float*in,uint64_t columns){
 __shared__ float reduction[32];float values[Registers];uint64_t base=uint64_t(blockIdx.x)*columns;float maximum=-FLT_MAX;
 #pragma unroll
 for(int j=0;j<Registers;j++){uint64_t c=threadIdx.x+uint64_t(j)*blockDim.x;if(c<columns){values[j]=in[base+c];maximum=::max(maximum,values[j]);}}
 maximum=decode_hf_block_reduce<true>(maximum,reduction);float sum=0;
 #pragma unroll
 for(int j=0;j<Registers;j++){uint64_t c=threadIdx.x+uint64_t(j)*blockDim.x;if(c<columns)sum=sum+std::exp(values[j]-maximum);}
 sum=decode_hf_block_reduce<false>(sum,reduction);
 #pragma unroll
 for(int j=0;j<Registers;j++){uint64_t c=threadIdx.x+uint64_t(j)*blockDim.x;if(c<columns)out[base+c]=std::exp(values[j]-maximum)/sum;}
}
