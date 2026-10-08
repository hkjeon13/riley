#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "source-fragment.cuh"

static void checked(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << '\n';
    std::exit(2);
  }
}
static void read(const char* path, void* bytes, size_t count) {
  FILE* file=std::fopen(path,"rb");
  if (!file || std::fread(bytes,1,count,file)!=count || std::fgetc(file)!=EOF) std::exit(3);
  std::fclose(file);
}
static float as_float(uint16_t value) {
  uint32_t word=uint32_t(value)<<16;float result;
  std::memcpy(&result,&word,sizeof(result));return result;
}
int main(int argc,char** argv) {
  if(argc!=4)return 1;
  constexpr size_t elements=16*2049,bytes=elements*2;
  std::array<uint16_t,elements> input{},expected{},observed{};
  read(argv[1],input.data(),bytes);read(argv[2],expected.data(),bytes);
  __nv_bfloat16* device=nullptr;cudaStream_t stream;
  checked(cudaStreamCreate(&stream));checked(cudaMalloc(&device,bytes));
  checked(cudaMemcpy(device,input.data(),bytes,cudaMemcpyHostToDevice));
  hf_regular_softmax_kernel<3><<<16,1024,0,stream>>>(device,2049);
  checked(cudaGetLastError());checked(cudaStreamSynchronize(stream));
  checked(cudaMemcpy(observed.data(),device,bytes,cudaMemcpyDeviceToHost));
  checked(cudaFree(device));checked(cudaStreamDestroy(stream));
  FILE* file=std::fopen(argv[3],"wb");
  if(!file || std::fwrite(observed.data(),1,bytes,file)!=bytes)return 4;
  std::fclose(file);
  size_t unequal=0;std::array<size_t,16> heads{};float maximum=0;
  for(size_t i=0;i<elements;++i)if(observed[i]!=expected[i]){
    ++unequal;++heads[i/2049];maximum=std::fmax(maximum,std::fabs(as_float(observed[i])-as_float(expected[i])));
  }
  std::cout << "{\"elements\":" << elements << ",\"unequal\":" << unequal
            << ",\"max_abs\":" << maximum << ",\"unequal_by_head\":[";
  for(size_t i=0;i<16;++i){if(i)std::cout<<',';std::cout<<heads[i];}
  std::cout << "]}\n";
  return 0; // Diagnostic process success does not imply BF16 equality.
}
