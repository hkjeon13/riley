#include "decode_shared_result.cuh"
#include <fstream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define OK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;OK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
std::vector<uint8_t> read(std::string path){std::ifstream f(path,std::ios::binary);std::vector<uint8_t>b((std::istreambuf_iterator<char>(f)),{});if(b.empty())exit(8);return b;}
int main(int argc,char**argv){if(argc!=2)return 1;std::string root=argv[1];auto*m=alloc<uint8_t>(17536);auto*l=alloc<__nv_bfloat16>(8*49152);auto*status=alloc<uint32_t>(1);auto*out=alloc<uint8_t>(riley_shared_result::batch_bytes);
 for(int active:{1,2,4,8}){auto p=read(root+"/request-"+std::to_string(active)+".bin"),expected=read(root+"/result-"+std::to_string(active)+".bin");if(p.size()!=17536||expected.size()!=riley_shared_result::batch_bytes)return 9;
  std::memcpy(m,p.data(),p.size());*status=0;for(int row=0;row<8;++row)std::memcpy(l+row*49152,expected.data()+row*98432+128,98304);
  OK(riley_shared_result::enqueue(0,m,l,status,out));OK(cudaDeviceSynchronize());if(std::memcmp(out,expected.data(),expected.size())){fprintf(stderr,"byte mismatch active=%d\n",active);return 3;}
  std::ofstream f(root+"/gpu-result-"+std::to_string(active)+".bin",std::ios::binary);f.write((char*)out,expected.size());f.close();
  // A nonfinite row must carry an error, not a successful completion.
  l[(active-1)*49152]=__ushort_as_bfloat16(0x7fc0);OK(riley_shared_result::enqueue(0,m,l,status,out));OK(cudaDeviceSynchronize());if(*reinterpret_cast<uint32_t*>(out+(active-1)*98432+12)==0)return 4;
  printf("active=%d gpu_rust_bytes_exact=true inactive_zero=true nonfinite_error=true\n",active);
 }
 for(void*p:{(void*)m,(void*)l,(void*)status,(void*)out})OK(cudaFree(p));
}
