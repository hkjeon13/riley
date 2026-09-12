#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "prefill_shape_model.cuh"
#include "mixed_model_v49.cuh"
#define CK(e) do{auto x=(e);if(x!=cudaSuccess){fprintf(stderr,"line%d %s\n",__LINE__,cudaGetErrorString(x));exit(2);}}while(0)
template<class T>T read(std::ifstream&f){T x{};f.read((char*)&x,sizeof(x));if(!f)exit(8);return x;}
int main(int argc,char**argv){if(argc<2)return 1;bool timing=argc>2;std::string root=argv[1];std::ifstream wf(root+"/weights.bin",std::ios::binary);std::vector<const void*>w(273);std::vector<void*>owned;
 auto alloc=[&](size_t n){void*p;CK(cudaMalloc(&p,n));owned.push_back(p);return p;};
 for(int i=0;i<273;++i){auto bytes=read<uint64_t>(wf);if(!bytes){w[i]=w[0];continue;}std::vector<uint16_t>v(bytes/2);wf.read((char*)v.data(),bytes);if(!wf)exit(8);
  if(i>=3&&(i-3)%9>=6){int part=(i-3)%9,N=part==8?576:1536,K=part==8?1536:576;std::vector<uint16_t>t(v.size());for(int n=0;n<N;n+=8)for(int k=0;k<K;k+=16)for(int lane=0;lane<32;++lane)for(int hi=0;hi<2;++hi)for(int z=0;z<2;++z)t[((n/8)*(K/16)+k/16)*128+hi*64+lane*2+z]=v[(n+lane/4)*K+k+hi*8+2*(lane%4)+z];v.swap(t);}
  void*p=alloc(bytes);CK(cudaMemcpy(p,v.data(),bytes,cudaMemcpyHostToDevice));w[i]=p;
 }
 constexpr unsigned cap=1024,physical=1024,context=1024,packed_bytes=61568,old_bytes=17536;size_t cachebytes=30ull*physical*16*192*2;std::vector<void*>scratch(12);unsigned widths[]={1152,1152,1152,1152,1152,384,384,384,3072,3072,2304,3072};for(int i=0;i<12;++i)scratch[i]=alloc(cap*widths[i]);
 void*keys=alloc(cachebytes),*values=alloc(cachebytes),*cos=alloc(context*32*4),*sin=alloc(context*32*4),*selected=alloc(32*1152+32),*old_selected=alloc(1152),*metadata=alloc(packed_bytes);auto*status=(uint32_t*)alloc(4);auto*publish=(uint32_t*)alloc(32*4);
 uint8_t *host,*old_host;CK(cudaMallocHost(&host,packed_bytes));CK(cudaMallocHost(&old_host,32*old_bytes));std::ifstream rf(root+"/rope.bin",std::ios::binary);std::vector<float>table(context*32);for(void*p:{cos,sin}){rf.read((char*)table.data(),table.size()*4);if(!rf)exit(8);CK(cudaMemcpy(p,table.data(),table.size()*4,cudaMemcpyHostToDevice));}
 std::ifstream input(root+"/requests.bin",std::ios::binary);std::vector<uint32_t>tokens;for(unsigned c=read<uint32_t>(input);c;--c){unsigned n=read<uint32_t>(input);std::vector<uint32_t>x(n);input.read((char*)x.data(),n*4);if(x.size()>tokens.size())tokens=x;}if(tokens.empty())exit(8);
 cudaStream_t stream;CK(cudaStreamCreate(&stream));
 auto put=[](uint8_t*p,unsigned at,uint32_t v){memcpy(p+at,&v,4);};
 auto packet=[&](uint8_t*p,bool packed,unsigned owner,unsigned descriptor,unsigned start,unsigned count,unsigned prompt,unsigned offset){unsigned b=128+descriptor*1664;put(p,b,tokens[(owner*7+start)%tokens.size()]);put(p,b+4,start+count-1);put(p,b+8,count);put(p,b+16,start);put(p,b+20,start+count);put(p,b+32,prompt);put(p,b+36,context);put(p,b+44,start+count==prompt?count-1:UINT32_MAX);if(packed)put(p,b+64,offset);for(unsigned i=0;i<32;++i)put(p,b+128+i*4,owner*32+(i*5+17)%32);for(unsigned i=0;i<count;++i)put(p,(packed?53376:13440)+(offset+i)*4,tokens[(owner*7+start+i)%tokens.size()]);};
 auto old_run=[&](uint8_t*p,unsigned capacity){CK(cudaMemcpyAsync(metadata,p,old_bytes,cudaMemcpyHostToDevice,stream));CK(enqueue_v3_prefill_model<8>(stream,scratch.data(),w.data(),metadata,keys,values,cos,sin,old_selected,status,publish,capacity,physical,true));};
 unsigned cases=0;uint64_t compared=0;int lists[4]={1,7,31,33};
 for(int pattern=0;pattern<6;++pattern)for(unsigned owners:{1U,4U,8U,16U,32U}){unsigned counts[32],starts[32],prompts[32],total=0,max_count=0;for(unsigned i=0;i<owners;++i){unsigned n=1;if(pattern==0)n=i==0?128:1;if(pattern==1)n=i<4?128:1;if(pattern==3)n=lists[i%4];if(pattern==4)n=i==0?398:1;if(pattern==5)n=31;counts[i]=n;starts[i]=n==1?128+13*(i%4):std::array<unsigned,4>{0,13,73,129}[i%4];prompts[i]=starts[i]+counts[i]+(n>1&&i%2?7:0);total+=counts[i];max_count=std::max(max_count,counts[i]);}
  unsigned packed_cap=total<=128?128:(total<=512?512:1024),old_cap=max_count<=128?128:512;
  std::vector<uint8_t>expected(32*1152,0),ka(cachebytes),va(cachebytes),actual(32*1152+32),kb(cachebytes),vb(cachebytes);std::vector<uint32_t>flags(32);
  for(int candidate=0;candidate<2;++candidate){CK(cudaMemsetAsync(keys,0,cachebytes,stream));CK(cudaMemsetAsync(values,0,cachebytes,stream));CK(cudaMemsetAsync(selected,0xa5,actual.size(),stream));CK(cudaStreamSynchronize(stream));
   // Populate real per-owner prefixes using the established model before either comparison.
   for(unsigned i=0;i<owners;++i)if(starts[i]){memset(old_host,0,old_bytes);packet(old_host,false,i,0,0,starts[i],prompts[i],0);old_run(old_host,512);CK(cudaStreamSynchronize(stream));}
   if(candidate){memset(host,0,packed_bytes);put(host,20,owners);put(host,36,total);unsigned offset=0,tiles=0;for(unsigned i=0;i<owners;++i){packet(host,true,i,i,starts[i],counts[i],prompts[i],offset);put(host,128+i*1664+68,tiles);offset+=counts[i];unsigned nt=counts[i]<32?counts[i]:(counts[i]+7)/8;for(unsigned j=0;j<nt;++j)put(host,57472+(tiles+j)*4,(i<<16)|j);tiles+=nt;}put(host,96,tiles);CK(cudaMemcpyAsync(metadata,host,packed_bytes,cudaMemcpyHostToDevice,stream));CK(enqueue_packed_prefill_model<32>(stream,scratch.data(),w.data(),metadata,keys,values,cos,sin,selected,status,publish,packed_cap,physical,true));CK(cudaStreamSynchronize(stream));CK(cudaMemcpy(actual.data(),selected,actual.size(),cudaMemcpyDeviceToHost));CK(cudaMemcpy(flags.data(),publish,128,cudaMemcpyDeviceToHost));CK(cudaMemcpy(kb.data(),keys,cachebytes,cudaMemcpyDeviceToHost));CK(cudaMemcpy(vb.data(),values,cachebytes,cudaMemcpyDeviceToHost));}
   else{for(unsigned i=0;i<owners;++i){auto*p=old_host+i*old_bytes;memset(p,0,old_bytes);packet(p,false,i,0,starts[i],counts[i],prompts[i],0);old_run(p,old_cap);CK(cudaStreamSynchronize(stream));CK(cudaMemcpy(expected.data()+i*1152,old_selected,1152,cudaMemcpyDeviceToHost));}CK(cudaMemcpy(ka.data(),keys,cachebytes,cudaMemcpyDeviceToHost));CK(cudaMemcpy(va.data(),values,cachebytes,cudaMemcpyDeviceToHost));}
  }
  uint32_t error;CK(cudaMemcpy(&error,status,4,cudaMemcpyDeviceToHost));if(error||memcmp(expected.data(),actual.data(),expected.size())||ka!=kb||va!=vb){fprintf(stderr,"packed mismatch pattern%d owners%u total%u hidden%d keys%d values%d status%u\n",pattern,owners,total,memcmp(expected.data(),actual.data(),expected.size())!=0,ka!=kb,va!=vb,error);exit(3);}for(unsigned i=expected.size();i<actual.size();++i)if(actual[i]!=0xa5)exit(4);for(unsigned i=0;i<32;++i)if(flags[i]!=(i<owners&&starts[i]+counts[i]==prompts[i]))exit(5);
  compared+=expected.size()+2*cachebytes;++cases;
  if(!timing){printf("mixed_geometry pattern%d owners%u total%u prefix=true hidden_exact=true entire_kv_exact=true partial_and_inactive_zero=true guard=true\n",pattern,owners,total);fflush(stdout);}

 }
 CK(cudaStreamDestroy(stream));CK(cudaFreeHost(host));CK(cudaFreeHost(old_host));for(void*p:owned)CK(cudaFree(p));fprintf(stderr,"mixed geometry full model cases=%u compared_bytes=%llu\n",cases,(unsigned long long)compared);
}
