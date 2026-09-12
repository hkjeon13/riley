from pathlib import Path
p=Path('/tmp/riley-g04-native-profile-source-260911/kernels/src/graph_resources.cu');s=p.read_text();s='#include <vector>\n'+s
s=s.replace('  bool diagnostic_trace = false;','  bool diagnostic_trace = false;\n  void* diagnostic_keys=nullptr;void* diagnostic_values=nullptr;')
s=s.replace('r->diagnostic_trace=profile==2;r->decode_capacity=capacity;', 'r->diagnostic_trace=profile==2;r->diagnostic_keys=d[15]->device_data;r->diagnostic_values=d[16]->device_data;r->decode_capacity=capacity;')
needle='  r->completion_unknown = true;\n  auto launched = cudaGraphLaunch'
s=s.replace(needle,'''  uint32_t diagnostic_position=0;if(r->diagnostic_trace)std::memcpy(&diagnostic_position,source+4,4);
  if(r->diagnostic_trace&&diagnostic_position==128){
    for(int layer=0;layer<30;++layer)for(int kind=0;kind<2;++kind){
      char path[256];std::snprintf(path,sizeof(path),"/tmp/riley-g04-serving-trace-260911/layer%d-call0-%s.bf16",layer,kind==0?"k":"v");
      auto* file=std::fopen(path,"rb");std::vector<uint8_t> raw(128*192*2),pool(r->decode_physical*16*192*2);
      if(file==nullptr)return scope.leave(reject(error,"prefill diagnostic file absent"),error,RILEY_CUDA_ERROR_STAGE_COPY,kTransfer);
      const bool ok=std::fread(raw.data(),raw.size(),1,file)==1;std::fclose(file);
      if(!ok)return scope.leave(reject(error,"prefill diagnostic read failed"),error,RILEY_CUDA_ERROR_STAGE_COPY,kTransfer);
      for(int token=0;token<128;++token){uint32_t physical=0;std::memcpy(&physical,source+16+4*(token/16),4);
        for(int head=0;head<3;++head)std::memcpy(pool.data()+((physical*3+head)*16+token%16)*128,raw.data()+(token*3+head)*128,128);}
      auto* destination=static_cast<uint8_t*>(kind==0?r->diagnostic_keys:r->diagnostic_values)+layer*pool.size();
      auto copied=cudaMemcpy(destination,pool.data(),pool.size(),cudaMemcpyHostToDevice);
      if(copied!=cudaSuccess)return scope.leave(runtime_error(copied,error,RILEY_CUDA_ERROR_STAGE_COPY,kTransfer),error,RILEY_CUDA_ERROR_STAGE_COPY,kTransfer);
    }
  }
  r->completion_unknown = true;
  auto launched = cudaGraphLaunch''')
s=s.replace('native-trace.bin','native-trace-injected.bin');p.write_text(s)
