from pathlib import Path
p=Path('/tmp/riley-g04-native-profile-source-260911/kernels/src/graph_resources.cu')
s=p.read_text();s='#include <cstdio>\n'+s
s=s.replace('  bool completion_visible = false;','  bool completion_visible = false;\n  bool diagnostic_trace = false;')
needle='  std::memmove(destination, static_cast<uint8_t*>(r->output->host_data)'
pos=s.index(needle)
s=s[:pos]+'''  if(r->diagnostic_trace){
    const auto* h=static_cast<const uint8_t*>(r->input->host_data);
    uint32_t position=0;std::memcpy(&position,h+4,4);
    if(position==0||position>=127){
      auto* file=std::fopen("/tmp/riley-g04-native-profile-260911/native-trace.bin","ab");
      if(file==nullptr)return reject(error,"diagnostic trace open failed",RILEY_CUDA_STATUS_INVALID_STATE);
      const bool ok=std::fwrite(&position,4,1,file)==1&&std::fwrite(h+256,3072,30,file)==30;
      std::fclose(file);if(!ok)return reject(error,"diagnostic trace write failed",RILEY_CUDA_STATUS_INVALID_STATE);
    }
  }
'''+s[pos:]
needle='      if(s==RILEY_CUDA_STATUS_SUCCESS) s=gemm(3,3);'
s=s.replace(needle,'''      if(s==RILEY_CUDA_STATUS_SUCCESS&&profile==2){
        const size_t ids[4]={4,8,7,5};const uint64_t lengths[4]={1152,384,384,1152};uint64_t offset=256+l*3072;
        for(size_t i=0;i<4&&s==RILEY_CUDA_STATUS_SUCCESS;++i){s=copy(host+offset,d[ids[i]]->device_data,lengths[i],cudaMemcpyDeviceToHost);offset+=lengths[i];}
      }
'''+needle)
s=s.replace('if(status==RILEY_CUDA_STATUS_SUCCESS){r->decode_capacity=capacity;', 'if(status==RILEY_CUDA_STATUS_SUCCESS){r->diagnostic_trace=profile==2;r->decode_capacity=capacity;')
# Trace storage must fit inside the unused input staging tail.
s=s.replace('const uint64_t transfer=result>metadata?result:metadata;','const uint64_t transfer=result>metadata?result:metadata;\n  if(profile==2&&transfer<92416)return reject(error,"trace staging too small",RILEY_CUDA_STATUS_INVALID_ARGUMENT);')
p.write_text(s)
