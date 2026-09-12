// Standalone diagnostic: the oracle links the complete frozen precise TU.
// Candidates link a separate TU with the frozen attention --use_fast_math flags.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace riley_cuda_internal {
cudaError_t enqueue_compiled_packed_decode_rope_kv(cudaStream_t,const void*,const void*,const void*,void*,void*,void*,const void*,const void*,const void*) noexcept;
}
extern "C" cudaError_t riley_rope_probe_explicit(int,cudaStream_t,const void*,const void*,const void*,void*,void*,void*,const void*,const void*,const void*);
namespace {
constexpr size_t kGuard=256,kCases=436,kQ=1152,kKV=98304,kTables=160*32*4;
void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
void check(cudaError_t status,const char* operation){if(status!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));}
std::string quote(const std::string& text){
 std::ostringstream out;out<<'"';
 for(unsigned char c:text){if(c=='"'||c=='\\')out<<'\\'<<c;else if(c<32)out<<"?";else out<<c;}
 out<<'"';return out.str();
}
int integer(const std::string& text){
 char* end=nullptr;errno=0;long value=std::strtol(text.c_str(),&end,10);
 require(errno==0 && end!=text.c_str() && *end=='\0' && value>=0 && value<=100000,"invalid integer in case index");
 return static_cast<int>(value);
}
std::vector<uint8_t> read_bytes(const std::string& path,size_t count){
 std::ifstream in(path,std::ios::binary|std::ios::ate);require(bool(in),"cannot open fixture: "+path);
 require(in.tellg()==static_cast<std::streamoff>(count),"fixture byte length differs: "+path);
 in.seekg(0);std::vector<uint8_t> bytes(count);in.read(reinterpret_cast<char*>(bytes.data()),count);
 require(bool(in),"cannot read fixture: "+path);return bytes;
}
uint16_t word(const uint8_t* p){return uint16_t(p[0])|(uint16_t(p[1])<<8);}
bool finite(const std::vector<uint8_t>& bytes,size_t start,size_t count){
 for(size_t i=start;i<start+count;i+=2)if((word(bytes.data()+i)&0x7f80)==0x7f80)return false;
 return true;
}
struct State {
 cudaStream_t stream=nullptr;size_t created=0,freed=0,live_bytes=0,cleanup_errors=0;bool stream_destroyed=false;
 void close_stream() noexcept {if(stream){auto status=cudaStreamDestroy(stream);stream=nullptr;if(status==cudaSuccess)stream_destroyed=true;else ++cleanup_errors;}}
 ~State(){close_stream();}
};
struct Buffer {
 State& state;void* allocation=nullptr;size_t bytes=0;std::vector<uint8_t> expected;
 explicit Buffer(State& s):state(s){}
 Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
 void create(const std::vector<uint8_t>& payload){
  require(allocation==nullptr,"buffer initialized twice");bytes=payload.size()+2*kGuard;
  expected.assign(bytes,0xa5);std::copy(payload.begin(),payload.end(),expected.begin()+kGuard);
  std::fill(expected.end()-kGuard,expected.end(),0x5a);
  check(cudaMalloc(&allocation,bytes),"cudaMalloc");++state.created;state.live_bytes+=bytes;
  require(reinterpret_cast<uintptr_t>(data())%256==0,"device payload is not 256-byte aligned");
  check(cudaMemcpyAsync(allocation,expected.data(),bytes,cudaMemcpyHostToDevice,state.stream),"upload guarded buffer");
 }
 void* data() const {return static_cast<uint8_t*>(allocation)+kGuard;}
 std::vector<uint8_t> download() const {
  std::vector<uint8_t> result(bytes);check(cudaMemcpyAsync(result.data(),allocation,bytes,cudaMemcpyDeviceToHost,state.stream),"download guarded buffer");
  check(cudaStreamSynchronize(state.stream),"complete guarded download");return result;
 }
 bool guards(const std::vector<uint8_t>& observed) const {
  return observed.size()==expected.size() && std::equal(observed.begin(),observed.begin()+kGuard,expected.begin())
   && std::equal(observed.end()-kGuard,observed.end(),expected.end()-kGuard);
 }
 void close() noexcept {
  if(allocation){const auto status=cudaFree(allocation);allocation=nullptr;
   if(status==cudaSuccess){++state.freed;state.live_bytes-=bytes;}else ++state.cleanup_errors;}
 }
 ~Buffer(){close();}
};
std::vector<uint8_t> poison(size_t count,uint16_t value){
 std::vector<uint8_t> out(count);for(size_t i=0;i<count;i+=2){out[i]=uint8_t(value);out[i+1]=uint8_t(value>>8);}return out;
}
struct Case {int id,phase,position;std::string qkv,cos,sin,metadata;};
std::vector<Case> read_cases(const std::string& path){
 std::ifstream in(path);require(bool(in),"cannot open cases index");std::vector<Case> cases;std::string line;
 while(std::getline(in,line)){
  std::vector<std::string> f;std::istringstream row(line);std::string field;
  while(std::getline(row,field,'\t'))f.push_back(field);
  require(f.size()==7,"case index must have seven TSV fields");
  Case c{integer(f[0]),integer(f[1]),integer(f[2]),f[3],f[4],f[5],f[6]};
  const int expected_phase=c.id<340?c.id/85:4+(c.id-340)/32;
  const int expected_position=128+(c.id<340?(c.id%85)%32:(c.id-340)%32);
  require(cases.size()<kCases && c.id==static_cast<int>(cases.size()) && c.phase==expected_phase && c.position==expected_position,"case coverage/order differs");
  cases.push_back(c);
 }
 require(in.eof() && cases.size()==kCases,"requires all 436 arithmetic cases");return cases;
}
uint32_t dword(const uint8_t* p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}
struct Comparison {
 std::array<size_t,3> mismatches{{0,0,0}};int first_region=-1;size_t first_word=0;uint16_t oracle=0,candidate=0;
 size_t total() const {return mismatches[0]+mismatches[1]+mismatches[2];}
 void add(int region,const std::vector<uint8_t>& a,const std::vector<uint8_t>& b,size_t size){
  for(size_t i=0;i<size;i+=2)if(word(a.data()+kGuard+i)!=word(b.data()+kGuard+i)){
   ++mismatches[region];if(first_region<0){first_region=region;first_word=i/2;oracle=word(a.data()+kGuard+i);candidate=word(b.data()+kGuard+i);}
  }
 }
 void emit(int order) const {
  std::cout<<"{\"order\":"<<order<<",\"equal\":"<<(total()==0?"true":"false")<<",\"mismatch_words\":"<<total()<<",\"region_mismatches\":["<<mismatches[0]<<","<<mismatches[1]<<","<<mismatches[2]<<"],\"first_mismatch\":";
  if(first_region<0)std::cout<<"null";
  else std::cout<<"{\"region\":"<<quote(first_region==0?"q_rotary":first_region==1?"keys":"values")<<",\"word\":"<<first_word<<",\"oracle_bits\":"<<oracle<<",\"candidate_bits\":"<<candidate<<"}";
  std::cout<<"}";
 }
};
bool output_contract(const std::array<std::vector<uint8_t>,3>& out,const std::vector<uint8_t>& qkv,int position,int physical){
 // Q and current K must overwrite their NaN poisons. Infinite rounded results
 // are permitted for finite extreme arithmetic; untouched cache slots are exact.
 for(size_t i=0;i<kQ;i+=2){const auto w=word(out[0].data()+kGuard+i);if((w&0x7fff)>0x7f80)return false;}
 for(size_t i=0;i<kKV/2;++i){
  const size_t block=i/(3*16*64),head=(i/(16*64))%3,token=(i/64)%16,dim=i%64;
  const bool active=block==static_cast<size_t>(physical)&&token==static_cast<size_t>(position%16);
  const auto k=word(out[1].data()+kGuard+i*2),v=word(out[2].data()+kGuard+i*2);
  if(active){if((k&0x7fff)>0x7f80 || v!=word(qkv.data()+1536+2*(head*64+dim)))return false;}
  else if(k!=0x3e81 || v!=0xbe83)return false;
 }
 return true;
}
std::array<Comparison,2> run_case(State& state,const Case& c){
 const auto qkv=read_bytes(c.qkv,1920),cos=read_bytes(c.cos,kTables),sin=read_bytes(c.sin,kTables),metadata=read_bytes(c.metadata,116);
 require(finite(qkv,0,1536),"Q/K fixture includes nonfinite BF16 input");
 for(const auto* t:{&cos,&sin})for(size_t i=0;i<t->size();i+=4)require((dword(t->data()+i)&0x7fffffff)<=0x3f800000,"table input outside finite [-1,1]");
 require(dword(metadata.data()+4)==static_cast<uint32_t>(c.position),"metadata position differs");
 const auto physical=dword(metadata.data()+16+(c.position/16)*4);require(physical<16,"metadata physical block out of range");
 Buffer input(state),cost(state),sint(state),meta(state);
 Buffer oq(state),ok(state),ov(state),aq(state),ak(state),av(state),bq(state),bk(state),bv(state);
 std::array<Buffer*,4> inputs{{&input,&cost,&sint,&meta}};
 std::array<std::array<Buffer*,3>,3> outputs{{{{&oq,&ok,&ov}},{{&aq,&ak,&av}},{{&bq,&bk,&bv}}}};
 input.create(qkv);cost.create(cos);sint.create(sin);meta.create(metadata);
 for(size_t order=0;order<3;++order){
  auto key_initial=poison(kKV,0x3e81);
  for(size_t head=0;head<3;++head)for(size_t dim=0;dim<64;++dim){
   const size_t offset=2*((physical*3+head)*16+c.position%16)*64+2*dim;
   key_initial[offset]=uint8_t(0xc1+order);key_initial[offset+1]=0x7f;
  }
  outputs[order][0]->create(poison(kQ,0x7fc1+order));outputs[order][1]->create(key_initial);outputs[order][2]->create(poison(kKV,0xbe83));
 }
 check(cudaStreamSynchronize(state.stream),"complete fixture uploads");check(cudaGetLastError(),"clean CUDA launch state");
 auto* ptr=static_cast<uint8_t*>(input.data());
 check(riley_cuda_internal::enqueue_compiled_packed_decode_rope_kv(state.stream,ptr,ptr+1152,ptr+1536,oq.data(),ok.data(),ov.data(),cost.data(),sint.data(),meta.data()),"precise packed RoPE/KV oracle");
 check(cudaGetLastError(),"oracle launch status");check(cudaStreamSynchronize(state.stream),"complete oracle");
 std::array<std::vector<uint8_t>,3> original{{oq.download(),ok.download(),ov.download()}};
 require(output_contract(original,qkv,c.position,physical),"oracle coverage/cache/V contract failed");
 std::array<Comparison,2> comparisons;
 bool guards=true,contracts=true;
 for(size_t order=0;order<2;++order){
  auto& out=outputs[order+1];
  check(riley_rope_probe_explicit(order,state.stream,ptr,ptr+1152,ptr+1536,out[0]->data(),out[1]->data(),out[2]->data(),cost.data(),sint.data(),meta.data()),"explicit arithmetic candidate");
  check(cudaGetLastError(),"candidate launch status");check(cudaStreamSynchronize(state.stream),"complete candidate");
  std::array<std::vector<uint8_t>,3> result{{out[0]->download(),out[1]->download(),out[2]->download()}};
  contracts=contracts&&output_contract(result,qkv,c.position,physical);
  for(int region=0;region<3;++region){guards=guards&&out[region]->guards(result[region]);comparisons[order].add(region,original[region],result[region],region==0?kQ:kKV);}
 }
 bool immutable=true;
 for(auto* buffer:inputs){auto observed=buffer->download();immutable=immutable&&observed==buffer->expected;guards=guards&&buffer->guards(observed);}
 bool oracle_unchanged=true;
 for(int region=0;region<3;++region){auto observed=outputs[0][region]->download();oracle_unchanged=oracle_unchanged&&observed==original[region];guards=guards&&outputs[0][region]->guards(original[region])&&outputs[0][region]->guards(observed);}
 for(auto* buffer:inputs)buffer->close();for(auto& group:outputs)for(auto* buffer:group)buffer->close();
 const bool freed=state.created==state.freed&&state.live_bytes==0&&state.cleanup_errors==0;
 const bool complete=guards&&contracts&&immutable&&oracle_unchanged&&freed;
 std::cout<<"{\"schema\":\"riley.rope-arithmetic-native.v1\",\"kind\":\"case\",\"case_id\":"<<c.id<<",\"phase\":"<<c.phase<<",\"position\":"<<c.position<<",\"mapping\":"<<c.id%3<<",\"complete\":"<<(complete?"true":"false")
  <<",\"qkv_bytes\":1920,\"q_bytes_compared\":1152,\"kv_bytes_each_compared\":98304,\"guards_intact\":"<<(guards?"true":"false")<<",\"inputs_unchanged\":"<<(immutable?"true":"false")
  <<",\"oracle_outputs_unchanged\":"<<(oracle_unchanged?"true":"false")<<",\"output_contract\":"<<(contracts?"true":"false")<<",\"all_allocations_freed\":"<<(freed?"true":"false")<<",\"candidates\":[";
 comparisons[0].emit(0);std::cout<<",";comparisons[1].emit(1);std::cout<<"]}\n"<<std::flush;
 require(complete,"guard/input/cache/cleanup contract failed");return comparisons;
}
} // namespace
int main(int argc,char** argv){
 State state;size_t completed=0;std::array<size_t,2> mismatch_cases{{0,0}},mismatch_words{{0,0}};std::string error;
 try{
  require(argc==5&&std::string(argv[1])=="--cases"&&std::string(argv[3])=="--device","usage: rope_arithmetic_probe --cases cases.tsv --device 0");
  const auto cases=read_cases(argv[2]);require(integer(argv[4])==0,"fixed profile requires device0");
  check(cudaSetDevice(0),"select device");cudaDeviceProp properties{};check(cudaGetDeviceProperties(&properties,0),"device properties");
  require(properties.major==8&&properties.minor==9,"probe requires SM89 hardware");
  int driver=0,runtime=0;check(cudaDriverGetVersion(&driver),"driver version");check(cudaRuntimeGetVersion(&runtime),"runtime version");require(runtime==13000,"probe requires CUDA runtime13.0");
  std::ostringstream uuid;uuid<<std::hex;for(unsigned char b:properties.uuid.bytes){uuid.width(2);uuid.fill('0');uuid<<unsigned(b);}
  std::cout<<"{\"schema\":\"riley.rope-arithmetic-native.v1\",\"kind\":\"device\",\"ordinal\":0,\"name\":"<<quote(properties.name)<<",\"uuid_hex\":"<<quote(uuid.str())<<",\"compute_major\":8,\"compute_minor\":9,\"driver_version\":"<<driver<<",\"runtime_version\":"<<runtime<<"}\n"<<std::flush;
  check(cudaStreamCreateWithFlags(&state.stream,cudaStreamNonBlocking),"create stream");
  for(const auto& c:cases){const auto result=run_case(state,c);for(size_t order=0;order<2;++order){mismatch_cases[order]+=result[order].total()!=0;mismatch_words[order]+=result[order].total();}++completed;}
  check(cudaStreamSynchronize(state.stream),"final stream completion");
 }catch(const std::exception& e){error=e.what();}
 state.close_stream();const bool freed=state.created==state.freed&&state.live_bytes==0&&state.cleanup_errors==0;
 const bool complete=error.empty()&&completed==kCases&&freed&&state.stream_destroyed;
 std::cout<<"{\"schema\":\"riley.rope-arithmetic-native.v1\",\"kind\":\"summary\",\"complete\":"<<(complete?"true":"false")<<",\"cases\":"<<completed
  <<",\"candidate_mismatch_cases\":["<<mismatch_cases[0]<<","<<mismatch_cases[1]<<"],\"candidate_mismatch_words\":["<<mismatch_words[0]<<","<<mismatch_words[1]<<"],\"candidate_equal\":["<<(complete&&mismatch_cases[0]==0?"true":"false")<<","<<(complete&&mismatch_cases[1]==0?"true":"false")<<"]"
  <<",\"device_allocations_created\":"<<state.created<<",\"device_allocations_freed\":"<<state.freed<<",\"live_device_allocations\":"<<(state.created-state.freed)<<",\"live_device_bytes\":"<<state.live_bytes<<",\"cleanup_errors\":"<<state.cleanup_errors
  <<",\"all_allocations_freed\":"<<(freed?"true":"false")<<",\"stream_destroyed\":"<<(state.stream_destroyed?"true":"false")<<",\"fusion_qualified\":false,\"performance_measured\":false,\"performance_claim_eligible\":false,\"error\":"<<quote(error)<<"}\n"<<std::flush;
 return complete?0:1;
}
