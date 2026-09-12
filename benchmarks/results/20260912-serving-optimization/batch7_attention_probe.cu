// Correctness-only synthetic attention probe. The unchanged production TU is
// compiled separately with its actual CMake flags, including --use_fast_math.
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
#ifndef RILEY_ATTENTION_CANDIDATE
#error "An explicit distinct candidate wrapper is required"
#endif
namespace riley_cuda_internal {
cudaError_t enqueue_compiled_attention_rows(cudaStream_t,const void*,const void*,const void*,void*,const void*,uint32_t) noexcept;
cudaError_t enqueue_compiled_packed_decode_attention(cudaStream_t,const void*,const void*,const void*,void*,const void*) noexcept;
cudaError_t RILEY_ATTENTION_CANDIDATE(cudaStream_t,const void*,const void*,const void*,void*,const void*) noexcept;
}
namespace {
constexpr size_t kGuard=256,kCases=8640,kOutput=1152,kCache=98304,kMetadata=116;
const char* patterns[]={"bounded_fingerprint","signed_zero_impulses","tail_cancellation"};
const char* mappings[]={"identity","reverse","affine_7_3"};
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

struct Case {int id,layer,position,pattern,mapping;};
std::vector<Case> read_cases(const std::string& path){
 std::ifstream in(path);require(bool(in),"cannot open cases index");std::vector<Case> cases;std::string line;
 while(std::getline(in,line)){
  std::istringstream row(line);std::vector<std::string> fields;std::string field;
  while(std::getline(row,field,'\t'))fields.push_back(field);
  require(fields.size()==5,"case index requires five TSV fields");
  Case c{integer(fields[0]),integer(fields[1]),integer(fields[2]),integer(fields[3]),integer(fields[4])};
  require(cases.size()<kCases && c.id==static_cast<int>(cases.size()) && c.layer==c.id/288 && c.position==128+(c.id/9)%32 && c.pattern==(c.id/3)%3 && c.mapping==c.id%3,"case coverage/order differs");
  cases.push_back(c);
 }
 require(in.eof()&&cases.size()==kCases,"requires 30 synthetic layers x 32 positions x 3 patterns x 3 maps");return cases;
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

void put16(std::vector<uint8_t>& bytes,size_t offset,uint16_t value){bytes[offset]=uint8_t(value);bytes[offset+1]=uint8_t(value>>8);}
void put32(std::vector<uint8_t>& bytes,size_t offset,uint32_t value){for(int b=0;b<4;++b)bytes[offset+b]=uint8_t(value>>(8*b));}
int physical(int logical,int mapping){return mapping==0?logical:mapping==1?15-logical:(7*logical+3)%16;}
// Integer BF16 encodings only. No float conversion or real checkpoint inputs.
uint16_t sample(int kind,int layer,int position,int token,int head,int dim,int pattern){
 const uint32_t seed=uint32_t(layer*43+position*19+token*31+head*17+dim*13+kind*7);
 const uint16_t sign=uint16_t((seed&1)*0x8000);
 if(pattern==0)return uint16_t(0x3d80+(seed%128))|sign; // 1/16 .. 1/8
 if(pattern==1){
  const int pivot=(layer*7+token*11+head*13+(kind==0?position:0))%64;
  return dim==pivot?uint16_t(0x3e80+(seed%128))|sign:uint16_t(((dim+token+head+layer)&1)*0x8000);
 }
 if(kind==0)return uint16_t(0x3f00+(seed%128)); // positive query, [1/2,1)
 if(kind==1){
  const uint16_t polarity=uint16_t(((token<128)^(layer&1))*0x8000);
  return uint16_t(0x3e00+((layer*23+token*3+head*7+dim*5)%128))|polarity;
 }
 // Adjacent values cancel; tail/full scores have different maxima and alpha.
 return uint16_t(0x3e80+((layer*29+token*11+head*19+(dim/2)*13)%128))|uint16_t((dim&1)*0x8000);
}
struct Fixtures {std::vector<uint8_t> q,k,v,metadata;};
Fixtures fixture(const Case& c){
 Fixtures f{std::vector<uint8_t>(kOutput),std::vector<uint8_t>(kCache),std::vector<uint8_t>(kCache),std::vector<uint8_t>(kMetadata)};
 for(int head=0;head<9;++head)for(int dim=0;dim<64;++dim)put16(f.q,(head*64+dim)*2,sample(0,c.layer,c.position,0,head,dim,c.pattern));
 // All physical slots, including unmapped and future tokens, start finite.
 for(size_t i=0;i<kCache;i+=2){put16(f.k,i,uint16_t(0xbe80+(i/2)%128));put16(f.v,i,uint16_t(0x3e80+(i/2)%128));}
 for(int token=0;token<160;++token)for(int head=0;head<3;++head)for(int dim=0;dim<64;++dim){
  size_t offset=((physical(token/16,c.mapping)*3+head)*16+token%16)*64*2+dim*2;
  // KV depends on layer/token/head/dim; it is stable across query positions.
  put16(f.k,offset,sample(1,c.layer,0,token,head,dim,c.pattern));
  put16(f.v,offset,sample(2,c.layer,0,token,head,dim,c.pattern));
 }
 const int live=c.position/16+1;put32(f.metadata,4,c.position);put32(f.metadata,12,live);
 for(int i=0;i<live;++i){put32(f.metadata,16+i*4,physical(i,c.mapping));put16(f.metadata,80+i*2,i==live-1?c.position%16+1:16);}
 return f;
}
bool run_case(State& state,const Case& c,std::array<std::vector<uint8_t>,3>& map_reference){
 const auto f=fixture(c);
 require(finite(f.q,0,f.q.size())&&finite(f.k,0,f.k.size())&&finite(f.v,0,f.v.size()),"nonfinite synthetic fixture");
 Buffer q(state),k(state),v(state),metadata(state),original(state),accepted(state),candidate(state);
 q.create(f.q);k.create(f.k);v.create(f.v);metadata.create(f.metadata);
 original.create(poison(kOutput,0x7fc1));accepted.create(poison(kOutput,0x7fc3));candidate.create(poison(kOutput,0x7fc5));
 check(cudaStreamSynchronize(state.stream),"complete fixture upload");check(cudaGetLastError(),"clean CUDA launch state");
 check(riley_cuda_internal::enqueue_compiled_attention_rows(state.stream,q.data(),k.data(),v.data(),original.data(),metadata.data(),1),"enqueue original rows1 attention");
 check(cudaGetLastError(),"original launch status");check(cudaStreamSynchronize(state.stream),"complete original attention");const auto before=original.download();
 check(riley_cuda_internal::enqueue_compiled_packed_decode_attention(state.stream,q.data(),k.data(),v.data(),accepted.data(),metadata.data()),"enqueue accepted packed attention");
 check(cudaGetLastError(),"accepted launch status");check(cudaStreamSynchronize(state.stream),"complete accepted attention");const auto packed=accepted.download();
 check(riley_cuda_internal::RILEY_ATTENTION_CANDIDATE(state.stream,q.data(),k.data(),v.data(),candidate.data(),metadata.data()),"enqueue candidate attention");
 check(cudaGetLastError(),"candidate launch status");check(cudaStreamSynchronize(state.stream),"complete candidate attention");const auto result=candidate.download();
 const auto qa=q.download(),ka=k.download(),va=v.download(),ma=metadata.download(),oa=original.download(),pa=accepted.download();
 const bool unchanged=qa==q.expected&&ka==k.expected&&va==v.expected&&ma==metadata.expected;
 const bool oracle_unchanged=oa==before&&pa==packed;
 const bool guards=q.guards(qa)&&k.guards(ka)&&v.guards(va)&&metadata.guards(ma)&&original.guards(before)&&original.guards(oa)&&accepted.guards(packed)&&accepted.guards(pa)&&candidate.guards(result);
 const bool finite_outputs=finite(before,kGuard,kOutput)&&finite(packed,kGuard,kOutput)&&finite(result,kGuard,kOutput);
 size_t accepted_mismatch=0,candidate_mismatch=0,first=576;
 for(size_t i=0;i<576;++i){
  const auto old=word(before.data()+kGuard+2*i),a=word(packed.data()+kGuard+2*i),b=word(result.data()+kGuard+2*i);
  accepted_mismatch+=old!=a;candidate_mismatch+=old!=b;if((old!=a||old!=b)&&first==576)first=i;
 }
 std::array<std::vector<uint8_t>,3> outputs{{std::vector<uint8_t>(before.begin()+kGuard,before.end()-kGuard),std::vector<uint8_t>(packed.begin()+kGuard,packed.end()-kGuard),std::vector<uint8_t>(result.begin()+kGuard,result.end()-kGuard)}};
 if(c.mapping==0)map_reference=outputs;
 const bool mapping_invariant=outputs==map_reference;
 q.close();k.close();v.close();metadata.close();original.close();accepted.close();candidate.close();
 const bool freed=state.created==state.freed&&state.live_bytes==0&&state.cleanup_errors==0;
 const bool exact=accepted_mismatch==0&&candidate_mismatch==0;
 const bool passed=exact&&finite_outputs&&guards&&unchanged&&oracle_unchanged&&mapping_invariant&&freed;
 std::cout<<"{\"schema\":\"riley.batch7-attention-native.v1\",\"kind\":\"case\",\"case_id\":"<<c.id<<",\"layer\":"<<c.layer<<",\"position\":"<<c.position
  <<",\"pattern\":"<<quote(patterns[c.pattern])<<",\"mapping\":"<<quote(mappings[c.mapping])<<",\"output_bytes_compared_per_pair\":1152,\"comparison_pairs\":2,\"passed\":"<<(passed?"true":"false")
  <<",\"exact_outputs\":"<<(exact?"true":"false")<<",\"finite_outputs\":"<<(finite_outputs?"true":"false")<<",\"guards_intact\":"<<(guards?"true":"false")
  <<",\"inputs_unchanged\":"<<(unchanged?"true":"false")<<",\"oracle_outputs_unchanged\":"<<(oracle_unchanged?"true":"false")<<",\"mapping_invariant\":"<<(mapping_invariant?"true":"false")
  <<",\"all_allocations_freed\":"<<(freed?"true":"false")<<",\"accepted_mismatch_words\":"<<accepted_mismatch<<",\"candidate_mismatch_words\":"<<candidate_mismatch;
 if(first<576)std::cout<<",\"first_mismatch\":{\"head\":"<<first/64<<",\"dim\":"<<first%64<<",\"original_bits\":"<<word(before.data()+kGuard+2*first)<<",\"accepted_bits\":"<<word(packed.data()+kGuard+2*first)<<",\"candidate_bits\":"<<word(result.data()+kGuard+2*first)<<"}";
 std::cout<<"}\n"<<std::flush;return passed;
}
} // namespace
int main(int argc,char** argv){
 State state;size_t completed=0,failed=0;std::string error;
 try{
  require(argc==5 && std::string(argv[1])=="--cases" && std::string(argv[3])=="--device","usage: batch7_attention_probe --cases cases.tsv --device 0");
  auto cases=read_cases(argv[2]);int device=integer(argv[4]);require(device==0,"fixed workload requires CUDA device0");
  check(cudaSetDevice(device),"select device");cudaDeviceProp properties{};check(cudaGetDeviceProperties(&properties,device),"device properties");
  require(properties.major==8 && properties.minor==9,"probe requires SM89 hardware");
  int driver=0,runtime=0;check(cudaDriverGetVersion(&driver),"driver version");check(cudaRuntimeGetVersion(&runtime),"runtime version");
  require(runtime==13000,"fixed profile requires CUDA runtime13.0");
  std::ostringstream uuid;uuid<<std::hex;for(unsigned char b:properties.uuid.bytes){uuid.width(2);uuid.fill('0');uuid<<unsigned(b);}
  std::cout<<"{\"schema\":\"riley.batch7-attention-native.v1\",\"kind\":\"device\",\"ordinal\":0,\"name\":"<<quote(properties.name)<<",\"uuid_hex\":"<<quote(uuid.str())<<",\"compute_major\":8,\"compute_minor\":9,\"driver_version\":"<<driver<<",\"runtime_version\":"<<runtime<<"}\n"<<std::flush;
  check(cudaStreamCreateWithFlags(&state.stream,cudaStreamNonBlocking),"create stream");
  std::array<std::vector<uint8_t>,3> map_reference;
  for(const auto& c:cases){if(!run_case(state,c,map_reference))++failed;++completed;}
  check(cudaStreamSynchronize(state.stream),"final stream completion");
 }catch(const std::exception& exception){error=exception.what();}
 state.close_stream();const bool freed=state.created==state.freed && state.live_bytes==0 && state.cleanup_errors==0;
 const bool passed=error.empty() && completed==kCases && failed==0 && freed && state.stream_destroyed;
 std::cout<<"{\"schema\":\"riley.batch7-attention-native.v1\",\"kind\":\"summary\",\"passed\":"<<(passed?"true":"false")
  <<",\"cases\":"<<completed<<",\"failed_cases\":"<<failed<<",\"layers\":30,\"positions\":32,\"patterns\":3,\"mappings\":3"
  <<",\"device_allocations_created\":"<<state.created<<",\"device_allocations_freed\":"<<state.freed<<",\"live_device_allocations\":"<<(state.created-state.freed)<<",\"live_device_bytes\":"<<state.live_bytes
  <<",\"cleanup_errors\":"<<state.cleanup_errors<<",\"all_allocations_freed\":"<<(freed?"true":"false")<<",\"stream_destroyed\":"<<(state.stream_destroyed?"true":"false")
  <<",\"performance_measured\":false,\"performance_claim_eligible\":false,\"error\":"<<quote(error)<<"}\n"<<std::flush;
 return passed?0:1;
}
