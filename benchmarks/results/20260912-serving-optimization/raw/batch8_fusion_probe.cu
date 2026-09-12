// Correctness-only synthetic RoPE/KV/attention fusion probe. The unchanged production TU is
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
cudaError_t enqueue_compiled_packed_decode_rope_kv(cudaStream_t,const void*,const void*,const void*,void*,void*,void*,const void*,const void*,const void*) noexcept;
cudaError_t enqueue_compiled_packed_decode_attention_two_warp(cudaStream_t,const void*,const void*,const void*,void*,const void*) noexcept;
cudaError_t RILEY_ATTENTION_CANDIDATE(cudaStream_t,const void*,const void*,const void*,void*,void*,void*,void*,const void*,const void*,const void*) noexcept;
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
struct Fixtures {std::vector<uint8_t> qkv,k,v,cos,sin,metadata;};
Fixtures fixture(const Case& c){
 Fixtures f{std::vector<uint8_t>(1920),std::vector<uint8_t>(kCache),std::vector<uint8_t>(kCache),
            std::vector<uint8_t>(20480),std::vector<uint8_t>(20480),std::vector<uint8_t>(kMetadata)};
 for(int head=0;head<9;++head)for(int dim=0;dim<64;++dim)
  put16(f.qkv,(head*64+dim)*2,sample(0,c.layer,c.position,0,head,dim,c.pattern));
 for(int head=0;head<3;++head)for(int dim=0;dim<64;++dim){
  put16(f.qkv,1152+(head*64+dim)*2,sample(1,c.layer,0,c.position,head,dim,c.pattern));
  put16(f.qkv,1536+(head*64+dim)*2,sample(2,c.layer,0,c.position,head,dim,c.pattern));
 }
 for(size_t i=0;i<kCache;i+=2){put16(f.k,i,uint16_t(0xbe80+(i/2)%128));put16(f.v,i,uint16_t(0x3e80+(i/2)%128));}
 for(int token=0;token<160;++token)for(int head=0;head<3;++head)for(int dim=0;dim<64;++dim){
  size_t offset=((physical(token/16,c.mapping)*3+head)*16+token%16)*128+dim*2;
  put16(f.k,offset,token==c.position?0x7fc1:sample(1,c.layer,0,token,head,dim,c.pattern));
  put16(f.v,offset,token==c.position?0x7fc3:sample(2,c.layer,0,token,head,dim,c.pattern));
 }
 // Bounded synthetic F32 table bits, including signed zero and BF16 rounding
 // boundaries. These are not checkpoint trigonometric tables.
 constexpr uint32_t table[]={0,0x80000000,0x3f800000,0xbf800000,0x3f000000,0xbf000000,
  0x3f007fff,0x3f008000,0x3f008001,0xbf007fff,0xbf008000,0xbf008001,
  0x3f7f7fff,0x3f7f8000,0x3f7f8001,0xbf7f7fff,0xbf7f8000,0xbf7f8001,
  0x00800000,0x80800000,0x00000001,0x80000001,0x3e807fff,0xbe808001};
 for(int pos=0;pos<160;++pos)for(int dim=0;dim<32;++dim){
  const uint32_t seed=uint32_t(c.layer*7+c.pattern*11+pos*13+dim*17);
  put32(f.cos,(pos*32+dim)*4,table[seed%24]);put32(f.sin,(pos*32+dim)*4,table[(seed+7)%24]);
 }
 const int live=c.position/16+1;put32(f.metadata,4,c.position);put32(f.metadata,12,live);
 for(int i=0;i<live;++i){put32(f.metadata,16+i*4,physical(i,c.mapping));put16(f.metadata,80+i*2,i==live-1?c.position%16+1:16);}
 return f;
}
bool run_case(State& state,const Case& c,std::array<std::vector<uint8_t>,2>& map_reference){
 const auto f=fixture(c);
 require(finite(f.qkv,0,f.qkv.size()),"nonfinite synthetic QKV");
 Buffer qkv(state),cos(state),sin(state),meta(state),oq(state),ok(state),ov(state),oa(state),
        cq(state),ck(state),cv(state),ca(state);
 qkv.create(f.qkv);cos.create(f.cos);sin.create(f.sin);meta.create(f.metadata);
 oq.create(poison(kOutput,0x7fc5));ok.create(f.k);ov.create(f.v);oa.create(poison(kOutput,0x7fc7));
 cq.create(poison(kOutput,0x7fc9));ck.create(f.k);cv.create(f.v);ca.create(poison(kOutput,0x7fcb));
 check(cudaStreamSynchronize(state.stream),"complete fixture upload");check(cudaGetLastError(),"clean launch state");
 auto* packed=static_cast<uint8_t*>(qkv.data());
 check(riley_cuda_internal::enqueue_compiled_packed_decode_rope_kv(state.stream,packed,packed+1152,packed+1536,
       oq.data(),ok.data(),ov.data(),cos.data(),sin.data(),meta.data()),"enqueue precise RoPE/KV oracle");
 check(riley_cuda_internal::enqueue_compiled_packed_decode_attention_two_warp(state.stream,oq.data(),ok.data(),ov.data(),
       oa.data(),meta.data()),"enqueue accepted two-warp oracle");
 check(cudaGetLastError(),"oracle launch status");check(cudaStreamSynchronize(state.stream),"complete oracle chain");
 const std::array<std::vector<uint8_t>,4> before{{oa.download(),oq.download(),ok.download(),ov.download()}};
 check(riley_cuda_internal::RILEY_ATTENTION_CANDIDATE(state.stream,packed,packed+1152,packed+1536,
       cq.data(),ck.data(),cv.data(),ca.data(),cos.data(),sin.data(),meta.data()),"enqueue fused candidate");
 check(cudaGetLastError(),"candidate launch status");check(cudaStreamSynchronize(state.stream),"complete fused candidate");
 const std::array<std::vector<uint8_t>,4> observed{{ca.download(),cq.download(),ck.download(),cv.download()}};
 const std::array<std::vector<uint8_t>,4> after{{oa.download(),oq.download(),ok.download(),ov.download()}};
 const auto qa=qkv.download(),co=cos.download(),si=sin.download(),ma=meta.download();
 const bool unchanged=qa==qkv.expected&&co==cos.expected&&si==sin.expected&&ma==meta.expected;
 const bool oracle_unchanged=before==after;
 const std::array<Buffer*,4> ob{{&oa,&oq,&ok,&ov}},cb{{&ca,&cq,&ck,&cv}};
 const std::array<size_t,4> extents{{kOutput,kOutput,kCache,kCache}};
 const char* regions[]={"attention","q_rotary","keys","values"};
 std::array<size_t,4> mismatches{};size_t first_region=4,first_word=0;
 bool guards=qkv.guards(qa)&&cos.guards(co)&&sin.guards(si)&&meta.guards(ma),finite_outputs=true;
 for(size_t region=0;region<4;++region){
  guards=guards&&ob[region]->guards(before[region])&&ob[region]->guards(after[region])&&cb[region]->guards(observed[region]);
  finite_outputs=finite_outputs&&finite(before[region],kGuard,extents[region])&&finite(observed[region],kGuard,extents[region]);
  for(size_t i=0;i<extents[region]/2;++i)if(word(before[region].data()+kGuard+2*i)!=word(observed[region].data()+kGuard+2*i)){
   ++mismatches[region];if(first_region==4){first_region=region;first_word=i;}
  }
 }
 // Every non-current KV byte must remain the original fixture, independently
 // of oracle equality. Decode rows never write other physical slots.
 bool inactive_unchanged=true,current_values_exact=true;
 for(size_t offset=0;offset<kCache;offset+=2){
  const size_t physical_block=offset/(3*16*128),within_block=offset%(3*16*128);
  const size_t head=within_block/(16*128),within_head=within_block%(16*128);
  const bool current=physical_block==size_t(physical(c.position/16,c.mapping))&&within_head/128==size_t(c.position%16);
  if(!current){
   inactive_unchanged=inactive_unchanged&&word(before[2].data()+kGuard+offset)==word(f.k.data()+offset)
    &&word(observed[2].data()+kGuard+offset)==word(f.k.data()+offset)
    &&word(before[3].data()+kGuard+offset)==word(f.v.data()+offset)
    &&word(observed[3].data()+kGuard+offset)==word(f.v.data()+offset);
  }else{
   const auto raw=word(f.qkv.data()+1536+head*128+within_head%128);
   current_values_exact=current_values_exact&&word(before[3].data()+kGuard+offset)==raw&&word(observed[3].data()+kGuard+offset)==raw;
  }
 }
 std::array<std::vector<uint8_t>,2> outputs{{
  std::vector<uint8_t>(observed[0].begin()+kGuard,observed[0].end()-kGuard),
  std::vector<uint8_t>(observed[1].begin()+kGuard,observed[1].end()-kGuard)}};
 if(c.mapping==0)map_reference=outputs;
 const bool mapping_invariant=outputs==map_reference;
 qkv.close();cos.close();sin.close();meta.close();oq.close();ok.close();ov.close();oa.close();cq.close();ck.close();cv.close();ca.close();
 const bool freed=state.created==state.freed&&state.live_bytes==0&&state.cleanup_errors==0;
 const bool exact=std::all_of(mismatches.begin(),mismatches.end(),[](size_t count){return count==0;});
 const bool passed=exact&&guards&&unchanged&&oracle_unchanged&&finite_outputs&&inactive_unchanged&&current_values_exact&&mapping_invariant&&freed;
 std::cout<<"{\"schema\":\"riley.batch8-fusion-native.v1\",\"kind\":\"case\",\"case_id\":"<<c.id<<",\"layer\":"<<c.layer<<",\"position\":"<<c.position
  <<",\"pattern\":"<<quote(patterns[c.pattern])<<",\"mapping\":"<<quote(mappings[c.mapping])<<",\"passed\":"<<(passed?"true":"false")
  <<",\"exact_outputs\":"<<(exact?"true":"false")<<",\"finite_outputs\":"<<(finite_outputs?"true":"false")<<",\"guards_intact\":"<<(guards?"true":"false")
  <<",\"inputs_unchanged\":"<<(unchanged?"true":"false")<<",\"oracle_outputs_unchanged\":"<<(oracle_unchanged?"true":"false")
  <<",\"inactive_kv_unchanged\":"<<(inactive_unchanged?"true":"false")<<",\"current_values_raw_exact\":"<<(current_values_exact?"true":"false")
  <<",\"mapping_invariant\":"<<(mapping_invariant?"true":"false")<<",\"all_allocations_freed\":"<<(freed?"true":"false")
  <<",\"compared_bytes\":[1152,1152,98304,98304],\"mismatch_words\":["<<mismatches[0]<<","<<mismatches[1]<<","<<mismatches[2]<<","<<mismatches[3]<<"]";
 if(first_region<4)std::cout<<",\"first_mismatch\":{\"region\":"<<quote(regions[first_region])<<",\"word\":"<<first_word
  <<",\"oracle_bits\":"<<word(before[first_region].data()+kGuard+first_word*2)<<",\"candidate_bits\":"<<word(observed[first_region].data()+kGuard+first_word*2)<<"}";
 std::cout<<"}\n"<<std::flush;return passed;
}
} // namespace
int main(int argc,char** argv){
 State state;size_t completed=0,failed=0;std::string error;
 try{
  require(argc==5&&std::string(argv[1])=="--cases"&&std::string(argv[3])=="--device","usage: batch8_fusion_probe --cases cases.tsv --device 0");
  auto cases=read_cases(argv[2]);require(integer(argv[4])==0,"requires CUDA device0");check(cudaSetDevice(0),"select device");
  cudaDeviceProp properties{};check(cudaGetDeviceProperties(&properties,0),"device properties");require(properties.major==8&&properties.minor==9,"requires SM89");
  int driver=0,runtime=0;check(cudaDriverGetVersion(&driver),"driver version");check(cudaRuntimeGetVersion(&runtime),"runtime version");require(runtime==13000,"requires CUDA runtime13.0");
  std::ostringstream uuid;uuid<<std::hex;for(unsigned char b:properties.uuid.bytes){uuid.width(2);uuid.fill('0');uuid<<unsigned(b);}
  std::cout<<"{\"schema\":\"riley.batch8-fusion-native.v1\",\"kind\":\"device\",\"ordinal\":0,\"name\":"<<quote(properties.name)<<",\"uuid_hex\":"<<quote(uuid.str())
   <<",\"compute_major\":8,\"compute_minor\":9,\"driver_version\":"<<driver<<",\"runtime_version\":"<<runtime<<"}\n"<<std::flush;
  check(cudaStreamCreateWithFlags(&state.stream,cudaStreamNonBlocking),"create stream");std::array<std::vector<uint8_t>,2> map_reference;
  for(const auto& c:cases){if(!run_case(state,c,map_reference))++failed;++completed;}
  check(cudaStreamSynchronize(state.stream),"final completion");
 }catch(const std::exception& exception){error=exception.what();}
 state.close_stream();const bool freed=state.created==state.freed&&state.live_bytes==0&&state.cleanup_errors==0;
 const bool passed=error.empty()&&completed==kCases&&failed==0&&freed&&state.stream_destroyed;
 std::cout<<"{\"schema\":\"riley.batch8-fusion-native.v1\",\"kind\":\"summary\",\"passed\":"<<(passed?"true":"false")<<",\"cases\":"<<completed
  <<",\"failed_cases\":"<<failed<<",\"layers\":30,\"positions\":32,\"patterns\":3,\"mappings\":3,\"device_allocations_created\":"<<state.created<<",\"device_allocations_freed\":"<<state.freed
  <<",\"live_device_allocations\":"<<(state.created-state.freed)<<",\"live_device_bytes\":"<<state.live_bytes<<",\"cleanup_errors\":"<<state.cleanup_errors
  <<",\"all_allocations_freed\":"<<(freed?"true":"false")<<",\"stream_destroyed\":"<<(state.stream_destroyed?"true":"false")<<",\"performance_measured\":false,\"performance_claim_eligible\":false,\"error\":"<<quote(error)<<"}\n"<<std::flush;
 return passed?0:1;
}
