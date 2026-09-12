// Correctness-only standalone probe. Numerical kernels are compiled separately
// from the frozen production graph_numerics_precise.cu with its CMake flags.
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
cudaError_t enqueue_compiled_prefill_m16_gemm(cudaStream_t,const void*,const void*,void*,int,int,int,const void*) noexcept;
cudaError_t enqueue_compiled_prefill_gemm_rows(cudaStream_t,const void*,const void*,void*,int,int,int,const void*,uint32_t) noexcept;
}
namespace {
constexpr size_t kRows=128,kGuard=256,kCases=630;
const char* projections[]={"q","k","v","o","gate","up","down"};
const char* patterns[]={"row_fingerprint","signed_zero_impulses","bounded_cancellation"};
constexpr int widths[]={576,192,192,576,1536,1536,576};
constexpr int depths[]={576,576,576,576,576,576,1536};
constexpr int intervals[]={192,192,192,128,0,0,320};
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
struct Case {int id,layer,projection,pattern,n,k,interval;std::string input,weight;};
std::vector<Case> read_cases(const std::string& path){
 std::ifstream in(path);require(bool(in),"cannot open cases index");std::vector<Case> cases;std::string line;
 while(std::getline(in,line)){
  std::vector<std::string> fields;std::istringstream row(line);std::string field;
  while(std::getline(row,field,'\t'))fields.push_back(field);
  require(fields.size()==9,"case index must have nine TSV fields");
  Case c{integer(fields[0]),integer(fields[1]),integer(fields[2]),integer(fields[3]),integer(fields[4]),integer(fields[5]),integer(fields[6]),fields[7],fields[8]};
  require(cases.size()<kCases && c.id==static_cast<int>(cases.size()) && c.layer==c.id/21 && c.projection==(c.id/3)%7 && c.pattern==c.id%3,"case coverage/order differs");
  require(c.n==widths[c.projection] && c.k==depths[c.projection] && c.interval==intervals[c.projection],"projection shape or rounding interval differs");
  require(!c.input.empty() && !c.weight.empty(),"empty fixture path");cases.push_back(c);
 }
 require(in.eof() && cases.size()==kCases,"requires all 30 layers x 7 projections x 3 patterns");return cases;
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
bool run_case(State& state,const Case& c){
 const auto x=read_bytes(c.input,kRows*c.k*2),w=read_bytes(c.weight,size_t(c.n)*c.k*2);
 require(finite(x,0,x.size()) && finite(w,0,w.size()),"input or checkpoint weight is nonfinite");
 const size_t output_bytes=kRows*c.n*2;
 Buffer input(state),weight(state),oracle(state),candidate(state),position(state);
 input.create(x);weight.create(w);oracle.create(poison(output_bytes,0x7fc1));candidate.create(poison(output_bytes,0x7fc3));position.create({127,0,0,0});
 check(cudaStreamSynchronize(state.stream),"complete fixture uploads");
 check(cudaGetLastError(),"clean CUDA launch state");
 check(riley_cuda_internal::enqueue_compiled_prefill_gemm_rows(state.stream,input.data(),weight.data(),oracle.data(),c.n,c.k,c.interval,position.data(),128),"enqueue original rows128 projection");
 check(cudaGetLastError(),"original projection launch status");
 check(cudaStreamSynchronize(state.stream),"complete original projection");
 const auto original=oracle.download();
 check(riley_cuda_internal::enqueue_compiled_prefill_m16_gemm(state.stream,input.data(),weight.data(),candidate.data(),c.n,c.k,c.interval,position.data()),"enqueue candidate M16 projection");
 check(cudaGetLastError(),"M16 projection launch status");
 check(cudaStreamSynchronize(state.stream),"complete M16 projection");
 const auto result=candidate.download(),input_after=input.download(),weight_after=weight.download(),position_after=position.download(),oracle_after=oracle.download();
 const bool unchanged_x=input_after==input.expected,unchanged_w=weight_after==weight.expected,unchanged_pos=position_after==position.expected;
 const bool guards=input.guards(input_after)&&weight.guards(weight_after)&&position.guards(position_after)&&oracle.guards(original)&&oracle.guards(oracle_after)&&candidate.guards(result);
 const bool finite_outputs=finite(original,kGuard,output_bytes)&&finite(result,kGuard,output_bytes);
 const bool equal=std::memcmp(original.data()+kGuard,result.data()+kGuard,output_bytes)==0;
 const bool oracle_unchanged=oracle_after==original;
 size_t mismatch=output_bytes;
 for(size_t i=0;i<output_bytes;i+=2)if(word(original.data()+kGuard+i)!=word(result.data()+kGuard+i)){mismatch=i;break;}
 input.close();weight.close();position.close();oracle.close();candidate.close();
 const bool freed=state.created==state.freed && state.live_bytes==0 && state.cleanup_errors==0;
 const bool passed=equal&&finite_outputs&&guards&&unchanged_x&&unchanged_w&&unchanged_pos&&oracle_unchanged&&freed;
 std::cout<<"{\"schema\":\"riley.batch6-projection-native.v1\",\"kind\":\"case\",\"case_id\":"<<c.id<<",\"layer\":"<<c.layer
  <<",\"projection\":"<<quote(projections[c.projection])<<",\"pattern\":"<<quote(patterns[c.pattern])<<",\"rows\":128,\"n\":"<<c.n<<",\"k\":"<<c.k<<",\"interval\":"<<c.interval
  <<",\"output_bytes_compared\":"<<output_bytes<<",\"passed\":"<<(passed?"true":"false")<<",\"exact_outputs\":"<<(equal?"true":"false")
  <<",\"finite_outputs\":"<<(finite_outputs?"true":"false")<<",\"guards_intact\":"<<(guards?"true":"false")<<",\"inputs_unchanged\":"<<(unchanged_x?"true":"false")
  <<",\"weights_unchanged\":"<<(unchanged_w?"true":"false")<<",\"position_unchanged\":"<<(unchanged_pos?"true":"false")<<",\"oracle_output_unchanged\":"<<(oracle_unchanged?"true":"false")
  <<",\"all_allocations_freed\":"<<(freed?"true":"false");
 if(mismatch<output_bytes)std::cout<<",\"first_mismatch\":{\"row\":"<<mismatch/(2*c.n)<<",\"column\":"<<(mismatch/2)%c.n<<",\"original_bits\":"<<word(original.data()+kGuard+mismatch)<<",\"candidate_bits\":"<<word(result.data()+kGuard+mismatch)<<"}";
 std::cout<<"}\n"<<std::flush;return passed;
}
} // namespace
int main(int argc,char** argv){
 State state;size_t completed=0,failed=0;std::string error;
 try{
  require(argc==5 && std::string(argv[1])=="--cases" && std::string(argv[3])=="--device","usage: batch6_projection_probe --cases cases.tsv --device 0");
  auto cases=read_cases(argv[2]);int device=integer(argv[4]);require(device==0,"fixed workload requires CUDA device0");
  check(cudaSetDevice(device),"select device");cudaDeviceProp properties{};check(cudaGetDeviceProperties(&properties,device),"device properties");
  require(properties.major==8 && properties.minor==9,"probe requires SM89 hardware");
  int driver=0,runtime=0;check(cudaDriverGetVersion(&driver),"driver version");check(cudaRuntimeGetVersion(&runtime),"runtime version");
  require(runtime==13000,"fixed profile requires CUDA runtime13.0");
  std::ostringstream uuid;uuid<<std::hex;for(unsigned char b:properties.uuid.bytes){uuid.width(2);uuid.fill('0');uuid<<unsigned(b);}
  std::cout<<"{\"schema\":\"riley.batch6-projection-native.v1\",\"kind\":\"device\",\"ordinal\":0,\"name\":"<<quote(properties.name)<<",\"uuid_hex\":"<<quote(uuid.str())<<",\"compute_major\":8,\"compute_minor\":9,\"driver_version\":"<<driver<<",\"runtime_version\":"<<runtime<<"}\n"<<std::flush;
  check(cudaStreamCreateWithFlags(&state.stream,cudaStreamNonBlocking),"create stream");
  for(const auto& c:cases){if(!run_case(state,c))++failed;++completed;}
  check(cudaStreamSynchronize(state.stream),"final stream completion");
 }catch(const std::exception& exception){error=exception.what();}
 state.close_stream();const bool freed=state.created==state.freed && state.live_bytes==0 && state.cleanup_errors==0;
 const bool passed=error.empty() && completed==kCases && failed==0 && freed && state.stream_destroyed;
 std::cout<<"{\"schema\":\"riley.batch6-projection-native.v1\",\"kind\":\"summary\",\"passed\":"<<(passed?"true":"false")
  <<",\"cases\":"<<completed<<",\"failed_cases\":"<<failed<<",\"layers\":30,\"projections\":7,\"patterns\":3,\"rows\":128"
  <<",\"device_allocations_created\":"<<state.created<<",\"device_allocations_freed\":"<<state.freed<<",\"live_device_allocations\":"<<(state.created-state.freed)<<",\"live_device_bytes\":"<<state.live_bytes
  <<",\"cleanup_errors\":"<<state.cleanup_errors<<",\"all_allocations_freed\":"<<(freed?"true":"false")<<",\"stream_destroyed\":"<<(state.stream_destroyed?"true":"false")
  <<",\"performance_measured\":false,\"performance_claim_eligible\":false,\"error\":"<<quote(error)<<"}\n"<<std::flush;
 return passed?0:1;
}
