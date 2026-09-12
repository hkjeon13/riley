// Synthetic correctness host. Generated fixture_support.hpp reuses the pinned
// attention prototype's guard, descriptor authority, input and mapping helpers.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace riley_cuda_internal {
cudaError_t enqueue_compiled_packed_decode_rope_kv(cudaStream_t,const void*,const void*,const void*,void*,void*,void*,const void*,const void*,const void*) noexcept;
cudaError_t enqueue_compiled_swiglu(cudaStream_t,const void*,const void*,void*) noexcept;
}
namespace riley_multisequence_precise_probe {
cudaError_t enqueue_rope_rows(cudaStream_t,const void*,void*,void*,void*,const void*,const void*,const void*,uint32_t) noexcept;
cudaError_t enqueue_swiglu_rows(cudaStream_t,const void*,void*,const void*,uint32_t) noexcept;
}
namespace {
constexpr size_t kGuard=256,kCases=3456,kOutput=1152,kCache=393216,kMetadata=1280,kProduct=3072,kTable=160*32*4;
const char* patterns[]={"bounded_fingerprint","signed_zero_impulses","tail_cancellation"};
const char* mappings[]={"identity64","reverse64","affine7_3_mod64"};
const int buckets[]={1,2,4,4},active_rows[]={1,2,3,4},seeds[]={0,15,29};
struct Case {int id,layer,tuple,pattern,mode,mapping;};
void require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
void check(cudaError_t status,const char* operation){if(status!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));}
std::string quote(const std::string& text){std::ostringstream out;out<<'"';for(unsigned char c:text){if(c=='"'||c=='\\')out<<'\\'<<c;else if(c<32)out<<"?";else out<<c;}out<<'"';return out.str();}
std::string strings(const std::vector<std::string>& values){std::ostringstream out;out<<'[';for(size_t i=0;i<values.size();++i){if(i)out<<',';out<<quote(values[i]);}out<<']';return out.str();}
int integer(const std::string& text){char* end=nullptr;errno=0;long n=std::strtol(text.c_str(),&end,10);require(errno==0&&end!=text.c_str()&&*end=='\0'&&n>=0&&n<=100000,"invalid index integer");return int(n);}
std::vector<Case> read_cases(const char* path){std::ifstream in(path);require(bool(in),"cannot open case index");std::vector<Case> result;std::string line;
 while(std::getline(in,line)){std::istringstream row(line);std::vector<std::string> f;std::string field;while(std::getline(row,field,'\t'))f.push_back(field);
  require(f.size()==6&&result.size()<kCases,"case TSV extent differs");Case c{integer(f[0]),integer(f[1]),integer(f[2]),integer(f[3]),integer(f[4]),integer(f[5])};
  require(c.id==int(result.size())&&c.layer==seeds[c.id/1152]&&c.tuple==(c.id/36)%32&&c.pattern==(c.id/12)%3&&c.mode==(c.id/3)%4&&c.mapping==c.id%3,"case coverage/order differs");result.push_back(c);}
 require(in.eof()&&result.size()==kCases,"incomplete case index");return result;}
}
#include "fixture_support.hpp"
namespace {
enum Global {QKV,GU,PACKET,COS,SIN,QOUT,PRODUCT,KEYS,VALUES,OKEYS,OVALUES,GLOBALS};
enum Oracle {Q,K,V,META,G,U,OQ,OY,ORACLES};
using Bytes=std::vector<uint8_t>;
Bytes slice(const Bytes& value,size_t begin,size_t count){require(begin+count<=value.size(),"host slice exceeds fixture");return Bytes(value.begin()+begin,value.begin()+begin+count);}
struct Payload {Fixtures fixture;std::array<Bytes,GLOBALS> global;std::vector<bool> kv_written;};
Payload payload(const Case& c){
 auto f=fixture(c);const int active=active_rows[c.mode],bucket=buckets[c.mode];Payload p{std::move(f),{},std::vector<bool>(kCache/2,false)};
 p.global[QKV]=poison(bucket*1920,0x7fc5);p.global[GU]=poison(bucket*6144,0x7fc5);
 p.global[PACKET]=p.fixture.packet;p.global[COS]=Bytes(kTable);p.global[SIN]=Bytes(kTable);
 // Exact FP32 bit fixtures include BF16 halfway values, signed zero and finite
 // subnormals. No host trigonometry or conversion changes the table bytes.
 const uint32_t table[]={0x3f800000,0x3f000000,0xbf000000,0x3f008000,0xbf808000,0,0x80000000,0x00000001,0x80000001,0x00800001};
 for(size_t i=0;i<160*32;++i){put32(p.global[COS],i*4,table[(i+c.layer+c.pattern)%10]);put32(p.global[SIN],i*4,table[(i*3+c.tuple+7)%10]);}
 p.global[QOUT]=poison(bucket*kOutput,0x7fc7);p.global[PRODUCT]=poison(bucket*kProduct,0x7fc7);
 p.global[KEYS]=p.fixture.k;p.global[VALUES]=p.fixture.v;
 for(int r=0;r<active;++r){const auto& row=p.fixture.rows[r];
  std::copy(p.fixture.q.begin()+r*kOutput,p.fixture.q.begin()+(r+1)*kOutput,p.global[QKV].begin()+r*1920);
  for(int h=0;h<3;++h)for(int d=0;d<64;++d){size_t at=((row.blocks[row.pos/16]*3+h)*16+row.pos%16)*64+d;
   put16(p.global[QKV],r*1920+1152+(h*64+d)*2,word(p.fixture.k.data()+2*at));
   put16(p.global[QKV],r*1920+1536+(h*64+d)*2,word(p.fixture.v.data()+2*at));
   p.kv_written[at]=true;put16(p.global[KEYS],2*at,0x7fc3);put16(p.global[VALUES],2*at,0x7fc3);}
  for(int i=0;i<1536;++i){uint16_t gate=sample(0,c.layer*4+r,row.pos,i/64,i%9,i%64,c.pattern);
   uint16_t up=sample(2,c.layer*4+r,row.pos,i/64,i%9,i%64,c.pattern);
   if(c.pattern==1&&i%31==0){gate=uint16_t((i&1?0x8000:0)|1);up=uint16_t((i&1?0:0x8000)|1);}
   put16(p.global[GU],r*6144+2*i,gate);put16(p.global[GU],r*6144+3072+2*i,up);}
 }
 p.global[OKEYS]=p.global[KEYS];p.global[OVALUES]=p.global[VALUES];return p;
}
struct Checked {
 bool finite=true,guards=true,inputs=true,oracles=true,other=true,inactive=true,invariant=true;
 std::array<size_t,4> mismatch{};size_t first_region=0,first_word=0;uint16_t first_a=0,first_b=0;
 size_t total() const {return mismatch[0]+mismatch[1]+mismatch[2]+mismatch[3];}
 bool passed() const {return total()==0&&finite&&guards&&inputs&&oracles&&other&&inactive&&invariant;}
 void compare(const Bytes& a,size_t ao,const Bytes& b,size_t bo,size_t count,size_t region){for(size_t i=0;i<count;++i){auto x=word(a.data()+ao+2*i),y=word(b.data()+bo+2*i);if(x!=y){if(total()==0){first_region=region;first_word=(bo-kGuard)/2+i;first_a=x;first_b=y;}++mismatch[region];}}}
};
struct Workspace {
 State& state;int bucket,capacity;Payload host;std::array<std::unique_ptr<Buffer>,GLOBALS> global;
 std::vector<std::array<std::unique_ptr<Buffer>,ORACLES>> oracle;
 std::vector<std::array<Bytes,2>> before;std::array<Bytes,2> cache_before;
 Workspace(State& s,const Case& c,int rows):state(s),bucket(buckets[c.mode]),capacity(rows),oracle(rows){
  for(auto& item:global)item.reset(new Buffer(state));for(auto& row:oracle)for(auto& item:row)item.reset(new Buffer(state));refresh(c);
 }
 void update(Buffer& buffer,const Bytes& bytes){if(buffer.allocation)buffer.upload(bytes);else buffer.create(bytes);}
 void refresh(const Case& c){require(buckets[c.mode]==bucket&&active_rows[c.mode]<=capacity,"cold workspace shape changed");host=payload(c);
  for(size_t i=0;i<GLOBALS;++i)update(*global[i],host.global[i]);
  for(int r=0;r<capacity;++r){std::array<Bytes,ORACLES> inputs{{slice(host.global[QKV],r*1920,1152),slice(host.global[QKV],r*1920+1152,384),slice(host.global[QKV],r*1920+1536,384),
      slice(host.global[PACKET],128+r*128,80),slice(host.global[GU],r*6144,3072),slice(host.global[GU],r*6144+3072,3072),poison(kOutput,0x7fc9),poison(kProduct,0x7fc9)}};
   for(size_t i=0;i<ORACLES;++i)update(*oracle[r][i],inputs[i]);}
  check(cudaStreamSynchronize(state.stream),"complete fresh guarded fixture uploads");
 }
 void run_oracle(int active){before.clear();for(int r=0;r<capacity;++r){auto& o=oracle[r];if(r<active){
   check(riley_cuda_internal::enqueue_compiled_packed_decode_rope_kv(state.stream,o[Q]->data(),o[K]->data(),o[V]->data(),o[OQ]->data(),global[OKEYS]->data(),global[OVALUES]->data(),global[COS]->data(),global[SIN]->data(),o[META]->data()),"accepted aligned M1 RoPE/KV");
   check(riley_cuda_internal::enqueue_compiled_swiglu(state.stream,o[G]->data(),o[U]->data(),o[OY]->data()),"accepted aligned M1 SwiGLU");}
   before.push_back({o[OQ]->download(),o[OY]->download()});}
  cache_before={global[OKEYS]->download(),global[OVALUES]->download()};
 }
 void enqueue(){check(riley_multisequence_precise_probe::enqueue_rope_rows(state.stream,global[QKV]->data(),global[QOUT]->data(),global[KEYS]->data(),global[VALUES]->data(),global[COS]->data(),global[SIN]->data(),global[PACKET]->data(),bucket),"candidate independent-row RoPE/KV");
  check(riley_multisequence_precise_probe::enqueue_swiglu_rows(state.stream,global[GU]->data(),global[PRODUCT]->data(),global[PACKET]->data(),bucket),"candidate packed-row SwiGLU");}
 Checked inspect(const Case& c,Bytes* mapping_reference){
  const int active=active_rows[c.mode];Checked result;std::array<Bytes,GLOBALS> actual;
  for(size_t i=0;i<GLOBALS;++i){actual[i]=global[i]->download();result.guards=result.guards&&global[i]->guards(actual[i]);if(i<=SIN)result.inputs=result.inputs&&actual[i]==global[i]->expected;}
  result.oracles=actual[OKEYS]==cache_before[0]&&actual[OVALUES]==cache_before[1];
  result.compare(cache_before[0],kGuard,actual[KEYS],kGuard,kCache/2,2);result.compare(cache_before[1],kGuard,actual[VALUES],kGuard,kCache/2,3);
  for(int r=0;r<capacity;++r){for(size_t i=0;i<ORACLES;++i){auto bytes=oracle[r][i]->download();result.guards=result.guards&&oracle[r][i]->guards(bytes);
    if(i<OQ)result.inputs=result.inputs&&bytes==oracle[r][i]->expected;else result.oracles=result.oracles&&bytes==before[r][i-OQ];}
   if(r<active){result.compare(before[r][0],kGuard,actual[QOUT],kGuard+r*kOutput,576,0);result.compare(before[r][1],kGuard,actual[PRODUCT],kGuard+r*kProduct,1536,1);
    result.finite=result.finite&&finite(before[r][0],kGuard,kOutput)&&finite(before[r][1],kGuard,kProduct);}}
  result.finite=result.finite&&finite(actual[QOUT],kGuard,bucket*kOutput)&&finite(actual[PRODUCT],kGuard,bucket*kProduct)
    &&finite(actual[KEYS],kGuard,kCache)&&finite(actual[VALUES],kGuard,kCache)&&finite(cache_before[0],kGuard,kCache)&&finite(cache_before[1],kGuard,kCache);
  for(size_t i=0;i<kCache/2;++i)if(!host.kv_written[i]){result.other=result.other&&word(actual[KEYS].data()+kGuard+2*i)==word(host.global[KEYS].data()+2*i)
    &&word(actual[VALUES].data()+kGuard+2*i)==word(host.global[VALUES].data()+2*i);}
  for(auto pair:std::array<std::pair<int,size_t>,2>{{{QOUT,kOutput},{PRODUCT,kProduct}}})result.inactive=result.inactive&&std::all_of(actual[pair.first].begin()+kGuard+active*pair.second,actual[pair.first].begin()+kGuard+bucket*pair.second,[](uint8_t x){return x==0;});
  if(mapping_reference){Bytes normalized=slice(actual[QOUT],kGuard,bucket*kOutput);auto product=slice(actual[PRODUCT],kGuard,bucket*kProduct);normalized.insert(normalized.end(),product.begin(),product.end());
   for(const auto& row:host.fixture.rows)for(int kind: {KEYS,VALUES})for(int head=0;head<3;++head)for(int dim=0;dim<64;++dim){size_t at=((row.blocks[row.pos/16]*3+head)*16+row.pos%16)*64+dim;
    normalized.push_back(actual[kind][kGuard+2*at]);normalized.push_back(actual[kind][kGuard+2*at+1]);}
   if(c.mapping==0)*mapping_reference=normalized;result.invariant=normalized==*mapping_reference;}
  return result;
 }
 void close(){for(auto& row:oracle)for(auto& item:row)item->close();for(auto& item:global)item->close();}
};
void report(const Case& c,const Checked& result,bool freed,bool transition,size_t step,size_t allocations){
 int active=active_rows[c.mode],bucket=buckets[c.mode];bool numeric=false;for(int r=0;r<active;++r)numeric=numeric||position(c,r)==159;
 std::cout<<"{\"schema\":\"riley.multisequence-precise-native.v1\",\"kind\":"<<quote(transition?"graph_transition":"case")<<",\"case_id\":"<<c.id
  <<",\"synthetic_seed\":"<<c.layer<<",\"position_tuple\":"<<c.tuple<<",\"pattern\":"<<quote(patterns[c.pattern])<<",\"mapping\":"<<quote(mappings[c.mapping])
  <<",\"bucket\":"<<bucket<<",\"active_rows\":"<<active<<",\"positions\":[";
 for(int r=0;r<active;++r){if(r)std::cout<<',';std::cout<<position(c,r);}std::cout<<"],\"numerical_only_position159\":"<<(numeric?"true":"false")
  <<",\"candidate_invocations\":2,\"oracle_invocations\":"<<2*active<<",\"q_words_compared\":"<<active*576<<",\"product_words_compared\":"<<active*1536
  <<",\"key_words_compared\":"<<kCache/2<<",\"value_words_compared\":"<<kCache/2<<",\"inactive_words_checked\":"<<(bucket-active)*(576+1536)
  <<",\"passed\":"<<(result.passed()&&freed?"true":"false")<<",\"exact_outputs\":"<<(result.total()==0?"true":"false")
  <<",\"finite_outputs\":"<<(result.finite?"true":"false")<<",\"guards_intact\":"<<(result.guards?"true":"false")<<",\"inputs_unchanged\":"<<(result.inputs?"true":"false")
  <<",\"oracle_outputs_unchanged\":"<<(result.oracles?"true":"false")<<",\"other_kv_slots_unchanged\":"<<(result.other?"true":"false")
  <<",\"inactive_outputs_zero\":"<<(result.inactive?"true":"false")<<",\"mapping_invariant\":"<<(result.invariant?"true":"false")
  <<",\"mapping_invariant_checked\":"<<(transition?"false":"true")<<",\"all_allocations_freed\":"<<(freed?"true":"false")
  <<",\"allocations_created\":"<<allocations<<",\"mismatch_words\":"<<result.total()<<",\"mismatch_by_region\":[";
 for(size_t i=0;i<4;++i){if(i)std::cout<<',';std::cout<<result.mismatch[i];}std::cout<<']';
 if(result.total())std::cout<<",\"first_mismatch\":{\"region\":"<<result.first_region<<",\"word\":"<<result.first_word<<",\"oracle_bits\":"<<result.first_a<<",\"candidate_bits\":"<<result.first_b<<'}';
 if(transition)std::cout<<",\"step\":"<<step<<",\"capture_count\":1,\"graph_nodes\":2,\"parameter_updates\":0,\"replay_index\":"<<step+1;
 std::cout<<"}\n"<<std::flush;
}
struct Graph {
 State& state;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;bool capturing=false;
 explicit Graph(State& s):state(s){}
 void capture(Workspace& work){check(cudaStreamBeginCapture(state.stream,cudaStreamCaptureModeThreadLocal),"begin precise graph capture");capturing=true;work.enqueue();
  auto status=cudaStreamEndCapture(state.stream,&graph);capturing=false;if(graph)++state.graphs_created;check(status,"end precise graph capture");
  size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count),"graph node count");require(count==2,"precise graph needs exactly two kernel nodes");
  std::array<cudaGraphNode_t,2> nodes{};check(cudaGraphGetNodes(graph,nodes.data(),&count),"graph nodes");for(auto node:nodes){cudaGraphNodeType type{};check(cudaGraphNodeGetType(node,&type),"graph node type");require(type==cudaGraphNodeTypeKernel,"unexpected graph node");}
  status=cudaGraphInstantiate(&exec,graph,0);if(exec)++state.execs_created;check(status,"instantiate precise graph");}
 void close() noexcept {if(capturing){cudaGraph_t recovered=nullptr;auto status=cudaStreamEndCapture(state.stream,&recovered);capturing=false;if(status!=cudaSuccess)++state.cleanup_errors;
   if(recovered){++state.graphs_created;if(cudaGraphDestroy(recovered)==cudaSuccess)++state.graphs_destroyed;else ++state.cleanup_errors;}}
  if(exec){if(cudaGraphExecDestroy(exec)==cudaSuccess)++state.execs_destroyed;else ++state.cleanup_errors;exec=nullptr;}
  if(graph){if(cudaGraphDestroy(graph)==cudaSuccess)++state.graphs_destroyed;else ++state.cleanup_errors;graph=nullptr;}}
 ~Graph(){close();}
};
bool allocation_clean(const State& state){return state.created==state.freed&&state.live_bytes==0&&state.cleanup_errors==0;}
size_t graph_transitions(State& state){
 const std::array<Case,3> cases{{{0,0,0,0,2,0},{1,15,17,1,3,2},{2,29,5,2,2,1}}};
 size_t start=state.created;Workspace workspace(state,cases[0],4);Graph graph(state);graph.capture(workspace);const size_t allocated=state.created;
 std::array<Checked,3> results;
 for(size_t i=0;i<cases.size();++i){workspace.refresh(cases[i]);workspace.run_oracle(active_rows[cases[i].mode]);
  check(cudaGraphLaunch(graph.exec,state.stream),"launch retained precise graph");check(cudaStreamSynchronize(state.stream),"complete retained precise graph");
  results[i]=workspace.inspect(cases[i],nullptr);require(state.created==allocated,"hot graph path allocated device memory");}
 graph.close();workspace.close();const bool freed=allocation_clean(state);size_t failed=0;
 for(size_t i=0;i<cases.size();++i){report(cases[i],results[i],freed,true,i,state.created-start);if(!results[i].passed()||!freed)++failed;}
 return failed;
}
}
int main(int argc,char** argv){
 State state;size_t completed=0,failed=0,transitions=0,transition_failed=0;std::string error;std::vector<std::string> maps;
 try{require(argc==5&&std::string(argv[1])=="--cases"&&std::string(argv[3])=="--device"&&integer(argv[4])==0,"usage: probe --cases cases.tsv --device 0");
  auto cases=read_cases(argv[2]);check(cudaSetDevice(0),"select device");cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0),"device properties");require(prop.major==8&&prop.minor==9,"SM89 required");
  int driver=0,runtime=0;check(cudaDriverGetVersion(&driver),"driver version");check(cudaRuntimeGetVersion(&runtime),"runtime version");require(runtime==13000,"CUDA13.0 required");
  std::ostringstream uuid;uuid<<std::hex;for(unsigned char value:prop.uuid.bytes){uuid.width(2);uuid.fill('0');uuid<<unsigned(value);}require(uuid.str()=="9087e4256acab722b8c9cc0423b39fb0","GPU UUID differs");
  maps=runtime_maps();std::cout<<"{\"schema\":\"riley.multisequence-precise-native.v1\",\"kind\":\"device\",\"uuid_hex\":"<<quote(uuid.str())<<",\"name\":"<<quote(prop.name)
   <<",\"compute_major\":8,\"compute_minor\":9,\"runtime_version\":"<<runtime<<",\"driver_version\":"<<driver<<",\"runtime_maps\":"<<strings(maps)<<"}\n"<<std::flush;
  check(cudaStreamCreateWithFlags(&state.stream,cudaStreamNonBlocking),"create stream");Bytes mapping_reference;
  for(const auto& c:cases){size_t count=state.created;Workspace work(state,c,active_rows[c.mode]);work.run_oracle(active_rows[c.mode]);work.enqueue();
   check(cudaStreamSynchronize(state.stream),"complete precise row kernels");auto result=work.inspect(c,&mapping_reference);work.close();bool freed=allocation_clean(state);
   report(c,result,freed,false,0,state.created-count);if(!result.passed()||!freed)++failed;++completed;}
  transition_failed=graph_transitions(state);transitions=3;check(cudaStreamSynchronize(state.stream),"final completion");maps=runtime_maps();
 }catch(const std::exception& failure){error=failure.what();}
 state.close_stream();bool complete=error.empty()&&completed==kCases&&transitions==3&&allocation_clean(state)&&state.stream_destroyed
  &&state.graphs_created==1&&state.graphs_destroyed==1&&state.execs_created==1&&state.execs_destroyed==1;
 std::cout<<"{\"schema\":\"riley.multisequence-precise-native.v1\",\"kind\":\"summary\",\"completed\":"<<(complete?"true":"false")
  <<",\"precise_bitwise_equal\":"<<(complete&&failed==0&&transition_failed==0?"true":"false")<<",\"cases\":"<<completed<<",\"failed_cases\":"<<failed
  <<",\"synthetic_seeds\":3,\"position_tuples\":32,\"patterns\":3,\"mappings\":3,\"modes\":4,\"graph_transitions\":"<<transitions<<",\"failed_graph_transitions\":"<<transition_failed
  <<",\"allocations_created\":"<<state.created<<",\"allocations_freed\":"<<state.freed<<",\"live_bytes\":"<<state.live_bytes<<",\"cleanup_errors\":"<<state.cleanup_errors
  <<",\"graphs_created\":"<<state.graphs_created<<",\"graphs_destroyed\":"<<state.graphs_destroyed<<",\"execs_created\":"<<state.execs_created<<",\"execs_destroyed\":"<<state.execs_destroyed
  <<",\"stream_destroyed\":"<<(state.stream_destroyed?"true":"false")<<",\"runtime_maps\":"<<strings(maps)<<",\"performance_claim\":false,\"error\":"<<quote(error)<<"}\n"<<std::flush;
 return complete?0:1;
}
