// Standalone CPU fixture/host harness. Numerical TUs are compiled separately
// with the exact accepted Batch7 CMake flags, including --use_fast_math.
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
cudaError_t enqueue_compiled_packed_decode_attention_two_warp(cudaStream_t,const void*,const void*,const void*,void*,const void*) noexcept;
}
namespace riley_multisequence_attention_probe {
cudaError_t enqueue_rows(cudaStream_t,const void*,const void*,const void*,void*,const void*,uint32_t) noexcept;
}
namespace {
constexpr size_t kGuard=256,kCases=34560,kOutput=1152,kCache=393216,kMetadata=1280;
const char* patterns[]={"bounded_fingerprint","signed_zero_impulses","tail_cancellation"};
const char* mappings[]={"identity64","reverse64","affine7_3_mod64"};
const int buckets[]={1,2,4,4},active_rows[]={1,2,3,4};
void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
void check(cudaError_t status,const char* operation){if(status!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));}
std::string quote(const std::string& text){std::ostringstream out;out<<'"';for(unsigned char c:text){if(c=='"'||c=='\\')out<<'\\'<<c;else if(c<32)out<<"?";else out<<c;}out<<'"';return out.str();}
std::string strings(const std::vector<std::string>& values){std::ostringstream s;s<<'[';for(size_t i=0;i<values.size();++i){if(i)s<<',';s<<quote(values[i]);}s<<']';return s.str();}
int integer(const std::string& text){char* end=nullptr;errno=0;long value=std::strtol(text.c_str(),&end,10);require(errno==0&&end!=text.c_str()&&*end=='\0'&&value>=0&&value<=100000,"invalid index integer");return int(value);}
struct Case {int id,layer,tuple,pattern,mode,mapping;};
std::vector<Case> read_cases(const std::string& path){
 std::ifstream in(path);require(bool(in),"cannot open case index");std::vector<Case> result;std::string line;
 while(std::getline(in,line)){std::istringstream row(line);std::vector<std::string> fields;std::string field;while(std::getline(row,field,'\t'))fields.push_back(field);
  require(fields.size()==6,"six TSV fields required");Case c{integer(fields[0]),integer(fields[1]),integer(fields[2]),integer(fields[3]),integer(fields[4]),integer(fields[5])};
  require(result.size()<kCases&&c.id==int(result.size())&&c.layer==c.id/1152&&c.tuple==(c.id/36)%32&&c.pattern==(c.id/12)%3&&c.mode==(c.id/3)%4&&c.mapping==c.id%3,"exact fixture coverage/order differs");result.push_back(c);}
 require(in.eof()&&result.size()==kCases,"incomplete fixture index");return result;
}
uint16_t word(const uint8_t* p){return uint16_t(p[0])|(uint16_t(p[1])<<8);}
uint32_t dword(const uint8_t* p){uint32_t value=0;for(int i=0;i<4;++i)value|=uint32_t(p[i])<<(8*i);return value;}
uint64_t qword(const uint8_t* p){uint64_t value=0;for(int i=0;i<8;++i)value|=uint64_t(p[i])<<(8*i);return value;}
bool finite(const std::vector<uint8_t>& bytes,size_t start,size_t count){for(size_t i=start;i<start+count;i+=2)if((word(bytes.data()+i)&0x7f80)==0x7f80)return false;return true;}
struct State {
 cudaStream_t stream=nullptr;size_t created=0,freed=0,live_bytes=0,cleanup_errors=0;bool stream_destroyed=false;
 size_t graphs_created=0,graphs_destroyed=0,execs_created=0,execs_destroyed=0;
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
 void upload(const std::vector<uint8_t>& payload){
  require(allocation&&payload.size()+2*kGuard==bytes,"fixed parent extent changed");
  expected.assign(bytes,0xa5);std::copy(payload.begin(),payload.end(),expected.begin()+kGuard);std::fill(expected.end()-kGuard,expected.end(),0x5a);
  check(cudaMemcpyAsync(allocation,expected.data(),bytes,cudaMemcpyHostToDevice,state.stream),"refresh fixed guarded parent");
 }
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
void put64(std::vector<uint8_t>& bytes,size_t offset,uint64_t value){for(int b=0;b<8;++b)bytes[offset+b]=uint8_t(value>>(8*b));}
int physical(int row,int logical,int mapping){int index=row*10+logical;return mapping==0?index:mapping==1?63-index:(7*index+3)%64;}
int position(const Case& c,int row){return 128+(c.tuple+11*row)%32;}
struct Row {uint64_t tag,cookie;uint32_t token,pos,slot;std::array<uint32_t,10> blocks;};
struct Fixtures {
 std::vector<uint8_t> q,k,v,canonical,packet;
 std::vector<Row> rows;
 std::array<int,64> physical_owner;
};
void validate_descriptor(const Fixtures& f,const Case& c){
 const auto& p=f.canonical;const int active=active_rows[c.mode],bucket=buckets[c.mode];
 require(p.size()==1280&&dword(p.data())==0x31444d52&&word(p.data()+4)==1&&word(p.data()+6)==128,"descriptor header magic/version");
 for(auto pair:std::vector<std::pair<int,uint32_t>>{{8,1280},{12,128},{16,1},{20,uint32_t(bucket)},{24,uint32_t(active)},{28,uint32_t(active)},{32,4},{36,10},{40,49152},{44,64},{104,0},{108,0},{112,0},{116,0}})
  require(dword(p.data()+pair.first)==pair.second,"descriptor header geometry");
 require(qword(p.data()+48)==1&&qword(p.data()+56)==uint64_t(c.id+1)&&qword(p.data()+64)==uint64_t(c.id+1)&&qword(p.data()+120)==0,"descriptor replay identity");
 require(std::all_of(p.begin()+72,p.begin()+104,[](uint8_t x){return x==0x5a;}),"descriptor synthetic catalog identity");
 std::set<uint32_t> reserved,live;std::set<uint64_t> tags,cookies;std::set<uint32_t> slots;
 for(int r=0;r<active;++r){const Row& row=f.rows[r];size_t b=128+r*128;int count=row.pos/16+1;
  require(row.pos>=128&&row.pos<160&&row.pos==uint32_t(position(c,r))&&row.token<49152,"synthetic row position/token");
  require(tags.insert(row.tag).second&&cookies.insert(row.cookie).second&&row.tag&&row.cookie&&slots.insert(row.slot).second&&row.slot<uint32_t(active),"synthetic row identity");
  for(int block=0;block<10;++block){auto id=row.blocks[block];require(id<64&&reserved.insert(id).second&&f.physical_owner[id]==r,"full reserved ten-block authority aliases");}
  require(dword(p.data()+b)==row.token&&dword(p.data()+b+4)==row.pos&&dword(p.data()+b+8)==0&&dword(p.data()+b+12)==uint32_t(count)&&dword(p.data()+b+76)==0,"C10 row prefix");
  int total=0;
  for(int block=0;block<10;++block){uint32_t id=dword(p.data()+b+16+block*4);uint16_t valid=word(p.data()+b+56+block*2);
   if(block<count){require(id==row.blocks[block]&&f.physical_owner[id]==r&&live.insert(id).second,"live block authority differs");require(valid==(block==count-1?row.pos%16+1:16),"valid target count differs");total+=valid;}
   else require(id==0&&valid==0,"unused C10 fields nonzero");}
  require(total==int(row.pos+1)&&qword(p.data()+b+80)==row.tag&&qword(p.data()+b+88)==row.cookie,"row reservation identity");
  require(dword(p.data()+b+96)==row.pos+1&&dword(p.data()+b+100)==row.slot&&dword(p.data()+b+104)==1&&dword(p.data()+b+108)==row.pos-127&&dword(p.data()+b+112)==1&&word(p.data()+b+116)==1,"descriptor row fields");
  require(std::all_of(p.begin()+b+118,p.begin()+b+128,[](uint8_t x){return x==0;}),"row reserved bytes nonzero");
 }
 require(std::all_of(p.begin()+128+active*128,p.end(),[](uint8_t x){return x==0;}),"inactive/P128/padding bytes nonzero");
}
Fixtures fixture(const Case& c){
 int active=active_rows[c.mode],bucket=buckets[c.mode];
 Fixtures f{poison(bucket*kOutput,0x7fc5),std::vector<uint8_t>(kCache),std::vector<uint8_t>(kCache),std::vector<uint8_t>(kMetadata),{}, {},{}};
 f.physical_owner.fill(-1);
 for(size_t i=0;i<kCache;i+=2){put16(f.k,i,uint16_t(0xbe80+(i/2)%128));put16(f.v,i,uint16_t(0x3e80+(i/2)%128));}
 auto& p=f.canonical;
 put32(p,0,0x31444d52);put16(p,4,1);put16(p,6,128);
 for(auto pair:std::vector<std::pair<int,uint32_t>>{{8,1280},{12,128},{16,1},{20,uint32_t(bucket)},{24,uint32_t(active)},{28,uint32_t(active)},{32,4},{36,10},{40,49152},{44,64}})put32(p,pair.first,pair.second);
 put64(p,48,1);put64(p,56,c.id+1);put64(p,64,c.id+1);std::fill(p.begin()+72,p.begin()+104,0x5a);
 for(int r=0;r<active;++r){
  Row row{uint64_t(100+c.layer*4+r),uint64_t(c.id*4+r+1),uint32_t(28+r),uint32_t(position(c,r)),uint32_t((r+active-1)%active),{}};
  for(int block=0;block<10;++block){int id=physical(r,block,c.mapping);require(f.physical_owner[id]==-1,"fixture full reservation aliases");f.physical_owner[id]=r;row.blocks[block]=id;}
  f.rows.push_back(row);int synthetic_layer=c.layer*4+r;
  for(int head=0;head<9;++head)for(int dim=0;dim<64;++dim)put16(f.q,r*kOutput+(head*64+dim)*2,sample(0,synthetic_layer,row.pos,0,head,dim,c.pattern));
  for(int token=0;token<160;++token)for(int head=0;head<3;++head)for(int dim=0;dim<64;++dim){
   size_t offset=((row.blocks[token/16]*3+head)*16+token%16)*64*2+dim*2;
   put16(f.k,offset,sample(1,synthetic_layer,0,token,head,dim,c.pattern));put16(f.v,offset,sample(2,synthetic_layer,0,token,head,dim,c.pattern));}
  size_t b=128+r*128;int live=row.pos/16+1;put32(p,b,row.token);put32(p,b+4,row.pos);put32(p,b+12,live);
  for(int i=0;i<live;++i){put32(p,b+16+i*4,row.blocks[i]);put16(p,b+56+i*2,i==live-1?row.pos%16+1:16);}
  put64(p,b+80,row.tag);put64(p,b+88,row.cookie);put32(p,b+96,row.pos+1);put32(p,b+100,row.slot);put32(p,b+104,1);put32(p,b+108,row.pos-127);put32(p,b+112,1);put16(p,b+116,1);
 }
 validate_descriptor(f,c);
 // Deliberate device-only poison after validating the canonical fixture. This
 // tests the kernel's inactive gate, not a production descriptor admission.
 f.packet=f.canonical;std::fill(f.packet.begin()+128+active*128,f.packet.begin()+640,0xff);
 require(finite(f.q,0,active*kOutput)&&finite(f.k,0,kCache)&&finite(f.v,0,kCache),"nonfinite active fixture");
 return f;
}
std::vector<std::string> runtime_maps(){
 std::ifstream file("/proc/self/maps");require(bool(file),"cannot inspect native runtime maps");std::vector<std::string> lines;std::string line;
 while(std::getline(file,line))if(line.find("/libcuda.so")!=std::string::npos||line.find("/libcudart.so")!=std::string::npos){require(line.find(" (deleted)")==std::string::npos,"deleted native library");lines.push_back(line);}
 require(!lines.empty(),"native runtime mappings absent");return lines;
}
bool run_case(State& state,const Case& c,std::vector<uint8_t>& mapping_reference){
 auto f=fixture(c);int active=active_rows[c.mode],bucket=buckets[c.mode];
 Buffer q(state),k(state),v(state),packet(state),candidate(state);
 q.create(f.q);k.create(f.k);v.create(f.v);packet.create(f.packet);candidate.create(poison(bucket*kOutput,0x7fc7));
 std::vector<std::unique_ptr<Buffer>> oq,om,oy;std::vector<std::vector<uint8_t>> before;
 for(int r=0;r<active;++r){
  oq.emplace_back(new Buffer(state));om.emplace_back(new Buffer(state));oy.emplace_back(new Buffer(state));
  oq.back()->create(std::vector<uint8_t>(f.q.begin()+r*kOutput,f.q.begin()+(r+1)*kOutput));
  om.back()->create(std::vector<uint8_t>(f.canonical.begin()+128+r*128,f.canonical.begin()+208+r*128));
  oy.back()->create(poison(kOutput,0x7fc9));
 }
 check(cudaStreamSynchronize(state.stream),"fixture upload completion");check(cudaGetLastError(),"clean launch state");
 for(int r=0;r<active;++r){
  check(riley_cuda_internal::enqueue_compiled_packed_decode_attention_two_warp(state.stream,oq[r]->data(),k.data(),v.data(),oy[r]->data(),om[r]->data()),"independent accepted M1 attention");
  check(cudaStreamSynchronize(state.stream),"M1 completion");before.push_back(oy[r]->download());
 }
 check(riley_multisequence_attention_probe::enqueue_rows(state.stream,q.data(),k.data(),v.data(),candidate.data(),packet.data(),bucket),"true row attention");
 check(cudaStreamSynchronize(state.stream),"row attention completion");auto result=candidate.download();
 auto qa=q.download(),ka=k.download(),va=v.download(),pa=packet.download();
 bool inputs=qa==q.expected&&pa==packet.expected,kv=ka==k.expected&&va==v.expected,oracles=true;
 bool guards=q.guards(qa)&&k.guards(ka)&&v.guards(va)&&packet.guards(pa)&&candidate.guards(result),allfinite=finite(result,kGuard,bucket*kOutput);
 size_t mismatches=0,first_row=0,first_word=0;uint16_t first_a=0,first_b=0;
 for(int r=0;r<active;++r){
  auto xa=oq[r]->download(),ma=om[r]->download(),ya=oy[r]->download();inputs=inputs&&xa==oq[r]->expected&&ma==om[r]->expected;oracles=oracles&&ya==before[r];
  guards=guards&&oq[r]->guards(xa)&&om[r]->guards(ma)&&oy[r]->guards(ya)&&oy[r]->guards(before[r]);allfinite=allfinite&&finite(before[r],kGuard,kOutput);
  for(size_t i=0;i<576;++i){auto a=word(before[r].data()+kGuard+2*i),b=word(result.data()+kGuard+r*kOutput+2*i);if(a!=b){if(mismatches==0){first_row=r;first_word=i;first_a=a;first_b=b;}++mismatches;}}
 }
 bool inactive=std::all_of(result.begin()+kGuard+active*kOutput,result.begin()+kGuard+bucket*kOutput,[](uint8_t x){return x==0;});
 std::vector<uint8_t> payload(result.begin()+kGuard,result.end()-kGuard);
 if(c.mapping==0)mapping_reference=payload;bool invariant=payload==mapping_reference;
 for(int r=0;r<active;++r){oq[r]->close();om[r]->close();oy[r]->close();}
 q.close();k.close();v.close();packet.close();candidate.close();
 bool freed=state.created==state.freed&&state.live_bytes==0&&state.cleanup_errors==0;
 bool exact=mismatches==0,passed=exact&&allfinite&&guards&&inputs&&kv&&oracles&&inactive&&invariant&&freed;
 bool numerical_only=false;for(int r=0;r<active;++r)numerical_only=numerical_only||position(c,r)==159;
 std::cout<<"{\"schema\":\"riley.multisequence-attention-native.v1\",\"kind\":\"case\",\"case_id\":"<<c.id<<",\"layer\":"<<c.layer<<",\"position_tuple\":"<<c.tuple
  <<",\"pattern\":"<<quote(patterns[c.pattern])<<",\"mapping\":"<<quote(mappings[c.mapping])<<",\"bucket\":"<<bucket<<",\"active_rows\":"<<active<<",\"positions\":[";
 for(int r=0;r<active;++r){if(r)std::cout<<',';std::cout<<position(c,r);}
 std::cout<<"],\"numerical_only_position159\":"<<(numerical_only?"true":"false")<<",\"candidate_invocations\":1,\"oracle_invocations\":"<<active
  <<",\"active_words_compared\":"<<active*576<<",\"inactive_words_checked\":"<<(bucket-active)*576<<",\"passed\":"<<(passed?"true":"false")
  <<",\"exact_outputs\":"<<(exact?"true":"false")<<",\"finite_outputs\":"<<(allfinite?"true":"false")<<",\"guards_intact\":"<<(guards?"true":"false")
  <<",\"inputs_unchanged\":"<<(inputs?"true":"false")<<",\"kv_unchanged\":"<<(kv?"true":"false")<<",\"oracle_outputs_unchanged\":"<<(oracles?"true":"false")
  <<",\"inactive_output_zero\":"<<(inactive?"true":"false")<<",\"mapping_invariant\":"<<(invariant?"true":"false")<<",\"all_allocations_freed\":"<<(freed?"true":"false")<<",\"mismatch_words\":"<<mismatches;
 if(mismatches)std::cout<<",\"first_mismatch\":{\"row\":"<<first_row<<",\"word\":"<<first_word<<",\"oracle_bits\":"<<first_a<<",\"candidate_bits\":"<<first_b<<"}";
 std::cout<<"}\n"<<std::flush;return passed;
}
struct RetainedGraph {
 State& state;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;bool capturing=false;size_t nodes=0;
 explicit RetainedGraph(State& value):state(value){}
 void capture(Buffer& q,Buffer& k,Buffer& v,Buffer& out,Buffer& packet){
  require(!graph&&!exec&&!capturing,"graph captured twice");
  check(cudaStreamBeginCapture(state.stream,cudaStreamCaptureModeThreadLocal),"begin retained attention capture");capturing=true;
  check(riley_multisequence_attention_probe::enqueue_rows(state.stream,q.data(),k.data(),v.data(),out.data(),packet.data(),4),"capture fixed bucket4 attention");
  auto status=cudaStreamEndCapture(state.stream,&graph);capturing=false;if(graph)++state.graphs_created;check(status,"end retained attention capture");
  check(cudaGraphGetNodes(graph,nullptr,&nodes),"query retained graph nodes");require(nodes==1,"attention graph must contain exactly one kernel node");
  cudaGraphNode_t node{};size_t count=1;check(cudaGraphGetNodes(graph,&node,&count),"read retained graph node");cudaGraphNodeType type{};
  check(cudaGraphNodeGetType(node,&type),"retained graph node type");require(type==cudaGraphNodeTypeKernel,"retained graph is not one kernel");
  status=cudaGraphInstantiate(&exec,graph,0);if(exec)++state.execs_created;check(status,"instantiate retained attention graph");
 }
 void close() noexcept {
  if(capturing){cudaGraph_t recovered=nullptr;auto status=cudaStreamEndCapture(state.stream,&recovered);capturing=false;if(status!=cudaSuccess)++state.cleanup_errors;
   if(recovered){++state.graphs_created;if(cudaGraphDestroy(recovered)==cudaSuccess)++state.graphs_destroyed;else ++state.cleanup_errors;}}
  if(exec){if(cudaGraphExecDestroy(exec)==cudaSuccess)++state.execs_destroyed;else ++state.cleanup_errors;exec=nullptr;}
  if(graph){if(cudaGraphDestroy(graph)==cudaSuccess)++state.graphs_destroyed;else ++state.cleanup_errors;graph=nullptr;}
 }
 ~RetainedGraph(){close();}
};
size_t graph_transitions(State& state){
 // The same five candidate parents and same graph executable survive all3
 // fresh packets. H2D refresh is outside capture; no node parameters change.
 const std::array<Case,3> cases{{{0,0,0,0,2,0},{1,15,17,1,3,2},{2,29,5,2,2,1}}};
 auto first=fixture(cases[0]);Buffer q(state),k(state),v(state),packet(state),candidate(state);
 q.create(first.q);k.create(first.k);v.create(first.v);packet.create(first.packet);candidate.create(poison(4*kOutput,0x7fc7));
 std::vector<std::unique_ptr<Buffer>> oq,om,oy;
 for(int r=0;r<4;++r){oq.emplace_back(new Buffer(state));om.emplace_back(new Buffer(state));oy.emplace_back(new Buffer(state));
  oq.back()->create(poison(kOutput,0x7fc5));om.back()->create(std::vector<uint8_t>(80,0xff));oy.back()->create(poison(kOutput,0x7fc9));}
 check(cudaStreamSynchronize(state.stream),"complete fixed graph-parent allocation");
 RetainedGraph retained(state);retained.capture(q,k,v,candidate,packet);
 size_t failed=0;
 for(size_t step=0;step<cases.size();++step){const auto& c=cases[step];auto f=fixture(c);int active=active_rows[c.mode];
  q.upload(f.q);k.upload(f.k);v.upload(f.v);packet.upload(f.packet);candidate.upload(poison(4*kOutput,0x7fc7));
  for(int r=0;r<4;++r){oq[r]->upload(std::vector<uint8_t>(f.q.begin()+r*kOutput,f.q.begin()+(r+1)*kOutput));
   om[r]->upload(std::vector<uint8_t>(f.packet.begin()+128+r*128,f.packet.begin()+208+r*128));oy[r]->upload(poison(kOutput,0x7fc9));}
  check(cudaStreamSynchronize(state.stream),"fresh packet upload completion");
  std::vector<std::vector<uint8_t>> before;
  for(int r=0;r<4;++r){if(r<active){check(riley_cuda_internal::enqueue_compiled_packed_decode_attention_two_warp(state.stream,oq[r]->data(),k.data(),v.data(),oy[r]->data(),om[r]->data()),"transition M1 oracle");check(cudaStreamSynchronize(state.stream),"transition oracle completion");}before.push_back(oy[r]->download());}
  check(cudaGraphLaunch(retained.exec,state.stream),"replay retained bucket4 graph");check(cudaStreamSynchronize(state.stream),"transition graph completion");auto result=candidate.download();
  auto qa=q.download(),ka=k.download(),va=v.download(),pa=packet.download();
  bool inputs=qa==q.expected&&pa==packet.expected,kv=ka==k.expected&&va==v.expected,oracles=true;
  bool guards=q.guards(qa)&&k.guards(ka)&&v.guards(va)&&packet.guards(pa)&&candidate.guards(result),allfinite=finite(result,kGuard,4*kOutput);
  size_t mismatches=0,first_row=0,first_word=0;uint16_t first_a=0,first_b=0;
  for(int r=0;r<4;++r){auto xa=oq[r]->download(),ma=om[r]->download(),ya=oy[r]->download();inputs=inputs&&xa==oq[r]->expected&&ma==om[r]->expected;oracles=oracles&&ya==before[r];
   guards=guards&&oq[r]->guards(xa)&&om[r]->guards(ma)&&oy[r]->guards(ya)&&oy[r]->guards(before[r]);
   if(r<active){allfinite=allfinite&&finite(before[r],kGuard,kOutput);for(size_t i=0;i<576;++i){auto a=word(before[r].data()+kGuard+2*i),b=word(result.data()+kGuard+r*kOutput+2*i);
    if(a!=b){if(mismatches==0){first_row=r;first_word=i;first_a=a;first_b=b;}++mismatches;}}}}
  bool inactive=std::all_of(result.begin()+kGuard+active*kOutput,result.begin()+kGuard+4*kOutput,[](uint8_t x){return x==0;});
  bool exact=mismatches==0,passed=exact&&allfinite&&guards&&inputs&&kv&&oracles&&inactive;if(!passed)++failed;
  std::cout<<"{\"schema\":\"riley.multisequence-attention-native.v1\",\"kind\":\"graph_transition\",\"step\":"<<step<<",\"layer\":"<<c.layer<<",\"position_tuple\":"<<c.tuple
   <<",\"pattern\":"<<quote(patterns[c.pattern])<<",\"mapping\":"<<quote(mappings[c.mapping])<<",\"bucket\":4,\"active_rows\":"<<active<<",\"positions\":[";
  for(int r=0;r<active;++r){if(r)std::cout<<',';std::cout<<position(c,r);}std::cout<<"],\"graph_nodes\":"<<retained.nodes<<",\"capture_count\":1,\"parameter_updates\":0,\"replay_index\":"<<step+1
   <<",\"oracle_invocations\":"<<active<<",\"active_words_compared\":"<<active*576<<",\"inactive_words_checked\":"<<(4-active)*576<<",\"passed\":"<<(passed?"true":"false")
   <<",\"exact_outputs\":"<<(exact?"true":"false")<<",\"finite_outputs\":"<<(allfinite?"true":"false")<<",\"guards_intact\":"<<(guards?"true":"false")
   <<",\"inputs_unchanged\":"<<(inputs?"true":"false")<<",\"kv_unchanged\":"<<(kv?"true":"false")<<",\"oracle_outputs_unchanged\":"<<(oracles?"true":"false")<<",\"inactive_output_zero\":"<<(inactive?"true":"false")<<",\"mismatch_words\":"<<mismatches;
  if(mismatches)std::cout<<",\"first_mismatch\":{\"row\":"<<first_row<<",\"word\":"<<first_word<<",\"oracle_bits\":"<<first_a<<",\"candidate_bits\":"<<first_b<<"}";
  std::cout<<"}\n"<<std::flush;
 }
 retained.close(); // Graph/executable must release their references first.
 for(int r=0;r<4;++r){oq[r]->close();om[r]->close();oy[r]->close();}
 q.close();k.close();v.close();packet.close();candidate.close();return failed;
}
} // namespace
int main(int argc,char** argv){
 State state;size_t completed=0,failed=0,transition_count=0,transition_failed=0;std::string error;std::vector<std::string> maps;
 try{
  require(argc==5&&std::string(argv[1])=="--cases"&&std::string(argv[3])=="--device","usage: probe --cases cases.tsv --device 0");
  auto cases=read_cases(argv[2]);require(integer(argv[4])==0,"device0 required");check(cudaSetDevice(0),"select CUDA device");
  cudaDeviceProp properties{};check(cudaGetDeviceProperties(&properties,0),"device properties");require(properties.major==8&&properties.minor==9,"SM89 required");
  int driver=0,runtime=0;check(cudaDriverGetVersion(&driver),"driver version");check(cudaRuntimeGetVersion(&runtime),"runtime version");require(runtime==13000,"CUDA13.0 runtime required");
  std::ostringstream uuid;uuid<<std::hex;for(unsigned char b:properties.uuid.bytes){uuid.width(2);uuid.fill('0');uuid<<unsigned(b);}require(uuid.str()=="9087e4256acab722b8c9cc0423b39fb0","GPU UUID differs");
  maps=runtime_maps();std::cout<<"{\"schema\":\"riley.multisequence-attention-native.v1\",\"kind\":\"device\",\"uuid_hex\":"<<quote(uuid.str())<<",\"name\":"<<quote(properties.name)<<",\"compute_major\":8,\"compute_minor\":9,\"runtime_version\":"<<runtime<<",\"driver_version\":"<<driver<<",\"runtime_maps\":"<<strings(maps)<<"}\n"<<std::flush;
  check(cudaStreamCreateWithFlags(&state.stream,cudaStreamNonBlocking),"create stream");std::vector<uint8_t> reference;
  for(const auto& c:cases){if(!run_case(state,c,reference))++failed;++completed;}
  transition_failed=graph_transitions(state);transition_count=3;check(cudaStreamSynchronize(state.stream),"final completion");maps=runtime_maps();
 }catch(const std::exception& e){error=e.what();}
 state.close_stream();bool complete=error.empty()&&completed==kCases&&state.created==state.freed&&state.live_bytes==0&&state.cleanup_errors==0&&state.stream_destroyed&&transition_count==3&&state.graphs_created==1&&state.graphs_destroyed==1&&state.execs_created==1&&state.execs_destroyed==1;
 std::cout<<"{\"schema\":\"riley.multisequence-attention-native.v1\",\"kind\":\"summary\",\"completed\":"<<(complete?"true":"false")<<",\"attention_bitwise_equal\":"<<(complete&&failed==0&&transition_failed==0?"true":"false")
  <<",\"cases\":"<<completed<<",\"failed_cases\":"<<failed<<",\"layers\":30,\"position_tuples\":32,\"patterns\":3,\"mappings\":3,\"modes\":4"
  <<",\"graph_transitions\":"<<transition_count<<",\"failed_graph_transitions\":"<<transition_failed<<",\"graphs_created\":"<<state.graphs_created<<",\"graphs_destroyed\":"<<state.graphs_destroyed<<",\"execs_created\":"<<state.execs_created<<",\"execs_destroyed\":"<<state.execs_destroyed
  <<",\"allocations_created\":"<<state.created<<",\"allocations_freed\":"<<state.freed<<",\"live_bytes\":"<<state.live_bytes<<",\"cleanup_errors\":"<<state.cleanup_errors<<",\"stream_destroyed\":"<<(state.stream_destroyed?"true":"false")
  <<",\"runtime_maps\":"<<strings(maps)<<",\"performance_claim\":false,\"error\":"<<quote(error)<<"}\n"<<std::flush;
 return complete?0:1;
}
