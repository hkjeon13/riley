// Strided M1 arithmetic: arm0 compact, arm1 padded. Fresh child descriptors.
// Correctness-only host harness. Links the frozen production native archive;
// Arm 0 uses compact strides; arm 1 uses padded strides. No timing or fallback.
#include "gemm.cu"
#include <cuda.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t GUARD = 256, STAGING = 1 << 20;
constexpr std::array<size_t,2> WORKSPACE_CAPS{{0,16u*1024u*1024u}};
constexpr const char* SCHEMA = "riley.multisequence-strided-owned-plan-native.v1";
using Bytes = std::vector<uint8_t>;
std::string json_quote(const std::string& s) {
  std::ostringstream o; o << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') o << '\\' << c;
    else if (c < 32) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
    else o << c;
  }
  o << '"'; return o.str();
}
const char* boolean(bool value) { return value ? "true" : "false"; }
void require(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
RileyCudaErrorInfo err() { RileyCudaErrorInfo e{}; e.struct_size = sizeof(e); return e; }
void checked(RileyCudaStatus status, const RileyCudaErrorInfo& e, const char* site) {
  if (status) throw std::runtime_error(std::string(site) + ": status=" + std::to_string(status) + " " + e.message);
}
#define CALL(expression) do { auto error = err(); auto status = (expression); checked(status, error, #expression); } while (false)
void record(const std::string& kind, const std::string& fields) {
  std::cout << "{\"schema_version\":" << json_quote(SCHEMA) << ",\"kind\":" << json_quote(kind) << ',' << fields << "}\n" << std::flush;
}
Bytes read_bytes(const std::string& path, size_t expected) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  require(bool(f) && f.tellg() == std::streamoff(expected), "fixture length differs: " + path);
  f.seekg(0); Bytes out(expected); f.read(reinterpret_cast<char*>(out.data()), expected);
  require(bool(f), "fixture read failed: " + path); return out;
}
uint16_t word(const Bytes& b, size_t index) { return uint16_t(b[2*index]) | (uint16_t(b[2*index+1]) << 8); }
bool finite(const Bytes& b) {
  for (size_t i = 0; i < b.size()/2; ++i) if ((word(b, i) & 0x7f80) == 0x7f80) return false;
  return true;
}
struct Shape { const char* name; size_t n, k; unsigned custom; };
constexpr std::array<Shape, 5> SHAPES{{{"qkv",960,576,74}, {"gate_up",3072,576,74},
  {"o",576,576,74}, {"down",576,1536,75}, {"head",49152,576,89}}};
struct Case { size_t id, layer, shape, m, active, n, k, pattern, arm; std::string x, w; };
std::vector<Case> cases(const std::string& path) {
  std::ifstream f(path); require(bool(f), "cannot read cases"); std::vector<Case> result;
  std::string line;
  while (std::getline(f, line)) {
    std::vector<std::string> fields; std::istringstream row(line); std::string part;
    while (std::getline(row, part, '\t')) fields.push_back(part);
    require(fields.size() == 11, "case column count differs");
    Case c{}; size_t* numbers[] = {&c.id,&c.layer,&c.shape,&c.m,&c.active,&c.n,&c.k,&c.pattern,&c.arm};
    for (size_t i=0;i<9;++i) { size_t used=0; *numbers[i]=std::stoull(fields[i],&used); require(used==fields[i].size(),"invalid integer"); }
    c.x=fields[9]; c.w=fields[10];
    size_t group=result.size()/18, variant=result.size()%18;
    size_t expected_shape=group<120 ? group%4 : 4, expected_layer=group<120 ? group/4 : 30;
    size_t mode=(variant/2)%3;
    require(c.id==result.size() && group<121 && c.layer==expected_layer && c.shape==expected_shape
      && c.m==(mode==0 ? 2u : 4u) && c.active==(mode==0 ? 2u : mode==1 ? 4u : 3u)
      && c.pattern==variant/6 && c.arm==variant%2 && c.n==SHAPES[c.shape].n && c.k==SHAPES[c.shape].k, "case coverage/order differs");
    result.push_back(c);
  }
  require(result.size()==2178,"expected 2178 cases"); return result;
}
struct Device { size_t slot, bytes; };
struct Session {
  RileyCudaContext* context=nullptr; RileyCudaStream* stream=nullptr;
  RileyCudaPinnedHostBuffer* staging=nullptr; RileyCudaCopy* pending=nullptr;
  std::vector<RileyCudaDeviceBuffer*> devices;
  std::array<std::array<RileyCudaGemmPlan*,5>,5> plans{};
  std::array<std::array<RileyCudaGemmAlgorithmInfo,5>,5> infos{};
  bool cleanup_attempted=false;
  RileyCudaAllocationStats stats() {
    RileyCudaAllocationStats result{}; result.struct_size=sizeof(result);
    CALL(riley_cuda_context_allocation_stats(context,&result,&error)); return result;
  }
  void copy(Device d, Bytes& b, bool upload) {
    require(b.size()==d.bytes+2*GUARD,"copy size differs");
    for (size_t offset=0;offset<b.size();offset+=STAGING) {
      size_t count=std::min(STAGING,b.size()-offset);
      if (upload) {
        CALL(riley_cuda_pinned_host_buffer_write(staging,0,b.data()+offset,count,&error));
        CALL(riley_cuda_copy_h2d_async(devices.at(d.slot),offset,staging,0,count,stream,&pending,&error));
      } else CALL(riley_cuda_copy_d2h_async(staging,0,devices.at(d.slot),offset,count,stream,&pending,&error));
      CALL(riley_cuda_copy_close(&pending,&error));
      if (!upload) CALL(riley_cuda_pinned_host_buffer_read(staging,0,b.data()+offset,count,&error));
    }
  }
  Device allocate(const Bytes& raw, bool poison=false) {
    // Insert before create: even ambiguous rollback retains a visible handle.
    devices.push_back(nullptr); Device d{devices.size()-1,raw.size()};
    CALL(riley_cuda_device_buffer_create(context,d.bytes+2*GUARD,&devices.back(),&error));
    Bytes b(d.bytes+2*GUARD,0xa5);
    if (poison) for(size_t i=0;i<d.bytes;i+=2) { b[GUARD+i]=0xc1; b[GUARD+i+1]=0x7f; }
    else std::copy(raw.begin(),raw.end(),b.begin()+GUARD);
    copy(d,b,true); return d;
  }
  Bytes download(Device d, bool& guards) {
    Bytes b(d.bytes+2*GUARD); copy(d,b,false);
    guards=guards && std::all_of(b.begin(),b.begin()+GUARD,[](uint8_t x){return x==0xa5;})
      && std::all_of(b.end()-GUARD,b.end(),[](uint8_t x){return x==0xa5;});
    return Bytes(b.begin()+GUARD,b.end()-GUARD);
  }
  RileyCudaBufferSpan span(Device d, RileyCudaDType dtype=RILEY_CUDA_DTYPE_BF16) {
    RileyCudaBufferSpan s{}; s.struct_size=sizeof(s); s.dtype=dtype;
    s.buffer=devices.at(d.slot); s.byte_offset=GUARD; s.byte_len=d.bytes; return s;
  }
  void execute(RileyCudaGemmPlan* plan, Device x, Device w, Device y, Device workspace) {
    auto a=span(x), b=span(w), c=span(y), scratch=span(workspace,RILEY_CUDA_DTYPE_U8);
    // The ABI requires a real span and buffer even for an exact zero-byte payload.
    CALL(riley_cuda_gemm_plan_execute(plan,&a,&b,&c,&scratch,stream,&error));
  }
  void close_case() {
    for (auto it=devices.rbegin();it!=devices.rend();++it) CALL(riley_cuda_device_buffer_close(&*it,&error));
    devices.clear(); auto a=stats();
    require(a.device_live_allocations==0 && a.device_live_bytes==0 && a.pinned_host_live_allocations==1
      && a.pinned_host_live_bytes==STAGING,"case allocation accounting differs");
  }
  bool close() noexcept {
    if (cleanup_attempted) return false; cleanup_attempted=true; bool ok=true;
    auto attempt=[&](RileyCudaStatus status){ if(status) ok=false; };
    auto e=err();
    if(pending) attempt(riley_cuda_copy_close(&pending,&e));
    for(auto it=plans.rbegin();it!=plans.rend();++it)
      for(auto p=it->rbegin();p!=it->rend();++p) if(*p) attempt(riley_cuda_gemm_plan_close(&*p,&e));
    for(auto it=devices.rbegin();it!=devices.rend();++it) if(*it) attempt(riley_cuda_device_buffer_close(&*it,&e));
    if(staging) attempt(riley_cuda_pinned_host_buffer_close(&staging,&e));
    if(stream) attempt(riley_cuda_stream_close(&stream,&e));
    if(context) {
      RileyCudaAllocationStats s{}; s.struct_size=sizeof(s);
      attempt(riley_cuda_context_allocation_stats(context,&s,&e));
      ok=ok && s.device_live_allocations==0 && s.device_live_bytes==0
        && s.pinned_host_live_allocations==0 && s.pinned_host_live_bytes==0;
      attempt(riley_cuda_context_close(&context,&e));
    }
    return ok && !context && !stream && !staging && !pending;
  }
  ~Session(){ if(!cleanup_attempted) close(); }
};
bool same_stats(const RileyCudaAllocationStats& a,const RileyCudaAllocationStats& b) {
  return a.device_live_bytes==b.device_live_bytes && a.device_live_allocations==b.device_live_allocations
    && a.pinned_host_live_bytes==b.pinned_host_live_bytes && a.pinned_host_live_allocations==b.pinned_host_live_allocations;
}
RileyCudaGemmConfig config(size_t m,const Shape& shape,size_t cap) {
  RileyCudaGemmConfig c{}; c.struct_size=sizeof(c); c.m=m;c.n=shape.n;c.k=shape.k;
  c.input_dtype=c.weight_dtype=c.output_dtype=RILEY_CUDA_DTYPE_BF16; c.accumulator_dtype=RILEY_CUDA_DTYPE_F32;
  c.input_transpose=RILEY_CUDA_GEMM_TRANSPOSE_N;c.weight_transpose=RILEY_CUDA_GEMM_TRANSPOSE_T;
  c.input_layout=c.weight_layout=c.output_layout=RILEY_CUDA_GEMM_LAYOUT_ROW_MAJOR;
  c.deterministic=RILEY_CUDA_GEMM_DETERMINISTIC_REQUIRED;c.max_workspace_bytes=cap;c.flags=0;
  return c;
}
std::string metadata(const RileyCudaGemmAlgorithmInfo& a) {
  std::ostringstream o; o<<"{";
#define FIELD(name) o << "\"" #name "\":" << a.name << ','
  FIELD(struct_size); FIELD(backend); FIELD(algorithm_id); FIELD(tile_id); FIELD(stages_id); FIELD(split_k);
  FIELD(reduction_scheme); FIELD(cta_swizzling); FIELD(custom_option); FIELD(deterministic); FIELD(workspace_bytes);
  FIELD(numerical_implementation_flags); FIELD(compute_capability_major); FIELD(compute_capability_minor);
  FIELD(runtime_version); FIELD(cublaslt_version); FIELD(m); FIELD(n); FIELD(k);
#undef FIELD
  o<<"\"reserved\":["<<a.reserved[0]<<','<<a.reserved[1]<<"]}";return o.str();
}
bool qualified(const RileyCudaGemmAlgorithmInfo& a,size_t m,const Shape& s) {
  return a.struct_size==sizeof(a) && a.backend==1 && a.algorithm_id==13 && a.tile_id==0 && a.stages_id==0
    && a.split_k==1 && a.reduction_scheme==0 && a.cta_swizzling==0 && a.custom_option==s.custom
    && a.deterministic==1 && a.workspace_bytes==0 && a.numerical_implementation_flags==131585
    && a.compute_capability_major==8 && a.compute_capability_minor==9 && a.runtime_version==13000
    && a.cublaslt_version==130101 && a.m==m && a.n==s.n && a.k==s.k && !a.reserved[0] && !a.reserved[1];
}
bool strict_candidate(const RileyCudaGemmAlgorithmInfo& a,size_t m,const Shape& s,size_t cap) {
  return a.struct_size==sizeof(a) && a.backend==1 && a.algorithm_id>=0 && a.split_k<=1 && a.reduction_scheme==0
    && a.deterministic==1 && a.workspace_bytes<=cap && a.compute_capability_major==8 && a.compute_capability_minor==9
    && a.runtime_version==13000 && a.cublaslt_version==130101 && a.m==m && a.n==s.n && a.k==s.k
    && !a.reserved[0] && !a.reserved[1];
}
bool prepare_plans(Session& session) {
 auto base=config(1,SHAPES[0],16u*1024u*1024u);auto before=session.stats();
 for(unsigned mode=0;mode<8;++mode){auto c=base;uint32_t batch=2;uint64_t xs=c.k,ys=c.n;
  if(mode<4)batch=std::array<uint32_t,4>{0,1,3,8}[mode];
  if(mode==4)xs=c.k+1;if(mode==5)ys=c.n+1;if(mode==6)c.m=2;if(mode==7)c.flags=1;
  auto e=err();auto* rejected=reinterpret_cast<RileyCudaGemmPlan*>(uintptr_t(1));
  auto status=riley_cuda_gemm_plan_create_strided_m1(session.context,&c,batch,xs,ys,&rejected,&e);
  require(status==RILEY_CUDA_STATUS_INVALID_ARGUMENT && rejected==nullptr,"invalid strided config not rejected");
  require(same_stats(before,session.stats()),"rejection changed allocation stats");
 }
 record("rejections","\"cases\":8,\"all_rejected_before_publication\":true");
 for(size_t shape=0;shape<5;++shape)for(size_t index=0;index<5;++index){
  auto c=config(1,SHAPES[shape],16u*1024u*1024u);auto& plan=session.plans[shape][index];
  if(index==0){CALL(riley_cuda_gemm_plan_create(session.context,&c,&plan,&error));}
  else {size_t batch=(index%2)==1 ? 2:4;bool padded=index>=3;
   size_t xs=padded ? ((c.k+127)/128)*128:c.k,ys=padded ? ((c.n+127)/128)*128:c.n;
   CALL(riley_cuda_gemm_plan_create_strided_m1(session.context,&c,batch,xs,ys,&plan,&error));}
  auto& info=session.infos[shape][index];info.struct_size=sizeof(info);CALL(riley_cuda_gemm_plan_info(plan,&info,&error));
  require(qualified(info,1,SHAPES[shape]),"M1 anchor differs");
  require(riley_cuda_internal::canonical_gemm_bf16_plan_is_ready(plan)==(index==0),"legacy graph admission did not separate strided plan");
  record("plan","\"shape\":"+std::to_string(shape)+",\"index\":"+std::to_string(index)+",\"metadata\":"+metadata(info));
 }
 return true;
}
bool run_case(Session& s,const Case& c) {
  size_t plan_index=1+2*c.arm+(c.m==4);auto plan=s.plans[c.shape][plan_index];
  std::string prefix="\"case_id\":"+std::to_string(c.id)+",\"layer\":"+std::to_string(c.layer)
    +",\"shape\":"+std::to_string(c.shape)+",\"m\":"+std::to_string(c.m)+",\"active_rows\":"+std::to_string(c.active)
    +",\"n\":"+std::to_string(c.n)+",\"k\":"+std::to_string(c.k)+",\"pattern\":"+std::to_string(c.pattern)
    +",\"arm\":"+std::to_string(c.arm)+",\"historical_fixture_workspace_cap\":"+std::to_string(WORKSPACE_CAPS[c.arm]);
  if(!plan) { record("case",prefix+",\"executed\":false,\"reason\":\"strict_descriptor_not_supported\"");return false; }
  Bytes x=read_bytes(c.x,c.m*c.k*2), w=read_bytes(c.w,c.n*c.k*2);
  require(finite(x)&&finite(w),"nonfinite fixture");
  for(size_t i=c.active*c.k*2;i<x.size();++i) require(x[i]==0,"inactive row is not raw positive zero");
  size_t xs=c.arm ? ((c.k+127)/128)*128:c.k,ys=c.arm ? ((c.n+127)/128)*128:c.n;
  Bytes upload(c.m*xs*2,0x5a),out_init(c.m*ys*2,0x5a);
  for(size_t row=0;row<c.m;++row)std::copy(x.begin()+row*c.k*2,x.begin()+(row+1)*c.k*2,upload.begin()+row*xs*2);
  Device dx=s.allocate(upload),dw=s.allocate(w),dy=s.allocate(out_init);
  size_t workspace_bytes=size_t(s.infos[c.shape][plan_index].workspace_bytes);
  Device workspace=s.allocate(Bytes(workspace_bytes)),oracle_workspace=s.allocate(Bytes());
  std::vector<Bytes> rows; std::vector<Device> rx,ry;
  for(size_t row=0;row<c.m;++row) {
    rows.emplace_back(x.begin()+row*c.k*2,x.begin()+(row+1)*c.k*2);
    // Independent allocation per oracle row: no misaligned interior row view.
    rx.push_back(s.allocate(rows.back()));ry.push_back(s.allocate(Bytes(c.n*2),true));
  }
  require(s.devices.size()==5+2*c.m,"case allocation inventory differs");
  auto before=s.stats();s.execute(plan,dx,dw,dy,workspace);
  for(size_t row=0;row<c.m;++row)s.execute(s.plans[c.shape][0],rx[row],dw,ry[row],oracle_workspace);
  bool stable=same_stats(before,s.stats()),guards=true;
  bool workspace_guards=true;s.download(workspace,workspace_guards);s.download(oracle_workspace,workspace_guards);
  guards=guards&&workspace_guards;
  RileyCudaGemmAlgorithmInfo candidate_after{},oracle_after{};candidate_after.struct_size=sizeof(candidate_after);oracle_after.struct_size=sizeof(oracle_after);
  CALL(riley_cuda_gemm_plan_info(plan,&candidate_after,&error));CALL(riley_cuda_gemm_plan_info(s.plans[c.shape][0],&oracle_after,&error));
  bool metadata_unchanged=metadata(candidate_after)==metadata(s.infos[c.shape][plan_index])&&metadata(oracle_after)==metadata(s.infos[c.shape][0]);
  auto padded=s.download(dy,guards);Bytes got(c.m*c.n*2);
  for(size_t row=0;row<c.m;++row){std::copy(padded.begin()+row*ys*2,padded.begin()+(row*ys+c.n)*2,got.begin()+row*c.n*2);
    for(size_t j=c.n*2;j<ys*2;++j)guards=guards && padded[row*ys*2+j]==0x5a;}
  bool inputs=s.download(dx,guards)==upload,weights=s.download(dw,guards)==w;
  bool allfinite=finite(got);size_t total=0;std::ostringstream summaries;summaries<<'[';
  for(size_t row=0;row<c.m;++row) {
    auto expected=s.download(ry[row],guards);inputs=inputs && s.download(rx[row],guards)==rows[row];
    allfinite=allfinite&&finite(expected);size_t mismatches=0,first=c.n;
    for(size_t j=0;j<c.n;++j)if(word(expected,j)!=word(got,row*c.n+j)){++mismatches;if(first==c.n)first=j;}
    total+=mismatches;
    if(row)summaries<<',';
    summaries<<"{\"row\":"<<row<<",\"words\":"<<c.n<<",\"mismatches\":"<<mismatches<<",\"first_mismatch\":";
    if(first==c.n)summaries<<"null,\"oracle_bits\":null,\"batched_bits\":null}";
    else summaries<<first<<",\"oracle_bits\":"<<word(expected,first)<<",\"batched_bits\":"<<word(got,row*c.n+first)<<'}';
  }
  summaries<<']';s.close_case();
  bool exact=!total && allfinite && inputs && weights && guards && stable && metadata_unchanged;
  record("case",prefix+",\"executed\":true,\"words\":"+std::to_string(c.m*c.n)+",\"mismatches\":"+std::to_string(total)
    +",\"exact_outputs\":"+boolean(!total)+",\"finite_outputs\":"+boolean(allfinite)+",\"guards_intact\":"+boolean(guards)
    +",\"inputs_unchanged\":"+boolean(inputs)+",\"weights_unchanged\":"+boolean(weights)
    +",\"allocation_stats_unchanged_across_execute\":"+boolean(stable)+",\"case_allocations_freed\":true,\"rows\":"+summaries.str()
    +",\"workspace_bytes\":"+std::to_string(workspace_bytes)+",\"oracle_workspace_bytes\":0,\"workspace_parent_bytes\":"+std::to_string(workspace_bytes+2*GUARD)
    +",\"oracle_workspace_parent_bytes\":512,\"workspace_span_dtype\":\"U8\",\"workspace_span_offset\":256,\"candidate_workspace_span_passed\":true,\"oracle_workspace_span_passed\":true"
    +",\"case_device_allocations\":"+std::to_string(5+2*c.m)+",\"workspace_guards_intact\":"+boolean(workspace_guards)+",\"plan_metadata_unchanged\":"+boolean(metadata_unchanged));
  require(inputs&&weights&&guards&&stable&&metadata_unchanged,"memory integrity/accounting or plan metadata failed");return exact;
}
std::string maps() {
  std::ifstream f("/proc/self/maps");require(bool(f),"cannot inspect loaded runtime maps");
  std::ostringstream out;out<<f.rdbuf();return out.str();
}
} // namespace

int main(int argc,char** argv) {
  Session session;size_t processed=0;bool exact=true,admitted=false;std::string failure;
  try {
    require(argc==5 && std::string(argv[1])=="--cases" && std::string(argv[3])=="--device" && std::string(argv[4])=="0","usage: --cases TSV --device 0");
    auto rows=cases(argv[2]);require(riley_cuda_abi_version()==1,"ABI differs");
    CALL(riley_cuda_context_create(0,&session.context,&error));CALL(riley_cuda_stream_create(session.context,&session.stream,&error));
    CALL(riley_cuda_pinned_host_buffer_create(session.context,STAGING,&session.staging,&error));
    RileyCudaDeviceProperties p{};p.struct_size=sizeof(p);CALL(riley_cuda_device_properties(0,&p,&error));
    CUuuid uuid{};require(cuDeviceGetUuid(&uuid,0)==CUDA_SUCCESS,"device UUID query failed");
    std::ostringstream id;for(unsigned char b:uuid.bytes)id<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(b);
    record("device","\"uuid\":"+json_quote(id.str())+",\"runtime_version\":"+std::to_string(p.runtime_version)
      +",\"compute_capability_major\":"+std::to_string(p.compute_capability_major)+",\"compute_capability_minor\":"+std::to_string(p.compute_capability_minor)
      +",\"abi\":1,\"native_build_info\":"+json_quote(riley_cuda_build_info())+",\"maps\":"+json_quote(maps()));
    require(id.str()=="9087e4256acab722b8c9cc0423b39fb0" && p.runtime_version==13000 && p.compute_capability_major==8 && p.compute_capability_minor==9
      && std::string(riley_cuda_build_info())=="riley-cuda-native abi=1 nvcc=13.3.73","device/runtime/native build differs");
    admitted=prepare_plans(session);
    for(const auto& row:rows){exact=run_case(session,row)&&exact;++processed;}
    record("runtime_end","\"maps\":"+json_quote(maps()));
  } catch(const std::exception& e) { failure=e.what(); }
  bool closed=session.close(),complete=failure.empty()&&processed==2178&&closed;
  record("summary","\"completed\":"+std::string(boolean(complete))+",\"cases\":"+std::to_string(processed)
    +",\"all_plans_admitted\":"+boolean(admitted)+",\"all_outputs_exact\":"+boolean(complete&&admitted&&exact)
    +",\"all_resources_closed\":"+boolean(closed)+",\"performance_measured\":false,\"performance_claim_eligible\":false,\"failure\":"+json_quote(failure));
  return complete ? 0 : 2;
}
