// Strictly extracted shared CPU fixture/guard helpers; no main or CUDA kernels.
namespace {
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
}
