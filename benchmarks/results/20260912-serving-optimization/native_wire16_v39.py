from pathlib import Path
root=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=root/'kernels/src/prefill_shape_packet.hpp';s=p.read_text()
s=s.replace('inline bool valid_prefill_shape_packet(', 'template<uint32_t Rows>\ninline bool valid_variable_shape_packet(')
s=s.replace(' constexpr uint32_t header=128,stride=1664,tokens=13440,extent=17536;', ''' static_assert(Rows==8||Rows==16,"unsupported wire capacity");
 constexpr uint32_t header=128,stride=1664,tokens=header+Rows*stride,extent=tokens+4096;
 constexpr uint32_t magic=Rows==8?0x33444d52:0x34444d52,version=Rows==8?3:4;''')
s=s.replace('capacity==4||capacity==8)', 'capacity==4||capacity==8||(Rows==16&&capacity==16))')
s=s.replace('u32(0)!=0x33444d52||u32(4)!=3','u32(0)!=magic||u32(4)!=version')
s+='''
inline bool valid_prefill_shape_packet(const uint8_t* p,uint64_t bytes,uint32_t physical,uint32_t capacity) noexcept {
 return valid_variable_shape_packet<8>(p,bytes,physical,capacity);
}
inline bool valid_v4_shape_packet(const uint8_t* p,uint64_t bytes,uint32_t physical,uint32_t capacity) noexcept {
 return valid_variable_shape_packet<16>(p,bytes,physical,capacity);
}
''';p.write_text(s)
p=root/'crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs';s=p.read_text();pos=s.index('    #[test] fn sixteen_row_contract')
s=s[:pos]+'''    #[test] fn export_versioned_wire_fixtures_when_requested() {
        let Some(dir)=std::env::var_os("RILEY_V39_WIRE_FIXTURES") else {return;};
        let dir=std::path::PathBuf::from(dir);std::fs::create_dir_all(&dir).unwrap();
        for active in [1,2,4,8,9,15,16] {
            let e=fixture_rows::<16>(InputStage::Decode,active);let mut packet=vec![0;Layout::<16>::REQUEST_BYTES];encode_into(&mut packet,&e).unwrap();
            std::fs::write(dir.join(format!("v4-decode-{active}.bin")),packet).unwrap();
        }
        let e=fixture_rows::<16>(InputStage::Prefill,1);let mut packet=vec![0;Layout::<16>::REQUEST_BYTES];encode_into(&mut packet,&e).unwrap();
        std::fs::write(dir.join("v4-prefill.bin"),packet).unwrap();
        for(stage,name)in[(InputStage::Decode,"v3-decode.bin"),(InputStage::Prefill,"v3-prefill.bin")] {
            let e=fixture(stage,if stage==InputStage::Decode{8}else{1});let mut packet=vec![0;REQUEST_BYTES];encode_into(&mut packet,&e).unwrap();
            std::fs::write(dir.join(name),packet).unwrap();
        }
    }
'''+s[pos:];p.write_text(s)
(root/'kernels/tests/variable_wire16_packet_test.cpp').write_text('''#include "prefill_shape_packet.hpp"
#include <cassert>
#include <fstream>
#include <iterator>
#include <vector>
#include <iostream>
int main(int argc,char**argv){
 assert(argc==11);unsigned accepted=0,rejected=0;
 for(int arg=1;arg<argc;++arg){
  std::ifstream f(argv[arg],std::ios::binary);std::vector<uint8_t> p((std::istreambuf_iterator<char>(f)),{});
  bool wide=p.size()==30848;auto valid=[&](const std::vector<uint8_t>&q){return wide?valid_v4_shape_packet(q.data(),q.size(),4096,16):valid_prefill_shape_packet(q.data(),q.size(),4096,8);};
  assert(valid(p));++accepted;
  assert(!(wide?valid_prefill_shape_packet(p.data(),p.size(),4096,8):valid_v4_shape_packet(p.data(),p.size(),4096,16)));++rejected;
  for(unsigned at:{0,4,8,12,24,28,36,96,128+64}) {auto q=p;q.at(at)^=0x80;assert(!valid(q));++rejected;}
  auto set=[](std::vector<uint8_t>&q,unsigned at,uint32_t v){std::memcpy(q.data()+at,&v,4);};
  auto get=[&](unsigned at){uint32_t v;std::memcpy(&v,p.data()+at,4);return v;};
  unsigned active=get(20),rows=wide?16:8,tokens=128+rows*1664;
  {auto q=p;set(q,20,rows+1);assert(!valid(q));++rejected;}
  {auto q=p;set(q,0,wide?0x33444d52:0x34444d52);set(q,4,wide?3:4);assert(!valid(q));++rejected;}
  if(active>1){auto q=p;set(q,128+(active-1)*1664+128,get(256));assert(!valid(q));++rejected;}
  for(unsigned at=128+active*1664;at<tokens;++at){auto q=p;q[at]=1;assert(!valid(q));++rejected;}
  {auto q=p;set(q,tokens,49152);assert(!valid(q));++rejected;}
  {auto q=p;q.pop_back();assert(!valid(q));++rejected;}
  {auto q=p;q.push_back(0);assert(!valid(q));++rejected;}
  // Every byte mutation is memory safe; some semantic changes require live Rust authority.
  for(size_t at=0;at<p.size();++at){p[at]^=0xff;(void)valid(p);p[at]^=0xff;}
 }
 std::cout<<"accepted="<<accepted<<" rejected="<<rejected<<"\\n";
}
''')
