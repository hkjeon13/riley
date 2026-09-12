from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'kernels/src/prefill_shape_packet.hpp'
p.write_text('''#pragma once
#include <cstdint>
#include <cstring>
// Structural check only. Rust retained-owner authority must bind identities,
// complete page ownership, catalog and replay order before GPU dispatch.
inline bool valid_prefill_shape_packet(const uint8_t* p,uint64_t bytes,uint32_t physical,uint32_t capacity) noexcept {
 constexpr uint32_t header=128,stride=1664,tokens=13440,extent=17536;
 if(!p||bytes!=extent||!physical||physical>4096||!(capacity==1||capacity==2||capacity==4||capacity==8))return false;
 auto u32=[&](uint32_t at){uint32_t v;std::memcpy(&v,p+at,4);return v;};
 auto u64=[&](uint32_t at){uint64_t v;std::memcpy(&v,p+at,8);return v;};
 auto u16=[&](uint32_t at){uint16_t v;std::memcpy(&v,p+at,2);return v;};
 auto zero=[&](uint32_t a,uint32_t b){for(;a<b;++a)if(p[a])return false;return true;};
 uint32_t stage=u32(16),active=u32(20);
 if(u32(0)!=0x33444d52||u32(4)!=3||u32(8)!=extent||u32(12)!=stride||stage>1||!active||active>capacity||u32(24)!=capacity||u32(28)!=physical||u32(32)>1||!zero(36,40)||!u64(40)||!u64(48)||!u64(56)||zero(64,96)||!zero(96,128)||(stage==0&&active!=1))return false;
 bool used[4096]{};uint32_t slots=0,prefill_count=0;
 for(uint32_t row=0;row<active;++row){
  uint32_t b=header+row*stride,count=u32(b+8),live=u32(b+12),committed=u32(b+16),target=u32(b+20),generated=u32(b+24),limit=u32(b+28),prompt=u32(b+32),context=u32(b+36),slot=u32(b+40),logit=u32(b+44);
  if(!count||count>1024||!prompt||!limit||!context||context>4096||uint64_t(prompt)+limit-1>context||generated>=limit||uint64_t(committed)+count!=target||!target||target>context||u32(b+4)!=target-1||u32(b)>=49152||live!=(target+15)/16||slot>=active||(slots&(1U<<slot))||!u64(b+48)||!u64(b+56)||!zero(b+64,b+128))return false;
  slots|=1U<<slot;
  if(stage==0){if(generated||committed>=prompt||target>prompt||logit!=(target==prompt?count-1:UINT32_MAX))return false;prefill_count=count;}
  else if(count!=1||!generated||uint64_t(prompt)+generated-1!=committed||logit!=0)return false;
  for(uint32_t prior=0;prior<row;++prior)if(u64(b+48)==u64(header+prior*stride+48)||u64(b+56)==u64(header+prior*stride+56))return false;
  for(uint32_t i=0;i<256;++i){uint32_t id=u32(b+128+i*4);uint16_t valid=u16(b+1152+i*2);
   if(i>=live){if(id||valid)return false;continue;}
   if(id>=physical||used[id]||valid!=(i+1<live?16:(target-1)%16+1))return false;used[id]=true;
  }
 }
 if(!zero(header+active*stride,tokens))return false;
 for(uint32_t i=0;i<1024;++i){uint32_t t=u32(tokens+i*4);if(i<prefill_count){if(t>=49152||(i==0&&t!=u32(header)))return false;}else if(t)return false;}
 return true;
}
''')
# Export fixtures from the already exercised Rust canonical encoder for native parity.
p=r/'crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs';s=p.read_text();needle='encode_into(&mut p,&e).unwrap();validate_packet(&p,&mut scratch,&e).unwrap();';assert needle in s
s=s.replace(needle,needle+'''\n            if let Some(dir)=std::env::var_os("RILEY_V3_FIXTURES") {let dir=std::path::PathBuf::from(dir);std::fs::create_dir_all(&dir).unwrap();std::fs::write(dir.join(if e.stage==InputStage::Prefill{"prefill.bin"}else{"decode.bin"}),&p).unwrap();}''');p.write_text(s)
p=r/'kernels/tests/prefill_shape_packet_test.cpp';p.write_text('''#include "prefill_shape_packet.hpp"
#include <vector>
#include <fstream>
#include <iterator>
#include <cstdio>
#include <cassert>
int main(int argc,char**argv){assert(argc==3);int mutations=0;
 for(int n=1;n<argc;++n){std::ifstream f(argv[n],std::ios::binary);std::vector<uint8_t>p((std::istreambuf_iterator<char>(f)),{});assert(valid_prefill_shape_packet(p.data(),p.size(),4096,8));
  for(auto at:{0,4,8,12,24,28,36,96,128+64,128+1152}){auto q=p;q[at]^=0x80;assert(!valid_prefill_shape_packet(q.data(),q.size(),4096,8));++mutations;}
  assert(!valid_prefill_shape_packet(p.data(),p.size()-1,4096,8));assert(!valid_prefill_shape_packet(p.data(),p.size(),4095,8));mutations+=2;
 }
 printf("native_v3 canonical_packets=2 rejected_mutations=%d\\n",mutations);
}
''')
