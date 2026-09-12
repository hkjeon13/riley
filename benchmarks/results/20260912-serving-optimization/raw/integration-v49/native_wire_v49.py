from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'kernels/src/prefill_shape_packet.hpp';s=p.read_text();s+='''
// V7 mixed stages retain the V6 token slab and append1024 canonical tile entries.
inline bool valid_v7_shape_packet(const uint8_t* p,uint64_t bytes,uint32_t physical,uint32_t capacity) noexcept {
 constexpr uint32_t header=128,stride=1664,tokens=53376,map=57472,extent=61568;
 if(!p||bytes!=extent||!physical||physical>4096||!(capacity==1||capacity==2||capacity==4||capacity==8||capacity==16||capacity==32))return false;
 auto u32=[&](uint32_t at){uint32_t v;std::memcpy(&v,p+at,4);return v;};
 auto u64=[&](uint32_t at){uint64_t v;std::memcpy(&v,p+at,8);return v;};
 auto u16=[&](uint32_t at){uint16_t v;std::memcpy(&v,p+at,2);return v;};
 auto zero=[&](uint32_t a,uint32_t b){for(;a<b;++a)if(p[a])return false;return true;};
 uint32_t stage=u32(16),active=u32(20);
 if(u32(0)!=0x37444d52||u32(4)!=7||u32(8)!=extent||u32(12)!=stride||stage>2||!active||active>capacity||u32(24)!=capacity||u32(28)!=physical||u32(32)>1||!u64(40)||!u64(48)||!u64(56)||zero(64,96)||!zero(100,128))return false;
 if(stage==1?(u32(36)||u32(96)):(!u32(36)||u32(36)>1024||!u32(96)||u32(96)>1024))return false;
 bool used[4096]{};uint32_t slots=0,input=0,tiles=0,prefills=0,decodes=0,published=0;
 for(uint32_t row=0;row<active;++row){
  uint32_t b=header+row*stride,count=u32(b+8),live=u32(b+12),committed=u32(b+16),target=u32(b+20),generated=u32(b+24),limit=u32(b+28),prompt=u32(b+32),context=u32(b+36),slot=u32(b+40),logit=u32(b+44),kind=u32(b+72);
  if(!count||count>1024||!prompt||!limit||!context||context>4096||uint64_t(prompt)+limit-1>context||generated>=limit||uint64_t(committed)+count!=target||!target||target>context||u32(b+4)!=target-1||u32(b)>=49152||live!=(target+15)/16||slot>=active||(slots&(1U<<slot))||!u64(b+48)||!u64(b+56)||kind>1||!zero(b+76,b+128))return false;
  slots|=1U<<slot;
  if(kind==0){++prefills;if(generated||committed>=prompt||target>prompt||logit!=(target==prompt?count-1:UINT32_MAX))return false;}
  else {++decodes;if(count!=1||!generated||uint64_t(prompt)+generated-1!=committed||logit!=0)return false;}
  published+=logit!=UINT32_MAX;
  if(stage!=1){
   if(u32(b+64)!=input||count>1024-input||u32(tokens+input*4)!=u32(b))return false;input+=count;
   uint32_t nt=count<32?count:(count+7)/8;
   if(u32(b+68)!=tiles||nt>1024-tiles)return false;
   for(uint32_t local=0;local<nt;++local)if(u32(map+(tiles+local)*4)!=((row<<16)|local))return false;
   tiles+=nt;
  }else if(u32(b+64)||u32(b+68))return false;
  for(uint32_t prior=0;prior<row;++prior)if(u64(b+48)==u64(header+prior*stride+48)||u64(b+56)==u64(header+prior*stride+56))return false;
  for(uint32_t i=0;i<256;++i){uint32_t id=u32(b+128+i*4);uint16_t valid=u16(b+1152+i*2);
   if(i>=live){if(id||valid)return false;continue;}
   if(id>=physical||used[id]||valid!=(i+1<live?16:(target-1)%16+1))return false;used[id]=true;
  }
 }
 if(prefills>4||(stage==0&&decodes)||(stage==1&&prefills)||(stage==2&&(!prefills||!decodes))||u32(36)!=input||u32(96)!=tiles||!zero(header+active*stride,tokens))return false;
 for(uint32_t row=0;row<active;++row){uint32_t b=header+row*stride;if((u32(b+40)<published)!=(u32(b+44)!=UINT32_MAX))return false;}
 for(uint32_t i=0;i<1024;++i){uint32_t t=u32(tokens+i*4);if(i<input){if(t>=49152)return false;}else if(t)return false;if(i>=tiles&&u32(map+i*4))return false;}
 return true;
}
''';p.write_text(s)
p=r/'kernels/tests/mixed_wire_v49_test.cpp';s=(r/'kernels/tests/packed_wire_v48_test.cpp').read_text().replace('argc==20','argc==6').replace('valid_v6_shape_packet','valid_v7_shape_packet').replace('valid_v5_shape_packet','valid_v6_shape_packet').replace('get(16)==0','get(16)!=1').replace('19*57472','5*61568')
s=s.replace('for(unsigned at:{0,4,8,12,24,28,36,96,128+68})', 'for(unsigned at:{0,4,8,12,24,28,36,96,128+68,128+72,57472})')
p.write_text(s)
print('V7 native parser and cross-language corruption probe added')
