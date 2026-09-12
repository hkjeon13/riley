#pragma once
#include <cstdint>
#include <cstring>
// Structural check only. Rust retained-owner authority must bind identities,
// complete page ownership, catalog and replay order before GPU dispatch.
template<uint32_t Rows>
inline bool valid_variable_shape_packet(const uint8_t* p,uint64_t bytes,uint32_t physical,uint32_t capacity) noexcept {
 static_assert(Rows==8||Rows==16,"unsupported wire capacity");
 constexpr uint32_t header=128,stride=1664,tokens=header+Rows*stride,extent=tokens+4096;
 constexpr uint32_t magic=Rows==8?0x33444d52:0x34444d52,version=Rows==8?3:4;
 if(!p||bytes!=extent||!physical||physical>4096||!(capacity==1||capacity==2||capacity==4||capacity==8||(Rows==16&&capacity==16)))return false;
 auto u32=[&](uint32_t at){uint32_t v;std::memcpy(&v,p+at,4);return v;};
 auto u64=[&](uint32_t at){uint64_t v;std::memcpy(&v,p+at,8);return v;};
 auto u16=[&](uint32_t at){uint16_t v;std::memcpy(&v,p+at,2);return v;};
 auto zero=[&](uint32_t a,uint32_t b){for(;a<b;++a)if(p[a])return false;return true;};
 uint32_t stage=u32(16),active=u32(20);
 if(u32(0)!=magic||u32(4)!=version||u32(8)!=extent||u32(12)!=stride||stage>1||!active||active>capacity||u32(24)!=capacity||u32(28)!=physical||u32(32)>1||!zero(36,40)||!u64(40)||!u64(48)||!u64(56)||zero(64,96)||!zero(96,128)||(stage==0&&active!=1))return false;
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
   if(id>=physical||used[id]||valid!=(i+1<live?16:(target-1)%16+1))return false;
   used[id]=true;
  }
 }
 if(!zero(header+active*stride,tokens))return false;
 for(uint32_t i=0;i<1024;++i){uint32_t t=u32(tokens+i*4);if(i<prefill_count){if(t>=49152||(i==0&&t!=u32(header)))return false;}else if(t)return false;}
 return true;
}

inline bool valid_prefill_shape_packet(const uint8_t* p,uint64_t bytes,uint32_t physical,uint32_t capacity) noexcept {
 return valid_variable_shape_packet<8>(p,bytes,physical,capacity);
}
inline bool valid_v4_shape_packet(const uint8_t* p,uint64_t bytes,uint32_t physical,uint32_t capacity) noexcept {
 return valid_variable_shape_packet<16>(p,bytes,physical,capacity);
}
