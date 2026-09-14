#pragma once
#include <cstdint>
#include <cstring>
// Structural check only. Rust retained-owner authority must bind identities,
// complete page ownership, catalog and replay order before GPU dispatch.
template<uint32_t Rows,bool Packed=false>
inline bool valid_variable_shape_packet(const uint8_t* p,uint64_t bytes,uint32_t physical,uint32_t capacity) noexcept {
 static_assert(Rows==8||Rows==16||Rows==32,"unsupported wire capacity");
 constexpr uint32_t header=128,stride=1664,tokens=header+Rows*stride,extent=tokens+4096;
 static_assert(!Packed||Rows==32,"packed prefill requires V6 capacity");
 constexpr uint32_t magic=Packed?0x36444d52:(Rows==8?0x33444d52:(Rows==16?0x34444d52:0x35444d52)),version=Packed?6:(Rows==8?3:(Rows==16?4:5));
 if(!p||bytes!=extent||!physical||physical>4096||!(capacity==1||capacity==2||capacity==4||capacity==8||(Rows>=16&&capacity==16)||(Rows==32&&capacity==32)))return false;
 auto u32=[&](uint32_t at){uint32_t v;std::memcpy(&v,p+at,4);return v;};
 auto u64=[&](uint32_t at){uint64_t v;std::memcpy(&v,p+at,8);return v;};
 auto u16=[&](uint32_t at){uint16_t v;std::memcpy(&v,p+at,2);return v;};
 auto zero=[&](uint32_t a,uint32_t b){for(;a<b;++a)if(p[a])return false;return true;};
 uint32_t stage=u32(16),active=u32(20);
 if(u32(0)!=magic||u32(4)!=version||u32(8)!=extent||u32(12)!=stride||stage>1||!active||active>capacity||u32(24)!=capacity||u32(28)!=physical||u32(32)>1||(Packed&&stage==0?u32(36)==0||u32(36)>1024:!zero(36,40))||!u64(40)||!u64(48)||!u64(56)||zero(64,96)||!zero(96,128)||(stage==0&&active>(Packed?4U:1U)))return false;
 bool used[4096]{};uint32_t slots=0,prefill_count=0;
 for(uint32_t row=0;row<active;++row){
  uint32_t b=header+row*stride,count=u32(b+8),live=u32(b+12),committed=u32(b+16),target=u32(b+20),generated=u32(b+24),limit=u32(b+28),prompt=u32(b+32),context=u32(b+36),slot=u32(b+40),logit=u32(b+44);
  if(!count||count>1024||!prompt||!limit||!context||context>4096||uint64_t(prompt)+limit-1>context||generated>=limit||uint64_t(committed)+count!=target||!target||target>context||u32(b+4)!=target-1||u32(b)>=49152||live!=(target+15)/16||slot>=active||(slots&(1U<<slot))||!u64(b+48)||!u64(b+56)||!zero(b+(Packed&&stage==0?68:64),b+128))return false;
  slots|=1U<<slot;
  if(stage==0){if(generated||committed>=prompt||target>prompt||logit!=(target==prompt?count-1:UINT32_MAX))return false;if(Packed){if(u32(b+64)!=prefill_count||count>1024-prefill_count)return false;}
   if(u32(tokens+prefill_count*4)!=u32(b))return false;
   prefill_count+=count;}
  else if(count!=1||!generated||uint64_t(prompt)+generated-1!=committed||logit!=0)return false;
  for(uint32_t prior=0;prior<row;++prior)if(u64(b+48)==u64(header+prior*stride+48)||u64(b+56)==u64(header+prior*stride+56))return false;
  for(uint32_t i=0;i<256;++i){uint32_t id=u32(b+128+i*4);uint16_t valid=u16(b+1152+i*2);
   if(i>=live){if(id||valid)return false;continue;}
   if(id>=physical||used[id]||valid!=(i+1<live?16:(target-1)%16+1))return false;
   used[id]=true;
  }
 }
 if(Packed&&stage==0&&u32(36)!=prefill_count)return false;
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

inline bool valid_v5_shape_packet(const uint8_t* p,uint64_t bytes,uint32_t physical,uint32_t capacity) noexcept {return valid_variable_shape_packet<32>(p,bytes,physical,capacity);}

inline bool valid_v6_shape_packet(const uint8_t* p,uint64_t bytes,uint32_t physical,uint32_t capacity) noexcept {return valid_variable_shape_packet<32,true>(p,bytes,physical,capacity);}

// V7 mixed stages retain the V6 token slab and append1024 canonical tile entries.
template<bool SharedPrefixes,bool Verification=false,bool Wide=false>
inline bool valid_v7_shape_packet_impl(const uint8_t* p,uint64_t bytes,uint32_t physical,uint32_t capacity) noexcept {
 constexpr uint32_t header=128,stride=1664,tokens=53376,map=57472,extent=61568;
 if(!p||bytes!=extent||!physical||physical>4096||!(capacity==1||capacity==2||capacity==4||capacity==8||capacity==16||capacity==32))return false;
 auto u32=[&](uint32_t at){uint32_t v;std::memcpy(&v,p+at,4);return v;};
 auto u64=[&](uint32_t at){uint64_t v;std::memcpy(&v,p+at,8);return v;};
 auto u16=[&](uint32_t at){uint16_t v;std::memcpy(&v,p+at,2);return v;};
 auto zero=[&](uint32_t a,uint32_t b){for(;a<b;++a)if(p[a])return false;return true;};
 uint32_t stage=u32(16),active=u32(20);
 if(u32(0)!=0x37444d52||u32(4)!=7||u32(8)!=extent||u32(12)!=stride||stage>(Verification?3u:2u)||!active||active>capacity||u32(24)!=capacity||u32(28)!=physical||u32(32)>1||!u64(40)||!u64(48)||!u64(56)||zero(64,96)||!zero(100,128))return false;
 if(stage==1?(u32(36)||u32(96)):(!u32(36)||u32(36)>1024||!u32(96)||u32(96)>1024))return false;
 bool used[4096]{};
 // Uninitialized slots are read only after used[id]. The legacy specialization
 // eliminates this ledger and retains its existing exclusive-page fast path.
 uint16_t shared_slots[SharedPrefixes?4096:1];
 uint32_t slots=0,input=0,tiles=0,prefills=0,decodes=0,published=0;
 for(uint32_t row=0;row<active;++row){
  uint32_t b=header+row*stride,count=u32(b+8),live=u32(b+12),committed=u32(b+16),target=u32(b+20),generated=u32(b+24),limit=u32(b+28),prompt=u32(b+32),context=u32(b+36),slot=u32(b+40),logit=u32(b+44),kind=u32(b+72);
  if(!count||count>1024||!prompt||!limit||!context||context>4096||uint64_t(prompt)+limit-1>context||generated>=limit||uint64_t(committed)+count!=target||!target||target>context||u32(b+4)!=target-1||u32(b)>=49152||live!=(target+15)/16||slot>=active||(slots&(1U<<slot))||!u64(b+48)||!u64(b+56)||(kind>1 && !(Verification && kind==3))||!zero(b+76,b+128))return false;
  slots|=1U<<slot;
  if(kind==3){++prefills;if(stage!=3 || !generated || uint64_t(prompt)+generated-1!=committed || count>8 || count>limit-generated || logit!=count-1)return false;}
  else if(kind==0){++prefills;if(stage==3||generated||committed>=prompt||target>prompt||logit!=(target==prompt?count-1:UINT32_MAX))return false;}
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
   if(id>=physical||valid!=(i+1<live?16:(target-1)%16+1))return false;
   if constexpr(SharedPrefixes){
    const bool writes=i>=committed/16;
    if(used[id]){
     const uint16_t prior=shared_slots[id];
     if(writes||(prior&256U)||(prior&255U)!=i||(prior>>9)==row)return false;
    }else shared_slots[id]=static_cast<uint16_t>((row<<9)|(writes?256U:0U)|i);
   }else if(used[id])return false;
   used[id]=true;
  }
 }
 if(prefills>(stage==3&&Wide?32:4)||((stage==0||stage==3)&&decodes)||(stage==1&&prefills)||(stage==2&&(!prefills||!decodes))||u32(36)!=input||u32(96)!=tiles||!zero(header+active*stride,tokens))return false;
 for(uint32_t row=0;row<active;++row){uint32_t b=header+row*stride;if((u32(b+40)<published)!=(u32(b+44)!=UINT32_MAX))return false;}
 for(uint32_t i=0;i<1024;++i){uint32_t t=u32(tokens+i*4);if(i<input){if(t>=49152)return false;}else if(t)return false;if(i>=tiles&&u32(map+i*4))return false;}
 return true;
}

// This capability comes from retained owner configuration, never packet bytes.
// Rust must additionally bind each reader to the complete live ownership ledger.
inline bool valid_v7_shape_packet(const uint8_t* p,uint64_t bytes,uint32_t physical,
    uint32_t capacity,bool shared_prefixes=false,bool verification=false,bool wide=false) noexcept {
 if(verification && wide && !shared_prefixes)return valid_v7_shape_packet_impl<false,true,true>(p,bytes,physical,capacity);
 if(verification && !shared_prefixes)return valid_v7_shape_packet_impl<false,true>(p,bytes,physical,capacity);
 return shared_prefixes?valid_v7_shape_packet_impl<true>(p,bytes,physical,capacity)
                       :valid_v7_shape_packet_impl<false>(p,bytes,physical,capacity);
}
