// Structural admission for the low-level numerical recorder. Scheduler ownership
// and live reservation authorization must additionally be checked by its Rust
// runtime owner; these echoed wire identities do not establish that authority.
static bool valid_multisequence_packet(const uint8_t* source,uint64_t bytes,
    uint32_t bucket,uint32_t physical,uint32_t full) noexcept {
  if(bytes<1280)return false;
  auto u32=[&](size_t offset){uint32_t value;std::memcpy(&value,source+offset,4);return value;};
  auto u64=[&](size_t offset){uint64_t value;std::memcpy(&value,source+offset,8);return value;};
  auto u16=[&](size_t offset){uint16_t value;std::memcpy(&value,source+offset,2);return value;};
  auto zero=[&](size_t begin,size_t end){for(size_t i=begin;i<end;++i)if(source[i])return false;return true;};
  const uint32_t active=u32(24);
  if(u32(0)!=0x31444d52||u16(4)!=1||u16(6)!=128||u32(8)!=1280||u32(12)!=128||
      u32(16)!=1||u32(20)!=bucket||active<2||active>bucket||(bucket==4&&active<3)||
      u32(28)!=active||u32(32)!=4||u32(36)!=10||u32(40)!=49152||u32(44)!=physical||
      !u64(48)||!u64(56)||!u64(64)||zero(72,104)||u32(104)!=full||!zero(108,128)||
      !zero(128+active*128,static_cast<size_t>(bytes)))return false;
  // At most 40 live block IDs; no allocations or unbounded work on replay.
  uint32_t blocks[40]{},count=0,slots=0;
  for(uint32_t row=0;row<active;++row){
    const size_t base=128+row*128;
    const uint32_t pos=u32(base+4),live=u32(base+12),slot=u32(base+100);
    if(u32(base)>=49152||pos<128||pos>=160||u32(base+8)||live!=pos/16+1||u32(base+76)||
        !u64(base+80)||!u64(base+88)||u32(base+96)!=pos+1||slot>=active||
        (slots&(1U<<slot))||u32(base+104)!=1||u32(base+108)!=pos-127||u32(base+112)!=1||
        u16(base+116)!=1||!zero(base+118,base+128))return false;
    slots|=1U<<slot;
    for(uint32_t prior=0;prior<row;++prior)
      if(u64(base+80)==u64(128+prior*128+80)||u64(base+88)==u64(128+prior*128+88))return false;
    for(uint32_t i=0;i<10;++i){
      const uint32_t block=u32(base+16+4*i),valid=u16(base+56+2*i);
      if(i>=live){if(block||valid)return false;continue;}
      if(block>=physical||valid!=(i+1<live?16:pos%16+1))return false;
      for(uint32_t j=0;j<count;++j)if(blocks[j]==block)return false;
      blocks[count++]=block;
    }
  }
  return true;
}
