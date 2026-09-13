#include "prefill_shape_packet.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#ifdef NDEBUG
#error "This validation executable requires assertions"
#endif

int main(int argc,char**argv) {
  assert(argc==2);
  std::ifstream input(argv[1],std::ios::binary);
  std::vector<uint8_t> packet((std::istreambuf_iterator<char>(input)),{});
  assert(packet.size()==61568);
  auto get=[](const std::vector<uint8_t>& p,unsigned at){uint32_t v;std::memcpy(&v,p.data()+at,4);return v;};
  auto set=[](std::vector<uint8_t>&p,unsigned at,uint32_t v){std::memcpy(p.data()+at,&v,4);};
  auto set16=[](std::vector<uint8_t>&p,unsigned at,uint16_t v){std::memcpy(p.data()+at,&v,2);};
  const uint32_t physical=get(packet,28),capacity=get(packet,24);
  auto valid=[&](const std::vector<uint8_t>& p,bool shared){
    return valid_v7_shape_packet(p.data(),p.size(),physical,capacity,shared);
  };
  assert(valid(packet,true));
  assert(!valid(packet,false));
  unsigned accepted=1,rejected=1;
  constexpr unsigned first=128,second=128+1664;
  // The old exclusive shape remains valid in both owner modes.
  {auto p=packet;set(p,second+128,4);set(p,second+132,5);
   assert(valid(p,false)&&valid(p,true));accepted+=2;}
  // Aliased append tails, shifted logical positions, and duplicate local pages.
  for(unsigned kind=0;kind<3;++kind){auto p=packet;
    if(kind==0)set(p,second+136,get(p,first+136));
    if(kind==1){set(p,second+128,1);set(p,second+132,0);}
    if(kind==2)set(p,second+132,0);
    assert(!valid(p,true));++rejected;
  }
  // Reader/writer conflicts in either row order, with every other row field
  // kept canonical so the sharing check is the reason for rejection.
  for(unsigned row:{first,second}){auto p=packet;
    set(p,row+4,16);set(p,row+12,2);set(p,row+16,16);set(p,row+20,17);
    set(p,row+32,16);set(p,row+136,0);set16(p,row+1154,1);set16(p,row+1156,0);
    assert(!valid(p,true));++rejected;
  }
  {auto p=packet;set(p,second+128,physical);assert(!valid(p,true));++rejected;}
  {auto p=packet;set(p,100,1);assert(!valid(p,true)&&!valid(p,false));rejected+=2;}
  // Structural validation cannot establish the off-batch ownership ledger;
  // that check belongs to the Rust retained authority.
  for(size_t at=0;at<packet.size();++at){packet[at]^=0xff;
    (void)valid(packet,true);(void)valid(packet,false);packet[at]^=0xff;
  }
  assert(valid(packet,true));
  std::cout<<"shared_prefix_packet accepted="<<accepted<<" rejected="<<rejected
           <<" byte_mutations="<<packet.size()<<" modes=2\n";
}
