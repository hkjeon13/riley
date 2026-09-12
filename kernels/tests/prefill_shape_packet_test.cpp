#include "prefill_shape_packet.hpp"
#include <vector>
#include <fstream>
#include <iterator>
#include <cstdio>
#include <cassert>
int main(int argc,char**argv){assert(argc==3);int mutations=0;
 for(int n=1;n<argc;++n){std::ifstream f(argv[n],std::ios::binary);std::vector<uint8_t>p((std::istreambuf_iterator<char>(f)),{});assert(valid_prefill_shape_packet(p.data(),p.size(),4096,8));
  for(auto at:{0,4,8,12,24,28,36,96,128+64,128+1152}){auto q=p;q[at]^=0x80;assert(!valid_prefill_shape_packet(q.data(),q.size(),4096,8));++mutations;}
  for(size_t i=0;i<p.size();++i){p[i]^=0xff;valid_prefill_shape_packet(p.data(),p.size(),4096,8);p[i]^=0xff;}
  assert(!valid_prefill_shape_packet(p.data(),p.size()-1,4096,8));assert(!valid_prefill_shape_packet(p.data(),p.size(),4095,8));mutations+=2;
 }
 printf("native_v3 canonical_packets=2 rejected_mutations=%d\n",mutations);
}
