#include <fstream>
#include <vector>
#include <iterator>
#include <cstdio>
#include "../../kernels/src/prefill_shape_packet.hpp"
int main(int argc,char**argv){
 if(argc!=2)return 2;
 std::ifstream f(argv[1],std::ios::binary);
 std::vector<uint8_t> p((std::istreambuf_iterator<char>(f)),{});
 auto valid=[&](const std::vector<uint8_t>& b,bool shared,bool verification,bool wide){return valid_v7_shape_packet(b.data(),b.size(),8,32,shared,verification,wide);};
 if(!valid(p,true,true,true)||valid(p,false,true,true)||valid(p,true,false,false)||valid(p,true,true,false))return 3;
 unsigned rejected=0;
 // Shared write alias, position alias and duplicate page within a row.
 for(auto edit:std::vector<std::pair<unsigned,unsigned>>{{128+1664+128+8,2},{128+1664+128,1},{128+1664+128+4,0},{128+24,0},{128+72,0}}){
  auto bad=p;std::memcpy(bad.data()+edit.first,&edit.second,4);
  if(valid(bad,true,true,true))return 4;++rejected;
 }
 std::printf("shared_verification_packet capabilities_bound=true corruptions_rejected=%u\n",rejected);
}
