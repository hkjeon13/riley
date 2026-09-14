#include <fstream>
#include <vector>
#include <iterator>
#include <cstdio>
#include "../../kernels/src/prefill_shape_packet.hpp"
int main(int argc,char**argv){
 if(argc!=2)return 2;std::ifstream f(argv[1],std::ios::binary);std::vector<uint8_t> p((std::istreambuf_iterator<char>(f)),{});
 auto valid=[&](const std::vector<uint8_t>& b,bool capability){return valid_v7_shape_packet(b.data(),b.size(),64,32,false,capability);};
 if(!valid(p,true)||valid(p,false))return 3;
 unsigned rejected=0;
 for(auto edit:std::vector<std::pair<unsigned,unsigned>>{{16,2},{16,0},{128+24,0},{128+16,33},{128+28,8},{128+72,0},{128+8,9},{128+32,40},{128+44,0},{128+64,1}}){
  auto bad=p;std::memcpy(bad.data()+edit.first,&edit.second,4);if(valid(bad,true))return 4;++rejected;
 }
 std::printf("verification_packet valid_explicit_stage=true ordinary_rejected=true corruptions_rejected=%u\n",rejected);
}
