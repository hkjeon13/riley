#include "prefill_shape_packet.hpp"
#include <cassert>
#include <fstream>
#include <iterator>
#include <vector>
#include <iostream>
int main(int argc,char**argv){
 assert(argc==6);unsigned accepted=0,rejected=0;
 for(int arg=1;arg<argc;++arg){
  std::ifstream f(argv[arg],std::ios::binary);std::vector<uint8_t>p((std::istreambuf_iterator<char>(f)),{});
  auto valid=[](const std::vector<uint8_t>&q){return valid_v7_shape_packet(q.data(),q.size(),4096,32);};
  auto get=[&](unsigned at){uint32_t v;std::memcpy(&v,p.data()+at,4);return v;};
  auto set=[](std::vector<uint8_t>&q,unsigned at,uint32_t v){std::memcpy(q.data()+at,&v,4);};
  assert(valid(p));++accepted;assert(!valid_v6_shape_packet(p.data(),p.size(),4096,32));++rejected;
  for(unsigned at:{0,4,8,12,24,28,36,96,128+68,128+72,57472}){auto q=p;q.at(at)^=0x80;assert(!valid(q));++rejected;}
  const unsigned active=get(20),tokens=53376;
  for(unsigned row=0;row<active;++row){
   unsigned b=128+row*1664;
   for(unsigned value:{1U,1024U,UINT32_MAX}){auto q=p;set(q,b+64,get(b+64)+value);assert(!valid(q));++rejected;}
   if(get(16)!=1){auto q=p;set(q,tokens+get(b+64)*4,(get(b)+1)%49152);assert(!valid(q));++rejected;}
  }
  if(active>1){auto q=p;set(q,128+1664+128,get(256));assert(!valid(q));++rejected;}
  {auto q=p;set(q,20,33);assert(!valid(q));++rejected;}
  {auto q=p;set(q,tokens,49152);assert(!valid(q));++rejected;}
  {auto q=p;q.pop_back();assert(!valid(q));++rejected;}
  {auto q=p;q.push_back(0);assert(!valid(q));++rejected;}
  // Exhaustive single-byte fuzzing checks parser memory safety. Semantic
  // differences to retained identities are separately rejected in Rust.
  for(size_t at=0;at<p.size();++at){p[at]^=0xff;(void)valid(p);p[at]^=0xff;}
 }
 std::cout<<"accepted="<<accepted<<" rejected="<<rejected<<" exhaustive_byte_mutations="<<5*61568<<"\n";
}
