#include "prefill_shape_packet.hpp"
#include <cassert>
#include <fstream>
#include <iterator>
#include <vector>
#include <iostream>
int main(int argc,char**argv){
 assert(argc==12);unsigned accepted=0,rejected=0;
 for(int arg=1;arg<argc;++arg){
  std::ifstream f(argv[arg],std::ios::binary);std::vector<uint8_t> p((std::istreambuf_iterator<char>(f)),{});
  bool wide=p.size()==30848;auto valid=[&](const std::vector<uint8_t>&q){return wide?valid_v4_shape_packet(q.data(),q.size(),4096,16):valid_prefill_shape_packet(q.data(),q.size(),4096,8);};
  assert(valid(p));++accepted;
  assert(!(wide?valid_prefill_shape_packet(p.data(),p.size(),4096,8):valid_v4_shape_packet(p.data(),p.size(),4096,16)));++rejected;
  for(unsigned at:{0,4,8,12,24,28,36,96,128+64}) {auto q=p;q.at(at)^=0x80;assert(!valid(q));++rejected;}
  auto set=[](std::vector<uint8_t>&q,unsigned at,uint32_t v){std::memcpy(q.data()+at,&v,4);};
  auto get=[&](unsigned at){uint32_t v;std::memcpy(&v,p.data()+at,4);return v;};
  unsigned active=get(20),rows=wide?16:8,tokens=128+rows*1664;
  {auto q=p;set(q,20,rows+1);assert(!valid(q));++rejected;}
  {auto q=p;set(q,0,wide?0x33444d52:0x34444d52);set(q,4,wide?3:4);assert(!valid(q));++rejected;}
  if(active>1){auto q=p;set(q,128+(active-1)*1664+128,get(256));assert(!valid(q));++rejected;}
  for(unsigned at=128+active*1664;at<tokens;++at){auto q=p;q[at]=1;assert(!valid(q));++rejected;}
  {auto q=p;set(q,tokens,49152);assert(!valid(q));++rejected;}
  {auto q=p;q.pop_back();assert(!valid(q));++rejected;}
  {auto q=p;q.push_back(0);assert(!valid(q));++rejected;}
  // Every byte mutation is memory safe; some semantic changes require live Rust authority.
  for(size_t at=0;at<p.size();++at){p[at]^=0xff;(void)valid(p);p[at]^=0xff;}
 }
 std::cout<<"accepted="<<accepted<<" rejected="<<rejected<<"\n";
}
