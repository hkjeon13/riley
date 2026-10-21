#include "ffi_internal.hpp"
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>
int main(){
 using namespace riley_cuda_internal;
 RileyCudaStream stream(nullptr,nullptr);stream.command_batch_owner.store(command_batch_thread_token());
 std::array<std::atomic<uint32_t>,1025> counters{};RileyCudaErrorInfo error{};
 std::thread foreign([&]{assert(command_batch_register_use(&stream,&counters[0],&error,"test","busy")==RILEY_CUDA_STATUS_INVALID_STATE);});foreign.join();assert(stream.command_batch_use_count==0&&counters[0]==0);
 for(size_t i=0;i<1024;i++)assert(command_batch_register_use(&stream,&counters[i],&error,"test","busy")==RILEY_CUDA_STATUS_SUCCESS);
 assert(command_batch_register_use(&stream,&counters[1024],&error,"test","busy")==RILEY_CUDA_STATUS_OUT_OF_RANGE);assert(counters[1024]==0&&stream.command_batch_use_count==1024);
 auto start=std::chrono::steady_clock::now();
 for(size_t i=0;i<2000000;i++)assert(command_batch_register_use(&stream,&counters[(i*727)%1024],&error,"test","busy")==RILEY_CUDA_STATUS_SUCCESS);
 auto elapsed=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/2000000;
 for(size_t i=0;i<1024;i++)assert(counters[i].load()==1);
 RileyCudaStream other(nullptr,nullptr);other.command_batch_owner.store(command_batch_thread_token());assert(command_batch_register_use(&other,&counters[0],&error,"test","busy")==RILEY_CUDA_STATUS_INVALID_STATE);assert(other.command_batch_use_count==0);
#ifdef RILEY_LEDGER_INDEX_CHECK
 for(size_t i=0;i<1024;i++){assert(stream.command_batch_uses[i]==&counters[i]);assert(counters[i].exchange(0)==1);stream.command_batch_uses[i]=nullptr;}
 stream.command_batch_use_count=0;stream.command_batch_index.clear();
 for(size_t generation=0;generation<4;generation++){
  for(size_t i=0;i<1024;i++)assert(command_batch_register_use(&stream,&counters[(i*727+generation)%1024],&error,"test","busy")==RILEY_CUDA_STATUS_SUCCESS);
  assert(stream.command_batch_use_count==1024);
  for(size_t i=0;i<1024;i++){assert(stream.command_batch_uses[i]->exchange(0)==1);stream.command_batch_uses[i]=nullptr;}
  stream.command_batch_use_count=0;stream.command_batch_index.clear();
 }
 RileyCudaCommandBatchIndex<std::atomic<uint32_t>,8> tiny;
 for(size_t i=0;i<8;i++){auto slot=tiny.find(&counters[i]);assert(slot<8&&tiny.slots[slot]==nullptr);tiny.slots[slot]=&counters[i];}
 for(size_t i=0;i<8;i++)assert(tiny.slots[tiny.find(&counters[i])]==&counters[i]);
 assert(tiny.find(&counters[8])==8);tiny.clear();for(auto*p:tiny.slots)assert(p==nullptr);
#endif
 std::printf("{\"duplicate_register_ns\":%.3f,\"capacity\":1024,\"duplicates\":2000000,\"foreign_thread_rejected\":true,\"foreign_stream_rejected\":true,\"overflow_unleased\":true}\n",elapsed);
}
