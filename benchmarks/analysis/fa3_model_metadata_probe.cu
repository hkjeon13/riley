#include "fa3_api.h"
#include "fa3_model_metadata.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>
#include <string>
#define OK(x) do {auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"CUDA line %d: %d\n",__LINE__,int(e));std::exit(1);}}while(0)
#define REQUIRE(x) do{if(!(x)){std::fprintf(stderr,"failed line %d: %s\n",__LINE__,#x);std::exit(2);}}while(0)
using riley_fa3_model::Workspace;
using riley_fa3_model::PacketWords;
struct Case {std::vector<unsigned> packet;unsigned expected=0,inherited=0;};
Case make(std::vector<std::pair<unsigned,unsigned>> shapes) {
    Case c{std::vector<unsigned>(PacketWords,0)};c.packet[5]=shapes.size();
    unsigned cursor=0;
    for(unsigned i=0;i<shapes.size();++i) {
        auto [q,n]=shapes[i];auto *s=c.packet.data()+32+i*416;
        s[1]=n-1;s[2]=q;s[16]=cursor;s[18]=q==1?1:0;cursor+=q;
        for(unsigned p=0;p<(n+15)/16;++p)s[32+p]=(2047-i*17-p*13)%2048;
    }
    c.packet[9]=cursor;return c;
}
int main() {
    std::vector<Case> cases{make({}),make({{1,1}}),make({{15,32}}),make({{16,32}}),
        make({{17,32}}),make({{1,4096},{128,256},{17,398}}),make({{1024,4096}}),
        make(std::vector<std::pair<unsigned,unsigned>>(32,{32,4096})),
        make(std::vector<std::pair<unsigned,unsigned>>(32,{1,31}))};
    auto decode=cases[8];decode.packet[4]=1;decode.packet[9]=0;
    for(unsigned r=0;r<32;++r)decode.packet[32+r*416+16]=0;
    cases.push_back(decode);
    auto bad=decode;bad.packet[9]=32;bad.expected=1;cases.push_back(bad);
    bad=decode;bad.packet[32+31*416+16]=31;bad.expected=2;cases.push_back(bad);
    bad=cases[5];bad.packet[32+2*416+32+24]=2048;bad.expected=4;cases.push_back(bad);
    bad=cases[5];bad.packet[32+2*416+1]=4096;bad.expected=2;cases.push_back(bad);
    bad=cases[5];bad.packet[32+2*416+16]++;bad.expected=2;cases.push_back(bad);
    bad=cases[5];bad.packet[9]++;bad.expected=16;cases.push_back(bad);
    bad=cases[5];bad.packet[5]=33;bad.expected=1;cases.push_back(bad);
    bad=cases[5];bad.packet[32+2*416+18]=1;bad.expected=2;cases.push_back(bad);
    bad=cases[5];bad.inherited=8;bad.expected=8;cases.push_back(bad);
    bad=cases[5];bad.packet[32+2*416+1]=~0u;bad.expected=2;cases.push_back(bad);
    unsigned *packet,*status;Workspace *work;cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;
    OK(cudaMalloc(&packet,PacketWords*4));OK(cudaMalloc(&status,4));OK(cudaMalloc(&work,sizeof(Workspace)));
    OK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    REQUIRE(riley_fa3_model_workspace_bytes()==sizeof(Workspace));
    REQUIRE(riley_fa3_model_metadata_prepare(stream,packet,4,work,sizeof(Workspace),2048,1024,4096,status)==cudaErrorInvalidValue);
    OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    OK(riley_fa3_model_metadata_prepare(stream,packet,PacketWords*4,work,sizeof(Workspace),2048,1024,4096,status));
    OK(cudaStreamEndCapture(stream,&graph));OK(cudaGraphInstantiate(&exec,graph,0));
    unsigned comparisons=0;
    for(unsigned replay=0;replay<2;++replay)for(auto c:cases) {
        if(replay && !c.expected) for(unsigned r=0;r<c.packet[5];++r) {
            auto *s=c.packet.data()+32+r*416;
            for(unsigned p=0;p<(s[1]+16)/16;++p)s[32+p]=(s[32+p]+71)%2048;
        }
        OK(cudaMemsetAsync(work,0xa5,sizeof(Workspace),stream));
        OK(cudaMemcpyAsync(packet,c.packet.data(),PacketWords*4,cudaMemcpyHostToDevice,stream));
        OK(cudaMemcpyAsync(status,&c.inherited,4,cudaMemcpyHostToDevice,stream));
        OK(cudaGraphLaunch(exec,stream));Workspace got;unsigned got_status;
        OK(cudaMemcpyAsync(&got,work,sizeof(got),cudaMemcpyDeviceToHost,stream));
        OK(cudaMemcpyAsync(&got_status,status,4,cudaMemcpyDeviceToHost,stream));OK(cudaStreamSynchronize(stream));
        REQUIRE(got_status==c.expected);
        for(unsigned r=0;r<33;++r) {
            unsigned expected=c.expected?0:c.packet[4]==1?(r<c.packet[5]?r:c.packet[5]):r<c.packet[5]?c.packet[32+r*416+16]:c.packet[9];
            REQUIRE(got.q_indptr[r]==int(expected));
        }
        for(unsigned r=0;r<32;++r) {
            unsigned n=(!c.expected && r<c.packet[5])?c.packet[32+r*416+1]+1:0;
            REQUIRE(got.kv_lengths[r]==int(n));
            for(unsigned p=0;p<256;++p) {
                REQUIRE(got.pages[r*256+p]>=0 && got.pages[r*256+p]<2048);
                if(!c.expected) REQUIRE(got.pages[r*256+p]==int(p<(n+15)/16?c.packet[32+r*416+32+p]:0));
            }
        }
        ++comparisons;
    }
    OK(cudaGraphExecDestroy(exec));OK(cudaGraphDestroy(graph));OK(cudaStreamDestroy(stream));
    OK(cudaFree(work));OK(cudaFree(status));OK(cudaFree(packet));
    std::printf("{\"graph_metadata_comparisons\":%u,\"workspace_bytes\":%zu,\"fa3_attention_executed\":false}\n",comparisons,sizeof(Workspace));
}
