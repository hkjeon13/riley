#include "fa3_contract.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <climits>
#include <cstdlib>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"failed line %d: %s\n",__LINE__,#x); std::exit(1); } } while(0)
int main() {
    int32_t q[]={0,1,18}, k[]={31,32}, pi[]={0,2,4}, pages[]={7,1,4,2};
    RileyFa3Spec s{1,0,2,18,8,q,k,pi,pages,4};
    uint64_t bytes=0;
    CHECK(riley_fa3_validate(&s,&bytes)==0 && bytes%256==0);
    riley_fa3::Layout l;
    CHECK(riley_fa3::make_layout(&s,l)==0);
    CHECK(l.max_k==32 && l.max_q==17 && l.columns==2);
    CHECK(l.metadata[0]==7 && l.metadata[1]==1 && l.metadata[2]==4 && l.metadata[3]==2);
    CHECK(l.metadata[l.qptr/4+2]==18 && l.metadata[l.lengths/4]==31);
    CHECK(l.bytes>=l.lse+18*9*4 && l.lse>=l.scheduler+(4*l.rounded_b+1)*4);
    s.mode=1; CHECK(riley_fa3_validate(&s,&bytes)==RILEY_FA3_INVALID && bytes==0); s.mode=0;
    pages[1]=8; CHECK(riley_fa3_validate(&s,&bytes)==RILEY_FA3_INVALID); pages[1]=1;
    pi[1]=INT_MIN; CHECK(riley_fa3_validate(&s,&bytes)==RILEY_FA3_INVALID); pi[1]=2;
    q[1]=INT_MAX; CHECK(riley_fa3_validate(&s,&bytes)==RILEY_FA3_INVALID); q[1]=1;
    k[0]=4097; CHECK(riley_fa3_validate(&s,&bytes)==RILEY_FA3_INVALID); k[0]=31;
    pi[2]=3; CHECK(riley_fa3_validate(&s,&bytes)==RILEY_FA3_INVALID); pi[2]=4;
    CHECK(riley_fa3_validate(nullptr,&bytes)==RILEY_FA3_INVALID);
    // Full batch, longest context, noncontiguous page IDs and packed Q boundary.
    int32_t bq[33], bk[32], bp[33], ids[8192];
    for(int i=0;i<=32;++i) { bq[i]=i*32; bp[i]=i*256; }
    for(int i=0;i<32;++i) bk[i]=4096;
    for(int i=0;i<8192;++i) ids[i]=(i*17)%8192;
    RileyFa3Spec maximum{1,0,32,1024,8192,bq,bk,bp,ids,8192};
    CHECK(riley_fa3_validate(&maximum,&bytes)==0);
    CHECK(riley_fa3::make_layout(&maximum,l)==0 && l.columns==256);
    CHECK(l.metadata[8191]==ids[8191] && l.metadata[l.qptr/4+32]==1024);
    maximum.total_q=1025; CHECK(riley_fa3_validate(&maximum,&bytes)==RILEY_FA3_INVALID);
    CHECK(riley_fa3_enqueue(nullptr)==RILEY_FA3_INVALID);
    CHECK(riley_fa3_destroy(nullptr)==RILEY_FA3_INVALID);
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess);
    RileyFa3Buffers buffers{}; RileyFa3Plan *plan=nullptr;
    int device=0; cudaDeviceProp props{};
    CHECK(cudaGetDevice(&device)==cudaSuccess);
    CHECK(cudaGetDeviceProperties(&props,device)==cudaSuccess);
    CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
    CHECK(riley_fa3_create(&s,&buffers,stream,&plan)==RILEY_FA3_CAPTURE && plan==nullptr);
    cudaGraph_t graph;
    CHECK(cudaStreamEndCapture(stream,&graph)==cudaSuccess);
    CHECK(cudaGraphDestroy(graph)==cudaSuccess);
    if (props.major!=9 || props.minor!=0) {
        CHECK(riley_fa3_create(&s,&buffers,stream,&plan)==RILEY_FA3_UNSUPPORTED && plan==nullptr);
        void *workspace=nullptr;
        CHECK(cudaMalloc(&workspace,riley_fa3_model_workspace_bytes())==cudaSuccess);
        CHECK(riley_fa3_model_schedule(stream,workspace,riley_fa3_model_workspace_bytes(),1024,4096)==cudaErrorNotSupported);
        CHECK(cudaFree(workspace)==cudaSuccess);
        std::puts("{\"host_layout\":\"pass\",\"unsupported_device_rejection\":\"pass\",\"hopper_runtime\":\"skip_no_hopper\"}");
    } else {
        // This is a contract probe, not a substitute for the Hopper numerical suite.
        CHECK(riley_fa3_create(&s,&buffers,stream,&plan)==RILEY_FA3_INVALID && plan==nullptr);
        std::puts("{\"host_layout\":\"pass\",\"null_buffer_rejection\":\"pass\",\"hopper_attention\":\"not_run\"}");
    }
    CHECK(cudaStreamDestroy(stream)==cudaSuccess);
}
