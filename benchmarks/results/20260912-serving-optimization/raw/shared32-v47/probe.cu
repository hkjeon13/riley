#include "decode_shared16.cuh"
#include "decode_shared16_attention.cuh"
#include "decode_shared32_v47.cuh"
#include "decode_shared32_attention_v47.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#define OK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line %d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;OK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
void exact(const __nv_bfloat16*a,const __nv_bfloat16*b,int valid,int size,const char*kind,int rows){for(int i=0;i<size+16;++i)if(__bfloat16_as_ushort(b[i])!=(i<valid?__bfloat16_as_ushort(a[i]):0x4123)){fprintf(stderr,"%s rows=%d mismatch=%d\n",kind,rows,i);exit(3);}}
template<class F>void timing(const char*kind,int rows,int context,F launch){
 cudaStream_t stream;OK(cudaStreamCreate(&stream));cudaEvent_t a,b;OK(cudaEventCreate(&a));OK(cudaEventCreate(&b));
 for(int pair=0;pair<4;++pair)for(int order=0;order<2;++order){bool candidate=(pair+order)%2;cudaGraph_t graph;cudaGraphExec_t exec;
 OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch(candidate,stream);OK(cudaStreamEndCapture(stream,&graph));OK(cudaGraphInstantiate(&exec,graph,0,0,0));
 for(int i=0;i<40;++i)OK(cudaGraphLaunch(exec,stream));OK(cudaStreamSynchronize(stream));
 OK(cudaEventRecord(a,stream));for(int i=0;i<200;++i)OK(cudaGraphLaunch(exec,stream));OK(cudaEventRecord(b,stream));OK(cudaEventSynchronize(b));float ms;OK(cudaEventElapsedTime(&ms,a,b));printf("{\"kind\":\"%s\",\"rows\":%d,\"context\":%d,\"pair\":%d,\"candidate\":%s,\"us\":%.6f}\n",kind,rows,context,pair,candidate?"true":"false",ms*5);
 OK(cudaGraphExecDestroy(exec));OK(cudaGraphDestroy(graph));}
 OK(cudaEventDestroy(a));OK(cudaEventDestroy(b));OK(cudaStreamDestroy(stream));
}
void live_set(uint32_t*l,int rows){l[0]=rows;l[1]=rows>=1&&rows<=32?min(rows,16):0;l[2]=rows>16&&rows<=32?rows-16:0;}
template<int N,int K,int I,bool Tiled>void projection(bool bench){
 auto*x=alloc<__nv_bfloat16>(32*K);auto*w=alloc<__nv_bfloat16>(N*K);auto*tw=alloc<__nv_bfloat16>(N*K);auto*a=alloc<__nv_bfloat16>(32*N);auto*b=alloc<__nv_bfloat16>(32*N+16);auto*p=alloc<float>(8*32*N);auto*l=alloc<uint32_t>(3);
 auto launch=[&](bool candidate,cudaStream_t stream){if(candidate)enqueue_shared32_projection<N,K,I,Tiled>(stream,x,Tiled?tw:w,b,p,l);else{enqueue_shared16_projection<N,K,I,Tiled>(stream,x,Tiled?tw:w,a,p,l+1);if(l[0]>16)enqueue_shared16_projection<N,K,I,Tiled>(stream,x+16*K,Tiled?tw:w,a+16*N,p,l+2);}};
 for(int seed:{1,17,99}){
 for(int i=0;i<32*K;++i)x[i]=__float2bfloat16_rn(float((i*17+seed)%127-63)/128);
 for(int i=0;i<N*K;++i)w[i]=__float2bfloat16_rn(float((i*13+seed)%251-125)/256);
 for(int n=0;n<N;n+=8)for(int k=0;k<K;k+=16)for(int lane=0;lane<32;++lane)for(int hi=0;hi<2;++hi)for(int z=0;z<2;++z)tw[((n/8)*(K/16)+k/16)*128+hi*64+lane*2+z]=w[(n+lane/4)*K+k+hi*8+2*(lane%4)+z];
 for(int rows=0;rows<=33;++rows){live_set(l,rows);for(int i=0;i<32*N+16;++i)b[i]=__ushort_as_bfloat16(0x4123);if(rows>=1&&rows<=32)launch(false,0);launch(true,0);OK(cudaDeviceSynchronize());exact(a,b,rows>=1&&rows<=32?rows*N:0,32*N,"projection",rows);}
 }
 if(bench){char name[80];snprintf(name,sizeof(name),"projection-%d-%d-%d-%d",N,K,I,Tiled);for(int rows:{4,8,16,24,32}){live_set(l,rows);OK(cudaDeviceSynchronize());timing(name,rows,0,launch);}}
 else printf("projection N=%d K=%d interval=%d seeds=3 active0..33 exact guards=true\n",N,K,I);
 for(void*z:{(void*)x,(void*)w,(void*)tw,(void*)a,(void*)b,(void*)p,(void*)l})OK(cudaFree(z));
}
void gate(bool bench){
 auto*x=alloc<__nv_bfloat16>(32*576);auto*g=alloc<__nv_bfloat16>(1536*576);auto*u=alloc<__nv_bfloat16>(1536*576);auto*a=alloc<__nv_bfloat16>(32*1536);auto*b=alloc<__nv_bfloat16>(32*1536+16);auto*l=alloc<uint32_t>(3);
 for(int i=0;i<32*576;++i)x[i]=__float2bfloat16_rn(float(i*17%127-63)/128);for(int i=0;i<1536*576;++i){g[i]=__float2bfloat16_rn(float(i*13%251-125)/256);u[i]=__float2bfloat16_rn(float(i*7%127-63)/256);}
 auto launch=[&](bool candidate,cudaStream_t stream){if(candidate)shared32_gate_up_swiglu<true><<<192,64,0,stream>>>(x,g,u,b,l);else{shared16_gate_up_swiglu<true><<<192,64,0,stream>>>(x,g,u,a,l+1);if(l[0]>16)shared16_gate_up_swiglu<true><<<192,64,0,stream>>>(x+16*576,g,u,a+16*1536,l+2);}};
 for(int rows=0;rows<=33;++rows){live_set(l,rows);for(int i=0;i<32*1536+16;++i)b[i]=__ushort_as_bfloat16(0x4123);if(rows>=1&&rows<=32)launch(false,0);launch(true,0);OK(cudaDeviceSynchronize());exact(a,b,rows>=1&&rows<=32?rows*1536:0,32*1536,"gate",rows);}
 if(bench)for(int rows:{4,8,16,24,32}){live_set(l,rows);OK(cudaDeviceSynchronize());timing("gate-up-swiglu",rows,0,launch);}else printf("gate active0..33 exact guards=true\n");
 for(void*z:{(void*)x,(void*)g,(void*)u,(void*)a,(void*)b,(void*)l})OK(cudaFree(z));
}
void attention(bool bench){
 auto*q=alloc<__nv_bfloat16>(32*576);auto*k=alloc<__nv_bfloat16>(8192*16*192);auto*v=alloc<__nv_bfloat16>(8192*16*192);auto*a=alloc<__nv_bfloat16>(32*576);auto*b=alloc<__nv_bfloat16>(32*576+16);auto*p=alloc<float>(32*9*4096);auto*shape=alloc<uint32_t>(32*416);auto*l=alloc<uint32_t>(3);
 int counts[16]={1,16,17,127,128,129,2049,4096,2,15,73,255,256,398,1024,4095};
 for(int i=0;i<32*576;++i)q[i]=__float2bfloat16_rn(float(i*7%127-63)/128);for(int i=0;i<8192*16*192;++i){k[i]=__float2bfloat16_rn(float(i*17%251-125)/256);v[i]=__float2bfloat16_rn(float(i*13%127-63)/64);}
 for(int row=0;row<32;++row){shape[row*416+1]=counts[row%16]-1;for(int page=0;page<256;++page)shape[row*416+32+page]=row*256+(page*17+3)%256;}
 auto launch=[&](bool candidate,cudaStream_t stream){if(candidate)riley_shared32_attention::enqueue(stream,q,k,v,b,p,shape,shape+32,l,4096);else{riley_shared16_attention::enqueue(stream,q,k,v,a,p,shape,shape+32,l+1,4096);if(l[0]>16)riley_shared16_attention::enqueue(stream,q+16*576,k,v,a+16*576,p,shape+16*416,shape+16*416+32,l+2,4096);}};
 for(int rows=0;rows<=33;++rows){live_set(l,rows);for(int i=0;i<32*576+16;++i)b[i]=__ushort_as_bfloat16(0x4123);if(rows>=1&&rows<=32)launch(false,0);launch(true,0);OK(cudaDeviceSynchronize());exact(a,b,rows>=1&&rows<=32?rows*576:0,32*576,"attention",rows);}
 if(bench)for(int count:{16,128,398,1024,4096,-1})for(int rows:{4,8,16,24,32}){live_set(l,rows);for(int row=0;row<32;++row)shape[row*416+1]=(count<0?counts[row%16]:count)-1;launch(false,0);launch(true,0);OK(cudaDeviceSynchronize());exact(a,b,rows*576,rows*576-16,"attention timing",rows);timing("attention",rows,count,launch);}else printf("attention active0..33 mixed contexts1..4096 disjoint permuted pages exact guards=true\n");
 for(void*z:{(void*)q,(void*)k,(void*)v,(void*)a,(void*)b,(void*)p,(void*)shape,(void*)l})OK(cudaFree(z));
}
int main(int argc,char**){bool bench=argc>1;projection<576,576,192,false>(bench);projection<192,576,192,false>(bench);projection<576,576,128,false>(bench);projection<1536,576,0,true>(bench);projection<576,1536,320,true>(bench);gate(bench);attention(bench);}
