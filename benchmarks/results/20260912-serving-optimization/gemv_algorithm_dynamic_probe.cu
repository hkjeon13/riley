#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#define CU(x) do{auto probe_status=(x);if(probe_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(probe_status));}while(0)
#define BL(x) do{auto probe_status=(x);if(probe_status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cublas status "+std::to_string(probe_status));}while(0)
struct Plan{
 cublasLtHandle_t h; cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t w,x,y; cublasLtMatmulAlgo_t algo; int n,k,parts,group,batch;
 Plan(int N,int K,int B,int P,int G,int C=-1):n(N),k(K),parts(P),group(G),batch(B){
 BL(cublasLtCreate(&h)); BL(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));auto trans=CUBLAS_OP_T;BL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
 BL(cublasLtMatrixLayoutCreate(&w,CUDA_R_16BF,k,n/parts,k));BL(cublasLtMatrixLayoutCreate(&x,CUDA_R_16BF,k,1,k));BL(cublasLtMatrixLayoutCreate(&y,CUDA_R_16BF,n/parts,1,n/parts));
 cublasLtMatrixLayout_t ls[]={w,x,y};int64_t strides[]={0,k,n};
 for(int i=0;i<3;i++){BL(cublasLtMatrixLayoutSetAttribute(ls[i],CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&group,sizeof(group)));BL(cublasLtMatrixLayoutSetAttribute(ls[i],CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&strides[i],sizeof(int64_t)));}
 BL(cublasLtMatmulAlgoInit(h,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,13,&algo));
 uint32_t custom=C>=0?C:n==49152?89:k==1536?75:74,split=1,zero=0;
 BL(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,&custom,sizeof(custom)));BL(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split)));BL(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&zero,sizeof(zero)));
 cublasLtMatmulHeuristicResult_t checked{};BL(cublasLtMatmulAlgoCheck(h,op,w,x,y,y,&algo,&checked));if(checked.state!=CUBLAS_STATUS_SUCCESS||checked.workspaceSize)throw std::runtime_error("unsupported geometry");
 }
 void run(cudaStream_t s,__nv_bfloat16* W,__nv_bfloat16* X,__nv_bfloat16* Y){float a=1,b=0;for(int row=0;row<batch;row+=group)for(int p=0;p<parts;p++)BL(cublasLtMatmul(h,op,&a,W+size_t(p)*(n/parts)*k,w,X+row*k,x,&b,Y+row*n+p*(n/parts),y,Y+row*n+p*(n/parts),y,&algo,nullptr,0,s));}
 ~Plan(){cublasLtMatrixLayoutDestroy(w);cublasLtMatrixLayoutDestroy(x);cublasLtMatrixLayoutDestroy(y);cublasLtMatmulDescDestroy(op);cublasLtDestroy(h);}
};
int main(){try{cudaStream_t s;CU(cudaStreamCreate(&s));for(auto shape:std::vector<std::pair<int,int>>{{960,576},{3072,576},{576,576},{576,1536},{49152,576}}){auto[n,k]=shape;for(int batch:{4,8}){
 std::vector<__nv_bfloat16> hw(size_t(n)*k),hx(batch*k),ref(batch*n),got(batch*n);__nv_bfloat16 *w,*x,*y;CU(cudaMalloc(&w,hw.size()*2));CU(cudaMalloc(&x,hx.size()*2));CU(cudaMalloc(&y,ref.size()*2));
 for(int pattern=0;pattern<4;pattern++){
 auto fill=[&](auto& v,uint32_t seed){for(size_t i=0;i<v.size();i++){uint32_t h=uint32_t(i)^seed;h^=h>>16;h*=0x7feb352d;h^=h>>15;h*=0x846ca68b;h^=h>>16;uint16_t bits=uint16_t((h&0x8000)|((118+(h>>16)%14)<<7)|(h&127));reinterpret_cast<uint16_t*>(v.data())[i]=bits;}};
 fill(hw,pattern*1777+773);fill(hx,pattern*991+17);
 CU(cudaMemcpy(w,hw.data(),hw.size()*2,cudaMemcpyHostToDevice));CU(cudaMemcpy(x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));{Plan base(n,k,batch,1,1);base.run(s,w,x,y);CU(cudaStreamSynchronize(s));CU(cudaMemcpy(ref.data(),y,ref.size()*2,cudaMemcpyDeviceToHost));}
 for(int custom:{74,75,89})for(int parts:{1,2,16})for(int group:{batch}){if(group>batch||n%parts)continue;Plan plan(n,k,batch,parts,group,custom);CU(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));plan.run(s,w,x,y);cudaGraph_t g;CU(cudaStreamEndCapture(s,&g));cudaGraphExec_t e;CU(cudaGraphInstantiate(&e,g,0));CU(cudaGraphLaunch(e,s));CU(cudaStreamSynchronize(s));CU(cudaMemcpy(got.data(),y,got.size()*2,cudaMemcpyDeviceToHost));size_t diff=0;for(size_t i=0;i<got.size();i++)diff+=reinterpret_cast<uint16_t*>(got.data())[i]!=reinterpret_cast<uint16_t*>(ref.data())[i];
 cudaEvent_t a,b;CU(cudaEventCreate(&a));CU(cudaEventCreate(&b));for(int i=0;i<5;i++)CU(cudaGraphLaunch(e,s));CU(cudaEventRecord(a,s));for(int i=0;i<30;i++)CU(cudaGraphLaunch(e,s));CU(cudaEventRecord(b,s));CU(cudaEventSynchronize(b));float ms;CU(cudaEventElapsedTime(&ms,a,b));std::cout<<"{\"custom\":"<<custom<<",\"n\":"<<n<<",\"k\":"<<k<<",\"batch\":"<<batch<<",\"pattern\":"<<pattern<<",\"parts\":"<<parts<<",\"group\":"<<group<<",\"mismatches\":"<<diff<<",\"graph_us\":"<<ms*1000/30<<"}"<<std::endl;CU(cudaEventDestroy(a));CU(cudaEventDestroy(b));CU(cudaGraphExecDestroy(e));CU(cudaGraphDestroy(g));
 }
 }CU(cudaFree(w));CU(cudaFree(x));CU(cudaFree(y));} }CU(cudaStreamDestroy(s));return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
