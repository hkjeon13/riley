#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring>
#define CU(x) do{auto probe_status=(x);if(probe_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(probe_status));}while(0)
#define BL(x) do{auto probe_status=(x);if(probe_status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cublas status "+std::to_string(probe_status));}while(0)
struct Plan{
 cublasLtHandle_t h; cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t w,x,y; cublasLtMatmulAlgo_t algo; int n,k,parts,group,batch;
 Plan(int N,int K,int B,int P,int G):n(N),k(K),parts(P),group(G),batch(B){
 BL(cublasLtCreate(&h)); BL(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));auto trans=CUBLAS_OP_T;BL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
 BL(cublasLtMatrixLayoutCreate(&w,CUDA_R_16BF,k,n/parts,k));BL(cublasLtMatrixLayoutCreate(&x,CUDA_R_16BF,k,1,k));BL(cublasLtMatrixLayoutCreate(&y,CUDA_R_16BF,n/parts,1,n/parts));
 cublasLtMatrixLayout_t ls[]={w,x,y};int64_t strides[]={0,k,n};
 for(int i=0;i<3;i++){BL(cublasLtMatrixLayoutSetAttribute(ls[i],CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&group,sizeof(group)));BL(cublasLtMatrixLayoutSetAttribute(ls[i],CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&strides[i],sizeof(int64_t)));}
 BL(cublasLtMatmulAlgoInit(h,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,13,&algo));
 uint32_t custom=n==49152?89:k==1536?75:74,split=1,zero=0;
 BL(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,&custom,sizeof(custom)));BL(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split)));BL(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&zero,sizeof(zero)));
 cublasLtMatmulHeuristicResult_t checked{};BL(cublasLtMatmulAlgoCheck(h,op,w,x,y,y,&algo,&checked));if(checked.state!=CUBLAS_STATUS_SUCCESS||checked.workspaceSize)throw std::runtime_error("unsupported geometry");
 }
 void run(cudaStream_t s,__nv_bfloat16* W,__nv_bfloat16* X,__nv_bfloat16* Y){float a=1,b=0;for(int row=0;row<batch;row+=group)for(int p=0;p<parts;p++)BL(cublasLtMatmul(h,op,&a,W+size_t(p)*(n/parts)*k,w,X+row*k,x,&b,Y+row*n+p*(n/parts),y,Y+row*n+p*(n/parts),y,&algo,nullptr,0,s));}
 ~Plan(){cublasLtMatrixLayoutDestroy(w);cublasLtMatrixLayoutDestroy(x);cublasLtMatrixLayoutDestroy(y);cublasLtMatmulDescDestroy(op);cublasLtDestroy(h);}
};
struct Dense {
 cublasLtHandle_t h; cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t w,x,y;cublasLtMatmulHeuristicResult_t a{};void* workspace=nullptr;
 Dense(int n,int k){BL(cublasLtCreate(&h));BL(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));auto tr=CUBLAS_OP_T;BL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&tr,sizeof(tr)));BL(cublasLtMatrixLayoutCreate(&w,CUDA_R_16BF,k,n,k));BL(cublasLtMatrixLayoutCreate(&x,CUDA_R_16BF,k,8,k));BL(cublasLtMatrixLayoutCreate(&y,CUDA_R_16BF,n,8,n));cublasLtMatmulPreference_t pref;BL(cublasLtMatmulPreferenceCreate(&pref));size_t budget=32*1024*1024;BL(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&budget,sizeof(budget)));int count=0;BL(cublasLtMatmulAlgoGetHeuristic(h,op,w,x,y,y,pref,1,&a,&count));BL(cublasLtMatmulPreferenceDestroy(pref));if(count!=1||a.state!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("dense algorithm missing");uint32_t split=1,red=0;BL(cublasLtMatmulAlgoConfigSetAttribute(&a.algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split)));BL(cublasLtMatmulAlgoConfigSetAttribute(&a.algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&red,sizeof(red)));cublasLtMatmulHeuristicResult_t checked{};BL(cublasLtMatmulAlgoCheck(h,op,w,x,y,y,&a.algo,&checked));if(checked.state!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("no-split unsupported");a.workspaceSize=checked.workspaceSize;if(a.workspaceSize)CU(cudaMalloc(&workspace,a.workspaceSize));}
 void run(cudaStream_t s,__nv_bfloat16*W,__nv_bfloat16*X,__nv_bfloat16*Y){float one=1,zero=0;BL(cublasLtMatmul(h,op,&one,W,w,X,x,&zero,Y,y,Y,y,&a.algo,workspace,a.workspaceSize,s));}
 ~Dense(){if(workspace)cudaFree(workspace);cublasLtMatrixLayoutDestroy(w);cublasLtMatrixLayoutDestroy(x);cublasLtMatrixLayoutDestroy(y);cublasLtMatmulDescDestroy(op);cublasLtDestroy(h);}
};
std::vector<__nv_bfloat16> read(const std::string&p,size_t n){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=std::streamoff(n*2))throw std::runtime_error("fixture size");f.seekg(0);std::vector<__nv_bfloat16> v(n);f.read((char*)v.data(),n*2);if(!f)throw std::runtime_error("fixture read");return v;}
struct Stats{double err2=0,ref2=0,maxerr=0;size_t bad=0;void add(double actual,double ref,double abs_sum,int k){double d=std::abs(actual-ref);err2+=d*d;ref2+=ref*ref;maxerr=std::max(d,maxerr);bad+=!std::isfinite(actual)||d>2.0*k*std::ldexp(1.0,-24)*abs_sum+std::abs(ref)/256+1e-30;}};
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("case index required");std::ifstream index(argv[1]);if(!index)throw std::runtime_error("index missing");cudaStream_t s;CU(cudaStreamCreate(&s));std::string line;int count=0;while(std::getline(index,line)){std::stringstream in(line);std::string id,ns,ks,wp,xp;std::getline(in,id,'\t');std::getline(in,ns,'\t');std::getline(in,ks,'\t');std::getline(in,wp,'\t');std::getline(in,xp,'\t');int n=std::stoi(ns),k=std::stoi(ks);auto hw=read(wp,size_t(n)*k),hx=read(xp,8*k);std::vector<__nv_bfloat16> old(8*n),fresh(8*n);__nv_bfloat16 *w,*x,*y;CU(cudaMalloc(&w,hw.size()*2));CU(cudaMalloc(&x,hx.size()*2));CU(cudaMalloc(&y,old.size()*2));CU(cudaMemcpy(w,hw.data(),hw.size()*2,cudaMemcpyHostToDevice));CU(cudaMemcpy(x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
 Plan base(n,k,8,n==49152?16:1,8);Dense dense(n,k);float times[2];for(int mode=0;mode<2;mode++){auto launch=[&](){if(mode)dense.run(s,w,x,y);else base.run(s,w,x,y);};launch();CU(cudaStreamSynchronize(s));CU(cudaMemcpy((mode?fresh:old).data(),y,old.size()*2,cudaMemcpyDeviceToHost));CU(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));launch();cudaGraph_t g;CU(cudaStreamEndCapture(s,&g));cudaGraphExec_t e;CU(cudaGraphInstantiate(&e,g,0));cudaEvent_t a,b;CU(cudaEventCreate(&a));CU(cudaEventCreate(&b));for(int i=0;i<5;i++)CU(cudaGraphLaunch(e,s));CU(cudaEventRecord(a,s));for(int i=0;i<30;i++)CU(cudaGraphLaunch(e,s));CU(cudaEventRecord(b,s));CU(cudaEventSynchronize(b));CU(cudaEventElapsedTime(&times[mode],a,b));times[mode]*=1000.0f/30;CU(cudaEventDestroy(a));CU(cudaEventDestroy(b));CU(cudaGraphExecDestroy(e));CU(cudaGraphDestroy(g));}
 Stats a,b;size_t diff=0;for(int r=0;r<8;r++)for(int j=0;j<n;j++){double ref=0,as=0;for(int t=0;t<k;t++){double p=double(__bfloat162float(hw[size_t(j)*k+t]))*__bfloat162float(hx[r*k+t]);ref+=p;as+=std::abs(p);}size_t at=r*n+j;a.add(__bfloat162float(old[at]),ref,as,k);b.add(__bfloat162float(fresh[at]),ref,as,k);diff+=reinterpret_cast<uint16_t*>(old.data())[at]!=reinterpret_cast<uint16_t*>(fresh.data())[at];}
 std::cout<<"{\"case\":\""<<id<<"\",\"n\":"<<n<<",\"k\":"<<k<<",\"outputs\":"<<8*n<<",\"bf16_differences\":"<<diff<<",\"baseline_rel_l2\":"<<std::sqrt(a.err2/a.ref2)<<",\"dense_rel_l2\":"<<std::sqrt(b.err2/b.ref2)<<",\"baseline_max_abs\":"<<a.maxerr<<",\"dense_max_abs\":"<<b.maxerr<<",\"baseline_bound_failures\":"<<a.bad<<",\"dense_bound_failures\":"<<b.bad<<",\"baseline_graph_us\":"<<times[0]<<",\"dense_graph_us\":"<<times[1]<<",\"workspace_bytes\":"<<dense.a.workspaceSize<<"}"<<std::endl;
 CU(cudaFree(w));CU(cudaFree(x));CU(cudaFree(y));count++;}CU(cudaStreamDestroy(s));if(count!=121)throw std::runtime_error("missing cases");return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
