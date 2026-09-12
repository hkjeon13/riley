// Admission-only diagnostic. Compile the unchanged pinned gemm.cu in this TU
// to inspect its actual opaque algorithm. No GEMM or tensor allocation occurs.
#include "gemm.cu"
#include <iostream>
#include <stdexcept>
#include <array>

namespace diagnostic {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Layout {
 cublasLtMatrixLayout_t p=nullptr;
 Layout(uint64_t rows,uint64_t cols,int64_t ld,int batch,int64_t stride){
  check(cublasLtMatrixLayoutCreate(&p,CUDA_R_16BF,rows,cols,ld)==CUBLAS_STATUS_SUCCESS,"layout create");
  if(cublasLtMatrixLayoutSetAttribute(p,CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(batch))!=CUBLAS_STATUS_SUCCESS ||
     cublasLtMatrixLayoutSetAttribute(p,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&stride,sizeof(stride))!=CUBLAS_STATUS_SUCCESS){cublasLtMatrixLayoutDestroy(p);p=nullptr;throw std::runtime_error("batch attributes");}
 }
 ~Layout(){if(p)cublasLtMatrixLayoutDestroy(p);}
};
struct Owner {
 RileyCudaContext* context=nullptr;RileyCudaGemmPlan* plan=nullptr;
 ~Owner(){RileyCudaErrorInfo e{};e.struct_size=sizeof(e);if(plan)riley_cuda_gemm_plan_close(&plan,&e);if(context)riley_cuda_context_close(&context,&e);}
};
}
int main(){using namespace diagnostic;try{
 Owner o;RileyCudaErrorInfo e{};e.struct_size=sizeof(e);
 check(riley_cuda_context_create(0,&o.context,&e)==0,"context create");
 CUuuid uuid{};check(cuDeviceGetUuid(&uuid,o.context->device)==CUDA_SUCCESS,"uuid");
 std::cout<<"{\"kind\":\"device\",\"uuid_hex\":\"";const char* hex="0123456789abcdef";for(auto c:uuid.bytes){auto b=static_cast<unsigned char>(c);std::cout<<hex[b>>4]<<hex[b&15];}std::cout<<"\"}\n";
 const std::array<std::array<uint64_t,3>,5> shapes={{{960,576,74},{3072,576,74},{576,576,74},{576,1536,75},{49152,576,89}}};
 unsigned cases=0;
 for(unsigned shape=0;shape<shapes.size();++shape){auto n=shapes[shape][0],k=shapes[shape][1];
  RileyCudaGemmConfig c{};c.struct_size=sizeof(c);c.m=1;c.n=n;c.k=k;c.input_dtype=c.weight_dtype=c.output_dtype=RILEY_CUDA_DTYPE_BF16;c.accumulator_dtype=RILEY_CUDA_DTYPE_F32;
  c.input_transpose=RILEY_CUDA_GEMM_TRANSPOSE_N;c.weight_transpose=RILEY_CUDA_GEMM_TRANSPOSE_T;c.input_layout=c.weight_layout=c.output_layout=RILEY_CUDA_GEMM_LAYOUT_ROW_MAJOR;c.deterministic=RILEY_CUDA_GEMM_DETERMINISTIC_REQUIRED;c.max_workspace_bytes=16u*1024u*1024u;
  check(riley_cuda_gemm_plan_create(o.context,&c,&o.plan,&e)==0,"M1 plan create");
  auto original=o.plan->algorithm;auto info=o.plan->algorithm_info;
  check(info.algorithm_id==13 && info.custom_option==shapes[shape][2] && info.numerical_implementation_flags==131585 && info.workspace_bytes==0 && info.m==1 && info.n==n && info.k==k && info.cublaslt_version==130101,"M1 anchor differs");
  check(cuCtxPushCurrent(o.context->context)==CUDA_SUCCESS,"push context");
  try {
   int support=-1;size_t written=0;
   auto status=cublasLtMatmulAlgoCapGetAttribute(&original,CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT,&support,sizeof(support),&written);
   check(status==CUBLAS_STATUS_SUCCESS && written==sizeof(support),"strided capability query");
   uint32_t alignment[4]{};cublasLtMatmulAlgoCapAttributes_t attrs[]={CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES,CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES,CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES};
   for(int i=0;i<4;++i){written=0;check(cublasLtMatmulAlgoCapGetAttribute(&original,attrs[i],&alignment[i],sizeof(alignment[i]),&written)==CUBLAS_STATUS_SUCCESS && written==sizeof(alignment[i]),"alignment query");}
   for(int padded=0;padded<2;++padded)for(int batch:{2,4}){
    int64_t xs=padded ? ((k+127)/128)*128:k,ys=padded ? ((n+127)/128)*128:n;
    Layout a(k,n,k,batch,0),b(k,1,k,batch,xs),cc(n,1,n,batch,ys),d(n,1,n,batch,ys);
    cublasLtMatmulHeuristicResult_t result{};
    auto algo=original;auto checked=cublasLtMatmulAlgoCheck(o.plan->handle,o.plan->operation,a.p,b.p,cc.p,d.p,&algo,&result);
    check(std::memcmp(&original,&o.plan->algorithm,sizeof(original))==0 && std::memcmp(&info,&o.plan->algorithm_info,sizeof(info))==0,"anchor mutated");
    std::cout<<"{\"kind\":\"case\",\"shape\":"<<shape<<",\"n\":"<<n<<",\"k\":"<<k<<",\"m\":1,\"batch\":"<<batch<<",\"padded\":"<<padded<<",\"input_stride_bf16\":"<<xs<<",\"output_stride_bf16\":"<<ys<<",\"strided_capability\":"<<support<<",\"alignments\":["<<alignment[0]<<','<<alignment[1]<<','<<alignment[2]<<','<<alignment[3]<<"],\"algo_id\":"<<info.algorithm_id<<",\"custom\":"<<info.custom_option<<",\"check_status\":"<<checked<<",\"result_state\":"<<result.state<<",\"workspace_bytes\":"<<result.workspaceSize<<",\"waves_count\":"<<result.wavesCount<<",\"anchor_unchanged\":true}\n";
    ++cases;
   }
  }catch(...){CUcontext popped;cuCtxPopCurrent(&popped);throw;}
  CUcontext popped=nullptr;check(cuCtxPopCurrent(&popped)==CUDA_SUCCESS && popped==o.context->context,"pop context");
  check(riley_cuda_gemm_plan_close(&o.plan,&e)==0,"plan close");
 }
 RileyCudaAllocationStats stats{};stats.struct_size=sizeof(stats);check(riley_cuda_context_allocation_stats(o.context,&stats,&e)==0,"allocation stats");
 check(!stats.device_live_allocations && !stats.pinned_host_live_allocations,"unexpected live allocations");check(riley_cuda_context_close(&o.context,&e)==0,"context close");
 std::cout<<"{\"kind\":\"summary\",\"cases\":"<<cases<<",\"all_plans_closed\":true,\"context_closed\":true,\"gemm_executed\":false,\"performance_measured\":false}\n";
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
