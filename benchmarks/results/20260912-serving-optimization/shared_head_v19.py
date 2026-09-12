from pathlib import Path
p=Path('/tmp/full_shared_model_v17.cu');s=p.read_text().replace('#include "decode_shared_model.cuh"','#include "decode_shared_model.cuh"\n#include "decode_shared_result.cuh"\n#include <cublasLt.h>')
a=s.index('int main(')
helper='''
void cb(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cublas status %d\\n",int(s));exit(12);}}
struct Head{
 cublasLtHandle_t handle; cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t w,x,y;cublasLtMatmulAlgo_t algo;
 Head(int rows){cb(cublasLtCreate(&handle));cb(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));cublasOperation_t trans=CUBLAS_OP_T;cb(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
 cb(cublasLtMatrixLayoutCreate(&w,CUDA_R_16BF,576,49152,576));cb(cublasLtMatrixLayoutCreate(&x,CUDA_R_16BF,576,rows,576));cb(cublasLtMatrixLayoutCreate(&y,CUDA_R_16BF,49152,rows,49152));
 cublasLtMatmulPreference_t pref;cb(cublasLtMatmulPreferenceCreate(&pref));size_t workspace=0;cb(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace)));cublasLtMatmulHeuristicResult_t choices[32];int count=0;cb(cublasLtMatmulAlgoGetHeuristic(handle,op,w,x,y,y,pref,32,choices,&count));bool found=false;
 for(int i=0;i<count;++i){uint32_t split=0,reduce=0;size_t size;cb(cublasLtMatmulAlgoConfigGetAttribute(&choices[i].algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split),&size));cb(cublasLtMatmulAlgoConfigGetAttribute(&choices[i].algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&reduce,sizeof(reduce),&size));if(choices[i].state==CUBLAS_STATUS_SUCCESS&&choices[i].workspaceSize==0&&split<=1&&reduce==0){algo=choices[i].algo;found=true;break;}}
 if(!found)exit(13);int id;size_t size;cb(cublasLtMatmulAlgoConfigGetAttribute(&algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&size));printf("head rows=%d algorithm=%d workspace=0\\n",rows,id);cb(cublasLtMatmulPreferenceDestroy(pref));
 }
 void run(cudaStream_t stream,const void* weight,const void* input,void* output){float alpha=1,beta=0;cb(cublasLtMatmul(handle,op,&alpha,weight,w,input,x,&beta,output,y,output,y,&algo,nullptr,0,stream));}
 ~Head(){cublasLtMatrixLayoutDestroy(w);cublasLtMatrixLayoutDestroy(x);cublasLtMatrixLayoutDestroy(y);cublasLtMatmulDescDestroy(op);cublasLtDestroy(handle);}
};
'''
s=s[:a]+helper+s[a:]
s=s.replace(' cudaStream_t stream;ck(cudaStreamCreate(&stream));',' cudaStream_t stream;ck(cudaStreamCreate(&stream));\n Head single(1),shared(8);void *logits,*records;alloc(&logits,8*98304);alloc(&records,riley_shared_result::batch_bytes);')
s=s.replace(' ck(cudaStreamEndCapture(stream,&graph));',' shared.run(stream,weights[2],scratch[1],logits);ck(riley_shared_result::enqueue(stream,metadata,(const __nv_bfloat16*)logits,status,records));\n ck(cudaStreamEndCapture(stream,&graph));')
s=s.replace('reset();std::vector<uint8_t> expected(active*1152),actual(active*1152);','reset();std::vector<uint8_t> expected(active*1152),actual(active*1152),expected_logits(active*98304),actual_records(riley_shared_result::batch_bytes);')
s=s.replace('ck(cudaMemcpy(expected.data()+request*1152,selected,1152,cudaMemcpyDeviceToHost));','ck(cudaMemcpy(expected.data()+request*1152,selected,1152,cudaMemcpyDeviceToHost));\n   single.run(stream,weights[2],selected,logits);ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(expected_logits.data()+request*98304,logits,98304,cudaMemcpyDeviceToHost));')
s=s.replace('  initial_k=actual_k;initial_v=actual_v;','''  ck(cudaMemcpy(actual_records.data(),records,actual_records.size(),cudaMemcpyDeviceToHost));
  for(int request=0;request<active;++request){auto* record=actual_records.data()+request*98432;auto* expected_row=expected_logits.data()+request*98304;
   if(std::memcmp(record+128,expected_row,98304)){fprintf(stderr,"head logits differ active=%d step=%d row=%d\\n",active,step,request);exit(15);}
   uint32_t token=0;float maximum=-INFINITY;for(uint32_t i=0;i<49152;++i){uint16_t bits;std::memcpy(&bits,expected_row+2*i,2);uint32_t fbits=uint32_t(bits)<<16;float value;std::memcpy(&value,&fbits,4);if(value>maximum){maximum=value;token=i;}}
   auto* words=reinterpret_cast<const uint32_t*>(record);if(words[0]||words[1]!=1||words[2]!=token||words[3])exit(16);
  }
  for(size_t i=active*98432;i<actual_records.size();++i)if(actual_records[i])exit(17);
  initial_k=actual_k;initial_v=actual_v;''')
s=s.replace('hidden_exact=true full_kv_exact=true graph_replay=true','hidden_exact=true full_kv_exact=true logits_exact=true result_argmax_exact=true inactive_records_zero=true graph_replay=true')
Path('/tmp/full_shared_head_v19.cu').write_text(s)
