from pathlib import Path
r=Path('/tmp/riley-opt-260912');o=r/'grouped-values-v45';s=(o/'probe.cu').read_text();a=s.index(' for(int rows=0;rows<=17;++rows){',s.index('void attention()'))
b=s.index(' for(void*z:',a)
body='''
 cudaEvent_t begin,finish;OK(cudaEventCreate(&begin));OK(cudaEventCreate(&finish));
 for(int context:{16,128,398,1024,4096,-1})for(int rows:{4,8,16}){
  *live=rows;for(int row=0;row<16;++row)shape[row*416+1]=(context<0?counts[row]:context)-1;
  riley_shared16_attention::enqueue(0,q,k,v,a,scratch,shape,shape+32,live,4096);OK(cudaDeviceSynchronize());
  riley_shared16_grouped::enqueue(0,q,k,v,b,scratch,shape,shape+32,live,4096);OK(cudaDeviceSynchronize());
  for(int i=0;i<rows*576;++i)if(__bfloat16_as_ushort(a[i])!=__bfloat16_as_ushort(b[i])){fprintf(stderr,"timing correctness failure\\n");exit(5);}
  for(int pipeline:{0,1})for(int pair=0;pair<4;++pair)for(int order=0;order<2;++order){
   bool candidate=(pair+order)%2;
   auto launch=[&](){
    if(pipeline){if(candidate)riley_shared16_grouped::enqueue(0,q,k,v,b,scratch,shape,shape+32,live,4096);else riley_shared16_attention::enqueue(0,q,k,v,b,scratch,shape,shape+32,live,4096);}
    else if(candidate)riley_shared16_grouped::values<<<dim3(8,48),32>>>(scratch,v,b,shape,shape+32,live);
    else riley_shared16_attention::values<<<dim3(8,144),32>>>(scratch,v,b,shape,shape+32,live);
    OK(cudaGetLastError());
   };
   for(int i=0;i<40;++i)launch();OK(cudaDeviceSynchronize());
   OK(cudaEventRecord(begin));for(int i=0;i<200;++i)launch();OK(cudaEventRecord(finish));OK(cudaEventSynchronize(finish));float ms;OK(cudaEventElapsedTime(&ms,begin,finish));
   printf("{\\"rows\\":%d,\\"context\\":%d,\\"pipeline\\":%d,\\"pair\\":%d,\\"candidate\\":%s,\\"us\\":%.6f}\\n",rows,context,pipeline,pair,candidate?"true":"false",ms*5.F);
  }
 }
 OK(cudaEventDestroy(begin));OK(cudaEventDestroy(finish));
'''
s=s[:a]+body+s[b:];s=s[:s.index('int main()')]+'int main(){attention();}\n';(o/'timing.cu').write_text(s)
