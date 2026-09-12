from pathlib import Path
r=Path('/tmp/riley-opt-260912/query-tile-v44');s=(r/'probe.cu').read_text();start=s.index(' ck(riley_prefill_query_tile::launch');end=s.index('\nint main()')
body='''
 auto launch=[&](bool candidate){
  if(candidate)ck(riley_prefill_query_tile::launch(0,(__nv_bfloat16*)dq,(__nv_bfloat16*)dk,(__nv_bfloat16*)dv,(__nv_bfloat16*)da,dp,start,rows,context));
  else ck(riley_prefill_shape::launch(0,(__nv_bfloat16*)dq,(__nv_bfloat16*)dk,(__nv_bfloat16*)dv,(__nv_bfloat16*)da,dp,start,rows,context));
 };
 cudaEvent_t begin,finish;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&finish));
 for(int pair=0;pair<4;++pair)for(int order=0;order<2;++order){
  bool candidate=(pair+order)%2;
  for(int i=0;i<20;++i)launch(candidate);ck(cudaDeviceSynchronize());
  ck(cudaEventRecord(begin));for(int i=0;i<100;++i)launch(candidate);ck(cudaEventRecord(finish));ck(cudaEventSynchronize(finish));float ms;ck(cudaEventElapsedTime(&ms,begin,finish));
  printf("{\\"tile\\":%d,\\"rows\\":%d,\\"start\\":%d,\\"pair\\":%d,\\"candidate\\":%s,\\"kernel_us\\":%.6f}\\n",RILEY_QUERY_TILE,rows,start,pair,candidate?"true":"false",ms*10.F);
 }
 ck(cudaEventDestroy(begin));ck(cudaEventDestroy(finish));
 for(void* p:{(void*)dq,(void*)dk,(void*)dv,(void*)da,(void*)db,(void*)dp})ck(cudaFree(p));
}
'''
s=s[:start]+body+'\nint main(){for(int start:{0,128,1024})for(int rows:{16,128,398})run(rows,start);run(1024,3072);}\n';(r/'timing.cu').write_text(s)
