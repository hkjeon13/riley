from pathlib import Path
p=Path('/tmp/riley-opt-260912/query-tile-v44/probe.cu');s=p.read_text()
marker='for(int i=0;i<576;++i)if(a[row*576+i]!=b[i])exit(6);'
extra='''
  // Keep the full query tile active while poisoning a suffix: finite earlier
  // queries must remain exact even when later queries legitimately see NaNs.
  ck(riley_prefill_query_tile::launch(0,(__nv_bfloat16*)dq,(__nv_bfloat16*)dk,(__nv_bfloat16*)dv,(__nv_bfloat16*)db,dp,start,rows,context));ck(cudaDeviceSynchronize());
  std::vector<unsigned short> mixed(a.size()),reference(a.size());
  ck(cudaMemcpy(mixed.data(),db,mixed.size()*2,cudaMemcpyDeviceToHost));
  for(int i=0;i<(row+1)*576;++i)if(mixed[i]!=a[i]){fprintf(stderr,"mixed causal prefix mismatch rows=%d start=%d poison_after=%d at=%d\\n",rows,start,row,i);exit(7);}
  oracle::attention<<<dim3(rows,3),96>>>((__nv_bfloat16*)dq,(__nv_bfloat16*)dk,(__nv_bfloat16*)dv,(__nv_bfloat16*)da,rows,end,nullptr,dp);ck(cudaDeviceSynchronize());
  ck(cudaMemcpy(reference.data(),da,reference.size()*2,cudaMemcpyDeviceToHost));
  for(size_t i=0;i<mixed.size();++i)if(mixed[i]!=reference[i]&&!(std::isnan(bf(mixed[i]))&&std::isnan(bf(reference[i])))){fprintf(stderr,"mixed causal oracle mismatch at=%zu\\n",i);exit(8);}
'''
assert marker in s;s=s.replace(marker,marker+extra);p.write_text(s)
