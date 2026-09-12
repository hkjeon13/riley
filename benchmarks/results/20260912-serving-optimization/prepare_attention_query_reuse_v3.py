from pathlib import Path
s=Path('/tmp/attention_query_reuse_probe_v2.cu').read_text()
a=s.index('template<int R>');b=s.index('#include <vector>',a);k=s[a:b]
k=k.replace(' for(int token=0;token<end;token+=8){',' float mx=-CUDART_INF_F;\n for(int token=0;token<end;token+=8){',1)
k=k.replace('  if(g<R)for(int j=0;j<2;++j)if(token+2*t+j<count)scores[warp][qr][token+2*t+j]=d[j]*.125F;','  for(int j=0;j<2;++j)if(token+2*t+j<count){float score=d[j]*.125F;mx=fmaxf(mx,score);if(g<R)scores[warp][qr][token+2*t+j]=score;}')
k=k.replace(' float mx=-CUDART_INF_F;\n for(int i=t;i<count;i+=4)mx=fmaxf(mx,scores[warp][qr][i]);\n','')
x=k.index(' // All score maxima');y=k.index(' den+=__shfl',x)
k=k[:x]+''' float den=0.;
 #pragma unroll 1
 for(int j=0;j<16;++j){
  int i=2*t+j*8;
  float p0=i<count?exponential(scores[warp][qr][i],mx):0.;
  float p1=i+1<count?exponential(scores[warp][qr][i+1],mx):0.;
  if(i<count)den+=p0;if(i+1<count)den+=p1;
  if(g<R)*reinterpret_cast<uint32_t*>(&probs[warp][qr][i])=pair(__float2bfloat16_rn(p0),__float2bfloat16_rn(p1));
 }
 __syncwarp();
'''+k[y:]
s=s[:a]+k+s[b:]
s=s.replace('for(int rows:{4,8})','for(int rows:{2,4,8})').replace('if(rows==4)attention_queries<4>', 'if(rows==2)attention_queries<2><<<dim3(64,3),96>>>(dq,dk,dv,db,dt);else if(rows==4)attention_queries<4>')
s=s.replace('for(int variant:{0,4,8,8,4,0})','for(int variant:{0,2,4,8,8,4,2,0})').replace('else if(variant==4)attention_queries<4>', 'else if(variant==2)attention_queries<2><<<dim3(64,3),96>>>(dq,dk,dv,db,dt);else if(variant==4)attention_queries<4>')
Path('/tmp/attention_query_reuse_probe_v3.cu').write_text(s)
