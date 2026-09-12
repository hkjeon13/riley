from pathlib import Path
s=Path('/tmp/attention_query_reuse_probe_v3.cu').read_text();a=s.index('template<int R>');b=s.index('#include <vector>',a);k=s[a:b]
k=k.replace('template<int R>','template<int R,int S=2>').replace(' int qr=g%R',' int head=warp%3;\n int qr=g%R').replace('qh=kvh*3+warp','qh=kvh*3+head').replace('[warp][qr]','[head][qr]')
k=k.replace(' float mx=-CUDART_INF_F;',' __shared__ float inverses[3][R];\n if(warp<3){\n float mx=-CUDART_INF_F;',1)
k=k.replace(' float inverse=1.F/den;\n for(int block=0;block<8;++block){',' float inverse=1.F/den;\n if(g<R&&t==0)inverses[head][qr]=inverse;\n }\n __syncthreads();\n float inverse=inverses[head][qr];\n for(int block=(8/S)*(warp/3);block<(8/S)*(warp/3+1);++block){')
s=s[:a]+k+s[b:]
# Only the candidate launch sizes change; keep the original oracle at96 threads.
s=s.replace('attention_queries<2><<<dim3(64,3),96>>>','attention_queries<2><<<dim3(64,3),192>>>').replace('attention_queries<4><<<dim3(32,3),96>>>','attention_queries<4><<<dim3(32,3),192>>>').replace('attention_queries<8><<<dim3(16,3),96>>>','attention_queries<8><<<dim3(16,3),192>>>')
Path('/tmp/attention_query_reuse_probe_v4.cu').write_text(s)
s=s.replace('int S=2','int S=4').replace(',192>>>',',384>>>')
Path('/tmp/attention_query_reuse_probe_v5.cu').write_text(s)
