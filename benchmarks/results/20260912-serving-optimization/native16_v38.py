from pathlib import Path
r=Path('/tmp/riley-opt-260912');d=r/'prefill-shapes-source-v11/kernels/src'
s=(d/'decode_shared.cuh').read_text().replace('shared_','shared16_')
s=s.replace('rows>8','rows>16').replace('chunk_index*8+g','chunk_index*16+g').replace('chunk*8*N','chunk*16*N').replace('3*8*','3*16*').replace('rows*N','rows*N').replace('(8*N+255)','(16*N+255)')
s=s.replace('uint32_t a[4],aa[4],b[4],bb[4];','uint32_t a[4],aa[4],a1[4],a3[4],b[4],bb[4];')
s=s.replace('   if constexpr(Tiled){','   a1[step]=valid&&g+8<rows?*reinterpret_cast<const uint32_t*>(xp+8*K+offset):0;\n   a3[step]=valid&&g+8<rows?*reinterpret_cast<const uint32_t*>(xp+8*K+offset+8):0;\n   if constexpr(Tiled){',1)
s=s.replace('mma(d,a[step],0,aa[step],0,b[step],bb[step])','mma(d,a[step],a1[step],aa[step],a3[step],b[step],bb[step])')
a=s.index(' if(g<rows)for(int j=0;j<2;++j){');b=s.index('\n}\n\ntemplate',a)
s=s[:a]+''' for(int half=0;half<2;++half){int row=g+half*8;if(row>=rows)continue;
  for(int j=0;j<2;++j){auto v=__float2bfloat16_rn(d[half*2+j]);
   if constexpr(Interval>0)parts[(chunk_index*16+row)*N+base+2*t+j]=__bfloat162float(v);
   else if constexpr(Compact)out[row*8+2*t+j]=v;
   else out[row*N+base+2*t+j]=v;
  }
 }
''' +s[b:]
s=s.replace('partials[2][64]','partials[2][128]')
old=' int row=threadIdx.x/8,col=threadIdx.x%8;if(row<rows){float g=__bfloat162float(partials[0][threadIdx.x]);out[row*1536+blockIdx.x*8+col]=__float2bfloat16_rn((g/(1.F+expf(-g)))*__bfloat162float(partials[1][threadIdx.x]));}'
new=' for(int i=threadIdx.x;i<128;i+=blockDim.x){int row=i/8,col=i%8;if(row<rows){float g=__bfloat162float(partials[0][i]);out[row*1536+blockIdx.x*8+col]=__float2bfloat16_rn((g/(1.F+expf(-g)))*__bfloat162float(partials[1][i]));}}'
assert old in s;s=s.replace(old,new);s=s.replace('up to eight independent','up to sixteen independent');(d/'decode_shared16.cuh').write_text(s)
s=(d/'decode_shared_attention.cuh').read_text().replace('riley_shared_attention','riley_shared16_attention').replace('active>8','active>16').replace(',72)',',144)');(d/'decode_shared16_attention.cuh').write_text(s)
s=(d/'decode_shared_model.cuh').read_text().replace('decode_shared','decode_shared16').replace('riley_shared_','riley_shared16_').replace('shared_qkv','shared16_qkv').replace('shared_gate','shared16_gate').replace('enqueue_shared_projection','enqueue_shared16_projection')
s=s.replace('rows>8','rows>16').replace('chunk*8*n','chunk*16*n').replace('3*8*','3*16*').replace('<<<8,','<<<16,').replace('dim3(2,8)','dim3(2,16)').replace('pointwise,8)','pointwise,16)');(d/'decode_shared16_model.cuh').write_text(s)
s=(d/'decode_shared_result.cuh').read_text().replace('riley_shared_result','riley_shared16_result').replace('batch_bytes=8*','batch_bytes=16*').replace('active>8','active>16').replace('finish<<<8,','finish<<<16,');(d/'decode_shared16_result.cuh').write_text(s)
s=(r/'full_fusion_v29.cu').read_text().replace('decode_shared_model','decode_shared16_model').replace('decode_shared_result','decode_shared16_result').replace('riley_shared_model','riley_shared16_model').replace('riley_shared_result','riley_shared16_result')
s=s.replace('physical=512','physical=1024').replace('i==7?8*9*4096*4','i==7?16*9*4096*4').replace('17536','30848').replace('shared(8)','shared(16)').replace('alloc(&logits,8*98304)','alloc(&logits,16*98304)').replace('request<8','request<16').replace('for(int active:{1,2,4,8})','for(int active:{1,2,4,8,9,15,16})');(r/'full_model16_v38.cu').write_text(s)
