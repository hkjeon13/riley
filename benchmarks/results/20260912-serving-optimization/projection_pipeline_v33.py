from pathlib import Path
r=Path('/tmp/riley-opt-260912');p=r/'prefill-shapes-source-v11/kernels/src/decode_shared.cuh';s=p.read_text();a=s.index(' #pragma unroll\n for(int depth=0;depth<Chunk;depth+=16)');b=s.index('\n if(g<rows)',a)
s=s[:a]+''' // Hoist row/tile bases and issue four independent load groups before the
 // ordered MMA recurrence. Bound code size instead of fully unrolling K.
 const auto* xp=x+g*K+begin+2*t;
 const auto* wp=Tiled?w+column*(K/16)*128+(begin/16)*128+lane*2:w+(base+g)*K+begin+2*t;
 #pragma unroll 1
 for(int depth=0;depth<Chunk;depth+=64){
  uint32_t a[4],aa[4],b[4],bb[4];
  #pragma unroll
  for(int step=0;step<4;++step){
   int offset=depth+step*16;bool valid=begin+offset<end;
   a[step]=valid&&g<rows?*reinterpret_cast<const uint32_t*>(xp+offset):0;
   aa[step]=valid&&g<rows?*reinterpret_cast<const uint32_t*>(xp+offset+8):0;
   if constexpr(Tiled){
    b[step]=valid?*reinterpret_cast<const uint32_t*>(wp+(offset/16)*128):0;
    bb[step]=valid?*reinterpret_cast<const uint32_t*>(wp+(offset/16)*128+64):0;
   }else{
    b[step]=valid?*reinterpret_cast<const uint32_t*>(wp+offset):0;
    bb[step]=valid?*reinterpret_cast<const uint32_t*>(wp+offset+8):0;
   }
  }
  #pragma unroll
  for(int step=0;step<4;++step)if(begin+depth+step*16<end)riley_prefill_shape::mma(d,a[step],0,aa[step],0,b[step],bb[step]);
 }
''' +s[b:];p.write_text(s)
s=(r/'build_test_v32.py').read_text().replace('v32','v33');(r/'build_test_v33.py').write_text(s)
for suffix in ('','_c8'):(r/f'run_v3_http_shared_v33{suffix}.py').write_text((r/f'run_v3_http_shared_v32{suffix}.py').read_text().replace('v32','v33').replace('shared-r2-final','shared-final'))
