from pathlib import Path
p=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11/crates/riley-runtime/src/llama/multi_descriptor/result_scan.rs');s=p.read_text().replace('use super::{check, Result};','#![deny(unsafe_op_in_unsafe_fn)]\nuse super::{check, Result};')
for name in ['avx2_zero','avx2_argmax']:
 a=s.index('unsafe fn '+name);start=s.index('{',a);b=start+1;depth=1
 while depth:
  depth+=(s[b]=='{')-(s[b]=='}');b+=1
 body=s[start+1:b-1]
 body=body.replace('unsafe{_mm256_loadu_si256(chunk.as_ptr().cast())}', '_mm256_loadu_si256(chunk.as_ptr().cast())').replace('unsafe{_mm_loadu_si128(chunk[group*16..].as_ptr().cast())}', '_mm_loadu_si128(chunk[group*16..].as_ptr().cast())').replace('unsafe {_mm256_storeu_ps(values.as_mut_ptr(),maxima[group]);_mm256_storeu_si256(ids.as_mut_ptr().cast(),indices[group]);}', '_mm256_storeu_ps(values.as_mut_ptr(),maxima[group]);_mm256_storeu_si256(ids.as_mut_ptr().cast(),indices[group]);')
 s=s[:start+1]+'\n    // SAFETY: private entry requires AVX2; all memory accesses are bounded below.\n    unsafe {'+body+'    }\n'+s[b-1:]
p.write_text(s)
