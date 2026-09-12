from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11/crates/riley-runtime/src/llama/multi_descriptor')
p=r/'variable_wire.rs';s=p.read_text().replace('fn all_zero(bytes:&[u8])->bool {bytes.iter().fold(0u8,|bits,&value|bits|value)==0}', 'fn all_zero(bytes:&[u8])->bool {super::result_scan::all_zero(bytes)}').replace('fn finite_argmax(logits:&[u8])->Result<u32>{','fn finite_argmax(logits:&[u8])->Result<u32>{super::result_scan::finite_argmax(logits)}\npub(super) fn scalar_finite_argmax(logits:&[u8])->Result<u32>{');p.write_text(s)
p=r/'mod.rs';s=p.read_text().replace('pub mod variable_wire16;', 'pub mod variable_wire16;\n// SIMD loads are confined to a checked, runtime-dispatched internal module.\n#[allow(unsafe_code)]\nmod result_scan;');p.write_text(s)
(r/'result_scan.rs').write_text('''//! Complete host validation with runtime-dispatched AVX2 and scalar fallback.
//! No caller alignment requirement. Vector loads occur only inside full chunks;
//! tails use checked slices. AVX2 entry points are private and feature-guarded.
use super::{check, Result};
pub(super) fn finite_argmax(bytes:&[u8])->Result<u32>{
    check(bytes.len()%2==0,"logits","incomplete BF16 word")?;
    #[cfg(target_arch="x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: feature checked; callee bounds every unaligned vector load.
        return unsafe {avx2_argmax(bytes)};
    }
    super::variable_wire::scalar_finite_argmax(bytes)
}
pub(super) fn all_zero(bytes:&[u8])->bool{
    #[cfg(target_arch="x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: feature checked; only complete 32-byte chunks are loaded.
        return unsafe {avx2_zero(bytes)};
    }
    bytes.iter().fold(0u8,|bits,&value|bits|value)==0
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn avx2_zero(bytes:&[u8])->bool{
    use std::arch::x86_64::*;
    let mut bits=_mm256_setzero_si256();let mut chunks=bytes.chunks_exact(32);
    for chunk in &mut chunks {
        // SAFETY: chunk contains exactly32 bytes; loadu permits any alignment.
        bits=_mm256_or_si256(bits,unsafe{_mm256_loadu_si256(chunk.as_ptr().cast())});
    }
    _mm256_testz_si256(bits,bits)!=0 && chunks.remainder().iter().all(|&x|x==0)
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn avx2_argmax(bytes:&[u8])->Result<u32>{
    use std::arch::x86_64::*;
    let mut maxima=_mm256_set1_ps(f32::NEG_INFINITY);
    let mut indices=_mm256_setzero_si256();let mut current=_mm256_setr_epi32(0,1,2,3,4,5,6,7);
    let mut bad=_mm256_setzero_si256();let exponent=_mm256_set1_epi32(0x7f80);
    let mut chunks=bytes.chunks_exact(16);
    for chunk in &mut chunks {
        // SAFETY: complete16-byte chunk; widening reads only those eight words.
        let words=_mm256_cvtepu16_epi32(unsafe{_mm_loadu_si128(chunk.as_ptr().cast())});
        bad=_mm256_or_si256(bad,_mm256_cmpeq_epi32(_mm256_and_si256(words,exponent),exponent));
        let values=_mm256_castsi256_ps(_mm256_slli_epi32::<16>(words));
        let greater=_mm256_cmp_ps::<_CMP_GT_OQ>(values,maxima);
        maxima=_mm256_blendv_ps(maxima,values,greater);
        indices=_mm256_blendv_epi8(indices,current,_mm256_castps_si256(greater));
        current=_mm256_add_epi32(current,_mm256_set1_epi32(8));
    }
    check(_mm256_testz_si256(bad,bad)!=0,"logits","nonfinite logit")?;
    let mut values=[0f32;8];let mut ids=[0u32;8];
    // SAFETY: destinations each contain32 writable bytes, unaligned stores allowed.
    unsafe {_mm256_storeu_ps(values.as_mut_ptr(),maxima);_mm256_storeu_si256(ids.as_mut_ptr().cast(),indices);}
    let mut maximum=f32::NEG_INFINITY;let mut token=0;
    for lane in 0..8 {if values[lane]>maximum || (values[lane]==maximum&&ids[lane]<token){maximum=values[lane];token=ids[lane];}}
    let base=bytes.len()/16*8;
    for(i,b)in chunks.remainder().chunks_exact(2).enumerate(){
        let v=f32::from_bits(u32::from(u16::from_le_bytes([b[0],b[1]]))<<16);
        check(v.is_finite(),"logits","nonfinite logit")?;
        if v>maximum{maximum=v;token=(base+i)as u32;}
    }
    Ok(token)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn every_bf16_word_and_unaligned_tail_matches_scalar(){
        for offset in 0..32 {for count in [0,1,7,8,9,15,16,17,31,32,33,49152] {
            let mut backing=vec![0xa5;offset+count*2+32];
            for(i,b)in backing[offset..offset+count*2].chunks_exact_mut(2).enumerate(){b.copy_from_slice(&((i*139+offset*73)%0x7f80).to_le_bytes()[..2]);}
            let bytes=&backing[offset..offset+count*2];
            assert_eq!(finite_argmax(bytes),super::super::variable_wire::scalar_finite_argmax(bytes));
        }}
        for word in 0..=u16::MAX {
            let mut b=vec![0;34];b[16..18].copy_from_slice(&word.to_le_bytes());
            assert_eq!(finite_argmax(&b),super::super::variable_wire::scalar_finite_argmax(&b));
        }
        assert!(finite_argmax(&[0]).is_err());
    }
    #[test] fn zero_scan_covers_unaligned_vectors_and_every_tail_byte(){
        for offset in 0..32 {for len in [0,1,7,31,32,33,63,64,65,257,98432] {
            let mut b=vec![0xa5;offset+len+32];b[offset..offset+len].fill(0);assert!(all_zero(&b[offset..offset+len]));
            for i in if len<258 {(0..len).collect::<Vec<_>>()}else{vec![0,31,32,len/2,len-1]} {
                b[offset+i]=1;assert!(!all_zero(&b[offset..offset+len]));b[offset+i]=0;
            }
        }}
    }
}
''')
