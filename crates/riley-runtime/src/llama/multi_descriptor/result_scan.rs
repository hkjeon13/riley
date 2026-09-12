//! Complete host validation with runtime-dispatched AVX2 and scalar fallback.
//! No caller alignment requirement. Vector loads occur only inside full chunks;
//! tails use checked slices. AVX2 entry points are private and feature-guarded.
#![deny(unsafe_op_in_unsafe_fn)]
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
    // SAFETY: private entry requires AVX2; all memory accesses are bounded below.
    unsafe {
    use std::arch::x86_64::*;
    let mut bits=_mm256_setzero_si256();let mut chunks=bytes.chunks_exact(32);
    for chunk in &mut chunks {
        // SAFETY: chunk contains exactly32 bytes; loadu permits any alignment.
        bits=_mm256_or_si256(bits,_mm256_loadu_si256(chunk.as_ptr().cast()));
    }
    _mm256_testz_si256(bits,bits)!=0 && chunks.remainder().iter().all(|&x|x==0)
    }
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn avx2_argmax(bytes:&[u8])->Result<u32>{
    // SAFETY: private entry requires AVX2; all memory accesses are bounded below.
    unsafe {
    use std::arch::x86_64::*;
    // Four independent recurrences hide compare/blend dependency latency.
    let mut maxima=[_mm256_set1_ps(f32::NEG_INFINITY);4];
    let mut indices=[_mm256_setzero_si256();4];
    let mut current=std::array::from_fn::<_,4,_>(|g|_mm256_setr_epi32(g as i32*8,g as i32*8+1,g as i32*8+2,g as i32*8+3,g as i32*8+4,g as i32*8+5,g as i32*8+6,g as i32*8+7));
    let mut bad=[_mm256_setzero_si256();4];let exponent=_mm256_set1_epi32(0x7f80);
    let mut chunks=bytes.chunks_exact(64);
    for chunk in &mut chunks {
        for group in 0..4 {
            // SAFETY: each group selects16 bytes within a complete64-byte chunk.
            let words=_mm256_cvtepu16_epi32(_mm_loadu_si128(chunk[group*16..].as_ptr().cast()));
            bad[group]=_mm256_or_si256(bad[group],_mm256_cmpeq_epi32(_mm256_and_si256(words,exponent),exponent));
            let values=_mm256_castsi256_ps(_mm256_slli_epi32::<16>(words));
            let greater=_mm256_cmp_ps::<_CMP_GT_OQ>(values,maxima[group]);
            maxima[group]=_mm256_blendv_ps(maxima[group],values,greater);
            indices[group]=_mm256_blendv_epi8(indices[group],current[group],_mm256_castps_si256(greater));
            current[group]=_mm256_add_epi32(current[group],_mm256_set1_epi32(32));
        }
    }
    let mut maximum=f32::NEG_INFINITY;let mut token=0;
    for group in 0..4 {
        check(_mm256_testz_si256(bad[group],bad[group])!=0,"logits","nonfinite logit")?;
        let mut values=[0f32;8];let mut ids=[0u32;8];
        // SAFETY: destinations each contain32 writable bytes, unaligned stores allowed.
        _mm256_storeu_ps(values.as_mut_ptr(),maxima[group]);_mm256_storeu_si256(ids.as_mut_ptr().cast(),indices[group]);
        for lane in 0..8 {if values[lane]>maximum || (values[lane]==maximum&&ids[lane]<token){maximum=values[lane];token=ids[lane];}}
    }
    let base=bytes.len()/64*32;
    for(i,b)in chunks.remainder().chunks_exact(2).enumerate(){
        let v=f32::from_bits(u32::from(u16::from_le_bytes([b[0],b[1]]))<<16);
        check(v.is_finite(),"logits","nonfinite logit")?;
        if v>maximum{maximum=v;token=(base+i)as u32;}
    }
    Ok(token)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore="CPU scan timing diagnostic, not serving qualification"]
    fn scan_timing_diagnostic(){
        use std::{hint::black_box,time::Instant};
        let bytes:Vec<u8>=(0..49152).flat_map(|i|(((i*139+73)%0x7f80)as u16).to_le_bytes()).collect();
        for run in 0..4 {
            for vector in if run%2==0 {[false,true]}else{[true,false]} {
                let start=Instant::now();
                for _ in 0..4096 {let result=if vector {finite_argmax(black_box(&bytes))}else{super::super::variable_wire::scalar_finite_argmax(black_box(&bytes))};black_box(result.unwrap());}
                eprintln!("SCAN vector={} iterations=4096 ns_per_row={}",vector,start.elapsed().as_nanos()/4096);
            }
        }
    }
    #[test] fn every_bf16_word_and_unaligned_tail_matches_scalar(){
        for offset in 0..32 {for count in [0,1,7,8,9,15,16,17,31,32,33,49152] {
            let mut backing=vec![0xa5;offset+count*2+32];
            for(i,b)in backing[offset..offset+count*2].chunks_exact_mut(2).enumerate(){b.copy_from_slice(&((i*139+offset*73)%0x7f80).to_le_bytes()[..2]);}
            let bytes=&backing[offset..offset+count*2];
            assert_eq!(finite_argmax(bytes),super::super::variable_wire::scalar_finite_argmax(bytes));
        }}
        for word in 0..=u16::MAX {
            for at in [word as usize%32,32+word as usize%32,64] {
                let mut b=vec![0;130];b[at*2..at*2+2].copy_from_slice(&word.to_le_bytes());
                assert_eq!(finite_argmax(&b),super::super::variable_wire::scalar_finite_argmax(&b));
            }
        }
        assert!(finite_argmax(&[0]).is_err());
    }
    #[cfg(all(target_os="linux",target_arch="x86_64"))]
    #[test] fn vector_reads_stop_at_an_inaccessible_guard_page(){
        use std::ffi::c_void;
        unsafe extern "C" {
            fn getpagesize()->i32;
            fn mmap(addr:*mut c_void,len:usize,prot:i32,flags:i32,fd:i32,offset:isize)->*mut c_void;
            fn mprotect(addr:*mut c_void,len:usize,prot:i32)->i32;
            fn munmap(addr:*mut c_void,len:usize)->i32;
        }
        // SAFETY: test owns this Linux anonymous mapping; no pointers escape.
        unsafe {
            let page=getpagesize() as usize;assert!(page>=4096);
            let ptr=mmap(std::ptr::null_mut(),page*2,3,0x22,-1,0);assert_ne!(ptr as isize,-1);
            assert_eq!(mprotect(ptr.cast::<u8>().add(page).cast(),page,0),0);
            std::slice::from_raw_parts_mut(ptr.cast::<u8>(),page).fill(0);
            for len in (0..=129).chain([257,4095,4096]) {
                let bytes=std::slice::from_raw_parts_mut(ptr.cast::<u8>().add(page-len),len);
                assert!(all_zero(bytes));
                if len%2==0 {assert_eq!(finite_argmax(bytes).unwrap(),0);}
                if len>0 {bytes[len-1]=1;assert!(!all_zero(bytes));bytes[len-1]=0;}
            }
            assert_eq!(munmap(ptr,page*2),0);
        }
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
