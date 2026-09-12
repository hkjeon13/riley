from pathlib import Path
p=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11/crates/riley-runtime/src/llama/multi_descriptor/result_scan.rs');s=p.read_text();a=s.index('    let mut maxima=_mm256_set1_ps');b=s.index('    Ok(token)\n}',a)
s=s[:a]+'''    // Four independent recurrences hide compare/blend dependency latency.
    let mut maxima=[_mm256_set1_ps(f32::NEG_INFINITY);4];
    let mut indices=[_mm256_setzero_si256();4];
    let mut current=std::array::from_fn::<_,4,_>(|g|_mm256_setr_epi32(g as i32*8,g as i32*8+1,g as i32*8+2,g as i32*8+3,g as i32*8+4,g as i32*8+5,g as i32*8+6,g as i32*8+7));
    let mut bad=[_mm256_setzero_si256();4];let exponent=_mm256_set1_epi32(0x7f80);
    let mut chunks=bytes.chunks_exact(64);
    for chunk in &mut chunks {
        for group in 0..4 {
            // SAFETY: each group selects16 bytes within a complete64-byte chunk.
            let words=_mm256_cvtepu16_epi32(unsafe{_mm_loadu_si128(chunk[group*16..].as_ptr().cast())});
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
        unsafe {_mm256_storeu_ps(values.as_mut_ptr(),maxima[group]);_mm256_storeu_si256(ids.as_mut_ptr().cast(),indices[group]);}
        for lane in 0..8 {if values[lane]>maximum || (values[lane]==maximum&&ids[lane]<token){maximum=values[lane];token=ids[lane];}}
    }
    let base=bytes.len()/64*32;
    for(i,b)in chunks.remainder().chunks_exact(2).enumerate(){
        let v=f32::from_bits(u32::from(u16::from_le_bytes([b[0],b[1]]))<<16);
        check(v.is_finite(),"logits","nonfinite logit")?;
        if v>maximum{maximum=v;token=(base+i)as u32;}
    }
'''+s[b:];p.write_text(s)
