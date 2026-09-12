from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs';s=p.read_text();at=s.index('#[cfg(test)]')
s=s[:at]+'''pub const RESULT_BYTES:usize=128+49152*2;
/// Validate a completed read from the caller's one outstanding retained owner.
/// This cannot prove quiescence or perform scheduler settlement on its own.
pub fn validate_result<'a>(bytes:&'a[u8],e:&Expectation)->Result<(Option<u32>,&'a[u8])>{
    validate(e)?;
    check(e.stage==InputStage::Prefill && e.rows.len()==1 && bytes.len()==RESULT_BYTES,"result","unsupported result shape")?;
    let u32_at=|at:usize|u32::from_le_bytes(bytes[at..at+4].try_into().unwrap());
    check(u32_at(0)==0 && u32_at(12)==0,"status","GPU execution or argmax failed")?;
    let row=&e.rows[0];let progress=row.progress.validate()?;let published=progress.logits_input_row.is_some();
    let token=u32_at(8);check(token<49152,"argmax","token out of vocabulary")?;
    let mut expected=[0u8;128];result_identity_into(&mut expected,e,token)?;
    check(bytes[..128]==expected,"result_identity","completion differs from outstanding expectation")?;
    let logits=&bytes[128..];
    if !published {check(logits.iter().all(|&v|v==0) && token==0,"partial_output","partial prefill produced output")?;return Ok((None,logits));}
    let mut maximum=f32::NEG_INFINITY;let mut host_token=0;
    for (i,word) in logits.chunks_exact(2).enumerate(){let v=f32::from_bits(u32::from(u16::from_le_bytes([word[0],word[1]]))<<16);
        check(v.is_finite(),"logits","nonfinite logit")?;if v>maximum {maximum=v;host_token=i as u32;}}
    check(token==host_token,"argmax","token differs from published logits")?;
    Ok((Some(token),logits))
}
fn result_identity_into(out:&mut[u8;128],e:&Expectation,token:u32)->Result<()>{
    let row=&e.rows[0];let p=row.progress;let v=p.validate()?;out.fill(0);
    for(at,x)in[(4,u32::from(v.logits_input_row.is_some())),(8,token),(56,v.target_tokens),(60,p.prompt_tokens),(64,p.generated_index),(68,p.input_tokens),(72,p.context_tokens),(76,if e.stage==InputStage::Prefill{0}else{1}),(112,v.last_position),(116,row.output_slot),(120,e.mode as u32),(124,0x33524d52)]{u32_at(out,at,x);}
    for(at,x)in[(16,e.owner_generation),(24,e.replay_id),(32,e.iteration_id),(40,row.sequence_tag),(48,row.cookie)]{u64_at(out,at,x);}
    out[80..112].copy_from_slice(&e.catalog_digest);Ok(())
}
'''+s[at:]
pos=s.rfind('\n}')
s=s[:pos]+'''
    #[test] fn result_binds_replay_and_suppresses_partial_output(){
        let mut e=fixture(InputStage::Prefill,1);let mut bytes=vec![0;RESULT_BYTES];let mut header=[0;128];result_identity_into(&mut header,&e,0).unwrap();bytes[..128].copy_from_slice(&header);
        assert_eq!(validate_result(&bytes,&e).unwrap().0,None);
        for offset in 0..128 {bytes[offset]^=1;assert!(validate_result(&bytes,&e).is_err(),"offset{offset}");bytes[offset]^=1;}
        bytes[128]=1;assert!(validate_result(&bytes,&e).is_err());bytes[128]=0;
        e.rows[0].progress.prompt_tokens=201;result_identity_into(&mut header,&e,7).unwrap();bytes[..128].copy_from_slice(&header);bytes[128+7*2..130+7*2].copy_from_slice(&0x3f80u16.to_le_bytes());
        assert_eq!(validate_result(&bytes,&e).unwrap().0,Some(7));
        e.replay_id+=1;e.last_accepted_replay+=1;assert!(validate_result(&bytes,&e).is_err());
    }
'''+s[pos:];p.write_text(s)
