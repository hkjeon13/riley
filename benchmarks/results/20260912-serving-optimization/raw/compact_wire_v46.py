from pathlib import Path
r=Path('/tmp/riley-opt-260912');p=r/'prefill-shapes-source-v11/crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs';s=p.read_text()
s=s.replace('    pub const OUTPUT_OFFSET:usize=Self::BATCH_RESULT_BYTES;','''    pub const COMPACT_RESULT_BYTES:usize=ROWS*HEADER_BYTES;
    pub const COMPACT_RESULT_MAGIC:u32=Self::RESULT_MAGIC^0x8000_0000;
    pub const OUTPUT_OFFSET:usize=Self::BATCH_RESULT_BYTES;''',1)
pos=s.index('\n#[cfg(test)]\nmod tests {')
new='''
/// Validate compact records downloaded from a prepared GPU-greedy reservation.
/// The GPU must scan every published logit and report any nonfinite value in
/// argmax status. This validates identities/status, not a CPU rescan of logits;
/// callers must not treat arbitrary bytes as evidence of device completion.
/// No record escapes before all active and inactive records have validated.
pub fn validate_compact_result<'a,const ROWS:usize>(bytes:&'a[u8],e:&Expectation<ROWS>)->Result<Vec<RowResult<'a>>>{
    validate(e)?;
    check(e.mode==ResultMode::Greedy,"mode","compact result requires greedy mode")?;
    check(bytes.len()==Layout::<ROWS>::COMPACT_RESULT_BYTES,"result","exact compact result extent required")?;
    let mut result=Vec::with_capacity(e.rows.len());
    for (index,row) in e.rows.iter().enumerate(){
        let record=&bytes[index*HEADER_BYTES..(index+1)*HEADER_BYTES];
        let token=u32::from_le_bytes(record[8..12].try_into().unwrap());
        let published=row.progress.validate()?.logits_input_row.is_some();
        check(token<49152 && (published||token==0),"argmax","invalid compact token publication")?;
        let mut expected=[0u8;HEADER_BYTES];result_row_identity_into(&mut expected,e,index,token)?;
        u32_at(&mut expected,124,Layout::<ROWS>::COMPACT_RESULT_MAGIC);
        check(record==expected,"result_identity","compact identity or GPU validation status differs")?;
        result.push(RowResult{output_slot:row.output_slot,token:published.then_some(token),logits:&[]});
    }
    check(all_zero(&bytes[e.rows.len()*HEADER_BYTES..]),"inactive_result","inactive compact record published bytes")?;
    Ok(result)
}
'''
s=s[:pos]+new+s[pos:]
pos=s.index('    fn fixture(stage:InputStage,active:u32)')
tests='''    fn compact_fixture<const ROWS:usize>(e:&Expectation<ROWS>)->Vec<u8>{
        let mut bytes=vec![0;Layout::<ROWS>::COMPACT_RESULT_BYTES];
        for i in 0..e.rows.len(){let mut h=[0;128];let token=if e.rows[i].progress.validate().unwrap().logits_input_row.is_some(){i as u32+7}else{0};
            result_row_identity_into(&mut h,e,i,token).unwrap();u32_at(&mut h,124,Layout::<ROWS>::COMPACT_RESULT_MAGIC);bytes[i*128..(i+1)*128].copy_from_slice(&h);}
        bytes
    }
    fn compact_contract<const ROWS:usize>(){
        for active in 1..=ROWS{
            let e=fixture_rows::<ROWS>(InputStage::Decode,active as u32);let mut bytes=compact_fixture(&e);
            let rows=validate_compact_result(&bytes,&e).unwrap();assert_eq!(rows.len(),active);
            for (i,row) in rows.iter().enumerate(){assert_eq!(row.output_slot,e.rows[i].output_slot);assert_eq!(row.token,Some(i as u32+7));assert!(row.logits.is_empty());}
            for row in 0..active{for offset in 0..128{if (8..12).contains(&offset){continue;}let at=row*128+offset;bytes[at]^=1;assert!(validate_compact_result(&bytes,&e).is_err(),"row{row} offset{offset}");bytes[at]^=1;}}
            let old=bytes[8..12].to_vec();bytes[8..12].copy_from_slice(&49152u32.to_le_bytes());assert!(validate_compact_result(&bytes,&e).is_err());bytes[8..12].copy_from_slice(&old);
            if active<ROWS{for at in [active*128,bytes.len()-1]{bytes[at]=1;assert!(validate_compact_result(&bytes,&e).is_err());bytes[at]=0;}}
            assert!(validate_compact_result(&bytes[..bytes.len()-1],&e).is_err());assert!(validate_batch_result(&bytes,&e).is_err());
            let mut wrong=e.clone();wrong.mode=ResultMode::FullLogits;assert!(validate_compact_result(&bytes,&wrong).is_err());
            if active>1{for j in 0..128{bytes.swap(j,128+j);}assert!(validate_compact_result(&bytes,&e).is_err());}
        }
        let e=fixture_rows::<ROWS>(InputStage::Prefill,1);let mut bytes=compact_fixture(&e);
        assert_eq!(validate_compact_result(&bytes,&e).unwrap()[0].token,None);
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());assert!(validate_compact_result(&bytes,&e).is_err());
    }
    #[test] fn compact_eight_identity_status_and_publication(){compact_contract::<8>();}
    #[test] fn compact_sixteen_identity_status_and_publication(){compact_contract::<16>();}
'''
s=s[:pos]+tests+s[pos:];p.write_text(s)
