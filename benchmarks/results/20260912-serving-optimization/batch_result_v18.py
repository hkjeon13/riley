from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs';s=p.read_text();a='    check(e.rows.len()==1 && bytes.len()==RESULT_BYTES,"result","unsupported result shape")?;';b=a+'\n    validate_result_row(bytes,e,0)\n}\nfn validate_result_row<\'a>(bytes:&\'a[u8],e:&Expectation,index:usize)->Result<(Option<u32>,&\'a[u8])>{';assert a in s;s=s.replace(a,b)
s=s.replace('    let row=&e.rows[0];let progress=row.progress.validate()?;', '    let row=&e.rows[index];let progress=row.progress.validate()?;')
s=s.replace('result_identity_into(&mut expected,e,token)?;', 'result_row_identity_into(&mut expected,e,index,token)?;')
a='fn result_identity_into(out:&mut[u8;128],e:&Expectation,token:u32)->Result<()>{\n    let row=&e.rows[0];';b='''#[cfg(test)]
fn result_identity_into(out:&mut[u8;128],e:&Expectation,token:u32)->Result<()>{result_row_identity_into(out,e,0,token)}
fn result_row_identity_into(out:&mut[u8;128],e:&Expectation,index:usize,token:u32)->Result<()>{
    let row=&e.rows[index];''';assert a in s;s=s.replace(a,b)
pos=s.index('#[cfg(test)]\nmod tests')
s=s[:pos]+'''/// Fixed-capacity readback. Unused records must be zero, and no record is
/// returned until every active identity/status/logit row has validated.
pub const BATCH_RESULT_BYTES:usize=8*RESULT_BYTES;
#[derive(Debug)]
pub struct RowResult<'a>{pub output_slot:u32,pub token:Option<u32>,pub logits:&'a[u8]}
pub fn validate_batch_result<'a>(bytes:&'a[u8],e:&Expectation)->Result<Vec<RowResult<'a>>>{
    validate(e)?;
    check(bytes.len()==BATCH_RESULT_BYTES,"result","exact eight-record extent required")?;
    let mut result=Vec::with_capacity(e.rows.len());
    for (index,row) in e.rows.iter().enumerate(){
        let record=&bytes[index*RESULT_BYTES..(index+1)*RESULT_BYTES];
        let (token,logits)=validate_result_row(record,e,index)?;
        result.push(RowResult{output_slot:row.output_slot,token,logits});
    }
    check(bytes[e.rows.len()*RESULT_BYTES..].iter().all(|&v|v==0),"inactive_result","inactive record published bytes")?;
    Ok(result)
}

'''+s[pos:]
pos=s.rfind('\n}')
s=s[:pos]+'''
    #[test] fn batch_results_bind_all_rows_before_publication(){
        for active in [1,2,4,8]{
            let e=fixture(InputStage::Decode,active);let mut bytes=vec![0;BATCH_RESULT_BYTES];
            for i in 0..active as usize {let mut h=[0;128];result_row_identity_into(&mut h,&e,i,i as u32+7).unwrap();let b=i*RESULT_BYTES;bytes[b..b+128].copy_from_slice(&h);bytes[b+128+(i+7)*2..b+130+(i+7)*2].copy_from_slice(&0x3f80u16.to_le_bytes());}
            let rows=validate_batch_result(&bytes,&e).unwrap();assert_eq!(rows.len(),active as usize);
            for(i,row)in rows.iter().enumerate(){assert_eq!(row.output_slot,active-1-i as u32);assert_eq!(row.token,Some(i as u32+7));}
            for row in 0..active as usize {for offset in 0..128 {let at=row*RESULT_BYTES+offset;bytes[at]^=1;assert!(validate_batch_result(&bytes,&e).is_err());bytes[at]^=1;}}
            if active<8 {let at=active as usize*RESULT_BYTES;bytes[at]=1;assert!(validate_batch_result(&bytes,&e).is_err());bytes[at]=0;}
            let last=(active as usize-1)*RESULT_BYTES;bytes[last+128..last+130].copy_from_slice(&0x7fc0u16.to_le_bytes());assert!(validate_batch_result(&bytes,&e).is_err());
        }
    }
'''+s[pos:];p.write_text(s)
