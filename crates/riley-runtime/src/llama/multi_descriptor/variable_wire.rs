//! V3 request encoding. Packet bytes are checked against external live authority;
//! they neither create ownership nor demonstrate GPU completion. Not yet dispatched.
use super::{check, overflow, BlockOwnership, Result, ResultMode};
use super::shape_progress::{InputStage, Progress};
use std::collections::{BTreeMap, BTreeSet};
pub const HEADER_BYTES:usize=128;
pub const ROW_BYTES:usize=1664; // 128 header +256*u32 pages +256*u16 valid counts
pub const TOKENS_OFFSET:usize=HEADER_BYTES+8*ROW_BYTES;
pub const REQUEST_BYTES:usize=TOKENS_OFFSET+1024*4;
pub const MAGIC:u32=0x33444d52;
#[derive(Clone, Debug)]
pub struct Row {
    pub sequence_tag:u64, pub cookie:u64, pub output_slot:u32,
    pub progress:Progress, pub input_tokens:Vec<u32>,
    pub physical_ids:Vec<u32>, pub valid_tokens:Vec<u16>,
}
#[derive(Clone, Debug)]
pub struct Expectation {
    pub owner_generation:u64, pub last_accepted_replay:u64, pub replay_id:u64,
    pub iteration_id:u64, pub catalog_digest:[u8;32],
    pub physical_block_count:u32, pub max_active_rows:u32,
    pub stage:InputStage, pub mode:ResultMode, pub rows:Vec<Row>,
    pub block_ownership:Vec<BlockOwnership>,
}
pub fn validate(e:&Expectation)->Result<()> {
    check(e.owner_generation!=0 && e.iteration_id!=0 && e.catalog_digest!=[0;32],"owner","missing retained owner identity")?;
    check(e.replay_id==e.last_accepted_replay.checked_add(1).ok_or_else(||overflow("replay"))?,"replay","not next replay")?;
    check(matches!(e.max_active_rows,1|2|4|8) && !e.rows.is_empty() && e.rows.len()<=e.max_active_rows as usize,"rows","unsupported active rows")?;
    check(e.stage!=InputStage::Prefill || e.rows.len()==1,"stage","prefill must describe one request")?;
    check(e.physical_block_count>0 && e.physical_block_count<=4096,"pool","unsupported physical pool")?;
    let mut ownership=BTreeMap::new();
    for x in &e.block_ownership {
        check(x.sequence_tag!=0 && x.physical_id<e.physical_block_count && ownership.insert(x.physical_id,x.sequence_tag).is_none(),"ownership","invalid or duplicate ledger entry")?;
    }
    let (mut tags,mut cookies,mut slots,mut used)=(BTreeSet::new(),BTreeSet::new(),BTreeSet::new(),BTreeSet::new());
    for row in &e.rows {
        check(row.sequence_tag!=0 && tags.insert(row.sequence_tag) && row.cookie!=0 && cookies.insert(row.cookie),"row_identity","duplicate or zero request identity")?;
        check(row.output_slot<e.rows.len() as u32 && slots.insert(row.output_slot),"slot","slots not unique and dense")?;
        check(row.progress.stage==e.stage,"stage","row stage differs")?;
        let v=row.progress.validate()?;
        check(row.input_tokens.len()==row.progress.input_tokens as usize && row.input_tokens.iter().all(|&t|t<49152),"tokens","invalid token count or vocabulary")?;
        check(row.physical_ids.len()==v.live_pages as usize && row.valid_tokens.len()==v.live_pages as usize,"pages","page map does not cover target exactly")?;
        for (i,(&id,&valid)) in row.physical_ids.iter().zip(&row.valid_tokens).enumerate() {
            check(id<e.physical_block_count && used.insert(id),"pages","out of pool or aliased pages")?;
            check(ownership.get(&id)==Some(&row.sequence_tag),"ownership","page not reserved by this request")?;
            check(valid==if i+1==row.physical_ids.len(){v.last_page_tokens}else{16},"valid_tokens","invalid page occupancy")?;
        }
    }
    Ok(())
}
fn u32_at(p:&mut [u8],at:usize,v:u32){p[at..at+4].copy_from_slice(&v.to_le_bytes());}
fn u64_at(p:&mut [u8],at:usize,v:u64){p[at..at+8].copy_from_slice(&v.to_le_bytes());}
/// Caller can reuse a retained buffer. Reject before mutating it on invalid authority.
pub fn encode_into(packet:&mut [u8],e:&Expectation)->Result<()> {
    check(packet.len()==REQUEST_BYTES,"bytes","exact V3 request extent required")?;
    validate(e)?;packet.fill(0);
    for (at,v) in [(0,MAGIC),(4,3),(8,REQUEST_BYTES as u32),(12,ROW_BYTES as u32),
        (16,if e.stage==InputStage::Prefill{0}else{1}),(20,e.rows.len() as u32),(24,e.max_active_rows),(28,e.physical_block_count),(32,e.mode as u32)] {u32_at(packet,at,v);}
    u64_at(packet,40,e.owner_generation);u64_at(packet,48,e.replay_id);u64_at(packet,56,e.iteration_id);packet[64..96].copy_from_slice(&e.catalog_digest);
    for (i,row) in e.rows.iter().enumerate() {
        let b=HEADER_BYTES+i*ROW_BYTES;let p=row.progress;let v=p.validate()?;
        for (at,value) in [(0,row.input_tokens[0]),(4,v.last_position),(8,p.input_tokens),(12,v.live_pages),
            (16,p.committed_tokens),(20,v.target_tokens),(24,p.generated_index),(28,p.output_limit),
            (32,p.prompt_tokens),(36,p.context_tokens),(40,row.output_slot),(44,v.logits_input_row.unwrap_or(u32::MAX))] {u32_at(packet,b+at,value);}
        u64_at(packet,b+48,row.sequence_tag);u64_at(packet,b+56,row.cookie);
        for (j,(&id,&valid)) in row.physical_ids.iter().zip(&row.valid_tokens).enumerate(){u32_at(packet,b+128+j*4,id);packet[b+1152+j*2..b+1154+j*2].copy_from_slice(&valid.to_le_bytes());}
    }
    if e.stage==InputStage::Prefill {for (i,&t) in e.rows[0].input_tokens.iter().enumerate(){u32_at(packet,TOKENS_OFFSET+i*4,t);}}
    Ok(())
}
/// Canonical comparison binds every byte, including padding, to live authority.
/// Scratch is caller-owned to avoid a replay-time allocation.
pub fn validate_packet(packet:&[u8],scratch:&mut [u8],e:&Expectation)->Result<()> {
    check(packet.len()==REQUEST_BYTES,"bytes","exact V3 request extent required")?;
    encode_into(scratch,e)?;check(packet==scratch,"packet","V3 bytes differ from live expectation")
}
pub const RESULT_BYTES:usize=128+49152*2;
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
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(stage:InputStage,active:u32)->Expectation {
        let mut e=Expectation{owner_generation:1,last_accepted_replay:4,replay_id:5,iteration_id:7,catalog_digest:[19;32],physical_block_count:4096,max_active_rows:8,stage,mode:ResultMode::Greedy,rows:vec![],block_ownership:vec![]};
        for i in 0..active {
            let progress=if stage==InputStage::Prefill {Progress{prompt_tokens:398,output_limit:128,context_tokens:1024,committed_tokens:128,input_tokens:73,generated_index:0,stage}} else {Progress{prompt_tokens:129+i*17,output_limit:128,context_tokens:1024,committed_tokens:129+i*17+63,input_tokens:1,generated_index:64,stage}};
            let v=progress.validate().unwrap();let ids:Vec<_>=(0..v.live_pages).map(|p|i*256+p).collect();
            e.block_ownership.extend(ids.iter().map(|&physical_id|BlockOwnership{physical_id,sequence_tag:u64::from(i)+1}));
            let mut valid=vec![16;ids.len()];*valid.last_mut().unwrap()=v.last_page_tokens;
            e.rows.push(Row{sequence_tag:u64::from(i)+1,cookie:u64::from(i)+29,output_slot:active-1-i,progress,input_tokens:vec![33;progress.input_tokens as usize],physical_ids:ids,valid_tokens:valid});
        } e
    }
    #[test] fn variable_shapes_and_partial_output_round_trip_with_every_byte_bound(){
        for e in [fixture(InputStage::Prefill,1),fixture(InputStage::Decode,8)] {
            let mut p=vec![0;REQUEST_BYTES];let mut scratch=p.clone();encode_into(&mut p,&e).unwrap();validate_packet(&p,&mut scratch,&e).unwrap();
            if let Some(dir)=std::env::var_os("RILEY_V3_FIXTURES") {let dir=std::path::PathBuf::from(dir);std::fs::create_dir_all(&dir).unwrap();std::fs::write(dir.join(if e.stage==InputStage::Prefill{"prefill.bin"}else{"decode.bin"}),&p).unwrap();}
            for offset in 0..REQUEST_BYTES {p[offset]^=1;assert!(validate_packet(&p,&mut scratch,&e).is_err(),"offset{offset}");p[offset]^=1;}
            if e.stage==InputStage::Prefill {assert_eq!(&p[HEADER_BYTES+44..HEADER_BYTES+48],&u32::MAX.to_le_bytes());}
        }
    }
    #[test] fn rejects_stale_replay_foreign_page_and_overlapping_reservations_without_writes(){
        let e=fixture(InputStage::Decode,8);
        let mut cases=vec![];let mut x=e.clone();x.replay_id=4;cases.push(x);
        let mut x=e.clone();x.block_ownership[0].sequence_tag=999;cases.push(x);
        let mut x=e.clone();x.rows[1].physical_ids[0]=x.rows[0].physical_ids[0];cases.push(x);
        let mut x=e.clone();x.rows[0].progress.generated_index=65;cases.push(x);
        let mut x=e.clone();x.rows[0].valid_tokens[0]=15;cases.push(x);
        for x in cases {let mut p=vec![0xa5;REQUEST_BYTES];assert!(encode_into(&mut p,&x).is_err());assert!(p.iter().all(|&v|v==0xa5));}
    }
    #[test] fn result_binds_replay_and_suppresses_partial_output(){
        let mut e=fixture(InputStage::Prefill,1);let mut bytes=vec![0;RESULT_BYTES];let mut header=[0;128];result_identity_into(&mut header,&e,0).unwrap();bytes[..128].copy_from_slice(&header);
        assert_eq!(validate_result(&bytes,&e).unwrap().0,None);
        for offset in 0..128 {bytes[offset]^=1;assert!(validate_result(&bytes,&e).is_err(),"offset{offset}");bytes[offset]^=1;}
        bytes[128]=1;assert!(validate_result(&bytes,&e).is_err());bytes[128]=0;
        e.rows[0].progress.prompt_tokens=201;result_identity_into(&mut header,&e,7).unwrap();bytes[..128].copy_from_slice(&header);bytes[128+7*2..130+7*2].copy_from_slice(&0x3f80u16.to_le_bytes());
        assert_eq!(validate_result(&bytes,&e).unwrap().0,Some(7));
        e.replay_id+=1;e.last_accepted_replay+=1;assert!(validate_result(&bytes,&e).is_err());
    }

}
