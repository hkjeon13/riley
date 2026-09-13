//! Versioned request encoding checked against external live authority.
//! V3 retains eight rows; V4 uses sixteen. Packet bytes never establish ownership.
use super::{check, overflow, BlockOwnership, Result, ResultMode};
use super::shape_progress::{InputStage, Progress};
use std::collections::{BTreeMap, BTreeSet};
pub const HEADER_BYTES:usize=128;
pub const ROW_BYTES:usize=1664; // 128 header +256*u32 pages +256*u16 valid counts
pub const TOKENS_OFFSET:usize=HEADER_BYTES+8*ROW_BYTES;
pub const REQUEST_BYTES:usize=TOKENS_OFFSET+1024*4;
pub const MAGIC:u32=0x33444d52;
/// Compile-time capacities. Public operations reject unsupported ROWS before writing.
pub struct Layout<const ROWS:usize>;
impl<const ROWS:usize> Layout<ROWS> {
    pub const VERSION:u32=if ROWS==8 {3} else if ROWS==16 {4} else {5};
    pub const MAGIC:u32=if ROWS==8 {0x33444d52} else if ROWS==16 {0x34444d52} else {0x35444d52};
    pub const RESULT_MAGIC:u32=if ROWS==8 {0x33524d52} else if ROWS==16 {0x34524d52} else {0x35524d52};
    pub const TOKENS_OFFSET:usize=HEADER_BYTES+ROWS*ROW_BYTES;
    pub const REQUEST_BYTES:usize=Self::TOKENS_OFFSET+1024*4;
    pub const BATCH_RESULT_BYTES:usize=ROWS*RESULT_BYTES;
    pub const COMPACT_RESULT_BYTES:usize=ROWS*HEADER_BYTES;
    pub const COMPACT_RESULT_MAGIC:u32=Self::RESULT_MAGIC^0x8000_0000;
    pub const OUTPUT_OFFSET:usize=Self::BATCH_RESULT_BYTES;
    pub const STAGING_BYTES:usize=2*Self::BATCH_RESULT_BYTES;
}
#[derive(Clone, Debug)]
pub struct Row {
    pub sequence_tag:u64, pub cookie:u64, pub output_slot:u32,
    pub progress:Progress, pub input_tokens:Vec<u32>,
    pub physical_ids:Vec<u32>, pub valid_tokens:Vec<u16>,
}
#[derive(Clone, Debug)]
pub struct Expectation<const ROWS:usize=8> {
    pub owner_generation:u64, pub last_accepted_replay:u64, pub replay_id:u64,
    pub iteration_id:u64, pub catalog_digest:[u8;32],
    pub physical_block_count:u32, pub max_active_rows:u32,
    pub stage:InputStage, pub mode:ResultMode, pub rows:Vec<Row>,
    pub block_ownership:Vec<BlockOwnership>,
    /// Explicit V6 capability, bound to the retained native graph by the session.
    pub packed_prefill:bool,
    pub mixed_execution:bool,
}
fn result_magic<const ROWS:usize>(e:&Expectation<ROWS>)->u32 {if e.mixed_execution{0x37524d52}else if e.packed_prefill{0x36524d52}else{Layout::<ROWS>::RESULT_MAGIC}}
fn compact_magic<const ROWS:usize>(e:&Expectation<ROWS>)->u32 {result_magic(e)^0x8000_0000}
pub const MIXED_REQUEST_BYTES:usize=Layout::<32>::REQUEST_BYTES+4096;
pub fn request_bytes<const ROWS:usize>(e:&Expectation<ROWS>)->usize{if e.mixed_execution{MIXED_REQUEST_BYTES}else{Layout::<ROWS>::REQUEST_BYTES}}
fn stage_number(stage:InputStage)->u32{if stage==InputStage::Prefill{0}else{1}}
fn packet_stage<const ROWS:usize>(e:&Expectation<ROWS>)->u32{if e.mixed_execution && e.rows.iter().any(|r|r.progress.stage==InputStage::Prefill) && e.rows.iter().any(|r|r.progress.stage==InputStage::Decode){2}else{stage_number(e.stage)}}
pub fn validate<const ROWS:usize>(e:&Expectation<ROWS>)->Result<()> {
    check(matches!(ROWS,8|16|32),"capacity","unsupported wire capacity")?;
    check(e.owner_generation!=0 && e.iteration_id!=0 && e.catalog_digest!=[0;32],"owner","missing retained owner identity")?;
    check(e.replay_id==e.last_accepted_replay.checked_add(1).ok_or_else(||overflow("replay"))?,"replay","not next replay")?;
    check(matches!(e.max_active_rows,1|2|4|8|16|32) && e.max_active_rows as usize<=ROWS && !e.rows.is_empty() && e.rows.len()<=e.max_active_rows as usize,"rows","unsupported active rows")?;
    check(!e.mixed_execution || (e.packed_prefill&&ROWS==32),"capacity","mixed execution requires packed32 capability")?;
    check(!e.packed_prefill || ROWS==32,"capacity","packed prefill requires V6 capacity")?;
    let prefills=e.rows.iter().filter(|r|r.progress.stage==InputStage::Prefill).count();
    check(e.stage==if prefills>0{InputStage::Prefill}else{InputStage::Decode},"stage","aggregate stage differs from rows")?;
    check(prefills<=if e.packed_prefill{4}else{1},"stage","unsupported prefill owner count")?;
    let total=e.rows.iter().try_fold(0usize,|n,r|n.checked_add(r.input_tokens.len()).ok_or_else(||overflow("tokens")))?;
    check(e.stage!=InputStage::Prefill || total<=1024,"tokens","packed token capacity exceeded")?;
    check(e.physical_block_count>0 && e.physical_block_count<=4096,"pool","unsupported physical pool")?;
    let mut ownership=BTreeMap::new();
    for x in &e.block_ownership {
        check(x.sequence_tag!=0 && x.physical_id<e.physical_block_count && ownership.insert(x.physical_id,x.sequence_tag).is_none(),"ownership","invalid or duplicate ledger entry")?;
    }
    let published=e.rows.iter().filter(|r|r.progress.committed_tokens.checked_add(r.progress.input_tokens)==Some(r.progress.prompt_tokens) || r.progress.stage==InputStage::Decode).count();
    let (mut tags,mut cookies,mut slots,mut used)=(BTreeSet::new(),BTreeSet::new(),BTreeSet::new(),BTreeSet::new());
    for row in &e.rows {
        check(row.sequence_tag!=0 && tags.insert(row.sequence_tag) && row.cookie!=0 && cookies.insert(row.cookie),"row_identity","duplicate or zero request identity")?;
        check(row.output_slot<e.rows.len() as u32 && slots.insert(row.output_slot),"slot","slots not unique and dense")?;
        check(e.mixed_execution || row.progress.stage==e.stage,"stage","row stage differs")?;
        let v=row.progress.validate()?;
        if e.packed_prefill {check(((row.output_slot as usize)<published)==v.logits_input_row.is_some(),"slot","published slots must precede internal partial slots")?;}
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
pub fn encode_into<const ROWS:usize>(packet:&mut [u8],e:&Expectation<ROWS>)->Result<()> {
    check(packet.len()==request_bytes(e),"bytes","exact versioned request extent required")?;
    let checked = checked_expectation(e)?;
    encode_checked_into(packet, &checked)
}

/// The immutable borrow prevents changing the expectation after validation.
pub(super) struct CheckedExpectation<'a, const ROWS: usize> { expectation: &'a Expectation<ROWS> }
pub(super) fn checked_expectation<const ROWS: usize>(e: &Expectation<ROWS>) -> Result<CheckedExpectation<'_, ROWS>> {
    validate(e)?;
    Ok(CheckedExpectation { expectation: e })
}
pub(super) fn encode_checked_into<const ROWS: usize>(packet: &mut [u8], checked: &CheckedExpectation<'_, ROWS>) -> Result<()> {
    let e = checked.expectation;
    check(packet.len()==request_bytes(e),"bytes","exact versioned request extent required")?;
    packet.fill(0);
    for (at,v) in [(0,if e.mixed_execution{0x37444d52}else if e.packed_prefill{0x36444d52}else{Layout::<ROWS>::MAGIC}),(4,if e.mixed_execution{7}else if e.packed_prefill{6}else{Layout::<ROWS>::VERSION}),(8,request_bytes(e) as u32),(12,ROW_BYTES as u32),
        (16,packet_stage(e)),(20,e.rows.len() as u32),(24,e.max_active_rows),(28,e.physical_block_count),(32,e.mode as u32)] {u32_at(packet,at,v);}
    u64_at(packet,40,e.owner_generation);u64_at(packet,48,e.replay_id);u64_at(packet,56,e.iteration_id);packet[64..96].copy_from_slice(&e.catalog_digest);
    let mut token_offset=0usize;let mut tile_offset=0usize;
    for (i,row) in e.rows.iter().enumerate() {
        let b=HEADER_BYTES+i*ROW_BYTES;let p=row.progress;let v=p.validate()?;
        for (at,value) in [(0,row.input_tokens[0]),(4,v.last_position),(8,p.input_tokens),(12,v.live_pages),
            (16,p.committed_tokens),(20,v.target_tokens),(24,p.generated_index),(28,p.output_limit),
            (32,p.prompt_tokens),(36,p.context_tokens),(40,row.output_slot),(44,v.logits_input_row.unwrap_or(u32::MAX))] {u32_at(packet,b+at,value);}
        u64_at(packet,b+48,row.sequence_tag);u64_at(packet,b+56,row.cookie);
        if e.mixed_execution {u32_at(packet,b+72,stage_number(p.stage));}
        if e.stage==InputStage::Prefill {
            if e.mixed_execution {
                u32_at(packet,b+68,tile_offset as u32);
                let tiles=if p.input_tokens<32{p.input_tokens}else{p.input_tokens.div_ceil(8)};
                for local in 0..tiles {u32_at(packet,Layout::<32>::REQUEST_BYTES+tile_offset*4,((i as u32)<<16)|local);tile_offset+=1;}
            }
            if e.packed_prefill {u32_at(packet,b+64,token_offset as u32);}
            for &t in &row.input_tokens {u32_at(packet,Layout::<ROWS>::TOKENS_OFFSET+token_offset*4,t);token_offset+=1;}
        }
        for (j,(&id,&valid)) in row.physical_ids.iter().zip(&row.valid_tokens).enumerate(){u32_at(packet,b+128+j*4,id);packet[b+1152+j*2..b+1154+j*2].copy_from_slice(&valid.to_le_bytes());}
    }
    if e.packed_prefill && e.stage==InputStage::Prefill {u32_at(packet,36,token_offset as u32);}
    if e.mixed_execution {u32_at(packet,96,tile_offset as u32);}
    Ok(())
}
/// Canonical comparison binds every byte, including padding, to live authority.
/// Scratch is caller-owned to avoid a replay-time allocation.
pub fn validate_packet<const ROWS:usize>(packet:&[u8],scratch:&mut [u8],e:&Expectation<ROWS>)->Result<()> {
    check(packet.len()==request_bytes(e),"bytes","exact versioned request extent required")?;
    encode_into(scratch,e)?;check(packet==scratch,"packet","versioned bytes differ from live expectation")
}
pub const RESULT_BYTES:usize=128+49152*2;
/// Validate a completed read from the caller's one outstanding retained owner.
/// This cannot prove quiescence or perform scheduler settlement on its own.
// A reduction permits vectorized scanning while checking every inactive byte.
fn all_zero(bytes:&[u8])->bool {super::result_scan::all_zero(bytes)}
// Independent accumulators shorten the maximum dependency chain. All lanes
// reject nonfinite values; final ties still choose the lowest vocabulary ID.
fn finite_argmax(logits:&[u8])->Result<u32>{super::result_scan::finite_argmax(logits)}
pub(super) fn scalar_finite_argmax(logits:&[u8])->Result<u32>{
 let mut maxima=[f32::NEG_INFINITY;8];let mut indices=[0u32;8];
 let mut chunks=logits.chunks_exact(16);
 for (block,chunk) in chunks.by_ref().enumerate(){
  let mut nonfinite=false;
  for lane in 0..8 {let bits=u16::from_le_bytes([chunk[2*lane],chunk[2*lane+1]]);nonfinite|=bits&0x7f80==0x7f80;let value=f32::from_bits(u32::from(bits)<<16);
   if value>maxima[lane]{maxima[lane]=value;indices[lane]=(block*8+lane)as u32;}}
  check(!nonfinite,"logits","nonfinite logit")?;
 }
 let mut maximum=f32::NEG_INFINITY;let mut token=0;
 for lane in 0..8 {if maxima[lane]>maximum || (maxima[lane]==maximum && indices[lane]<token){maximum=maxima[lane];token=indices[lane];}}
 let base=logits.len()/16*8;
 for (i,word) in chunks.remainder().chunks_exact(2).enumerate(){let value=f32::from_bits(u32::from(u16::from_le_bytes([word[0],word[1]]))<<16);check(value.is_finite(),"logits","nonfinite logit")?;if value>maximum{maximum=value;token=(base+i)as u32;}}
 Ok(token)
}
pub fn validate_result<'a,const ROWS:usize>(bytes:&'a[u8],e:&Expectation<ROWS>)->Result<(Option<u32>,&'a[u8])>{
    validate(e)?;
    check(e.rows.len()==1 && bytes.len()==RESULT_BYTES,"result","unsupported result shape")?;
    validate_result_row(bytes,e,0)
}
fn validate_result_row<'a,const ROWS:usize>(bytes:&'a[u8],e:&Expectation<ROWS>,index:usize)->Result<(Option<u32>,&'a[u8])>{
    let u32_at=|at:usize|u32::from_le_bytes(bytes[at..at+4].try_into().unwrap());
    check(u32_at(0)==0 && u32_at(12)==0,"status","GPU execution or argmax failed")?;
    let row=&e.rows[index];let progress=row.progress.validate()?;let published=progress.logits_input_row.is_some();
    let token=u32_at(8);check(token<49152,"argmax","token out of vocabulary")?;
    let mut expected=[0u8;128];result_row_identity_into(&mut expected,e,index,token)?;
    check(bytes[..128]==expected,"result_identity","completion differs from outstanding expectation")?;
    let logits=&bytes[128..];
    if !published {check(all_zero(logits) && token==0,"partial_output","partial prefill produced output")?;return Ok((None,logits));}
    let host_token=finite_argmax(logits)?;
    check(token==host_token,"argmax","token differs from published logits")?;
    Ok((Some(token),logits))
}
#[cfg(test)]
fn result_identity_into<const ROWS:usize>(out:&mut[u8;128],e:&Expectation<ROWS>,token:u32)->Result<()>{result_row_identity_into(out,e,0,token)}
pub(super) fn result_row_identity_into<const ROWS:usize>(out:&mut[u8;128],e:&Expectation<ROWS>,index:usize,token:u32)->Result<()>{
    let row=&e.rows[index];let p=row.progress;let v=p.validate()?;out.fill(0);
    for(at,x)in[(4,u32::from(v.logits_input_row.is_some())),(8,token),(56,v.target_tokens),(60,p.prompt_tokens),(64,p.generated_index),(68,p.input_tokens),(72,p.context_tokens),(76,stage_number(p.stage)),(112,v.last_position),(116,row.output_slot),(120,e.mode as u32),(124,result_magic(e))]{u32_at(out,at,x);}
    for(at,x)in[(16,e.owner_generation),(24,e.replay_id),(32,e.iteration_id),(40,row.sequence_tag),(48,row.cookie)]{u64_at(out,at,x);}
    out[80..112].copy_from_slice(&e.catalog_digest);Ok(())
}
/// Fixed-capacity readback. Unused records must be zero, and no record is
/// returned until every active identity/status/logit row has validated.
pub const BATCH_RESULT_BYTES:usize=8*RESULT_BYTES;
#[derive(Debug)]
pub struct RowResult<'a>{pub output_slot:u32,pub token:Option<u32>,pub logits:&'a[u8]}
pub fn validate_batch_result<'a,const ROWS:usize>(bytes:&'a[u8],e:&Expectation<ROWS>)->Result<Vec<RowResult<'a>>>{
    validate(e)?;
    check(bytes.len()==Layout::<ROWS>::BATCH_RESULT_BYTES,"result","exact capacity result extent required")?;
    let mut result=Vec::with_capacity(e.rows.len());
    for (index,row) in e.rows.iter().enumerate(){
        let record=&bytes[index*RESULT_BYTES..(index+1)*RESULT_BYTES];
        let (token,logits)=validate_result_row(record,e,index)?;
        result.push(RowResult{output_slot:row.output_slot,token,logits});
    }
    check(all_zero(&bytes[e.rows.len()*RESULT_BYTES..]),"inactive_result","inactive record published bytes")?;
    Ok(result)
}

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
        u32_at(&mut expected,124,compact_magic(e));
        check(record==expected,"result_identity","compact identity or GPU validation status differs")?;
        result.push(RowResult{output_slot:row.output_slot,token:published.then_some(token),logits:&[]});
    }
    check(all_zero(&bytes[e.rows.len()*HEADER_BYTES..]),"inactive_result","inactive compact record published bytes")?;
    Ok(result)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub(crate) fn compact_fixture<const ROWS:usize>(e:&Expectation<ROWS>)->Vec<u8>{
        let mut bytes=vec![0;Layout::<ROWS>::COMPACT_RESULT_BYTES];
        for i in 0..e.rows.len(){let mut h=[0;128];let token=if e.rows[i].progress.validate().unwrap().logits_input_row.is_some(){i as u32+7}else{0};
            result_row_identity_into(&mut h,e,i,token).unwrap();u32_at(&mut h,124,compact_magic(e));bytes[i*128..(i+1)*128].copy_from_slice(&h);}
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
    #[test] fn compact_thirtytwo_identity_status_and_publication(){compact_contract::<32>();}
    fn fixture(stage:InputStage,active:u32)->Expectation {fixture_rows::<8>(stage,active)}
    pub(crate) fn fixture_rows<const ROWS:usize>(stage:InputStage,active:u32)->Expectation<ROWS> {
        let mut e=Expectation{owner_generation:1,last_accepted_replay:4,replay_id:5,iteration_id:7,catalog_digest:[19;32],physical_block_count:4096,max_active_rows:ROWS as u32,stage,mode:ResultMode::Greedy,rows:vec![],block_ownership:vec![],packed_prefill:false,mixed_execution:false};
        for i in 0..active {
            let progress=if stage==InputStage::Prefill {Progress{prompt_tokens:398,output_limit:128,context_tokens:1024,committed_tokens:128,input_tokens:73,generated_index:0,stage}} else {Progress{prompt_tokens:129+i*17,output_limit:128,context_tokens:1024,committed_tokens:129+i*17+63,input_tokens:1,generated_index:64,stage}};
            let v=progress.validate().unwrap();let ids:Vec<_>=(0..v.live_pages).map(|p|i*(if ROWS==32{128}else{256})+p).collect();
            e.block_ownership.extend(ids.iter().map(|&physical_id|BlockOwnership{physical_id,sequence_tag:u64::from(i)+1}));
            let mut valid=vec![16;ids.len()];*valid.last_mut().unwrap()=v.last_page_tokens;
            e.rows.push(Row{sequence_tag:u64::from(i)+1,cookie:u64::from(i)+29,output_slot:active-1-i,progress,input_tokens:vec![33;progress.input_tokens as usize],physical_ids:ids,valid_tokens:valid});
        } e
    }

    fn packed_fixture(active:u32,chunk:u32)->Expectation<32>{
        let mut e=fixture_rows::<32>(InputStage::Prefill,active);e.packed_prefill=true;
        e.block_ownership.clear();
        let published=(active+1)/2;let(mut output,mut partial)=(0,published);
        for (i,row) in e.rows.iter_mut().enumerate(){
            let committed=13*i as u32;let target=committed+chunk;
            row.progress=Progress{prompt_tokens:target+if i%2==0{0}else{7},output_limit:32,context_tokens:4096,committed_tokens:committed,input_tokens:chunk,generated_index:0,stage:InputStage::Prefill};
            row.input_tokens=(0..chunk).map(|j|(i as u32*97+j)%49152).collect();
            row.output_slot=if i%2==0{let n=output;output+=1;n}else{let n=partial;partial+=1;n};
            let v=row.progress.validate().unwrap();row.physical_ids=(0..v.live_pages).map(|j|i as u32*128+j).collect();row.valid_tokens=vec![16;v.live_pages as usize];*row.valid_tokens.last_mut().unwrap()=v.last_page_tokens;
            e.block_ownership.extend(row.physical_ids.iter().map(|&physical_id|BlockOwnership{physical_id,sequence_tag:row.sequence_tag}));
        }e
    }
    #[test] fn packed_v6_offsets_publication_and_corruption(){
        let export=std::env::var_os("RILEY_V48_WIRE_FIXTURES").map(std::path::PathBuf::from);
        if let Some(d)=&export{std::fs::create_dir_all(d).unwrap();}
        for active in 1..=4 {for chunk in [1,73,128,256] {
            let e=packed_fixture(active,chunk);let mut p=vec![0;Layout::<32>::REQUEST_BYTES];encode_into(&mut p,&e).unwrap();
            assert_eq!(u32::from_le_bytes(p[36..40].try_into().unwrap()),active*chunk);
            let mut offset=0;
            for (i,row) in e.rows.iter().enumerate(){let b=HEADER_BYTES+i*ROW_BYTES;assert_eq!(&p[b+64..b+68],&(offset as u32).to_le_bytes());for &token in &row.input_tokens{let at=Layout::<32>::TOKENS_OFFSET+offset*4;assert_eq!(&p[at..at+4],&token.to_le_bytes());offset+=1;}}
            let mut scratch=vec![0;p.len()];validate_packet(&p,&mut scratch,&e).unwrap();
            for at in 0..p.len(){p[at]^=0x80;assert!(validate_packet(&p,&mut scratch,&e).is_err(),"byte{at}");p[at]^=0x80;}
            let compact=compact_fixture(&e);let rows=validate_compact_result(&compact,&e).unwrap();assert_eq!(rows.iter().filter(|r|r.token.is_some()).count(),((active+1)/2)as usize);
            let mut wrong=e.clone();wrong.packed_prefill=false;assert!(validate_compact_result(&compact,&wrong).is_err());
            if active>1{let mut wrong=e.clone();wrong.rows.swap(0,1);let slot=wrong.rows[0].output_slot;wrong.rows[0].output_slot=wrong.rows[1].output_slot;wrong.rows[1].output_slot=slot;assert!(validate(&wrong).is_err());}
            if let Some(d)=&export{std::fs::write(d.join(format!("v6-prefill-{active}-{chunk}.bin")),&p).unwrap();}
        }}
        for active in [1,16,32]{let mut e=fixture_rows::<32>(InputStage::Decode,active);e.packed_prefill=true;let mut p=vec![0;Layout::<32>::REQUEST_BYTES];encode_into(&mut p,&e).unwrap();assert_eq!(&p[36..40],&[0;4]);if let Some(d)=&export{std::fs::write(d.join(format!("v6-decode-{active}.bin")),p).unwrap();}}
    }
    #[test] fn packed_v6_rejects_capacity_before_write(){
        for e in [packed_fixture(4,257),packed_fixture(5,128)]{let mut p=vec![0xa5;Layout::<32>::REQUEST_BYTES];assert!(encode_into(&mut p,&e).is_err());assert!(p.iter().all(|&x|x==0xa5));}
        let mut e=fixture_rows::<16>(InputStage::Prefill,1);e.packed_prefill=true;assert!(validate(&e).is_err());
        let mut e=packed_fixture(4,256);e.rows[1].physical_ids[0]=e.rows[0].physical_ids[0];assert!(validate(&e).is_err());
    }

    fn mixed_fixture(prefills:u32,decodes:u32,chunk:u32)->Expectation<32>{
        let mut e=packed_fixture(prefills,chunk);e.mixed_execution=true;e.stage=if prefills>0{InputStage::Prefill}else{InputStage::Decode};
        let decode=fixture_rows::<32>(InputStage::Decode,decodes);
        for mut row in decode.rows{row.sequence_tag+=u64::from(prefills);row.cookie+=u64::from(prefills);for p in &mut row.physical_ids{*p+=prefills*128;}e.rows.push(row);}
        e.block_ownership=e.rows.iter().flat_map(|r|r.physical_ids.iter().map(move |&physical_id|BlockOwnership{physical_id,sequence_tag:r.sequence_tag})).collect();
        let mut published=0;let mut partial=e.rows.iter().filter(|r|r.progress.validate().unwrap().logits_input_row.is_some()).count() as u32;
        for row in &mut e.rows{row.output_slot=if row.progress.validate().unwrap().logits_input_row.is_some(){let n=published;published+=1;n}else{let n=partial;partial+=1;n};}e
    }
    #[test] fn mixed_v7_stage_tiles_and_result_identity(){
        for (prefills,decodes,chunk) in [(0,32,1),(1,31,128),(4,28,249),(4,0,256),(2,3,31)] {
            let e=mixed_fixture(prefills,decodes,chunk);let mut p=vec![0;MIXED_REQUEST_BYTES];encode_into(&mut p,&e).unwrap();
            let get=|at|u32::from_le_bytes(p[at..at+4].try_into().unwrap());assert_eq!(get(0),0x37444d52);assert_eq!(get(4),7);assert_eq!(get(16),if prefills==0{1}else if decodes==0{0}else{2});
            let mut tile=0;for (owner,row) in e.rows.iter().enumerate(){let b=128+owner*ROW_BYTES;assert_eq!(get(b+72),stage_number(row.progress.stage));if prefills>0{assert_eq!(get(b+68),tile);let n=row.progress.input_tokens;for local in 0..if n<32{n}else{n.div_ceil(8)}{assert_eq!(get(Layout::<32>::REQUEST_BYTES+tile as usize*4),((owner as u32)<<16)|local);tile+=1;}}}
            assert_eq!(get(96),tile);let mut scratch=vec![0;p.len()];for at in 0..p.len(){p[at]^=0x80;assert!(validate_packet(&p,&mut scratch,&e).is_err(),"byte{at}");p[at]^=0x80;}
            let bytes=compact_fixture(&e);assert_eq!(validate_compact_result(&bytes,&e).unwrap().len(),(prefills+decodes)as usize);
            let mut wrong=e.clone();wrong.mixed_execution=false;assert!(validate_compact_result(&bytes,&wrong).is_err());
            if let Some(dir)=std::env::var_os("RILEY_V49_WIRE_FIXTURES"){let dir=std::path::PathBuf::from(dir);std::fs::create_dir_all(&dir).unwrap();std::fs::write(dir.join(format!("v7-{prefills}-{decodes}-{chunk}.bin")),p).unwrap();}
        }
    }
    #[test] fn mixed_v7_over_budget_and_stage_fail_before_write(){
        let e=mixed_fixture(4,28,250);let mut p=vec![0xa5;MIXED_REQUEST_BYTES];assert!(encode_into(&mut p,&e).is_err());assert!(p.iter().all(|&x|x==0xa5));
        let mut e=mixed_fixture(1,3,128);e.stage=InputStage::Decode;assert!(validate(&e).is_err());e.stage=InputStage::Prefill;e.packed_prefill=false;assert!(validate(&e).is_err());
        let e=mixed_fixture(1,3,128);assert!(encode_into(&mut vec![0;Layout::<32>::REQUEST_BYTES],&e).is_err());
    }
    fn maximum_prefill16()->Expectation<16> {
        let mut e=fixture_rows::<16>(InputStage::Prefill,1);
        e.rows[0].progress=Progress{prompt_tokens:2048,output_limit:128,context_tokens:4096,committed_tokens:0,input_tokens:1024,generated_index:0,stage:InputStage::Prefill};
        e.rows[0].input_tokens=(0..1024).map(|i|i*31%49152).collect();
        e.rows[0].physical_ids=(0..64).collect();e.rows[0].valid_tokens=vec![16;64];
        e.block_ownership=(0..64).map(|physical_id|BlockOwnership{physical_id,sequence_tag:1}).collect();e
    }
    #[test] fn maximum_prefill16_tokens_do_not_overlap_unused_descriptors() {
        let e=maximum_prefill16();let mut packet=vec![0xa5;Layout::<16>::REQUEST_BYTES];encode_into(&mut packet,&e).unwrap();
        assert!(packet[HEADER_BYTES+ROW_BYTES..Layout::<16>::TOKENS_OFFSET].iter().all(|&x|x==0));
        for(i,&token)in e.rows[0].input_tokens.iter().enumerate(){let at=Layout::<16>::TOKENS_OFFSET+i*4;assert_eq!(&packet[at..at+4],&token.to_le_bytes());}
        let mut result=vec![0;Layout::<16>::BATCH_RESULT_BYTES];let mut header=[0;128];result_identity_into(&mut header,&e,0).unwrap();result[..128].copy_from_slice(&header);
        assert_eq!(validate_batch_result(&result,&e).unwrap()[0].token,None);
        result[128]=1;assert!(validate_batch_result(&result,&e).is_err());
    }
    #[test] fn export_versioned_wire_fixtures_when_requested() {
        let Some(dir)=std::env::var_os("RILEY_V39_WIRE_FIXTURES") else {return;};
        let dir=std::path::PathBuf::from(dir);std::fs::create_dir_all(&dir).unwrap();
        for active in [1,2,4,8,9,15,16] {
            let e=fixture_rows::<16>(InputStage::Decode,active);let mut packet=vec![0;Layout::<16>::REQUEST_BYTES];encode_into(&mut packet,&e).unwrap();
            std::fs::write(dir.join(format!("v4-decode-{active}.bin")),packet).unwrap();
        }
        let e=fixture_rows::<16>(InputStage::Prefill,1);let mut packet=vec![0;Layout::<16>::REQUEST_BYTES];encode_into(&mut packet,&e).unwrap();
        std::fs::write(dir.join("v4-prefill.bin"),packet).unwrap();
        let e=maximum_prefill16();let mut packet=vec![0;Layout::<16>::REQUEST_BYTES];encode_into(&mut packet,&e).unwrap();
        std::fs::write(dir.join("v4-prefill-max.bin"),packet).unwrap();
        for(stage,name)in[(InputStage::Decode,"v3-decode.bin"),(InputStage::Prefill,"v3-prefill.bin")] {
            let e=fixture(stage,if stage==InputStage::Decode{8}else{1});let mut packet=vec![0;REQUEST_BYTES];encode_into(&mut packet,&e).unwrap();
            std::fs::write(dir.join(name),packet).unwrap();
        }
    }
    #[test] fn sixteen_row_contract_rejects_old_versions_and_binds_all_bytes() {
        assert_eq!(Layout::<8>::REQUEST_BYTES,REQUEST_BYTES);
        assert_eq!(Layout::<8>::BATCH_RESULT_BYTES,BATCH_RESULT_BYTES);
        assert_eq!(Layout::<16>::TOKENS_OFFSET,26752);
        assert_eq!(Layout::<16>::REQUEST_BYTES,30848);
        assert_eq!(Layout::<16>::BATCH_RESULT_BYTES,1574912);
        assert!(Layout::<16>::REQUEST_BYTES<Layout::<16>::OUTPUT_OFFSET);
        assert_eq!(Layout::<16>::STAGING_BYTES,3149824);
        for stage in [InputStage::Prefill,InputStage::Decode] {
            let e=fixture_rows::<16>(stage,if stage==InputStage::Prefill{1}else{16});
            let mut packet=vec![0;Layout::<16>::REQUEST_BYTES];let mut scratch=packet.clone();
            encode_into(&mut packet,&e).unwrap();
            for offset in 0..packet.len() {packet[offset]^=1;assert!(validate_packet(&packet,&mut scratch,&e).is_err(),"offset{offset}");packet[offset]^=1;}
            validate_packet(&packet,&mut scratch,&e).unwrap();
            if stage==InputStage::Prefill {
                assert!(packet[HEADER_BYTES+ROW_BYTES..Layout::<16>::TOKENS_OFFSET].iter().all(|&x|x==0));
                assert_eq!(&packet[Layout::<16>::TOKENS_OFFSET..Layout::<16>::TOKENS_OFFSET+4],&33u32.to_le_bytes());
            }
            let old=fixture(stage,if stage==InputStage::Prefill{1}else{8});
            assert!(validate_packet(&packet,&mut vec![0;REQUEST_BYTES],&old).is_err());
            let mut old_packet=vec![0;REQUEST_BYTES];encode_into(&mut old_packet,&old).unwrap();
            assert!(validate_packet(&old_packet,&mut scratch,&e).is_err());
            // Matching extents cannot disguise old magic/version, even for one row.
            u32_at(&mut packet,0,MAGIC);u32_at(&mut packet,4,3);
            assert!(validate_packet(&packet,&mut scratch,&e).is_err());
        }
    }
    #[test] fn sixteen_row_results_bind_upper_rows_and_reject_old_single_result_magic() {
        for active in [1,2,4,8,9,15,16] {
            let e=fixture_rows::<16>(InputStage::Decode,active);
            let mut bytes=vec![0;Layout::<16>::BATCH_RESULT_BYTES];
            for i in 0..active as usize {
                let mut h=[0;128];result_row_identity_into(&mut h,&e,i,0).unwrap();
                bytes[i*RESULT_BYTES..i*RESULT_BYTES+128].copy_from_slice(&h);
            }
            let rows=validate_batch_result(&bytes,&e).unwrap();assert_eq!(rows.len(),active as usize);
            for(i,row)in rows.iter().enumerate(){assert_eq!(row.output_slot,active-1-i as u32);assert_eq!(row.token,Some(0));}
            for row in 0..active as usize {for offset in 0..128 {
                let at=row*RESULT_BYTES+offset;bytes[at]^=1;assert!(validate_batch_result(&bytes,&e).is_err());bytes[at]^=1;
            }}
            if active<16 {for at in [active as usize*RESULT_BYTES,bytes.len()-1] {bytes[at]=1;assert!(validate_batch_result(&bytes,&e).is_err());bytes[at]=0;}}
            let at=(active as usize-1)*RESULT_BYTES+128;
            bytes[at..at+2].copy_from_slice(&0x7f80u16.to_le_bytes());assert!(validate_batch_result(&bytes,&e).is_err());
        }
        let e=fixture_rows::<16>(InputStage::Decode,1);let old=fixture(InputStage::Decode,1);
        let mut bytes=vec![0;RESULT_BYTES];let mut h=[0;128];result_identity_into(&mut h,&e,0).unwrap();bytes[..128].copy_from_slice(&h);
        validate_result(&bytes,&e).unwrap();assert!(validate_result(&bytes,&old).is_err());
        result_identity_into(&mut h,&old,0).unwrap();bytes[..128].copy_from_slice(&h);
        validate_result(&bytes,&old).unwrap();assert!(validate_result(&bytes,&e).is_err());
    }
    #[test] fn sixteen_row_invalid_authority_and_capacity_never_mutate_packet() {
        let e=fixture_rows::<16>(InputStage::Decode,16);let mut cases=vec![];
        let mut x=e.clone();x.max_active_rows=8;cases.push(x);
        let mut x=e.clone();x.max_active_rows=32;cases.push(x);
        let mut x=e.clone();x.rows[15].physical_ids[0]=x.rows[0].physical_ids[0];cases.push(x);
        let mut x=e.clone();x.rows[15].cookie=x.rows[0].cookie;cases.push(x);
        let mut x=e.clone();x.rows[15].output_slot=x.rows[0].output_slot;cases.push(x);
        let mut x=e.clone();x.replay_id=4;cases.push(x);
        let mut x=e.clone();x.rows.push(x.rows[0].clone());cases.push(x);
        for x in cases {let mut packet=vec![0xa5;Layout::<16>::REQUEST_BYTES];assert!(encode_into(&mut packet,&x).is_err());assert!(packet.iter().all(|&x|x==0xa5));}
        let unsupported=fixture_rows::<64>(InputStage::Decode,1);let mut packet=vec![0xa5;Layout::<64>::REQUEST_BYTES];
        assert!(encode_into(&mut packet,&unsupported).is_err());assert!(packet.iter().all(|&x|x==0xa5));
        let mut old=fixture(InputStage::Decode,8);old.max_active_rows=16;
        assert!(validate(&old).is_err());
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

    #[test] fn single_decode_completion_is_bound_to_its_generated_index(){
        let mut e=fixture(InputStage::Decode,1);let mut bytes=vec![0;RESULT_BYTES];let mut header=[0;128];result_identity_into(&mut header,&e,0).unwrap();bytes[..128].copy_from_slice(&header);
        assert_eq!(validate_result(&bytes,&e).unwrap().0,Some(0));
        e.rows[0].progress.generated_index+=1;e.rows[0].progress.committed_tokens+=1;*e.rows[0].valid_tokens.last_mut().unwrap()+=1;
        assert!(validate_result(&bytes,&e).is_err());
    }

    #[test] fn batch_results_bind_all_rows_before_publication(){
        for active in [1,2,4,8]{
            let e=fixture(InputStage::Decode,active);let mut bytes=vec![0;BATCH_RESULT_BYTES];
            for i in 0..active as usize {let mut h=[0;128];result_row_identity_into(&mut h,&e,i,i as u32+7).unwrap();let b=i*RESULT_BYTES;bytes[b..b+128].copy_from_slice(&h);bytes[b+128+(i+7)*2..b+130+(i+7)*2].copy_from_slice(&0x3f80u16.to_le_bytes());}
            if let Some(dir)=std::env::var_os("RILEY_V18_FIXTURES") {
                let dir=std::path::PathBuf::from(dir);std::fs::create_dir_all(&dir).unwrap();let mut packet=vec![0;REQUEST_BYTES];encode_into(&mut packet,&e).unwrap();
                std::fs::write(dir.join(format!("request-{active}.bin")),packet).unwrap();std::fs::write(dir.join(format!("result-{active}.bin")),&bytes).unwrap();
            }
            if let Some(dir)=std::env::var_os("RILEY_V18_GPU_RESULTS") {
                let gpu=std::fs::read(std::path::PathBuf::from(dir).join(format!("gpu-result-{active}.bin"))).unwrap();
                assert_eq!(gpu,bytes);assert_eq!(validate_batch_result(&gpu,&e).unwrap().len(),active as usize);
            }
            let rows=validate_batch_result(&bytes,&e).unwrap();assert_eq!(rows.len(),active as usize);
            for(i,row)in rows.iter().enumerate(){assert_eq!(row.output_slot,active-1-i as u32);assert_eq!(row.token,Some(i as u32+7));}
            for row in 0..active as usize {for offset in 0..128 {let at=row*RESULT_BYTES+offset;bytes[at]^=1;assert!(validate_batch_result(&bytes,&e).is_err());bytes[at]^=1;}}
            if active<8 {let at=active as usize*RESULT_BYTES;bytes[at]=1;assert!(validate_batch_result(&bytes,&e).is_err());bytes[at]=0;}
            let last=(active as usize-1)*RESULT_BYTES;bytes[last+128..last+130].copy_from_slice(&0x7fc0u16.to_le_bytes());assert!(validate_batch_result(&bytes,&e).is_err());
        }
    }

}

#[cfg(test)]
mod lane_argmax_tests {
 use super::*;
 #[test]
 fn independent_lanes_match_scalar_and_reject_nonfinite(){
  for size in [1,7,8,9,16,257,49152] {for seed in 0..8 {
   let words:Vec<u16>=(0..size).map(|i|match (i+seed)%13 {0=>0,1=>0x8000,_=>((i*139+seed*73)%0x7f80)as u16 | if i%2==0{0x8000}else{0}}).collect();let mut bytes:Vec<_>=words.iter().flat_map(|x|x.to_le_bytes()).collect();
   let mut maximum=f32::NEG_INFINITY;let mut expected=0;
   for(i,&word)in words.iter().enumerate(){let value=f32::from_bits(u32::from(word)<<16);if value>maximum{maximum=value;expected=i as u32;}}
   assert_eq!(finite_argmax(&bytes).unwrap(),expected);
   for bad in [0x7f80u16,0xff80,0x7fc0,0xff81] {for at in [0,size/2,size-1]{let old=[bytes[at*2],bytes[at*2+1]];bytes[at*2..at*2+2].copy_from_slice(&bad.to_le_bytes());assert!(finite_argmax(&bytes).is_err());bytes[at*2..at*2+2].copy_from_slice(&old);}}
  }}
  for winner in 0..32 {let mut values=vec![0xbf80u16;32];values[winner]=0;values[(winner+8).min(31)]=0x8000;let bytes:Vec<_>=values.iter().flat_map(|x|x.to_le_bytes()).collect();assert_eq!(finite_argmax(&bytes).unwrap(),winner as u32);}
 }
}
