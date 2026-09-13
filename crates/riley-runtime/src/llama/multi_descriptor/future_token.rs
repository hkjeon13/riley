//! Tentative wire preparation, not permission to launch or evidence of commit.
//! The scheduler must own both reservations and retain all predecessor pages.
use super::{check, overflow, Result, ResultMode};
use super::shape_progress::InputStage;
use super::variable_wire::{self as wire, Expectation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenSource { Host, PreviousRow(u32) }

/// Native Reference: source row, 32 canonical result words, destination cookie.
pub const REFERENCE_BYTES: usize = 140;
pub const SIDECAR_BYTES: usize = 32 * REFERENCE_BYTES;

pub struct PreparedFutureBatch {
    packet: Vec<u8>,
    references: Vec<u8>,
    predecessor_replay: u64,
    committed_replay: u64,
}
impl PreparedFutureBatch {
    pub fn packet(&self) -> &[u8] { &self.packet }
    pub fn references(&self) -> &[u8] { &self.references }
    pub fn predecessor_replay(&self) -> u64 { self.predecessor_replay }
    /// This remains the real last accepted replay, not the tentative predecessor.
    pub fn committed_replay(&self) -> u64 { self.committed_replay }
}

/// Owns the exact expectations whose future bytes were validated and encoded.
/// No mutable expectation or packet access escapes after construction.
/// ```compile_fail,E0594
/// use riley_runtime::llama::multi_descriptor::future_token::PreparedFutureWindow;
/// fn cannot_change_checked_owner(window: &mut PreparedFutureWindow) {
///     window.first().owner_generation += 1;
/// }
/// ```
pub struct PreparedFutureWindow {
    first: Expectation<32>,
    second: Expectation<32>,
    future: PreparedFutureBatch,
}
impl PreparedFutureWindow {
    pub fn new(first: Expectation<32>, second: Expectation<32>, sources: &[TokenSource]) -> Result<Self> {
        let future = prepare(&first, &second, sources)?;
        Ok(Self { first, second, future })
    }
    pub fn first(&self) -> &Expectation<32> { &self.first }
    pub fn second(&self) -> &Expectation<32> { &self.second }
    pub fn future(&self) -> &PreparedFutureBatch { &self.future }
    #[cfg(feature = "cuda")]
    pub(crate) fn into_parts(self) -> (Expectation<32>, Expectation<32>, PreparedFutureBatch) {
        (self.first, self.second, self.future)
    }
}

/// Validate the pair and encode a tentative successor without changing either
/// expectation. Future tokens use an explicit zero placeholder. This object
/// cannot be submitted through the existing single-inflight session API.
pub fn prepare(previous: &Expectation<32>, successor: &Expectation<32>, sources: &[TokenSource]) -> Result<PreparedFutureBatch> {
    let checked=wire::checked_expectation(previous)?;
    prepare_checked(&checked,successor,sources)
}
pub(crate) fn prepare_checked(checked_previous:&wire::CheckedExpectation<'_,32>, successor:&Expectation<32>, sources:&[TokenSource])->Result<PreparedFutureBatch> {
    let previous=checked_previous.expectation();
    check(previous.shared_prefixes==successor.shared_prefixes && previous.mixed_execution && successor.mixed_execution && previous.mode==ResultMode::Greedy && successor.mode==ResultMode::Greedy,
          "future_profile", "requires V7 mixed32 greedy expectations")?;
    check(previous.owner_generation==successor.owner_generation && previous.catalog_digest==successor.catalog_digest && previous.physical_block_count==successor.physical_block_count,
          "future_owner", "retained owner differs")?;
    check(successor.last_accepted_replay==previous.last_accepted_replay &&
          previous.replay_id.checked_add(1)==Some(successor.replay_id) &&
          previous.iteration_id.checked_add(1)==Some(successor.iteration_id),
          "future_order", "successor must depend on uncommitted predecessor")?;
    check(sources.len()==successor.rows.len(), "future_rows", "source count differs")?;
    // Structural validation needs the immediate predecessor number. This clone
    // never changes session state and is not returned as an accepted expectation.
    let mut structural=successor.clone();
    structural.last_accepted_replay=previous.replay_id;
    let checked_structural = wire::checked_expectation(&structural)?;
    let last_cookie=previous.rows.iter().map(|r|r.cookie).max().unwrap();
    check(successor.rows.iter().all(|r|r.cookie>last_cookie), "future_cookie", "successor cookies must be freshly issued")?;
    check(checked_structural.retains_ownership(previous),
          "future_pages", "predecessor ownership must remain retained")?;
    let mut references=vec![0u8;SIDECAR_BYTES];
    for row in 0..32 {references[row*REFERENCE_BYTES..row*REFERENCE_BYTES+4].copy_from_slice(&u32::MAX.to_le_bytes());}
    let mut used=0u32;
    for (index,source) in sources.iter().enumerate() {
        let TokenSource::PreviousRow(source)=*source else {
            let tag=successor.rows[index].sequence_tag;
            for old in previous.rows.iter().filter(|r|r.sequence_tag==tag) {
                let new=&successor.rows[index];let p=old.progress.validate()?;
                check(p.logits_input_row.is_none(), "future_token", "published predecessor requires a future token reference")?;
                check(new.progress.stage==InputStage::Prefill && new.progress.committed_tokens==p.target_tokens && new.progress.prompt_tokens==old.progress.prompt_tokens && new.progress.output_limit==old.progress.output_limit && new.progress.context_tokens==old.progress.context_tokens && new.physical_ids.starts_with(&old.physical_ids),
                      "future_prefill", "prompt continuation must append to retained predecessor pages")?;
            }
            continue
        };
        check((source as usize)<previous.rows.len() && (used&(1u32<<source))==0, "future_source", "invalid or reused source row")?;
        used|=1u32<<source;
        let old=&previous.rows[source as usize];let new=&successor.rows[index];
        let progress=old.progress.validate()?;
        check(progress.logits_input_row.is_some() && old.sequence_tag==new.sequence_tag && new.progress.stage==InputStage::Decode && new.input_tokens==[0],
              "future_token", "source must publish for the same request and successor must use a decode placeholder")?;
        check(new.progress.committed_tokens==progress.target_tokens &&
              old.progress.generated_index.checked_add(1)==Some(new.progress.generated_index) &&
              old.progress.prompt_tokens==new.progress.prompt_tokens && old.progress.output_limit==new.progress.output_limit && old.progress.context_tokens==new.progress.context_tokens,
              "future_progress", "successor does not append the next decode token")?;
        check(new.physical_ids.starts_with(&old.physical_ids), "future_pages", "successor remaps predecessor pages")?;
        let mut expected=[0u8;128];wire::result_row_identity_into(&mut expected,previous,source as usize,0)?;
        expected[124..128].copy_from_slice(&0xb7524d52u32.to_le_bytes());
        let base=index.checked_mul(REFERENCE_BYTES).ok_or_else(||overflow("future_reference"))?;
        references[base..base+4].copy_from_slice(&source.to_le_bytes());
        references[base+4..base+132].copy_from_slice(&expected);
        references[base+132..base+140].copy_from_slice(&new.cookie.to_le_bytes());
    }
    let mut packet=vec![0;wire::request_bytes(&structural)];
    wire::encode_checked_into(&mut packet,&checked_structural)?;
    Ok(PreparedFutureBatch {packet,references,predecessor_replay:previous.replay_id,committed_replay:previous.last_accepted_replay})
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{BlockOwnership,shape_progress::Progress};
    use wire::Row;
    fn pair()->(Expectation<32>,Expectation<32>) {
        let old=Expectation {owner_generation:7,last_accepted_replay:9,replay_id:10,iteration_id:20,catalog_digest:[1;32],physical_block_count:8,max_active_rows:32,stage:InputStage::Prefill,mode:ResultMode::Greedy,packed_prefill:true,shared_prefixes:false,mixed_execution:true,
            rows:(0..2).map(|i|Row {sequence_tag:10+i as u64,cookie:100+i as u64,output_slot:i,progress:Progress {prompt_tokens:16,output_limit:4,context_tokens:64,committed_tokens:0,input_tokens:16,generated_index:0,stage:InputStage::Prefill},input_tokens:vec![3;16],physical_ids:vec![i],valid_tokens:vec![16]}).collect(),
            block_ownership:(0..2).map(|i|BlockOwnership{physical_id:i,sequence_tag:10+i as u64}).collect()};
        let mut new=old.clone();new.replay_id=11;new.iteration_id=21;new.stage=InputStage::Decode;new.rows.reverse();
        for(i,row)in new.rows.iter_mut().enumerate(){row.cookie=200+i as u64;row.output_slot=i as u32;row.progress.stage=InputStage::Decode;row.progress.committed_tokens=16;row.progress.input_tokens=1;row.progress.generated_index=1;row.input_tokens=vec![0];row.physical_ids.push(2+i as u32);row.valid_tokens.push(1);new.block_ownership.push(BlockOwnership{physical_id:2+i as u32,sequence_tag:row.sequence_tag});}
        (old,new)
    }
    fn export(old:&Expectation<32>,prepared:&PreparedFutureBatch,child:&str){
        if let Ok(directory)=std::env::var("RILEY_FUTURE_WIRE_FIXTURE") {
            let dir=std::path::Path::new(&directory).join(child);std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("packet.bin"),prepared.packet()).unwrap();
            std::fs::write(dir.join("references.bin"),prepared.references()).unwrap();
            let mut previous=vec![0u8;4096];
            for i in 0..old.rows.len(){let mut record=[0;128];wire::result_row_identity_into(&mut record,&old,i,0).unwrap();record[124..128].copy_from_slice(&0xb7524d52u32.to_le_bytes());previous[i*128..(i+1)*128].copy_from_slice(&record);}
            std::fs::write(dir.join("previous.bin"),previous).unwrap();
        }
    }
    #[test] fn page_boundary_reorder_preserves_commit_and_encodes_distinct_cookies(){
        let(old,new)=pair();assert!(wire::validate(&new).is_err());
        let prepared=prepare(&old,&new,&[TokenSource::PreviousRow(1),TokenSource::PreviousRow(0)]).unwrap();
        assert_eq!(prepared.committed_replay(),9);assert_eq!(prepared.predecessor_replay(),10);assert_eq!(new.last_accepted_replay,9);
        assert_eq!(prepared.references().len(),4480);
        assert_eq!(u32::from_le_bytes(prepared.references()[0..4].try_into().unwrap()),1);
        assert_eq!(u64::from_le_bytes(prepared.references()[52..60].try_into().unwrap()),101);
        assert_eq!(u64::from_le_bytes(prepared.references()[132..140].try_into().unwrap()),200);
        assert_eq!(&prepared.packet()[128..132],&[0;4]);
        export(&old,&prepared,"");
    }
    #[test] fn owned_window_keeps_the_exact_cookie_and_page_binding() {
        let (mut old, mut new) = pair();
        let window = PreparedFutureWindow::new(old.clone(), new.clone(), &[TokenSource::PreviousRow(1), TokenSource::PreviousRow(0)]).unwrap();
        old.rows[1].cookie = 999;
        new.rows[0].physical_ids[0] = 7;
        assert_eq!(window.first().rows[1].cookie, 101);
        assert_eq!(window.second().rows[0].physical_ids, [1, 2]);
        assert_eq!(u64::from_le_bytes(window.future().references()[52..60].try_into().unwrap()), 101);
        assert_eq!(u64::from_le_bytes(window.future().references()[132..140].try_into().unwrap()), 200);
        let mut structural = window.second().clone();
        structural.last_accepted_replay = window.first().replay_id;
        let mut expected = vec![0; wire::request_bytes(&structural)];
        wire::encode_into(&mut expected, &structural).unwrap();
        assert_eq!(window.future().packet(), expected);
    }
    #[test] fn host_prefill_admission_keeps_previous_pages_retained(){
        let(old,mut new)=pair();new.stage=InputStage::Prefill;
        let row=&mut new.rows[0];row.sequence_tag=99;row.progress=old.rows[0].progress;
        row.input_tokens=vec![3;16];row.physical_ids=vec![4];row.valid_tokens=vec![16];
        new.block_ownership.push(BlockOwnership{physical_id:4,sequence_tag:99});
        let p=prepare(&old,&new,&[TokenSource::Host,TokenSource::PreviousRow(0)]).unwrap();
        assert_eq!(&p.references()[..4],&u32::MAX.to_le_bytes());
        assert_eq!(p.packet().len(),wire::MIXED_REQUEST_BYTES);
        export(&old,&p,"mixed");
    }
    #[test] fn partial_prefill_can_continue_without_inventing_a_sampled_token(){
        let(mut old,mut new)=pair();for row in &mut old.rows {row.progress.prompt_tokens=32;}
        new.stage=InputStage::Prefill;
        for row in &mut new.rows {row.progress.prompt_tokens=32;row.progress.stage=InputStage::Prefill;row.progress.input_tokens=16;row.progress.generated_index=0;row.input_tokens=vec![3;16];row.valid_tokens=vec![16,16];}
        assert!(prepare(&old,&new,&[TokenSource::Host;2]).is_ok());
        new.rows[0].physical_ids.swap(0,1);
        assert!(prepare(&old,&new,&[TokenSource::Host;2]).is_err());
    }
    #[test]
    fn indexed_owner_retention_keeps_off_batch_readers_and_is_order_independent() {
        let(mut old,mut new)=pair();
        let extra=BlockOwnership{physical_id:7,sequence_tag:77};
        old.block_ownership.push(extra);new.block_ownership.push(extra);
        let sources=[TokenSource::PreviousRow(1),TokenSource::PreviousRow(0)];
        let expected=prepare(&old,&new,&sources).unwrap();
        new.block_ownership.reverse();
        let owned=wire::OwnedCheckedExpectation::new(old).unwrap();
        let indexed=prepare_checked(&owned.checked(),&new,&sources).unwrap();
        assert_eq!(expected.packet(),indexed.packet());assert_eq!(expected.references(),indexed.references());
        new.block_ownership.retain(|p|p.sequence_tag!=77);
        assert_eq!(prepare_checked(&owned.checked(),&new,&sources).err().unwrap().field,"future_pages");
    }
    #[test] fn invalid_pairs_do_not_gain_authority(){
        let(old,new)=pair();let sources=[TokenSource::PreviousRow(1),TokenSource::PreviousRow(0)];
        for fault in 0..10 {let mut n=new.clone();match fault {
            0=>n.last_accepted_replay=10,1=>n.replay_id=12,2=>n.owner_generation=8,3=>n.rows[0].cookie=101,
            4=>n.rows[0].input_tokens[0]=5,5=>n.rows[0].progress.generated_index=2,
            6=>n.block_ownership.retain(|o|o.physical_id!=0),7=>n.rows[0].physical_ids.swap(0,1),
            8=>n.rows[0].sequence_tag=99,_=>n.catalog_digest=[2;32]}
            assert!(prepare(&old,&n,&sources).is_err(),"fault={fault}");
            assert!(PreparedFutureWindow::new(old.clone(),n,&sources).is_err(),"owned fault={fault}");}
        assert!(prepare(&old,&new,&[TokenSource::PreviousRow(1);2]).is_err());
        assert!(prepare(&old,&new,&[TokenSource::PreviousRow(32);2]).is_err());
        assert!(prepare(&old,&new,&[]).is_err());
        assert!(prepare(&old,&new,&[TokenSource::Host;2]).is_err());
    }
}
