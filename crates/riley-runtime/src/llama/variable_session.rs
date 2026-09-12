//! Retained single-request V3 transactions. Production model preparation is separate.
use super::multi_descriptor::{variable_wire as wire, Error, Result};
use riley_cuda::BorrowedGraphResourceReservation;
use std::sync::atomic::{AtomicU64, Ordering};
static GENERATION: AtomicU64 = AtomicU64::new(1);
#[derive(Clone, Copy)]
pub struct VariableSessionIdentity {
    pub generation:u64, pub last_accepted_replay:u64, pub catalog_digest:[u8;32],
    pub physical_block_count:u32, pub context_tokens:u32,
}
/// Owns the prepared reservation; no mutable native handle escapes while active.
/// A result remains outstanding until scheduler commit is explicitly confirmed.
pub struct BorrowedVariableSession<'a> {
    graph:BorrowedGraphResourceReservation<'a>, identity:VariableSessionIdentity,
    issued:Option<u64>, next_cookie:u64, retained:Option<wire::Expectation>,
    started:bool, poisoned:bool, completed:bool, input:Vec<u8>, output:Vec<u8>,
}
fn bad(reason:&'static str)->Error {Error{field:"V3 retained session",reason}}
impl<'a> BorrowedVariableSession<'a> {
    /// The caller must have recorded the V3 model and bound the digest to its
    /// actual prepared model/kernel catalog. This constructor does not create it.
    pub fn new(graph:BorrowedGraphResourceReservation<'a>,catalog_digest:[u8;32],physical_block_count:u32,context_tokens:u32)->Result<Self> {
        if catalog_digest==[0;32] || !(1..=4096).contains(&physical_block_count) || !(1..=4096).contains(&context_tokens) {return Err(bad("invalid prepared geometry"));}
        let generation=GENERATION.fetch_update(Ordering::Relaxed,Ordering::Relaxed,|n|n.checked_add(1)).map_err(|_|bad("generation exhausted"))?;
        Ok(Self{graph,identity:VariableSessionIdentity{generation,last_accepted_replay:0,catalog_digest,physical_block_count,context_tokens},
            issued:None,next_cookie:1,retained:None,started:false,poisoned:false,completed:false,input:vec![0;wire::REQUEST_BYTES],output:vec![0;wire::RESULT_BYTES]})
    }
    pub fn issue(&mut self)->Result<(VariableSessionIdentity,u64,u64)> {
        if self.poisoned || self.retained.is_some() || self.issued.is_some() {return Err(bad("session busy or poisoned"));}
        let replay=self.identity.last_accepted_replay.checked_add(1).ok_or_else(||bad("replay exhausted"))?;
        let cookie=self.next_cookie;self.next_cookie=cookie.checked_add(1).ok_or_else(||bad("cookie exhausted"))?;
        self.issued=Some(cookie);self.started=false;Ok((self.identity,replay,cookie))
    }
    pub fn abandon_issued(&mut self)->Result<()> {
        if self.retained.is_some() || self.poisoned {return Err(bad("cannot abandon admitted work"));}
        self.issued=None;Ok(())
    }
    pub fn submission_started(&self)->bool {self.started}
    pub fn execute(&mut self,e:wire::Expectation)->Result<(Option<u32>,&[u8])> {
        if self.poisoned || self.retained.is_some() {return Err(bad("session busy or poisoned"));}
        let i=self.identity;
        if e.rows.len()!=1 || e.max_active_rows!=8 || e.owner_generation!=i.generation
            || e.last_accepted_replay!=i.last_accepted_replay || e.catalog_digest!=i.catalog_digest
            || e.physical_block_count!=i.physical_block_count || e.rows[0].progress.context_tokens!=i.context_tokens
            || self.issued!=Some(e.rows[0].cookie) {return Err(bad("submission differs from issued owner"));}
        wire::encode_into(&mut self.input,&e)?;
        self.retained=Some(e);self.issued=None;self.started=true;self.completed=false;
        if self.graph.replay_transfer(&self.input).and_then(|()|self.graph.read_transfer(&mut self.output)).is_err() {
            self.poisoned=true;return Err(bad("native execution failed; close required before KV release"));
        }
        let e=self.retained.as_ref().unwrap();
        match wire::validate_result(&self.output,e) {
            Ok(result)=>{self.completed=true;Ok(result)},
            Err(_)=>{self.poisoned=true;Err(bad("invalid GPU completion; close required"))}
        }
    }
    /// Only after successful scheduler settlement of this exact iteration.
    pub fn confirm_scheduler_commit(&mut self,iteration:u64)->Result<()> {
        if self.poisoned || !self.completed || self.retained.as_ref().map(|e|e.iteration_id)!=Some(iteration) {return Err(bad("commit differs from completed iteration"));}
        self.identity.last_accepted_replay=self.retained.take().unwrap().replay_id;self.completed=false;Ok(())
    }
    /// Native destruction remains the authority for GPU parent release.
    pub fn close(self)->riley_cuda::CudaResult<()> {self.graph.close()}
}
