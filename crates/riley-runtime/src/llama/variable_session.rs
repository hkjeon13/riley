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
pub struct VariableSession<G: VariableGraph> {
    graph:G, identity:VariableSessionIdentity,
    issued:Option<u64>, next_cookie:u64, retained:Option<wire::Expectation>,
    started:bool, poisoned:bool, completed:bool, input:Vec<u8>, output:Vec<u8>,
}
fn bad(reason:&'static str)->Error {Error{field:"V3 retained session",reason}}
impl<G: VariableGraph> VariableSession<G> {
    /// The caller must have recorded the V3 model and bound the digest to its
    /// actual prepared model/kernel catalog. This constructor does not create it.
    pub fn new(graph:G,catalog_digest:[u8;32],physical_block_count:u32,context_tokens:u32)->Result<Self> {
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
    pub(crate) fn into_recorded_parts(self)->(G,VariableSessionIdentity) {(self.graph,self.identity)}
}

/// Cold scratch for the fixed SmolLM2 V3 implementation. KV and weights remain
/// in the loaded model owner and are exclusively borrowed during the session.
pub struct VariableGraphBuffers {
    pub(crate) devices:Vec<riley_cuda::CudaDeviceBuffer>,
    pub(crate) staging:riley_cuda::CudaPinnedHostBuffer,
    pub(crate) head:riley_cuda::CudaPreparedGemm,
    pub(crate) capacity:u32,
}
impl VariableGraphBuffers {
    pub fn prepare(context:&riley_cuda::CudaContext,capacity:u32)->riley_cuda::CudaResult<Self> {
        // Invalid capacity is rejected again by the native recorder; allocating
        // zero/oversized geometry is prevented with the GEMM config validator.
        let checked=if (1..=1024).contains(&capacity) {capacity}else{0};
        let config=riley_cuda::CudaGemmConfig::new(checked as u64,49152,576,0)?;
        let _=config;
        let sizes=[1152,1152,1152,1152,1152,384,384,384,3072,3072,2304,3072];
        let mut devices=Vec::with_capacity(18);
        for (i,bytes) in sizes.into_iter().enumerate() {let bytes=bytes*capacity as u64;devices.push(context.allocate_device_buffer(if i==7 {bytes.max(9*4096*4)}else{bytes})?);}
        for bytes in [17536,1152,128,4,98304,8] {devices.push(context.allocate_device_buffer(bytes)?);}
        Ok(Self{devices,staging:context.allocate_pinned_host_buffer(196864)?,
            head:context.prepare_gemm(riley_cuda::CudaGemmConfig::new(1,49152,576,0)?)?,capacity})
    }
    pub fn close(self)->riley_cuda::CudaResult<()> {self.head.close()}
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for riley_cuda::BorrowedGraphResourceReservation<'_> {}
    impl Sealed for riley_cuda::OwnedGraphResourceReservation<super::VariableModelParents> {}
}
/// Native reservation operations; sealed so a caller cannot fabricate completion.
pub trait VariableGraph: sealed::Sealed {
    fn replay_transfer(&mut self,input:&[u8])->riley_cuda::CudaResult<()>;
    fn read_transfer(&mut self,output:&mut[u8])->riley_cuda::CudaResult<()>;
}
impl VariableGraph for BorrowedGraphResourceReservation<'_> {
    fn replay_transfer(&mut self,input:&[u8])->riley_cuda::CudaResult<()> {BorrowedGraphResourceReservation::replay_transfer(self,input)}
    fn read_transfer(&mut self,output:&mut[u8])->riley_cuda::CudaResult<()> {BorrowedGraphResourceReservation::read_transfer(self,output)}
}
impl VariableGraph for riley_cuda::OwnedGraphResourceReservation<VariableModelParents> {
    fn replay_transfer(&mut self,input:&[u8])->riley_cuda::CudaResult<()> {riley_cuda::OwnedGraphResourceReservation::replay_transfer(self,input)}
    fn read_transfer(&mut self,output:&mut[u8])->riley_cuda::CudaResult<()> {riley_cuda::OwnedGraphResourceReservation::read_transfer(self,output)}
}
pub type BorrowedVariableSession<'a> = VariableSession<BorrowedGraphResourceReservation<'a>>;
pub type OwnedVariableSession = VariableSession<riley_cuda::OwnedGraphResourceReservation<VariableModelParents>>;
/// No fields are exposed while the native graph owns parent leases.
pub struct VariableModelParents {
    executor:super::PreparedLlamaBatchExecutor,
    stream:riley_cuda::CudaStream,
    scratch:VariableGraphBuffers,
}
impl VariableSession<BorrowedGraphResourceReservation<'_>> {
    pub fn close(self)->riley_cuda::CudaResult<()> {self.graph.close()}
}
impl OwnedVariableSession {
    pub fn close(self)->super::LlamaBatchExecutorResult<()> {
        let cuda=|e|super::executor::error::cuda_error(super::ExecutionSite::global(super::LlamaOp::IterationCompletion),e);
        let parents=self.graph.close().map_err(cuda)?;
        let scratch=parents.scratch.close().map_err(cuda);
        let executor=parents.executor.close();
        let stream=parents.stream.close().map_err(cuda);
        scratch.and(executor).and(stream)
    }
}
impl super::PreparedLlamaBatchExecutor {
    /// Moves the loaded model and every graph parent into an owned session.
    /// The native reservation is destroyed before model/stream/scratch release.
    pub fn into_owned_variable_session(self,context:&riley_cuda::CudaContext,capacity:u32)->super::LlamaBatchExecutorResult<OwnedVariableSession> {
        let cuda=|e|super::executor::error::cuda_error(super::ExecutionSite::global(super::LlamaOp::IterationCompletion),e);
        let parents=VariableModelParents{executor:self,stream:context.create_stream().map_err(cuda)?,scratch:VariableGraphBuffers::prepare(context,capacity).map_err(cuda)?};
        let mut identity=None;
        let graph=riley_cuda::OwnedGraphResourceReservation::prepare(parents,|p| {
            let session=p.executor.prepare_variable_session(&mut p.stream,&mut p.scratch)?;
            let (graph,i)=session.into_recorded_parts();identity=Some(i);Ok::<_,super::LlamaBatchExecutorError>(graph)
        })?;
        let i=identity.expect("successful recording provides identity");
        OwnedVariableSession::new(graph,i.catalog_digest,i.physical_block_count,i.context_tokens)
            .map_err(|_|super::LlamaBatchExecutorError::InvalidConfiguration{field:"V3 owned session",reason:"identity creation failed"})
    }
}
