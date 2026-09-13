//! Retained V3 transactions with single-row or shared native reservations.
use super::multi_descriptor::{variable_wire as wire, Error, Result};
use riley_cuda::BorrowedGraphResourceReservation;
use std::sync::atomic::{AtomicU64, Ordering};
static GENERATION: AtomicU64 = AtomicU64::new(1);
#[derive(Clone, Copy)]
pub struct VariableSessionIdentity {
    pub generation:u64, pub last_accepted_replay:u64, pub catalog_digest:[u8;32],
    pub physical_block_count:u32, pub context_tokens:u32, pub packed_prefill:bool, pub mixed_execution:bool,
}
/// Owns the prepared reservation; no mutable native handle escapes while active.
/// A result remains outstanding until scheduler commit is explicitly confirmed.
pub struct VariableSession<G: VariableGraph,const ROWS:usize=8> {
    graph:G, identity:VariableSessionIdentity,
    issued:Option<Vec<u64>>, next_cookie:u64, retained:Option<wire::Expectation<ROWS>>,
    buffered:bool, buffered_ticket:Option<u64>, async_completion:bool,compact:bool,shared:bool,started:bool, poisoned:bool, completed:bool, input:Vec<u8>, output:Vec<u8>,
}
fn bad(reason:&'static str)->Error {Error{field:"V3 retained session",reason}}
impl<G: VariableGraph,const ROWS:usize> VariableSession<G,ROWS> {
    /// The caller must have recorded the V3 model and bound the digest to its
    /// actual prepared model/kernel catalog. This constructor does not create it.
    pub fn new(graph:G,catalog_digest:[u8;32],physical_block_count:u32,context_tokens:u32)->Result<Self> {
        if !matches!(ROWS,8|16|32) || catalog_digest==[0;32] || !(1..=4096).contains(&physical_block_count) || !(1..=4096).contains(&context_tokens) {return Err(bad("invalid prepared geometry"));}
        let generation=GENERATION.fetch_update(Ordering::Relaxed,Ordering::Relaxed,|n|n.checked_add(1)).map_err(|_|bad("generation exhausted"))?;
        Ok(Self{graph,identity:VariableSessionIdentity{generation,last_accepted_replay:0,catalog_digest,physical_block_count,context_tokens,packed_prefill:false,mixed_execution:false},
            buffered:false,buffered_ticket:None,async_completion:false,compact:false,shared:false,issued:None,next_cookie:1,retained:None,started:false,poisoned:false,completed:false,input:vec![0;wire::Layout::<ROWS>::REQUEST_BYTES],output:vec![0;wire::RESULT_BYTES]})
    }
    pub fn new_shared(graph:G,digest:[u8;32],physical:u32,context:u32)->Result<Self>{let mut s=Self::new(graph,digest,physical,context)?;s.shared=true;s.output=vec![0;wire::Layout::<ROWS>::BATCH_RESULT_BYTES];Ok(s)}
    pub(crate) fn new_shared_compact(graph:G,digest:[u8;32],physical:u32,context:u32)->Result<Self>{if !matches!(ROWS,16|32){return Err(bad("compact requires sixteen or thirty-two rows"));}let mut s=Self::new_shared(graph,digest,physical,context)?;s.compact=true;Ok(s)}
    pub(crate) fn new_shared_packed(graph:G,digest:[u8;32],physical:u32,context:u32,compact:bool)->Result<Self>{
        if ROWS!=32{return Err(bad("packed requires32 rows"));}
        let mut s=if compact{Self::new_shared_compact(graph,digest,physical,context)?}else{Self::new_shared(graph,digest,physical,context)?};s.identity.packed_prefill=true;Ok(s)
    }
    pub(crate) fn new_shared_mixed(graph:G,digest:[u8;32],physical:u32,context:u32,compact:bool)->Result<Self>{let mut s=Self::new_shared_packed(graph,digest,physical,context,compact)?;s.identity.mixed_execution=true;s.input.resize(wire::MIXED_REQUEST_BYTES,0);Ok(s)}
    pub(crate) fn set_buffered_completion(&mut self,enabled:bool) {self.buffered=enabled;}
    pub fn supports_mixed_execution(&self)->bool{self.identity.mixed_execution}
    pub fn supports_packed_prefill(&self)->bool{self.identity.packed_prefill}
    pub fn supports_compact_greedy(&self)->bool{self.compact}
    pub fn issue(&mut self)->Result<(VariableSessionIdentity,u64,u64)>{let(i,r,c)=self.issue_rows(1)?;Ok((i,r,c[0]))}
    pub fn issue_rows(&mut self,rows:usize)->Result<(VariableSessionIdentity,u64,Vec<u64>)> {
        if rows==0||rows>if self.shared{ROWS}else{1}{return Err(bad("unsupported row count"));}
        if self.poisoned || self.retained.is_some() || self.issued.is_some() {return Err(bad("session busy or poisoned"));}
        let replay=self.identity.last_accepted_replay.checked_add(1).ok_or_else(||bad("replay exhausted"))?;
        let cookie=self.next_cookie;self.next_cookie=cookie.checked_add(rows as u64).ok_or_else(||bad("cookie exhausted"))?;
        let cookies:Vec<_>=(cookie..self.next_cookie).collect();self.issued=Some(cookies.clone());self.started=false;Ok((self.identity,replay,cookies))
    }
    pub fn abandon_issued(&mut self)->Result<()> {
        if self.retained.is_some() || self.poisoned {return Err(bad("cannot abandon admitted work"));}
        self.issued=None;Ok(())
    }
    pub fn submission_started(&self)->bool {self.started}
    pub fn execute(&mut self,e:wire::Expectation<ROWS>)->Result<(Option<u32>,&[u8])> {
        if e.rows.len()!=1{return Err(bad("single result requires one row"));}
        let rows=self.execute_rows(e)?;Ok((rows[0].token,rows[0].logits))
    }
    /// Select event completion for the compatibility execute_rows path.
    /// This does not enable scheduling ahead; it may only change while idle.
    pub fn set_async_completion(&mut self, enabled: bool)->Result<()> {
        if self.poisoned || self.retained.is_some() || self.issued.is_some() { return Err(bad("session busy or poisoned")); }
        self.async_completion=enabled; Ok(())
    }
    fn retain_submission(&mut self,e:wire::Expectation<ROWS>)->Result<()> {
        if self.poisoned || self.retained.is_some() {return Err(bad("session busy or poisoned"));}
        let i=self.identity;
        if e.mixed_execution!=i.mixed_execution || e.packed_prefill!=i.packed_prefill || e.rows.is_empty() || e.rows.len()>if self.shared{ROWS}else{1} || e.max_active_rows!=ROWS as u32 || e.owner_generation!=i.generation
            || e.last_accepted_replay!=i.last_accepted_replay || e.catalog_digest!=i.catalog_digest
            || e.physical_block_count!=i.physical_block_count || e.rows.iter().any(|r|r.progress.context_tokens!=i.context_tokens)
            || self.issued.as_deref()!=Some(e.rows.iter().map(|r|r.cookie).collect::<Vec<_>>().as_slice()) {return Err(bad("submission differs from issued owner"));}
        wire::encode_into(&mut self.input,&e)?;
        self.retained=Some(e);self.issued=None;self.started=true;self.completed=false;
        Ok(())
    }
    pub fn execute_rows(&mut self,e:wire::Expectation<ROWS>)->Result<Vec<wire::RowResult<'_>>> {
        if self.async_completion || self.buffered { self.submit_rows(e)?; return self.wait_rows(); }
        self.retain_submission(e)?;
        if self.graph.replay_transfer(&self.input).is_err() {
            self.poisoned=true;return Err(bad("native execution failed; close required before KV release"));
        }
        self.collect_rows()
    }
    /// Stage and submit one iteration. Its buffers remain owned until commit.
    pub fn submit_rows(&mut self,e:wire::Expectation<ROWS>)->Result<()> {
        self.retain_submission(e)?;
        let result=if self.buffered {
            self.graph.submit_buffered_transfer(&self.input).map(|ticket|self.buffered_ticket=Some(ticket))
        } else {self.graph.submit_transfer(&self.input)};
        if result.is_err() {
            self.poisoned=true;return Err(bad("native submission failed; close required before KV release"));
        }
        Ok(())
    }
    pub fn query_completion(&mut self)->Result<bool> {
        if self.poisoned || self.retained.is_none() || self.completed {return Err(bad("no pending iteration"));}
        let result=if let Some(ticket)=self.buffered_ticket {self.graph.query_buffered_transfer(ticket,false)} else {self.graph.query_transfer()};
        result.map_err(|_|{self.poisoned=true;bad("native completion failed; close required")})
    }
    pub fn wait_rows(&mut self)->Result<Vec<wire::RowResult<'_>>> {
        if self.poisoned || self.retained.is_none() || self.completed {return Err(bad("no pending iteration"));}
        let result=if let Some(ticket)=self.buffered_ticket {self.graph.query_buffered_transfer(ticket,true).map(|_|())} else {self.graph.wait_transfer()};
        if result.is_err() {self.poisoned=true;return Err(bad("native completion failed; close required"));}
        self.collect_rows()
    }
    fn collect_rows(&mut self)->Result<Vec<wire::RowResult<'_>>> {
        if self.poisoned || self.retained.is_none() || self.completed {return Err(bad("no pending iteration"));}
        let compact=self.compact && self.retained.as_ref().unwrap().mode==super::multi_descriptor::ResultMode::Greedy;
        let output_bytes=if compact{wire::Layout::<ROWS>::COMPACT_RESULT_BYTES}else{self.output.len()};
        let result=if let Some(ticket)=self.buffered_ticket {self.graph.read_buffered_transfer(ticket,&mut self.output[..output_bytes])} else {self.graph.read_transfer(&mut self.output[..output_bytes])};
        if result.is_err() {self.poisoned=true;return Err(bad("native read failed; close required"));}
        self.buffered_ticket=None;
        let e=self.retained.as_ref().unwrap();
        match if compact {wire::validate_compact_result(&self.output[..output_bytes],e)}else if self.shared {wire::validate_batch_result(&self.output,e)}else{wire::validate_result(&self.output,e).map(|(token,logits)|vec![wire::RowResult{output_slot:0,token,logits}])} {
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
    pub(crate) flashinfer_workspace:Option<riley_cuda::CudaDeviceBuffer>,
    pub(crate) devices:Vec<riley_cuda::CudaDeviceBuffer>,
    pub(crate) tiled:Vec<riley_cuda::CudaDeviceBuffer>,
    pub(crate) shared_devices:Vec<riley_cuda::CudaDeviceBuffer>,
    pub(crate) shared_head:Option<riley_cuda::CudaPreparedGemm>,
    pub(crate) staging:riley_cuda::CudaPinnedHostBuffer,
    pub(crate) buffered_staging:Vec<riley_cuda::CudaPinnedHostBuffer>,
    pub(crate) head:riley_cuda::CudaPreparedGemm,
    pub(crate) capacity:u32,
    pub(crate) wire_rows:usize,
    pub(crate) compact:bool,
    pub(crate) packed_prefill:bool,
    pub(crate) mixed_execution:bool,
}
impl VariableGraphBuffers {
    pub fn prepare(context:&riley_cuda::CudaContext,capacity:u32)->riley_cuda::CudaResult<Self> {Self::prepare_base(context,capacity,true)}
    fn prepare_base(context:&riley_cuda::CudaContext,capacity:u32,packed:bool)->riley_cuda::CudaResult<Self>{
        // Invalid capacity is rejected again by the native recorder; allocating
        // zero/oversized geometry is prevented with the GEMM config validator.
        let checked=if (1..=1024).contains(&capacity) {capacity}else{0};
        let config=riley_cuda::CudaGemmConfig::new(checked as u64,49152,576,0)?;
        let _=config;
        let sizes=[1152,1152,1152,1152,1152,384,384,384,3072,3072,2304,3072];
        let mut devices=Vec::with_capacity(18);
        for (i,bytes) in sizes.into_iter().enumerate() {let bytes=bytes*capacity as u64;devices.push(context.allocate_device_buffer(if i==7 {bytes.max(9*4096*4)}else{bytes})?);}
        for bytes in [17536,1152,128,4,98304,8] {devices.push(context.allocate_device_buffer(bytes)?);}
        let tiled=(0..if packed{90}else{0}).map(|_|context.allocate_device_buffer(1769472)).collect::<riley_cuda::CudaResult<Vec<_>>>()?;
        Ok(Self{flashinfer_workspace:None,devices,tiled,shared_devices:vec![],shared_head:None,staging:context.allocate_pinned_host_buffer(196864)?,
            buffered_staging:Vec::new(),head:context.prepare_gemm(riley_cuda::CudaGemmConfig::new(1,49152,576,0)?)?,capacity,wire_rows:8,compact:false,packed_prefill:false,mixed_execution:false})
    }
    pub fn prepare_shared(context:&riley_cuda::CudaContext,capacity:u32)->riley_cuda::CudaResult<Self>{Self::prepare_shared_rows::<8>(context,capacity)}
    pub fn prepare_shared16(context:&riley_cuda::CudaContext,capacity:u32)->riley_cuda::CudaResult<Self>{Self::prepare_shared_rows::<16>(context,capacity)}
    fn prepare_shared_rows<const ROWS:usize>(context:&riley_cuda::CudaContext,capacity:u32)->riley_cuda::CudaResult<Self>{
        let mut s=Self::prepare_base(context,if capacity==0{0}else{capacity.max(ROWS as u32)},true)?;
        s.wire_rows=ROWS;
        s.devices[7]=context.allocate_device_buffer((s.capacity as u64*384).max(ROWS as u64*9*4096*4))?;
        s.devices[12]=context.allocate_device_buffer(wire::Layout::<ROWS>::REQUEST_BYTES as u64)?;
        s.staging=context.allocate_pinned_host_buffer(wire::Layout::<ROWS>::STAGING_BYTES as u64)?;
        for bytes in [ROWS as u64*1152,ROWS as u64*98304,wire::Layout::<ROWS>::BATCH_RESULT_BYTES as u64]{s.shared_devices.push(context.allocate_device_buffer(bytes)?);}
        s.shared_head=Some(context.prepare_gemm(riley_cuda::CudaGemmConfig::new(ROWS as u64,49152,576,0)?)?);Ok(s)
    }
    pub fn close(self)->riley_cuda::CudaResult<()> {let a=self.head.close();let b=self.shared_head.map_or(Ok(()),|h|h.close());a.and(b)}
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for riley_cuda::BorrowedGraphResourceReservation<'_> {}
    impl Sealed for riley_cuda::OwnedGraphResourceReservation<super::VariableModelParents> {}
}
/// Native reservation operations; sealed so a caller cannot fabricate completion.
pub trait VariableGraph: sealed::Sealed {
    fn submit_buffered_transfer(&mut self,input:&[u8])->riley_cuda::CudaResult<u64>;
    fn query_buffered_transfer(&mut self,ticket:u64,wait:bool)->riley_cuda::CudaResult<bool>;
    fn read_buffered_transfer(&mut self,ticket:u64,output:&mut[u8])->riley_cuda::CudaResult<()>;
    fn replay_transfer(&mut self,input:&[u8])->riley_cuda::CudaResult<()>;
    fn submit_transfer(&mut self,input:&[u8])->riley_cuda::CudaResult<()>;
    fn query_transfer(&mut self)->riley_cuda::CudaResult<bool>;
    fn wait_transfer(&mut self)->riley_cuda::CudaResult<()>;
    fn read_transfer(&mut self,output:&mut[u8])->riley_cuda::CudaResult<()>;
}
impl VariableGraph for BorrowedGraphResourceReservation<'_> {
    fn submit_buffered_transfer(&mut self,input:&[u8])->riley_cuda::CudaResult<u64> {BorrowedGraphResourceReservation::submit_buffered_transfer(self,input)}
    fn query_buffered_transfer(&mut self,ticket:u64,wait:bool)->riley_cuda::CudaResult<bool> {BorrowedGraphResourceReservation::query_buffered_transfer(self,ticket,wait)}
    fn read_buffered_transfer(&mut self,ticket:u64,output:&mut[u8])->riley_cuda::CudaResult<()> {BorrowedGraphResourceReservation::read_buffered_transfer(self,ticket,output)}
    fn submit_transfer(&mut self,input:&[u8])->riley_cuda::CudaResult<()> {BorrowedGraphResourceReservation::submit_transfer(self,input)}
    fn query_transfer(&mut self)->riley_cuda::CudaResult<bool> {BorrowedGraphResourceReservation::query_transfer(self)}
    fn wait_transfer(&mut self)->riley_cuda::CudaResult<()> {BorrowedGraphResourceReservation::wait_transfer(self)}
    fn replay_transfer(&mut self,input:&[u8])->riley_cuda::CudaResult<()> {BorrowedGraphResourceReservation::replay_transfer(self,input)}
    fn read_transfer(&mut self,output:&mut[u8])->riley_cuda::CudaResult<()> {BorrowedGraphResourceReservation::read_transfer(self,output)}
}
impl VariableGraph for riley_cuda::OwnedGraphResourceReservation<VariableModelParents> {
    fn submit_buffered_transfer(&mut self,input:&[u8])->riley_cuda::CudaResult<u64> {riley_cuda::OwnedGraphResourceReservation::submit_buffered_transfer(self,input)}
    fn query_buffered_transfer(&mut self,ticket:u64,wait:bool)->riley_cuda::CudaResult<bool> {riley_cuda::OwnedGraphResourceReservation::query_buffered_transfer(self,ticket,wait)}
    fn read_buffered_transfer(&mut self,ticket:u64,output:&mut[u8])->riley_cuda::CudaResult<()> {riley_cuda::OwnedGraphResourceReservation::read_buffered_transfer(self,ticket,output)}
    fn submit_transfer(&mut self,input:&[u8])->riley_cuda::CudaResult<()> {riley_cuda::OwnedGraphResourceReservation::submit_transfer(self,input)}
    fn query_transfer(&mut self)->riley_cuda::CudaResult<bool> {riley_cuda::OwnedGraphResourceReservation::query_transfer(self)}
    fn wait_transfer(&mut self)->riley_cuda::CudaResult<()> {riley_cuda::OwnedGraphResourceReservation::wait_transfer(self)}
    fn replay_transfer(&mut self,input:&[u8])->riley_cuda::CudaResult<()> {riley_cuda::OwnedGraphResourceReservation::replay_transfer(self,input)}
    fn read_transfer(&mut self,output:&mut[u8])->riley_cuda::CudaResult<()> {riley_cuda::OwnedGraphResourceReservation::read_transfer(self,output)}
}
pub type BorrowedVariableSession<'a,const ROWS:usize=8> = VariableSession<BorrowedGraphResourceReservation<'a>,ROWS>;
pub type OwnedVariableSession<const ROWS:usize=8> = VariableSession<riley_cuda::OwnedGraphResourceReservation<VariableModelParents>,ROWS>;
/// No fields are exposed while the native graph owns parent leases.
pub struct VariableModelParents {
    executor:super::PreparedLlamaBatchExecutor,
    stream:riley_cuda::CudaStream,
    scratch:VariableGraphBuffers,
}
impl<const ROWS:usize> VariableSession<BorrowedGraphResourceReservation<'_>,ROWS> {
    pub fn close(self)->riley_cuda::CudaResult<()> {self.graph.close()}
}
impl<const ROWS:usize> OwnedVariableSession<ROWS> {
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
    pub fn into_owned_variable_session(self,context:&riley_cuda::CudaContext,capacity:u32)->super::LlamaBatchExecutorResult<OwnedVariableSession> {self.into_variable_session::<8>(context,capacity,false,false,false,false)}
    pub fn into_owned_variable_shared_session(self,context:&riley_cuda::CudaContext,capacity:u32)->super::LlamaBatchExecutorResult<OwnedVariableSession>{self.into_variable_session::<8>(context,capacity,true,false,false,false)}
    pub fn into_owned_variable_shared_session_rows<const ROWS:usize>(self,context:&riley_cuda::CudaContext,capacity:u32)->super::LlamaBatchExecutorResult<OwnedVariableSession<ROWS>>{self.into_variable_session::<ROWS>(context,capacity,true,false,false,false)}
    pub fn into_owned_variable_shared16_session(self,context:&riley_cuda::CudaContext,capacity:u32)->super::LlamaBatchExecutorResult<OwnedVariableSession<16>>{self.into_variable_session::<16>(context,capacity,true,false,false,false)}
    pub fn into_owned_variable_shared_greedy_session_rows<const ROWS:usize>(self,context:&riley_cuda::CudaContext,capacity:u32)->super::LlamaBatchExecutorResult<OwnedVariableSession<ROWS>>{self.into_variable_session::<ROWS>(context,capacity,true,true,false,false)}
    pub fn into_owned_variable_shared16_greedy_session(self,context:&riley_cuda::CudaContext,capacity:u32)->super::LlamaBatchExecutorResult<OwnedVariableSession<16>>{self.into_variable_session::<16>(context,capacity,true,true,false,false)}
    pub fn into_owned_variable_packed_session(self,context:&riley_cuda::CudaContext,capacity:u32,compact:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<32>>{self.into_variable_session::<32>(context,capacity,true,compact,true,false)}
    pub fn into_owned_variable_mixed_session(self,context:&riley_cuda::CudaContext,capacity:u32,compact:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<32>>{self.into_variable_session::<32>(context,capacity,true,compact,true,true)}
    /// Opt-in staging double buffering. Scheduler admission still permits only
    /// one retained expectation; this does not enable GPU future-token decoding.
    pub fn into_owned_buffered_variable_mixed_session(self,context:&riley_cuda::CudaContext,capacity:u32,compact:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<32>>{self.into_variable_session_mode::<32>(context,capacity,true,compact,true,true,true,false)}
    fn into_variable_session<const ROWS:usize>(self,context:&riley_cuda::CudaContext,capacity:u32,shared:bool,compact:bool,packed:bool,mixed:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<ROWS>>{
        self.into_variable_session_mode::<ROWS>(context,capacity,shared,compact,packed,mixed,false,false)
    }
    /// Explicit experimental numerical profile: FlashInfer 0.6.16.post3 unsplit
    /// pure decode; existing packed/mixed prefill arithmetic. Quality is unaccepted.
    /// Fails if optional native build support is absent; never falls back silently.
    pub fn into_owned_variable_flashinfer_experimental_session(self,context:&riley_cuda::CudaContext,capacity:u32,compact:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<32>> {
        self.into_variable_session_mode::<32>(context,capacity,true,compact,true,true,false,true)
    }
    fn into_variable_session_mode<const ROWS:usize>(self,context:&riley_cuda::CudaContext,capacity:u32,shared:bool,compact:bool,packed:bool,mixed:bool,buffered:bool,flashinfer:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<ROWS>>{
        let cuda=|e|super::executor::error::cuda_error(super::ExecutionSite::global(super::LlamaOp::IterationCompletion),e);
        if (mixed&&!packed) || (packed && (ROWS!=32||!shared)) || !matches!(ROWS,8|16|32) || (compact && (!shared || !matches!(ROWS,16|32))) {return Err(super::LlamaBatchExecutorError::InvalidConfiguration{field:"wire rows",reason:"unsupported execution width"});}
        let mut parents=VariableModelParents{executor:self,stream:context.create_stream().map_err(cuda)?,scratch:if shared {VariableGraphBuffers::prepare_shared_rows::<ROWS>(context,capacity)}else{VariableGraphBuffers::prepare(context,capacity)}.map_err(cuda)?};
        parents.scratch.compact=compact;parents.scratch.packed_prefill=packed;parents.scratch.mixed_execution=mixed;
        if flashinfer {parents.scratch.flashinfer_workspace=Some(context.allocate_device_buffer(33456).map_err(cuda)?);}
        if mixed {parents.scratch.devices[12]=context.allocate_device_buffer(wire::MIXED_REQUEST_BYTES as u64).map_err(cuda)?;}
        if buffered {for _ in 0..1 {parents.scratch.buffered_staging.push(context.allocate_pinned_host_buffer(wire::Layout::<ROWS>::STAGING_BYTES as u64).map_err(cuda)?);}}
        let mut identity=None;
        let graph=riley_cuda::OwnedGraphResourceReservation::prepare(parents,|p| {
            let session=p.executor.prepare_variable_session_rows::<ROWS>(&mut p.stream,&mut p.scratch)?;
            let (graph,i)=session.into_recorded_parts();identity=Some(i);Ok::<_,super::LlamaBatchExecutorError>(graph)
        })?;
        let i=identity.expect("successful recording provides identity");
        let mut session=(if mixed {OwnedVariableSession::new_shared_mixed(graph,i.catalog_digest,i.physical_block_count,i.context_tokens,compact)}else if packed {OwnedVariableSession::new_shared_packed(graph,i.catalog_digest,i.physical_block_count,i.context_tokens,compact)}else if compact {OwnedVariableSession::new_shared_compact(graph,i.catalog_digest,i.physical_block_count,i.context_tokens)}else if shared {OwnedVariableSession::new_shared(graph,i.catalog_digest,i.physical_block_count,i.context_tokens)}else{OwnedVariableSession::new(graph,i.catalog_digest,i.physical_block_count,i.context_tokens)})
            .map_err(|_|super::LlamaBatchExecutorError::InvalidConfiguration{field:"V3 owned session",reason:"identity creation failed"})?;
        session.set_buffered_completion(buffered);Ok(session)
    }
}
