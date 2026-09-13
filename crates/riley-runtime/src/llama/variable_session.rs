//! Retained V3 transactions with single-row or shared native reservations.
use super::multi_descriptor::{variable_wire as wire, Error, Result};
use riley_cuda::BorrowedGraphResourceReservation;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
#[derive(Default)]
struct HostRuntimeTiming { counts: [u64; 6], nanos: [u128; 6] }
fn runtime_phase_record(timing: &mut Option<HostRuntimeTiming>, stage: usize, start: Option<Instant>) {
    if let (Some(timing), Some(start)) = (timing.as_mut(), start) {
        timing.counts[stage] = timing.counts[stage].saturating_add(1);
        timing.nanos[stage] = timing.nanos[stage].saturating_add(start.elapsed().as_nanos());
    }
}
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
    host_phase_timing: Option<HostRuntimeTiming>,
    window_predecessor_ticket:Option<u64>, issued_successor:Option<Vec<u64>>, window:Option<DecodeWindowState<ROWS>>,
    issued:Option<Vec<u64>>, next_cookie:u64, retained:Option<wire::Expectation<ROWS>>,
    buffered:bool, buffered_ticket:Option<u64>, async_completion:bool,compact:bool,shared:bool,started:bool, poisoned:bool, completed:bool, input:Vec<u8>, output:Vec<u8>,
}
struct DecodeWindowState<const ROWS:usize> {
    successor:wire::Expectation<ROWS>, tickets:[u64;2], output:Vec<u8>, predecessor_validated:bool, complete:bool,
}

fn bad(reason:&'static str)->Error {Error{field:"V3 retained session",reason}}
impl<G: VariableGraph,const ROWS:usize> VariableSession<G,ROWS> {
    /// The caller must have recorded the V3 model and bound the digest to its
    /// actual prepared model/kernel catalog. This constructor does not create it.
    pub fn new(graph:G,catalog_digest:[u8;32],physical_block_count:u32,context_tokens:u32)->Result<Self> {
        if !matches!(ROWS,8|16|32) || catalog_digest==[0;32] || !(1..=4096).contains(&physical_block_count) || !(1..=4096).contains(&context_tokens) {return Err(bad("invalid prepared geometry"));}
        let generation=GENERATION.fetch_update(Ordering::Relaxed,Ordering::Relaxed,|n|n.checked_add(1)).map_err(|_|bad("generation exhausted"))?;
        Ok(Self{graph,host_phase_timing:(std::env::var("RILEY_SERVING_PHASE_TIMING").ok().as_deref()==Some("1")).then(HostRuntimeTiming::default),window_predecessor_ticket:None,issued_successor:None,window:None,identity:VariableSessionIdentity{generation,last_accepted_replay:0,catalog_digest,physical_block_count,context_tokens,packed_prefill:false,mixed_execution:false},
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
    /// Opt-in diagnostic wall time; native execution includes GPU waits.
    pub fn report_host_phase_timing(&self) {
        if let Some(timing) = self.host_phase_timing.as_ref() {
            for (stage, name) in ["retain_encode", "sync_transfer", "buffered_submit", "buffered_wait", "read_validate", "future_prepare"].iter().enumerate() {
                eprintln!("RILEY_RUNTIME_PHASE kind={} calls={} wall_ns={}", name, timing.counts[stage], timing.nanos[stage]);
            }
        }
    }
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
        self.issued=None;self.issued_successor=None;Ok(())
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
        let phase = self.host_phase_timing.as_ref().map(|_| Instant::now());
        if self.issued_successor.is_some() || (self.window.is_some() || self.window_predecessor_ticket.is_some()) {return Err(bad("window requires paired submission"));}
        if self.poisoned || self.retained.is_some() {return Err(bad("session busy or poisoned"));}
        let i=self.identity;
        if e.mixed_execution!=i.mixed_execution || e.packed_prefill!=i.packed_prefill || e.rows.is_empty() || e.rows.len()>if self.shared{ROWS}else{1} || e.max_active_rows!=ROWS as u32 || e.owner_generation!=i.generation
            || e.last_accepted_replay!=i.last_accepted_replay || e.catalog_digest!=i.catalog_digest
            || e.physical_block_count!=i.physical_block_count || e.rows.iter().any(|r|r.progress.context_tokens!=i.context_tokens)
            || self.issued.as_deref()!=Some(e.rows.iter().map(|r|r.cookie).collect::<Vec<_>>().as_slice()) {return Err(bad("submission differs from issued owner"));}
        wire::encode_into(&mut self.input,&e)?;
        self.retained=Some(e);self.issued=None;self.started=true;self.completed=false;
        runtime_phase_record(&mut self.host_phase_timing, 0, phase);
        Ok(())
    }
    pub fn execute_rows(&mut self,e:wire::Expectation<ROWS>)->Result<Vec<wire::RowResult<'_>>> {
        if self.async_completion || self.buffered { self.submit_rows(e)?; return self.wait_rows(); }
        self.retain_submission(e)?;
        let phase = self.host_phase_timing.as_ref().map(|_| Instant::now());
        if self.graph.replay_transfer(&self.input).is_err() {
            self.poisoned=true;return Err(bad("native execution failed; close required before KV release"));
        }
        runtime_phase_record(&mut self.host_phase_timing, 1, phase);
        self.collect_rows()
    }
    /// Stage and submit one iteration. Its buffers remain owned until commit.
    pub fn submit_rows(&mut self,e:wire::Expectation<ROWS>)->Result<()> {
        self.retain_submission(e)?;
        let phase = self.host_phase_timing.as_ref().map(|_| Instant::now());
        let result=if self.buffered {
            self.graph.submit_buffered_transfer(&self.input).map(|ticket|self.buffered_ticket=Some(ticket))
        } else {self.graph.submit_transfer(&self.input)};
        if result.is_err() {
            self.poisoned=true;return Err(bad("native submission failed; close required before KV release"));
        }
        runtime_phase_record(&mut self.host_phase_timing, 2, phase);
        Ok(())
    }
    pub fn query_completion(&mut self)->Result<bool> {
        if self.poisoned || (self.window.is_some() || self.window_predecessor_ticket.is_some()) || self.retained.is_none() || self.completed {return Err(bad("no pending iteration"));}
        let result=if let Some(ticket)=self.buffered_ticket {self.graph.query_buffered_transfer(ticket,false)} else {self.graph.query_transfer()};
        result.map_err(|_|{self.poisoned=true;bad("native completion failed; close required")})
    }
    pub fn wait_rows(&mut self)->Result<Vec<wire::RowResult<'_>>> {
        if self.poisoned || (self.window.is_some() || self.window_predecessor_ticket.is_some()) || self.retained.is_none() || self.completed {return Err(bad("no pending iteration"));}
        let phase = self.host_phase_timing.as_ref().map(|_| Instant::now());
        let result=if let Some(ticket)=self.buffered_ticket {self.graph.query_buffered_transfer(ticket,true).map(|_|())} else {self.graph.wait_transfer()};
        if result.is_err() {self.poisoned=true;return Err(bad("native completion failed; close required"));}
        runtime_phase_record(&mut self.host_phase_timing, 3, phase);
        self.collect_rows()
    }
    fn collect_rows(&mut self)->Result<Vec<wire::RowResult<'_>>> {
        let phase = self.host_phase_timing.as_ref().map(|_| Instant::now());
        if self.poisoned || (self.window.is_some() || self.window_predecessor_ticket.is_some()) || self.retained.is_none() || self.completed {return Err(bad("no pending iteration"));}
        let compact=self.compact && self.retained.as_ref().unwrap().mode==super::multi_descriptor::ResultMode::Greedy;
        let output_bytes=if compact{wire::Layout::<ROWS>::COMPACT_RESULT_BYTES}else{self.output.len()};
        let result=if let Some(ticket)=self.buffered_ticket {self.graph.read_buffered_transfer(ticket,&mut self.output[..output_bytes])} else {self.graph.read_transfer(&mut self.output[..output_bytes])};
        if result.is_err() {self.poisoned=true;return Err(bad("native read failed; close required"));}
        self.buffered_ticket=None;
        let e=self.retained.as_ref().unwrap();
        match if compact {wire::validate_compact_result(&self.output[..output_bytes],e)}else if self.shared {wire::validate_batch_result(&self.output,e)}else{wire::validate_result(&self.output,e).map(|(token,logits)|vec![wire::RowResult{output_slot:0,token,logits}])} {
            Ok(result)=>{self.completed=true;runtime_phase_record(&mut self.host_phase_timing, 4, phase);Ok(result)},
            Err(_)=>{self.poisoned=true;Err(bad("invalid GPU completion; close required"))}
        }
    }
    /// Only after successful scheduler settlement of this exact iteration.
    pub fn confirm_scheduler_commit(&mut self,iteration:u64)->Result<()> {
        if self.poisoned || (self.window.is_some() || self.window_predecessor_ticket.is_some()) || !self.completed || self.retained.as_ref().map(|e|e.iteration_id)!=Some(iteration) {return Err(bad("commit differs from completed iteration"));}
        self.identity.last_accepted_replay=self.retained.take().unwrap().replay_id;self.completed=false;Ok(())
    }
    pub(crate) fn into_recorded_parts(self)->(G,VariableSessionIdentity) {(self.graph,self.identity)}
}

/// Cold scratch for the fixed SmolLM2 V3 implementation. KV and weights remain
/// in the loaded model owner and are exclusively borrowed during the session.
pub struct VariableGraphBuffers {
    pub(crate) fa3_attention:bool,
    pub(crate) flashinfer_prefill_only:bool,
    pub(crate) attention_workspace:Option<riley_cuda::CudaDeviceBuffer>,
    pub(crate) ffn_pipeline:bool,
    pub(crate) prefill_ffn_pipeline:bool,
    pub(crate) adaptive_decode:bool,
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
        Ok(Self{fa3_attention:false,flashinfer_prefill_only:false,ffn_pipeline:false,prefill_ffn_pipeline:false,adaptive_decode:false,attention_workspace:None,devices,tiled,shared_devices:vec![],shared_head:None,staging:context.allocate_pinned_host_buffer(196864)?,
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
    fn submit_future_transfer(&mut self,input:&[u8],references:&[u8],predecessor:u64)->riley_cuda::CudaResult<u64>;
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
    fn submit_future_transfer(&mut self,input:&[u8],references:&[u8],predecessor:u64)->riley_cuda::CudaResult<u64> {BorrowedGraphResourceReservation::submit_future_transfer(self,input,references,predecessor)}
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
    fn submit_future_transfer(&mut self,input:&[u8],references:&[u8],predecessor:u64)->riley_cuda::CudaResult<u64> {riley_cuda::OwnedGraphResourceReservation::submit_future_transfer(self,input,references,predecessor)}
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
    /// single iterations and opt-in paired decode windows.
    pub fn into_owned_buffered_variable_mixed_session(self,context:&riley_cuda::CudaContext,capacity:u32,compact:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<32>>{self.into_variable_session_mode::<32>(context,capacity,true,compact,true,true,true,false,false,false,false)}
    fn into_variable_session<const ROWS:usize>(self,context:&riley_cuda::CudaContext,capacity:u32,shared:bool,compact:bool,packed:bool,mixed:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<ROWS>>{
        self.into_variable_session_mode::<ROWS>(context,capacity,shared,compact,packed,mixed,false,false,false,false,false)
    }
    /// Explicit experimental numerical profile: FlashInfer 0.6.16.post3 unsplit
    /// decode rows in pure and mixed stages; existing prefill arithmetic. Quality is unaccepted.
    /// Fails if optional native build support is absent; never falls back silently.
    pub fn into_owned_variable_flashinfer_experimental_session(self,context:&riley_cuda::CudaContext,capacity:u32,compact:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<32>> {
        self.into_variable_session_mode::<32>(context,capacity,true,compact,true,true,false,true,false,false,false)
    }
    /// Experimental FFN transport, existing attention and prefill arithmetic.
    pub fn into_owned_variable_ffn_pipeline_session(self,context:&riley_cuda::CudaContext,capacity:u32,compact:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<32>> {
        self.into_variable_session_mode::<32>(context,capacity,true,compact,true,true,false,false,true,false,false)
    }
    /// Experimental prefill attention; existing decode arithmetic in both stages.
    pub fn into_owned_variable_flashinfer_prefill_session(self,context:&riley_cuda::CudaContext,capacity:u32,compact:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<32>> {
        self.into_variable_session_mode::<32>(context,capacity,true,compact,true,true,false,false,false,true,false)
    }
    /// Opt-in mixed/prefill FFN transport with unchanged pure V7 decode, optionally buffered.
    pub fn into_owned_variable_prefill_ffn_pipeline_session(self,context:&riley_cuda::CudaContext,capacity:u32,compact:bool,buffered:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<32>> {
        self.into_variable_session_mode::<32>(context,capacity,true,compact,true,true,buffered,false,false,false,true)
    }
    fn into_variable_session_mode<const ROWS:usize>(self,context:&riley_cuda::CudaContext,capacity:u32,shared:bool,compact:bool,packed:bool,mixed:bool,buffered:bool,flashinfer:bool,ffn_pipeline:bool,prefill_only:bool,prefill_ffn_pipeline:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<ROWS>>{
        self.into_variable_session_profile::<ROWS>(context,capacity,shared,compact,packed,mixed,buffered,flashinfer,ffn_pipeline,prefill_only,prefill_ffn_pipeline,false,false)
    }
    /// Experimental Hopper FA3 model graph. Not an exact numerical profile.
    pub fn into_owned_variable_fa3_session(self,context:&riley_cuda::CudaContext,capacity:u32,compact:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<32>> {
        if !riley_cuda::FA3_COMPILED {return Err(super::LlamaBatchExecutorError::InvalidConfiguration{field:"FA3 backend",reason:"FA3 was not compiled"});}
        self.into_variable_session_profile::<32>(context,capacity,true,compact,true,true,false,false,false,false,false,true,false)
    }
    /// Exact-order adaptive row tiles for pure decode, including paired graphs.
    pub fn into_owned_variable_adaptive_decode_session(self,context:&riley_cuda::CudaContext,capacity:u32,compact:bool,buffered:bool,prefill_ffn:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<32>> {
        self.into_variable_session_profile::<32>(context,capacity,true,compact,true,true,buffered,false,false,false,prefill_ffn,false,true)
    }
    fn into_variable_session_profile<const ROWS:usize>(self,context:&riley_cuda::CudaContext,capacity:u32,shared:bool,compact:bool,packed:bool,mixed:bool,buffered:bool,flashinfer:bool,ffn_pipeline:bool,prefill_only:bool,prefill_ffn_pipeline:bool,fa3:bool,adaptive:bool)->super::LlamaBatchExecutorResult<OwnedVariableSession<ROWS>>{
        let cuda=|e|super::executor::error::cuda_error(super::ExecutionSite::global(super::LlamaOp::IterationCompletion),e);
        if (mixed&&!packed) || (packed && (ROWS!=32||!shared)) || !matches!(ROWS,8|16|32) || (compact && (!shared || !matches!(ROWS,16|32))) {return Err(super::LlamaBatchExecutorError::InvalidConfiguration{field:"wire rows",reason:"unsupported execution width"});}
        let mut parents=VariableModelParents{executor:self,stream:context.create_stream().map_err(cuda)?,scratch:if shared {VariableGraphBuffers::prepare_shared_rows::<ROWS>(context,capacity)}else{VariableGraphBuffers::prepare(context,capacity)}.map_err(cuda)?};
        parents.scratch.compact=compact;parents.scratch.packed_prefill=packed;parents.scratch.mixed_execution=mixed;
        parents.scratch.ffn_pipeline=ffn_pipeline;
        parents.scratch.prefill_ffn_pipeline=prefill_ffn_pipeline;
        parents.scratch.flashinfer_prefill_only=prefill_only;
        parents.scratch.fa3_attention=fa3;
        parents.scratch.adaptive_decode=adaptive;
        if fa3 {parents.scratch.attention_workspace=Some(context.allocate_device_buffer(riley_cuda::FA3_MODEL_WORKSPACE_BYTES).map_err(cuda)?);}
        if flashinfer || prefill_only {parents.scratch.attention_workspace=Some(context.allocate_device_buffer(if prefill_only{33996}else{78256}).map_err(cuda)?);}
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

// Window state uses the same retained native owner and stream. No Python path.
impl<G:VariableGraph> VariableSession<G,32> {
    pub fn issue_decode_window(&mut self,rows:usize)->Result<(VariableSessionIdentity,u64,Vec<u64>,Vec<u64>)> {
        if !self.buffered || !self.compact || !self.identity.mixed_execution {return Err(bad("window requires buffered V7 compact session"));}
        self.identity.last_accepted_replay.checked_add(2).ok_or_else(||bad("window replay exhausted"))?;
        self.next_cookie.checked_add((rows as u64).checked_mul(2).ok_or_else(||bad("window cookies exhausted"))?).ok_or_else(||bad("window cookies exhausted"))?;
        let(owner,replay,first)=self.issue_rows(rows)?;
        let start=self.next_cookie;self.next_cookie+=rows as u64;
        let second:Vec<_>=(start..self.next_cookie).collect();self.issued_successor=Some(second.clone());
        Ok((owner,replay,first,second))
    }
    fn check_decode_window_submission(&self, first: &wire::Expectation<32>, second: &wire::Expectation<32>) -> Result<()> {
        if self.poisoned || self.retained.is_some() || (self.window.is_some() || self.window_predecessor_ticket.is_some()) || !self.buffered || !self.compact {return Err(bad("window session busy or unavailable"));}
        let cookies:Vec<_>=second.rows.iter().map(|r|r.cookie).collect();
        if self.issued_successor.as_deref()!=Some(cookies.as_slice()) {return Err(bad("successor cookies differ from issue"));}
        if first.rows.len()!=second.rows.len() || first.rows.iter().chain(&second.rows).any(|r|r.progress.stage!=super::multi_descriptor::shape_progress::InputStage::Decode) {return Err(bad("window requires two pure decode batches"));}
        Ok(())
    }
    /// Submit the already-reserved predecessor before successor CPU preparation.
    /// The issued successor cookies and all model parents remain retained.
    pub fn submit_decode_window_predecessor(&mut self,first:wire::Expectation<32>)->Result<()> {
        if self.poisoned || self.retained.is_some() || self.window.is_some() || self.window_predecessor_ticket.is_some()
            || !self.buffered || !self.compact || !self.identity.mixed_execution
            || self.issued_successor.as_ref().map(Vec::len)!=Some(first.rows.len())
            || first.rows.iter().any(|r|r.progress.stage!=super::multi_descriptor::shape_progress::InputStage::Decode) {
            return Err(bad("no eligible issued window predecessor"));
        }
        let issued=self.issued_successor.take();
        let retained=self.retain_submission(first);
        self.issued_successor=issued;
        retained?;
        let phase=self.host_phase_timing.as_ref().map(|_|Instant::now());
        match self.graph.submit_buffered_transfer(&self.input) {
            Ok(ticket)=>self.window_predecessor_ticket=Some(ticket),
            Err(_)=>{self.poisoned=true;return Err(bad("predecessor submission failed; close required before KV release"));}
        }
        runtime_phase_record(&mut self.host_phase_timing,2,phase);
        Ok(())
    }
    /// Run CPU preparation after predecessor submission, then bind and submit its
    /// successor. Any preparation/validation/submit failure quarantines this pair;
    /// it must never be reported as NotDispatched or free its KV before close.
    pub fn submit_decode_window_successor(&mut self,prepare:impl FnOnce()->Result<wire::Expectation<32>>)->Result<()> {
        if self.poisoned || self.window_predecessor_ticket.is_none() || self.window.is_some() || self.retained.is_none() {
            return Err(bad("no retained predecessor awaiting preparation"));
        }
        let result=(|| {
            let phase=self.host_phase_timing.as_ref().map(|_|Instant::now());
            let second=prepare()?;
            let first=self.retained.as_ref().unwrap();
            if first.rows.len()!=second.rows.len() || second.rows.iter().any(|r|r.progress.stage!=super::multi_descriptor::shape_progress::InputStage::Decode)
                || self.issued_successor.as_deref()!=Some(second.rows.iter().map(|r|r.cookie).collect::<Vec<_>>().as_slice()) {
                return Err(bad("successor differs from issued window"));
            }
            let sources=(0..second.rows.len()).map(|i|super::multi_descriptor::future_token::TokenSource::PreviousRow(i as u32)).collect::<Vec<_>>();
            let future=super::multi_descriptor::future_token::prepare(first,&second,&sources)?;
            let output=vec![0;wire::Layout::<32>::COMPACT_RESULT_BYTES];
            runtime_phase_record(&mut self.host_phase_timing,5,phase);
            let first_ticket=self.window_predecessor_ticket.take().unwrap();
            self.window=Some(DecodeWindowState{successor:second,tickets:[first_ticket,0],output,predecessor_validated:false,complete:false});
            self.issued_successor=None;
            let phase=self.host_phase_timing.as_ref().map(|_|Instant::now());
            let ticket=self.graph.submit_future_transfer(future.packet(),future.references(),first_ticket).map_err(|_|bad("successor submission failed; close required before KV release"))?;
            self.window.as_mut().unwrap().tickets[1]=ticket;
            runtime_phase_record(&mut self.host_phase_timing,2,phase);
            Ok(())
        })();
        if result.is_err(){self.poisoned=true;}result
    }
    /// Compatibility entry point: construct one checked, immutable future window.
    pub fn submit_decode_window(&mut self,first:wire::Expectation<32>,second:wire::Expectation<32>)->Result<()> {
        use super::multi_descriptor::future_token::{PreparedFutureWindow, TokenSource};
        self.check_decode_window_submission(&first, &second)?;
        let phase = self.host_phase_timing.as_ref().map(|_| Instant::now());
        let sources:Vec<_>=(0..second.rows.len()).map(|r|TokenSource::PreviousRow(r as u32)).collect();
        let prepared = PreparedFutureWindow::new(first, second, &sources)?;
        runtime_phase_record(&mut self.host_phase_timing, 5, phase);
        self.submit_prepared_decode_window(prepared)
    }
    /// Consume expectations and their bound bytes without rebuilding the future packet.
    /// Current owner, issued cookies and eligibility are still checked at admission.
    pub fn submit_prepared_decode_window(&mut self, prepared: super::multi_descriptor::future_token::PreparedFutureWindow) -> Result<()> {
        let (first, second, future) = prepared.into_parts();
        self.check_decode_window_submission(&first, &second)?;
        let output=vec![0;wire::Layout::<32>::COMPACT_RESULT_BYTES];
        // retain_submission validates actual owner, geometry and first cookies.
        let issued=self.issued_successor.take();
        if let Err(e)=self.retain_submission(first) {self.issued_successor=issued;return Err(e);}
        self.window=Some(DecodeWindowState{successor:second,tickets:[0;2],output,predecessor_validated:false,complete:false});
        let phase = self.host_phase_timing.as_ref().map(|_| Instant::now());
        let submit=(|| {
            let first=self.graph.submit_buffered_transfer(&self.input)?;
            self.window.as_mut().unwrap().tickets[0]=first;
            let second=self.graph.submit_future_transfer(future.packet(),future.references(),first)?;
            self.window.as_mut().unwrap().tickets[1]=second;
            Ok::<_,riley_cuda::CudaError>(())
        })();
        if submit.is_err() {self.poisoned=true;return Err(bad("window submission failed; close required before KV release"));}
        runtime_phase_record(&mut self.host_phase_timing, 2, phase);
        Ok(())
    }
    /// Compatibility drain. Validation of the first result overlaps the second graph.
    pub fn wait_decode_window(&mut self)->Result<[Vec<(u32,u32)>;2]> {
        let first=self.wait_decode_window_first()?;
        let second=self.wait_decode_window_second()?;
        Ok([first,second])
    }
    /// Read only the predecessor event. The successor remains retained and pending;
    /// this neither commits replay nor allows another issue or slot reuse.
    pub fn wait_decode_window_first(&mut self)->Result<Vec<(u32,u32)>> {
        if self.poisoned || !self.window.as_ref().is_some_and(|w|!w.complete && !w.predecessor_validated) {return Err(bad("no unread window predecessor"));}
        let result=(|| {
            let window=self.window.as_mut().unwrap();
            let phase=self.host_phase_timing.as_ref().map(|_|Instant::now());
            if !self.graph.query_buffered_transfer(window.tickets[0],true).map_err(|_|bad("predecessor completion failed"))? {return Err(bad("predecessor completion not established"));}
            runtime_phase_record(&mut self.host_phase_timing,3,phase);
            let phase=self.host_phase_timing.as_ref().map(|_|Instant::now());
            let bytes=wire::Layout::<32>::COMPACT_RESULT_BYTES;
            self.graph.read_buffered_transfer(window.tickets[0],&mut self.output[..bytes]).map_err(|_|bad("predecessor read failed"))?;
            let first=wire::validate_compact_result(&self.output[..bytes],self.retained.as_ref().unwrap())?;
            let rows=first.iter().map(|r|r.token.map(|t|(r.output_slot,t)).ok_or_else(||bad("missing decode output"))).collect::<Result<Vec<_>>>()?;
            // Bind the successor to validated GPU predecessor tokens while its
            // device execution continues. This expectation is host-owned only.
            window.successor.last_accepted_replay=self.retained.as_ref().unwrap().replay_id;
            for (row,previous) in window.successor.rows.iter_mut().zip(first.iter()) {
                row.input_tokens[0]=previous.token.ok_or_else(||bad("missing future input"))?;
            }
            window.predecessor_validated=true;
            runtime_phase_record(&mut self.host_phase_timing,4,phase);
            Ok(rows)
        })();
        if result.is_err(){self.poisoned=true;}result
    }
    /// Fence and validate the successor after the predecessor was validated.
    /// Both reservations remain retained until the scheduler commits the pair.
    pub fn wait_decode_window_second(&mut self)->Result<Vec<(u32,u32)>> {
        if self.poisoned || !self.window.as_ref().is_some_and(|w|!w.complete && w.predecessor_validated) {return Err(bad("no pending validated window successor"));}
        let result=(|| {
            let window=self.window.as_mut().unwrap();
            let phase=self.host_phase_timing.as_ref().map(|_|Instant::now());
            if !self.graph.query_buffered_transfer(window.tickets[1],true).map_err(|_|bad("window completion failed"))? {return Err(bad("successor completion not established"));}
            runtime_phase_record(&mut self.host_phase_timing,3,phase);
            let phase=self.host_phase_timing.as_ref().map(|_|Instant::now());
            self.graph.read_buffered_transfer(window.tickets[1],&mut window.output).map_err(|_|bad("successor read failed"))?;
            let rows=wire::validate_compact_result(&window.output,&window.successor)?.iter().map(|r|r.token.map(|t|(r.output_slot,t)).ok_or_else(||bad("missing successor output"))).collect::<Result<Vec<_>>>()?;
            window.complete=true;
            runtime_phase_record(&mut self.host_phase_timing,4,phase);
            Ok(rows)
        })();
        if result.is_err(){self.poisoned=true;}result
    }
    /// Call only after scheduler settlement of both drained iterations succeeds.
    pub fn confirm_decode_window_commit(&mut self,first:u64,second:u64)->Result<()> {
        let w=self.window.as_ref().ok_or_else(||bad("no completed window"))?;
        if self.poisoned || !w.complete || w.successor.iteration_id!=second || self.retained.as_ref().map(|e|e.iteration_id)!=Some(first) {return Err(bad("window commit differs from completed pair"));}
        self.identity.last_accepted_replay=w.successor.replay_id;
        self.window=None;self.retained=None;self.completed=false;Ok(())
    }
}

#[cfg(test)]
mod staged_window_tests {
    use super::*;
    use super::super::multi_descriptor::shape_progress::InputStage;
    struct FakeGraph { outputs:[Vec<u8>;2], calls:Vec<(&'static str,u64)> }
    impl sealed::Sealed for FakeGraph {}
    impl VariableGraph for FakeGraph {
        fn query_buffered_transfer(&mut self,ticket:u64,wait:bool)->riley_cuda::CudaResult<bool>{assert!(wait);self.calls.push(("wait",ticket));Ok(true)}
        fn read_buffered_transfer(&mut self,ticket:u64,out:&mut[u8])->riley_cuda::CudaResult<()>{self.calls.push(("read",ticket));out.copy_from_slice(&self.outputs[ticket as usize-1]);Ok(())}
        fn submit_future_transfer(&mut self,_:&[u8],_:&[u8],predecessor:u64)->riley_cuda::CudaResult<u64>{assert_eq!(predecessor,1);self.calls.push(("submit_future",2));Ok(2)}
        fn submit_buffered_transfer(&mut self,_:&[u8])->riley_cuda::CudaResult<u64>{self.calls.push(("submit",1));Ok(1)}
        fn replay_transfer(&mut self,_:&[u8])->riley_cuda::CudaResult<()>{unreachable!()}
        fn submit_transfer(&mut self,_:&[u8])->riley_cuda::CudaResult<()>{unreachable!()}
        fn query_transfer(&mut self)->riley_cuda::CudaResult<bool>{unreachable!()}
        fn wait_transfer(&mut self)->riley_cuda::CudaResult<()>{unreachable!()}
        fn read_transfer(&mut self,_:&mut[u8])->riley_cuda::CudaResult<()>{unreachable!()}
    }
    fn pending()->VariableSession<FakeGraph,32>{
        let mut first=wire::tests::fixture_rows::<32>(InputStage::Decode,1);
        first.mixed_execution=true;first.packed_prefill=true;
        let mut second=first.clone();second.iteration_id+=1;second.replay_id+=1;
        second.last_accepted_replay=first.replay_id;
        let row=&mut second.rows[0];row.cookie+=1;row.progress.committed_tokens+=1;row.progress.generated_index+=1;*row.valid_tokens.last_mut().unwrap()+=1;row.input_tokens[0]=7;
        let graph=FakeGraph{outputs:[wire::tests::compact_fixture(&first),wire::tests::compact_fixture(&second)],calls:vec![]};
        second.last_accepted_replay=first.last_accepted_replay;second.rows[0].input_tokens[0]=0;
        let mut s=VariableSession::new_shared_mixed(graph,first.catalog_digest,first.physical_block_count,first.rows[0].progress.context_tokens,true).unwrap();
        s.identity.last_accepted_replay=first.last_accepted_replay;s.retained=Some(first);s.buffered=true;
        s.window=Some(DecodeWindowState{successor:second,tickets:[1,2],output:vec![0;wire::Layout::<32>::COMPACT_RESULT_BYTES],predecessor_validated:false,complete:false});s
    }
    fn issued()->(VariableSession<FakeGraph,32>,wire::Expectation<32>,wire::Expectation<32>){
        let mut s=pending();let mut first=s.retained.take().unwrap();let mut second=s.window.take().unwrap().successor;
        first.owner_generation=s.identity.generation;second.owner_generation=s.identity.generation;
        s.issued=Some(first.rows.iter().map(|r|r.cookie).collect());s.issued_successor=Some(second.rows.iter().map(|r|r.cookie).collect());
        s.graph.outputs[0]=wire::tests::compact_fixture(&first);
        let mut actual=second.clone();actual.last_accepted_replay=first.replay_id;actual.rows[0].input_tokens[0]=7;
        s.graph.outputs[1]=wire::tests::compact_fixture(&actual);
        (s,first,second)
    }
    #[test]
    fn successor_preparation_follows_first_submit_and_keeps_pair_exclusive(){
        let(mut s,first,second)=issued();let ids=(first.iteration_id,second.iteration_id);
        s.submit_decode_window_predecessor(first).unwrap();assert_eq!(s.graph.calls,vec![("submit",1)]);
        assert!(s.query_completion().is_err());assert!(s.wait_rows().is_err());assert!(s.wait_decode_window_first().is_err());
        assert!(s.confirm_scheduler_commit(ids.0).is_err());assert!(s.issue_rows(1).is_err());assert!(s.abandon_issued().is_err());
        let mut prepared=false;
        s.submit_decode_window_successor(||{prepared=true;Ok(second)}).unwrap();assert!(prepared);
        assert_eq!(s.graph.calls,vec![("submit",1),("submit_future",2)]);
        s.wait_decode_window().unwrap();s.confirm_decode_window_commit(ids.0,ids.1).unwrap();assert!(s.issue_rows(1).is_ok());
    }
    #[test]
    fn failed_late_preparation_quarantines_live_predecessor(){
        let(mut s,first,_)=issued();s.submit_decode_window_predecessor(first).unwrap();
        assert!(s.submit_decode_window_successor(||Err(bad("injected preparation failure"))).is_err());
        assert!(s.poisoned);assert!(s.retained.is_some());assert_eq!(s.window_predecessor_ticket,Some(1));
        assert_eq!(s.graph.calls,vec![("submit",1)]);assert!(s.abandon_issued().is_err());assert!(s.issue_rows(1).is_err());
    }
    #[test]
    fn late_foreign_cookie_cannot_attach_successor(){
        let(mut s,first,mut second)=issued();s.submit_decode_window_predecessor(first).unwrap();second.rows[0].cookie+=99;
        assert!(s.submit_decode_window_successor(||Ok(second)).is_err());assert!(s.poisoned);assert!(s.retained.is_some());
        assert_eq!(s.graph.calls,vec![("submit",1)]);
    }
    #[test]
    fn predecessor_processing_does_not_wait_successor_or_release_reservations(){
        let mut s=pending();let first=s.retained.as_ref().unwrap().iteration_id;let second=s.window.as_ref().unwrap().successor.iteration_id;
        assert!(s.wait_decode_window_second().is_err());assert!(s.graph.calls.is_empty());
        assert_eq!(s.wait_decode_window_first().unwrap(),vec![(0,7)]);
        assert_eq!(s.graph.calls,vec![("wait",1),("read",1)]);
        assert!(s.issue_rows(1).is_err());assert!(s.confirm_decode_window_commit(first,second).is_err());assert!(s.wait_decode_window_first().is_err());
        assert_eq!(s.wait_decode_window_second().unwrap(),vec![(0,7)]);
        assert_eq!(s.graph.calls,vec![("wait",1),("read",1),("wait",2),("read",2)]);
        assert!(s.issue_rows(1).is_err());assert!(s.wait_decode_window_second().is_err());
        s.confirm_decode_window_commit(first,second).unwrap();assert!(s.issue_rows(1).is_ok());
    }
    #[test]
    fn corrupt_successor_keeps_pair_retained_and_poisoned(){
        let mut s=pending();s.graph.outputs[1][24]^=1;
        s.wait_decode_window_first().unwrap();assert!(s.wait_decode_window_second().is_err());
        assert!(s.poisoned);assert!(s.retained.is_some());assert!(s.window.is_some());assert!(s.issue_rows(1).is_err());
    }
    #[test]
    fn corrupt_predecessor_never_reads_successor(){
        let mut s=pending();s.graph.outputs[0][24]^=1;
        assert!(s.wait_decode_window_first().is_err());assert!(s.wait_decode_window_second().is_err());
        assert_eq!(s.graph.calls,vec![("wait",1),("read",1)]);assert!(s.poisoned);assert!(s.retained.is_some());
    }
}
