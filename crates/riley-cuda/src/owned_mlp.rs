//! Managed fixed-address HF M1-per-request MLP graphs. A stale graph never
//! replays on a new allocation: keys retain unique identity tags, not buffers.
use crate::error::{CudaError, CudaResult};
use crate::runtime::{CudaResourceIdentity, execution_stream_mut};
use crate::{
    CublasGemmProbeParams, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType,
    CudaDeviceBuffer, CudaExecutionStream, CudaGemmConfig, CudaPreparedCublasGemmProbe,
    CudaStepGraph, GatedMultiplyParams,
};
use std::collections::HashMap;

/// Caller-owned fixed MLP operands. Scratch/output allocations must be distinct.
pub struct CudaOwnedHfMlpParams<'a> {
    pub input: CudaBufferSpan<'a>,
    pub gate_weight: CudaBufferSpan<'a>,
    pub up_weight: CudaBufferSpan<'a>,
    pub down_weight: CudaBufferSpan<'a>,
    pub gate: &'a mut CudaDeviceBuffer,
    pub up: &'a mut CudaDeviceBuffer,
    pub product: &'a mut CudaDeviceBuffer,
    pub output: &'a mut CudaDeviceBuffer,
    /// Each active row is an independent M1 request, never a prefill group.
    pub active_rows: u32,
}
#[derive(Clone, Eq, PartialEq, Hash)]
struct Binding {
    identity: CudaResourceIdentity,
    offset: u64,
    bytes: u64,
}
#[derive(Eq, PartialEq, Hash)]
struct Key {
    stream: CudaResourceIdentity,
    rows: u32,
    bindings: [Binding; 8],
}
/// Logical counters; captures and capacity fallbacks are explicit.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaOwnedHfMlpGraphStats {
    pub hits: u64,
    pub captures: u64,
    pub eager_fallbacks: u64,
    pub stream_resets: u64,
}
/// Bounded MLP graph owner. Every execution validates the live operand bindings
/// and registers plan/buffer leases through command-batch completion. Cached
/// graphs are destroyed before their cuBLAS plans. CUDA allocations may be
/// replaced between calls: their new opaque identities force a fresh capture.
pub struct CudaOwnedHfMlpGraph {
    entries: HashMap<Key, CudaStepGraph>,
    cache_stream: Option<CudaResourceIdentity>,
    gate_plan: CudaPreparedCublasGemmProbe,
    down_plan: CudaPreparedCublasGemmProbe,
    max_entries: usize,
    max_rows: u32,
    hidden: u64,
    intermediate: u64,
    stats: CudaOwnedHfMlpGraphStats,
    poisoned: bool,
}
const OP: &str = "owned HF MLP graph";
impl CudaOwnedHfMlpGraph {
    /// Prepares plans and a bounded host cache; operand storage stays caller-owned.
    pub fn prepare(
        c: &CudaContext,
        max_rows: u32,
        hidden: u64,
        intermediate: u64,
        max_entries: usize,
    ) -> CudaResult<Self> {
        if max_rows == 0 || max_rows > 32 || max_entries == 0 || max_entries > 4096 {
            return Err(CudaError::invalid_argument(
                OP,
                "rows1..32 and cache entries1..4096 required",
            ));
        }
        let gate_plan = c.prepare_owned_hf_gemm(CudaGemmConfig::new(
            u64::from(max_rows),
            intermediate,
            hidden,
            0,
        )?)?;
        let down_plan = c.prepare_owned_hf_gemm(CudaGemmConfig::new(
            u64::from(max_rows),
            hidden,
            intermediate,
            0,
        )?)?;
        Ok(Self {
            entries: HashMap::with_capacity(max_entries),
            cache_stream: None,
            gate_plan,
            down_plan,
            max_entries,
            max_rows,
            hidden,
            intermediate,
            stats: CudaOwnedHfMlpGraphStats::default(),
            poisoned: false,
        })
    }
    pub fn stats(&self) -> CudaOwnedHfMlpGraphStats {
        self.stats
    }
    /// Submits one exact per-request MLP on an existing command batch. No graph
    /// survives an execution failure as a reusable cache entry.
    pub fn execute<S: CudaExecutionStream + ?Sized>(
        &mut self,
        p: &mut CudaOwnedHfMlpParams<'_>,
        stream: &mut S,
    ) -> CudaResult<()> {
        if self.poisoned {
            return Err(CudaError::invalid_state(
                OP,
                "owner poisoned by previous execution failure",
            ));
        }
        if p.active_rows == 0 || p.active_rows > self.max_rows {
            return Err(CudaError::invalid_argument(
                OP,
                "active rows exceed prepared MLP capacity",
            ));
        }
        let stream = execution_stream_mut(stream);
        let binding = |s: CudaBufferSpan<'_>| Binding {
            identity: s.buffer().identity.clone(),
            offset: s.byte_offset(),
            bytes: s.byte_len(),
        };
        let buffer_binding = |b: &CudaDeviceBuffer| Binding {
            identity: b.identity.clone(),
            offset: 0,
            bytes: b.byte_len(),
        };
        for span in [p.input, p.gate_weight, p.up_weight, p.down_weight] {
            if span.dtype() != CudaDType::BF16 {
                return Err(CudaError::invalid_argument(OP, "BF16 operands required"));
            }
        }
        let key = Key {
            stream: stream.identity.clone(),
            rows: p.active_rows,
            bindings: [
                binding(p.input),
                binding(p.gate_weight),
                binding(p.up_weight),
                binding(p.down_weight),
                buffer_binding(p.gate),
                buffer_binding(p.up),
                buffer_binding(p.product),
                buffer_binding(p.output),
            ],
        };
        // Native validates all capacities, contexts, aliases, batch thread and leases
        // before either a cache hit or a cold capture can submit GPU work.
        self.gate_plan.register_mlp_uses(
            &self.down_plan,
            [
                p.input
                    .subspan(0, self.gate_plan.config().input_bytes())?
                    .raw(),
                p.gate_weight
                    .subspan(0, self.gate_plan.config().weight_bytes())?
                    .raw(),
                p.up_weight
                    .subspan(0, self.gate_plan.config().weight_bytes())?
                    .raw(),
                p.down_weight
                    .subspan(0, self.down_plan.config().weight_bytes())?
                    .raw(),
                p.gate.native_handle().span(
                    crate::ffi::DTYPE_BF16,
                    0,
                    self.gate_plan.config().output_bytes(),
                ),
                p.up.native_handle().span(
                    crate::ffi::DTYPE_BF16,
                    0,
                    self.gate_plan.config().output_bytes(),
                ),
                p.product.native_handle().span(
                    crate::ffi::DTYPE_BF16,
                    0,
                    self.gate_plan.config().output_bytes(),
                ),
                p.output.native_handle().span(
                    crate::ffi::DTYPE_BF16,
                    0,
                    self.down_plan.config().output_bytes(),
                ),
            ],
            stream,
        )?;
        // A classic cuBLAS handle can change its internal workspace when its
        // stream changes. Retire old graphs before any such handle rebinding;
        // a stream identity match alone cannot preserve library workspace.
        if self
            .cache_stream
            .as_ref()
            .is_some_and(|id| *id != stream.identity)
        {
            for (_, graph) in self.entries.drain() {
                if let Err(e) = graph.close() {
                    self.poisoned = true;
                    return Err(e);
                }
            }
            self.stats.stream_resets = self.stats.stream_resets.saturating_add(1);
        }
        self.cache_stream = Some(stream.identity.clone());
        if let Some(graph) = self.entries.get(&key) {
            let result = crate::launch_step_graph(stream, graph);
            if result.is_err() {
                self.poisoned = true;
            } else {
                self.stats.hits = self.stats.hits.saturating_add(1);
            }
            return result;
        }
        if self.entries.len() == self.max_entries {
            self.stats.eager_fallbacks = self.stats.eager_fallbacks.saturating_add(1);
            let result = self.eager(p, stream);
            if result.is_err() {
                self.poisoned = true;
            }
            return result;
        }
        // Prime shape-specific lazy cuBLAS state outside capture. This computes the
        // same pure MLP twice on a cold miss; no residual/metadata state is modified.
        if let Err(e) = self.eager(p, stream) {
            self.poisoned = true;
            return Err(e);
        }
        if let Err(e) = crate::begin_step_capture(stream) {
            self.poisoned = true;
            return Err(e);
        }
        if let Err(e) = self.eager(p, stream) {
            let _ = crate::abort_step_capture(stream);
            self.poisoned = true;
            return Err(e);
        }
        let graph = match crate::end_step_capture(stream) {
            Ok(g) => g,
            Err(e) => {
                self.poisoned = true;
                return Err(e);
            }
        };
        if let Err(e) = crate::launch_step_graph(stream, &graph) {
            self.poisoned = true;
            return Err(e);
        }
        self.entries.insert(key, graph);
        self.stats.captures = self.stats.captures.saturating_add(1);
        Ok(())
    }
    fn eager<S: CudaExecutionStream + ?Sized>(
        &mut self,
        p: &mut CudaOwnedHfMlpParams<'_>,
        s: &mut S,
    ) -> CudaResult<()> {
        let mut slots = [0u32; 32];
        for (i, v) in slots[..p.active_rows as usize].iter_mut().enumerate() {
            *v = i as u32;
        }
        let slots = &slots[..p.active_rows as usize];
        let gate_bytes = self.gate_plan.config().output_bytes();
        self.gate_plan.execute_row_groups(
            &mut CublasGemmProbeParams {
                input: p.input.subspan(0, self.gate_plan.config().input_bytes())?,
                weight: p
                    .gate_weight
                    .subspan(0, self.gate_plan.config().weight_bytes())?,
                output: CudaBufferSpanMut::new(p.gate, CudaDType::BF16, 0, gate_bytes)?,
            },
            slots,
            None,
            0,
            s,
        )?;
        self.gate_plan.execute_row_groups(
            &mut CublasGemmProbeParams {
                input: p.input.subspan(0, self.gate_plan.config().input_bytes())?,
                weight: p
                    .up_weight
                    .subspan(0, self.gate_plan.config().weight_bytes())?,
                output: CudaBufferSpanMut::new(p.up, CudaDType::BF16, 0, gate_bytes)?,
            },
            slots,
            None,
            0,
            s,
        )?;
        crate::silu_gated_multiply(
            &mut GatedMultiplyParams {
                activated_gate: CudaBufferSpan::new(p.gate, CudaDType::BF16, 0, gate_bytes)?,
                up: CudaBufferSpan::new(p.up, CudaDType::BF16, 0, gate_bytes)?,
                output: CudaBufferSpanMut::new(p.product, CudaDType::BF16, 0, gate_bytes)?,
                element_count: u64::from(p.active_rows) * self.intermediate,
            },
            s,
        )?;
        self.down_plan.execute_row_groups(
            &mut CublasGemmProbeParams {
                input: CudaBufferSpan::new(p.product, CudaDType::BF16, 0, gate_bytes)?,
                weight: p
                    .down_weight
                    .subspan(0, self.down_plan.config().weight_bytes())?,
                output: CudaBufferSpanMut::new(
                    p.output,
                    CudaDType::BF16,
                    0,
                    u64::from(self.max_rows) * self.hidden * 2,
                )?,
            },
            slots,
            None,
            0,
            s,
        )
    }
    pub fn close(mut self) -> CudaResult<()> {
        for (_, g) in self.entries.drain() {
            g.close()?;
        }
        self.gate_plan.close()?;
        self.down_plan.close()
    }
}
