//! Borrowed gather/argmax/D2H with parent spans and completion-scoped reads.
use crate::{CudaDeviceBuffer, CudaError, CudaPinnedHostBuffer, CudaResult, CudaStream};

/// Actual logits and result buffers, exclusively borrowed through close.
pub struct BorrowedOutputResources<'a> {
    /// Capture and replay stream.
    pub stream: &'a mut CudaStream,
    /// BF16 input.
    pub input: &'a mut CudaDeviceBuffer,
    /// Packed index parent.
    pub indices: &'a mut CudaDeviceBuffer,
    /// Intermediate gathered logits.
    pub gathered: &'a mut CudaDeviceBuffer,
    /// Actual pinned result parent.
    pub pinned: &'a mut CudaPinnedHostBuffer,
    /// U32 token/status records.
    pub output: &'a mut CudaDeviceBuffer,
}

/// Three-node output graph; does not admit an aggregate decode graph.
pub struct BorrowedOutputGraph<'a> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    resources: BorrowedOutputResources<'a>,
    terminal: bool,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    result_bytes: usize,
}
#[cfg(feature = "cuda")]
struct CaptureGuard {
    native: crate::ffi::GraphCaptureHandle,
    active: bool,
}

#[cfg(feature = "cuda")]
impl Drop for CaptureGuard {
    fn drop(&mut self) {
        if self.active && self.native.abort_with_transition().resource_release_known {
            super::finish_deferred_capture_contexts();
        }
    }
}

impl<'a> BorrowedOutputGraph<'a> {
    /// Captures gather, deterministic argmax and D2H on actual parent allocations.
    /// # Errors
    /// Rejects row-map geometry, size, context, alias, lease or CUDA failures.
    pub fn prepare(
        resources: BorrowedOutputResources<'a>,
        input_rows: u64,
        indices: &[u32],
        vocabulary: u64,
        offset: u64,
    ) -> CudaResult<Self> {
        let rows = indices.len() as u64;
        let result_bytes = indices
            .len()
            .checked_mul(8)
            .ok_or_else(|| CudaError::out_of_range("output graph", "result size overflow"))?;
        #[cfg(feature = "cuda")]
        {
            validate(&resources, input_rows, indices, vocabulary, offset)?;
            let native = resources
                .stream
                .native
                .begin_graph_output_parent_d2h_capture(
                    resources.input.native_handle(),
                    resources.indices.native_handle(),
                    resources.gathered.native_handle(),
                    resources.output.native_handle(),
                    resources.pinned.native_handle(),
                    input_rows,
                    rows,
                    vocabulary,
                    offset,
                    super::CudaGraphCaptureMode::ThreadLocal as u32,
                )?;
            super::begin_deferred_capture_contexts();
            let mut capture = CaptureGuard {
                native,
                active: true,
            };
            capture.native.enqueue_bf16_row_gather_argmax_d2h()?;
            let transition = capture.native.end();
            capture.active = !transition.owner_consumed;
            if transition.resource_release_known {
                super::finish_deferred_capture_contexts();
            }
            let mut graph = transition.result?;
            let native = graph.instantiate()?;
            Ok(Self {
                native,
                resources,
                terminal: false,
                result_bytes,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                resources,
                rows,
                vocabulary,
                input_rows,
                offset,
                result_bytes,
            );
            Err(CudaError::unavailable("BorrowedOutputGraph::prepare"))
        }
    }
    /// Replays the full chain and copies result records only after known completion.
    ///
    /// # Errors
    /// Any CUDA failure permanently prevents further replay.
    pub fn replay(&mut self) -> CudaResult<Vec<u8>> {
        const OP: &str = "BorrowedOutputGraph::replay";
        if self.terminal {
            return Err(CudaError::invalid_state(OP, "graph is terminal"));
        }
        self.terminal = true;
        #[cfg(feature = "cuda")]
        {
            self.native
                .launch(&mut self.resources.stream.native)?
                .complete()?;
            let mut bytes = vec![0; self.result_bytes];
            self.native
                .read_bf16_row_gather_argmax_d2h_results(&mut bytes)?;
            self.terminal = false;
            Ok(bytes)
        }
        #[cfg(not(feature = "cuda"))]
        Err(CudaError::unavailable(OP))
    }

    /// Releases graph leases before returning access to the executor allocations.
    ///
    /// # Errors
    /// Rejects terminal state and propagates native destruction failures.
    pub fn close(mut self) -> CudaResult<()> {
        const OP: &str = "BorrowedOutputGraph::close";
        if self.terminal {
            return Err(CudaError::invalid_state(OP, "graph is terminal"));
        }
        #[cfg(feature = "cuda")]
        {
            self.native.close()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self;
            Err(CudaError::unavailable(OP))
        }
    }
}

#[cfg(feature = "cuda")]
fn validate(
    r: &BorrowedOutputResources<'_>,
    input_rows: u64,
    indices: &[u32],
    vocabulary: u64,
    offset: u64,
) -> CudaResult<()> {
    const OP: &str = "BorrowedOutputGraph::prepare";
    if input_rows == 0
        || indices.is_empty()
        || vocabulary == 0
        || vocabulary > u64::from(u32::MAX)
        || indices.iter().any(|i| u64::from(*i) >= input_rows)
        || offset % 4 != 0
    {
        return Err(CudaError::out_of_range(
            OP,
            "invalid output geometry or row map",
        ));
    }
    let rows = indices.len() as u64;
    let size = |a: u64, b: u64| {
        a.checked_mul(b)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| CudaError::out_of_range(OP, "size overflow"))
    };
    let index_end = rows
        .checked_mul(4)
        .and_then(|n| offset.checked_add(n))
        .ok_or_else(|| CudaError::out_of_range(OP, "index span overflow"))?;
    let result_bytes = rows
        .checked_mul(8)
        .ok_or_else(|| CudaError::out_of_range(OP, "result overflow"))?;
    let buffers = [
        (&*r.input, size(input_rows, vocabulary)?),
        (&*r.indices, index_end),
        (&*r.gathered, size(rows, vocabulary)?),
        (&*r.output, result_bytes),
    ];
    for (i, (buffer, needed)) in buffers.iter().enumerate() {
        super::ensure_same_context(&r.stream.context, buffer.context_owner(), OP)?;
        buffer.ensure_idle_for_operation(OP)?;
        if buffer.byte_len() < *needed {
            return Err(CudaError::out_of_range(OP, "device span too short"));
        }
        for (other, _) in &buffers[..i] {
            if buffer
                .native_handle()
                .same_allocation(other.native_handle())
            {
                return Err(CudaError::invalid_argument(
                    OP,
                    "device parents must be distinct",
                ));
            }
        }
    }
    super::ensure_same_context(&r.stream.context, r.pinned.context_owner(), OP)?;
    r.pinned.ensure_idle_for_operation(OP)?;
    if r.pinned.byte_len() < result_bytes {
        return Err(CudaError::out_of_range(
            OP,
            "pinned result prefix too short",
        ));
    }
    Ok(())
}
