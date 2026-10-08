//! Borrowed standalone GPU greedy selection; host transfer follows graph close.
use crate::{CudaDeviceBuffer, CudaError, CudaResult, CudaStream};

/// Actual logits and result buffers, exclusively borrowed through close.
pub struct BorrowedArgmaxResources<'a> {
    /// Capture and replay stream.
    pub stream: &'a mut CudaStream,
    /// BF16 input.
    pub input: &'a mut CudaDeviceBuffer,
    /// U32 token/status records.
    pub output: &'a mut CudaDeviceBuffer,
}

/// Cold standalone argmax graph; does not admit an aggregate decode graph.
pub struct BorrowedArgmaxGraph<'a> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    resources: BorrowedArgmaxResources<'a>,
    terminal: bool,
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

impl<'a> BorrowedArgmaxGraph<'a> {
    /// Captures deterministic argmax on the actual logits and result allocations.
    /// # Errors
    /// Rejects unsupported policy, size, context, alias, lease or CUDA failures.
    pub fn prepare(
        resources: BorrowedArgmaxResources<'a>,
        rows: u64,
        vocabulary: u64,
    ) -> CudaResult<Self> {
        #[cfg(feature = "cuda")]
        {
            super::validate_graph_bf16_argmax_capture_preflight(
                resources.stream,
                resources.input,
                resources.output,
                rows,
                vocabulary,
                "BorrowedArgmaxGraph::prepare",
            )?;
            let native = resources.stream.native.begin_graph_bf16_argmax_capture(
                resources.input.native_handle(),
                resources.output.native_handle(),
                rows,
                vocabulary,
                super::CudaGraphCaptureMode::ThreadLocal as u32,
            )?;
            super::begin_deferred_capture_contexts();
            let mut capture = CaptureGuard {
                native,
                active: true,
            };
            capture.native.enqueue_bf16_argmax()?;
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
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (resources, rows, vocabulary);
            Err(CudaError::unavailable("BorrowedArgmaxGraph::prepare"))
        }
    }
    /// Replays once and waits for known completion.
    ///
    /// # Errors
    /// Any CUDA failure permanently prevents further replay.
    pub fn replay(&mut self) -> CudaResult<()> {
        const OP: &str = "BorrowedArgmaxGraph::replay";
        if self.terminal {
            return Err(CudaError::invalid_state(OP, "graph is terminal"));
        }
        self.terminal = true;
        #[cfg(feature = "cuda")]
        {
            self.native
                .launch(&mut self.resources.stream.native)?
                .complete()?;
            self.terminal = false;
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        Err(CudaError::unavailable(OP))
    }

    /// Releases graph leases before returning access to the executor allocations.
    ///
    /// # Errors
    /// Rejects terminal state and propagates native destruction failures.
    pub fn close(mut self) -> CudaResult<()> {
        const OP: &str = "BorrowedArgmaxGraph::close";
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
