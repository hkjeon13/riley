//! Qualified M1 strided GEMM graph with an optional actual workspace parent.
use crate::{CudaDeviceBuffer, CudaError, CudaPreparedStridedGemm, CudaResult, CudaStream};

/// Prepared plan, exact I/O and optional workspace parent borrowed through close.
pub struct BorrowedStridedGemmResources<'a> {
    /// Capture and replay stream.
    pub stream: &'a mut CudaStream,
    /// Existing canonical plan; never reselected during capture.
    pub plan: &'a mut CudaPreparedStridedGemm,
    /// BF16 input.
    pub input: &'a mut CudaDeviceBuffer,
    /// Physical BF16 weight allocation.
    pub weight: &'a mut CudaDeviceBuffer,
    /// BF16 output.
    pub output: &'a mut CudaDeviceBuffer,
    /// Actual workspace parent, absent when no workspace is allocated.
    pub workspace: Option<&'a mut CudaDeviceBuffer>,
}

/// Cold standalone GEMM graph; does not admit an aggregate decode graph.
pub struct BorrowedStridedGemmGraph<'a> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    resources: BorrowedStridedGemmResources<'a>,
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

impl<'a> BorrowedStridedGemmGraph<'a> {
    /// Captures the selected selected no-split plan without replacement resources.
    /// # Errors
    /// Rejects unsupported policy, size, context, alias, lease or CUDA failures.
    pub fn prepare(resources: BorrowedStridedGemmResources<'a>) -> CudaResult<Self> {
        #[cfg(feature = "cuda")]
        {
            resources.plan.validate_strided_graph_capture(
                resources.stream,
                resources.input,
                resources.weight,
                resources.output,
                resources.workspace.as_deref(),
                "BorrowedStridedGemmGraph::prepare",
            )?;
            let native = resources.plan.begin_strided_graph_capture_native(
                resources.stream,
                resources.input,
                resources.weight,
                resources.output,
                resources.workspace.as_deref(),
                super::CudaGraphCaptureMode::ThreadLocal as u32,
            )?;
            super::begin_deferred_capture_contexts();
            let mut capture = CaptureGuard {
                native,
                active: true,
            };
            capture.native.enqueue_canonical_gemm_bf16()?;
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
            let _ = resources;
            Err(CudaError::unavailable("BorrowedStridedGemmGraph::prepare"))
        }
    }
    /// Replays once and waits for known completion.
    ///
    /// # Errors
    /// Any CUDA failure permanently prevents further replay.
    pub fn replay(&mut self) -> CudaResult<()> {
        const OP: &str = "BorrowedStridedGemmGraph::replay";
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
        const OP: &str = "BorrowedStridedGemmGraph::close";
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
