//! Borrowed whole-slab H2D graph; each replay requires guarded fresh-source staging.
use crate::{CudaDeviceBuffer, CudaError, CudaPinnedHostBuffer, CudaResult, CudaStream};

/// Actual pinned source and device destination, exclusively borrowed through close.
pub struct BorrowedH2DResources<'a> {
    /// Capture and replay stream.
    pub stream: &'a mut CudaStream,
    /// Pinned source slab.
    pub input: &'a mut CudaPinnedHostBuffer,
    /// Equal-sized device destination slab.
    pub output: &'a mut CudaDeviceBuffer,
}

/// Cold standalone H2D graph; does not admit an aggregate decode graph.
pub struct BorrowedH2DGraph<'a> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    resources: BorrowedH2DResources<'a>,
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

impl<'a> BorrowedH2DGraph<'a> {
    /// Captures one whole-slab copy on actual source/destination allocations.
    /// # Errors
    /// Rejects unsupported policy, size, context, alias, lease or CUDA failures.
    pub fn prepare(resources: BorrowedH2DResources<'a>) -> CudaResult<Self> {
        #[cfg(feature = "cuda")]
        {
            super::validate_graph_h2d_capture_preflight(
                resources.stream,
                resources.input,
                resources.output,
                "BorrowedH2DGraph::prepare",
            )?;
            let native = resources.stream.native.begin_graph_h2d_capture(
                resources.output.native_handle(),
                resources.input.native_handle(),
                super::CudaGraphCaptureMode::ThreadLocal as u32,
            )?;
            super::begin_deferred_capture_contexts();
            let mut capture = CaptureGuard {
                native,
                active: true,
            };
            capture.native.enqueue_h2d()?;
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
            Err(CudaError::unavailable("BorrowedH2DGraph::prepare"))
        }
    }
    /// Stages one fresh exact payload into the retained pinned owner, then replays.
    ///
    /// # Errors
    /// Any CUDA failure permanently prevents further replay.
    pub fn replay(&mut self, bytes: &[u8]) -> CudaResult<()> {
        const OP: &str = "BorrowedH2DGraph::replay";
        if self.terminal {
            return Err(CudaError::invalid_state(OP, "graph is terminal"));
        }
        if u64::try_from(bytes.len()).ok() != Some(self.resources.input.byte_len()) {
            return Err(CudaError::out_of_range(
                OP,
                "payload must exactly match source slab",
            ));
        }
        self.terminal = true;
        #[cfg(feature = "cuda")]
        {
            self.native
                .stage_h2d_source(self.resources.input.native_handle(), bytes)?;
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
        const OP: &str = "BorrowedH2DGraph::close";
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
