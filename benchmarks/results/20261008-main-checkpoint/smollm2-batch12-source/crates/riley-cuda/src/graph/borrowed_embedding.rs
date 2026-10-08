//! Borrowed BF16 embedding with completion-validated invalid-token status.
use super::Bf16EmbeddingStatusD2HStatus;
use crate::{CudaDeviceBuffer, CudaError, CudaPinnedHostBuffer, CudaResult, CudaStream};

/// Exact nonempty BF16 table and token batch dimensions.
#[derive(Clone, Copy, Debug)]
pub struct EmbeddingGraphGeometry {
    /// Number of U32 token IDs at allocation byte zero.
    pub tokens: u64,
    /// Number of table rows.
    pub vocabulary: u64,
    /// BF16 elements per table row.
    pub hidden: u64,
}
impl EmbeddingGraphGeometry {
    /// Checks multiplication bounds before allocation access.
    /// # Errors
    /// Rejects zero dimensions and overflowing byte spans.
    pub fn validate(self) -> CudaResult<()> {
        if self.tokens == 0
            || self.vocabulary == 0
            || self.hidden == 0
            || self.tokens.checked_mul(4).is_none()
            || self
                .tokens
                .checked_mul(self.hidden)
                .and_then(|n| n.checked_mul(2))
                .is_none()
            || self
                .vocabulary
                .checked_mul(self.hidden)
                .and_then(|n| n.checked_mul(2))
                .is_none()
        {
            return Err(CudaError::out_of_range(
                "EmbeddingGraphGeometry::validate",
                "invalid or overflowing BF16 embedding geometry",
            ));
        }
        Ok(())
    }
}
/// Actual exclusive device buffers, fixed 32-byte pinned report, and stream.
pub struct BorrowedEmbeddingResources<'a> {
    /// Capture/replay stream.
    pub stream: &'a mut CudaStream,
    /// BF16 embedding table.
    pub table: &'a mut CudaDeviceBuffer,
    /// U32 token prefix, possibly inside a larger metadata parent.
    pub token_ids: &'a mut CudaDeviceBuffer,
    /// Distinct BF16 output allocation.
    pub output: &'a mut CudaDeviceBuffer,
    /// Existing embedding error scratch allocation.
    pub error_scratch: &'a mut CudaDeviceBuffer,
    /// Exact 32-byte pinned status owner.
    pub report: &'a mut CudaPinnedHostBuffer,
}
/// Completion-scoped embedding status with all resources retained through close.
pub struct BorrowedEmbeddingGraph<'a> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    resources: BorrowedEmbeddingResources<'a>,
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

impl<'a> BorrowedEmbeddingGraph<'a> {
    /// Captures the existing validation/gather/status-D2H native primitive.
    /// # Errors
    /// Rejects geometry, resource identity, report size, leases or CUDA failures.
    pub fn prepare(
        resources: BorrowedEmbeddingResources<'a>,
        geometry: EmbeddingGraphGeometry,
    ) -> CudaResult<Self> {
        const OP: &str = "BorrowedEmbeddingGraph::prepare";
        geometry.validate()?;
        #[cfg(feature = "cuda")]
        {
            super::validate_graph_bf16_embedding_status_d2h_capture_preflight(
                resources.stream,
                resources.table,
                resources.token_ids,
                resources.output,
                resources.error_scratch,
                resources.report,
                geometry.tokens,
                geometry.vocabulary,
                geometry.hidden,
                OP,
            )?;
            let native = resources
                .stream
                .native
                .begin_graph_bf16_embedding_status_d2h_capture(
                    resources.table.native_handle(),
                    resources.token_ids.native_handle(),
                    resources.output.native_handle(),
                    resources.error_scratch.native_handle(),
                    resources.report.native_handle(),
                    geometry.tokens,
                    geometry.vocabulary,
                    geometry.hidden,
                    super::CudaGraphCaptureMode::ThreadLocal as u32,
                )?;
            super::begin_deferred_capture_contexts();
            let mut capture = CaptureGuard {
                native,
                active: true,
            };
            capture.native.enqueue_bf16_embedding_status_d2h()?;
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
            Err(CudaError::unavailable(OP))
        }
    }
    /// Replays, waits and validates the native status before returning a value.
    /// An invalid token is a reusable no-output-write outcome, not a CUDA error.
    /// # Errors
    /// CUDA or malformed-report failures permanently prevent replay.
    pub fn replay(&mut self) -> CudaResult<Bf16EmbeddingStatusD2HStatus> {
        const OP: &str = "BorrowedEmbeddingGraph::replay";
        if self.terminal {
            return Err(CudaError::invalid_state(OP, "graph is terminal"));
        }
        self.terminal = true;
        #[cfg(feature = "cuda")]
        {
            self.native
                .launch(&mut self.resources.stream.native)?
                .complete()?;
            let report = self.native.read_bf16_embedding_status_d2h_report()?;
            let status = match report.code {
                0 if report.token_position == 0 && report.token_id == 0 => {
                    Bf16EmbeddingStatusD2HStatus::Success
                }
                1 => Bf16EmbeddingStatusD2HStatus::TokenOutOfRange {
                    token_position: report.token_position,
                    token_id: report.token_id,
                },
                _ => {
                    return Err(CudaError::invalid_state(
                        OP,
                        "inconsistent native embedding status report",
                    ));
                }
            };
            self.terminal = false;
            Ok(status)
        }
        #[cfg(not(feature = "cuda"))]
        Err(CudaError::unavailable(OP))
    }
    /// Releases graph leases before returning access to the executor allocations.
    ///
    /// # Errors
    /// Rejects terminal state and propagates native destruction failures.
    pub fn close(mut self) -> CudaResult<()> {
        const OP: &str = "BorrowedEmbeddingGraph::close";
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

#[cfg(test)]
mod tests {
    use super::EmbeddingGraphGeometry as Shape;
    #[test]
    fn embedding_geometry_rejects_empty_or_overflowing_spans() {
        assert!(
            Shape {
                tokens: 1,
                vocabulary: 49152,
                hidden: 576
            }
            .validate()
            .is_ok()
        );
        for (tokens, vocabulary, hidden) in [
            (0, 10, 64),
            (1, 0, 64),
            (1, 10, 0),
            (u64::MAX, 10, 64),
            (1, u64::MAX, 64),
            (1, 10, u64::MAX),
        ] {
            assert!(
                Shape {
                    tokens,
                    vocabulary,
                    hidden
                }
                .validate()
                .is_err()
            );
        }
    }
}
