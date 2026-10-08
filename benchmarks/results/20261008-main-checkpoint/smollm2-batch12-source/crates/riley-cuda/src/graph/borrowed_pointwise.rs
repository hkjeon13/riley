//! Borrowed standalone BF16 pointwise graph primitives.
use crate::{CudaDeviceBuffer, CudaError, CudaResult, CudaStream};

/// Exact primitive; variants never inherit one another's graph evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PointwiseGraphOperation {
    /// Standalone BF16 activation.
    Silu,
    /// Activated gate multiplied by up projection.
    GatedMultiply,
    /// Standalone residual addition.
    ResidualAdd,
}

impl PointwiseGraphOperation {
    /// Validates nonempty BF16 size and exact unary/binary arity.
    /// # Errors
    /// Rejects zero/overflowing size or wrong number of inputs.
    pub fn validate(self, elements: u64, has_other: bool) -> CudaResult<()> {
        if elements == 0 || elements.checked_mul(2).is_none() || has_other != (self != Self::Silu) {
            return Err(CudaError::invalid_argument(
                "PointwiseGraphOperation::validate",
                "invalid BF16 element count or input arity",
            ));
        }
        Ok(())
    }
}

/// Distinct fixed inputs and output retained with the stream through close.
pub struct BorrowedPointwiseResources<'a> {
    /// Capture and replay stream.
    pub stream: &'a mut CudaStream,
    /// Activation input or first binary operand.
    pub input: &'a mut CudaDeviceBuffer,
    /// Required for binary operations, absent for activation.
    pub other: Option<&'a mut CudaDeviceBuffer>,
    /// Distinct BF16 output allocation.
    pub output: &'a mut CudaDeviceBuffer,
}

/// One exact pointwise node, with exclusive native and Rust resource leases.
pub struct BorrowedPointwiseGraph<'a> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    resources: BorrowedPointwiseResources<'a>,
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

impl<'a> BorrowedPointwiseGraph<'a> {
    /// Captures the existing native primitive on the actual borrowed allocations.
    /// # Errors
    /// Rejects arity, geometry, aliases, context, leases and CUDA failures.
    pub fn prepare(
        resources: BorrowedPointwiseResources<'a>,
        operation: PointwiseGraphOperation,
        elements: u64,
    ) -> CudaResult<Self> {
        const OP: &str = "BorrowedPointwiseGraph::prepare";
        operation.validate(elements, resources.other.is_some())?;
        #[cfg(feature = "cuda")]
        {
            validate_resources(&resources, elements)?;
            let mode = super::CudaGraphCaptureMode::ThreadLocal as u32;
            let native = match operation {
                PointwiseGraphOperation::Silu => {
                    resources.stream.native.begin_graph_silu_bf16_capture(
                        resources.input.native_handle(),
                        resources.output.native_handle(),
                        elements,
                        mode,
                    )?
                }
                PointwiseGraphOperation::GatedMultiply => resources
                    .stream
                    .native
                    .begin_graph_gated_multiply_bf16_capture(
                        resources.input.native_handle(),
                        resources
                            .other
                            .as_deref()
                            .ok_or_else(|| {
                                CudaError::invalid_argument(OP, "missing binary operand")
                            })?
                            .native_handle(),
                        resources.output.native_handle(),
                        elements,
                        mode,
                    )?,
                PointwiseGraphOperation::ResidualAdd => resources
                    .stream
                    .native
                    .begin_graph_residual_add_bf16_capture(
                        resources.input.native_handle(),
                        resources
                            .other
                            .as_deref()
                            .ok_or_else(|| {
                                CudaError::invalid_argument(OP, "missing binary operand")
                            })?
                            .native_handle(),
                        resources.output.native_handle(),
                        elements,
                        mode,
                    )?,
            };
            super::begin_deferred_capture_contexts();
            let mut capture = CaptureGuard {
                native,
                active: true,
            };
            match operation {
                PointwiseGraphOperation::Silu => capture.native.enqueue_silu_bf16()?,
                PointwiseGraphOperation::GatedMultiply => {
                    capture.native.enqueue_gated_multiply_bf16()?;
                }
                PointwiseGraphOperation::ResidualAdd => {
                    capture.native.enqueue_residual_add_bf16()?;
                }
            }
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
    /// Replays once and waits for known completion.
    ///
    /// # Errors
    /// Any CUDA failure permanently prevents further replay.
    pub fn replay(&mut self) -> CudaResult<()> {
        const OP: &str = "BorrowedPointwiseGraph::replay";
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
        const OP: &str = "BorrowedPointwiseGraph::close";
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
    use super::PointwiseGraphOperation as Op;
    #[test]
    fn pointwise_graph_requires_exact_arity_and_nonoverflowing_bf16_span() {
        for op in [Op::Silu, Op::GatedMultiply, Op::ResidualAdd] {
            let binary = op != Op::Silu;
            assert!(op.validate(257, binary).is_ok());
            assert!(op.validate(257, !binary).is_err());
            assert!(op.validate(0, binary).is_err());
            assert!(op.validate(u64::MAX, binary).is_err());
        }
    }
}

#[cfg(feature = "cuda")]
fn validate_resources(resources: &BorrowedPointwiseResources<'_>, elements: u64) -> CudaResult<()> {
    const OP: &str = "BorrowedPointwiseGraph::prepare";
    let buffers = [
        Some(&*resources.input),
        resources.other.as_deref(),
        Some(&*resources.output),
    ];
    for (i, buffer) in buffers.iter().enumerate() {
        let Some(buffer) = buffer else {
            continue;
        };
        super::ensure_same_context(&resources.stream.context, buffer.context_owner(), OP)?;
        buffer.ensure_idle_for_operation(OP)?;
        if buffer.byte_len() < elements * 2 {
            return Err(CudaError::out_of_range(OP, "BF16 span exceeds allocation"));
        }
        for other in buffers[..i].iter().flatten() {
            if buffer
                .native_handle()
                .same_allocation(other.native_handle())
            {
                return Err(CudaError::invalid_argument(
                    OP,
                    "pointwise allocations must be distinct",
                ));
            }
        }
    }
    Ok(())
}
