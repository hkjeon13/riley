//! Profile-specific `RMSNorm` capture borrowing the executor's actual allocations.

use crate::{CudaDeviceBuffer, CudaError, CudaResult, CudaStream};

/// Distinct immutable reduction profiles; neither inherits the other's evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormGraphProfile {
    /// Generic BF16 reduction.
    Canonical,
    /// Reviewed HF `SmolLM2` H576 reduction with exact epsilon 1e-5.
    HuggingFaceSmolLm2,
}

impl NormGraphProfile {
    /// Checks profile-specific geometry before entering CUDA.
    ///
    /// # Errors
    /// Rejects zero rows, unsupported HF geometry and noncanonical epsilon bits.
    pub fn validate(self, rows: u64, hidden: u64, epsilon: f32) -> CudaResult<()> {
        if rows == 0
            || hidden == 0
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || (self == Self::HuggingFaceSmolLm2
                && (rows > 8192 || hidden != 576 || epsilon.to_bits() != 1e-5_f32.to_bits()))
        {
            return Err(CudaError::invalid_argument(
                "NormGraphProfile::validate",
                "unsupported norm profile geometry or epsilon",
            ));
        }
        Ok(())
    }
}

/// Fixed BF16 inputs and out-of-place output retained through graph close.
pub struct BorrowedNormResources<'a> {
    /// Exact capture and replay stream.
    pub stream: &'a mut CudaStream,
    /// Row-major input, starting at byte zero.
    pub input: &'a mut CudaDeviceBuffer,
    /// Canonical weight vector, starting at byte zero.
    pub weight: &'a mut CudaDeviceBuffer,
    /// Distinct row-major output, starting at byte zero.
    pub output: &'a mut CudaDeviceBuffer,
}

/// One profile-specific BF16 `RMSNorm` node. Native leases also protect forgotten owners.
pub struct BorrowedNormGraph<'a> {
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphExecHandle,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    resources: BorrowedNormResources<'a>,
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

impl<'a> BorrowedNormGraph<'a> {
    /// Captures existing canonical `RMSNorm` using opaque owners, without copying weights.
    ///
    /// # Errors
    /// Rejects invalid geometry, context or leases and propagates CUDA failures.
    /// This default API selects the canonical profile.
    pub fn prepare(
        resources: BorrowedNormResources<'a>,
        rows: u64,
        hidden: u64,
        epsilon: f32,
    ) -> CudaResult<Self> {
        Self::prepare_with_profile(
            resources,
            rows,
            hidden,
            epsilon,
            NormGraphProfile::Canonical,
        )
    }

    /// Captures the selected exact reduction with the same opaque-owner lifecycle.
    ///
    /// # Errors
    /// Rejects unsupported geometry, mismatched context/leases or CUDA failures.
    pub fn prepare_with_profile(
        resources: BorrowedNormResources<'a>,
        rows: u64,
        hidden: u64,
        epsilon: f32,
        profile: NormGraphProfile,
    ) -> CudaResult<Self> {
        const OP: &str = "BorrowedNormGraph::prepare";
        profile.validate(rows, hidden, epsilon)?;
        #[cfg(feature = "cuda")]
        {
            super::validate_graph_canonical_rms_norm_bf16_capture_preflight(
                resources.stream,
                resources.input,
                resources.weight,
                resources.output,
                rows,
                hidden,
                epsilon,
                OP,
            )?;
            let native = match profile {
                NormGraphProfile::Canonical => resources
                    .stream
                    .native
                    .begin_graph_canonical_rms_norm_bf16_capture(
                        resources.input.native_handle(),
                        resources.weight.native_handle(),
                        resources.output.native_handle(),
                        rows,
                        hidden,
                        epsilon,
                        super::CudaGraphCaptureMode::ThreadLocal as u32,
                    )?,
                NormGraphProfile::HuggingFaceSmolLm2 => resources
                    .stream
                    .native
                    .begin_graph_hf_smollm2_rms_norm_bf16_capture(
                        resources.input.native_handle(),
                        resources.weight.native_handle(),
                        resources.output.native_handle(),
                        rows,
                        hidden,
                        epsilon,
                        super::CudaGraphCaptureMode::ThreadLocal as u32,
                    )?,
            };
            super::begin_deferred_capture_contexts();
            let mut capture = CaptureGuard {
                native,
                active: true,
            };
            match profile {
                NormGraphProfile::Canonical => capture.native.enqueue_canonical_rms_norm_bf16()?,
                NormGraphProfile::HuggingFaceSmolLm2 => {
                    capture.native.enqueue_hf_smollm2_rms_norm_bf16()?;
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
            let _ = (resources, rows, hidden, epsilon);
            Err(CudaError::unavailable(OP))
        }
    }

    /// Replays once and waits for known completion.
    ///
    /// # Errors
    /// Any CUDA failure permanently prevents further replay.
    pub fn replay(&mut self) -> CudaResult<()> {
        const OP: &str = "BorrowedNormGraph::replay";
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
        const OP: &str = "BorrowedNormGraph::close";
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
    use super::NormGraphProfile;

    #[test]
    fn hf_norm_profile_has_exact_geometry_and_epsilon() {
        let hf = NormGraphProfile::HuggingFaceSmolLm2;
        for rows in [1, 3, 8, 16, 17, 8192] {
            assert!(hf.validate(rows, 576, 1e-5).is_ok());
        }
        for (rows, hidden, epsilon) in [
            (0, 576, 1e-5),
            (8193, 576, 1e-5),
            (1, 575, 1e-5),
            (1, 576, f32::from_bits(1e-5_f32.to_bits() + 1)),
            (1, 576, f32::NAN),
        ] {
            assert!(hf.validate(rows, hidden, epsilon).is_err());
        }
        assert!(NormGraphProfile::Canonical.validate(1, 575, 1e-4).is_ok());
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA; raw HF begin/abort/profile isolation"]
    fn hf_norm_native_capture_rejects_profile_swap_and_recovers_after_abort()
    -> Result<(), Box<dyn std::error::Error>> {
        use super::{BorrowedNormGraph, BorrowedNormResources, CaptureGuard};
        use crate::CudaRuntime;
        let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
        let mut stream = context.create_stream()?;
        let mut input = context.allocate_device_buffer(1152)?;
        let mut weight = context.allocate_device_buffer(1152)?;
        let mut output = context.allocate_device_buffer(1152)?;
        let mut staging = context.allocate_pinned_host_buffer(1152)?;
        let ones: Vec<u8> = (0..576).flat_map(|_| 0x3f80_u16.to_le_bytes()).collect();
        input.upload_from_slice(0, &ones, &mut staging, &mut stream)?;
        weight.upload_from_slice(0, &ones, &mut staging, &mut stream)?;
        for (rows, hidden, epsilon) in [
            (0, 576, 1e-5),
            (8193, 576, 1e-5),
            (1, 64, 1e-5),
            (1, 576, 1e-6),
        ] {
            assert!(
                stream
                    .native
                    .begin_graph_hf_smollm2_rms_norm_bf16_capture(
                        input.native_handle(),
                        weight.native_handle(),
                        output.native_handle(),
                        rows,
                        hidden,
                        epsilon,
                        1,
                    )
                    .is_err()
            );
        }
        for hf in [false, true] {
            let native = if hf {
                stream.native.begin_graph_hf_smollm2_rms_norm_bf16_capture(
                    input.native_handle(),
                    weight.native_handle(),
                    output.native_handle(),
                    1,
                    576,
                    1e-5,
                    1,
                )?
            } else {
                stream.native.begin_graph_canonical_rms_norm_bf16_capture(
                    input.native_handle(),
                    weight.native_handle(),
                    output.native_handle(),
                    1,
                    576,
                    1e-5,
                    1,
                )?
            };
            super::super::begin_deferred_capture_contexts();
            let mut guard = CaptureGuard {
                native,
                active: true,
            };
            if hf {
                assert!(guard.native.enqueue_canonical_rms_norm_bf16().is_err());
                guard.native.enqueue_hf_smollm2_rms_norm_bf16()?;
                assert!(guard.native.enqueue_hf_smollm2_rms_norm_bf16().is_err());
            } else {
                assert!(guard.native.enqueue_hf_smollm2_rms_norm_bf16().is_err());
                guard.native.enqueue_canonical_rms_norm_bf16()?;
            }
            // CaptureGuard aborts a queued graph and releases its leases.
            drop(guard);
            let mut graph = BorrowedNormGraph::prepare_with_profile(
                BorrowedNormResources {
                    stream: &mut stream,
                    input: &mut input,
                    weight: &mut weight,
                    output: &mut output,
                },
                1,
                576,
                1e-5,
                NormGraphProfile::HuggingFaceSmolLm2,
            )?;
            graph.replay()?;
            graph.close()?;
        }
        let mut actual = vec![0; 1152];
        output.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
        assert_eq!(actual, ones);
        input.close()?;
        weight.close()?;
        output.close()?;
        staging.close()?;
        stream.close()?;
        assert!(context.allocation_stats()?.is_zero());
        context.close()?;
        Ok(())
    }
}
