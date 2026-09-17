//! G02C: exact final-normalization binding, distinct from per-layer Norm.

use super::PhysicalWeightId;
use super::forward::LlamaRmsNormProfile;

/// Typed reasons remain distinct even though inventory projection is Unknown.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FinalNormBindingMismatch {
    Profile,
    Geometry,
    Weight,
    Epsilon,
    Terminal,
    Site,
}

/// A layer probe cannot qualify `FinalNorm` or the aggregate Norm slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NormBindingSite {
    Final,
    Input(usize),
    PostAttention(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FinalNormBinding {
    site: NormBindingSite,
    profile: LlamaRmsNormProfile,
    rows: u64,
    hidden: u64,
    epsilon_bits: u32,
    weight: PhysicalWeightId,
}

impl FinalNormBinding {
    pub(crate) fn new(
        profile: LlamaRmsNormProfile,
        rows: u64,
        hidden: u64,
        epsilon: f32,
        weight: PhysicalWeightId,
    ) -> Result<Self, FinalNormBindingMismatch> {
        if profile == LlamaRmsNormProfile::FixedContiguous37Balanced {
            return Err(FinalNormBindingMismatch::Profile);
        }
        if rows == 0
            || hidden == 0
            || rows
                .checked_mul(hidden)
                .and_then(|n| n.checked_mul(2))
                .is_none()
        {
            return Err(FinalNormBindingMismatch::Geometry);
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(FinalNormBindingMismatch::Epsilon);
        }
        if profile == LlamaRmsNormProfile::HuggingFaceSmolLm2 {
            if rows > 8192 || hidden != 576 {
                return Err(FinalNormBindingMismatch::Geometry);
            }
            if epsilon.to_bits() != 1e-5_f32.to_bits() {
                return Err(FinalNormBindingMismatch::Epsilon);
            }
        }
        Ok(Self {
            site: NormBindingSite::Final,
            profile,
            rows,
            hidden,
            epsilon_bits: epsilon.to_bits(),
            weight,
        })
    }

    pub(crate) fn at_site(mut self, site: NormBindingSite) -> Self {
        self.site = site;
        self
    }

    fn compare(self, expected: Self) -> Result<(), FinalNormBindingMismatch> {
        if self.site != expected.site {
            Err(FinalNormBindingMismatch::Site)
        } else if self.profile != expected.profile {
            Err(FinalNormBindingMismatch::Profile)
        } else if self.weight != expected.weight {
            Err(FinalNormBindingMismatch::Weight)
        } else if self.rows != expected.rows || self.hidden != expected.hidden {
            Err(FinalNormBindingMismatch::Geometry)
        } else if self.epsilon_bits != expected.epsilon_bits {
            Err(FinalNormBindingMismatch::Epsilon)
        } else {
            Ok(())
        }
    }
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::{FinalNormBinding, FinalNormBindingMismatch, NormBindingSite};
    use crate::llama::executor::error::{
        LlamaBatchExecutorError, LlamaBatchExecutorResult, cuda_error,
    };
    use crate::llama::graph::GraphOperatorCapability;
    use crate::llama::graph_decode_capture_inventory::{
        PureDecodeGraphV1CaptureCapabilityInventory, PureDecodeGraphV1CaptureOperation,
    };
    use crate::llama::{ExecutionSite, LlamaOp};
    use riley_cuda::{
        BorrowedNormGraph, BorrowedNormResources, CudaBufferSpan, CudaBufferSpanMut, CudaDType,
        CudaDeviceBuffer, CudaPinnedHostBuffer, CudaStream, RmsNormParams,
    };
    use sha2::{Digest, Sha256};

    fn rejected(reason: &'static str) -> LlamaBatchExecutorError {
        LlamaBatchExecutorError::InvalidConfiguration {
            field: "C07 final norm owner",
            reason,
        }
    }

    /// Evidence is inseparable from the retained stream/input/weight/output borrows.
    /// No standalone digest or primitive capability can construct this owner.
    pub(crate) struct C07FinalNormGraphOwner<'a> {
        graph: Option<BorrowedNormGraph<'a>>,
        poisoned: &'a mut bool,
        binding: FinalNormBinding,
        hashes: [[u8; 32]; 3],
        terminal: bool,
    }

    impl<'a> C07FinalNormGraphOwner<'a> {
        // Keep the ordered probe/readback/parity/recapture transaction together.
        #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
        pub(crate) fn prepare(
            stream: &'a mut CudaStream,
            input: &'a mut CudaDeviceBuffer,
            weight: &'a mut CudaDeviceBuffer,
            output: &'a mut CudaDeviceBuffer,
            staging: &mut CudaPinnedHostBuffer,
            binding: FinalNormBinding,
            poisoned: &'a mut bool,
        ) -> LlamaBatchExecutorResult<Self> {
            if *poisoned {
                return Err(rejected("executor is poisoned"));
            }
            let mut cuda = |source| {
                crate::llama::forward::poison_for_cuda_error(poisoned, &source);
                cuda_error(
                    match binding.site {
                        NormBindingSite::Final => ExecutionSite::global(LlamaOp::FinalNorm),
                        NormBindingSite::Input(layer) => {
                            ExecutionSite::layer(layer, LlamaOp::InputNorm)
                        }
                        NormBindingSite::PostAttention(layer) => {
                            ExecutionSite::layer(layer, LlamaOp::PostAttentionNorm)
                        }
                    },
                    source,
                )
            };
            let matrix_bytes = binding.rows * binding.hidden * 2;
            let weight_bytes = binding.hidden * 2;
            if input.byte_len() < matrix_bytes
                || output.byte_len() < matrix_bytes
                || weight.byte_len() != weight_bytes
            {
                return Err(rejected("exact final norm spans do not fit their owners"));
            }
            let host_len = |bytes| {
                usize::try_from(bytes).map_err(|_| rejected("norm span exceeds host address space"))
            };
            let used = host_len(matrix_bytes)?;
            let mut before_input = vec![0; host_len(input.byte_len())?];
            let mut before_weight = vec![0; host_len(weight_bytes)?];
            let mut before_output = vec![0; host_len(output.byte_len())?];
            input
                .download_to_slice(0, &mut before_input, staging, stream)
                .map_err(&mut cuda)?;
            weight
                .download_to_slice(0, &mut before_weight, staging, stream)
                .map_err(&mut cuda)?;
            output
                .download_to_slice(0, &mut before_output, staging, stream)
                .map_err(&mut cuda)?;
            let epsilon = f32::from_bits(binding.epsilon_bits);
            let graph_profile = match binding.profile {
                crate::llama::forward::LlamaRmsNormProfile::Canonical => {
                    riley_cuda::NormGraphProfile::Canonical
                }
                crate::llama::forward::LlamaRmsNormProfile::HuggingFaceSmolLm2 => {
                    riley_cuda::NormGraphProfile::HuggingFaceSmolLm2
                }
                crate::llama::forward::LlamaRmsNormProfile::FixedContiguous37Balanced => {
                    return Err(rejected("unsupported norm graph profile"));
                }
                #[cfg(feature = "cuda-cublas-gemm-probe")]
                crate::llama::forward::LlamaRmsNormProfile::HuggingFaceCudaQwenP2051Probe => {
                    return Err(rejected("P2051 RMSNorm probe is not graph-qualified"));
                }
                #[cfg(feature = "cuda-cublas-gemm-probe")]
                crate::llama::forward::LlamaRmsNormProfile::HuggingFaceCudaQwenP2048CacheOnProbe => {
                    return Err(rejected(
                        "P2048 cache-on RMSNorm probe is not graph-qualified",
                    ));
                }
            };
            let mut probe = BorrowedNormGraph::prepare_with_profile(
                BorrowedNormResources {
                    stream,
                    input,
                    weight,
                    output,
                },
                binding.rows,
                binding.hidden,
                epsilon,
                graph_profile,
            )
            .map_err(&mut cuda)?;
            probe.replay().map_err(&mut cuda)?;
            probe.close().map_err(&mut cuda)?;
            let mut graph_bytes = vec![0; before_output.len()];
            output
                .download_to_slice(0, &mut graph_bytes, staging, stream)
                .map_err(&mut cuda)?;
            crate::llama::forward::execute_profile_rms_norm(
                binding.profile,
                &mut RmsNormParams {
                    input: CudaBufferSpan::new(input, CudaDType::BF16, 0, matrix_bytes)
                        .map_err(&mut cuda)?,
                    weight: CudaBufferSpan::new(weight, CudaDType::BF16, 0, weight_bytes)
                        .map_err(&mut cuda)?,
                    output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, matrix_bytes)
                        .map_err(&mut cuda)?,
                    row_count: binding.rows,
                    hidden_size: binding.hidden,
                    epsilon,
                },
                stream,
            )
            .map_err(&mut cuda)?;
            let mut eager_bytes = vec![0; graph_bytes.len()];
            output
                .download_to_slice(0, &mut eager_bytes, staging, stream)
                .map_err(&mut cuda)?;
            let mut after_input = vec![0; before_input.len()];
            let mut after_weight = vec![0; before_weight.len()];
            input
                .download_to_slice(0, &mut after_input, staging, stream)
                .map_err(&mut cuda)?;
            weight
                .download_to_slice(0, &mut after_weight, staging, stream)
                .map_err(&mut cuda)?;
            if eager_bytes != graph_bytes
                || before_input != after_input
                || before_weight != after_weight
                || graph_bytes[used..] != before_output[used..]
            {
                // The probe has altered executor workspace: a failed parity check
                // cannot be treated like harmless preflight rejection.
                *poisoned = true;
                return Err(rejected(
                    "final norm graph parity or owner preservation failed",
                ));
            }
            if eager_bytes[..used]
                .chunks_exact(2)
                .any(|word| u16::from_le_bytes([word[0], word[1]]) & 0x7f80 == 0x7f80)
            {
                *poisoned = true;
                return Err(rejected("final norm output is not finite"));
            }
            let hashes = [
                Sha256::digest(&before_input[..used]).into(),
                Sha256::digest(&before_weight).into(),
                Sha256::digest(&eager_bytes[..used]).into(),
            ];
            let graph = BorrowedNormGraph::prepare_with_profile(
                BorrowedNormResources {
                    stream,
                    input,
                    weight,
                    output,
                },
                binding.rows,
                binding.hidden,
                epsilon,
                graph_profile,
            )
            .map_err(&mut cuda)?;
            Ok(Self {
                graph: Some(graph),
                poisoned,
                binding,
                hashes,
                terminal: false,
            })
        }

        pub(crate) fn binding_decision(
            &self,
            expected: FinalNormBinding,
        ) -> Result<(), FinalNormBindingMismatch> {
            if self.terminal {
                Err(FinalNormBindingMismatch::Terminal)
            } else {
                self.binding.compare(expected)
            }
        }

        pub(crate) fn bind_inventory(
            &self,
            inventory: PureDecodeGraphV1CaptureCapabilityInventory,
            expected: FinalNormBinding,
        ) -> PureDecodeGraphV1CaptureCapabilityInventory {
            let slot = PureDecodeGraphV1CaptureOperation::FinalNorm;
            let capability =
                if inventory.capability_for(slot) == GraphOperatorCapability::Unsupported {
                    GraphOperatorCapability::Unsupported
                } else if self.binding.site == NormBindingSite::Final
                    && self.binding_decision(expected).is_ok()
                {
                    GraphOperatorCapability::Supported
                } else {
                    GraphOperatorCapability::Unknown
                };
            inventory.with_capability(slot, capability)
        }

        pub(crate) fn replay(&mut self) -> riley_cuda::CudaResult<()> {
            let result = self.graph.as_mut().expect("live graph").replay();
            if result.is_err() {
                self.terminal = true;
                *self.poisoned = true;
            }
            result
        }

        pub(crate) fn parity_hashes(&self) -> [[u8; 32]; 3] {
            self.hashes
        }

        pub(crate) fn close(mut self) -> riley_cuda::CudaResult<()> {
            let result = self.graph.take().expect("live graph").close();
            if result.is_err() {
                *self.poisoned = true;
            }
            result
        }
    }

    impl Drop for C07FinalNormGraphOwner<'_> {
        fn drop(&mut self) {
            if let Some(graph) = self.graph.take() {
                if graph.close().is_err() {
                    *self.poisoned = true;
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) use gpu::C07FinalNormGraphOwner;

#[cfg(test)]
mod tests {
    use super::{
        FinalNormBinding, FinalNormBindingMismatch, LlamaRmsNormProfile, PhysicalWeightId,
    };

    #[test]
    fn final_norm_binding_rejects_profile_geometry_weight_and_epsilon_substitution() {
        let weight = PhysicalWeightId::new(1, 3);
        let make = |profile, rows, hidden, epsilon, weight| {
            FinalNormBinding::new(profile, rows, hidden, epsilon, weight)
        };
        let canonical = LlamaRmsNormProfile::Canonical;
        let valid = make(canonical, 1, 64, 1e-5, weight).unwrap();
        for profile in [LlamaRmsNormProfile::FixedContiguous37Balanced] {
            assert_eq!(
                make(profile, 1, 64, 1e-5, weight),
                Err(FinalNormBindingMismatch::Profile)
            );
        }
        for (rows, hidden) in [(0, 64), (1, 0), (u64::MAX, 64)] {
            assert_eq!(
                make(canonical, rows, hidden, 1e-5, weight),
                Err(FinalNormBindingMismatch::Geometry)
            );
        }
        for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                make(canonical, 1, 64, epsilon, weight),
                Err(FinalNormBindingMismatch::Epsilon)
            );
        }
        for foreign in [PhysicalWeightId::new(2, 3), PhysicalWeightId::new(1, 4)] {
            assert_eq!(
                valid.compare(make(canonical, 1, 64, 1e-5, foreign).unwrap()),
                Err(FinalNormBindingMismatch::Weight)
            );
        }
        assert_eq!(
            valid.compare(make(canonical, 2, 64, 1e-5, weight).unwrap()),
            Err(FinalNormBindingMismatch::Geometry)
        );
        assert_eq!(
            valid.compare(make(canonical, 1, 64, 1e-6, weight).unwrap()),
            Err(FinalNormBindingMismatch::Epsilon)
        );
        assert_eq!(valid.compare(valid), Ok(()));
        for site in [
            super::NormBindingSite::Input(0),
            super::NormBindingSite::PostAttention(0),
            super::NormBindingSite::Input(1),
        ] {
            let layer = valid.at_site(site);
            assert_eq!(valid.compare(layer), Err(FinalNormBindingMismatch::Site));
            assert_eq!(layer.compare(layer), Ok(()));
        }
        assert_eq!(
            valid
                .at_site(super::NormBindingSite::Input(0))
                .compare(valid.at_site(super::NormBindingSite::PostAttention(0))),
            Err(FinalNormBindingMismatch::Site)
        );
        let hf = make(
            LlamaRmsNormProfile::HuggingFaceSmolLm2,
            1,
            576,
            1e-5,
            weight,
        )
        .unwrap();
        assert_eq!(
            hf.compare(make(canonical, 1, 576, 1e-5, weight).unwrap()),
            Err(FinalNormBindingMismatch::Profile)
        );
        assert_eq!(
            make(
                LlamaRmsNormProfile::HuggingFaceSmolLm2,
                8193,
                576,
                1e-5,
                weight
            ),
            Err(FinalNormBindingMismatch::Geometry)
        );
        assert_eq!(
            make(
                LlamaRmsNormProfile::HuggingFaceSmolLm2,
                1,
                576,
                1e-6,
                weight
            ),
            Err(FinalNormBindingMismatch::Epsilon)
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA; G02C canonical final norm owner parity and lifecycle"]
    fn c07_final_norm_owner_gpu_parity_binding_and_recovery()
    -> Result<(), Box<dyn std::error::Error>> {
        use super::C07FinalNormGraphOwner;
        use crate::llama::graph::GraphOperatorCapability as Capability;
        use crate::llama::graph_decode_capture_inventory::{
            PureDecodeGraphV1CaptureCapabilityInventory as Inventory,
            PureDecodeGraphV1CaptureOperation as Operation,
        };
        use riley_cuda::CudaRuntime;
        use sha2::{Digest, Sha256};
        let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
        let mut stream = context.create_stream()?;
        let mut staging = context.allocate_pinned_host_buffer(16384)?;
        for (profile, rows, hidden) in [
            (LlamaRmsNormProfile::Canonical, 1_u64, 64_u64),
            (LlamaRmsNormProfile::Canonical, 3, 257),
            (LlamaRmsNormProfile::Canonical, 1, 576),
            (LlamaRmsNormProfile::HuggingFaceSmolLm2, 1, 576),
            (LlamaRmsNormProfile::HuggingFaceSmolLm2, 3, 576),
            (LlamaRmsNormProfile::HuggingFaceSmolLm2, 8, 576),
            (LlamaRmsNormProfile::HuggingFaceSmolLm2, 16, 576),
            (LlamaRmsNormProfile::HuggingFaceSmolLm2, 17, 576),
        ] {
            let len = (rows * hidden * 2) as usize;
            let input_bytes: Vec<u8> = (0..len / 2 + 8)
                .flat_map(|i| (0x3e00_u16 + (i % 250) as u16).to_le_bytes())
                .collect();
            let weights: Vec<u8> = (0..hidden)
                .flat_map(|i| (0x3f00_u16 + (i % 127) as u16).to_le_bytes())
                .collect();
            let mut input = context.allocate_device_buffer(input_bytes.len() as u64)?;
            let mut output = context.allocate_device_buffer(input_bytes.len() as u64)?;
            let mut weight = context.allocate_device_buffer(weights.len() as u64)?;
            input.upload_from_slice(0, &input_bytes, &mut staging, &mut stream)?;
            weight.upload_from_slice(0, &weights, &mut staging, &mut stream)?;
            output.upload_from_slice(
                0,
                &vec![0xa5; input_bytes.len()],
                &mut staging,
                &mut stream,
            )?;
            let mut poisoned = false;
            let binding =
                FinalNormBinding::new(profile, rows, hidden, 1e-5, PhysicalWeightId::new(1, 3))
                    .unwrap();
            // Cross-context staging rejects before capture, without poisoning.
            let foreign = CudaRuntime::initialize()?.device(0)?.create_context()?;
            let mut foreign_staging = foreign.allocate_pinned_host_buffer(16384)?;
            assert!(
                C07FinalNormGraphOwner::prepare(
                    &mut stream,
                    &mut input,
                    &mut weight,
                    &mut output,
                    &mut foreign_staging,
                    binding,
                    &mut poisoned
                )
                .is_err()
            );
            assert!(!poisoned);
            foreign_staging.close()?;
            foreign.close()?;
            let stats = context.allocation_stats()?;
            for (site, close_explicitly) in [
                (super::NormBindingSite::Final, true),
                (super::NormBindingSite::Final, false),
                (super::NormBindingSite::Input(0), true),
                (super::NormBindingSite::PostAttention(0), false),
            ] {
                let binding = binding.at_site(site);
                let mut owner = C07FinalNormGraphOwner::prepare(
                    &mut stream,
                    &mut input,
                    &mut weight,
                    &mut output,
                    &mut staging,
                    binding,
                    &mut poisoned,
                )?;
                let initial =
                    Inventory::default().with_capability(Operation::Norm, Capability::Supported);
                let matched = owner.bind_inventory(initial, binding);
                assert_eq!(
                    matched.capability_for(Operation::FinalNorm),
                    if site == super::NormBindingSite::Final {
                        Capability::Supported
                    } else {
                        Capability::Unknown
                    }
                );
                assert_eq!(
                    matched.capability_for(Operation::Norm),
                    Capability::Supported
                );
                assert_eq!(matched.operator_capability(), Capability::Unknown);
                let mut wrong = binding;
                wrong.weight = PhysicalWeightId::new(2, 3);
                assert_eq!(
                    owner.binding_decision(wrong),
                    Err(FinalNormBindingMismatch::Weight)
                );
                assert_eq!(
                    owner
                        .bind_inventory(matched, wrong)
                        .capability_for(Operation::FinalNorm),
                    Capability::Unknown
                );
                assert_eq!(
                    owner
                        .bind_inventory(
                            initial.with_capability(Operation::FinalNorm, Capability::Unsupported),
                            binding
                        )
                        .capability_for(Operation::FinalNorm),
                    Capability::Unsupported
                );
                for _ in 0..64 {
                    owner.replay()?;
                }
                let hashes = owner.parity_hashes();
                assert_eq!(
                    hashes[0],
                    <[u8; 32]>::from(Sha256::digest(&input_bytes[..len]))
                );
                assert_eq!(hashes[1], <[u8; 32]>::from(Sha256::digest(&weights)));
                eprintln!(
                    "G02C profile={profile:?} rows={rows} hidden={hidden} input/weight/output SHA256={hashes:02x?}"
                );
                if close_explicitly {
                    owner.close()?;
                } else {
                    drop(owner);
                }
                assert!(!poisoned);
                assert_eq!(context.allocation_stats()?, stats);
                let mut observed = vec![0; input_bytes.len()];
                output.download_to_slice(0, &mut observed, &mut staging, &mut stream)?;
                assert_eq!(
                    hashes[2],
                    <[u8; 32]>::from(Sha256::digest(&observed[..len]))
                );
                assert_eq!(&observed[len..], &[0xa5; 16]);
                input.download_to_slice(0, &mut observed, &mut staging, &mut stream)?;
                assert_eq!(observed, input_bytes);
            }
            // Byte-equal NaNs are not usable evidence; workspace mutation poisons.
            input.upload_from_slice(
                0,
                &vec![0xff; input_bytes.len()],
                &mut staging,
                &mut stream,
            )?;
            assert!(
                C07FinalNormGraphOwner::prepare(
                    &mut stream,
                    &mut input,
                    &mut weight,
                    &mut output,
                    &mut staging,
                    binding,
                    &mut poisoned
                )
                .is_err()
            );
            assert!(poisoned);
            assert!(
                C07FinalNormGraphOwner::prepare(
                    &mut stream,
                    &mut input,
                    &mut weight,
                    &mut output,
                    &mut staging,
                    binding,
                    &mut poisoned
                )
                .is_err()
            );
            input.close()?;
            weight.close()?;
            output.close()?;
        }
        staging.close()?;
        stream.close()?;
        assert!(context.allocation_stats()?.is_zero());
        context.close()?;
        Ok(())
    }
}
