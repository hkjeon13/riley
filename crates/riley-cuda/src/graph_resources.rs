//! Resource ownership for staged transfer, `SwiGLU` and MLP diagnostic graphs.
use crate::{CudaDeviceBuffer, CudaPinnedHostBuffer, CudaPreparedGemm, CudaResult, CudaStream};

/// Versioned arithmetic for the retained full decode graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum DecodeNumericalProfile {
    /// Existing canonical graph arithmetic.
    Canonical = 0,
    /// Existing HF-compatible SmolLM2 graph arithmetic.
    HuggingFaceSmolLm2 = 1,
    /// SM89/CUDA13 SmolLM2-135M, P128 and at most 32 generated tokens.
    /// Preserves vLLM 0.27.1 compiled/FlashAttention2 arithmetic for this bucket.
    VllmSmolP128V1 = 2,
}

impl DecodeNumericalProfile {
    /// Stable arithmetic label for execution evidence, not a quality assertion.
    #[must_use]
    pub const fn evidence_id(self) -> &'static str {
        match self {
            Self::Canonical => "canonical-v1",
            Self::HuggingFaceSmolLm2 => "hf-smollm2-v1",
            Self::VllmSmolP128V1 => "vllm-smol-p128-v1",
        }
    }
}

/// Owns the Rust parents of a recorded graph. Native handles point to independent
/// CUDA allocations, never into the Rust container, so moving the container is safe.
/// No parent access is exposed until native graph destruction succeeds.
#[cfg(feature = "cuda")]
pub struct OwnedGraphResourceReservation<P> {
    native: crate::ffi::GraphResourcesHandle,
    parents: P,
}

#[cfg(feature = "cuda")]
impl<P> OwnedGraphResourceReservation<P> {
    /// Records using an exclusive, scoped borrow and retains the entire owner.
    /// # Errors
    /// Returns the recording error without publishing a partially prepared owner.
    pub fn prepare<E>(
        mut parents: P,
        record: impl for<'a> FnOnce(&'a mut P) -> Result<BorrowedGraphResourceReservation<'a>, E>,
    ) -> Result<Self, E> {
        let BorrowedGraphResourceReservation {
            native,
            parents: borrowed,
            strided_plans,
        } = record(&mut parents)?;
        drop(strided_plans);
        drop(borrowed);
        Ok(Self { native, parents })
    }

    /// Submit into the next free prepared staging slot on the retained stream.
    /// # Errors
    /// Rejects full slots, invalid input or an unprepared/failed owner.
    pub fn submit_buffered_transfer(&mut self, input: &[u8]) -> CudaResult<u64> { self.native.submit_buffered_transfer(input) }
    /// Submit V7 compact decode using the latest unconsumed predecessor output.
    /// # Errors
    /// Rejects wrong extents, stale tickets and unsupported prepared profiles.
    pub fn submit_future_transfer(&mut self, input: &[u8], references: &[u8], predecessor: u64) -> CudaResult<u64> { self.native.submit_future_transfer(input, references, predecessor) }

    /// Poll or wait for this owner's exact ticket; does not consume output.
    /// # Errors
    /// Rejects stale tickets and propagates CUDA completion errors.
    pub fn query_buffered_transfer(&mut self, ticket: u64, wait: bool) -> CudaResult<bool> { self.native.query_buffered_transfer(ticket, wait) }
    /// Copy completed output and release only its staging slot for reuse.
    /// # Errors
    /// Rejects stale/pending tickets or an incorrect destination size.
    pub fn read_buffered_transfer(&mut self, ticket: u64, output: &mut [u8]) -> CudaResult<()> { self.native.read_buffered_transfer(ticket, output) }

    /// Stages fresh input and waits for completion; errors never expose output.
    /// # Errors
    /// Returns validation or CUDA errors from the retained native graph.
    pub fn replay_transfer(&mut self, input: &[u8]) -> CudaResult<()> {
        self.native.replay_transfer(input)
    }

    /// Copies input into retained staging and submits the graph without waiting.
    /// # Errors
    /// Rejects pending work or invalid input and propagates CUDA failures.
    pub fn submit_transfer(&mut self, input: &[u8]) -> CudaResult<()> {
        self.native.submit_transfer(input)
    }
    /// Checks completion without blocking. Output is readable only after true.
    /// # Errors
    /// Rejects missing work or propagates CUDA completion errors.
    pub fn query_transfer(&mut self) -> CudaResult<bool> {
        self.native.query_transfer()
    }
    /// Waits for pending work. Close/drop also drain before releasing parents.
    /// # Errors
    /// On unknown completion native retains the parents instead of releasing them.
    pub fn wait_transfer(&mut self) -> CudaResult<()> {
        self.native.wait_transfer()
    }

    /// Reads only a successfully completed replay.
    /// # Errors
    /// Returns an error for stale, failed, or incorrectly sized output.
    pub fn read_transfer(&mut self, output: &mut [u8]) -> CudaResult<()> {
        self.native.read_transfer(output)
    }

    /// Replays one cold catalog entry while retaining every parent.
    /// # Errors
    /// Rejects an absent entry, invalid packet or CUDA execution failure.
    pub fn replay_catalog(&mut self, index: u32, input: &[u8]) -> CudaResult<()> {
        self.native.replay_catalog(index, input)
    }

    /// Reads only the most recently completed catalog entry.
    /// # Errors
    /// Rejects stale selection or an incorrectly sized output buffer.
    pub fn read_catalog(&mut self, index: u32, output: &mut [u8]) -> CudaResult<()> {
        self.native.read_catalog(index, output)
    }

    /// Returns parents only after the native reservation has been released.
    /// # Errors
    /// Unknown completion retains native parent protection and returns an error.
    pub fn close(mut self) -> CudaResult<P> {
        self.native.close()?;
        Ok(self.parents)
    }
}

/// Exclusive actual parents retained until reservation close or Drop.
/// Resource registration does not validate operator aliases or a decode DAG.
pub struct BorrowedGraphResourceParents<'a> {
    /// Exact stream reserved for future recording.
    pub stream: &'a mut CudaStream,
    /// Physical device allocations, including immutable weights and workspace.
    pub devices: Vec<&'a mut CudaDeviceBuffer>,
    /// Actual pinned metadata and result parents.
    pub pinned: Vec<&'a mut CudaPinnedHostBuffer>,
    /// Actual selected GEMM plans, preserving their algorithms and policies.
    pub plans: Vec<&'a mut CudaPreparedGemm>,
}

/// A thread-confined native ledger, optionally holding a staged diagnostic graph.
pub struct BorrowedGraphResourceReservation<'a> {
    // Native releases first, while every borrowed parent is still alive.
    #[cfg(feature = "cuda")]
    native: crate::ffi::GraphResourcesHandle,
    #[allow(dead_code)]
    parents: BorrowedGraphResourceParents<'a>,
    // Kept separate so existing single-row recording APIs cannot select these plans.
    #[allow(dead_code)]
    strided_plans: Vec<&'a mut crate::CudaPreparedStridedGemm>,
}
impl<'a> BorrowedGraphResourceReservation<'a> {
    /// Reserves all parents atomically with rollback on a busy resource.
    /// # Errors
    /// Rejects foreign contexts, busy resources, unsupported plans or capacity.
    #[cfg_attr(not(feature = "cuda"), allow(clippy::needless_pass_by_value))] // CUDA owns parents through close.
    pub fn reserve(parents: BorrowedGraphResourceParents<'a>) -> CudaResult<Self> {
        Self::reserve_with_strided(parents, Vec::new())
    }

    /// Retains additional strided plans under the same aggregate lease owner.
    /// Existing single-row recording methods only index `parents.plans`.
    /// This admission does not record a multi-row decode graph.
    /// # Errors
    /// Rejects poisoned, foreign, busy or unsupported plans and rolls back acquisitions.
    #[cfg_attr(not(feature = "cuda"), allow(clippy::needless_pass_by_value))]
    pub fn reserve_with_strided(
        parents: BorrowedGraphResourceParents<'a>,
        strided_plans: Vec<&'a mut crate::CudaPreparedStridedGemm>,
    ) -> CudaResult<Self> {
        #[cfg(feature = "cuda")]
        {
            let devices: Vec<_> = parents.devices.iter().map(|b| b.native_handle()).collect();
            let pinned: Vec<_> = parents.pinned.iter().map(|b| b.native_handle()).collect();
            let mut plans: Vec<_> = parents
                .plans
                .iter()
                .map(|p| p.graph_resource_handle())
                .collect::<CudaResult<_>>()?;
            for plan in &strided_plans {
                plans.push(plan.graph_resource_handle()?);
            }
            let native = crate::ffi::GraphResourcesHandle::reserve(
                &parents.stream.native,
                &devices,
                &pinned,
                &plans,
            )?;
            Ok(Self {
                native,
                parents,
                strided_plans,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (parents, strided_plans);
            Err(crate::CudaError::unavailable(
                "reserve aggregate graph resources",
            ))
        }
    }
    /// Releases the ledger before ending the exclusive parent borrows.
    /// # Errors
    /// Returns an error if native release cannot be established.
    pub fn close(mut self) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.native.close()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = &mut self;
            Ok(())
        }
    }
}

impl BorrowedGraphResourceReservation<'_> {
    /// Records H2D → device copy → D2H using indices into retained parents.
    /// This transport probe does not admit model operators or full decode replay.
    /// # Errors
    /// Rejects invalid indices, aliases, sizes, membership or native graph errors.
    pub fn record_transfer(
        &mut self,
        input: usize,
        first: usize,
        second: usize,
        output: usize,
    ) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            let bad = || {
                crate::CudaError::invalid_argument(
                    "record aggregate transfer",
                    "parent index out of range",
                )
            };
            let input = self.parents.pinned.get(input).ok_or_else(bad)?;
            let output = self.parents.pinned.get(output).ok_or_else(bad)?;
            let first = self.parents.devices.get(first).ok_or_else(bad)?;
            let second = self.parents.devices.get(second).ok_or_else(bad)?;
            self.native.record_transfer(
                input.native_handle(),
                first.native_handle(),
                second.native_handle(),
                output.native_handle(),
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (input, first, second, output);
            Err(crate::CudaError::unavailable("record aggregate transfer"))
        }
    }
    /// Stages fresh input and synchronously executes the retained staged graph.
    /// # Errors
    /// Rejects stale state or wrong size; CUDA failures retain native protection.
    pub fn replay_transfer(&mut self, source: &[u8]) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.native.replay_transfer(source)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = source;
            Err(crate::CudaError::unavailable("replay aggregate transfer"))
        }
    }
    /// Copies output only after a successful, completed replay.
    /// # Errors
    /// Rejects missing completion, failed replay or wrong output size.
    pub fn read_transfer(&mut self, output: &mut [u8]) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.native.read_transfer(output)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = output;
            Err(crate::CudaError::unavailable("read aggregate transfer"))
        }
    }
}

impl BorrowedGraphResourceReservation<'_> {
    /// Records two eager BF16 kernels with staged input/output on actual parents.
    /// Device indices are [gate, up, activated, product]. The pinned parent must
    /// fit all four arrays. This is a diagnostic subgraph, not full model replay.
    /// # Errors
    /// Rejects invalid parent indices, aliases, sizes, leases or graph state.
    pub fn record_swiglu(&mut self, devices: [usize; 4], staging: usize) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            let bad =
                || crate::CudaError::invalid_argument("record SwiGLU", "parent index out of range");
            let gate = self.parents.devices.get(devices[0]).ok_or_else(bad)?;
            let up = self.parents.devices.get(devices[1]).ok_or_else(bad)?;
            let activated = self.parents.devices.get(devices[2]).ok_or_else(bad)?;
            let product = self.parents.devices.get(devices[3]).ok_or_else(bad)?;
            let staging = self.parents.pinned.get(staging).ok_or_else(bad)?;
            self.native.record_swiglu(
                gate.native_handle(),
                up.native_handle(),
                activated.native_handle(),
                product.native_handle(),
                staging.native_handle(),
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (devices, staging);
            Err(crate::CudaError::unavailable("record SwiGLU"))
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod swiglu_gpu_tests {
    use super::*;
    use crate::{
        CudaBufferSpan, CudaBufferSpanMut, CudaDType, CudaRuntime, GatedMultiplyParams, SiluParams,
        gated_multiply, silu,
    };

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn swiglu_chain_matches_eager_across_sizes_and_fresh_inputs() -> CudaResult<()> {
        let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
        let mut stream = context.create_stream()?;
        let empty = context.allocation_stats()?;
        for elements in [1_u64, 257, 1536] {
            let bytes = elements * 2;
            let mut gate = context.allocate_device_buffer(bytes)?;
            let mut up = context.allocate_device_buffer(bytes)?;
            let mut activated = context.allocate_device_buffer(bytes)?;
            let mut product = context.allocate_device_buffer(bytes)?;
            let mut staging = context.allocate_pinned_host_buffer(bytes * 4 + 64)?;
            let mut eager_gate = context.allocate_device_buffer(bytes)?;
            let mut eager_up = context.allocate_device_buffer(bytes)?;
            let mut eager_activated = context.allocate_device_buffer(bytes)?;
            let mut eager_product = context.allocate_device_buffer(bytes)?;
            let mut eager_staging = context.allocate_pinned_host_buffer(bytes * 4)?;
            let mut eager_stream = context.create_stream()?;
            let expected_stats = context.allocation_stats()?;
            staging.write(0, &vec![0xa5; (bytes * 4 + 64) as usize])?;
            for cycle in 0..2 {
                let mut graph =
                    BorrowedGraphResourceReservation::reserve(BorrowedGraphResourceParents {
                        stream: &mut stream,
                        devices: vec![&mut gate, &mut up, &mut activated, &mut product],
                        pinned: vec![&mut staging],
                        plans: vec![],
                    })?;
                assert!(graph.record_swiglu([0, 1, 2, 99], 0).is_err());
                assert!(graph.record_swiglu([0, 0, 2, 3], 0).is_err());
                graph.record_swiglu([0, 1, 2, 3], 0)?;
                let mut output = vec![0; (bytes * 2) as usize];
                assert!(graph.read_transfer(&mut output).is_err());
                for step in 0..32_u32 {
                    // Finite, signed BF16 values; vary every input on one retained exec.
                    let payload: Vec<u8> = (0..elements * 2)
                        .flat_map(|i| {
                            let value = ((i % 97) as f32 - 48.0 + step as f32) / 8.0;
                            ((value.to_bits() >> 16) as u16).to_le_bytes()
                        })
                        .collect();
                    eager_gate.upload_from_slice(
                        0,
                        &payload[..bytes as usize],
                        &mut eager_staging,
                        &mut eager_stream,
                    )?;
                    eager_up.upload_from_slice(
                        0,
                        &payload[bytes as usize..],
                        &mut eager_staging,
                        &mut eager_stream,
                    )?;
                    silu(
                        &mut SiluParams {
                            input: CudaBufferSpan::new(&eager_gate, CudaDType::BF16, 0, bytes)?,
                            output: CudaBufferSpanMut::new(
                                &mut eager_activated,
                                CudaDType::BF16,
                                0,
                                bytes,
                            )?,
                            element_count: elements,
                        },
                        &mut eager_stream,
                    )?;
                    gated_multiply(
                        &mut GatedMultiplyParams {
                            activated_gate: CudaBufferSpan::new(
                                &eager_activated,
                                CudaDType::BF16,
                                0,
                                bytes,
                            )?,
                            up: CudaBufferSpan::new(&eager_up, CudaDType::BF16, 0, bytes)?,
                            output: CudaBufferSpanMut::new(
                                &mut eager_product,
                                CudaDType::BF16,
                                0,
                                bytes,
                            )?,
                            element_count: elements,
                        },
                        &mut eager_stream,
                    )?;
                    let mut expected = vec![0; output.len()];
                    eager_activated.download_to_slice(
                        0,
                        &mut expected[..bytes as usize],
                        &mut eager_staging,
                        &mut eager_stream,
                    )?;
                    eager_product.download_to_slice(
                        0,
                        &mut expected[bytes as usize..],
                        &mut eager_staging,
                        &mut eager_stream,
                    )?;
                    graph.replay_transfer(&payload)?;
                    graph.read_transfer(&mut output)?;
                    assert_eq!(output, expected, "elements={elements} step={step}");
                    assert!(
                        graph
                            .replay_transfer(&payload[..payload.len() - 1])
                            .is_err()
                    );
                    assert!(graph.read_transfer(&mut output).is_err());
                }
                if cycle == 0 {
                    graph.close()?;
                } else {
                    drop(graph);
                }
                assert_eq!(&staging.to_vec()?[(bytes * 4) as usize..], &[0xa5; 64]);
                assert_eq!(context.allocation_stats()?, expected_stats);
            }
            eager_stream.close()?;
            eager_staging.close()?;
            eager_product.close()?;
            eager_activated.close()?;
            eager_up.close()?;
            eager_gate.close()?;
            staging.close()?;
            product.close()?;
            activated.close()?;
            up.close()?;
            gate.close()?;
            assert_eq!(context.allocation_stats()?, empty);
        }
        stream.close()?;
        assert!(context.allocation_stats()?.is_zero());
        context.close()?;
        Ok(())
    }
}

impl BorrowedGraphResourceReservation<'_> {
    /// Records staged M=1 gate/up GEMMs, `SiLU`, multiply, down GEMM and residual.
    /// Device slots: input, residual, gate, up, activated, product, down, output,
    /// gate weight, up weight, down weight. Plans: intermediate and down.
    /// # Errors
    /// Rejects absent/aliased parents, mismatched geometry, plan policy or capture errors.
    pub fn record_mlp(
        &mut self,
        devices: [usize; 11],
        workspace: Option<usize>,
        plans: [usize; 2],
        staging: usize,
    ) -> CudaResult<()> {
        self.record_mlp_with_norm(devices, workspace, plans, staging, None, None)
    }
    /// Records post-attention `RMSNorm` followed by the MLP chain.
    /// `norm` contains the norm weight index, epsilon and HF profile selection.
    /// # Errors
    /// Rejects invalid parent indices, norm geometry/profile or capture failure.
    pub fn record_norm_mlp(
        &mut self,
        devices: [usize; 11],
        workspace: Option<usize>,
        plans: [usize; 2],
        staging: usize,
        norm: (usize, f32, bool),
    ) -> CudaResult<()> {
        self.record_mlp_with_norm(devices, workspace, plans, staging, Some(norm), None)
    }
    /// Records attention output projection and residual before norm and MLP.
    /// `projection` contains the output weight index and selected plan index.
    /// # Errors
    /// Rejects invalid parents, unsupported geometry or capture failure.
    pub fn record_layer_tail(
        &mut self,
        devices: [usize; 11],
        workspace: Option<usize>,
        plans: [usize; 2],
        staging: usize,
        norm: (usize, f32, bool),
        projection: (usize, usize),
    ) -> CudaResult<()> {
        self.record_mlp_with_norm(
            devices,
            workspace,
            plans,
            staging,
            Some(norm),
            Some(projection),
        )
    }
    #[cfg_attr(not(feature = "cuda"), allow(clippy::unused_self))] // CUDA uses the retained owner.
    fn record_mlp_with_norm(
        &mut self,
        devices: [usize; 11],
        workspace: Option<usize>,
        plans: [usize; 2],
        staging: usize,
        norm: Option<(usize, f32, bool)>,
        projection: Option<(usize, usize)>,
    ) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            let bad = || {
                crate::CudaError::invalid_argument("record MLP graph", "parent index out of range")
            };
            let get = |index: usize| {
                self.parents
                    .devices
                    .get(index)
                    .map(|b| b.native_handle())
                    .ok_or_else(bad)
            };
            let d = [
                get(devices[0])?,
                get(devices[1])?,
                get(devices[2])?,
                get(devices[3])?,
                get(devices[4])?,
                get(devices[5])?,
                get(devices[6])?,
                get(devices[7])?,
                get(devices[8])?,
                get(devices[9])?,
                get(devices[10])?,
            ];
            let workspace = workspace.map(get).transpose()?;
            let intermediate = self
                .parents
                .plans
                .get(plans[0])
                .ok_or_else(bad)?
                .graph_resource_handle()?;
            let down = self
                .parents
                .plans
                .get(plans[1])
                .ok_or_else(bad)?
                .graph_resource_handle()?;
            let staging = self.parents.pinned.get(staging).ok_or_else(bad)?;
            let norm = norm
                .map(|(index, epsilon, hf)| {
                    get(index).map(|weight| (weight, epsilon, u32::from(hf)))
                })
                .transpose()?;
            let projection = projection
                .map(|(weight, plan)| -> CudaResult<_> {
                    Ok((
                        get(weight)?,
                        self.parents
                            .plans
                            .get(plan)
                            .ok_or_else(bad)?
                            .graph_resource_handle()?,
                    ))
                })
                .transpose()?;
            self.native.record_mlp_with_tail(
                d,
                workspace,
                intermediate,
                down,
                staging.native_handle(),
                norm,
                projection,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (devices, workspace, plans, staging, norm, projection);
            Err(crate::CudaError::unavailable("record MLP graph"))
        }
    }
}

impl BorrowedGraphResourceReservation<'_> {
    /// Records input norm then Q/K/V with selected M=1 plans.
    /// Slots: input, norm, Q, K, V, norm weight, Q/K/V weights.
    /// # Errors
    /// Rejects missing parents, aliases, invalid profile/geometry or capture failure.
    #[cfg_attr(not(feature = "cuda"), allow(clippy::unused_self))]
    pub fn record_norm_qkv(
        &mut self,
        devices: [usize; 9],
        workspace: Option<usize>,
        plans: [usize; 2],
        staging: usize,
        norm: (f32, bool),
    ) -> CudaResult<()> {
        self.record_qkv_impl(devices, workspace, plans, staging, norm, None, None)
    }
    /// Records norm/QKV and indexed D64 `RoPE` with fresh replay positions.
    /// Extra parents are rotated Q/K, cos, sin and the packed positions parent.
    /// # Errors
    /// Rejects invalid parent indices, geometry, profile or capture failure.
    pub fn record_qkv_rope(
        &mut self,
        devices: [usize; 9],
        workspace: Option<usize>,
        plans: [usize; 2],
        staging: usize,
        norm: (f32, bool),
        rope: ([usize; 5], u64),
    ) -> CudaResult<()> {
        self.record_qkv_impl(devices, workspace, plans, staging, norm, Some(rope), None)
    }
    #[cfg_attr(not(feature = "cuda"), allow(clippy::unused_self))]
    // Geometry: layers, layer, physical block count, physical block, logical block, valid prefix.
    #[allow(clippy::too_many_arguments)] // Explicit retained parent groups and fixed cache binding.
    fn record_qkv_impl(
        &mut self,
        devices: [usize; 9],
        workspace: Option<usize>,
        plans: [usize; 2],
        staging: usize,
        norm: (f32, bool),
        rope: Option<([usize; 5], u64)>,
        caches: Option<([usize; 2], [u64; 6])>,
    ) -> CudaResult<()> {
        self.record_attention_impl(devices, workspace, plans, staging, norm, rope, caches, None)
    }
    #[cfg_attr(not(feature = "cuda"), allow(clippy::unused_self))]
    #[allow(clippy::too_many_arguments)]
    fn record_attention_impl(
        &mut self,
        devices: [usize; 9],
        workspace: Option<usize>,
        plans: [usize; 2],
        staging: usize,
        norm: (f32, bool),
        rope: Option<([usize; 5], u64)>,
        caches: Option<([usize; 2], [u64; 6])>,
        attention: Option<(usize, [u64; 4])>,
    ) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            let bad =
                || crate::CudaError::invalid_argument("norm QKV", "parent index out of range");
            let get = |i: usize| {
                self.parents
                    .devices
                    .get(i)
                    .map(|d| d.native_handle())
                    .ok_or_else(bad)
            };
            let d = [
                get(devices[0])?,
                get(devices[1])?,
                get(devices[2])?,
                get(devices[3])?,
                get(devices[4])?,
                get(devices[5])?,
                get(devices[6])?,
                get(devices[7])?,
                get(devices[8])?,
            ];
            let w = workspace.map(get).transpose()?;
            let q = self
                .parents
                .plans
                .get(plans[0])
                .ok_or_else(bad)?
                .graph_resource_handle()?;
            let kv = self
                .parents
                .plans
                .get(plans[1])
                .ok_or_else(bad)?
                .graph_resource_handle()?;
            let staging = self.parents.pinned.get(staging).ok_or_else(bad)?;
            let rope = rope
                .map(|(ids, offset)| -> CudaResult<_> {
                    Ok((
                        [
                            get(ids[0])?,
                            get(ids[1])?,
                            get(ids[2])?,
                            get(ids[3])?,
                            get(ids[4])?,
                        ],
                        offset,
                    ))
                })
                .transpose()?;
            let caches = caches
                .map(|(ids, geometry)| -> CudaResult<_> {
                    Ok(([get(ids[0])?, get(ids[1])?], geometry))
                })
                .transpose()?;
            let attention = attention
                .map(|(output, fields)| -> CudaResult<_> { Ok((get(output)?, fields)) })
                .transpose()?;
            self.native.record_attention_impl(
                d,
                w,
                [q, kv],
                staging.native_handle(),
                norm.0,
                norm.1,
                rope,
                caches,
                attention,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                devices, workspace, plans, staging, norm, rope, caches, attention,
            );
            Err(crate::CudaError::unavailable("record norm QKV"))
        }
    }
}

impl BorrowedGraphResourceReservation<'_> {
    /// Records QKV/RoPE and KV write to a fixed logical/physical block.
    /// Cache geometry is layers, layer, physical blocks, physical, logical, valid prefix.
    /// # Errors
    /// Rejects invalid parents, mapping geometry or capture failure.
    #[allow(clippy::too_many_arguments)] // Explicit retained parent groups and fixed cache binding.
    pub fn record_qkv_kv(
        &mut self,
        devices: [usize; 9],
        workspace: Option<usize>,
        plans: [usize; 2],
        staging: usize,
        norm: (f32, bool),
        rope: ([usize; 5], u64),
        caches: ([usize; 2], [u64; 6]),
    ) -> CudaResult<()> {
        self.record_qkv_impl(
            devices,
            workspace,
            plans,
            staging,
            norm,
            Some(rope),
            Some(caches),
        )
    }
}

impl BorrowedGraphResourceReservation<'_> {
    /// Records norm/QKV/RoPE/KV write and one-block grouped attention.
    /// # Errors
    /// Rejects unsupported mapping, parent geometry, field overlaps or capture errors.
    #[allow(clippy::too_many_arguments)]
    pub fn record_attention_chain(
        &mut self,
        devices: [usize; 9],
        workspace: Option<usize>,
        plans: [usize; 2],
        staging: usize,
        norm: (f32, bool),
        rope: ([usize; 5], u64),
        caches: ([usize; 2], [u64; 6]),
        attention: (usize, [u64; 4]),
    ) -> CudaResult<()> {
        self.record_attention_impl(
            devices,
            workspace,
            plans,
            staging,
            norm,
            Some(rope),
            Some(caches),
            Some(attention),
        )
    }
}

impl BorrowedGraphResourceReservation<'_> {
    /// Captures a full single-row, D64 canonical Llama model using retained parents.
    /// Devices: token, hidden, norm, projection, rotary-Q, context, raw-K, raw-V,
    /// rotary-K, gate, up, activated, product, logits, embedding-report, K/V pools,
    /// cos, sin, metadata, argmax. Plans: hidden, KV, intermediate, down, head.
    /// Weights: embedding, final norm, head, then nine weights per layer:
    /// input norm, Q, K, V, O, post norm, gate, up, down. Geometry: layers,
    /// physical blocks, block capacity, vocabulary. Epsilons: final, then two per layer.
    /// Replay uses an exact-sized payload: token/position u32, [0,live_blocks] u32,
    /// capacity physical u32s, capacity valid u16s, aligned zero row-slot u32.
    /// The rest is padding. Size is max(metadata length, 40 + optional vocab*2).
    /// Output is optional BF16 logits, argmax token/status (8 bytes), embedding report (32 bytes).
    /// # Errors
    /// Rejects unsupported geometry/profile, invalid parents, or capture failures.
    #[allow(clippy::too_many_arguments)]
    pub fn record_decode(
        &mut self,
        devices: [usize; 21],
        workspace: Option<usize>,
        weights: &[usize],
        plans: [usize; 5],
        staging: usize,
        geometry: [u64; 4],
        eps: &[f32],
        hf: bool,
        publish_logits: bool,
    ) -> CudaResult<()> {
        self.record_decode_with_profile(
            devices,
            workspace,
            weights,
            plans,
            staging,
            geometry,
            eps,
            if hf {
                DecodeNumericalProfile::HuggingFaceSmolLm2
            } else {
                DecodeNumericalProfile::Canonical
            },
            publish_logits,
        )
    }

    /// Records an explicitly selected numerical contract; no cross-profile fallback.
    /// # Errors
    /// Rejects unsupported geometry, environment, parents or capture failure.
    #[allow(clippy::too_many_arguments)]
    pub fn record_decode_with_profile(
        &mut self,
        devices: [usize; 21],
        workspace: Option<usize>,
        weights: &[usize],
        plans: [usize; 5],
        staging: usize,
        geometry: [u64; 4],
        eps: &[f32],
        profile: DecodeNumericalProfile,
        publish_logits: bool,
    ) -> CudaResult<()> {
        self.record_decode_with_profile_and_prefill(
            devices,
            workspace,
            weights,
            plans,
            staging,
            geometry,
            eps,
            profile,
            publish_logits,
            None,
        )
    }

    /// Records an optional P128 prefill graph alongside the retained M1 decode graph.
    /// Prefill indices select twelve separately reserved parents in this order:
    /// hidden, norm, projection, rotary Q, context, raw K, raw V, rotary K,
    /// gate, up, FP32 residual, product. Their byte sizes are 128 times
    /// `[1152, 1152, 1152, 1152, 1152, 384, 384, 384, 3072, 3072, 2304, 3072]`.
    /// Only `VllmSmolP128V1` accepts prefill parents. Native validation checks
    /// exact geometry, nonaliasing and the shared reservation before capture.
    /// # Errors
    /// Rejects an incompatible profile, missing parents or invalid native capture.
    #[allow(clippy::too_many_arguments)]
    pub fn record_decode_with_profile_and_prefill(
        &mut self,
        devices: [usize; 21],
        workspace: Option<usize>,
        weights: &[usize],
        plans: [usize; 5],
        staging: usize,
        geometry: [u64; 4],
        eps: &[f32],
        profile: DecodeNumericalProfile,
        publish_logits: bool,
        prefill: Option<[usize; 12]>,
    ) -> CudaResult<()> {
        self.record_decode_with_profile_prefill_and_packed(
            devices,
            workspace,
            weights,
            plans,
            staging,
            geometry,
            eps,
            profile,
            publish_logits,
            prefill,
            None,
            None,
        )
    }

    /// Records P128 prefill with optional packed M1 decode projections.
    /// Packed device parents alternate QKV `[960, 576]` and gate/up `[3072, 576]`
    /// row-major BF16 weights for each of thirty layers, followed by BF16 output
    /// parents of 1920 and 6144 bytes. The two packed plans are respectively
    /// M1/N960/K576 and M1/N3072/K576. All indices select retained parents.
    /// Native validation checks distinct, nonaliasing parents and the exact
    /// workspace-free, no-split plan metadata before capture.
    /// # Errors
    /// Rejects packed parents without packed plans, P128 prefill or the
    /// `VllmSmolP128V1` profile, out-of-range indices, or invalid native capture.
    #[allow(clippy::too_many_arguments)]
    pub fn record_decode_with_profile_prefill_and_packed(
        &mut self,
        devices: [usize; 21],
        workspace: Option<usize>,
        weights: &[usize],
        plans: [usize; 5],
        staging: usize,
        geometry: [u64; 4],
        eps: &[f32],
        profile: DecodeNumericalProfile,
        publish_logits: bool,
        prefill: Option<[usize; 12]>,
        packed: Option<[usize; 62]>,
        packed_plans: Option<[usize; 2]>,
    ) -> CudaResult<()> {
        if packed.is_some() != packed_plans.is_some() || (packed.is_some() && prefill.is_none()) {
            return Err(crate::CudaError::invalid_argument(
                "record decode with packed projections",
                "packed parents and plans must be supplied together with P128 prefill",
            ));
        }
        if prefill.is_some() && profile != DecodeNumericalProfile::VllmSmolP128V1 {
            return Err(crate::CudaError::invalid_argument(
                "record decode with P128 prefill",
                "P128 prefill requires VllmSmolP128V1",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let bad =
                || crate::CudaError::invalid_argument("record decode", "parent index out of range");
            let devices = devices
                .iter()
                .map(|&i| {
                    self.parents
                        .devices
                        .get(i)
                        .map(|d| d.native_handle())
                        .ok_or_else(bad)
                })
                .collect::<CudaResult<Vec<_>>>()?;
            let workspace = workspace
                .map(|i| {
                    self.parents
                        .devices
                        .get(i)
                        .map(|d| d.native_handle())
                        .ok_or_else(bad)
                })
                .transpose()?;
            let weights = weights
                .iter()
                .map(|&i| {
                    self.parents
                        .devices
                        .get(i)
                        .map(|d| d.native_handle())
                        .ok_or_else(bad)
                })
                .collect::<CudaResult<Vec<_>>>()?;
            let plans = plans
                .iter()
                .map(|&i| {
                    self.parents
                        .plans
                        .get(i)
                        .ok_or_else(bad)?
                        .graph_resource_handle()
                })
                .collect::<CudaResult<Vec<_>>>()?;
            let staging = self.parents.pinned.get(staging).ok_or_else(bad)?;
            let prefill = prefill
                .map(|indices| -> CudaResult<_> {
                    indices
                        .iter()
                        .map(|&index| {
                            self.parents
                                .devices
                                .get(index)
                                .map(|buffer| buffer.native_handle())
                                .ok_or_else(bad)
                        })
                        .collect::<CudaResult<Vec<_>>>()?
                        .try_into()
                        .map_err(|_| bad())
                })
                .transpose()?;
            let packed = packed
                .map(|indices| -> CudaResult<_> {
                    indices
                        .iter()
                        .map(|&index| {
                            self.parents
                                .devices
                                .get(index)
                                .map(|buffer| buffer.native_handle())
                                .ok_or_else(bad)
                        })
                        .collect::<CudaResult<Vec<_>>>()?
                        .try_into()
                        .map_err(|_| bad())
                })
                .transpose()?;
            let packed_plans = packed_plans
                .map(|indices| -> CudaResult<_> {
                    indices
                        .iter()
                        .map(|&index| {
                            self.parents
                                .plans
                                .get(index)
                                .ok_or_else(bad)?
                                .graph_resource_handle()
                        })
                        .collect::<CudaResult<Vec<_>>>()?
                        .try_into()
                        .map_err(|_| bad())
                })
                .transpose()?;
            self.native.record_decode(
                devices.try_into().map_err(|_| bad())?,
                workspace,
                &weights,
                plans.try_into().map_err(|_| bad())?,
                staging.native_handle(),
                geometry,
                eps,
                profile as u32,
                publish_logits,
                prefill,
                packed,
                packed_plans,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                devices,
                workspace,
                weights,
                plans,
                staging,
                geometry,
                eps,
                profile as u32,
                publish_logits,
                prefill,
                packed,
                packed_plans,
            );
            Err(crate::CudaError::unavailable("record decode"))
        }
    }
}

impl BorrowedGraphResourceReservation<'_> {
    /// Records a fixed SmolLM2 numerical decode DAG for two or four request rows.
    /// Device roles follow the native recorder's 18-parent contract; weights are
    /// the 273 legacy weights followed by 60 packed QKV/gate-up parents. This
    /// low-level owner does not authorize scheduler reservations or KV ownership.
    /// # Errors
    /// Rejects invalid indices, parent geometry, unsupported buckets or native failures.
    #[allow(clippy::too_many_arguments)]
    pub fn record_multisequence_decode(
        &mut self,
        devices: &[usize; 18],
        weights: &[usize],
        plans: &[usize; 5],
        staging: usize,
        bucket: u32,
        physical: u32,
        full: bool,
    ) -> CudaResult<()> {
        self.prepare_multisequence_entry(
            devices, weights, plans, staging, bucket, physical, full, false, false,
        )
    }
    pub fn append_multisequence_decode(
        &mut self,
        devices: &[usize; 18],
        weights: &[usize],
        plans: &[usize; 5],
        staging: usize,
        bucket: u32,
        physical: u32,
        full: bool,
    ) -> CudaResult<()> {
        self.prepare_multisequence_entry(
            devices, weights, plans, staging, bucket, physical, full, true, false,
        )
    }
    pub fn append_shared_multisequence_decode(
        &mut self,
        devices: &[usize; 18],
        weights: &[usize],
        plans: &[usize; 5],
        staging: usize,
        bucket: u32,
        physical: u32,
        full: bool,
    ) -> CudaResult<()> {
        self.prepare_multisequence_entry(
            devices, weights, plans, staging, bucket, physical, full, true, true,
        )
    }
    pub fn record_shared_multisequence_decode(
        &mut self,
        devices: &[usize; 18],
        weights: &[usize],
        plans: &[usize; 5],
        staging: usize,
        bucket: u32,
        physical: u32,
        full: bool,
    ) -> CudaResult<()> {
        self.prepare_multisequence_entry(
            devices, weights, plans, staging, bucket, physical, full, false, true,
        )
    }
    fn prepare_multisequence_entry(
        &mut self,
        devices: &[usize; 18],
        weights: &[usize],
        plans: &[usize; 5],
        staging: usize,
        bucket: u32,
        physical: u32,
        full: bool,
        append: bool,
        shared_rows: bool,
    ) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            let bad = || {
                crate::CudaError::invalid_argument(
                    "record multi-sequence decode",
                    "parent index out of range",
                )
            };
            let devices: Vec<_> = devices
                .iter()
                .map(|i| {
                    self.parents
                        .devices
                        .get(*i)
                        .map(|p| p.native_handle())
                        .ok_or_else(bad)
                })
                .collect::<CudaResult<_>>()?;
            let weights: Vec<_> = weights
                .iter()
                .map(|i| {
                    self.parents
                        .devices
                        .get(*i)
                        .map(|p| p.native_handle())
                        .ok_or_else(bad)
                })
                .collect::<CudaResult<_>>()?;
            let plans: Vec<_> = plans
                .iter()
                .enumerate()
                .map(|(role, i)| {
                    if shared_rows && matches!(role, 0 | 1 | 4) {
                        return self.parents.plans.get(*i).ok_or_else(bad)?.graph_resource_handle();
                    }
                    self.strided_plans
                        .get(*i)
                        .ok_or_else(bad)?
                        .graph_resource_handle()
                })
                .collect::<CudaResult<_>>()?;
            let staging = self.parents.pinned.get(staging).ok_or_else(bad)?;
            self.native.record_multisequence_decode(
                &devices,
                &weights,
                &plans,
                staging.native_handle(),
                bucket,
                physical,
                full,
                append,
                shared_rows,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (
                devices, weights, plans, staging, bucket, physical, full, append, shared_rows,
            );
            Err(crate::CudaError::unavailable(
                "record multi-sequence decode",
            ))
        }
    }
}

impl BorrowedGraphResourceReservation<'_> {
    /// Replays a pre-recorded entry: 0 original, 1 appended N2, 2 appended N4.
    pub fn replay_catalog(&mut self, index: u32, source: &[u8]) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.native.replay_catalog(index, source)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (index, source);
            Err(crate::CudaError::unavailable("replay graph catalog"))
        }
    }
    /// Reads only the entry selected by the most recent successful replay.
    pub fn read_catalog(&mut self, index: u32, output: &mut [u8]) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.native.read_catalog(index, output)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (index, output);
            Err(crate::CudaError::unavailable("read graph catalog"))
        }
    }
}

impl BorrowedGraphResourceReservation<'_> {
    /// Records V3 prefill and its canonical head under this retained ledger.
    /// This does not authorize scheduler publication of the returned result.
    pub fn record_v3_prefill(&mut self,devices:&[usize;22],workspace:Option<usize>,weights:&[usize],head:usize,staging:usize,capacity:u32,physical:u32)->CudaResult<()> {
        #[cfg(feature="cuda")]{
            let bad=||crate::CudaError::invalid_argument("record V3 prefill","parent index out of range");
            let devices:Vec<_>=devices.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let weights:Vec<_>=weights.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let workspace=workspace.map(|i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).transpose()?;
            let head=self.parents.plans.get(head).ok_or_else(bad)?.graph_resource_handle()?;
            let staging=self.parents.pinned.get(staging).ok_or_else(bad)?.native_handle();
            self.native.record_v3_prefill(&devices,workspace,&weights,head,staging,capacity,physical)
        }
        #[cfg(not(feature="cuda"))]{let _=(devices,workspace,weights,head,staging,capacity,physical);Err(crate::CudaError::unavailable("record V3 prefill"))}
    }
    pub fn record_v3_shared(&mut self,devices:&[usize;25],workspace:Option<usize>,weights:&[usize],head:usize,shared_head:usize,staging:usize,capacity:u32,physical:u32)->CudaResult<()> {
        #[cfg(feature="cuda")]{
            let bad=||crate::CudaError::invalid_argument("record V3 prefill","parent index out of range");
            let devices:Vec<_>=devices.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let weights:Vec<_>=weights.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let workspace=workspace.map(|i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).transpose()?;
            let head=self.parents.plans.get(head).ok_or_else(bad)?.graph_resource_handle()?;
            let shared_head=self.parents.plans.get(shared_head).ok_or_else(bad)?.graph_resource_handle()?;
            let staging=self.parents.pinned.get(staging).ok_or_else(bad)?.native_handle();
            self.native.record_v3_shared(&devices,workspace,&weights,head,shared_head,staging,capacity,physical)
        }
        #[cfg(not(feature="cuda"))]{let _=(devices,workspace,weights,head,shared_head,staging,capacity,physical);Err(crate::CudaError::unavailable("record V3 prefill"))}
    }
    pub fn record_v4_shared(&mut self,devices:&[usize;25],workspace:Option<usize>,weights:&[usize],head:usize,shared_head:usize,staging:usize,capacity:u32,physical:u32)->CudaResult<()> {
        #[cfg(feature="cuda")]{
            let bad=||crate::CudaError::invalid_argument("record V4 shared","parent index out of range");
            let devices:Vec<_>=devices.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let weights:Vec<_>=weights.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let workspace=workspace.map(|i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).transpose()?;
            let head=self.parents.plans.get(head).ok_or_else(bad)?.graph_resource_handle()?;
            let shared_head=self.parents.plans.get(shared_head).ok_or_else(bad)?.graph_resource_handle()?;
            let staging=self.parents.pinned.get(staging).ok_or_else(bad)?.native_handle();
            self.native.record_v4_shared(&devices,workspace,&weights,head,shared_head,staging,capacity,physical)
        }
        #[cfg(not(feature="cuda"))]{let _=(devices,workspace,weights,head,shared_head,staging,capacity,physical);Err(crate::CudaError::unavailable("record V4 shared"))}
    }
pub fn record_v5_shared(&mut self,devices:&[usize;25],workspace:Option<usize>,weights:&[usize],head:usize,shared_head:usize,staging:usize,capacity:u32,physical:u32)->CudaResult<()> {
        #[cfg(feature="cuda")]{
            let bad=||crate::CudaError::invalid_argument("record V4 shared","parent index out of range");
            let devices:Vec<_>=devices.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let weights:Vec<_>=weights.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let workspace=workspace.map(|i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).transpose()?;
            let head=self.parents.plans.get(head).ok_or_else(bad)?.graph_resource_handle()?;
            let shared_head=self.parents.plans.get(shared_head).ok_or_else(bad)?.graph_resource_handle()?;
            let staging=self.parents.pinned.get(staging).ok_or_else(bad)?.native_handle();
            self.native.record_v5_shared(&devices,workspace,&weights,head,shared_head,staging,capacity,physical)
        }
        #[cfg(not(feature="cuda"))]{let _=(devices,workspace,weights,head,shared_head,staging,capacity,physical);Err(crate::CudaError::unavailable("record V4 shared"))}
    }
pub fn record_v6_shared(&mut self,devices:&[usize;25],workspace:Option<usize>,weights:&[usize],head:usize,shared_head:usize,staging:usize,capacity:u32,physical:u32)->CudaResult<()> {
        #[cfg(feature="cuda")]{
            let bad=||crate::CudaError::invalid_argument("record V6 packed","parent index out of range");
            let devices:Vec<_>=devices.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let weights:Vec<_>=weights.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let workspace=workspace.map(|i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).transpose()?;
            let head=self.parents.plans.get(head).ok_or_else(bad)?.graph_resource_handle()?;
            let shared_head=self.parents.plans.get(shared_head).ok_or_else(bad)?.graph_resource_handle()?;
            let staging=self.parents.pinned.get(staging).ok_or_else(bad)?.native_handle();
            self.native.record_v6_shared(&devices,workspace,&weights,head,shared_head,staging,capacity,physical)
        }
        #[cfg(not(feature="cuda"))]{let _=(devices,workspace,weights,head,shared_head,staging,capacity,physical);Err(crate::CudaError::unavailable("record V6 packed"))}
    }
/// Records experimental FlashInfer decode arithmetic, retaining its workspace.
/// This is not numerically interchangeable with the existing exact profile.
pub fn record_v7_flashinfer_experimental(&mut self,devices:&[usize;25],workspace:Option<usize>,weights:&[usize],head:usize,shared_head:usize,staging:usize,capacity:u32,physical:u32,attention_workspace:usize,compact:bool)->CudaResult<()> {
        #[cfg(feature="cuda")]{
            let bad=||crate::CudaError::invalid_argument("record V7 mixed","parent index out of range");
            let devices:Vec<_>=devices.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let weights:Vec<_>=weights.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let workspace=workspace.map(|i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).transpose()?;
            let attention_workspace=self.parents.devices.get(attention_workspace).ok_or_else(bad)?.native_handle();
            let head=self.parents.plans.get(head).ok_or_else(bad)?.graph_resource_handle()?;
            let shared_head=self.parents.plans.get(shared_head).ok_or_else(bad)?.graph_resource_handle()?;
            let staging=self.parents.pinned.get(staging).ok_or_else(bad)?.native_handle();
            self.native.record_v7_flashinfer_experimental(&devices,workspace,&weights,head,shared_head,staging,capacity,physical,attention_workspace,compact)
        }
        #[cfg(not(feature="cuda"))]{let _=(devices,workspace,weights,head,shared_head,staging,capacity,physical,attention_workspace,compact);Err(crate::CudaError::unavailable("record V7 mixed"))}
    }
/// Records experimental prefill arithmetic; decode keeps the existing V7 backend.
/// The retained workspace must have the exact native prefill extent (33,996 bytes).
pub fn record_v7_flashinfer_prefill_only(&mut self,devices:&[usize;25],workspace:Option<usize>,weights:&[usize],head:usize,shared_head:usize,staging:usize,capacity:u32,physical:u32,attention_workspace:usize,compact:bool)->CudaResult<()> {
        #[cfg(feature="cuda")]{
            let bad=||crate::CudaError::invalid_argument("record V7 mixed","parent index out of range");
            let devices:Vec<_>=devices.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let weights:Vec<_>=weights.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let workspace=workspace.map(|i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).transpose()?;
            let attention_workspace=self.parents.devices.get(attention_workspace).ok_or_else(bad)?.native_handle();
            let head=self.parents.plans.get(head).ok_or_else(bad)?.graph_resource_handle()?;
            let shared_head=self.parents.plans.get(shared_head).ok_or_else(bad)?.graph_resource_handle()?;
            let staging=self.parents.pinned.get(staging).ok_or_else(bad)?.native_handle();
            self.native.record_v7_flashinfer_prefill_only(&devices,workspace,&weights,head,shared_head,staging,capacity,physical,attention_workspace,compact)
        }
        #[cfg(not(feature="cuda"))]{let _=(devices,workspace,weights,head,shared_head,staging,capacity,physical,attention_workspace,compact);Err(crate::CudaError::unavailable("record V7 mixed"))}
    }
/// Records unqualified experimental FA3 arithmetic with a retained workspace.
pub fn record_v7_fa3_experimental(&mut self,devices:&[usize;25],workspace:Option<usize>,weights:&[usize],head:usize,shared_head:usize,staging:usize,capacity:u32,physical:u32,attention_workspace:usize,compact:bool)->CudaResult<()> {
        #[cfg(feature="cuda")]{
            let bad=||crate::CudaError::invalid_argument("record V7 mixed","parent index out of range");
            let devices:Vec<_>=devices.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let weights:Vec<_>=weights.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let workspace=workspace.map(|i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).transpose()?;
            let attention_workspace=self.parents.devices.get(attention_workspace).ok_or_else(bad)?.native_handle();
            let head=self.parents.plans.get(head).ok_or_else(bad)?.graph_resource_handle()?;
            let shared_head=self.parents.plans.get(shared_head).ok_or_else(bad)?.graph_resource_handle()?;
            let staging=self.parents.pinned.get(staging).ok_or_else(bad)?.native_handle();
            self.native.record_v7_fa3_experimental(&devices,workspace,&weights,head,shared_head,staging,capacity,physical,attention_workspace,compact)
        }
        #[cfg(not(feature="cuda"))]{let _=(devices,workspace,weights,head,shared_head,staging,capacity,physical,attention_workspace,compact);Err(crate::CudaError::unavailable("record V7 mixed"))}
    }
/// Records the explicit exact-order FFN pipeline candidate.
pub fn record_v7_ffn_pipeline(&mut self,devices:&[usize;25],workspace:Option<usize>,weights:&[usize],head:usize,shared_head:usize,staging:usize,capacity:u32,physical:u32,compact:bool)->CudaResult<()> {
        #[cfg(feature="cuda")]{
            let bad=||crate::CudaError::invalid_argument("record V7 mixed","parent index out of range");
            let devices:Vec<_>=devices.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let weights:Vec<_>=weights.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let workspace=workspace.map(|i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).transpose()?;
            let head=self.parents.plans.get(head).ok_or_else(bad)?.graph_resource_handle()?;
            let shared_head=self.parents.plans.get(shared_head).ok_or_else(bad)?.graph_resource_handle()?;
            let staging=self.parents.pinned.get(staging).ok_or_else(bad)?.native_handle();
            self.native.record_v7_ffn_pipeline(&devices,workspace,&weights,head,shared_head,staging,capacity,physical,compact)
        }
        #[cfg(not(feature="cuda"))]{let _=(devices,workspace,weights,head,shared_head,staging,capacity,physical,compact);Err(crate::CudaError::unavailable("record V7 mixed"))}
    }
pub fn record_v7_prefill_ffn_pipeline(&mut self,devices:&[usize;25],workspace:Option<usize>,weights:&[usize],head:usize,shared_head:usize,staging:usize,capacity:u32,physical:u32,compact:bool)->CudaResult<()> {
        #[cfg(feature="cuda")]{
            let bad=||crate::CudaError::invalid_argument("record V7 mixed","parent index out of range");
            let devices:Vec<_>=devices.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let weights:Vec<_>=weights.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let workspace=workspace.map(|i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).transpose()?;
            let head=self.parents.plans.get(head).ok_or_else(bad)?.graph_resource_handle()?;
            let shared_head=self.parents.plans.get(shared_head).ok_or_else(bad)?.graph_resource_handle()?;
            let staging=self.parents.pinned.get(staging).ok_or_else(bad)?.native_handle();
            self.native.record_v7_prefill_ffn_pipeline(&devices,workspace,&weights,head,shared_head,staging,capacity,physical,compact)
        }
        #[cfg(not(feature="cuda"))]{let _=(devices,workspace,weights,head,shared_head,staging,capacity,physical,compact);Err(crate::CudaError::unavailable("record V7 mixed"))}
    }
pub fn record_v7_shared(&mut self,devices:&[usize;25],workspace:Option<usize>,weights:&[usize],head:usize,shared_head:usize,staging:usize,capacity:u32,physical:u32)->CudaResult<()> {
        #[cfg(feature="cuda")]{
            let bad=||crate::CudaError::invalid_argument("record V7 mixed","parent index out of range");
            let devices:Vec<_>=devices.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let weights:Vec<_>=weights.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let workspace=workspace.map(|i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).transpose()?;
            let head=self.parents.plans.get(head).ok_or_else(bad)?.graph_resource_handle()?;
            let shared_head=self.parents.plans.get(shared_head).ok_or_else(bad)?.graph_resource_handle()?;
            let staging=self.parents.pinned.get(staging).ok_or_else(bad)?.native_handle();
            self.native.record_v7_shared(&devices,workspace,&weights,head,shared_head,staging,capacity,physical)
        }
        #[cfg(not(feature="cuda"))]{let _=(devices,workspace,weights,head,shared_head,staging,capacity,physical);Err(crate::CudaError::unavailable("record V7 mixed"))}
    }
    pub fn record_v4_shared_greedy(&mut self,devices:&[usize;25],workspace:Option<usize>,weights:&[usize],head:usize,shared_head:usize,staging:usize,capacity:u32,physical:u32)->CudaResult<()> {
        #[cfg(feature="cuda")]{
            let bad=||crate::CudaError::invalid_argument("record V4 shared","parent index out of range");
            let devices:Vec<_>=devices.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let weights:Vec<_>=weights.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let workspace=workspace.map(|i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).transpose()?;
            let head=self.parents.plans.get(head).ok_or_else(bad)?.graph_resource_handle()?;
            let shared_head=self.parents.plans.get(shared_head).ok_or_else(bad)?.graph_resource_handle()?;
            let staging=self.parents.pinned.get(staging).ok_or_else(bad)?.native_handle();
            self.native.record_v4_shared_greedy(&devices,workspace,&weights,head,shared_head,staging,capacity,physical)
        }
        #[cfg(not(feature="cuda"))]{let _=(devices,workspace,weights,head,shared_head,staging,capacity,physical);Err(crate::CudaError::unavailable("record V4 shared"))}
    }
pub fn record_v5_shared_greedy(&mut self,devices:&[usize;25],workspace:Option<usize>,weights:&[usize],head:usize,shared_head:usize,staging:usize,capacity:u32,physical:u32)->CudaResult<()> {
        #[cfg(feature="cuda")]{
            let bad=||crate::CudaError::invalid_argument("record V4 shared","parent index out of range");
            let devices:Vec<_>=devices.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let weights:Vec<_>=weights.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let workspace=workspace.map(|i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).transpose()?;
            let head=self.parents.plans.get(head).ok_or_else(bad)?.graph_resource_handle()?;
            let shared_head=self.parents.plans.get(shared_head).ok_or_else(bad)?.graph_resource_handle()?;
            let staging=self.parents.pinned.get(staging).ok_or_else(bad)?.native_handle();
            self.native.record_v5_shared_greedy(&devices,workspace,&weights,head,shared_head,staging,capacity,physical)
        }
        #[cfg(not(feature="cuda"))]{let _=(devices,workspace,weights,head,shared_head,staging,capacity,physical);Err(crate::CudaError::unavailable("record V4 shared"))}
    }
pub fn record_v6_shared_greedy(&mut self,devices:&[usize;25],workspace:Option<usize>,weights:&[usize],head:usize,shared_head:usize,staging:usize,capacity:u32,physical:u32)->CudaResult<()> {
        #[cfg(feature="cuda")]{
            let bad=||crate::CudaError::invalid_argument("record V6 packed","parent index out of range");
            let devices:Vec<_>=devices.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let weights:Vec<_>=weights.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let workspace=workspace.map(|i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).transpose()?;
            let head=self.parents.plans.get(head).ok_or_else(bad)?.graph_resource_handle()?;
            let shared_head=self.parents.plans.get(shared_head).ok_or_else(bad)?.graph_resource_handle()?;
            let staging=self.parents.pinned.get(staging).ok_or_else(bad)?.native_handle();
            self.native.record_v6_shared_greedy(&devices,workspace,&weights,head,shared_head,staging,capacity,physical)
        }
        #[cfg(not(feature="cuda"))]{let _=(devices,workspace,weights,head,shared_head,staging,capacity,physical);Err(crate::CudaError::unavailable("record V6 packed"))}
    }
pub fn record_v7_shared_greedy(&mut self,devices:&[usize;25],workspace:Option<usize>,weights:&[usize],head:usize,shared_head:usize,staging:usize,capacity:u32,physical:u32)->CudaResult<()> {
        #[cfg(feature="cuda")]{
            let bad=||crate::CudaError::invalid_argument("record V7 mixed","parent index out of range");
            let devices:Vec<_>=devices.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let weights:Vec<_>=weights.iter().map(|&i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).collect::<CudaResult<_>>()?;
            let workspace=workspace.map(|i|self.parents.devices.get(i).map(|p|p.native_handle()).ok_or_else(bad)).transpose()?;
            let head=self.parents.plans.get(head).ok_or_else(bad)?.graph_resource_handle()?;
            let shared_head=self.parents.plans.get(shared_head).ok_or_else(bad)?.graph_resource_handle()?;
            let staging=self.parents.pinned.get(staging).ok_or_else(bad)?.native_handle();
            self.native.record_v7_shared_greedy(&devices,workspace,&weights,head,shared_head,staging,capacity,physical)
        }
        #[cfg(not(feature="cuda"))]{let _=(devices,workspace,weights,head,shared_head,staging,capacity,physical);Err(crate::CudaError::unavailable("record V7 mixed"))}
    }
}

#[cfg(feature = "cuda")]
impl BorrowedGraphResourceReservation<'_> {
    /// Cold preparation using two pinned parents already in the ledger.
    /// Slot zero may reuse an existing combined model input/output staging parent.
    /// Each slot must cover the recorded input/output staging extent.
    /// # Errors
    /// Rejects invalid parent indices, extents or graph state; CUDA errors propagate.
    pub fn prepare_buffered_transfers(&mut self, first: usize, second: usize) -> CudaResult<()> {
        let bad = || crate::CudaError::invalid_argument("prepare buffered transfers", "staging index outside retained parents");
        let first = self.parents.pinned.get(first).ok_or_else(bad)?.native_handle();
        let second = self.parents.pinned.get(second).ok_or_else(bad)?.native_handle();
        self.native.prepare_buffered_transfers(first, second)
    }
    /// Submit into the next free prepared staging slot on the retained stream.
    /// # Errors
    /// Rejects full slots, invalid input or an unprepared/failed owner.
    pub fn submit_buffered_transfer(&mut self, input: &[u8]) -> CudaResult<u64> { self.native.submit_buffered_transfer(input) }
    /// Submit V7 compact decode using the latest unconsumed predecessor output.
    /// # Errors
    /// Rejects wrong extents, stale tickets and unsupported prepared profiles.
    pub fn submit_future_transfer(&mut self, input: &[u8], references: &[u8], predecessor: u64) -> CudaResult<u64> { self.native.submit_future_transfer(input, references, predecessor) }

    /// Poll or wait for this owner's exact ticket; does not consume output.
    /// # Errors
    /// Rejects stale tickets and propagates CUDA completion errors.
    pub fn query_buffered_transfer(&mut self, ticket: u64, wait: bool) -> CudaResult<bool> { self.native.query_buffered_transfer(ticket, wait) }
    /// Copy completed output and release only its staging slot for reuse.
    /// # Errors
    /// Rejects stale/pending tickets or an incorrect destination size.
    pub fn read_buffered_transfer(&mut self, ticket: u64, output: &mut [u8]) -> CudaResult<()> { self.native.read_buffered_transfer(ticket, output) }

    /// Submit using retained parents; input is staged before return.
    /// # Errors
    /// Propagates validation and CUDA submission failures.
    pub fn submit_transfer(&mut self, input: &[u8]) -> CudaResult<()> { self.native.submit_transfer(input) }
    /// Query the retained completion event.
    /// # Errors
    /// Propagates absent or failed completion.
    pub fn query_transfer(&mut self) -> CudaResult<bool> { self.native.query_transfer() }
    /// Wait for the retained completion event.
    /// # Errors
    /// Failed completion keeps native parents retained.
    pub fn wait_transfer(&mut self) -> CudaResult<()> { self.native.wait_transfer() }
}
