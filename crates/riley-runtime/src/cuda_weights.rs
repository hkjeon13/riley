use std::collections::BTreeMap;
use std::error;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use riley_cuda::{
    CudaBufferSpan, CudaContext, CudaDType, CudaDeviceBuffer, CudaError, CudaResult, CudaStream,
};
use riley_model::{LoadedWeights, ModelError, TensorSource, WeightSlot};
use riley_tensor::DType;

use crate::llama::{PhysicalWeightId, PhysicalWeightMetadata};

static NEXT_WEIGHT_OWNER_ID: AtomicU64 = AtomicU64::new(1);

/// Result type for canonical checkpoint upload to one CUDA context.
pub type CudaWeightUploadResult<T> = Result<T, CudaWeightUploadError>;

/// Stable failure from the model-to-CUDA upload boundary.
#[derive(Debug)]
#[non_exhaustive]
pub enum CudaWeightUploadError {
    /// Canonical weight lookup failed after model validation.
    Model(ModelError),
    /// Device/pinned allocation, copy, synchronization, or close failed.
    Cuda(CudaError),
    /// The current primitive path cannot consume this checkpoint dtype.
    UnsupportedDType {
        /// Canonical slot that exposed the dtype.
        slot: WeightSlot,
        /// Unsupported tensor dtype.
        dtype: DType,
    },
    /// No reusable pinned bytes were permitted for a non-empty checkpoint.
    EmptyStagingCapacity,
    /// A host tensor byte length does not fit the fixed-width CUDA ABI.
    HostLengthOverflow {
        /// Canonical slot whose tensor length overflowed.
        slot: WeightSlot,
    },
    /// Accumulating uploaded physical bytes overflowed `u64`.
    TotalLengthOverflow,
    /// A requested canonical slot was absent from the uploaded mapping.
    MissingSlot {
        /// Missing canonical slot.
        slot: WeightSlot,
    },
    /// The process exhausted non-zero uploaded-weight owner identities.
    WeightOwnerIdentityExhausted,
    /// A physical ID was created by a different uploaded-weight owner.
    ForeignPhysicalWeightId,
    /// A physical ID index was absent from its uploaded-weight owner.
    MissingPhysicalWeight { index: usize },
}

impl fmt::Display for CudaWeightUploadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => error.fmt(formatter),
            Self::Cuda(error) => error.fmt(formatter),
            Self::UnsupportedDType { slot, dtype } => write!(
                formatter,
                "CUDA upload does not support {dtype} for {}",
                slot.name()
            ),
            Self::EmptyStagingCapacity => formatter
                .write_str("CUDA weight upload requires a non-zero pinned staging capacity"),
            Self::HostLengthOverflow { slot } => write!(
                formatter,
                "host byte length for {} does not fit the CUDA ABI",
                slot.name()
            ),
            Self::TotalLengthOverflow => {
                formatter.write_str("total uploaded physical bytes overflow u64")
            }
            Self::MissingSlot { slot } => {
                write!(formatter, "uploaded CUDA weights omit {}", slot.name())
            }
            Self::WeightOwnerIdentityExhausted => {
                formatter.write_str("uploaded CUDA weight owner identities are exhausted")
            }
            Self::ForeignPhysicalWeightId => {
                formatter.write_str("physical CUDA weight ID belongs to a different uploaded model")
            }
            Self::MissingPhysicalWeight { index } => {
                write!(
                    formatter,
                    "uploaded CUDA weights omit physical index {index}"
                )
            }
        }
    }
}

impl error::Error for CudaWeightUploadError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Cuda(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ModelError> for CudaWeightUploadError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

impl From<CudaError> for CudaWeightUploadError {
    fn from(error: CudaError) -> Self {
        Self::Cuda(error)
    }
}

/// One unique checkpoint tensor and its opaque device allocation.
pub struct CudaUploadedTensor {
    source: TensorSource,
    dtype: CudaDType,
    shape: Vec<usize>,
    byte_len: u64,
    /// `None` once the device copy was released because a merged copy of the
    /// same bytes replaced it (see `release_separate_gate_up`).
    buffer: Option<CudaDeviceBuffer>,
}

impl CudaUploadedTensor {
    /// Verified physical checkpoint identity.
    #[must_use]
    pub const fn source(&self) -> &TensorSource {
        &self.source
    }

    /// CUDA storage dtype copied without conversion.
    #[must_use]
    pub const fn dtype(&self) -> CudaDType {
        self.dtype
    }

    /// Canonical tensor dimensions.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Physical device bytes.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

impl fmt::Debug for CudaUploadedTensor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaUploadedTensor")
            .field("tensor_name", &self.source.tensor_name())
            .field("shard_path", &self.source.shard_path())
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("byte_len", &self.byte_len)
            .field("released", &self.buffer.is_none())
            .finish()
    }
}

/// Borrowed execution-facing view of one uploaded canonical weight.
#[derive(Debug)]
pub struct CudaUploadedWeight<'a> {
    slot: WeightSlot,
    tensor: &'a CudaUploadedTensor,
    span: CudaBufferSpan<'a>,
}

/// Direct-indexed physical weight view used only by the owning forward path.
#[allow(dead_code)]
pub(crate) struct CudaPhysicalWeight<'a> {
    tensor: &'a CudaUploadedTensor,
    span: CudaBufferSpan<'a>,
}

#[allow(dead_code)]
impl<'a> CudaPhysicalWeight<'a> {
    pub(crate) const fn tensor(&self) -> &'a CudaUploadedTensor {
        self.tensor
    }

    pub(crate) const fn span(&self) -> CudaBufferSpan<'a> {
        self.span
    }
}

impl<'a> CudaUploadedWeight<'a> {
    /// Originally requested canonical slot.
    #[must_use]
    pub const fn slot(&self) -> WeightSlot {
        self.slot
    }

    /// Unique physical tensor, after resolving tied aliases.
    #[must_use]
    pub const fn tensor(&self) -> &'a CudaUploadedTensor {
        self.tensor
    }

    /// Whole-allocation immutable CUDA span.
    #[must_use]
    pub const fn span(&self) -> CudaBufferSpan<'a> {
        self.span
    }
}

/// Canonical slot mapping over deduplicated physical device allocations.
pub struct CudaUploadedWeights {
    owner_id: u64,
    physical: Vec<CudaUploadedTensor>,
    slots: BTreeMap<WeightSlot, usize>,
    total_physical_bytes: u64,
    /// Optional per-layer `Q|K|V` projection weights concatenated along the
    /// output axis for the merged-QKV projection mode.
    merged_qkv: Vec<CudaDeviceBuffer>,
    merged_gate_up: Vec<CudaDeviceBuffer>,
    negative_zero_bias: Option<CudaDeviceBuffer>,
}

impl CudaUploadedWeights {
    /// Uploads every unique physical tensor in stable canonical slot order.
    ///
    /// Tied aliases deduplicate only when their verified `(shard path, tensor
    /// name)` identities match. Equal bytes from distinct source tensors are
    /// never coalesced. One caller-budgeted pinned buffer is allocated for the
    /// cold upload and explicitly closed before success returns.
    ///
    /// # Errors
    ///
    /// Returns a model lookup, unsupported dtype, length, allocation, copy,
    /// synchronization, or staging-close error. Partial device allocations are
    /// dropped on failure.
    pub fn upload(
        weights: &LoadedWeights,
        context: &CudaContext,
        stream: &mut CudaStream,
        staging_capacity_bytes: u64,
    ) -> CudaWeightUploadResult<Self> {
        if staging_capacity_bytes == 0 {
            return Err(CudaWeightUploadError::EmptyStagingCapacity);
        }
        let owner_id = next_weight_owner_id()?;
        let mut staging = context.allocate_pinned_host_buffer(staging_capacity_bytes)?;
        let mut physical = Vec::new();
        let mut physical_indices: BTreeMap<(PathBuf, String), usize> = BTreeMap::new();
        let mut slots = BTreeMap::new();
        let mut total_physical_bytes = 0_u64;

        for &slot in weights.bindings().keys() {
            let bound = weights.view(slot)?;
            let source = bound.source();
            let key = (
                source.shard_path().to_path_buf(),
                source.tensor_name().to_owned(),
            );
            let physical_index = if let Some(&existing) = physical_indices.get(&key) {
                existing
            } else {
                let view = bound.view();
                let dtype = cuda_dtype(slot, view.dtype())?;
                let bytes = view.storage();
                let byte_len = u64::try_from(bytes.len())
                    .map_err(|_| CudaWeightUploadError::HostLengthOverflow { slot })?;
                let mut buffer = context.allocate_device_buffer(byte_len)?;
                buffer.upload_from_slice(0, bytes, &mut staging, stream)?;
                total_physical_bytes = total_physical_bytes
                    .checked_add(byte_len)
                    .ok_or(CudaWeightUploadError::TotalLengthOverflow)?;
                let index = physical.len();
                physical.push(CudaUploadedTensor {
                    source: source.clone(),
                    dtype,
                    shape: view.shape().dimensions().to_vec(),
                    byte_len,
                    buffer: Some(buffer),
                });
                physical_indices.insert(key, index);
                index
            };
            slots.insert(slot, physical_index);
        }

        staging.close()?;
        Ok(Self {
            owner_id,
            physical,
            slots,
            total_physical_bytes,
            merged_qkv: Vec::new(),
            merged_gate_up: Vec::new(),
            negative_zero_bias: None,
        })
    }

    /// Uploads, for each decoder layer, one additional device tensor holding
    /// the query, key, and value projection weights concatenated along the
    /// output axis (`[query_width + 2 * key_value_width, hidden]`). The bytes
    /// are the unmodified checkpoint bytes of the three tensors in that order.
    ///
    /// # Errors
    ///
    /// Returns a model lookup, length, allocation, copy, or staging-close
    /// error. Already merged weights are left untouched.
    pub(crate) fn upload_merged_qkv(
        &mut self,
        weights: &LoadedWeights,
        layer_count: usize,
        context: &CudaContext,
        stream: &mut CudaStream,
        staging_capacity_bytes: u64,
        release_separate: bool,
    ) -> CudaWeightUploadResult<()> {
        if !self.merged_qkv.is_empty() {
            return Ok(());
        }
        if staging_capacity_bytes == 0 {
            return Err(CudaWeightUploadError::EmptyStagingCapacity);
        }
        let mut staging = context.allocate_pinned_host_buffer(staging_capacity_bytes)?;
        let mut merged = Vec::with_capacity(layer_count);
        for layer in 0..layer_count {
            let parts = [
                riley_model::DecoderWeight::QueryWeight,
                riley_model::DecoderWeight::KeyWeight,
                riley_model::DecoderWeight::ValueWeight,
            ];
            let mut total = 0_u64;
            for parameter in parts {
                let slot = WeightSlot::Decoder { layer, parameter };
                let bytes = weights.view(slot)?.view().storage().len();
                total = u64::try_from(bytes)
                    .ok()
                    .and_then(|bytes| total.checked_add(bytes))
                    .ok_or(CudaWeightUploadError::TotalLengthOverflow)?;
            }
            let mut buffer = context.allocate_device_buffer(total)?;
            let mut offset = 0_u64;
            for parameter in parts {
                let slot = WeightSlot::Decoder { layer, parameter };
                let bound = weights.view(slot)?;
                let bytes = bound.view().storage();
                buffer.upload_from_slice(offset, bytes, &mut staging, stream)?;
                offset += u64::try_from(bytes.len())
                    .map_err(|_| CudaWeightUploadError::HostLengthOverflow { slot })?;
            }
            merged.push(buffer);
            // Releasing layer by layer keeps the peak at one extra layer
            // instead of a second copy of every layer.
            if release_separate {
                self.release_layer_separate(layer, &parts)?;
            }
        }
        staging.close()?;
        self.merged_qkv = merged;
        Ok(())
    }

    /// Uploads a BF16 vector of `element_count` negative zeros.
    ///
    /// It stands for the Q/K/V biases of a model that has none when its
    /// merged Q|K|V projection goes through the split-with-bias kernel:
    /// `x + (-0.0)` is `x` bit for bit for every `x`, including both zeros.
    ///
    /// # Errors
    ///
    /// Returns a length, allocation, copy, or staging-close error. An already
    /// uploaded vector that is long enough is left untouched.
    pub(crate) fn upload_negative_zero_bias(
        &mut self,
        element_count: usize,
        context: &CudaContext,
        stream: &mut CudaStream,
    ) -> CudaWeightUploadResult<()> {
        let byte_len = element_count
            .checked_mul(2)
            .ok_or(CudaWeightUploadError::TotalLengthOverflow)?;
        let byte_len_u64 =
            u64::try_from(byte_len).map_err(|_| CudaWeightUploadError::TotalLengthOverflow)?;
        if self
            .negative_zero_bias
            .as_ref()
            .is_some_and(|buffer| buffer.byte_len() >= byte_len_u64)
            || byte_len == 0
        {
            return Ok(());
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_len)
            .map_err(|_| CudaWeightUploadError::TotalLengthOverflow)?;
        for _ in 0..element_count {
            bytes.extend_from_slice(&0x8000_u16.to_ne_bytes());
        }
        let mut staging = context.allocate_pinned_host_buffer(byte_len_u64)?;
        let mut buffer = context.allocate_device_buffer(byte_len_u64)?;
        buffer.upload_from_slice(0, &bytes, &mut staging, stream)?;
        staging.close()?;
        if let Some(previous) = self.negative_zero_bias.replace(buffer) {
            previous.close()?;
        }
        Ok(())
    }

    /// The negative-zero bias vector, when uploaded.
    pub(crate) fn negative_zero_bias(&self) -> Option<&CudaDeviceBuffer> {
        self.negative_zero_bias.as_ref()
    }

    /// Releases the separate device copies of every layer's gate and up
    /// projection weights after their merged copies were uploaded, so the
    /// merged MLP projection does not hold both. A slot whose tensor is shared
    /// with another slot is kept. Viewing a released weight fails, which
    /// keeps any path that still wants the separate projections fail-closed.
    ///
    /// # Errors
    ///
    /// Returns the first CUDA close failure; without merged copies nothing is
    /// released.
    pub(crate) fn release_separate_gate_up(&mut self, layer_count: usize) -> CudaResult<()> {
        if self.merged_gate_up.len() != layer_count {
            return Ok(());
        }
        self.release_separate(
            layer_count,
            &[
                riley_model::DecoderWeight::GateWeight,
                riley_model::DecoderWeight::UpWeight,
            ],
        )
    }

    fn release_separate(
        &mut self,
        layer_count: usize,
        parameters: &[riley_model::DecoderWeight],
    ) -> CudaResult<()> {
        let mut first_error = None;
        for layer in 0..layer_count {
            if let Err(error) = self.release_layer_separate(layer, parameters) {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Releases the device tensors of `parameters` in one layer, unless
    /// another slot shares the tensor.
    fn release_layer_separate(
        &mut self,
        layer: usize,
        parameters: &[riley_model::DecoderWeight],
    ) -> CudaResult<()> {
        let mut first_error = None;
        for &parameter in parameters {
            let slot = WeightSlot::Decoder { layer, parameter };
            let Some(&index) = self.slots.get(&slot) else {
                continue;
            };
            if self.slots.values().filter(|other| **other == index).count() != 1 {
                continue;
            }
            if let Some(buffer) = self.physical[index].buffer.take() {
                self.total_physical_bytes = self
                    .total_physical_bytes
                    .saturating_sub(self.physical[index].byte_len);
                if let Err(error) = buffer.close() {
                    first_error.get_or_insert(error);
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// The merged gate|up projection weight of one layer, when uploaded.
    pub(crate) fn merged_gate_up(&self, layer: usize) -> Option<&CudaDeviceBuffer> {
        self.merged_gate_up.get(layer)
    }

    /// Uploads, for every decoder layer, one device buffer holding the MLP
    /// gate and up projection weights concatenated along the output axis
    /// (`[2 * intermediate, hidden]`), as unmodified checkpoint bytes.
    ///
    /// # Errors
    ///
    /// Returns a model lookup, length, allocation, copy, or staging-close
    /// error. Already merged weights are left untouched.
    pub(crate) fn upload_merged_gate_up(
        &mut self,
        weights: &LoadedWeights,
        layer_count: usize,
        context: &CudaContext,
        stream: &mut CudaStream,
        staging_capacity_bytes: u64,
        release_separate: bool,
    ) -> CudaWeightUploadResult<()> {
        if !self.merged_gate_up.is_empty() {
            return Ok(());
        }
        if staging_capacity_bytes == 0 {
            return Err(CudaWeightUploadError::EmptyStagingCapacity);
        }
        let mut staging = context.allocate_pinned_host_buffer(staging_capacity_bytes)?;
        let mut merged = Vec::with_capacity(layer_count);
        for layer in 0..layer_count {
            let parts = [
                riley_model::DecoderWeight::GateWeight,
                riley_model::DecoderWeight::UpWeight,
            ];
            let mut total = 0_u64;
            for parameter in parts {
                let slot = WeightSlot::Decoder { layer, parameter };
                let bytes = weights.view(slot)?.view().storage().len();
                total = u64::try_from(bytes)
                    .ok()
                    .and_then(|bytes| total.checked_add(bytes))
                    .ok_or(CudaWeightUploadError::TotalLengthOverflow)?;
            }
            let mut buffer = context.allocate_device_buffer(total)?;
            let mut offset = 0_u64;
            for parameter in parts {
                let slot = WeightSlot::Decoder { layer, parameter };
                let bound = weights.view(slot)?;
                let bytes = bound.view().storage();
                buffer.upload_from_slice(offset, bytes, &mut staging, stream)?;
                offset += u64::try_from(bytes.len())
                    .map_err(|_| CudaWeightUploadError::HostLengthOverflow { slot })?;
            }
            merged.push(buffer);
            // Releasing layer by layer keeps the peak at one extra layer
            // instead of a second copy of every layer.
            if release_separate {
                self.release_layer_separate(layer, &parts)?;
            }
        }
        staging.close()?;
        self.merged_gate_up = merged;
        Ok(())
    }

    /// The merged `Q|K|V` projection weight of one layer, when uploaded.
    pub(crate) fn merged_qkv(&self, layer: usize) -> Option<&CudaDeviceBuffer> {
        self.merged_qkv.get(layer)
    }

    /// Number of unique physical CUDA allocations.
    #[must_use]
    pub fn physical_tensor_count(&self) -> usize {
        self.physical.len()
    }

    /// Sum of unique physical tensor bytes.
    #[must_use]
    pub const fn total_physical_bytes(&self) -> u64 {
        self.total_physical_bytes
    }

    /// Stable physical index backing a canonical slot.
    #[must_use]
    pub fn physical_index(&self, slot: WeightSlot) -> Option<usize> {
        self.slots.get(&slot).copied()
    }

    /// Unique physical tensors in first-canonical-use order.
    #[must_use]
    pub fn physical_tensors(&self) -> &[CudaUploadedTensor] {
        &self.physical
    }

    /// Resolves a canonical slot once during cold plan construction.
    pub(crate) fn resolve_slot(&self, slot: WeightSlot) -> Option<PhysicalWeightId> {
        self.slots
            .get(&slot)
            .copied()
            .map(|index| PhysicalWeightId::new(self.owner_id, index))
    }

    /// Returns cold shape/dtype/byte metadata for an owner-bound physical ID.
    pub(crate) fn physical_metadata(
        &self,
        id: PhysicalWeightId,
    ) -> Option<PhysicalWeightMetadata<'_>> {
        if id.owner() != self.owner_id {
            return None;
        }
        let tensor = self.physical.get(id.index())?;
        let dtype = match tensor.dtype {
            CudaDType::BF16 => DType::BF16,
            CudaDType::F32 => DType::F32,
            _ => return None,
        };
        Some(PhysicalWeightMetadata {
            dtype,
            shape: &tensor.shape,
            byte_len: tensor.byte_len(),
        })
    }

    /// Directly indexes an already-bound physical tensor without a map lookup.
    ///
    /// The owner cookie rejects IDs from another uploaded model. The final
    /// forward owner must keep this collection and its plan together.
    #[allow(dead_code)]
    pub(crate) fn view_physical(
        &self,
        id: PhysicalWeightId,
    ) -> CudaWeightUploadResult<CudaPhysicalWeight<'_>> {
        if id.owner() != self.owner_id {
            return Err(CudaWeightUploadError::ForeignPhysicalWeightId);
        }
        let tensor = self
            .physical
            .get(id.index())
            .ok_or(CudaWeightUploadError::MissingPhysicalWeight { index: id.index() })?;
        let buffer = tensor
            .buffer
            .as_ref()
            .ok_or(CudaWeightUploadError::MissingPhysicalWeight { index: id.index() })?;
        let span = CudaBufferSpan::new(buffer, tensor.dtype, 0, buffer.byte_len())?;
        Ok(CudaPhysicalWeight { tensor, span })
    }

    /// Unique physical parents for a cold aggregate lease reservation. Tied
    /// logical slots already resolve to the same entry in this owner.
    #[cfg(feature = "cuda")]
    pub(crate) fn borrow_graph_weight_parents(
        &mut self,
    ) -> impl Iterator<Item = &mut CudaDeviceBuffer> {
        self.physical
            .iter_mut()
            .filter_map(|tensor| tensor.buffer.as_mut())
    }

    /// Cold graph binding of one actual physical weight. The exclusive borrow
    /// prevents the uploaded owner or tied aliases from being used during capture.
    #[cfg(feature = "cuda")]
    pub(crate) fn borrow_graph_weight(
        &mut self,
        id: PhysicalWeightId,
    ) -> CudaWeightUploadResult<&mut CudaDeviceBuffer> {
        // Preserve the existing owner-ID check and dtype/span validation.
        self.view_physical(id)?;
        self.physical[id.index()]
            .buffer
            .as_mut()
            .ok_or(CudaWeightUploadError::MissingPhysicalWeight { index: id.index() })
    }

    /// Resolves a canonical slot into its immutable whole-allocation span.
    ///
    /// # Errors
    ///
    /// Returns an internal mapping or CUDA span-contract error.
    pub fn view(&self, slot: WeightSlot) -> CudaWeightUploadResult<CudaUploadedWeight<'_>> {
        let id = self
            .resolve_slot(slot)
            .ok_or(CudaWeightUploadError::MissingSlot { slot })?;
        let physical = self.view_physical(id)?;
        Ok(CudaUploadedWeight {
            slot,
            tensor: physical.tensor,
            span: physical.span,
        })
    }

    /// Explicitly closes every physical allocation, attempting all closes even
    /// after the first failure.
    ///
    /// # Errors
    ///
    /// Returns the first CUDA close failure after attempting the full set.
    pub fn close(self) -> CudaResult<()> {
        let mut first_error = None;
        for buffer in self.physical.into_iter().filter_map(|tensor| tensor.buffer) {
            if let Err(error) = buffer.close() {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        for buffer in self.merged_gate_up {
            if let Err(error) = buffer.close() {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        if let Some(buffer) = self.negative_zero_bias {
            if let Err(error) = buffer.close() {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        for buffer in self.merged_qkv {
            if let Err(error) = buffer.close() {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

impl fmt::Debug for CudaUploadedWeights {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaUploadedWeights")
            .field("owner_id", &self.owner_id)
            .field("physical", &self.physical)
            .field("slots", &self.slots)
            .field("total_physical_bytes", &self.total_physical_bytes)
            .finish()
    }
}

fn next_weight_owner_id() -> CudaWeightUploadResult<u64> {
    NEXT_WEIGHT_OWNER_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| CudaWeightUploadError::WeightOwnerIdentityExhausted)
}

fn cuda_dtype(slot: WeightSlot, dtype: DType) -> CudaWeightUploadResult<CudaDType> {
    match dtype {
        DType::BF16 => Ok(CudaDType::BF16),
        DType::F32 => Ok(CudaDType::F32),
        dtype => Err(CudaWeightUploadError::UnsupportedDType { slot, dtype }),
    }
}
