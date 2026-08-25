use std::collections::BTreeMap;
use std::error;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rustinfer_cuda::{
    CudaBufferSpan, CudaContext, CudaDType, CudaDeviceBuffer, CudaError, CudaResult, CudaStream,
};
use rustinfer_model::{LoadedWeights, ModelError, TensorSource, WeightSlot};
use rustinfer_tensor::DType;

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
    buffer: CudaDeviceBuffer,
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
        self.buffer.byte_len()
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
            .field("byte_len", &self.buffer.byte_len())
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
                    buffer,
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
        })
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
            CudaDType::U8 | CudaDType::U32 => return None,
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
        let span = CudaBufferSpan::new(&tensor.buffer, tensor.dtype, 0, tensor.buffer.byte_len())?;
        Ok(CudaPhysicalWeight { tensor, span })
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
        for tensor in self.physical {
            if let Err(error) = tensor.buffer.close() {
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
