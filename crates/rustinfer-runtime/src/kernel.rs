use std::error;
use std::fmt;

use rustinfer_tensor::DType;

/// Stable operation identifiers used by the minimal kernel registry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum OpId {
    /// Token embedding table gather.
    Embedding,
    /// Root-mean-square normalization.
    RmsNorm,
    /// Elementwise residual addition.
    ResidualAdd,
    /// Elementwise `SiLU` activation.
    Silu,
    /// Elementwise multiplication of an activated gate and value.
    GatedMultiply,
    /// Standard, non-interleaved Llama rotary position embedding.
    Rope,
    /// Explicit dtype conversion.
    Cast,
    /// Dense matrix multiplication.
    Gemm,
}

/// The exact dtype signature used to look up one operation implementation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KernelKey {
    op: OpId,
    input_dtype: DType,
    weight_dtype: Option<DType>,
    output_dtype: DType,
}

impl KernelKey {
    /// Constructs an exact operation and dtype signature.
    #[must_use]
    pub const fn new(
        op: OpId,
        input_dtype: DType,
        weight_dtype: Option<DType>,
        output_dtype: DType,
    ) -> Self {
        Self {
            op,
            input_dtype,
            weight_dtype,
            output_dtype,
        }
    }

    /// Returns the operation identifier.
    #[must_use]
    pub const fn op(self) -> OpId {
        self.op
    }

    /// Returns the primary input dtype.
    #[must_use]
    pub const fn input_dtype(self) -> DType {
        self.input_dtype
    }

    /// Returns the optional weight or secondary-input dtype.
    #[must_use]
    pub const fn weight_dtype(self) -> Option<DType> {
        self.weight_dtype
    }

    /// Returns the output dtype.
    #[must_use]
    pub const fn output_dtype(self) -> DType {
        self.output_dtype
    }
}

/// Capabilities which affect whether an implementation is safe to select.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelCapability {
    key: KernelKey,
    deterministic: bool,
    requires_contiguous: bool,
}

impl KernelCapability {
    /// Describes an implementation's exact dtype and execution properties.
    #[must_use]
    pub const fn new(key: KernelKey, deterministic: bool, requires_contiguous: bool) -> Self {
        Self {
            key,
            deterministic,
            requires_contiguous,
        }
    }

    /// Returns the exact registry key supported by this implementation.
    #[must_use]
    pub const fn key(self) -> KernelKey {
        self.key
    }

    /// Returns whether identical inputs have a deterministic execution path.
    #[must_use]
    pub const fn deterministic(self) -> bool {
        self.deterministic
    }

    /// Returns whether all inputs and outputs must be contiguous.
    #[must_use]
    pub const fn requires_contiguous(self) -> bool {
        self.requires_contiguous
    }
}

/// Provenance of a kernel implementation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum KernelOrigin {
    /// A native CUDA C++ kernel behind the stable C ABI.
    CudaCpp,
    /// A cuBLASLt adapter and selected algorithm.
    CuBlasLt,
    /// An approved CUTLASS implementation.
    Cutlass,
    /// Another native, non-Python implementation.
    ExternalNative,
    /// A prototype implemented with Triton Python JIT.
    ExperimentalTriton,
    /// The allocation-free CPU correctness reference.
    ReferenceCpu,
}

/// A named implementation and the capability it provides.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelImplementation {
    id: &'static str,
    origin: KernelOrigin,
    capability: KernelCapability,
}

impl KernelImplementation {
    /// Constructs immutable implementation metadata.
    #[must_use]
    pub const fn new(id: &'static str, origin: KernelOrigin, capability: KernelCapability) -> Self {
        Self {
            id,
            origin,
            capability,
        }
    }

    /// Returns the stable implementation identifier used in metrics.
    #[must_use]
    pub const fn id(self) -> &'static str {
        self.id
    }

    /// Returns the implementation's provenance.
    #[must_use]
    pub const fn origin(self) -> KernelOrigin {
        self.origin
    }

    /// Returns the implementation capability.
    #[must_use]
    pub const fn capability(self) -> KernelCapability {
        self.capability
    }
}

/// Runtime selection preference for the exact reference or an optimized path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelPreference {
    /// Require the CPU reference implementation.
    Reference,
    /// Prefer a production optimized implementation, falling back to reference.
    Optimized,
}

/// A minimal deterministic registry for production kernel selection.
#[derive(Debug, Default)]
pub struct KernelRegistry {
    implementations: Vec<KernelImplementation>,
}

impl KernelRegistry {
    /// Constructs an empty registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            implementations: Vec::new(),
        }
    }

    /// Registers one uniquely named implementation for its exact key.
    ///
    /// Experimental Triton metadata may be registered for comparison, but
    /// [`Self::select`] never returns it.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty identifier or a duplicate `(key, id)`.
    pub fn register(
        &mut self,
        implementation: KernelImplementation,
    ) -> Result<(), KernelRegistryError> {
        if implementation.id.is_empty() {
            return Err(KernelRegistryError::EmptyImplementationId);
        }
        if self.implementations.iter().any(|registered| {
            registered.id == implementation.id
                && registered.capability.key == implementation.capability.key
        }) {
            return Err(KernelRegistryError::DuplicateImplementationId {
                id: implementation.id,
            });
        }
        self.implementations.push(implementation);
        Ok(())
    }

    /// Returns all registered metadata, including experimental implementations.
    #[must_use]
    pub fn implementations(&self) -> &[KernelImplementation] {
        &self.implementations
    }

    /// Selects a deterministic production implementation for an exact key.
    ///
    /// Selection is independent of registration order. Optimized selection
    /// prefers the PR 06 production origin for the operation and falls back to
    /// the CPU reference. `ExperimentalTriton` is never production-selectable.
    ///
    /// # Errors
    ///
    /// Returns an error when no deterministic production implementation
    /// satisfies the requested key and preference.
    pub fn select(
        &self,
        key: KernelKey,
        preference: KernelPreference,
    ) -> Result<&KernelImplementation, KernelRegistryError> {
        let candidate = self
            .implementations
            .iter()
            .filter(|implementation| implementation.capability.key == key)
            .filter(|implementation| implementation.capability.deterministic)
            .filter(|implementation| implementation.origin != KernelOrigin::ExperimentalTriton)
            .filter(|implementation| match preference {
                KernelPreference::Reference => implementation.origin == KernelOrigin::ReferenceCpu,
                KernelPreference::Optimized => true,
            })
            .min_by_key(|implementation| selection_rank(**implementation, preference));

        candidate.ok_or_else(|| {
            let experimental_only = preference == KernelPreference::Optimized
                && self.implementations.iter().any(|implementation| {
                    implementation.capability.key == key
                        && implementation.capability.deterministic
                        && implementation.origin == KernelOrigin::ExperimentalTriton
                })
                && self.implementations.iter().all(|implementation| {
                    implementation.capability.key != key
                        || !implementation.capability.deterministic
                        || implementation.origin == KernelOrigin::ExperimentalTriton
                });
            if experimental_only {
                KernelRegistryError::ExperimentalTritonRejected { key }
            } else {
                KernelRegistryError::NoDeterministicImplementation { key, preference }
            }
        })
    }
}

fn selection_rank(
    implementation: KernelImplementation,
    preference: KernelPreference,
) -> (u8, u8, &'static str) {
    let reference_rank = u8::from(
        preference == KernelPreference::Optimized
            && implementation.origin == KernelOrigin::ReferenceCpu,
    );
    let origin_rank = match implementation.capability.key.op {
        OpId::Gemm => match implementation.origin {
            KernelOrigin::CuBlasLt => 0,
            KernelOrigin::Cutlass => 1,
            KernelOrigin::CudaCpp => 2,
            KernelOrigin::ExternalNative => 3,
            KernelOrigin::ReferenceCpu => 4,
            KernelOrigin::ExperimentalTriton => 5,
        },
        _ => match implementation.origin {
            KernelOrigin::CudaCpp => 0,
            KernelOrigin::ExternalNative => 1,
            KernelOrigin::CuBlasLt => 2,
            KernelOrigin::Cutlass => 3,
            KernelOrigin::ReferenceCpu => 4,
            KernelOrigin::ExperimentalTriton => 5,
        },
    };
    (reference_rank, origin_rank, implementation.id)
}

/// A stable kernel registration or production-selection failure.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum KernelRegistryError {
    /// An implementation identifier was empty.
    EmptyImplementationId,
    /// An implementation identifier was already registered for the same key.
    DuplicateImplementationId {
        /// The duplicate stable identifier.
        id: &'static str,
    },
    /// No deterministic production implementation matched.
    NoDeterministicImplementation {
        /// The exact requested operation signature.
        key: KernelKey,
        /// The requested reference or optimized policy.
        preference: KernelPreference,
    },
    /// Matching implementations existed only in the experimental Triton origin.
    ExperimentalTritonRejected {
        /// The exact requested operation signature.
        key: KernelKey,
    },
}

impl fmt::Display for KernelRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyImplementationId => formatter.write_str("kernel implementation ID is empty"),
            Self::DuplicateImplementationId { id } => {
                write!(
                    formatter,
                    "kernel implementation ID `{id}` is already registered"
                )
            }
            Self::NoDeterministicImplementation { key, preference } => write!(
                formatter,
                "no deterministic production kernel for {key:?} with {preference:?} preference"
            ),
            Self::ExperimentalTritonRejected { key } => write!(
                formatter,
                "ExperimentalTriton is not production-selectable for {key:?}"
            ),
        }
    }
}

impl error::Error for KernelRegistryError {}

#[cfg(test)]
mod tests {
    use super::*;

    const GEMM_F32: KernelKey =
        KernelKey::new(OpId::Gemm, DType::F32, Some(DType::F32), DType::F32);

    fn implementation(
        id: &'static str,
        origin: KernelOrigin,
        deterministic: bool,
    ) -> KernelImplementation {
        KernelImplementation::new(
            id,
            origin,
            KernelCapability::new(GEMM_F32, deterministic, true),
        )
    }

    #[test]
    fn reference_preference_requires_reference_origin() {
        let mut registry = KernelRegistry::new();
        registry
            .register(implementation(
                "reference",
                KernelOrigin::ReferenceCpu,
                true,
            ))
            .unwrap();
        registry
            .register(implementation("cublas", KernelOrigin::CuBlasLt, true))
            .unwrap();

        let selected = registry
            .select(GEMM_F32, KernelPreference::Reference)
            .unwrap();
        assert_eq!(selected.id(), "reference");
    }

    #[test]
    fn optimized_selection_is_deterministic_and_falls_back_to_reference() {
        let mut first = KernelRegistry::new();
        first
            .register(implementation("cutlass", KernelOrigin::Cutlass, true))
            .unwrap();
        first
            .register(implementation("cublas-z", KernelOrigin::CuBlasLt, true))
            .unwrap();
        first
            .register(implementation("cublas-a", KernelOrigin::CuBlasLt, true))
            .unwrap();

        let mut reversed = KernelRegistry::new();
        for implementation in first.implementations().iter().rev().copied() {
            reversed.register(implementation).unwrap();
        }

        for registry in [&first, &reversed] {
            let selected = registry
                .select(GEMM_F32, KernelPreference::Optimized)
                .unwrap();
            assert_eq!(selected.id(), "cublas-a");
        }

        let mut reference_only = KernelRegistry::new();
        reference_only
            .register(implementation(
                "reference",
                KernelOrigin::ReferenceCpu,
                true,
            ))
            .unwrap();
        assert_eq!(
            reference_only
                .select(GEMM_F32, KernelPreference::Optimized)
                .unwrap()
                .origin(),
            KernelOrigin::ReferenceCpu
        );
    }

    #[test]
    fn production_selection_rejects_triton_and_nondeterminism() {
        let mut registry = KernelRegistry::new();
        registry
            .register(implementation(
                "triton",
                KernelOrigin::ExperimentalTriton,
                true,
            ))
            .unwrap();
        assert!(matches!(
            registry.select(GEMM_F32, KernelPreference::Optimized),
            Err(KernelRegistryError::ExperimentalTritonRejected { .. })
        ));

        registry
            .register(implementation("cublas", KernelOrigin::CuBlasLt, false))
            .unwrap();
        assert!(matches!(
            registry.select(GEMM_F32, KernelPreference::Optimized),
            Err(KernelRegistryError::ExperimentalTritonRejected { .. })
        ));
    }

    #[test]
    fn registration_rejects_empty_and_duplicate_ids() {
        let mut registry = KernelRegistry::new();
        assert_eq!(
            registry.register(implementation("", KernelOrigin::ReferenceCpu, true)),
            Err(KernelRegistryError::EmptyImplementationId)
        );
        registry
            .register(implementation("same", KernelOrigin::ReferenceCpu, true))
            .unwrap();
        assert!(matches!(
            registry.register(implementation("same", KernelOrigin::CuBlasLt, true)),
            Err(KernelRegistryError::DuplicateImplementationId { id: "same" })
        ));

        let cast_key = KernelKey::new(OpId::Cast, DType::F32, None, DType::BF16);
        registry
            .register(KernelImplementation::new(
                "same",
                KernelOrigin::CudaCpp,
                KernelCapability::new(cast_key, true, true),
            ))
            .unwrap();
    }
}
