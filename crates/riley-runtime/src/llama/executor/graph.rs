//! Closed and allocation-free execution-graph dispatch policy.
//!
//! This C06-0/C06-1 module owns neither a CUDA Graph nor a model executor.
//! It turns already-observed scalar eligibility and inventory facts into one
//! of three modes, or a fail-closed `require` rejection, and defines an
//! immutable value-only cache identity. Native graph lookup and runtime
//! wiring remain separate follow-up slices.

use std::error;
use std::fmt;

use sha2::{Digest, Sha256};

/// Operator-specific graph-capture admission result supplied to the dispatcher.
///
/// This is a runtime policy value, not a CUDA ABI handle. A later C06 adapter
/// maps the C05 native capability record into this closed vocabulary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphOperatorCapability {
    /// No reviewed capture-safety result is available for the selected work.
    #[default]
    Unknown,
    /// At least one selected operator is not capture-safe.
    Unsupported,
    /// Every selected operator is admitted for the requested graph mode.
    Supported,
}

/// Workload stage considered by the C06 execution-graph policy.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum GraphWorkloadStage {
    /// A prefill-only iteration.
    Prefill,
    /// A pure decode iteration.
    PureDecode,
    /// An iteration that mixes prefill and decode work.
    Mixed,
    /// A stage with no reviewed execution-graph policy.
    Unsupported,
}

/// Sampling/output backend considered for full-graph admission.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum GraphSamplingBackend {
    /// The fixed GPU greedy backend required by the initial full-graph path.
    GpuGreedy,
    /// Any other or unreviewed sampling/output backend.
    ///
    /// It can remain an eager boundary around a piecewise graph, but cannot be
    /// part of a full graph replay.
    Unsupported,
}

/// Schema version embedded in every graph-cache identity.
///
/// Increment this value whenever equality-relevant signature meaning changes.
/// Cold graph inventory never reuses entries from a different schema version.
pub const GRAPH_SIGNATURE_SCHEMA_VERSION: u16 = 1;

/// Closed model topology family accepted by the initial graph cache.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum GraphModelArchitecture {
    /// The canonical dense Llama decoder topology, including dense Qwen2 IR.
    LlamaDecoder,
}

/// Closed tensor storage dtype carried by a graph signature.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum GraphDataType {
    /// IEEE 16-bit floating point storage.
    Float16,
    /// Brain floating-point 16-bit storage.
    BFloat16,
    /// IEEE 32-bit floating point storage.
    Float32,
}

/// Closed compute/reduction type carried by a graph signature.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum GraphComputeType {
    /// IEEE 32-bit floating point accumulation.
    Float32,
    /// TensorFloat-32 Tensor Core compute.
    TensorFloat32,
}

/// Fixed-size canonical model revision fingerprint.
///
/// The fingerprint is prepared on a cold path from a canonical model/config
/// representation. It deliberately contains no model pointer, path, or
/// process-unique address.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphRevisionFingerprint([u8; 32]);

impl GraphRevisionFingerprint {
    const DOMAIN_SEPARATOR: &[u8] = b"riley.graph-revision-fingerprint.v1\0";

    /// Wraps a reviewed fixed-width fingerprint without allocation.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Hashes canonical cold-path bytes with the fixed graph-revision domain.
    #[must_use]
    pub fn from_canonical_bytes(canonical_bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(Self::DOMAIN_SEPARATOR);
        hasher.update(canonical_bytes);
        Self(hasher.finalize().into())
    }

    /// Returns the fixed-width digest for logging-free identity composition.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Immutable model and uploaded-weight layout identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphModelSignature {
    architecture: GraphModelArchitecture,
    architecture_revision: u32,
    model_revision: GraphRevisionFingerprint,
    weight_layout_revision: u32,
}

impl GraphModelSignature {
    /// Creates the static identity for one canonical model topology and layout.
    #[must_use]
    pub const fn new(
        architecture: GraphModelArchitecture,
        architecture_revision: u32,
        model_revision: GraphRevisionFingerprint,
        weight_layout_revision: u32,
    ) -> Self {
        Self {
            architecture,
            architecture_revision,
            model_revision,
            weight_layout_revision,
        }
    }
}

/// CUDA/native provenance that changes graph replay compatibility.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphDeviceSignature {
    compute_capability_major: u32,
    compute_capability_minor: u32,
    cuda_runtime_version: u32,
    cublaslt_version: u32,
    native_abi_version: u32,
}

impl GraphDeviceSignature {
    /// Creates a stable device/runtime identity from cold runtime metadata.
    #[must_use]
    pub const fn new(
        compute_capability_major: u32,
        compute_capability_minor: u32,
        cuda_runtime_version: u32,
        cublaslt_version: u32,
        native_abi_version: u32,
    ) -> Self {
        Self {
            compute_capability_major,
            compute_capability_minor,
            cuda_runtime_version,
            cublaslt_version,
            native_abi_version,
        }
    }
}

/// Exact activation, weight, and accumulator types for a graph chain.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphTensorSignature {
    activation_dtype: GraphDataType,
    weight_dtype: GraphDataType,
    compute_type: GraphComputeType,
}

impl GraphTensorSignature {
    /// Creates a closed tensor and compute-type identity.
    #[must_use]
    pub const fn new(
        activation_dtype: GraphDataType,
        weight_dtype: GraphDataType,
        compute_type: GraphComputeType,
    ) -> Self {
        Self {
            activation_dtype,
            weight_dtype,
            compute_type,
        }
    }
}

/// Model geometry that fixes graph-buffer and kernel launch shapes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphGeometrySignature {
    layer_count: u32,
    hidden_size: u32,
    intermediate_size: u32,
    vocabulary_size: u32,
    query_heads: u32,
    key_value_heads: u32,
    head_dimension: u32,
}

impl GraphGeometrySignature {
    /// Creates the exact model dimensions used by graph preparation.
    #[must_use]
    pub const fn new(
        layer_count: u32,
        hidden_size: u32,
        intermediate_size: u32,
        vocabulary_size: u32,
        query_heads: u32,
        key_value_heads: u32,
        head_dimension: u32,
    ) -> Self {
        Self {
            layer_count,
            hidden_size,
            intermediate_size,
            vocabulary_size,
            query_heads,
            key_value_heads,
            head_dimension,
        }
    }
}

/// Canonical packed-metadata layout identity.
///
/// The digest is prepared on a cold path from schema version, field offsets,
/// field sizes, and alignments. It does not contain an allocation address.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphMetadataLayoutSignature {
    schema_version: u32,
    digest: [u8; 32],
}

impl GraphMetadataLayoutSignature {
    /// Creates the exact metadata schema and canonical-layout digest identity.
    #[must_use]
    pub const fn new(schema_version: u32, digest: [u8; 32]) -> Self {
        Self {
            schema_version,
            digest,
        }
    }

    /// Returns the packed-metadata schema version.
    #[must_use]
    pub const fn schema_version(self) -> u32 {
        self.schema_version
    }

    /// Returns the canonical fixed-width layout digest.
    #[must_use]
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }
}

/// Fixed KV and packed-metadata layout identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphLayoutSignature {
    maximum_sequence_tokens: u32,
    kv_page_size: u32,
    kv_layout_version: u32,
    metadata_layout: GraphMetadataLayoutSignature,
}

impl GraphLayoutSignature {
    /// Creates the cold layout identity for graph-owned fixed addresses.
    #[must_use]
    pub const fn new(
        maximum_sequence_tokens: u32,
        kv_page_size: u32,
        kv_layout_version: u32,
        metadata_layout: GraphMetadataLayoutSignature,
    ) -> Self {
        Self {
            maximum_sequence_tokens,
            kv_page_size,
            kv_layout_version,
            metadata_layout,
        }
    }
}

/// Fixed numeric identifier for one cold-selected graph implementation.
///
/// A later cold adapter owns the mapping from executable implementation to
/// this value; strings and runtime addresses never enter this identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphImplementationId(u32);

impl GraphImplementationId {
    /// Creates one fixed implementation identifier.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }
}

/// Fixed numeric identity for the complete selected GEMM-plan set.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphGemmPlanSetId(u32);

impl GraphGemmPlanSetId {
    /// Creates one cold-selected GEMM-plan-set identifier.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }
}

/// Fixed numeric identity for the selected reduction policy.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphReductionPolicyId(u32);

impl GraphReductionPolicyId {
    /// Creates one cold-selected reduction-policy identifier.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }
}

/// Closed implementation-plan identities used by an instantiated graph.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphImplementationSignature {
    attention: GraphImplementationId,
    projections: GraphImplementationId,
    mlp: GraphImplementationId,
    output: GraphImplementationId,
    gemm_plan_set: GraphGemmPlanSetId,
    reduction_policy: GraphReductionPolicyId,
}

impl GraphImplementationSignature {
    /// Creates the exact implementation and reduction-plan identity.
    #[must_use]
    pub const fn new(
        attention: GraphImplementationId,
        projections: GraphImplementationId,
        mlp: GraphImplementationId,
        output: GraphImplementationId,
        gemm_plan_set: GraphGemmPlanSetId,
        reduction_policy: GraphReductionPolicyId,
    ) -> Self {
        Self {
            attention,
            projections,
            mlp,
            output,
            gemm_plan_set,
            reduction_policy,
        }
    }
}

/// Every cold-prepared fact shared by all iteration signatures for one model.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphStaticSignature {
    model: GraphModelSignature,
    device: GraphDeviceSignature,
    tensors: GraphTensorSignature,
    geometry: GraphGeometrySignature,
    layout: GraphLayoutSignature,
    implementations: GraphImplementationSignature,
}

impl GraphStaticSignature {
    /// Combines only fixed-width, cold-prepared graph identity facts.
    #[must_use]
    pub const fn new(
        model: GraphModelSignature,
        device: GraphDeviceSignature,
        tensors: GraphTensorSignature,
        geometry: GraphGeometrySignature,
        layout: GraphLayoutSignature,
        implementations: GraphImplementationSignature,
    ) -> Self {
        Self {
            model,
            device,
            tensors,
            geometry,
            layout,
            implementations,
        }
    }
}

/// Iteration-specific identity that changes a graph cache lookup.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphIterationSignature {
    stage: GraphWorkloadStage,
    active_row_bucket: u32,
    sampling_backend: GraphSamplingBackend,
}

impl GraphIterationSignature {
    /// Creates the allocation-free identity for one planned iteration shape.
    #[must_use]
    pub const fn new(
        stage: GraphWorkloadStage,
        active_row_bucket: u32,
        sampling_backend: GraphSamplingBackend,
    ) -> Self {
        Self {
            stage,
            active_row_bucket,
            sampling_backend,
        }
    }

    /// Returns the workload stage encoded into this exact cache key.
    #[must_use]
    pub const fn stage(self) -> GraphWorkloadStage {
        self.stage
    }

    /// Returns the sampling/output backend encoded into this exact cache key.
    #[must_use]
    pub const fn sampling_backend(self) -> GraphSamplingBackend {
        self.sampling_backend
    }
}

/// Stable SHA-256 digest of one explicit graph signature encoding.
///
/// This is a trace and cache-prefilter value only. A graph owner must still
/// require full `GraphSignature` equality before it reuses an entry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphSignatureFingerprint([u8; 32]);

impl GraphSignatureFingerprint {
    /// Returns the fixed-width digest bytes without allocation.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Complete, immutable key for one exact prepared graph cache entry.
///
/// Pointer stability is intentionally not represented here: the future graph
/// owner validates its own instantiated-buffer addresses separately.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphSignature {
    schema_version: u16,
    static_signature: GraphStaticSignature,
    iteration: GraphIterationSignature,
}

impl GraphSignature {
    /// Creates a versioned graph-cache key without allocation or I/O.
    #[must_use]
    pub const fn new(
        static_signature: GraphStaticSignature,
        iteration: GraphIterationSignature,
    ) -> Self {
        Self {
            schema_version: GRAPH_SIGNATURE_SCHEMA_VERSION,
            static_signature,
            iteration,
        }
    }

    /// Returns the equality-relevant schema version embedded in this key.
    #[must_use]
    pub const fn schema_version(self) -> u16 {
        self.schema_version
    }

    /// Returns the cold identity portion of this graph-cache key.
    #[must_use]
    pub const fn static_signature(self) -> GraphStaticSignature {
        self.static_signature
    }

    /// Returns the iteration identity portion of this graph-cache key.
    #[must_use]
    pub const fn iteration(self) -> GraphIterationSignature {
        self.iteration
    }

    /// Calculates the fixed, domain-separated cache prefilter and trace digest.
    ///
    /// This encodes each field in a declared order and never hashes raw struct
    /// bytes. Full `GraphSignature` equality remains required before graph reuse.
    #[must_use]
    pub fn fingerprint(self) -> GraphSignatureFingerprint {
        let mut hasher = Sha256::new();
        hasher.update(GRAPH_SIGNATURE_FINGERPRINT_DOMAIN);
        hasher.update(self.schema_version.to_le_bytes());
        self.static_signature.update_fingerprint(&mut hasher);
        self.iteration.update_fingerprint(&mut hasher);
        GraphSignatureFingerprint(hasher.finalize().into())
    }
}

const GRAPH_SIGNATURE_FINGERPRINT_DOMAIN: &[u8] = b"riley.graph-signature-fingerprint.v1\0";

impl GraphModelArchitecture {
    const fn fingerprint_tag(self) -> u8 {
        match self {
            Self::LlamaDecoder => 1,
        }
    }
}

impl GraphDataType {
    const fn fingerprint_tag(self) -> u8 {
        match self {
            Self::Float16 => 1,
            Self::BFloat16 => 2,
            Self::Float32 => 3,
        }
    }
}

impl GraphComputeType {
    const fn fingerprint_tag(self) -> u8 {
        match self {
            Self::Float32 => 1,
            Self::TensorFloat32 => 2,
        }
    }
}

impl GraphWorkloadStage {
    const fn fingerprint_tag(self) -> u8 {
        match self {
            Self::Prefill => 1,
            Self::PureDecode => 2,
            Self::Mixed => 3,
            Self::Unsupported => 4,
        }
    }
}

impl GraphSamplingBackend {
    const fn fingerprint_tag(self) -> u8 {
        match self {
            Self::GpuGreedy => 1,
            Self::Unsupported => 2,
        }
    }
}

impl GraphStaticSignature {
    fn update_fingerprint(self, hasher: &mut Sha256) {
        hasher.update([b'S']);
        self.model.update_fingerprint(hasher);
        self.device.update_fingerprint(hasher);
        self.tensors.update_fingerprint(hasher);
        self.geometry.update_fingerprint(hasher);
        self.layout.update_fingerprint(hasher);
        self.implementations.update_fingerprint(hasher);
    }
}

impl GraphModelSignature {
    fn update_fingerprint(self, hasher: &mut Sha256) {
        hasher.update([b'M', self.architecture.fingerprint_tag()]);
        hasher.update(self.architecture_revision.to_le_bytes());
        hasher.update(self.model_revision.as_bytes());
        hasher.update(self.weight_layout_revision.to_le_bytes());
    }
}

impl GraphDeviceSignature {
    fn update_fingerprint(self, hasher: &mut Sha256) {
        hasher.update([b'D']);
        hasher.update(self.compute_capability_major.to_le_bytes());
        hasher.update(self.compute_capability_minor.to_le_bytes());
        hasher.update(self.cuda_runtime_version.to_le_bytes());
        hasher.update(self.cublaslt_version.to_le_bytes());
        hasher.update(self.native_abi_version.to_le_bytes());
    }
}

impl GraphTensorSignature {
    fn update_fingerprint(self, hasher: &mut Sha256) {
        hasher.update([
            b'T',
            self.activation_dtype.fingerprint_tag(),
            self.weight_dtype.fingerprint_tag(),
            self.compute_type.fingerprint_tag(),
        ]);
    }
}

impl GraphGeometrySignature {
    fn update_fingerprint(self, hasher: &mut Sha256) {
        hasher.update([b'G']);
        hasher.update(self.layer_count.to_le_bytes());
        hasher.update(self.hidden_size.to_le_bytes());
        hasher.update(self.intermediate_size.to_le_bytes());
        hasher.update(self.vocabulary_size.to_le_bytes());
        hasher.update(self.query_heads.to_le_bytes());
        hasher.update(self.key_value_heads.to_le_bytes());
        hasher.update(self.head_dimension.to_le_bytes());
    }
}

impl GraphLayoutSignature {
    fn update_fingerprint(self, hasher: &mut Sha256) {
        hasher.update([b'L']);
        hasher.update(self.maximum_sequence_tokens.to_le_bytes());
        hasher.update(self.kv_page_size.to_le_bytes());
        hasher.update(self.kv_layout_version.to_le_bytes());
        hasher.update(self.metadata_layout.schema_version.to_le_bytes());
        hasher.update(self.metadata_layout.digest);
    }
}

impl GraphImplementationSignature {
    fn update_fingerprint(self, hasher: &mut Sha256) {
        hasher.update([b'I']);
        hasher.update(self.attention.0.to_le_bytes());
        hasher.update(self.projections.0.to_le_bytes());
        hasher.update(self.mlp.0.to_le_bytes());
        hasher.update(self.output.0.to_le_bytes());
        hasher.update(self.gemm_plan_set.0.to_le_bytes());
        hasher.update(self.reduction_policy.0.to_le_bytes());
    }
}

impl GraphIterationSignature {
    fn update_fingerprint(self, hasher: &mut Sha256) {
        hasher.update([
            b'R',
            self.stage.fingerprint_tag(),
            self.sampling_backend.fingerprint_tag(),
        ]);
        hasher.update(self.active_row_bucket.to_le_bytes());
    }
}

/// Operator and backend facts that must both admit graph dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphCaptureSafety {
    sampling_backend: GraphSamplingBackend,
    operator_capability: GraphOperatorCapability,
    backend_capture_safe: bool,
}

impl GraphCaptureSafety {
    /// Creates an allocation-free capture-safety fact bundle.
    #[must_use]
    pub const fn new(
        sampling_backend: GraphSamplingBackend,
        operator_capability: GraphOperatorCapability,
        backend_capture_safe: bool,
    ) -> Self {
        Self {
            sampling_backend,
            operator_capability,
            backend_capture_safe,
        }
    }
}

/// Scalar eligibility facts observed before a graph inventory lookup result is
/// applied. Every fact is caller-owned and has no CUDA side effect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphDispatchEligibility {
    stage: GraphWorkloadStage,
    active_row_bucket_supported: bool,
    metadata_layout_matches: bool,
    inventory_enabled: bool,
    capture_safety: GraphCaptureSafety,
}

impl GraphDispatchEligibility {
    /// Creates the fixed policy input for one proposed execution-graph lookup.
    #[must_use]
    pub const fn new(
        stage: GraphWorkloadStage,
        active_row_bucket_supported: bool,
        metadata_layout_matches: bool,
        inventory_enabled: bool,
        capture_safety: GraphCaptureSafety,
    ) -> Self {
        Self {
            stage,
            active_row_bucket_supported,
            metadata_layout_matches,
            inventory_enabled,
            capture_safety,
        }
    }

    /// Returns the workload stage used for graph-policy admission.
    #[must_use]
    pub const fn stage(self) -> GraphWorkloadStage {
        self.stage
    }

    /// Returns the sampling backend used for graph-policy admission.
    #[must_use]
    pub const fn sampling_backend(self) -> GraphSamplingBackend {
        self.capture_safety.sampling_backend
    }

    /// Returns this eligibility bundle with only inventory availability changed.
    #[must_use]
    pub const fn with_inventory_enabled(self, inventory_enabled: bool) -> Self {
        Self {
            inventory_enabled,
            ..self
        }
    }
}

/// Policy selected at startup or by an explicit benchmark/qualification caller.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExecutionGraphPolicy {
    /// Preserve the established exact eager path without inspecting graph facts.
    #[default]
    Disabled,
    /// Select a matching prepared graph, otherwise execute exact eager work.
    Auto,
    /// Reject an iteration without a matching prepared graph.
    Require,
}

/// One cold-prepared inventory lookup result for the exact proposed signature.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphInventoryState {
    /// No graph was prepared for the requested signature.
    NotPrepared,
    /// A complete pure-decode graph is ready for replay.
    PreparedFull,
    /// An admitted fixed segment is ready for piecewise replay.
    PreparedPiecewise,
    /// A graph entry had an unrecoverable launch/completion failure.
    Poisoned,
}

/// Complete, scalar-only input for one dispatcher decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphDispatchRequest {
    policy: ExecutionGraphPolicy,
    eligibility: GraphDispatchEligibility,
    inventory: GraphInventoryState,
}

impl GraphDispatchRequest {
    /// Combines policy, eligibility, and the exact signature's inventory state.
    #[must_use]
    pub const fn new(
        policy: ExecutionGraphPolicy,
        eligibility: GraphDispatchEligibility,
        inventory: GraphInventoryState,
    ) -> Self {
        Self {
            policy,
            eligibility,
            inventory,
        }
    }

    /// Returns the scalar eligibility facts for this dispatch request.
    #[must_use]
    pub const fn eligibility(self) -> GraphDispatchEligibility {
        self.eligibility
    }

    /// Returns the configured execution-graph policy for this request.
    #[must_use]
    pub const fn policy(self) -> ExecutionGraphPolicy {
        self.policy
    }

    /// Returns this request with only its exact inventory fact replaced.
    ///
    /// A registry adapter uses this to preflight policy before it performs a
    /// lookup, preserving the `disabled` policy's lookup-free eager path.
    #[must_use]
    pub const fn with_inventory(self, inventory: GraphInventoryState) -> Self {
        Self { inventory, ..self }
    }

    /// Returns this request with only its scalar eligibility facts replaced.
    #[must_use]
    pub const fn with_eligibility(self, eligibility: GraphDispatchEligibility) -> Self {
        Self {
            eligibility,
            ..self
        }
    }
}

/// Mode selected for one iteration after graph policy evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExecutionMode {
    /// Replay a fully prepared pure-decode CUDA Graph.
    FullGraph,
    /// Replay a prepared graph segment around dynamic work boundaries.
    PiecewiseGraph,
    /// Use the existing, exact command-batch execution path.
    ExactEager,
}

/// Closed explanation for a graph miss or exact-eager fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphFallbackReason {
    /// The configured policy deliberately disables graph dispatch.
    PolicyDisabled,
    /// The requested signature has no prepared graph entry.
    NotPrepared,
    /// The iteration stage has no reviewed graph mode.
    UnsupportedStage,
    /// The active-row bucket is not part of the prepared graph inventory.
    UnsupportedShape,
    /// The sampling/output backend is not the admitted GPU-greedy backend.
    UnsupportedSampling,
    /// The runtime metadata layout differs from the prepared graph layout.
    LayoutMismatch,
    /// Signature facts disagree with independently observed request facts.
    SignatureMismatch,
    /// A selected backend or operator is not capture-safe.
    BackendNotCaptureSafe,
    /// The exact graph entry is poisoned and must never be replayed.
    GraphPoisoned,
    /// Cold graph inventory capacity was disabled or exhausted.
    CapacityDisabled,
    /// At least one selected operator's capture capability is unknown.
    OperatorCapabilityUnknown,
}

impl GraphFallbackReason {
    /// Stable allocation-free identifier for metrics and traces.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::PolicyDisabled => "policy-disabled",
            Self::NotPrepared => "not-prepared",
            Self::UnsupportedStage => "unsupported-stage",
            Self::UnsupportedShape => "unsupported-shape",
            Self::UnsupportedSampling => "unsupported-sampling",
            Self::LayoutMismatch => "layout-mismatch",
            Self::SignatureMismatch => "signature-mismatch",
            Self::BackendNotCaptureSafe => "backend-not-capture-safe",
            Self::GraphPoisoned => "graph-poisoned",
            Self::CapacityDisabled => "capacity-disabled",
            Self::OperatorCapabilityUnknown => "operator-capability-unknown",
        }
    }
}

/// A successful graph-dispatch decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphDispatchDecision {
    /// A full graph is ready and eligible for pure decode.
    FullGraph,
    /// A piecewise graph is ready and eligible for prefill or mixed work.
    PiecewiseGraph,
    /// No graph runs; the established exact eager path remains correct.
    ExactEager(GraphFallbackReason),
}

impl GraphDispatchDecision {
    /// Execution mode selected by this decision.
    #[must_use]
    pub const fn mode(self) -> ExecutionMode {
        match self {
            Self::FullGraph => ExecutionMode::FullGraph,
            Self::PiecewiseGraph => ExecutionMode::PiecewiseGraph,
            Self::ExactEager(_) => ExecutionMode::ExactEager,
        }
    }

    /// Exact-eager explanation, if no graph is selected.
    #[must_use]
    pub const fn fallback_reason(self) -> Option<GraphFallbackReason> {
        match self {
            Self::ExactEager(reason) => Some(reason),
            Self::FullGraph | Self::PiecewiseGraph => None,
        }
    }
}

/// Fail-closed outcome when the `require` policy cannot select a graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphDispatchError {
    /// The required graph path is unavailable for the exact request facts.
    RequiredGraphUnavailable {
        /// Closed reason that prevented graph selection.
        reason: GraphFallbackReason,
    },
}

impl fmt::Display for GraphDispatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequiredGraphUnavailable { reason } => write!(
                formatter,
                "execution graph is required but unavailable: {}",
                reason.id()
            ),
        }
    }
}

impl error::Error for GraphDispatchError {}

/// Selects graph replay, exact eager work, or a fail-closed `require` error.
///
/// This function is pure and allocation-free. It never opens a capture, looks
/// up a raw handle, mutates an executor, or turns a graph miss into capture on
/// the iteration hot path.
///
/// # Errors
///
/// Returns `GraphDispatchError::RequiredGraphUnavailable` only when the request
/// uses `ExecutionGraphPolicy::Require` and no eligible, matching prepared graph
/// is available.
pub const fn select_execution_graph(
    request: GraphDispatchRequest,
) -> Result<GraphDispatchDecision, GraphDispatchError> {
    match request.policy {
        ExecutionGraphPolicy::Disabled => {
            return Ok(GraphDispatchDecision::ExactEager(
                GraphFallbackReason::PolicyDisabled,
            ));
        }
        ExecutionGraphPolicy::Auto | ExecutionGraphPolicy::Require => {}
    }

    let eligibility = request.eligibility;
    match eligibility.stage {
        GraphWorkloadStage::Unsupported => {
            return fallback(request.policy, GraphFallbackReason::UnsupportedStage);
        }
        GraphWorkloadStage::Prefill
        | GraphWorkloadStage::PureDecode
        | GraphWorkloadStage::Mixed => {}
    }
    if !eligibility.inventory_enabled {
        return fallback(request.policy, GraphFallbackReason::CapacityDisabled);
    }
    if !eligibility.active_row_bucket_supported {
        return fallback(request.policy, GraphFallbackReason::UnsupportedShape);
    }
    if !eligibility.metadata_layout_matches {
        return fallback(request.policy, GraphFallbackReason::LayoutMismatch);
    }
    match eligibility.stage {
        GraphWorkloadStage::PureDecode => match eligibility.capture_safety.sampling_backend {
            GraphSamplingBackend::GpuGreedy => {}
            GraphSamplingBackend::Unsupported => {
                return fallback(request.policy, GraphFallbackReason::UnsupportedSampling);
            }
        },
        GraphWorkloadStage::Prefill | GraphWorkloadStage::Mixed => {}
        GraphWorkloadStage::Unsupported => {
            return fallback(request.policy, GraphFallbackReason::UnsupportedStage);
        }
    }
    match eligibility.capture_safety.operator_capability {
        GraphOperatorCapability::Unknown => {
            return fallback(
                request.policy,
                GraphFallbackReason::OperatorCapabilityUnknown,
            );
        }
        GraphOperatorCapability::Unsupported => {
            return fallback(request.policy, GraphFallbackReason::BackendNotCaptureSafe);
        }
        GraphOperatorCapability::Supported => {}
    }
    if !eligibility.capture_safety.backend_capture_safe {
        return fallback(request.policy, GraphFallbackReason::BackendNotCaptureSafe);
    }

    match (eligibility.stage, request.inventory) {
        (GraphWorkloadStage::PureDecode, GraphInventoryState::PreparedFull) => {
            Ok(GraphDispatchDecision::FullGraph)
        }
        (
            GraphWorkloadStage::Prefill | GraphWorkloadStage::Mixed,
            GraphInventoryState::PreparedPiecewise,
        ) => Ok(GraphDispatchDecision::PiecewiseGraph),
        (_, GraphInventoryState::Poisoned) => {
            fallback(request.policy, GraphFallbackReason::GraphPoisoned)
        }
        _ => fallback(request.policy, GraphFallbackReason::NotPrepared),
    }
}

const fn fallback(
    policy: ExecutionGraphPolicy,
    reason: GraphFallbackReason,
) -> Result<GraphDispatchDecision, GraphDispatchError> {
    match policy {
        ExecutionGraphPolicy::Disabled | ExecutionGraphPolicy::Auto => {
            Ok(GraphDispatchDecision::ExactEager(reason))
        }
        ExecutionGraphPolicy::Require => {
            Err(GraphDispatchError::RequiredGraphUnavailable { reason })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExecutionGraphPolicy, ExecutionMode, GraphCaptureSafety, GraphDispatchDecision,
        GraphDispatchEligibility, GraphDispatchError, GraphDispatchRequest, GraphFallbackReason,
        GraphInventoryState, GraphOperatorCapability, GraphSamplingBackend, GraphWorkloadStage,
        select_execution_graph,
    };

    const fn capture_safety() -> GraphCaptureSafety {
        GraphCaptureSafety::new(
            GraphSamplingBackend::GpuGreedy,
            GraphOperatorCapability::Supported,
            true,
        )
    }

    const fn request(
        policy: ExecutionGraphPolicy,
        stage: GraphWorkloadStage,
        inventory: GraphInventoryState,
    ) -> GraphDispatchRequest {
        GraphDispatchRequest::new(
            policy,
            GraphDispatchEligibility::new(stage, true, true, true, capture_safety()),
            inventory,
        )
    }

    #[test]
    fn disabled_policy_never_selects_a_ready_graph() {
        let decision = select_execution_graph(request(
            ExecutionGraphPolicy::Disabled,
            GraphWorkloadStage::PureDecode,
            GraphInventoryState::PreparedFull,
        ))
        .expect("disabled policy must preserve eager execution");

        assert_eq!(decision.mode(), ExecutionMode::ExactEager);
        assert_eq!(
            decision.fallback_reason(),
            Some(GraphFallbackReason::PolicyDisabled)
        );
    }

    #[test]
    fn auto_selects_only_the_stage_matching_prepared_mode() {
        assert_eq!(
            select_execution_graph(request(
                ExecutionGraphPolicy::Auto,
                GraphWorkloadStage::PureDecode,
                GraphInventoryState::PreparedFull,
            )),
            Ok(GraphDispatchDecision::FullGraph)
        );
        assert_eq!(
            select_execution_graph(request(
                ExecutionGraphPolicy::Auto,
                GraphWorkloadStage::Prefill,
                GraphInventoryState::PreparedPiecewise,
            )),
            Ok(GraphDispatchDecision::PiecewiseGraph)
        );
        assert_eq!(
            select_execution_graph(request(
                ExecutionGraphPolicy::Auto,
                GraphWorkloadStage::Mixed,
                GraphInventoryState::PreparedPiecewise,
            )),
            Ok(GraphDispatchDecision::PiecewiseGraph)
        );
    }

    #[test]
    fn auto_reports_closed_fallback_reasons_without_graph_work() {
        let unknown = GraphDispatchRequest::new(
            ExecutionGraphPolicy::Auto,
            GraphDispatchEligibility::new(
                GraphWorkloadStage::PureDecode,
                true,
                true,
                true,
                GraphCaptureSafety::new(
                    GraphSamplingBackend::GpuGreedy,
                    GraphOperatorCapability::Unknown,
                    true,
                ),
            ),
            GraphInventoryState::PreparedFull,
        );
        assert_eq!(
            select_execution_graph(unknown),
            Ok(GraphDispatchDecision::ExactEager(
                GraphFallbackReason::OperatorCapabilityUnknown
            ))
        );
        assert_eq!(
            select_execution_graph(request(
                ExecutionGraphPolicy::Auto,
                GraphWorkloadStage::PureDecode,
                GraphInventoryState::Poisoned,
            )),
            Ok(GraphDispatchDecision::ExactEager(
                GraphFallbackReason::GraphPoisoned
            ))
        );
        assert_eq!(GraphFallbackReason::LayoutMismatch.id(), "layout-mismatch");
    }

    #[test]
    fn require_turns_every_miss_into_a_fail_closed_error() {
        assert_eq!(
            select_execution_graph(request(
                ExecutionGraphPolicy::Require,
                GraphWorkloadStage::PureDecode,
                GraphInventoryState::NotPrepared,
            )),
            Err(GraphDispatchError::RequiredGraphUnavailable {
                reason: GraphFallbackReason::NotPrepared,
            })
        );

        let unsupported_sampling = GraphDispatchRequest::new(
            ExecutionGraphPolicy::Require,
            GraphDispatchEligibility::new(
                GraphWorkloadStage::PureDecode,
                true,
                true,
                true,
                GraphCaptureSafety::new(
                    GraphSamplingBackend::Unsupported,
                    GraphOperatorCapability::Supported,
                    true,
                ),
            ),
            GraphInventoryState::PreparedFull,
        );
        assert_eq!(
            select_execution_graph(unsupported_sampling),
            Err(GraphDispatchError::RequiredGraphUnavailable {
                reason: GraphFallbackReason::UnsupportedSampling,
            })
        );
    }
}
