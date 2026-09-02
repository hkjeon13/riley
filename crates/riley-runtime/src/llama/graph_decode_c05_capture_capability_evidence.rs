//! Exact C05 primitive evidence for C07's cold pure-decode inventory.
//!
//! This adapter consumes only the native C05 capability vocabulary.  It does
//! not create a CUDA runtime/context, allocate, capture, instantiate, launch,
//! or attach an inventory to C06 dispatch.  In particular, `FillF32` has no
//! pure-decode operation slot and therefore supplies no C07 evidence.
//!
//! A native `Supported` result is mapped only to the semantically identical
//! C07 operation: whole-slab H2D to metadata transfer, canonical generic
//! BF16 RMSNorm to the per-layer normalization slot, fixed-address BF16 SiLU
//! to the MLP BF16-SiLU activation, fixed-address BF16 multiplication of an
//! activated gate with an up projection to the MLP gated multiply, and
//! fixed-address BF16 residual addition to the decode residual slot. Every
//! other C07 operation stays unknown until an equally exact reviewed C05
//! primitive exists.

use riley_cuda::{CudaGraphCaptureCapability, CudaGraphCaptureOperation, CudaResult};

use super::{
    graph::GraphOperatorCapability,
    graph_decode_capture_inventory::{
        PureDecodeGraphV1CaptureCapabilityInventory, PureDecodeGraphV1CaptureOperation,
    },
};

fn map_exact_c05_capability(capability: CudaGraphCaptureCapability) -> GraphOperatorCapability {
    match capability {
        CudaGraphCaptureCapability::Unknown => GraphOperatorCapability::Unknown,
        CudaGraphCaptureCapability::Unsupported => GraphOperatorCapability::Unsupported,
        CudaGraphCaptureCapability::Supported => GraphOperatorCapability::Supported,
        _ => GraphOperatorCapability::Unknown,
    }
}

/// Queries reviewed C05 primitive evidence for the C07 pure-decode inventory.
///
/// The returned value remains an incomplete inventory: it can only carry the
/// exact metadata-H2D, canonical-BF16-RMSNorm, MLP-BF16-SiLU,
/// MLP-gated-multiply, and residual-add facts. Its aggregate capability
/// therefore remains `Unknown` unless future reviewed adapters fill every
/// remaining operation. Query failures are preserved rather than being
/// reinterpreted as execution permission.
///
/// # Errors
///
/// Returns a native C05 ABI error unchanged.  The native query itself is a
/// pure vocabulary lookup and does not initialize CUDA.
pub(crate) fn pure_decode_graph_v1_c05_capture_capability_evidence()
-> CudaResult<PureDecodeGraphV1CaptureCapabilityInventory> {
    let metadata_h2d =
        map_exact_c05_capability(CudaGraphCaptureOperation::H2D.capture_capability()?);
    let norm = map_exact_c05_capability(
        CudaGraphCaptureOperation::CanonicalRmsNormBf16.capture_capability()?,
    );
    let mlp_silu =
        map_exact_c05_capability(CudaGraphCaptureOperation::SiluBf16.capture_capability()?);
    let mlp_gated_multiply = map_exact_c05_capability(
        CudaGraphCaptureOperation::GatedMultiplyBf16.capture_capability()?,
    );
    let residual =
        map_exact_c05_capability(CudaGraphCaptureOperation::ResidualAddBf16.capture_capability()?);

    Ok(PureDecodeGraphV1CaptureCapabilityInventory::default()
        .with_capability(PureDecodeGraphV1CaptureOperation::MetadataH2d, metadata_h2d)
        .with_capability(PureDecodeGraphV1CaptureOperation::Norm, norm)
        .with_capability(PureDecodeGraphV1CaptureOperation::MlpSiluBf16, mlp_silu)
        .with_capability(
            PureDecodeGraphV1CaptureOperation::MlpGatedMultiply,
            mlp_gated_multiply,
        )
        .with_capability(PureDecodeGraphV1CaptureOperation::Residual, residual))
}

#[cfg(test)]
mod tests {
    use super::super::{
        graph::GraphOperatorCapability,
        graph_decode_capture_inventory::PureDecodeGraphV1CaptureOperation,
    };
    use super::{map_exact_c05_capability, pure_decode_graph_v1_c05_capture_capability_evidence};
    use riley_cuda::CudaGraphCaptureCapability;

    #[test]
    fn exact_c05_capability_mapping_is_closed() {
        assert_eq!(
            map_exact_c05_capability(CudaGraphCaptureCapability::Unknown),
            GraphOperatorCapability::Unknown
        );
        assert_eq!(
            map_exact_c05_capability(CudaGraphCaptureCapability::Unsupported),
            GraphOperatorCapability::Unsupported
        );
        assert_eq!(
            map_exact_c05_capability(CudaGraphCaptureCapability::Supported),
            GraphOperatorCapability::Supported
        );
    }

    #[test]
    fn native_c05_evidence_leaves_the_incomplete_decode_chain_unknown() {
        let inventory = pure_decode_graph_v1_c05_capture_capability_evidence()
            .expect("reviewed C05 primitive capability queries must link");

        assert_eq!(
            inventory.capability_for(PureDecodeGraphV1CaptureOperation::MetadataH2d),
            GraphOperatorCapability::Supported
        );
        assert_eq!(
            inventory.capability_for(PureDecodeGraphV1CaptureOperation::Norm),
            GraphOperatorCapability::Supported
        );
        assert_eq!(
            inventory.capability_for(PureDecodeGraphV1CaptureOperation::MlpSiluBf16),
            GraphOperatorCapability::Supported
        );
        assert_eq!(
            inventory.capability_for(PureDecodeGraphV1CaptureOperation::MlpGatedMultiply),
            GraphOperatorCapability::Supported
        );
        assert_eq!(
            inventory.capability_for(PureDecodeGraphV1CaptureOperation::Residual),
            GraphOperatorCapability::Supported
        );
        assert_eq!(
            inventory.operator_capability(),
            GraphOperatorCapability::Unknown,
            "five reviewed primitives must not admit the incomplete decode chain"
        );
        for operation in PureDecodeGraphV1CaptureOperation::all() {
            if matches!(
                operation,
                PureDecodeGraphV1CaptureOperation::MetadataH2d
                    | PureDecodeGraphV1CaptureOperation::Norm
                    | PureDecodeGraphV1CaptureOperation::MlpSiluBf16
                    | PureDecodeGraphV1CaptureOperation::MlpGatedMultiply
                    | PureDecodeGraphV1CaptureOperation::Residual
            ) {
                continue;
            }
            assert_eq!(
                inventory.capability_for(operation),
                GraphOperatorCapability::Unknown,
                "unmapped C07 operation must not inherit C05 primitive evidence: {operation:?}"
            );
        }
    }
}
