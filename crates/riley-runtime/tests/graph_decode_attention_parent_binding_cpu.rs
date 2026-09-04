//! Source boundary contract for G01 parent-KV attention graph binding.

const BINDING_SOURCE: &str = include_str!("../src/llama/graph_decode_attention_parent_binding.rs");
const LLAMA_MODULE_SOURCE: &str = include_str!("../src/llama/mod.rs");
const INVENTORY_SOURCE: &str = include_str!("../src/llama/graph_decode_capture_inventory.rs");

#[test]
fn parent_binding_is_private_cpu_only_and_never_exposes_raw_device_addresses() {
    for required in [
        "AttentionGraphParentAllocationId",
        "AttentionGraphLayerSpan",
        "GraphMetadataLayoutSignature",
        "KvLayout::layer_byte_offset",
        "ParentLeaseActive",
        "GraphLaunchInFlight",
        "GraphLeaseAlreadyReleased",
        "OverlappingKeyValueParents",
    ] {
        assert!(
            BINDING_SOURCE.contains(required),
            "G01 parent binding omitted required contract token {required:?}"
        );
    }
    for forbidden in [
        "unsafe",
        "extern \"C\"",
        "*const",
        "*mut",
        "CudaDeviceBuffer",
        "CudaStream",
        "riley_cuda",
        "GraphRegistry",
        "select_execution_graph",
        "PureDecodeGraphV1CaptureOperation::Attention,\n            GraphOperatorCapability::Supported",
    ] {
        assert!(
            !BINDING_SOURCE.contains(forbidden),
            "G01 parent binding crossed its CPU/lifetime boundary with {forbidden:?}"
        );
    }
    assert!(
        LLAMA_MODULE_SOURCE.contains(
            "#[allow(dead_code)] // G01 establishes CPU parent-span ownership before native C05-19 binding.\nmod graph_decode_attention_parent_binding;"
        ),
        "G01 binding must remain a private CPU-only precursor"
    );
    assert!(
        !LLAMA_MODULE_SOURCE.contains("pub use graph_decode_attention_parent_binding"),
        "G01 binding must not become public before the native owner exists"
    );
    assert!(
        INVENTORY_SOURCE.contains("Attention"),
        "the source inventory must keep its explicit attention slot"
    );
}
