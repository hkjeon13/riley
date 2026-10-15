//! Real pinned dense configs: the 4B query projection is wider than its residual.
//! These cold-contract checks do not establish GPU or generation correctness.
use riley_model::{ModelArchitecture, ModelConfig, ModelFamily, NormKind};

#[test]
fn pinned_dense_configs_keep_head_normalization_and_independent_projection_width() {
    for (bytes, hidden, layers, query_heads) in [
        (include_bytes!("fixtures/qwen3-1.7b-config.json").as_slice(), 2048, 28, 16),
        (include_bytes!("fixtures/qwen3-4b-config.json").as_slice(), 2560, 36, 32),
    ] {
        let config = ModelConfig::from_json_slice(bytes).expect("pinned Qwen3 config");
        assert_eq!(config.family(), ModelFamily::Qwen3);
        let spec = config.to_model_spec();
        assert_eq!(spec.architecture(), ModelArchitecture::Llama);
        assert_eq!(spec.source_architecture(), "Qwen3ForCausalLM");
        assert_eq!(spec.blocks().len(), layers);
        assert_eq!(spec.max_sequence_length(), 40960);
        for block in spec.blocks() {
            let attention = block.attention();
            assert_eq!(attention.hidden_size(), hidden);
            assert_eq!(attention.query_heads(), query_heads);
            assert_eq!(attention.key_value_heads(), 8);
            assert_eq!(attention.head_dimension(), 128);
            assert!(!attention.has_bias());
            assert!(!block.mlp().has_bias());
            assert_eq!(attention.query_key_norm().expect("Q/K norm").kind(), NormKind::RmsNorm);
            assert_eq!(attention.rope().theta().to_bits(), 1_000_000.0_f64.to_bits());
        }
        if hidden == 2560 {
            assert_ne!(hidden, query_heads * 128);
        }
    }
}

#[test]
fn qwen3_rejects_missing_head_dimension_and_biased_attention() {
    let config: serde_json::Value = serde_json::from_slice(
        include_bytes!("fixtures/qwen3-4b-config.json"),
    ).unwrap();
    let mut missing = config.clone();
    missing.as_object_mut().unwrap().remove("head_dim");
    assert!(ModelConfig::from_json_slice(&serde_json::to_vec(&missing).unwrap()).is_err());
    let mut biased = config;
    biased["attention_bias"] = serde_json::Value::Bool(true);
    assert!(ModelConfig::from_json_slice(&serde_json::to_vec(&biased).unwrap()).is_err());
}
