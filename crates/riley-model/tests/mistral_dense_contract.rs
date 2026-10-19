//! Cold contracts only: these do not qualify the Mistral tokenizer or CUDA output.
//! Source: mistralai/Mistral-7B-Instruct-v0.3 at
//! c170c708c41dac9275d15a8fff4eca08d52bab71, config.json.
use riley_model::{LlamaConfig, ModelArchitecture, ModelConfig, ModelError, ModelFamily};
use serde_json::{Value, json};

const CONFIG: &[u8] = include_bytes!("fixtures/mistral-7b-v0.3-config.json");

#[test]
fn full_attention_mistral_lowers_to_shared_dense_execution() {
    let config = ModelConfig::from_json_slice(CONFIG).unwrap();
    assert_eq!(config.family(), ModelFamily::Mistral);
    let spec = config.to_model_spec();
    assert_eq!(spec.architecture(), ModelArchitecture::Llama);
    assert_eq!(spec.source_architecture(), "MistralForCausalLM");
    assert_eq!(spec.embedding().vocabulary_size(), 32768);
    assert_eq!(spec.max_sequence_length(), 32768);
    assert_eq!(spec.blocks().len(), 32);
    for block in spec.blocks() {
        let attention = block.attention();
        assert_eq!(attention.hidden_size(), 4096);
        assert_eq!(attention.query_heads(), 32);
        assert_eq!(attention.key_value_heads(), 8);
        assert_eq!(attention.head_dimension(), 128);
        assert!(!attention.has_bias());
        assert!(attention.query_key_norm().is_none());
        assert!(!block.mlp().has_bias());
        assert_eq!(attention.rope().theta(), 1_000_000.0);
        assert!(attention.rope().scaling().is_none());
    }
    // The dedicated Llama parser must retain its strict family boundary.
    assert!(LlamaConfig::from_json_slice(CONFIG).is_err());
}

#[test]
fn mistral_never_silently_substitutes_full_attention_for_a_default_window() {
    let original: Value = serde_json::from_slice(CONFIG).unwrap();
    for window in [None, Some(json!(4096)), Some(json!(0))] {
        let mut changed = original.clone();
        if let Some(window) = window {
            changed["sliding_window"] = window;
        } else {
            changed.as_object_mut().unwrap().remove("sliding_window");
        }
        assert!(matches!(
            ModelConfig::from_json_slice(&serde_json::to_vec(&changed).unwrap()),
            Err(ModelError::UnsupportedConfig { field, .. }) if field == "sliding_window"
        ));
    }
}

#[test]
fn mistral_rejects_changed_projection_rope_and_architecture_semantics() {
    for (field, value) in [
        ("attention_bias", json!(true)),
        ("mlp_bias", json!(true)),
        ("architectures", json!(["MixtralForCausalLM"])),
        ("model_type", json!("mixtral")),
        ("rope_scaling", json!({"rope_type": "llama3", "factor": 8})),
        ("quantization_config", json!({"quant_method": "gptq"})),
    ] {
        let mut changed: Value = serde_json::from_slice(CONFIG).unwrap();
        changed[field] = value;
        assert!(
            ModelConfig::from_json_slice(&serde_json::to_vec(&changed).unwrap()).is_err(),
            "accepted unsupported {field}"
        );
    }
}
