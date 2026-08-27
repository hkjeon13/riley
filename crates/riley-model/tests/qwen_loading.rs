//! Remote-only PR12 aggregate Qwen artifact validation.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use riley_model::{
    CONFIG_FILENAME, ChatMessage, ChatRole, ChatTemplateOptions, LoadLimits, LoadedModel,
    ModelConfig, ModelFamily, Qwen2Tokenizer, Qwen2TokenizerConfig, TOKENIZER_CONFIG_FILENAME,
    TOKENIZER_FILENAME, WeightBinding, WeightSlot,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const QWEN_WEIGHT_LIMIT_BYTES: u64 = 1024 * 1024 * 1024;
const QWEN_ADDRESSABLE_TOKENS: usize = 151_665;
const QWEN_MODEL_TOKENS: usize = 151_936;

fn qwen_checkpoint() -> PathBuf {
    std::env::var_os("RILEY_QWEN_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RILEY_QWEN_CHECKPOINT must name the pinned remote checkpoint")
}

fn qwen_limits() -> TestResult<LoadLimits> {
    Ok(LoadLimits::default()
        .with_weight_byte_limits(QWEN_WEIGHT_LIMIT_BYTES, QWEN_WEIGHT_LIMIT_BYTES)?)
}

#[test]
#[ignore = "parses the pinned Qwen metadata only on server-4096"]
fn pinned_qwen_metadata_matches_the_strict_native_profile() -> TestResult {
    let checkpoint = qwen_checkpoint();
    let limits = LoadLimits::default();
    let config = ModelConfig::from_json_slice_with_limits(
        &fs::read(checkpoint.join(CONFIG_FILENAME))?,
        limits,
    )?;
    let tokenizer = Qwen2Tokenizer::from_json_slice_with_limits(
        &fs::read(checkpoint.join(TOKENIZER_FILENAME))?,
        limits,
    )?;
    let tokenizer_config = Qwen2TokenizerConfig::from_json_slice_with_limits(
        &fs::read(checkpoint.join(TOKENIZER_CONFIG_FILENAME))?,
        limits,
    )?;
    tokenizer_config.validate_tokenizer(&tokenizer)?;

    assert_eq!(config.family(), ModelFamily::Qwen2);
    assert_eq!(tokenizer.addressable_token_count(), QWEN_ADDRESSABLE_TOKENS);
    assert_eq!(tokenizer_config.model_max_length(), 131_072);
    Ok(())
}

#[test]
#[ignore = "loads the pinned 988 MB Qwen checkpoint only on server-4096"]
fn pinned_qwen_checkpoint_is_complete_bounded_and_chat_ready() -> TestResult {
    let model = LoadedModel::load(&qwen_checkpoint(), qwen_limits()?)?;

    assert_eq!(model.config().family(), ModelFamily::Qwen2);
    assert_eq!(model.spec().source_architecture(), "Qwen2ForCausalLM");
    assert_eq!(
        model.spec().embedding().vocabulary_size(),
        QWEN_MODEL_TOKENS
    );
    assert_eq!(model.spec().blocks().len(), 24);
    assert_eq!(model.spec().blocks()[0].attention().query_heads(), 14);
    assert_eq!(model.spec().blocks()[0].attention().key_value_heads(), 2);
    assert_eq!(model.spec().blocks()[0].attention().head_dimension(), 64);
    assert_eq!(
        model.tokenizer().addressable_token_count(),
        QWEN_ADDRESSABLE_TOKENS
    );
    assert_eq!(model.weights().bindings().len(), 291);
    assert!(matches!(
        model.weights().binding(WeightSlot::LmHead),
        Some(WeightBinding::Alias(WeightSlot::TokenEmbedding))
    ));

    let tokenizer_config = model
        .qwen2_tokenizer_config()
        .expect("Qwen load must retain validated chat metadata");
    assert_eq!(tokenizer_config.model_max_length(), 131_072);
    let rendered = tokenizer_config.render_chat(
        &[ChatMessage {
            role: ChatRole::User,
            content: "Hello",
            has_tool_calls: false,
        }],
        ChatTemplateOptions {
            add_generation_prompt: true,
            tools_requested: false,
        },
    )?;
    assert_eq!(
        rendered,
        "<|im_start|>system\n\
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n\
<|im_start|>user\nHello<|im_end|>\n\
<|im_start|>assistant\n"
    );

    let provenance = model.provenance();
    assert_eq!(provenance.source_model(), "Qwen/Qwen2.5-0.5B-Instruct");
    assert_eq!(
        provenance.source_revision(),
        "7ae557604adf67be50417f59c2c2f167def9a775"
    );
    let expected_files: [(&str, u64, &str); 4] = [
        (
            "config.json",
            659,
            "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
        ),
        (
            "tokenizer.json",
            7_031_645,
            "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        ),
        (
            TOKENIZER_CONFIG_FILENAME,
            7_305,
            "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
        ),
        (
            "model.safetensors",
            988_097_824,
            "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
        ),
    ];
    for (path, bytes, sha256) in expected_files {
        let file = provenance
            .files()
            .get(Path::new(path))
            .expect("pinned Qwen provenance entry");
        assert_eq!(file.byte_len(), bytes);
        assert_eq!(file.sha256(), sha256);
    }
    Ok(())
}

#[test]
#[ignore = "loads both pinned checkpoints sequentially only on server-4096"]
fn llama_and_qwen_load_and_unload_in_one_process() -> TestResult {
    let llama_checkpoint = std::env::var_os("RILEY_REAL_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RILEY_REAL_CHECKPOINT must name the pinned SmolLM2 checkpoint");
    {
        let llama = LoadedModel::load(&llama_checkpoint, LoadLimits::default())?;
        assert_eq!(llama.config().family(), ModelFamily::Llama);
        assert!(llama.qwen2_tokenizer_config().is_none());
    }
    {
        let qwen = LoadedModel::load(&qwen_checkpoint(), qwen_limits()?)?;
        assert_eq!(qwen.config().family(), ModelFamily::Qwen2);
        assert!(qwen.qwen2_tokenizer_config().is_some());
    }
    Ok(())
}
