//! Checks the Llama 3 loader pieces against real artifacts when a directory
//! with them is given in `RILEY_LLAMA3_ARTIFACT_DIR` (the test passes without
//! doing anything otherwise, since the artifacts are not part of the
//! repository): `config.json`, `tokenizer.json`, and
//! `llama3-tokenizer-reference.json`, a list of
//! `{"text", "with_bos", "without"}` records produced by the Hugging Face
//! tokenizer of the same checkpoint.

use std::path::PathBuf;

use riley_model::{
    DecodeOptions, EncodeOptions, LlamaConfig, LoadLimits, SmolLm2Tokenizer, Tokenizer,
};

#[test]
fn llama3_config_and_tokenizer_match_hugging_face_on_real_artifacts() {
    let Some(directory) = std::env::var_os("RILEY_LLAMA3_ARTIFACT_DIR").map(PathBuf::from) else {
        eprintln!("RILEY_LLAMA3_ARTIFACT_DIR is not set; nothing checked");
        return;
    };
    let config = LlamaConfig::from_json_slice(
        &std::fs::read(directory.join("config.json")).expect("config.json"),
    )
    .expect("Llama 3 config");
    let spec = config.to_model_spec();
    let rope = spec.blocks()[0].attention().rope();
    assert!(rope.scaling().is_some(), "llama3 rope scaling");

    let tokenizer = SmolLm2Tokenizer::from_llama3_json_slice_with_limits(
        &std::fs::read(directory.join("tokenizer.json")).expect("tokenizer.json"),
        LoadLimits::production(),
    )
    .expect("Llama 3 tokenizer");
    assert_eq!(tokenizer.bos_token_id(), spec.special_tokens().bos());
    assert_eq!(
        tokenizer.addressable_token_count(),
        spec.embedding().vocabulary_size()
    );

    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.join("llama3-tokenizer-reference.json")).expect("reference"),
    )
    .expect("reference JSON");
    let cases = reference.as_array().expect("reference list");
    assert!(!cases.is_empty());
    let ids = |value: &serde_json::Value| -> Vec<u32> {
        value
            .as_array()
            .expect("id list")
            .iter()
            .map(|id| u32::try_from(id.as_u64().expect("id")).expect("u32 id"))
            .collect()
    };
    let mut tokens = 0_usize;
    for case in cases {
        let text = case["text"].as_str().expect("text");
        let with_bos = tokenizer
            .encode(text, EncodeOptions::default())
            .expect("encode with special tokens");
        assert_eq!(with_bos, ids(&case["with_bos"]), "with BOS: {text:?}");
        let without = tokenizer
            .encode(
                text,
                EncodeOptions {
                    add_special_tokens: false,
                },
            )
            .expect("encode without special tokens");
        assert_eq!(without, ids(&case["without"]), "without BOS: {text:?}");
        assert_eq!(
            tokenizer
                .decode(&without, DecodeOptions::default())
                .expect("decode"),
            text
        );
        tokens += without.len();
    }
    println!(
        "llama3 artifacts: {} texts, {tokens} tokens match",
        cases.len()
    );
}
