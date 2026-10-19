//! CPU-only oracle checks against pinned HF artifacts and tokenizers 0.23.3.
//! Run explicitly with RILEY_MISTRAL_ARTIFACT_DIR and --ignored; never silently skip.
use riley_model::{
    DecodeOptions, EncodeOptions, LoadLimits, LoadedModel, MistralTokenizer, ModelFamily,
    TokenDecodeState, Tokenizer,
};
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Deserialize)]
struct Reference {
    encoding: Vec<EncodingCase>,
    decoding: Vec<DecodingCase>,
}
#[derive(Deserialize)]
struct EncodingCase {
    text: String,
    with_bos: Vec<u32>,
    without_bos: Vec<u32>,
    decoded: String,
    decoded_special: String,
}
#[derive(Deserialize)]
struct DecodingCase {
    ids: Vec<u32>,
    skip: bool,
    text: String,
}

fn directory() -> PathBuf {
    std::env::var_os("RILEY_MISTRAL_ARTIFACT_DIR")
        .map(PathBuf::from)
        .expect("set RILEY_MISTRAL_ARTIFACT_DIR to the pinned CPU oracle artifacts")
}
fn tokenizer() -> MistralTokenizer {
    MistralTokenizer::from_json_slice(&fs::read(directory().join("tokenizer.json")).unwrap())
        .unwrap()
}

#[test]
#[ignore = "loads the pinned 14.5 GB Mistral checkpoint on A100, CPU only"]
fn complete_mistral_checkpoint_loads_with_native_tokenizer_and_weight_bindings() {
    let root = std::env::var_os("RILEY_MISTRAL_CHECKPOINT")
        .map(PathBuf::from)
        .expect("set RILEY_MISTRAL_CHECKPOINT to the verified complete model");
    let limits = LoadLimits::production()
        .with_weight_byte_limits(32 << 30, 32 << 30)
        .unwrap();
    let model = LoadedModel::load(&root, limits).unwrap();
    assert_eq!(model.config().family(), ModelFamily::Mistral);
    assert_eq!(model.spec().source_architecture(), "MistralForCausalLM");
    assert_eq!(
        model.provenance().source_revision(),
        "c170c708c41dac9275d15a8fff4eca08d52bab71"
    );
    assert_eq!(model.weights().bindings().len(), 32 * 9 + 3);
    assert!(model.qwen2_tokenizer_config().is_none());
    let tokens = model
        .tokenizer()
        .encode("The capital of France is", EncodeOptions::default())
        .unwrap();
    assert_eq!(
        model
            .tokenizer()
            .decode(
                &tokens,
                DecodeOptions {
                    skip_special_tokens: true
                }
            )
            .unwrap(),
        "The capital of France is"
    );
}

#[test]
#[ignore = "requires pinned Mistral artifacts and independent HF tokenizers oracle"]
fn native_encoding_batch_and_incremental_decoding_match_hugging_face() {
    let tokenizer = tokenizer();
    let reference: Reference =
        serde_json::from_slice(&fs::read(directory().join("reference.json")).unwrap()).unwrap();
    assert!(reference.encoding.len() >= 300);
    assert!(reference.decoding.len() >= 32768);
    for case in &reference.encoding {
        let with_bos = tokenizer
            .encode(&case.text, EncodeOptions::default())
            .unwrap();
        let without = tokenizer
            .encode(
                &case.text,
                EncodeOptions {
                    add_special_tokens: false,
                },
            )
            .unwrap();
        assert_eq!(with_bos, case.with_bos, "with BOS: {:?}", case.text);
        assert_eq!(without, case.without_bos, "without BOS: {:?}", case.text);
        assert_eq!(
            tokenizer
                .decode(&with_bos, DecodeOptions::default())
                .unwrap(),
            case.decoded_special,
            "special decode: {:?}",
            case.text
        );
        let options = DecodeOptions {
            skip_special_tokens: true,
        };
        assert_eq!(
            tokenizer.decode(&with_bos, options).unwrap(),
            case.decoded,
            "batch decode: {:?}",
            case.text
        );
        let mut buffer = vec![0; tokenizer.maximum_decoded_token_bytes()];
        let mut streamed = Vec::new();
        let mut state = TokenDecodeState::default();
        for &id in &with_bos {
            let n = tokenizer
                .decode_token_bytes_with_state(id, options, &mut buffer, &mut state)
                .unwrap();
            streamed.extend_from_slice(&buffer[..n]);
        }
        assert_eq!(
            String::from_utf8(streamed).unwrap(),
            case.decoded,
            "incremental decode: {:?}",
            case.text
        );
    }
    for case in &reference.decoding {
        let options = DecodeOptions {
            skip_special_tokens: case.skip,
        };
        assert_eq!(
            tokenizer.decode(&case.ids, options).unwrap(),
            case.text,
            "decode IDs {:?}",
            case.ids
        );
        let n = tokenizer.decoded_bytes_len(&case.ids, options).unwrap();
        assert_eq!(n, case.text.len());
        let mut bytes = vec![0; n];
        assert_eq!(
            tokenizer
                .decode_bytes(&case.ids, options, &mut bytes)
                .unwrap(),
            n
        );
        assert_eq!(bytes, case.text.as_bytes());
    }
    println!(
        "Mistral HF oracle: {} encoding and {} decoding cases matched",
        reference.encoding.len(),
        reference.decoding.len()
    );
}

#[test]
#[ignore = "requires pinned Mistral artifacts"]
fn decoder_validates_before_writes_and_prefix_state_is_request_local() {
    let tokenizer = tokenizer();
    let options = DecodeOptions {
        skip_special_tokens: true,
    };
    let ids = tokenizer
        .encode("hello world", EncodeOptions::default())
        .unwrap();
    let mut dest = [0xa5; 128];
    let before = dest;
    assert!(
        tokenizer
            .decode_bytes(&[ids[1], u32::MAX], options, &mut dest)
            .is_err()
    );
    assert_eq!(dest, before);
    assert!(
        tokenizer
            .decode_bytes(&ids, options, &mut dest[..1])
            .is_err()
    );
    assert_eq!(dest, before);
    let mut one = TokenDecodeState::default();
    let mut two = TokenDecodeState::default();
    // Failure must not consume the first-token prefix of either request.
    assert!(
        tokenizer
            .decode_token_bytes_with_state(u32::MAX, options, &mut dest, &mut one)
            .is_err()
    );
    let mut a = Vec::new();
    let mut b = Vec::new();
    for &id in &ids {
        let n = tokenizer
            .decode_token_bytes_with_state(id, options, &mut dest, &mut one)
            .unwrap();
        a.extend_from_slice(&dest[..n]);
        let n = tokenizer
            .decode_token_bytes_with_state(id, options, &mut dest, &mut two)
            .unwrap();
        b.extend_from_slice(&dest[..n]);
    }
    assert_eq!(a, b"hello world");
    assert_eq!(b, a);
}

#[test]
#[ignore = "requires pinned Mistral artifacts"]
fn unsupported_tokenizer_semantics_fail_closed() {
    let raw: serde_json::Value =
        serde_json::from_slice(&fs::read(directory().join("tokenizer.json")).unwrap()).unwrap();
    for pointer in [
        "/pre_tokenizer/prepend_scheme",
        "/pre_tokenizer/split",
        "/model/byte_fallback",
        "/decoder/decoders/3/start",
    ] {
        let mut changed = raw.clone();
        *changed.pointer_mut(pointer).unwrap() = serde_json::Value::Null;
        assert!(
            MistralTokenizer::from_json_slice(&serde_json::to_vec(&changed).unwrap()).is_err(),
            "accepted {pointer}"
        );
    }
    let mut changed = raw;
    changed["model"]["vocab"]
        .as_object_mut()
        .unwrap()
        .remove("<0xFF>");
    assert!(MistralTokenizer::from_json_slice(&serde_json::to_vec(&changed).unwrap()).is_err());
}
