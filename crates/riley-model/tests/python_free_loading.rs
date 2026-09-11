use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use riley_model::{
    DecodeOptions, EncodeOptions, LoadLimits, LoadedModel, ModelError, Tokenizer, WeightSlot,
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
struct FixtureTensor {
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

struct TempCheckpoint {
    root: PathBuf,
}

impl TempCheckpoint {
    fn new() -> Self {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "riley-python-free-model-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&root).expect("create isolated synthetic checkpoint");
        Self { root }
    }

    fn root(&self) -> &Path {
        &self.root
    }
}

impl Drop for TempCheckpoint {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).expect("remove isolated synthetic checkpoint");
    }
}

#[test]
fn python_free_process_loads_complete_checkpoint() {
    if std::env::var_os("RILEY_REQUIRE_EMPTY_PATH").is_some() {
        assert_empty_executable_path();
    }
    let checkpoint = TempCheckpoint::new();
    write_complete_checkpoint(checkpoint.root());

    let model = LoadedModel::load(checkpoint.root(), LoadLimits::default())
        .expect("synthetic checkpoint must load without an external executable");

    assert_eq!(model.spec().blocks().len(), 1);
    assert_eq!(model.spec().embedding().vocabulary_size(), 8);
    assert_eq!(model.config().dtype(), model.spec().dtype());
    assert_eq!(model.provenance().source_model(), "fixture/tiny-llama");
    assert_eq!(
        model.verified_files(),
        &BTreeSet::from([
            PathBuf::from("config.json"),
            PathBuf::from("model.safetensors"),
            PathBuf::from("tokenizer.json"),
        ])
    );

    let token_ids = model
        .tokenizer()
        .encode("hello 12", EncodeOptions::default())
        .expect("Rust-native tokenizer encode");
    assert_eq!(token_ids, [1, 2, 3, 3, 4, 5, 6, 7]);
    assert_eq!(
        model
            .tokenizer()
            .decode(&token_ids, DecodeOptions::default())
            .expect("Rust-native tokenizer decode"),
        "hello 12"
    );

    let embedding = model
        .weights()
        .view(WeightSlot::TokenEmbedding)
        .expect("validated embedding view");
    let lm_head = model
        .weights()
        .view(WeightSlot::LmHead)
        .expect("validated tied LM-head view");
    assert_eq!(embedding.view().shape().dimensions(), &[8, 4]);
    assert_eq!(
        embedding.view().storage().as_ptr(),
        lm_head.view().storage().as_ptr()
    );
}

#[test]
#[ignore = "requires the pinned SmolLM2 checkpoint; run only on the remote validation server"]
fn remote_pinned_smollm2_checkpoint_matches_reference_contract() {
    let checkpoint = std::env::var_os("RILEY_REAL_CHECKPOINT")
        .map(PathBuf::from)
        .expect("RILEY_REAL_CHECKPOINT must name the remote checkpoint directory");
    let model = LoadedModel::load(&checkpoint, LoadLimits::default())
        .expect("pinned SmolLM2 checkpoint must load without Python");

    assert_eq!(
        model.provenance().source_model(),
        "HuggingFaceTB/SmolLM2-135M"
    );
    assert_eq!(
        model.provenance().source_revision(),
        "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
    );
    assert_eq!(model.spec().blocks().len(), 30);
    assert_eq!(model.spec().embedding().vocabulary_size(), 49_152);
    assert_eq!(model.spec().embedding().hidden_size(), 576);
    for (path, bytes, sha256) in [
        (
            "config.json",
            704,
            "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843",
        ),
        (
            "model.safetensors",
            269_060_552,
            "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
        ),
        (
            "tokenizer.json",
            2_104_556,
            "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
        ),
    ] {
        let file = model
            .provenance()
            .files()
            .get(Path::new(path))
            .expect("pinned artifact assertion");
        assert_eq!(file.byte_len(), bytes);
        assert_eq!(file.sha256(), sha256);
    }
    assert_eq!(
        model.verified_files(),
        &BTreeSet::from([
            PathBuf::from("config.json"),
            PathBuf::from("model.safetensors"),
            PathBuf::from("tokenizer.json"),
        ])
    );

    assert_reference_encoding(
        model.tokenizer(),
        "The quick brown fox checks a short English input.",
        &[504, 2365, 6354, 16438, 11139, 253, 1890, 2321, 3007, 30],
    );
    assert_reference_encoding(
        model.tokenizer(),
        "짧은 한국어 입력의 토큰화와 로짓을 확인합니다.",
        &[
            183, 117, 117, 26638, 218, 216, 33085, 246, 181, 130, 251, 183, 240, 129, 18601, 248,
            223, 182, 250, 115, 26638, 242, 48138, 224, 250, 184, 219, 125, 184, 243, 238, 183,
            243, 218, 25947, 111, 246, 183, 117, 237, 26638, 222, 48138, 243, 239, 26638, 133,
            33085, 119, 41259, 226, 41259, 114, 30,
        ],
    );
    assert_reference_encoding(
        model.tokenizer(),
        "let answer: i64 = (0x2A + 1_000 - 3) / 7; // []{}() <> == != && || π≈3.14159",
        &[
            2016, 2988, 42, 2056, 38, 36, 446, 365, 32, 104, 34, 49, 1232, 216, 33, 79, 32, 32, 32,
            731, 216, 35, 25, 2272, 216, 39, 43, 13241, 4389, 6150, 1000, 2067, 46, 1758, 6541,
            1456, 22, 14669, 25143, 40533, 226, 35, 30, 33, 36, 33, 37, 41,
        ],
    );
}

fn assert_reference_encoding(tokenizer: &dyn Tokenizer, text: &str, expected: &[u32]) {
    let actual = tokenizer
        .encode(text, EncodeOptions::default())
        .expect("Rust-native tokenizer encode must match the reference artifact");
    assert_eq!(actual, expected);
    assert_eq!(
        tokenizer
            .decode(&actual, DecodeOptions::default())
            .expect("Rust-native tokenizer decode must round trip"),
        text
    );
}

#[test]
fn aggregate_checks_payload_digests_before_parsing() {
    let config_checkpoint = TempCheckpoint::new();
    write_complete_checkpoint(config_checkpoint.root());
    corrupt_payload_without_changing_length(config_checkpoint.root(), "config.json");
    assert!(matches!(
        LoadedModel::load(config_checkpoint.root(), LoadLimits::default()),
        Err(ModelError::ChecksumMismatch { path, .. }) if path == Path::new("config.json")
    ));

    let tokenizer_checkpoint = TempCheckpoint::new();
    write_complete_checkpoint(tokenizer_checkpoint.root());
    corrupt_payload_without_changing_length(tokenizer_checkpoint.root(), "tokenizer.json");
    assert!(matches!(
        LoadedModel::load(tokenizer_checkpoint.root(), LoadLimits::default()),
        Err(ModelError::ChecksumMismatch { path, .. }) if path == Path::new("tokenizer.json")
    ));
}

#[test]
fn aggregate_rejects_unconsumed_or_undeclared_manifest_payloads() {
    let extra = TempCheckpoint::new();
    write_complete_checkpoint(extra.root());
    let unused = b"unused payload";
    fs::write(extra.root().join("unused.json"), unused).expect("write unconsumed payload");
    let mut manifest = read_manifest(extra.root());
    manifest["files"]
        .as_array_mut()
        .expect("manifest files array")
        .push(file_assertion("unused.json", unused));
    write_manifest_value(extra.root(), &manifest);
    let error = expect_load_error(
        extra.root(),
        "manifest payload not consumed by the aggregate loader must fail",
    );
    assert!(error.to_string().contains("file set mismatch"));
    assert!(error.to_string().contains("unused.json"));

    let missing = TempCheckpoint::new();
    write_complete_checkpoint(missing.root());
    let mut manifest = read_manifest(missing.root());
    manifest["files"]
        .as_array_mut()
        .expect("manifest files array")
        .retain(|entry| entry["path"] != "tokenizer.json");
    write_manifest_value(missing.root(), &manifest);
    let error = expect_load_error(
        missing.root(),
        "a consumed payload absent from provenance must fail",
    );
    assert!(error.to_string().contains("tokenizer.json"));
    assert!(
        error
            .to_string()
            .contains("absent from provenance manifest")
    );
}

#[test]
fn aggregate_cross_checks_dtype_vocabulary_and_special_tokens() {
    let dtype = TempCheckpoint::new();
    write_complete_checkpoint(dtype.root());
    replace_payload_and_assertion(dtype.root(), "tokenizer.json", b"not-json");
    let mut manifest = read_manifest(dtype.root());
    manifest["dtype"] = Value::String("f16".to_owned());
    write_manifest_value(dtype.root(), &manifest);
    let error = expect_load_error(
        dtype.root(),
        "manifest/config dtype mismatch must precede tokenizer parsing",
    );
    assert!(error.to_string().contains("manifest dtype f16 differs"));

    let vocabulary = TempCheckpoint::new();
    write_complete_checkpoint(vocabulary.root());
    let mut config: Value = serde_json::from_slice(&config_json()).expect("fixture config JSON");
    config["vocab_size"] = Value::from(9);
    let config = serde_json::to_vec(&config).expect("serialize mismatched config");
    replace_payload_and_assertion(vocabulary.root(), "config.json", &config);
    let error = expect_load_error(
        vocabulary.root(),
        "config/tokenizer vocabulary mismatch must fail",
    );
    assert!(error.to_string().contains("vocabulary size mismatch"));

    let special = TempCheckpoint::new();
    write_complete_checkpoint(special.root());
    let mut config: Value = serde_json::from_slice(&config_json()).expect("fixture config JSON");
    config["bos_token_id"] = Value::from(7);
    let config = serde_json::to_vec(&config).expect("serialize special-token mismatch");
    replace_payload_and_assertion(special.root(), "config.json", &config);
    let error = expect_load_error(special.root(), "a non-special BOS tokenizer ID must fail");
    assert!(
        error
            .to_string()
            .contains("bos_token_id 7 is not declared as a special added token")
    );
}

#[cfg(unix)]
#[test]
fn aggregate_rejects_final_component_symlinks() {
    use std::os::unix::fs::symlink;

    let checkpoint = TempCheckpoint::new();
    write_complete_checkpoint(checkpoint.root());
    let model_path = checkpoint.root().join("model.safetensors");
    let backing_path = checkpoint.root().join("backing.safetensors");
    fs::rename(&model_path, &backing_path).expect("move fixture model behind a symlink");
    symlink("backing.safetensors", &model_path).expect("create final-component symlink");

    assert!(matches!(
        LoadedModel::load(checkpoint.root(), LoadLimits::default()),
        Err(ModelError::UnsafePath { path }) if path == Path::new("model.safetensors")
    ));
}

fn assert_empty_executable_path() {
    let path = std::env::var_os("PATH").expect("CI gate supplies an empty PATH directory");
    let entries = std::env::split_paths(&path).collect::<Vec<_>>();
    assert_eq!(entries.len(), 1, "PATH must contain exactly one directory");
    assert!(
        entries[0].is_dir(),
        "PATH entry must be the CI-owned directory"
    );
    assert_eq!(
        fs::read_dir(&entries[0])
            .expect("inspect empty PATH directory")
            .count(),
        0,
        "PATH directory must be empty"
    );
    for command in ["python", "python3", "pip", "pip3"] {
        let error = Command::new(command)
            .status()
            .expect_err("forbidden executable unexpectedly resolved through PATH");
        assert_eq!(
            error.kind(),
            ErrorKind::NotFound,
            "unexpected {command} error"
        );
    }
}

fn expect_load_error(root: &Path, context: &str) -> ModelError {
    match LoadedModel::load(root, LoadLimits::default()) {
        Err(error) => error,
        Ok(_) => panic!("{context}"),
    }
}

fn write_complete_checkpoint(root: &Path) {
    let config = config_json();
    let tokenizer = tokenizer_json();
    let weights = serialize_safetensors(&canonical_tensors());
    fs::write(root.join("config.json"), &config).expect("write synthetic config");
    fs::write(root.join("tokenizer.json"), &tokenizer).expect("write synthetic tokenizer");
    fs::write(root.join("model.safetensors"), &weights).expect("write synthetic weights");

    let manifest = serde_json::to_vec_pretty(&json!({
        "format":"riley-checkpoint-v1",
        "source_model":"fixture/tiny-llama",
        "source_revision":"1111111111111111111111111111111111111111",
        "converter_revision":null,
        "transforms":[],
        "dtype":"bf16",
        "files":[
            file_assertion("config.json", &config),
            file_assertion("model.safetensors", &weights),
            file_assertion("tokenizer.json", &tokenizer)
        ]
    }))
    .expect("serialize synthetic provenance");
    fs::write(root.join("riley-checkpoint.json"), manifest).expect("write synthetic provenance");
}

fn read_manifest(root: &Path) -> Value {
    serde_json::from_slice(
        &fs::read(root.join("riley-checkpoint.json")).expect("read synthetic provenance"),
    )
    .expect("parse synthetic provenance")
}

fn write_manifest_value(root: &Path, manifest: &Value) {
    let bytes = serde_json::to_vec_pretty(manifest).expect("serialize synthetic provenance");
    fs::write(root.join("riley-checkpoint.json"), bytes).expect("rewrite synthetic provenance");
}

fn replace_payload_and_assertion(root: &Path, path: &str, bytes: &[u8]) {
    fs::write(root.join(path), bytes).expect("replace synthetic payload");
    let mut manifest = read_manifest(root);
    let entry = manifest["files"]
        .as_array_mut()
        .expect("manifest files array")
        .iter_mut()
        .find(|entry| entry["path"] == path)
        .expect("payload assertion exists");
    *entry = file_assertion(path, bytes);
    write_manifest_value(root, &manifest);
}

fn corrupt_payload_without_changing_length(root: &Path, path: &str) {
    let payload_path = root.join(path);
    let mut bytes = fs::read(&payload_path).expect("read synthetic payload for corruption");
    let first = bytes.first_mut().expect("synthetic payload is non-empty");
    *first ^= 1;
    fs::write(payload_path, bytes).expect("corrupt payload while preserving its length");
}

fn config_json() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "architectures":["LlamaForCausalLM"],
        "attention_bias":false,
        "bos_token_id":0,
        "eos_token_id":0,
        "hidden_act":"silu",
        "hidden_size":4,
        "intermediate_size":8,
        "max_position_embeddings":16,
        "model_type":"llama",
        "num_attention_heads":2,
        "num_hidden_layers":1,
        "num_key_value_heads":1,
        "rms_norm_eps":0.00001,
        "rope_scaling":null,
        "rope_theta":10000,
        "tie_word_embeddings":true,
        "torch_dtype":"bfloat16",
        "vocab_size":8
    }))
    .expect("serialize synthetic config")
}

fn tokenizer_json() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "version":"1.0",
        "truncation":null,
        "padding":null,
        "added_tokens":[{
            "id":0,
            "content":"<|endoftext|>",
            "single_word":false,
            "lstrip":false,
            "rstrip":false,
            "normalized":false,
            "special":true
        }],
        "normalizer":null,
        "pre_tokenizer":{
            "type":"Sequence",
            "pretokenizers":[
                {"type":"Digits","individual_digits":true},
                {
                    "type":"ByteLevel",
                    "add_prefix_space":false,
                    "trim_offsets":true,
                    "use_regex":true
                }
            ]
        },
        "post_processor":null,
        "decoder":{
            "type":"ByteLevel",
            "add_prefix_space":true,
            "trim_offsets":true,
            "use_regex":true
        },
        "model":{
            "type":"BPE",
            "dropout":null,
            "unk_token":null,
            "continuing_subword_prefix":null,
            "end_of_word_suffix":null,
            "fuse_unk":false,
            "byte_fallback":false,
            "ignore_merges":false,
            "vocab":{
                "<|endoftext|>":0,
                "h":1,
                "e":2,
                "l":3,
                "o":4,
                "Ġ":5,
                "1":6,
                "2":7
            },
            "merges":[]
        }
    }))
    .expect("serialize synthetic tokenizer")
}

fn canonical_tensors() -> BTreeMap<String, FixtureTensor> {
    let entries: &[(&str, &[usize])] = &[
        ("model.embed_tokens.weight", &[8, 4]),
        ("model.layers.0.input_layernorm.weight", &[4]),
        ("model.layers.0.self_attn.q_proj.weight", &[4, 4]),
        ("model.layers.0.self_attn.k_proj.weight", &[2, 4]),
        ("model.layers.0.self_attn.v_proj.weight", &[2, 4]),
        ("model.layers.0.self_attn.o_proj.weight", &[4, 4]),
        ("model.layers.0.post_attention_layernorm.weight", &[4]),
        ("model.layers.0.mlp.gate_proj.weight", &[8, 4]),
        ("model.layers.0.mlp.up_proj.weight", &[8, 4]),
        ("model.layers.0.mlp.down_proj.weight", &[4, 8]),
        ("model.norm.weight", &[4]),
    ];
    let mut tensors = BTreeMap::new();
    for (seed, (name, shape)) in entries.iter().enumerate() {
        let byte_len = shape.iter().product::<usize>() * 2;
        let bytes = (0..byte_len)
            .map(|index| {
                u8::try_from(seed)
                    .expect("small seed")
                    .wrapping_add(u8::try_from(index).expect("small fixture tensor"))
            })
            .collect();
        tensors.insert(
            (*name).to_owned(),
            FixtureTensor {
                shape: shape.to_vec(),
                bytes,
            },
        );
    }
    tensors
}

fn serialize_safetensors(tensors: &BTreeMap<String, FixtureTensor>) -> Vec<u8> {
    let mut header = Map::new();
    let mut data = Vec::new();
    for (name, tensor) in tensors {
        let start = data.len();
        data.extend_from_slice(&tensor.bytes);
        let end = data.len();
        header.insert(
            name.clone(),
            json!({
                "dtype":"BF16",
                "shape":tensor.shape,
                "data_offsets":[start,end]
            }),
        );
    }
    let header = serde_json::to_vec(&Value::Object(header)).expect("serialize safetensors header");
    let mut artifact = Vec::with_capacity(8 + header.len() + data.len());
    artifact.extend_from_slice(&(header.len() as u64).to_le_bytes());
    artifact.extend_from_slice(&header);
    artifact.extend_from_slice(&data);
    artifact
}

fn file_assertion(path: &str, bytes: &[u8]) -> Value {
    json!({"path":path,"bytes":bytes.len(),"sha256":sha256(bytes)})
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(output, "{byte:02x}").expect("writing to String is infallible");
    }
    output
}
