use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rustinfer_model::{
    CheckpointProvenance, LlamaConfig, LoadLimits, LoadedWeights, ModelError, WeightBinding,
    WeightSlot,
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
    fn new(label: &str) -> Self {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "rustinfer-pr05-{label}-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&root).expect("create isolated checkpoint directory");
        Self { root }
    }

    fn root(&self) -> &Path {
        &self.root
    }
}

impl Drop for TempCheckpoint {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).expect("remove isolated checkpoint directory");
    }
}

#[test]
fn single_shard_binds_all_slots_and_tied_alias_view() {
    let checkpoint = TempCheckpoint::new("single");
    let tensors = canonical_tensors(true, false);
    write_single_checkpoint(checkpoint.root(), &tensors);

    let spec = model_spec(true);
    let provenance = CheckpointProvenance::load(checkpoint.root(), LoadLimits::default()).unwrap();
    let loaded = LoadedWeights::load_weight_only(
        checkpoint.root(),
        &spec,
        provenance,
        LoadLimits::default(),
    )
    .unwrap();

    assert_eq!(loaded.bindings().len(), 12);
    assert_eq!(
        loaded.binding(WeightSlot::LmHead),
        Some(&WeightBinding::Alias(WeightSlot::TokenEmbedding))
    );
    let embedding = loaded.view(WeightSlot::TokenEmbedding).unwrap();
    let lm_head = loaded.view(WeightSlot::LmHead).unwrap();
    assert_eq!(embedding.view().shape().dimensions(), &[8, 4]);
    assert_eq!(lm_head.source().tensor_name(), "model.embed_tokens.weight");
    assert_eq!(
        embedding.view().storage().as_ptr(),
        lm_head.view().storage().as_ptr()
    );
    let binding_snapshot = loaded.binding_snapshot_json().unwrap();
    let expected_snapshot = include_str!("fixtures/tied-weight-bindings.json")
        .strip_suffix('\n')
        .expect("snapshot fixture has one final newline");
    assert_eq!(binding_snapshot, expected_snapshot);
    loaded
        .provenance()
        .require_exact_file_set(loaded.verified_weight_files())
        .unwrap();
}

#[test]
fn optional_physical_tied_head_must_be_byte_identical() {
    let accepted = TempCheckpoint::new("tied-identical");
    let tensors = canonical_tensors(true, true);
    write_single_checkpoint(accepted.root(), &tensors);
    let spec = model_spec(true);
    let provenance = CheckpointProvenance::load(accepted.root(), LoadLimits::default()).unwrap();
    LoadedWeights::load_weight_only(accepted.root(), &spec, provenance, LoadLimits::default())
        .expect("identical physical tied head is consumed as an alias");

    let rejected = TempCheckpoint::new("tied-mismatch");
    let mut tensors = canonical_tensors(true, true);
    tensors
        .get_mut("lm_head.weight")
        .expect("physical lm head")
        .bytes[0] ^= 0xff;
    write_single_checkpoint(rejected.root(), &tensors);
    let provenance = CheckpointProvenance::load(rejected.root(), LoadLimits::default()).unwrap();
    assert!(matches!(
        LoadedWeights::load_weight_only(rejected.root(), &spec, provenance, LoadLimits::default()),
        Err(ModelError::TiedWeightMismatch)
    ));
}

#[test]
fn missing_extra_and_shape_mismatches_are_distinct() {
    let spec = model_spec(true);

    let missing = TempCheckpoint::new("missing");
    let mut tensors = canonical_tensors(true, false);
    tensors.remove("model.layers.0.self_attn.q_proj.weight");
    write_single_checkpoint(missing.root(), &tensors);
    let provenance = CheckpointProvenance::load(missing.root(), LoadLimits::default()).unwrap();
    assert!(matches!(
        LoadedWeights::load_weight_only(
            missing.root(),
            &spec,
            provenance,
            LoadLimits::default()
        ),
        Err(ModelError::MissingTensors { names })
            if names == ["model.layers.0.self_attn.q_proj.weight"]
    ));

    let extra = TempCheckpoint::new("extra");
    let mut tensors = canonical_tensors(true, false);
    tensors.insert("unexpected.weight".to_owned(), tensor(&[1], 99));
    write_single_checkpoint(extra.root(), &tensors);
    let provenance = CheckpointProvenance::load(extra.root(), LoadLimits::default()).unwrap();
    assert!(matches!(
        LoadedWeights::load_weight_only(extra.root(), &spec, provenance, LoadLimits::default()),
        Err(ModelError::ExtraTensors { names }) if names == ["unexpected.weight"]
    ));

    let shape = TempCheckpoint::new("shape");
    let mut tensors = canonical_tensors(true, false);
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_owned(),
        tensor(&[2, 8], 7),
    );
    write_single_checkpoint(shape.root(), &tensors);
    let provenance = CheckpointProvenance::load(shape.root(), LoadLimits::default()).unwrap();
    assert!(matches!(
        LoadedWeights::load_weight_only(shape.root(), &spec, provenance, LoadLimits::default()),
        Err(ModelError::TensorShapeMismatch { name, .. })
            if name == "model.layers.0.self_attn.q_proj.weight"
    ));
}

#[test]
fn checksum_is_verified_before_checkpoint_parsing() {
    let checkpoint = TempCheckpoint::new("checksum");
    write_single_checkpoint(checkpoint.root(), &canonical_tensors(true, false));
    let model_path = checkpoint.root().join("model.safetensors");
    let mut bytes = fs::read(&model_path).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(&model_path, bytes).unwrap();

    let provenance = CheckpointProvenance::load(checkpoint.root(), LoadLimits::default()).unwrap();
    assert!(matches!(
        LoadedWeights::load_weight_only(
            checkpoint.root(),
            &model_spec(true),
            provenance,
            LoadLimits::default()
        ),
        Err(ModelError::ChecksumMismatch { path, .. }) if path == Path::new("model.safetensors")
    ));
}

#[test]
fn standalone_weight_load_rejects_unconsumed_manifest_files() {
    let checkpoint = TempCheckpoint::new("manifest-extra");
    write_single_checkpoint(checkpoint.root(), &canonical_tensors(true, false));
    let model = fs::read(checkpoint.root().join("model.safetensors")).unwrap();
    write_manifest(
        checkpoint.root(),
        &[
            file_assertion("model.safetensors", &model),
            file_assertion("config.json", b"{}"),
        ],
    );
    let provenance = CheckpointProvenance::load(checkpoint.root(), LoadLimits::default()).unwrap();
    let Err(error) = LoadedWeights::load_weight_only(
        checkpoint.root(),
        &model_spec(true),
        provenance,
        LoadLimits::default(),
    ) else {
        panic!("unconsumed manifest files must fail the standalone weight load");
    };
    assert!(error.to_string().contains("file set mismatch"));
    assert!(error.to_string().contains("config.json"));
}

#[test]
fn shuffled_two_shard_index_has_deterministic_bindings() {
    let checkpoint = TempCheckpoint::new("sharded");
    let tensors = canonical_tensors(false, true);
    write_sharded_checkpoint(checkpoint.root(), &tensors);
    let spec = model_spec(false);
    let provenance = CheckpointProvenance::load(checkpoint.root(), LoadLimits::default()).unwrap();
    let loaded = LoadedWeights::load_weight_only(
        checkpoint.root(),
        &spec,
        provenance,
        LoadLimits::default(),
    )
    .unwrap();

    assert!(matches!(
        loaded.binding(WeightSlot::LmHead),
        Some(WeightBinding::Tensor(source)) if source.tensor_name() == "lm_head.weight"
    ));
    let verified: BTreeSet<_> = [
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    .into_iter()
    .map(PathBuf::from)
    .collect();
    assert_eq!(loaded.verified_weight_files(), &verified);
    loaded
        .provenance()
        .require_exact_file_set(&verified)
        .unwrap();
    let snapshot = loaded.binding_snapshot_json().unwrap();
    assert!(snapshot.contains("model-00001-of-00002.safetensors"));
    assert!(snapshot.contains("model-00002-of-00002.safetensors"));
}

#[test]
fn shard_index_must_match_the_physical_inventory() {
    let checkpoint = TempCheckpoint::new("index-mismatch");
    write_sharded_checkpoint(checkpoint.root(), &canonical_tensors(false, true));
    let index_path = checkpoint.root().join("model.safetensors.index.json");
    let mut index: Value = serde_json::from_slice(&fs::read(&index_path).unwrap()).unwrap();
    index["weight_map"]["model.embed_tokens.weight"] =
        Value::String("model-00002-of-00002.safetensors".to_owned());
    let index_bytes = serde_json::to_vec(&index).unwrap();
    fs::write(&index_path, &index_bytes).unwrap();
    let first = fs::read(checkpoint.root().join("model-00001-of-00002.safetensors")).unwrap();
    let second = fs::read(checkpoint.root().join("model-00002-of-00002.safetensors")).unwrap();
    write_manifest(
        checkpoint.root(),
        &[
            file_assertion("model.safetensors.index.json", &index_bytes),
            file_assertion("model-00001-of-00002.safetensors", &first),
            file_assertion("model-00002-of-00002.safetensors", &second),
        ],
    );

    let provenance = CheckpointProvenance::load(checkpoint.root(), LoadLimits::default()).unwrap();
    let Err(error) = LoadedWeights::load_weight_only(
        checkpoint.root(),
        &model_spec(false),
        provenance,
        LoadLimits::default(),
    ) else {
        panic!("index-to-physical mismatch must fail");
    };
    assert!(error.to_string().contains("but index maps it"));
}

#[test]
fn ambiguous_single_and_sharded_layout_is_rejected_before_loading() {
    let checkpoint = TempCheckpoint::new("ambiguous");
    write_single_checkpoint(checkpoint.root(), &canonical_tensors(true, false));
    fs::write(
        checkpoint.root().join("model.safetensors.index.json"),
        b"{}",
    )
    .unwrap();
    let provenance = CheckpointProvenance::load(checkpoint.root(), LoadLimits::default()).unwrap();
    let Err(error) = LoadedWeights::load_weight_only(
        checkpoint.root(),
        &model_spec(true),
        provenance,
        LoadLimits::default(),
    ) else {
        panic!("ambiguous layouts must fail closed");
    };
    assert!(error.to_string().contains("both single-file and sharded"));
}

fn model_spec(tied: bool) -> rustinfer_model::ModelSpec {
    let config = format!(
        r#"{{
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
          "tie_word_embeddings":{tied},
          "torch_dtype":"bfloat16",
          "vocab_size":8
        }}"#
    );
    LlamaConfig::from_json_slice(config.as_bytes())
        .unwrap()
        .to_model_spec()
}

fn canonical_tensors(tied: bool, include_lm_head: bool) -> BTreeMap<String, FixtureTensor> {
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
        tensors.insert(
            (*name).to_owned(),
            tensor(shape, u8::try_from(seed).expect("small fixture seed")),
        );
    }
    if include_lm_head || !tied {
        let lm_head = if tied {
            tensors["model.embed_tokens.weight"].clone()
        } else {
            tensor(&[8, 4], 42)
        };
        tensors.insert("lm_head.weight".to_owned(), lm_head);
    }
    tensors
}

fn tensor(shape: &[usize], seed: u8) -> FixtureTensor {
    let byte_len = shape.iter().product::<usize>() * 2;
    let bytes = (0..byte_len)
        .map(|index| {
            seed.wrapping_add(u8::try_from(index).expect("tiny fixture tensor is below 256 bytes"))
        })
        .collect();
    FixtureTensor {
        shape: shape.to_vec(),
        bytes,
    }
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
    let header = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut artifact = Vec::with_capacity(8 + header.len() + data.len());
    artifact.extend_from_slice(&(header.len() as u64).to_le_bytes());
    artifact.extend_from_slice(&header);
    artifact.extend_from_slice(&data);
    artifact
}

fn write_single_checkpoint(root: &Path, tensors: &BTreeMap<String, FixtureTensor>) {
    let bytes = serialize_safetensors(tensors);
    fs::write(root.join("model.safetensors"), &bytes).unwrap();
    write_manifest(root, &[file_assertion("model.safetensors", &bytes)]);
}

fn write_sharded_checkpoint(root: &Path, tensors: &BTreeMap<String, FixtureTensor>) {
    let mut first = BTreeMap::new();
    let mut second = BTreeMap::new();
    let mut weight_map = Map::new();
    for (index, (name, tensor)) in tensors.iter().enumerate() {
        let (filename, destination) = if index % 2 == 0 {
            ("model-00002-of-00002.safetensors", &mut second)
        } else {
            ("model-00001-of-00002.safetensors", &mut first)
        };
        destination.insert(name.clone(), tensor.clone());
        weight_map.insert(name.clone(), Value::String(filename.to_owned()));
    }
    let first_bytes = serialize_safetensors(&first);
    let second_bytes = serialize_safetensors(&second);
    fs::write(root.join("model-00001-of-00002.safetensors"), &first_bytes).unwrap();
    fs::write(root.join("model-00002-of-00002.safetensors"), &second_bytes).unwrap();
    let total_size: usize = tensors.values().map(|tensor| tensor.bytes.len()).sum();
    let index = serde_json::to_vec(&json!({
        "metadata":{"total_size":total_size},
        "weight_map":weight_map
    }))
    .unwrap();
    fs::write(root.join("model.safetensors.index.json"), &index).unwrap();
    write_manifest(
        root,
        &[
            file_assertion("model.safetensors.index.json", &index),
            file_assertion("model-00001-of-00002.safetensors", &first_bytes),
            file_assertion("model-00002-of-00002.safetensors", &second_bytes),
        ],
    );
}

fn file_assertion(path: &str, bytes: &[u8]) -> Value {
    json!({"path":path,"bytes":bytes.len(),"sha256":sha256(bytes)})
}

fn write_manifest(root: &Path, files: &[Value]) {
    let manifest = serde_json::to_vec_pretty(&json!({
        "format":"rustinfer-checkpoint-v1",
        "source_model":"fixture/tiny-llama",
        "source_revision":"1111111111111111111111111111111111111111",
        "converter_revision":null,
        "transforms":[],
        "dtype":"bf16",
        "files":files
    }))
    .unwrap();
    fs::write(root.join("rustinfer-checkpoint.json"), manifest).unwrap();
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(output, "{byte:02x}").expect("writing to String is infallible");
    }
    output
}
