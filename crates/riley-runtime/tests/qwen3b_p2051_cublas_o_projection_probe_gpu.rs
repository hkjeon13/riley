//! Remote-only direct-cuBLAS qualifier for Qwen2.5-3B's eager O projection.
//!
//! The input and expected output are captured from the actual Hugging Face
//! ``layer0.self_attn.o_proj`` module.  This remains an isolated diagnostic:
//! it is not reachable from the ordinary GEMM selector, graph replay, or
//! serving batch path.

#![cfg(feature = "cuda-cublas-gemm-probe")]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    dead_code,
    unused_imports
)]

mod p2051_cublas_o_projection_probe_contract {
    include!("support/qwen3b_p2051_projection_p7_contract.rs");

    use riley_cuda::{CublasGemmProbeMetadata, CublasGemmProbeParams, CudaPreparedCublasGemmProbe};

    const O_MANIFEST_VARIABLE: &str = "RILEY_QWEN3B_P2051_O_PROJECTION_MANIFEST";
    const O_SIDECAR_VARIABLE: &str = "RILEY_QWEN3B_P2051_O_PROJECTION_SIDECAR";
    const O_OUTPUT_VARIABLE: &str = "RILEY_QWEN3B_P2051_CUBLAS_O_PROJECTION_PROBE_OUTPUT";

    const O_SCHEMA_VERSION: &str = "riley.qwen3b-hf-eager-p2051-o-projection-trace.v1";
    const O_ARTIFACT_KIND: &str = "qwen2.5-3b-hf-eager-bf16-p2051-cache-off-o-projection-trace";
    const O_TRACE_ID: &str = "qwen3b-p2051-cache-off-layer0-o-projection-boundary-v1";
    const O_RESULT_SCHEMA_VERSION: &str =
        "riley.qwen3b-p2051-direct-cublas-o-projection-boundary-comparison.v1";
    const O_RESULT_ARTIFACT_KIND: &str =
        "qwen2.5-3b-riley-p2051-direct-cublas-o-projection-boundary-comparison";
    const O_REFERENCE_COMPUTE_CAPABILITY: (u32, u32) = (8, 9);
    const O_TENSOR_NAMES: [&str; 2] = ["layer0.attention_context", "layer0.o_proj"];

    const O_SOURCE_RECORDS: [(&str, &str); 15] = [
        (
            "qwen_serving_oracle",
            "tools/python/reference/riley_reference/qwen3b_serving_oracle.py",
        ),
        (
            "teacher_forced_generation",
            "tools/python/reference/riley_reference/qwen3b_generation_trace.py",
        ),
        (
            "p2051_layer_stage_contract",
            "tools/python/reference/riley_reference/qwen3b_p2051_layer_stage_trace.py",
        ),
        (
            "stage_trace_support",
            "tools/python/reference/riley_reference/qwen3b_stage_trace.py",
        ),
        (
            "p2051_projection_trace",
            "tools/python/reference/riley_reference/qwen3b_p2051_projection_trace.py",
        ),
        (
            "p2051_o_projection_trace",
            "tools/python/reference/riley_reference/qwen3b_p2051_o_projection_trace.py",
        ),
        (
            "hf_calibration",
            "tools/python/reference/riley_reference/hf_calibration.py",
        ),
        (
            "rust_o_projection_consumer",
            "crates/riley-runtime/tests/qwen3b_p2051_cublas_o_projection_probe_gpu.rs",
        ),
        (
            "rust_o_projection_consumer_package",
            "crates/riley-runtime/Cargo.toml",
        ),
        (
            "rust_direct_cublas_wrapper",
            "crates/riley-cuda/src/gemm.rs",
        ),
        ("rust_direct_cublas_ffi", "crates/riley-cuda/src/ffi.rs"),
        (
            "rust_direct_cublas_kernel",
            "kernels/src/cublas_gemm_probe.cu",
        ),
        ("reference_project", "tools/python/reference/pyproject.toml"),
        ("reference_lock", "tools/python/reference/uv.lock"),
        ("reference_python", "tools/python/reference/.python-version"),
    ];

    #[derive(Debug)]
    struct OProjectionArtifact {
        manifest_path: PathBuf,
        manifest_sha256: String,
        sidecar_path: PathBuf,
        sidecar_sha256: String,
        checkpoint_receipt_filename: String,
        checkpoint_receipt_sha256: String,
        input_bf16_le: Vec<u8>,
        expected_bf16_le: Vec<u8>,
        input_token_ids_le_sha256: String,
        teacher_artifact_sha256: String,
        teacher_sidecar_sha256: String,
        teacher_full_token_ids_sha256: String,
        teacher_prefix_token_ids: Vec<u32>,
    }

    #[derive(Debug)]
    struct OutputProjectionWeight {
        native_bf16: Vec<u8>,
        provenance: Value,
    }

    #[derive(Debug)]
    struct ProbeExecution {
        device: Value,
        metadata: Value,
        output_bf16_le: Vec<u8>,
        repeated_output_bf16_exact: bool,
        allocation_accounting_unchanged: bool,
        reference_compute_capability_matches: bool,
    }

    fn o_shape(name: &str) -> TestResult<Vec<u64>> {
        let sequence = u64::try_from(CONTEXT_TOKEN_COUNT)?;
        let hidden = u64::try_from(QWEN3B_HIDDEN_SIZE)?;
        match name {
            "layer0.attention_context" | "layer0.o_proj" => Ok(vec![sequence, hidden]),
            _ => Err(format!("unknown O-projection trace tensor {name}").into()),
        }
    }

    fn o_rust_consumer() -> Value {
        json!({
            "api": "riley_cuda::CudaPreparedCublasGemmProbe",
            "execution": "direct-p2051-bf16-layer0-o-projection-qualification",
            "input_context_token_count": CONTEXT_TOKEN_COUNT,
            "input_tensor": "layer0.attention_context",
            "output_tensor": "layer0.o_proj",
            "operator": "actual Hugging Face eager layer0.self_attn.o_proj",
            "projection": "layer0.self_attn.o_proj",
            "bias": false,
            "dtype": "bfloat16",
            "source_path": "crates/riley-runtime/tests/qwen3b_p2051_cublas_o_projection_probe_gpu.rs",
            "sidecar_key_rule": "trace/{tensor_name.replace('.', '/')}",
            "tensor_layout": "full-sequence-bf16",
        })
    }

    fn o_execution_contract() -> Value {
        json!({
            "attention_implementation": "eager",
            "cache_free": true,
            "dtype": "bfloat16",
            "explicit_attention_mask": true,
            "explicit_input_ids": true,
            "explicit_position_ids": true,
            "inference_mode": true,
            "logits_to_keep": 1,
            "return_dict": true,
            "tf32_enabled": false,
            "use_cache": false,
        })
    }

    fn parse_o_sidecar(
        manifest: &Value,
        sidecar_path: &Path,
    ) -> TestResult<BTreeMap<String, Vec<u8>>> {
        let bytes = fs::read(sidecar_path)?;
        if bytes.len() < 8 {
            return Err("O-projection sidecar is too short".into());
        }
        let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
        if header_len == 0
            || header_len > MAX_SAFETENSORS_HEADER_BYTES
            || 8_usize
                .checked_add(header_len)
                .is_none_or(|end| end > bytes.len())
        {
            return Err("O-projection sidecar header length differs".into());
        }
        let data_start = 8 + header_len;
        let header: Value = serde_json::from_slice(&bytes[8..data_start])?;
        let header = header
            .as_object()
            .ok_or("O-projection sidecar header is not an object")?;
        let expected_keys = O_TENSOR_NAMES
            .into_iter()
            .map(|name| format!("trace/{}", name.replace('.', "/")))
            .collect::<BTreeSet<_>>();
        let actual_keys = header
            .keys()
            .filter(|name| name.as_str() != "__metadata__")
            .cloned()
            .collect::<BTreeSet<_>>();
        if actual_keys != expected_keys {
            return Err("O-projection sidecar tensor set differs".into());
        }
        let manifest_tensors = manifest["tensors"]
            .as_object()
            .ok_or("O-projection manifest tensors are missing")?;
        let mut ranges = Vec::with_capacity(O_TENSOR_NAMES.len());
        let mut tensors = BTreeMap::new();
        for name in O_TENSOR_NAMES {
            let shape = o_shape(name)?;
            let key = format!("trace/{}", name.replace('.', "/"));
            let reference = manifest_tensors
                .get(name)
                .ok_or("O-projection manifest tensor is missing")?;
            require_exact_fields(
                reference,
                &[
                    "key",
                    "shape",
                    "dtype",
                    "canonical_byte_order",
                    "bf16_le_sha256",
                    "bf16_le_bytes",
                ],
                "O-projection manifest tensor",
            )?;
            if reference.get("key").and_then(Value::as_str) != Some(key.as_str())
                || reference.get("dtype").and_then(Value::as_str) != Some("bfloat16")
                || reference
                    .get("canonical_byte_order")
                    .and_then(Value::as_str)
                    != Some("little-endian-u16")
                || reference.get("shape")
                    != Some(&Value::Array(
                        shape.iter().copied().map(Value::from).collect(),
                    ))
                || reference.get("bf16_le_bytes").and_then(Value::as_u64)
                    != Some(u64::try_from(shape_byte_len(&shape)?)?)
            {
                return Err(format!("O-projection tensor {name} metadata differs").into());
            }
            let expected_sha = json_sha256(
                reference
                    .get("bf16_le_sha256")
                    .ok_or("O-projection tensor SHA-256 is missing")?,
                "O-projection tensor SHA-256",
            )?;
            let entry = header
                .get(&key)
                .ok_or("O-projection sidecar tensor is missing")?;
            require_exact_fields(
                entry,
                &["dtype", "shape", "data_offsets"],
                "O-projection sidecar tensor",
            )?;
            if entry.get("dtype").and_then(Value::as_str) != Some("BF16")
                || entry.get("shape")
                    != Some(&Value::Array(
                        shape.iter().copied().map(Value::from).collect(),
                    ))
            {
                return Err(format!("O-projection sidecar tensor {name} metadata differs").into());
            }
            let offsets = entry
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or("O-projection sidecar offsets are missing")?;
            if offsets.len() != 2 {
                return Err("O-projection sidecar offset count differs".into());
            }
            let start = usize::try_from(offsets[0].as_u64().ok_or("O-projection offset start")?)?;
            let end = usize::try_from(offsets[1].as_u64().ok_or("O-projection offset end")?)?;
            let expected_bytes = shape_byte_len(&shape)?;
            if end < start || end - start != expected_bytes || data_start + end > bytes.len() {
                return Err(format!("O-projection sidecar tensor {name} range differs").into());
            }
            let raw = bytes[data_start + start..data_start + end].to_vec();
            if sha256_hex(&raw) != expected_sha {
                return Err(format!("O-projection sidecar tensor {name} hash differs").into());
            }
            validate_finite_bf16(&raw, &format!("O-projection tensor {name}"))?;
            ranges.push((start, end));
            tensors.insert(name.to_owned(), raw);
        }
        let mut expected_start = 0_usize;
        ranges.sort_unstable();
        for (start, end) in ranges {
            if start != expected_start {
                return Err("O-projection sidecar offsets are non-contiguous".into());
            }
            expected_start = end;
        }
        if data_start + expected_start != bytes.len() {
            return Err("O-projection sidecar has trailing bytes".into());
        }
        Ok(tensors)
    }

    fn load_o_projection_artifact(
        teacher: &TeacherPrefix,
        workload: &Workload,
    ) -> TestResult<OProjectionArtifact> {
        let manifest_path = regular_file(
            &required_path(O_MANIFEST_VARIABLE)?,
            "HF P2051 O-projection manifest",
        )?;
        let sidecar_path = regular_file(
            &required_path(O_SIDECAR_VARIABLE)?,
            "HF P2051 O-projection sidecar",
        )?;
        let payload = fs::read(&manifest_path)?;
        let manifest_sha256 = sha256_hex(&payload);
        let manifest: Value = serde_json::from_slice(&payload)?;
        require_exact_fields(
            &manifest,
            &[
                "schema_version",
                "artifact_kind",
                "trace_id",
                "performance_claim_eligible",
                "created_at",
                "producer",
                "trace_profile",
                "contract",
                "model",
                "provenance",
                "sidecar",
                "tensors",
            ],
            "HF P2051 O-projection manifest",
        )?;
        if manifest["schema_version"].as_str() != Some(O_SCHEMA_VERSION)
            || manifest["artifact_kind"].as_str() != Some(O_ARTIFACT_KIND)
            || manifest["trace_id"].as_str() != Some(O_TRACE_ID)
            || manifest["performance_claim_eligible"].as_bool() != Some(false)
        {
            return Err("HF P2051 O-projection manifest identity differs".into());
        }
        require_exact_fields(
            &manifest["trace_profile"],
            &["capture_domain", "id", "tensor_count", "rust_consumer"],
            "HF P2051 O-projection trace profile",
        )?;
        let profile = manifest["trace_profile"]
            .as_object()
            .ok_or("HF P2051 O-projection trace profile is missing")?;
        if profile.get("capture_domain").and_then(Value::as_str)
            != Some("cache-free-p2051-layer0-o-projection-boundary")
            || profile.get("id").and_then(Value::as_str) != Some(O_TRACE_ID)
            || profile.get("tensor_count").and_then(Value::as_u64)
                != Some(u64::try_from(O_TENSOR_NAMES.len())?)
            || profile.get("rust_consumer") != Some(&o_rust_consumer())
        {
            return Err("HF P2051 O-projection trace profile differs".into());
        }
        let producer = manifest["producer"]
            .as_object()
            .ok_or("HF P2051 O-projection producer is missing")?;
        if producer.get("implementation_id").and_then(Value::as_str)
            != Some("riley-python-qwen3b-hf-eager-p2051-o-projection-trace-v1")
        {
            return Err("HF P2051 O-projection producer identity differs".into());
        }
        for field in [
            "runtime_dependency_class",
            "torch_version",
            "transformers_version",
        ] {
            if producer
                .get(field)
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            {
                return Err("HF P2051 O-projection producer metadata differs".into());
            }
        }
        require_exact_fields(
            &manifest["model"],
            &[
                "checkpoint_path",
                "checkpoint_receipt_filename",
                "checkpoint_receipt_sha256",
            ],
            "HF P2051 O-projection model",
        )?;
        let model = manifest["model"]
            .as_object()
            .ok_or("HF P2051 O-projection model is missing")?;
        let checkpoint_receipt_filename = model
            .get("checkpoint_receipt_filename")
            .and_then(Value::as_str)
            .filter(|value| *value == CHECKPOINT_RECEIPT_FILENAME)
            .ok_or("HF P2051 O-projection checkpoint receipt filename differs")?
            .to_owned();
        let checkpoint_receipt_sha256 = json_sha256(
            model
                .get("checkpoint_receipt_sha256")
                .ok_or("HF P2051 O-projection checkpoint receipt SHA-256 is missing")?,
            "HF P2051 O-projection checkpoint receipt SHA-256",
        )?;
        let contract = manifest["contract"]
            .as_object()
            .ok_or("HF P2051 O-projection contract is missing")?;
        require_exact_fields(
            &manifest["contract"],
            &[
                "model_id",
                "model_revision",
                "workload",
                "execution",
                "input",
            ],
            "HF P2051 O-projection contract",
        )?;
        if contract.get("model_id").and_then(Value::as_str) != Some(QWEN3B_MODEL_ID)
            || contract.get("model_revision").and_then(Value::as_str) != Some(QWEN3B_REVISION)
            || contract.get("execution") != Some(&o_execution_contract())
        {
            return Err("HF P2051 O-projection execution contract differs".into());
        }
        let workload_contract = contract["workload"]
            .as_object()
            .ok_or("HF P2051 O-projection workload contract is missing")?;
        require_exact_fields(
            &contract["workload"],
            &[
                "schema_version",
                "case",
                "source_sha256",
                "prompt_token_count",
                "prompt_token_ids_le_u32_sha256",
            ],
            "HF P2051 O-projection workload contract",
        )?;
        if workload_contract
            .get("schema_version")
            .and_then(Value::as_str)
            != Some(QWEN3B_WORKLOAD_SCHEMA)
            || workload_contract.get("case").and_then(Value::as_str) != Some(QWEN3B_WORKLOAD_CASE)
            || workload_contract
                .get("source_sha256")
                .and_then(Value::as_str)
                != Some(QWEN3B_WORKLOAD_SHA256)
            || workload_contract
                .get("prompt_token_count")
                .and_then(Value::as_u64)
                != Some(u64::try_from(workload.prompt_token_ids.len())?)
            || workload_contract
                .get("prompt_token_ids_le_u32_sha256")
                .and_then(Value::as_str)
                != Some(QWEN3B_PROMPT_TOKEN_SHA256)
        {
            return Err("HF P2051 O-projection workload contract differs".into());
        }
        let input = contract["input"]
            .as_object()
            .ok_or("HF P2051 O-projection input contract is missing")?;
        require_exact_fields(
            &contract["input"],
            &[
                "construction",
                "context_token_count",
                "input_token_ids_le_u32_sha256",
                "prompt_token_count",
                "teacher_prefix_token_count",
                "teacher_prefix_token_ids",
                "teacher_prefix_token_ids_le_u32_sha256",
                "teacher_source",
            ],
            "HF P2051 O-projection input contract",
        )?;
        let expected_input = workload
            .prompt_token_ids
            .iter()
            .copied()
            .chain(teacher.token_ids.iter().copied())
            .collect::<Vec<_>>();
        let input_token_ids_le_sha256 = token_ids_sha256(&expected_input);
        if input.get("construction").and_then(Value::as_str)
            != Some("workload.prompt_token_ids+verified_hf_cache_off.teacher_token_ids[:3]")
            || input.get("context_token_count").and_then(Value::as_u64)
                != Some(u64::try_from(CONTEXT_TOKEN_COUNT)?)
            || input.get("prompt_token_count").and_then(Value::as_u64)
                != Some(u64::try_from(QWEN3B_PROMPT_TOKEN_COUNT)?)
            || input
                .get("teacher_prefix_token_count")
                .and_then(Value::as_u64)
                != Some(u64::try_from(TEACHER_PREFIX_TOKEN_COUNT)?)
            || input
                .get("input_token_ids_le_u32_sha256")
                .and_then(Value::as_str)
                != Some(input_token_ids_le_sha256.as_str())
            || json_u32_array(
                input
                    .get("teacher_prefix_token_ids")
                    .ok_or("HF P2051 O-projection teacher prefix is missing")?,
                "HF P2051 O-projection teacher prefix",
            )? != teacher.token_ids
        {
            return Err("HF P2051 O-projection input binding differs".into());
        }
        let teacher_source = input["teacher_source"]
            .as_object()
            .ok_or("HF P2051 O-projection teacher source is missing")?;
        require_exact_fields(
            &input["teacher_source"],
            &[
                "artifact_kind",
                "artifact_path",
                "artifact_schema_version",
                "artifact_sha256",
                "cache_mode",
                "cache_off_sidecar_path",
                "cache_off_sidecar_sha256",
                "cache_off_sidecar_tensor_key",
                "full_teacher_token_count",
                "full_teacher_token_ids_le_u32_sha256",
            ],
            "HF P2051 O-projection teacher source",
        )?;
        if teacher_source
            .get("artifact_schema_version")
            .and_then(Value::as_str)
            != Some(TEACHER_ARTIFACT_SCHEMA)
            || teacher_source.get("artifact_kind").and_then(Value::as_str)
                != Some(TEACHER_ARTIFACT_KIND)
            || teacher_source
                .get("artifact_sha256")
                .and_then(Value::as_str)
                != Some(teacher.artifact_sha256.as_str())
            || teacher_source.get("cache_mode").and_then(Value::as_str) != Some("cache-off")
            || teacher_source
                .get("cache_off_sidecar_sha256")
                .and_then(Value::as_str)
                != Some(teacher.cache_off_sidecar_sha256.as_str())
            || teacher_source
                .get("full_teacher_token_ids_le_u32_sha256")
                .and_then(Value::as_str)
                != Some(teacher.full_teacher_token_ids_sha256.as_str())
            || teacher_source
                .get("cache_off_sidecar_tensor_key")
                .and_then(Value::as_str)
                != Some(TEACHER_CACHE_OFF_SIDECAR_KEY)
            || teacher_source
                .get("full_teacher_token_count")
                .and_then(Value::as_u64)
                != Some(128)
        {
            return Err("HF P2051 O-projection teacher source binding differs".into());
        }
        require_exact_fields(
            &manifest["provenance"],
            &["source_repository"],
            "HF P2051 O-projection provenance",
        )?;
        let provenance = manifest["provenance"]["source_repository"]
            .as_object()
            .ok_or("HF P2051 O-projection source provenance is missing")?;
        require_exact_fields(
            &manifest["provenance"]["source_repository"],
            &[
                "git_revision",
                "source_dirty",
                "source_status_sha256",
                "sources",
            ],
            "HF P2051 O-projection source provenance",
        )?;
        if provenance.get("source_dirty").and_then(Value::as_bool) != Some(false) {
            return Err("HF P2051 O-projection source provenance is dirty".into());
        }
        let _revision = json_git_revision(
            provenance
                .get("git_revision")
                .ok_or("HF P2051 O-projection Git revision is missing")?,
            "HF P2051 O-projection Git revision",
        )?;
        let _status_sha = json_sha256(
            provenance
                .get("source_status_sha256")
                .ok_or("HF P2051 O-projection source status SHA-256 is missing")?,
            "HF P2051 O-projection source status SHA-256",
        )?;
        let sources = provenance
            .get("sources")
            .and_then(Value::as_object)
            .ok_or("HF P2051 O-projection source records are missing")?;
        let expected_source_names = O_SOURCE_RECORDS
            .iter()
            .map(|(name, _)| *name)
            .collect::<BTreeSet<_>>();
        if sources.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected_source_names {
            return Err("HF P2051 O-projection source record set differs".into());
        }
        let root = repository_root()?;
        for (name, path) in O_SOURCE_RECORDS {
            validate_source_record(
                &root,
                sources
                    .get(name)
                    .ok_or("HF P2051 O-projection source record is missing")?,
                path,
                &format!("HF P2051 O-projection source {name}"),
            )?;
        }
        require_exact_fields(
            &manifest["sidecar"],
            &["path", "sha256", "format", "tensor_count"],
            "HF P2051 O-projection sidecar",
        )?;
        if manifest["sidecar"]["path"].as_str()
            != sidecar_path.file_name().and_then(|name| name.to_str())
            || manifest["sidecar"]["format"].as_str() != Some("safetensors")
            || manifest["sidecar"]["tensor_count"].as_u64()
                != Some(u64::try_from(O_TENSOR_NAMES.len())?)
        {
            return Err("HF P2051 O-projection sidecar metadata differs".into());
        }
        let sidecar_sha256 = sha256_file(&sidecar_path)?;
        if manifest["sidecar"]["sha256"].as_str() != Some(sidecar_sha256.as_str()) {
            return Err("HF P2051 O-projection sidecar hash differs".into());
        }
        let tensors = parse_o_sidecar(&manifest, &sidecar_path)?;
        Ok(OProjectionArtifact {
            manifest_path,
            manifest_sha256,
            sidecar_path,
            sidecar_sha256,
            checkpoint_receipt_filename,
            checkpoint_receipt_sha256,
            input_bf16_le: tensors
                .get("layer0.attention_context")
                .ok_or("O-projection attention context is missing")?
                .clone(),
            expected_bf16_le: tensors
                .get("layer0.o_proj")
                .ok_or("O-projection output is missing")?
                .clone(),
            input_token_ids_le_sha256,
            teacher_artifact_sha256: teacher.artifact_sha256.clone(),
            teacher_sidecar_sha256: teacher.cache_off_sidecar_sha256.clone(),
            teacher_full_token_ids_sha256: teacher.full_teacher_token_ids_sha256.clone(),
            teacher_prefix_token_ids: teacher.token_ids.clone(),
        })
    }

    fn load_o_projection_model(artifact: &OProjectionArtifact) -> TestResult<LoadedModel> {
        let checkpoint = regular_directory(
            &required_path("RILEY_QWEN3B_CHECKPOINT")?,
            "Qwen checkpoint",
        )?;
        let receipt = regular_file(
            &checkpoint.join(&artifact.checkpoint_receipt_filename),
            "Qwen checkpoint receipt",
        )?;
        if sha256_file(&receipt)? != artifact.checkpoint_receipt_sha256 {
            return Err("Qwen checkpoint receipt differs from O-projection artifact".into());
        }
        let model = LoadedModel::load(
            &checkpoint,
            LoadLimits::default().with_weight_byte_limits(8 * ONE_GIB, 8 * ONE_GIB)?,
        )?;
        let spec = model.spec();
        if model.config().family() != ModelFamily::Qwen2
            || model.provenance().source_model() != QWEN3B_MODEL_ID
            || model.provenance().source_revision() != QWEN3B_REVISION
            || spec.architecture() != ModelArchitecture::Llama
            || spec.source_architecture() != EXPECTED_SOURCE_ARCHITECTURE
            || spec.dtype() != DType::BF16
            || spec.blocks().len() != QWEN3B_LAYER_COUNT
            || spec.embedding().hidden_size() != QWEN3B_HIDDEN_SIZE
            || spec.embedding().vocabulary_size() != QWEN3B_VOCABULARY_SIZE
        {
            return Err("loaded model differs from Qwen2.5-3B O-projection contract".into());
        }
        Ok(model)
    }

    fn output_projection_weight(model: &LoadedModel) -> TestResult<OutputProjectionWeight> {
        let slot = WeightSlot::Decoder {
            layer: 0,
            parameter: DecoderWeight::OutputWeight,
        };
        let weight = model.weights().view(slot)?;
        let view = weight.view();
        let expected_shape = [QWEN3B_HIDDEN_SIZE, QWEN3B_HIDDEN_SIZE];
        let expected_bytes = QWEN3B_HIDDEN_SIZE
            .checked_mul(QWEN3B_HIDDEN_SIZE)
            .and_then(|elements| elements.checked_mul(BF16_BYTES))
            .ok_or("Qwen O-projection weight byte count overflows")?;
        if view.dtype() != DType::BF16
            || view.shape().dimensions() != expected_shape.as_slice()
            || view.storage().len() != expected_bytes
        {
            return Err("Qwen O-projection checkpoint weight binding differs".into());
        }
        let bytes = view.storage();
        Ok(OutputProjectionWeight {
            native_bf16: bf16_le_to_native(bytes)?,
            provenance: json!({
                "slot": slot.name(),
                "source_tensor": weight.source().tensor_name(),
                "source_shard": weight.source().shard_path().display().to_string(),
                "shape": view.shape().dimensions(),
                "raw_checkpoint_storage_sha256": sha256_hex(bytes),
                "bias": false,
            }),
        })
    }

    fn metadata_json(metadata: CublasGemmProbeMetadata) -> Value {
        json!({
            "backend_id": metadata.backend_id(),
            "math_modes": metadata.math_modes(),
            "pointer_modes": metadata.pointer_modes(),
            "atomics_modes": metadata.atomics_modes(),
            "runtime_version": metadata.runtime_version(),
            "cublas_version": metadata.cublas_version(),
            "compute_capability": metadata.compute_capability(),
            "dimensions": metadata.dimensions(),
        })
    }

    fn validate_probe_metadata(
        metadata: CublasGemmProbeMetadata,
        config: CudaGemmConfig,
        compute_capability: (u32, u32),
    ) -> TestResult {
        if metadata.backend_id() != CublasGemmProbeMetadata::BACKEND_ID
            || metadata.math_modes() != (0, 0)
            || metadata.pointer_modes() != (0, 0)
            || metadata.atomics_modes() != (0, 0)
            || metadata.runtime_version() <= 0
            || metadata.cublas_version() <= 0
            || metadata.compute_capability() != compute_capability
            || metadata.dimensions() != (config.m(), config.n(), config.k())
        {
            return Err("direct-cuBLAS O-projection metadata violates the contract".into());
        }
        Ok(())
    }

    fn execute_probe(
        plan: &mut CudaPreparedCublasGemmProbe,
        config: CudaGemmConfig,
        input: &CudaDeviceBuffer,
        weight: &CudaDeviceBuffer,
        output: &mut CudaDeviceBuffer,
        stream: &mut CudaStream,
    ) -> TestResult {
        let mut params = CublasGemmProbeParams {
            input: CudaBufferSpan::new(input, CudaDType::BF16, 0, config.input_bytes())?,
            weight: CudaBufferSpan::new(weight, CudaDType::BF16, 0, config.weight_bytes())?,
            output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, config.output_bytes())?,
        };
        plan.execute(&mut params, stream)?;
        Ok(())
    }

    fn run_direct_cublas_probe(
        artifact: &OProjectionArtifact,
        weight: &OutputProjectionWeight,
    ) -> TestResult<ProbeExecution> {
        let config = CudaGemmConfig::new(
            u64::try_from(CONTEXT_TOKEN_COUNT)?,
            u64::try_from(QWEN3B_HIDDEN_SIZE)?,
            u64::try_from(QWEN3B_HIDDEN_SIZE)?,
            MAX_WORKSPACE_BYTES,
        )?;
        if artifact.input_bf16_le.len() != usize::try_from(config.input_bytes())?
            || artifact.expected_bf16_le.len() != usize::try_from(config.output_bytes())?
            || weight.native_bf16.len() != usize::try_from(config.weight_bytes())?
        {
            return Err("O-projection host tensor lengths differ from GEMM config".into());
        }
        let runtime = CudaRuntime::initialize()?;
        if runtime.device_count() == 0 {
            return Err("remote GPU runner has no CUDA device".into());
        }
        let device = runtime.device(0)?;
        let properties = device.properties().clone();
        let reference_compute_capability_matches =
            properties.compute_capability() == O_REFERENCE_COMPUTE_CAPABILITY;
        let device_record = json!({
            "ordinal": properties.ordinal(),
            "name": properties.name(),
            "total_memory_bytes": properties.total_memory_bytes(),
            "compute_capability": properties.compute_capability(),
            "multiprocessor_count": properties.multiprocessor_count(),
            "driver_version": properties.driver_version(),
            "runtime_version": properties.runtime_version(),
            "reference_compute_capability": O_REFERENCE_COMPUTE_CAPABILITY,
            "reference_compute_capability_matches": reference_compute_capability_matches,
        });
        let context = device.create_context()?;
        let mut stream = context.create_stream()?;
        let result = (|| -> TestResult<ProbeExecution> {
            let mut staging = context.allocate_pinned_host_buffer(UPLOAD_STAGING_BYTES)?;
            let input_native = bf16_le_to_native(&artifact.input_bf16_le)?;
            let mut input = context.allocate_device_buffer(config.input_bytes())?;
            input.upload_from_slice(0, &input_native, &mut staging, &mut stream)?;
            let mut weight_device = context.allocate_device_buffer(config.weight_bytes())?;
            weight_device.upload_from_slice(0, &weight.native_bf16, &mut staging, &mut stream)?;
            let mut output = context.allocate_device_buffer(config.output_bytes())?;
            let mut plan = context.prepare_cublas_gemm_probe(config)?;
            let metadata = plan.metadata();
            if plan.config() != config {
                return Err("direct-cuBLAS O-projection plan config differs after prepare".into());
            }
            validate_probe_metadata(metadata, config, properties.compute_capability())?;
            let allocations_before = context.allocation_stats()?;
            execute_probe(
                &mut plan,
                config,
                &input,
                &weight_device,
                &mut output,
                &mut stream,
            )?;
            let mut first_native = vec![0_u8; usize::try_from(config.output_bytes())?];
            output.download_to_slice(0, &mut first_native, &mut staging, &mut stream)?;
            execute_probe(
                &mut plan,
                config,
                &input,
                &weight_device,
                &mut output,
                &mut stream,
            )?;
            let mut repeated_native = vec![0_u8; usize::try_from(config.output_bytes())?];
            output.download_to_slice(0, &mut repeated_native, &mut staging, &mut stream)?;
            let repeated_output_bf16_exact = first_native == repeated_native;
            let allocation_accounting_unchanged = context.allocation_stats()? == allocations_before;
            let output_bf16_le = bf16_native_to_le(&first_native)?;
            plan.close()?;
            input.close()?;
            weight_device.close()?;
            output.close()?;
            staging.close()?;
            if !context.allocation_stats()?.is_zero() {
                return Err(
                    "direct-cuBLAS O-projection probe left native allocations before context close"
                        .into(),
                );
            }
            Ok(ProbeExecution {
                device: device_record,
                metadata: metadata_json(metadata),
                output_bf16_le,
                repeated_output_bf16_exact,
                allocation_accounting_unchanged,
                reference_compute_capability_matches,
            })
        })();
        let cleanup = close_context(stream, context);
        match (result, cleanup) {
            (Ok(result), Ok(())) => Ok(result),
            (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
            (Err(run_error), Err(cleanup_error)) => Err(format!(
                "direct-cuBLAS O-projection qualifier failed: {run_error}; cleanup also failed: {cleanup_error}"
            )
            .into()),
        }
    }

    #[test]
    #[ignore = "remote-only P2051 direct-cuBLAS O-projection arithmetic qualifier"]
    fn qwen3b_p2051_direct_cublas_matches_actual_hf_o_projection() -> TestResult {
        let output = required_path(O_OUTPUT_VARIABLE)?;
        validate_output_destination(&output)?;
        let workload = load_workload()?;
        let teacher = load_teacher_prefix()?;
        let artifact = load_o_projection_artifact(&teacher, &workload)?;
        let model = load_o_projection_model(&artifact)?;
        let weight = output_projection_weight(&model)?;
        let execution = run_direct_cublas_probe(&artifact, &weight)?;
        let comparison = metrics(&artifact.expected_bf16_le, &execution.output_bf16_le)?;
        let exact = comparison.get("bf16_exact").and_then(Value::as_bool) == Some(true);
        let quality_pass = execution.reference_compute_capability_matches
            && execution.repeated_output_bf16_exact
            && execution.allocation_accounting_unchanged
            && exact;
        let root = repository_root()?;
        let source_paths = [
            "crates/riley-runtime/tests/qwen3b_p2051_cublas_o_projection_probe_gpu.rs",
            "crates/riley-cuda/src/gemm.rs",
            "crates/riley-cuda/src/ffi.rs",
            "kernels/src/cublas_gemm_probe.cu",
        ];
        let source_hashes = source_paths
            .into_iter()
            .map(|source| {
                let path = regular_file(&root.join(source), "direct-cuBLAS O-projection source")?;
                Ok((source, sha256_file(&path)?))
            })
            .collect::<TestResult<BTreeMap<_, _>>>()?;
        let receipt = json!({
            "schema_version": O_RESULT_SCHEMA_VERSION,
            "artifact_kind": O_RESULT_ARTIFACT_KIND,
            "created_at_unix_seconds": unix_seconds()?,
            "quality_pass": quality_pass,
            "performance_claim_eligible": false,
            "serving_selector_eligible": false,
            "corrected_cache_on_eligible": false,
            "immutable_hf_artifact": {
                "manifest_path": artifact.manifest_path,
                "manifest_sha256": artifact.manifest_sha256,
                "sidecar_path": artifact.sidecar_path,
                "sidecar_sha256": artifact.sidecar_sha256,
                "input_token_ids_le_u32_sha256": artifact.input_token_ids_le_sha256,
                "teacher_artifact_sha256": artifact.teacher_artifact_sha256,
                "teacher_sidecar_sha256": artifact.teacher_sidecar_sha256,
                "teacher_full_token_ids_le_u32_sha256": artifact.teacher_full_token_ids_sha256,
                "teacher_prefix_token_ids": artifact.teacher_prefix_token_ids,
            },
            "contract": {
                "trace_id": O_TRACE_ID,
                "endpoint": "layer0.self_attn.o_proj",
                "shape": {"m": CONTEXT_TOKEN_COUNT, "n": QWEN3B_HIDDEN_SIZE, "k": QWEN3B_HIDDEN_SIZE},
                "operator": "direct cublasGemmEx OP_T(Wc), OP_N(Xc), BF16 I/O, FP32 compute, CUBLAS_DEFAULT_MATH",
                "bias": false,
                "python_in_hot_path": false,
                "serving_selector_changed": false,
                "cuda_graph": false,
                "command_batch": false,
            },
            "device": execution.device,
            "checkpoint_weight": weight.provenance,
            "direct_cublas": {
                "metadata": execution.metadata,
                "output_bf16_le_sha256": sha256_hex(&execution.output_bf16_le),
                "repeated_output_bf16_exact": execution.repeated_output_bf16_exact,
                "allocation_accounting_unchanged": execution.allocation_accounting_unchanged,
                "actual_hf_o_projection_comparison": comparison,
            },
            "source_hashes": source_hashes,
            "summary": {
                "reference_compute_capability_matches": execution.reference_compute_capability_matches,
                "direct_cublas_matches_actual_hf_o_projection": exact,
                "interpretation": "A true result establishes only the pinned full-sequence layer-zero O-projection correspondence. It does not qualify residual, MLP, cache-on, serving behavior, or performance.",
                "serving_selector_changed": false,
                "performance_claim_eligible": false,
            },
        });
        write_artifact_exclusive(&output, &receipt)?;
        if !quality_pass {
            return Err(
                "direct-cuBLAS O-projection candidate did not meet its pinned quality gate".into(),
            );
        }
        println!(
            "QWEN3B_P2051_DIRECT_CUBLAS_O_PROJECTION trace_id={} quality_pass=true performance_claim_eligible=false",
            O_TRACE_ID,
        );
        Ok(())
    }
}
