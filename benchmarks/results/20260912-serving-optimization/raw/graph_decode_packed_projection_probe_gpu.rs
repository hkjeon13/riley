//! Numerical feasibility experiment, not full-model or graph qualification.
//!
//! Declare as a child of `graph_decode_full` only in an isolated probe build:
//! `#[cfg(all(test, feature = "cuda"))]`
//! `#[path = "graph_decode_packed_projection_probe_gpu.rs"] mod packed_projection_probe_gpu;`
//!
//! Run `packed_projection_numerical_feasibility` with `--ignored --nocapture`
//! and `RILEY_REAL_CHECKPOINT`. Optional `RILEY_PACKED_PROJECTION_PROBE_OUTPUT`
//! names a new JSON file (existing files are never overwritten). The same JSON
//! is printed after the `RILEY_PACKED_PROJECTION_PROBE ` prefix.
//!
//! A completed experiment returns Ok even when `equal` is false. Read the JSON:
//! a passing Rust test means the probe completed, never that packing qualified.
//! CUDA/setup/cleanup errors fail the test. No timing or performance is measured.
use super::*;
use crate::llama::{
    LlamaBatchMetadataConfig, PreparedLlamaBatchExecutorConfig, PreparedLlamaForwardConfig,
};
use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaGemmConfig, CudaPreparedGemm,
    CudaRuntime, GemmParams,
};
use riley_model::{LoadLimits, LoadedModel};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::Write;

type ProbeResult<T = ()> = Result<T, Box<dyn std::error::Error>>;
const HIDDEN: usize = 576;
const LAYERS: usize = 30;
const PREFIX: &str = "RILEY_PACKED_PROJECTION_PROBE ";

#[derive(Clone, Copy)]
enum Projection {
    Query,
    Key,
    Value,
    Gate,
    Up,
}

impl Projection {
    const fn label(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Key => "key",
            Self::Value => "value",
            Self::Gate => "gate",
            Self::Up => "up",
        }
    }

    const fn width(self) -> usize {
        match self {
            Self::Query => HIDDEN,
            Self::Key | Self::Value => 192,
            Self::Gate | Self::Up => 1536,
        }
    }
}

struct Input {
    label: String,
    seed: Option<u32>,
    sign_inverted: bool,
    bytes: Vec<u8>,
}

fn bf16_bytes(words: impl IntoIterator<Item = u16>) -> Vec<u8> {
    words.into_iter().flat_map(u16::to_le_bytes).collect()
}

fn inputs() -> Vec<Input> {
    let mut result = vec![
        Input {
            label: "zero".to_owned(),
            seed: None,
            sign_inverted: false,
            bytes: vec![0; HIDDEN * 2],
        },
        Input {
            label: "unit_positive".to_owned(),
            seed: None,
            sign_inverted: false,
            bytes: bf16_bytes(std::iter::repeat_n(0x3f80, HIDDEN)),
        },
        Input {
            label: "unit_alternating".to_owned(),
            seed: None,
            sign_inverted: false,
            bytes: bf16_bytes((0..HIDDEN).map(|i| if i % 2 == 0 { 0x3f80 } else { 0xbf80 })),
        },
    ];
    for seed in [0x1234_5678_u32, 0x6d2b_79f5, 0xa341_316c] {
        let mut state = seed;
        let words: Vec<u16> = (0..HIDDEN)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                // Finite BF16 values with magnitudes in [0.25, 7.96875].
                let sign = ((state >> 16) & 0x8000) as u16;
                let exponent = (125 + (state >> 8) % 5) as u16;
                sign | (exponent << 7) | (state & 0x7f) as u16
            })
            .collect();
        for sign_inverted in [false, true] {
            result.push(Input {
                label: format!(
                    "seed_{seed:08x}_{}",
                    if sign_inverted { "negated" } else { "base" }
                ),
                seed: Some(seed),
                sign_inverted,
                bytes: bf16_bytes(
                    words
                        .iter()
                        .map(|&bits| bits ^ if sign_inverted { 0x8000 } else { 0 }),
                ),
            });
        }
    }
    result
}

fn sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn plan_record(plan: &CudaPreparedGemm) -> Value {
    let config = plan.config();
    let metadata = plan.algorithm_metadata();
    let (m, n, k) = metadata.dimensions();
    json!({
        "prepared": true,
        "m": m, "n": n, "k": k,
        "reduction_policy": config.reduction_policy().id(),
        "workspace_cap_bytes": config.max_workspace_bytes(),
        "backend_id": metadata.backend_id(), "algorithm_id": metadata.algorithm_id(),
        "tile_id": metadata.tile_id(), "stages_id": metadata.stages_id(),
        "split_k": metadata.split_k(), "reduction_scheme": metadata.reduction_scheme(),
        "cta_swizzling": metadata.cta_swizzling(), "custom_option": metadata.custom_option(),
        "workspace_bytes": metadata.workspace_bytes(), "deterministic": metadata.deterministic(),
        "numerical_implementation_flags": metadata.numerical_implementation_flags(),
        "compute_capability": metadata.compute_capability(),
        "runtime_version": metadata.runtime_version(), "cublaslt_version": metadata.cublaslt_version(),
        "effective_no_split": metadata.split_k() <= 1 && metadata.reduction_scheme() == 0,
    })
}

fn canonical(plan: &mut PreparedLlamaGemm) -> ProbeResult<&mut CudaPreparedGemm> {
    match plan {
        PreparedLlamaGemm::Canonical(plan) => Ok(plan),
        _ => Err("original selected plan is not canonical cuBLASLt".into()),
    }
}

struct PackedGroup {
    label: &'static str,
    projections: &'static [Projection],
    width: usize,
    plan: Option<CudaPreparedGemm>,
    weight: Option<CudaDeviceBuffer>,
    output: Option<CudaDeviceBuffer>,
    workspace: Option<CudaDeviceBuffer>,
}

impl PackedGroup {
    fn new(label: &'static str, projections: &'static [Projection]) -> Self {
        Self {
            label,
            projections,
            width: projections
                .iter()
                .map(|projection| projection.width())
                .sum(),
            plan: None,
            weight: None,
            output: None,
            workspace: None,
        }
    }
}

#[derive(Default)]
struct Resources {
    prepared: Option<PreparedLlamaBatchExecutor>,
    stream: Option<CudaStream>,
    io: Option<CudaPinnedHostBuffer>,
    groups: Vec<PackedGroup>,
}

fn close_result(
    error_list: &mut Vec<String>,
    label: &str,
    result: Result<(), impl std::fmt::Display>,
) {
    if let Err(error) = result {
        error_list.push(format!("{label}: {error}"));
    }
}

impl Resources {
    // Attempt every explicit close, including after a partial setup or execution failure.
    fn close(self) -> Vec<String> {
        let mut errors = Vec::new();
        for group in self.groups {
            if let Some(plan) = group.plan {
                close_result(&mut errors, group.label, plan.close());
            }
            for buffer in [group.weight, group.output, group.workspace]
                .into_iter()
                .flatten()
            {
                close_result(&mut errors, group.label, buffer.close());
            }
        }
        if let Some(prepared) = self.prepared {
            close_result(&mut errors, "original executor", prepared.close());
        }
        if let Some(io) = self.io {
            close_result(&mut errors, "probe staging", io.close());
        }
        if let Some(stream) = self.stream {
            close_result(&mut errors, "probe stream", stream.close());
        }
        errors
    }
}

fn read(
    buffer: &mut CudaDeviceBuffer,
    io: &mut CudaPinnedHostBuffer,
    stream: &mut CudaStream,
) -> ProbeResult<Vec<u8>> {
    let mut bytes = vec![0; usize::try_from(buffer.byte_len())?];
    buffer.download_to_slice(0, &mut bytes, io, stream)?;
    Ok(bytes)
}

#[allow(clippy::too_many_arguments)]
fn execute(
    plan: &mut CudaPreparedGemm,
    input: &CudaDeviceBuffer,
    weight: &CudaDeviceBuffer,
    output: &mut CudaDeviceBuffer,
    workspace: Option<&mut CudaDeviceBuffer>,
    io: &mut CudaPinnedHostBuffer,
    stream: &mut CudaStream,
) -> ProbeResult<Vec<u8>> {
    let config = plan.config();
    let algorithm = plan.algorithm_metadata();
    // Poison the complete destination so a no-op cannot pass the zero-input case.
    output.upload_from_slice(
        0,
        &vec![0xff; usize::try_from(config.output_bytes())?],
        io,
        stream,
    )?;
    let workspace = if algorithm.workspace_bytes() == 0 {
        None
    } else {
        Some(CudaBufferSpanMut::new(
            workspace.ok_or("selected workspace absent")?,
            CudaDType::U8,
            0,
            algorithm.workspace_bytes(),
        )?)
    };
    plan.execute(
        &mut GemmParams {
            input: CudaBufferSpan::new(input, CudaDType::BF16, 0, config.input_bytes())?,
            weight: CudaBufferSpan::new(weight, CudaDType::BF16, 0, config.weight_bytes())?,
            output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, config.output_bytes())?,
            workspace,
        },
        stream,
    )?;
    if plan.config() != config || plan.algorithm_metadata() != algorithm {
        return Err("selected plan changed during execution".into());
    }
    read(output, io, stream)
}

fn source_weight(
    prepared: &mut PreparedLlamaBatchExecutor,
    layer: usize,
    projection: Projection,
) -> ProbeResult<&mut CudaDeviceBuffer> {
    let forward = &mut prepared.owner.forward;
    let layer = &forward.plan.layers()[layer];
    let id = match projection {
        Projection::Query => layer.query_weight(),
        Projection::Key => layer.key_weight(),
        Projection::Value => layer.value_weight(),
        Projection::Gate => layer.gate_weight(),
        Projection::Up => layer.up_weight(),
    };
    Ok(forward.weights.borrow_graph_weight(id)?)
}

fn original_projection(
    prepared: &mut PreparedLlamaBatchExecutor,
    layer: usize,
    projection: Projection,
    io: &mut CudaPinnedHostBuffer,
    stream: &mut CudaStream,
) -> ProbeResult<Vec<u8>> {
    let forward = &mut prepared.owner.forward;
    let layer = &forward.plan.layers()[layer];
    let (plan, weight_id, output) = match projection {
        Projection::Query => (
            &mut forward.gemms.hidden,
            layer.query_weight(),
            &mut forward.buffers.hidden_projection,
        ),
        Projection::Key => (
            &mut forward.gemms.key_value,
            layer.key_weight(),
            &mut forward.buffers.key_raw,
        ),
        Projection::Value => (
            &mut forward.gemms.key_value,
            layer.value_weight(),
            &mut forward.buffers.value_raw,
        ),
        Projection::Gate => (
            &mut forward.gemms.intermediate,
            layer.gate_weight(),
            &mut forward.buffers.gate_raw,
        ),
        Projection::Up => (
            &mut forward.gemms.intermediate,
            layer.up_weight(),
            &mut forward.buffers.up_raw,
        ),
    };
    execute(
        canonical(plan)?,
        &forward.buffers.hidden_norm,
        forward.weights.borrow_graph_weight(weight_id)?,
        output,
        forward.buffers.gemm_workspace.as_mut(),
        io,
        stream,
    )
}

#[derive(Default)]
struct Difference {
    mismatches: usize,
    first: Option<Value>,
    baseline_nonfinite: usize,
    packed_nonfinite: usize,
    max_finite_abs_difference: f64,
}

fn difference(baseline: &[u8], packed: &[u8]) -> ProbeResult<Difference> {
    if baseline.len() != packed.len() || baseline.len() % 2 != 0 {
        return Err("projection comparison requires equal whole-BF16 byte lengths".into());
    }
    let mut result = Difference::default();
    for (index, (left, right)) in baseline
        .chunks_exact(2)
        .zip(packed.chunks_exact(2))
        .enumerate()
    {
        let left = u16::from_le_bytes([left[0], left[1]]);
        let right = u16::from_le_bytes([right[0], right[1]]);
        let left_finite = left & 0x7f80 != 0x7f80;
        let right_finite = right & 0x7f80 != 0x7f80;
        result.baseline_nonfinite += usize::from(!left_finite);
        result.packed_nonfinite += usize::from(!right_finite);
        if left != right {
            result.mismatches += 1;
            result.first.get_or_insert_with(|| {
                json!({
                    "element": index, "byte_offset": index * 2,
                    "baseline_bf16_hex": format!("{left:04x}"),
                    "packed_bf16_hex": format!("{right:04x}"),
                })
            });
        }
        if left_finite && right_finite {
            let delta = (f64::from(f32::from_bits(u32::from(left) << 16))
                - f64::from(f32::from_bits(u32::from(right) << 16)))
            .abs();
            result.max_finite_abs_difference = result.max_finite_abs_difference.max(delta);
        }
    }
    Ok(result)
}

struct ProjectionSummary {
    layer: usize,
    projection: Projection,
    inputs: usize,
    mismatches: usize,
    max_mismatches: usize,
    baseline_nonfinite: usize,
    packed_nonfinite: usize,
    max_abs_difference: f64,
    first: Option<Value>,
    mismatch_inputs: Vec<Value>,
    baseline_digest: Sha256,
    packed_digest: Sha256,
}

impl ProjectionSummary {
    fn new(layer: usize, projection: Projection) -> Self {
        Self {
            layer,
            projection,
            inputs: 0,
            mismatches: 0,
            max_mismatches: 0,
            baseline_nonfinite: 0,
            packed_nonfinite: 0,
            max_abs_difference: 0.,
            first: None,
            mismatch_inputs: Vec::new(),
            baseline_digest: Sha256::new(),
            packed_digest: Sha256::new(),
        }
    }

    fn compare(&mut self, input: &Input, baseline: &[u8], packed: &[u8]) -> ProbeResult<()> {
        if baseline.len() != self.projection.width() * 2 {
            return Err("original projection width differs from expected checkpoint shape".into());
        }
        let delta = difference(baseline, packed)?;
        self.inputs += 1;
        self.mismatches += delta.mismatches;
        self.max_mismatches = self.max_mismatches.max(delta.mismatches);
        self.baseline_nonfinite += delta.baseline_nonfinite;
        self.packed_nonfinite += delta.packed_nonfinite;
        self.max_abs_difference = self.max_abs_difference.max(delta.max_finite_abs_difference);
        if delta.mismatches != 0 || delta.baseline_nonfinite != 0 || delta.packed_nonfinite != 0 {
            let record = json!({
                "input": input.label, "mismatching_elements": delta.mismatches,
                "first_mismatch": delta.first, "baseline_nonfinite": delta.baseline_nonfinite,
                "packed_nonfinite": delta.packed_nonfinite,
                "max_finite_abs_difference": delta.max_finite_abs_difference,
            });
            if self.first.is_none() && delta.mismatches != 0 {
                self.first = Some(record.clone());
            }
            self.mismatch_inputs.push(record);
        }
        // Input order and each input's byte content bind both output digests.
        for digest in [&mut self.baseline_digest, &mut self.packed_digest] {
            digest.update(input.label.as_bytes());
            digest.update([0]);
            digest.update(&input.bytes);
        }
        self.baseline_digest.update(baseline);
        self.packed_digest.update(packed);
        Ok(())
    }

    fn into_json(self) -> Value {
        json!({
            "layer": self.layer, "projection": self.projection.label(),
            "width": self.projection.width(), "input_count": self.inputs,
            "compared_elements": self.inputs * self.projection.width(),
            "bitwise_equal": self.mismatches == 0,
            "finite": self.baseline_nonfinite == 0 && self.packed_nonfinite == 0,
            "mismatching_elements": self.mismatches,
            "max_mismatching_elements_per_input": self.max_mismatches,
            "first_mismatch": self.first, "mismatch_inputs": self.mismatch_inputs,
            "baseline_nonfinite": self.baseline_nonfinite, "packed_nonfinite": self.packed_nonfinite,
            "max_finite_abs_difference": self.max_abs_difference,
            "baseline_outputs_sha256": hex(&self.baseline_digest.finalize()),
            "packed_outputs_sha256": hex(&self.packed_digest.finalize()),
        })
    }
}

fn run(
    context: &CudaContext,
    resources: &mut Resources,
    report: &mut Value,
    inputs: &[Input],
) -> ProbeResult<()> {
    resources.stream = Some(context.create_stream()?);
    let checkpoint = std::env::var_os("RILEY_REAL_CHECKPOINT").ok_or("checkpoint missing")?;
    let model = LoadedModel::load(std::path::Path::new(&checkpoint), LoadLimits::default())?;
    let config = PreparedLlamaBatchExecutorConfig::new(
        LlamaBatchMetadataConfig::new(1, 1, 16, 1, 16)?,
        PreparedLlamaForwardConfig::default(),
    )
    .with_grouped_ragged_attention_heads()
    .with_separate_residual_norm()
    .with_iteration_batch_completion()
    .with_packed_async_metadata()
    .with_vllm_smol_p128_graph();
    resources.prepared = Some(PreparedLlamaBatchExecutor::prepare(
        &model,
        context,
        resources.stream.as_mut().expect("stream"),
        config,
    )?);
    let prepared = resources.prepared.as_mut().expect("prepared");
    if !prepared.full_decode_supported() {
        return Err("checkpoint does not support the existing M1 P128 profile".into());
    }
    let forward = &mut prepared.owner.forward;
    if forward.plan.layers().len() != LAYERS || forward.plan.dimensions().hidden_size() != HIDDEN {
        return Err("probe requires the actual 30-layer, hidden-576 checkpoint".into());
    }
    let mut workspace_cap = 0;
    let mut original_no_split = true;
    for (label, plan) in [
        ("query_and_output", &mut forward.gemms.hidden),
        ("key_and_value", &mut forward.gemms.key_value),
        ("gate_and_up", &mut forward.gemms.intermediate),
        ("down", &mut forward.gemms.down),
        ("lm_head", &mut forward.gemms.lm_head),
    ] {
        let plan = canonical(plan)?;
        workspace_cap = workspace_cap.max(plan.config().max_workspace_bytes());
        let metadata = plan.algorithm_metadata();
        original_no_split &= metadata.split_k() <= 1 && metadata.reduction_scheme() == 0;
        report["original_plans"][label] = plan_record(plan);
    }
    report["original_plans_effective_no_split"] = json!(original_no_split);
    resources.io = Some(context.allocate_pinned_host_buffer(4 * 1024 * 1024)?);
    resources.groups = vec![
        PackedGroup::new(
            "qkv",
            &[Projection::Query, Projection::Key, Projection::Value],
        ),
        PackedGroup::new("gate_up", &[Projection::Gate, Projection::Up]),
    ];
    for group in &mut resources.groups {
        // Generic strict-no-split selection is intentional: no anchored-N API,
        // private algorithm mutation, or tolerance-based fallback is involved.
        let config = CudaGemmConfig::new(1, group.width as u64, HIDDEN as u64, workspace_cap)?;
        match context.prepare_gemm(config) {
            Ok(plan) => {
                report["packed_plans"][group.label] = plan_record(&plan);
                group.plan = Some(plan);
            }
            Err(error) if error.kind() == riley_cuda::CudaErrorKind::NotSupported => {
                report["packed_plans"][group.label] = json!({
                    "prepared": false, "m": 1, "n": group.width, "k": HIDDEN,
                    "error": error.to_string(),
                });
                continue;
            }
            Err(error) => return Err(error.into()),
        }
        group.weight = Some(context.allocate_device_buffer(config.weight_bytes())?);
        group.output = Some(context.allocate_device_buffer(config.output_bytes())?);
        let bytes = group
            .plan
            .as_ref()
            .expect("plan")
            .algorithm_metadata()
            .workspace_bytes();
        if bytes != 0 {
            group.workspace = Some(context.allocate_device_buffer(bytes)?);
        }
    }
    let prepared = resources.prepared.as_mut().expect("prepared");
    let stream = resources.stream.as_mut().expect("stream");
    let io = resources.io.as_mut().expect("io");
    for group in &mut resources.groups {
        if group.plan.is_none() {
            continue;
        }
        for layer in 0..LAYERS {
            let mut packed_bytes = Vec::with_capacity(group.width * HIDDEN * 2);
            let mut weight_hashes = Vec::new();
            for &projection in group.projections {
                let bytes = read(source_weight(prepared, layer, projection)?, io, stream)?;
                if bytes.len() != projection.width() * HIDDEN * 2 {
                    return Err("source weight has an unexpected full-allocation shape".into());
                }
                weight_hashes.push(sha256(&bytes));
                report["weights"]
                    .as_array_mut()
                    .expect("weights")
                    .push(json!({
                        "layer": layer, "projection": projection.label(),
                        "shape": [projection.width(), HIDDEN], "bytes": bytes.len(),
                        "sha256": weight_hashes.last().expect("weight hash"),
                    }));
                packed_bytes.extend_from_slice(&bytes);
            }
            let weight = group.weight.as_mut().expect("packed weight");
            weight.upload_from_slice(0, &packed_bytes, io, stream)?;
            if read(weight, io, stream)? != packed_bytes {
                return Err(
                    "cold-packed weight readback differs from source row concatenation".into(),
                );
            }
            let mut summaries: Vec<_> = group
                .projections
                .iter()
                .map(|&p| ProjectionSummary::new(layer, p))
                .collect();
            for input in inputs {
                prepared
                    .owner
                    .forward
                    .buffers
                    .hidden_norm
                    .upload_from_slice(0, &input.bytes, io, stream)?;
                let mut originals = Vec::with_capacity(group.projections.len());
                for &projection in group.projections {
                    originals.push(original_projection(
                        prepared, layer, projection, io, stream,
                    )?);
                }
                let packed = execute(
                    group.plan.as_mut().expect("packed plan"),
                    &prepared.owner.forward.buffers.hidden_norm,
                    weight,
                    group.output.as_mut().expect("packed output"),
                    group.workspace.as_mut(),
                    io,
                    stream,
                )?;
                if read(&mut prepared.owner.forward.buffers.hidden_norm, io, stream)? != input.bytes
                {
                    return Err("projection modified the shared synthetic input".into());
                }
                let mut offset = 0;
                for (summary, original) in summaries.iter_mut().zip(&originals) {
                    let end = offset + summary.projection.width() * 2;
                    summary.compare(
                        input,
                        original,
                        packed
                            .get(offset..end)
                            .ok_or("packed output slice missing")?,
                    )?;
                    offset = end;
                }
                if offset != packed.len() {
                    return Err("packed output contains an unexamined suffix".into());
                }
            }
            // Verify all actual source weights and the new packed parent stayed immutable.
            for (&projection, hash) in group.projections.iter().zip(&weight_hashes) {
                if sha256(&read(
                    source_weight(prepared, layer, projection)?,
                    io,
                    stream,
                )?) != *hash
                {
                    return Err("original projection weight changed during execution".into());
                }
            }
            if read(weight, io, stream)? != packed_bytes {
                return Err("packed projection weight changed during execution".into());
            }
            report["packed_weights"]
                .as_array_mut()
                .expect("packed weights")
                .push(json!({
                    "layer": layer, "group": group.label, "shape": [group.width, HIDDEN],
                    "sha256": sha256(&packed_bytes), "readback_and_preservation_equal": true,
                }));
            report["layer_projection_results"]
                .as_array_mut()
                .expect("results")
                .extend(summaries.into_iter().map(ProjectionSummary::into_json));
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA and RILEY_REAL_CHECKPOINT; inspect JSON equal, not test status"]
fn packed_projection_numerical_feasibility() -> ProbeResult<()> {
    let mut artifact = std::env::var_os("RILEY_PACKED_PROJECTION_PROBE_OUTPUT")
        .map(|path| OpenOptions::new().write(true).create_new(true).open(path))
        .transpose()?;
    let inputs = inputs();
    let expected = LAYERS * 5 * inputs.len();
    let mut report = json!({
        "schema_version": "riley.packed-projection-numerical-probe.v1",
        "experiment_completed": false, "comparisons_complete": false, "equal": false,
        "qualification": "numerical-feasibility-only", "full_model_qualified": false,
        "graph_capture_tested": false, "performance_measured": false,
        "input_source": "deterministic synthetic BF16; not actual model activations",
        "layers": LAYERS, "expected_comparisons": expected,
        "inputs": inputs.iter().map(|input| json!({
            "id": input.label, "seed": input.seed, "sign_inverted": input.sign_inverted,
            "elements": HIDDEN, "sha256": sha256(&input.bytes),
        })).collect::<Vec<_>>(),
        "original_plans": {}, "packed_plans": {}, "weights": [], "packed_weights": [],
        "layer_projection_results": [], "execution_error": null, "cleanup_errors": [],
    });
    let mut resources = Resources::default();
    let mut context = None;
    let execution = (|| -> ProbeResult<()> {
        context = Some(CudaRuntime::initialize()?.device(0)?.create_context()?);
        run(
            context.as_ref().expect("context"),
            &mut resources,
            &mut report,
            &inputs,
        )
    })();
    report["experiment_completed"] = json!(execution.is_ok());
    report["execution_error"] = json!(execution.as_ref().err().map(ToString::to_string));
    let mut cleanup = resources.close();
    if let Some(context) = context {
        match context.allocation_stats() {
            Ok(stats) => {
                report["after_close_allocations"] = json!({
                    "device_live_bytes": stats.device_live_bytes(),
                    "device_live_allocations": stats.device_live_allocations(),
                    "pinned_host_live_bytes": stats.pinned_host_live_bytes(),
                    "pinned_host_live_allocations": stats.pinned_host_live_allocations(),
                    "zero": stats.is_zero(),
                });
                if !stats.is_zero() {
                    cleanup.push("tracked allocations remain after all explicit closes".to_owned());
                }
            }
            Err(error) => cleanup.push(format!("allocation stats: {error}")),
        }
    }
    let rows = report["layer_projection_results"]
        .as_array()
        .expect("results");
    let observed: u64 = rows
        .iter()
        .map(|row| row["input_count"].as_u64().expect("input count"))
        .sum();
    let mismatches: u64 = rows
        .iter()
        .map(|row| {
            row["mismatching_elements"]
                .as_u64()
                .expect("mismatch count")
        })
        .sum();
    let complete = rows.len() == LAYERS * 5 && observed == expected as u64 && execution.is_ok();
    let finite = rows.iter().all(|row| row["finite"] == true);
    report["observed_comparisons"] = json!(observed);
    report["comparisons_complete"] = json!(complete);
    report["mismatching_elements"] = json!(mismatches);
    report["finite"] = json!(complete && finite);
    report["equal"] = json!(complete && finite && mismatches == 0 && cleanup.is_empty());
    report["cleanup_errors"] = json!(cleanup);
    let encoded = serde_json::to_string(&report)?;
    println!("{PREFIX}{encoded}");
    if let Some(artifact) = artifact.as_mut() {
        artifact.write_all(encoded.as_bytes())?;
        artifact.write_all(b"\n")?;
        artifact.flush()?;
    }
    execution?;
    if !cleanup.is_empty() {
        return Err(format!("probe cleanup failed: {}", cleanup.join("; ")).into());
    }
    Ok(())
}

#[test]
fn packed_probe_difference_counts_every_raw_bf16_mismatch() -> ProbeResult<()> {
    let baseline = bf16_bytes([0x0000, 0x3f80, 0x4000, 0x7fc1]);
    let packed = bf16_bytes([0x8000, 0x3f81, 0x4001, 0x7fc1]);
    let delta = difference(&baseline, &packed)?;
    assert_eq!(
        delta.mismatches, 3,
        "signed zero is also an exact-byte mismatch"
    );
    assert_eq!(delta.first.as_ref().expect("mismatch")["element"], 0);
    assert_eq!(delta.baseline_nonfinite, 1);
    assert_eq!(delta.packed_nonfinite, 1);
    assert_eq!(delta.max_finite_abs_difference, 0.015625);
    assert!(difference(&baseline[..1], &packed[..1]).is_err());
    assert!(difference(&baseline, &packed[..2]).is_err());
    Ok(())
}

#[test]
fn packed_probe_inputs_are_finite_deterministic_and_sign_mirrored() {
    let generated = inputs();
    assert_eq!(generated.len(), 9);
    for (left, right) in generated.iter().zip(inputs()) {
        assert_eq!(left.bytes, right.bytes);
        assert_eq!(left.bytes.len(), HIDDEN * 2);
        assert!(
            left.bytes
                .chunks_exact(2)
                .all(|word| u16::from_le_bytes([word[0], word[1]]) & 0x7f80 != 0x7f80)
        );
    }
    for pair in generated[3..].chunks_exact(2) {
        assert_eq!(pair[0].seed, pair[1].seed);
        for (left, right) in pair[0]
            .bytes
            .chunks_exact(2)
            .zip(pair[1].bytes.chunks_exact(2))
        {
            assert_eq!(
                u16::from_le_bytes([left[0], left[1]]) ^ 0x8000,
                u16::from_le_bytes([right[0], right[1]])
            );
        }
    }
}
