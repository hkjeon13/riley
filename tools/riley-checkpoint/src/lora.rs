//! Merges a PEFT LoRA adapter into a BF16 checkpoint.
//!
//! The merged checkpoint is an ordinary one: serving pays nothing per token,
//! and no kernel, attention path or tokenizer changes. Each target weight
//! becomes `W + scale * (alpha / r) * B @ A`, computed in F32 from the F32
//! adapter weights and rounded to BF16 once, as PEFT's `merge_and_unload` does
//! for a BF16 base. The new manifest names the base in `source_model` and
//! binds the adapter and scale in `converter_revision`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use riley_model::PROVENANCE_FILENAME;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{Plan, Result, invalid, regular_file};

const ADAPTER_CONFIG: &str = "adapter_config.json";
const ADAPTER_WEIGHTS: &str = "adapter_model.safetensors";
const SAFETENSORS_HEADER_PREFIX: usize = 8;

fn lower_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// One tensor's location in a safetensors file.
struct TensorEntry {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

fn tensor_entries(file: &[u8]) -> Result<(usize, BTreeMap<String, TensorEntry>)> {
    let prefix: [u8; SAFETENSORS_HEADER_PREFIX] = file
        .get(..SAFETENSORS_HEADER_PREFIX)
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(|| invalid("safetensors file is truncated"))?;
    let header_len = usize::try_from(u64::from_le_bytes(prefix))
        .map_err(|_| invalid("safetensors header does not fit"))?;
    let data_start = SAFETENSORS_HEADER_PREFIX
        .checked_add(header_len)
        .filter(|end| *end <= file.len())
        .ok_or_else(|| invalid("safetensors header exceeds the file"))?;
    let header: BTreeMap<String, Value> =
        serde_json::from_slice(&file[SAFETENSORS_HEADER_PREFIX..data_start])?;
    let mut entries = BTreeMap::new();
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let dtype = value["dtype"].as_str().unwrap_or_default().to_owned();
        let shape: Vec<usize> = value["shape"]
            .as_array()
            .ok_or_else(|| invalid("safetensors tensor has no shape"))?
            .iter()
            .filter_map(Value::as_u64)
            .filter_map(|axis| usize::try_from(axis).ok())
            .collect();
        let offset = |index: usize| {
            value["data_offsets"][index]
                .as_u64()
                .and_then(|offset| usize::try_from(offset).ok())
                .and_then(|offset| offset.checked_add(data_start))
                .filter(|offset| *offset <= file.len())
                .ok_or_else(|| invalid("safetensors tensor offset is invalid"))
        };
        let (start, end) = (offset(0)?, offset(1)?);
        entries.insert(name, TensorEntry { dtype, shape, start, end });
    }
    Ok((data_start, entries))
}

fn to_f32(bytes: &[u8], dtype: &str) -> Result<Vec<f32>> {
    match dtype {
        "F32" => Ok(bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]]))
            .collect()),
        "BF16" => Ok(bytes
            .chunks_exact(2)
            .map(|value| bf16_to_f32(u16::from_le_bytes([value[0], value[1]])))
            .collect()),
        _ => Err(invalid("LoRA tensors must be F32 or BF16")),
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Rounds to nearest, ties to even, as PyTorch and BF16 hardware do.
fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return 0x7fc0;
    }
    let upper = (bits >> 16) as u16;
    let lower = bits & 0xffff;
    if lower > 0x8000 || (lower == 0x8000 && upper & 1 == 1) {
        upper + 1
    } else {
        upper
    }
}

/// `weight += scaling * B @ A`, for BF16 little-endian `weight`.
fn merge_into(
    weight: &mut [u8],
    rows: usize,
    columns: usize,
    a: &[f32],
    b: &[f32],
    rank: usize,
    scaling: f32,
) {
    let mut delta = vec![0.0_f32; columns];
    for row in 0..rows {
        delta.fill(0.0);
        for k in 0..rank {
            let left = b[row * rank + k];
            let right = &a[k * columns..(k + 1) * columns];
            for (cell, value) in delta.iter_mut().zip(right) {
                *cell += left * value;
            }
        }
        let start = row * columns * 2;
        for (column, value) in delta.iter().enumerate() {
            let at = start + column * 2;
            let base = bf16_to_f32(u16::from_le_bytes([weight[at], weight[at + 1]]));
            let merged = f32_to_bf16(base + value * scaling);
            weight[at..at + 2].copy_from_slice(&merged.to_le_bytes());
        }
    }
}

fn scaling_factor(config: &Value, scale: f32) -> Result<(f32, usize)> {
    if config["peft_type"].as_str() != Some("LORA") {
        return Err(invalid("adapter peft_type must be LORA"));
    }
    let rank = config["r"]
        .as_u64()
        .and_then(|rank| usize::try_from(rank).ok())
        .filter(|rank| *rank > 0)
        .ok_or_else(|| invalid("adapter r must be a positive integer"))?;
    let alpha = config["lora_alpha"]
        .as_f64()
        .ok_or_else(|| invalid("adapter lora_alpha is missing"))?;
    for (key, what) in [
        ("fan_in_fan_out", "fan_in_fan_out"),
        ("use_dora", "DoRA"),
    ] {
        if config[key].as_bool() == Some(true) {
            return Err(invalid(&format!("{what} adapters are not supported")));
        }
    }
    if config["bias"].as_str().is_some_and(|bias| bias != "none") {
        return Err(invalid("adapters with trained biases are not supported"));
    }
    for key in ["rank_pattern", "alpha_pattern"] {
        if config[key].as_object().is_some_and(|pattern| !pattern.is_empty()) {
            return Err(invalid("per-module rank or alpha is not supported"));
        }
    }
    if config["modules_to_save"].as_array().is_some_and(|saved| !saved.is_empty()) {
        return Err(invalid("modules_to_save is not supported"));
    }
    #[allow(clippy::cast_precision_loss)]
    let divisor = if config["use_rslora"].as_bool() == Some(true) {
        (rank as f64).sqrt()
    } else {
        rank as f64
    };
    #[allow(clippy::cast_possible_truncation)]
    Ok((scale * (alpha / divisor) as f32, rank))
}

/// The base-weight name a PEFT tensor key updates, and which factor it holds.
fn adapter_target(key: &str) -> Result<(String, bool)> {
    let module = key
        .strip_prefix("base_model.model.")
        .ok_or_else(|| invalid("adapter tensor name has no base_model.model. prefix"))?;
    for (suffix, is_a) in [(".lora_A.weight", true), (".lora_B.weight", false)] {
        if let Some(target) = module.strip_suffix(suffix) {
            return Ok((format!("{target}.weight"), is_a));
        }
    }
    Err(invalid("adapter holds a tensor other than lora_A and lora_B weights"))
}

/// Writes `destination`: `base` with `adapter` merged at `scale`.
///
/// # Errors
///
/// Returns an error for an unsupported adapter, a target missing from the
/// base, an existing destination, or a file failure.
pub fn merge_lora(base: &Path, adapter: &Path, destination: &Path, scale: f32) -> Result<()> {
    let plan = Plan::read(&base.join(PROVENANCE_FILENAME))?;
    let config_bytes = fs::read(adapter.join(ADAPTER_CONFIG))?;
    let config: Value = serde_json::from_slice(&config_bytes)?;
    let (scaling, rank) = scaling_factor(&config, scale)?;
    let adapter_file = fs::read(adapter.join(ADAPTER_WEIGHTS))?;
    let (_, adapter_entries) = tensor_entries(&adapter_file)?;
    let mut pairs: BTreeMap<String, [Option<&TensorEntry>; 2]> = BTreeMap::new();
    for (key, entry) in &adapter_entries {
        let (target, is_a) = adapter_target(key)?;
        pairs.entry(target).or_default()[usize::from(!is_a)] = Some(entry);
    }
    if pairs.is_empty() {
        return Err(invalid("adapter holds no tensors"));
    }

    fs::create_dir(destination)?;
    let mut remaining = pairs.len();
    let mut manifest: Value = serde_json::from_slice(&fs::read(base.join(PROVENANCE_FILENAME))?)?;
    let mut merged_files = BTreeMap::new();
    for entry in plan.provenance().files().values() {
        let path = entry.path();
        let source = base.join(path);
        regular_file(&source)?;
        let target = destination.join(path);
        if path.extension().is_none_or(|extension| extension != "safetensors") {
            fs::copy(&source, &target)?;
            continue;
        }
        let mut file = fs::read(&source)?;
        let (_, entries) = tensor_entries(&file)?;
        let mut changed = false;
        for (name, weight) in &entries {
            let Some(pair) = pairs.get(name) else { continue };
            let [Some(a), Some(b)] = *pair else {
                return Err(invalid("an adapter target lacks lora_A or lora_B"));
            };
            let [rows, columns] = weight.shape[..] else {
                return Err(invalid("a LoRA target must be a matrix"));
            };
            if weight.dtype != "BF16" {
                return Err(invalid("LoRA merge needs a BF16 base"));
            }
            if a.shape != [rank, columns] || b.shape != [rows, rank] {
                return Err(invalid("adapter shapes do not match the base weight"));
            }
            let a_values = to_f32(&adapter_file[a.start..a.end], &a.dtype)?;
            let b_values = to_f32(&adapter_file[b.start..b.end], &b.dtype)?;
            merge_into(
                &mut file[weight.start..weight.end],
                rows,
                columns,
                &a_values,
                &b_values,
                rank,
                scaling,
            );
            changed = true;
            remaining -= 1;
        }
        fs::write(&target, &file)?;
        if changed {
            merged_files.insert(
                path.to_string_lossy().into_owned(),
                lower_hex(&Sha256::digest(&file)),
            );
        }
    }
    if remaining != 0 {
        return Err(invalid("an adapter target is missing from the base"));
    }
    let mut binding = Sha256::new();
    binding.update(&config_bytes);
    binding.update(&adapter_file);
    binding.update(scale.to_le_bytes());
    let binding = lower_hex(&binding.finalize());
    manifest["converter_revision"] = Value::from(&binding[..40]);
    manifest["source_model"] = Value::from(format!(
        "{}-lora",
        manifest["source_model"].as_str().unwrap_or_default()
    ));
    for file in manifest["files"].as_array_mut().into_iter().flatten() {
        if let Some(sha256) = file["path"].as_str().and_then(|path| merged_files.get(path)) {
            file["sha256"] = Value::from(sha256.clone());
        }
    }
    fs::write(
        destination.join(PROVENANCE_FILENAME),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    crate::verify(destination)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounding_is_to_nearest_even() {
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
        assert_eq!(bf16_to_f32(f32_to_bf16(-2.5)), -2.5);
    }

    #[test]
    fn merge_adds_the_scaled_low_rank_product() {
        // W = [[1, 2], [3, 4]], A = [[1, 0]], B = [[2], [4]], scaling 0.5.
        let mut weight: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|value| f32_to_bf16(*value).to_le_bytes())
            .collect();
        merge_into(&mut weight, 2, 2, &[1.0, 0.0], &[2.0, 4.0], 1, 0.5);
        let merged = to_f32(&weight, "BF16").unwrap();
        assert_eq!(merged, [2.0, 2.0, 5.0, 4.0]);
    }

    #[test]
    fn scaling_reads_alpha_over_rank_and_refuses_what_it_cannot_merge() {
        let config = serde_json::json!({"peft_type": "LORA", "r": 4, "lora_alpha": 8});
        assert_eq!(scaling_factor(&config, 1.0).unwrap(), (2.0, 4));
        assert_eq!(scaling_factor(&config, 0.5).unwrap(), (1.0, 4));
        let rs = serde_json::json!({"peft_type": "LORA", "r": 4, "lora_alpha": 8, "use_rslora": true});
        assert_eq!(scaling_factor(&rs, 1.0).unwrap().0, 4.0);
        for extra in [
            serde_json::json!({"use_dora": true}),
            serde_json::json!({"fan_in_fan_out": true}),
            serde_json::json!({"bias": "all"}),
            serde_json::json!({"modules_to_save": ["lm_head"]}),
            serde_json::json!({"rank_pattern": {"x": 2}}),
        ] {
            let mut bad = config.clone();
            bad.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            assert!(scaling_factor(&bad, 1.0).is_err(), "{bad}");
        }
    }

    #[test]
    fn adapter_keys_map_to_base_weights() {
        let (target, is_a) = adapter_target(
            "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight",
        )
        .unwrap();
        assert_eq!(target, "model.layers.3.self_attn.q_proj.weight");
        assert!(is_a);
        assert!(adapter_target("base_model.model.x.lora_B.bias").is_err());
        assert!(adapter_target("model.x.lora_A.weight").is_err());
    }
}
