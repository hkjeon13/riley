use std::collections::BTreeMap;
use std::ops::Range;

use rustinfer_tensor::{DType, Shape};
use serde::Deserialize;
use serde_json::Value;

use crate::{ArtifactKind, LoadLimits, ModelError, ModelResult, strict_json};

const PREFIX_BYTES: usize = 8;

/// A validated safetensors shard whose tensor ranges borrow from owned bytes.
pub(crate) struct ParsedShard {
    tensors: BTreeMap<String, ParsedTensor>,
    bytes: Box<[u8]>,
}

impl ParsedShard {
    /// Parses and completely validates one in-memory safetensors shard.
    pub(crate) fn from_bytes(
        display_name: &str,
        bytes: Box<[u8]>,
        limits: LoadLimits,
    ) -> ModelResult<Self> {
        let (object, data_start) = parse_header(display_name, &bytes, limits)?;
        let tensor_count = object
            .len()
            .checked_sub(usize::from(object.contains_key("__metadata__")))
            .ok_or_else(|| ModelError::NumericOverflow {
                field: format!("{display_name} tensor count"),
            })?;
        if tensor_count > limits.tensors() {
            return Err(ModelError::LimitExceeded {
                resource: "safetensors tensors",
                limit: usize_to_u64(limits.tensors()),
                actual: Some(usize_to_u64(tensor_count)),
            });
        }

        if let Some(metadata) = object.get("__metadata__") {
            validate_metadata(display_name, metadata)?;
        }

        let mut tensors = BTreeMap::new();
        for (name, value) in &object {
            if name == "__metadata__" {
                continue;
            }
            if name.is_empty() || name.len() > 1024 || name.chars().any(char::is_control) {
                return Err(invalid(
                    display_name,
                    "tensor name must be bounded, non-empty, and printable",
                ));
            }
            let tensor = parse_tensor(
                display_name,
                name,
                value,
                data_start,
                bytes.len() - data_start,
                limits,
            )?;
            tensors.insert(name.clone(), tensor);
        }
        validate_coverage(display_name, &tensors, data_start, bytes.len())?;

        Ok(Self { tensors, bytes })
    }

    /// Returns tensors in deterministic name order.
    pub(crate) fn tensors(&self) -> &BTreeMap<String, ParsedTensor> {
        &self.tensors
    }

    /// Returns one tensor by its serialized name.
    pub(crate) fn tensor(&self, name: &str) -> Option<&ParsedTensor> {
        self.tensors.get(name)
    }

    /// Returns the validated storage bytes for one tensor.
    pub(crate) fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let tensor = self.tensor(name)?;
        self.bytes.get(tensor.byte_range.clone())
    }
}

/// Validated metadata for one tensor in a parsed shard.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ParsedTensor {
    dtype: DType,
    shape: Shape,
    byte_range: Range<usize>,
}

impl ParsedTensor {
    /// Returns the serialized scalar dtype.
    pub(crate) const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the validated logical shape.
    pub(crate) const fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the tensor's absolute byte range in the shard.
    pub(crate) fn byte_range(&self) -> Range<usize> {
        self.byte_range.clone()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

fn parse_header(
    display_name: &str,
    bytes: &[u8],
    limits: LoadLimits,
) -> ModelResult<(serde_json::Map<String, Value>, usize)> {
    enforce_u64_limit("safetensors shard", limits.shard_bytes(), bytes.len())?;
    let prefix = bytes
        .get(..PREFIX_BYTES)
        .ok_or_else(|| invalid(display_name, "missing 8-byte header-length prefix"))?;
    let header_len_u64 = u64::from_le_bytes(
        prefix
            .try_into()
            .map_err(|_| invalid(display_name, "invalid header-length prefix"))?,
    );
    if header_len_u64 > limits.safetensors_header_bytes() {
        return Err(ModelError::LimitExceeded {
            resource: "safetensors header",
            limit: limits.safetensors_header_bytes(),
            actual: Some(header_len_u64),
        });
    }
    let header_len = usize::try_from(header_len_u64).map_err(|_| ModelError::NumericOverflow {
        field: format!("{display_name} header length"),
    })?;
    let data_start =
        PREFIX_BYTES
            .checked_add(header_len)
            .ok_or_else(|| ModelError::NumericOverflow {
                field: format!("{display_name} data offset"),
            })?;
    let header = bytes
        .get(PREFIX_BYTES..data_start)
        .ok_or_else(|| invalid(display_name, "header length exceeds shard length"))?;
    if header.first() != Some(&b'{') {
        return Err(invalid(display_name, "JSON header must begin with '{'"));
    }
    let unpadded_end = header.iter().rposition(|byte| *byte != b' ');
    if unpadded_end.is_none_or(|index| header[index] != b'}') {
        return Err(invalid(
            display_name,
            "JSON header may only use ASCII spaces as trailing padding",
        ));
    }

    match strict_json::from_slice(header, ArtifactKind::Safetensors)? {
        Value::Object(object) => Ok((object, data_start)),
        _ => Err(invalid(display_name, "JSON header must be an object")),
    }
}

fn parse_tensor(
    display_name: &str,
    name: &str,
    value: &Value,
    data_start: usize,
    data_len: usize,
    limits: LoadLimits,
) -> ModelResult<ParsedTensor> {
    let raw: RawTensor = serde_json::from_value(value.clone()).map_err(|error| {
        invalid(
            display_name,
            format!("tensor {name:?} has invalid metadata: {error}"),
        )
    })?;
    let dtype = match raw.dtype.as_str() {
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        other => {
            return Err(invalid(
                display_name,
                format!("tensor {name:?} has unsupported dtype {other:?}"),
            ));
        }
    };
    if raw.shape.len() > limits.tensor_rank() {
        return Err(ModelError::LimitExceeded {
            resource: "safetensors tensor rank",
            limit: usize_to_u64(limits.tensor_rank()),
            actual: Some(usize_to_u64(raw.shape.len())),
        });
    }
    let dimensions = raw
        .shape
        .into_iter()
        .map(|extent| {
            usize::try_from(extent).map_err(|_| ModelError::NumericOverflow {
                field: format!("{display_name} tensor {name:?} shape"),
            })
        })
        .collect::<ModelResult<Vec<_>>>()?;
    let shape = Shape::new(dimensions).map_err(|_| ModelError::NumericOverflow {
        field: format!("{display_name} tensor {name:?} element count"),
    })?;
    let expected_len = shape
        .element_count()
        .checked_mul(dtype.size_bytes())
        .ok_or_else(|| ModelError::NumericOverflow {
            field: format!("{display_name} tensor {name:?} byte length"),
        })?;
    let start = checked_offset(display_name, name, "start", raw.data_offsets[0])?;
    let end = checked_offset(display_name, name, "end", raw.data_offsets[1])?;
    let actual_len = end.checked_sub(start).ok_or_else(|| {
        invalid(
            display_name,
            format!("tensor {name:?} has descending data offsets {start}..{end}"),
        )
    })?;
    if actual_len != expected_len {
        return Err(invalid(
            display_name,
            format!("tensor {name:?} byte length is {actual_len}, expected {expected_len}"),
        ));
    }
    if end > data_len {
        return Err(invalid(
            display_name,
            format!("tensor {name:?} data range {start}..{end} exceeds data length {data_len}"),
        ));
    }

    Ok(ParsedTensor {
        dtype,
        shape,
        byte_range: checked_absolute_range(display_name, name, data_start, start, end)?,
    })
}

fn checked_offset(display_name: &str, name: &str, kind: &str, value: u64) -> ModelResult<usize> {
    usize::try_from(value).map_err(|_| ModelError::NumericOverflow {
        field: format!("{display_name} tensor {name:?} {kind} offset"),
    })
}

fn checked_absolute_range(
    display_name: &str,
    name: &str,
    data_start: usize,
    start: usize,
    end: usize,
) -> ModelResult<Range<usize>> {
    let absolute_start =
        data_start
            .checked_add(start)
            .ok_or_else(|| ModelError::NumericOverflow {
                field: format!("{display_name} tensor {name:?} absolute start"),
            })?;
    let absolute_end = data_start
        .checked_add(end)
        .ok_or_else(|| ModelError::NumericOverflow {
            field: format!("{display_name} tensor {name:?} absolute end"),
        })?;
    Ok(absolute_start..absolute_end)
}

fn validate_coverage(
    display_name: &str,
    tensors: &BTreeMap<String, ParsedTensor>,
    data_start: usize,
    shard_len: usize,
) -> ModelResult<()> {
    let mut coverage = tensors.iter().collect::<Vec<_>>();
    coverage.sort_unstable_by(|left, right| {
        let left_range = &left.1.byte_range;
        let right_range = &right.1.byte_range;
        (left_range.start, left_range.end, left.0).cmp(&(
            right_range.start,
            right_range.end,
            right.0,
        ))
    });
    let mut cursor = data_start;
    for (name, tensor) in coverage {
        if tensor.byte_range.start != cursor {
            let kind = if tensor.byte_range.start < cursor {
                "overlaps"
            } else {
                "leaves a gap after"
            };
            return Err(invalid(
                display_name,
                format!("tensor {name:?} {kind} data offset {}", cursor - data_start),
            ));
        }
        cursor = tensor.byte_range.end;
    }
    if cursor != shard_len {
        return Err(invalid(
            display_name,
            format!(
                "tensor ranges cover {} of {} data bytes",
                cursor - data_start,
                shard_len - data_start
            ),
        ));
    }
    Ok(())
}

fn validate_metadata(display_name: &str, value: &Value) -> ModelResult<()> {
    let metadata = value.as_object().ok_or_else(|| {
        invalid(
            display_name,
            "__metadata__ must be a string-to-string object",
        )
    })?;
    for (key, value) in metadata {
        if !value.is_string() {
            return Err(invalid(
                display_name,
                format!("__metadata__ value for {key:?} must be a string"),
            ));
        }
    }
    Ok(())
}

fn enforce_u64_limit(resource: &'static str, limit: u64, actual: usize) -> ModelResult<()> {
    let actual = u64::try_from(actual).unwrap_or(u64::MAX);
    if actual > limit {
        return Err(ModelError::LimitExceeded {
            resource,
            limit,
            actual: Some(actual),
        });
    }
    Ok(())
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn invalid(display_name: &str, reason: impl Into<String>) -> ModelError {
    ModelError::InvalidArtifact {
        artifact: display_name.to_owned(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use rustinfer_tensor::DType;

    use super::ParsedShard;
    use crate::{LoadLimits, ModelError};

    fn shard(header: &str, data: &[u8]) -> Box<[u8]> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        bytes.into_boxed_slice()
    }

    fn parse(header: &str, data: &[u8]) -> Result<ParsedShard, ModelError> {
        ParsedShard::from_bytes(
            "fixture.safetensors",
            shard(header, data),
            LoadLimits::default(),
        )
    }

    #[test]
    fn parses_f16_bf16_scalar_empty_metadata_and_padding() {
        let header = concat!(
            r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[],"data_offsets":[4,6]},"empty":{"dtype":"F16","shape":[0,999999999999],"data_offsets":[6,6]}}"#,
            "  "
        );
        let parsed = parse(header, &[1, 2, 3, 4, 5, 6]).expect("valid shard");

        assert_eq!(parsed.tensors().len(), 3);
        assert_eq!(parsed.tensor("a").expect("a").dtype(), DType::F16);
        assert_eq!(parsed.tensor("a").expect("a").shape().dimensions(), &[2]);
        assert_eq!(
            parsed.tensor("a").expect("a").byte_range(),
            8 + header.len()..12 + header.len()
        );
        assert_eq!(parsed.tensor_bytes("a"), Some(&[1, 2, 3, 4][..]));
        assert_eq!(parsed.tensor_bytes("b"), Some(&[5, 6][..]));
        assert_eq!(parsed.tensor_bytes("empty"), Some(&[][..]));
    }

    #[test]
    fn every_truncation_returns_error_without_panicking() {
        let complete = shard(
            r#"{"a":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#,
            &[1, 2, 3, 4],
        );
        for length in 0..complete.len() {
            let truncated = complete[..length].to_vec().into_boxed_slice();
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                ParsedShard::from_bytes("truncated.safetensors", truncated, LoadLimits::default())
            }));
            assert!(outcome.is_ok(), "parser panicked at truncation {length}");
            assert!(
                outcome.expect("checked above").is_err(),
                "truncation {length} unexpectedly parsed"
            );
        }
    }

    #[test]
    fn rejects_duplicate_keys_at_every_depth() {
        for header in [
            r#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#,
            r#"{"a":{"dtype":"F16","dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#,
            r#"{"__metadata__":{"same":"1","same":"2"}}"#,
        ] {
            assert!(matches!(
                parse(header, &[0, 0]),
                Err(ModelError::InvalidJson { .. })
            ));
        }
    }

    #[test]
    fn rejects_unsupported_dtype_and_invalid_tensor_schema() {
        for header in [
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2],"extra":true}}"#,
            r#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0]}}"#,
        ] {
            assert!(matches!(
                parse(header, &[0; 4]),
                Err(ModelError::InvalidArtifact { .. })
            ));
        }
    }

    #[test]
    fn rejects_bad_prefix_header_and_metadata() {
        let oversized = (LoadLimits::default().safetensors_header_bytes() + 1)
            .to_le_bytes()
            .to_vec()
            .into_boxed_slice();
        assert!(matches!(
            ParsedShard::from_bytes("large", oversized, LoadLimits::default()),
            Err(ModelError::LimitExceeded { .. })
        ));
        assert!(matches!(
            parse(" []", &[]),
            Err(ModelError::InvalidArtifact { .. })
        ));
        assert!(matches!(
            parse("{}\n", &[]),
            Err(ModelError::InvalidArtifact { .. })
        ));
        assert!(matches!(
            parse("[]", &[]),
            Err(ModelError::InvalidArtifact { .. })
        ));
        assert!(matches!(
            parse(r#"{"__metadata__":{"bad":1}}"#, &[]),
            Err(ModelError::InvalidArtifact { .. })
        ));
        assert!(matches!(
            parse(r#"{"__metadata__":[]}"#, &[]),
            Err(ModelError::InvalidArtifact { .. })
        ));
    }

    #[test]
    fn rejects_bad_lengths_offsets_gaps_overlaps_and_trailing_data() {
        let cases: &[(&str, &[u8])] = &[
            (
                r#"{"a":{"dtype":"F16","shape":[2],"data_offsets":[0,2]}}"#,
                &[0, 0],
            ),
            (
                r#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[2,0]}}"#,
                &[0, 0],
            ),
            (
                r#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[1,3]}}"#,
                &[0, 0, 0],
            ),
            (
                r#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"F16","shape":[1],"data_offsets":[1,3]}}"#,
                &[0, 0, 0],
            ),
            (
                r#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#,
                &[0, 0, 0],
            ),
            (
                r#"{"a":{"dtype":"BF16","shape":[],"data_offsets":[0,2]}}"#,
                &[0],
            ),
        ];
        for (header, data) in cases {
            assert!(
                matches!(parse(header, data), Err(ModelError::InvalidArtifact { .. })),
                "unexpectedly accepted {header}"
            );
        }
    }

    #[test]
    fn rejects_rank_above_limit_and_shape_overflow() {
        let shape = vec!["1"; LoadLimits::default().tensor_rank() + 1].join(",");
        let header = format!(r#"{{"a":{{"dtype":"F16","shape":[{shape}],"data_offsets":[0,2]}}}}"#);
        assert!(matches!(
            parse(&header, &[0, 0]),
            Err(ModelError::LimitExceeded { .. })
        ));

        let header = format!(
            r#"{{"a":{{"dtype":"F16","shape":[{},2],"data_offsets":[0,0]}}}}"#,
            usize::MAX
        );
        assert!(matches!(
            parse(&header, &[]),
            Err(ModelError::NumericOverflow { .. })
        ));
    }
}
