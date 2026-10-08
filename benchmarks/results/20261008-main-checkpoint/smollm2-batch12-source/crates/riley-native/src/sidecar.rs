#![cfg_attr(all(test, not(feature = "cuda")), allow(dead_code))]

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

const SAFETENSORS_ALIGNMENT: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TensorDType {
    Bf16,
    F32,
}

impl TensorDType {
    pub(crate) const fn manifest_name(self) -> &'static str {
        match self {
            Self::Bf16 => "bfloat16",
            Self::F32 => "float32",
        }
    }

    const fn safetensors_name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
        }
    }

    const fn element_bytes(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TensorSpec {
    pub(crate) key: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: TensorDType,
}

impl TensorSpec {
    pub(crate) fn byte_len(&self) -> Result<u64, SidecarError> {
        if self.shape.is_empty() || self.shape.contains(&0) {
            return Err(SidecarError::InvalidTensor {
                key: self.key.clone(),
                reason: "shape must contain positive dimensions",
            });
        }
        self.shape
            .iter()
            .try_fold(self.dtype.element_bytes(), |bytes, &dimension| {
                bytes.checked_mul(u64::try_from(dimension).ok()?)
            })
            .ok_or_else(|| SidecarError::InvalidTensor {
                key: self.key.clone(),
                reason: "tensor byte length overflows u64",
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TensorLayout {
    data_offset: u64,
    byte_len: u64,
}

#[derive(Debug)]
pub(crate) enum SidecarError {
    Io(io::Error),
    Json(serde_json::Error),
    DuplicateKey(String),
    InvalidTensor {
        key: String,
        reason: &'static str,
    },
    UnknownKey(String),
    DuplicateWrite(String),
    InvalidWriteLength {
        key: String,
        expected: u64,
        actual: usize,
    },
    MissingWrites(usize),
    ArithmeticOverflow,
}

impl fmt::Display for SidecarError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(source) => source.fmt(formatter),
            Self::Json(source) => source.fmt(formatter),
            Self::DuplicateKey(key) => write!(formatter, "duplicate safetensors key {key:?}"),
            Self::InvalidTensor { key, reason } => {
                write!(formatter, "invalid safetensors tensor {key:?}: {reason}")
            }
            Self::UnknownKey(key) => write!(formatter, "unknown safetensors key {key:?}"),
            Self::DuplicateWrite(key) => {
                write!(formatter, "safetensors tensor {key:?} was written twice")
            }
            Self::InvalidWriteLength {
                key,
                expected,
                actual,
            } => write!(
                formatter,
                "safetensors tensor {key:?} has {actual} bytes, expected {expected}"
            ),
            Self::MissingWrites(count) => {
                write!(
                    formatter,
                    "safetensors sidecar is missing {count} tensor writes"
                )
            }
            Self::ArithmeticOverflow => formatter.write_str("safetensors offset overflow"),
        }
    }
}

impl Error for SidecarError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(source) => Some(source),
            Self::Json(source) => Some(source),
            _ => None,
        }
    }
}

impl From<io::Error> for SidecarError {
    fn from(source: io::Error) -> Self {
        Self::Io(source)
    }
}

impl From<serde_json::Error> for SidecarError {
    fn from(source: serde_json::Error) -> Self {
        Self::Json(source)
    }
}

pub(crate) struct SafeTensorWriter {
    path: PathBuf,
    file: Option<File>,
    data_start: u64,
    layouts: BTreeMap<String, TensorLayout>,
    written: BTreeSet<String>,
    committed: bool,
}

impl SafeTensorWriter {
    pub(crate) fn create(path: &Path, specs: &[TensorSpec]) -> Result<Self, SidecarError> {
        let mut layouts = BTreeMap::new();
        let mut header = Map::new();
        let mut data_offset = 0_u64;
        for spec in specs {
            let byte_len = spec.byte_len()?;
            let end = data_offset
                .checked_add(byte_len)
                .ok_or(SidecarError::ArithmeticOverflow)?;
            if layouts
                .insert(
                    spec.key.clone(),
                    TensorLayout {
                        data_offset,
                        byte_len,
                    },
                )
                .is_some()
            {
                return Err(SidecarError::DuplicateKey(spec.key.clone()));
            }
            header.insert(
                spec.key.clone(),
                json!({
                    "dtype": spec.dtype.safetensors_name(),
                    "shape": spec.shape,
                    "data_offsets": [data_offset, end],
                }),
            );
            data_offset = end;
        }
        let mut header_bytes = serde_json::to_vec(&Value::Object(header))?;
        let padding = (SAFETENSORS_ALIGNMENT - header_bytes.len() % SAFETENSORS_ALIGNMENT)
            % SAFETENSORS_ALIGNMENT;
        header_bytes.resize(header_bytes.len() + padding, b' ');
        let header_len =
            u64::try_from(header_bytes.len()).map_err(|_| SidecarError::ArithmeticOverflow)?;
        let data_start = 8_u64
            .checked_add(header_len)
            .ok_or(SidecarError::ArithmeticOverflow)?;
        let file_len = data_start
            .checked_add(data_offset)
            .ok_or(SidecarError::ArithmeticOverflow)?;
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let initialize = (|| -> Result<(), SidecarError> {
            file.write_all(&header_len.to_le_bytes())?;
            file.write_all(&header_bytes)?;
            file.set_len(file_len)?;
            Ok(())
        })();
        if let Err(error) = initialize {
            drop(file);
            let _ = std::fs::remove_file(path);
            return Err(error);
        }
        Ok(Self {
            path: path.to_path_buf(),
            file: Some(file),
            data_start,
            layouts,
            written: BTreeSet::new(),
            committed: false,
        })
    }

    pub(crate) fn write_tensor(&mut self, key: &str, bytes: &[u8]) -> Result<(), SidecarError> {
        let layout = self
            .layouts
            .get(key)
            .copied()
            .ok_or_else(|| SidecarError::UnknownKey(key.to_owned()))?;
        if self.written.contains(key) {
            return Err(SidecarError::DuplicateWrite(key.to_owned()));
        }
        if u64::try_from(bytes.len()).ok() != Some(layout.byte_len) {
            return Err(SidecarError::InvalidWriteLength {
                key: key.to_owned(),
                expected: layout.byte_len,
                actual: bytes.len(),
            });
        }
        let absolute = self
            .data_start
            .checked_add(layout.data_offset)
            .ok_or(SidecarError::ArithmeticOverflow)?;
        let file = self.file.as_mut().ok_or(SidecarError::MissingWrites(1))?;
        file.seek(SeekFrom::Start(absolute))?;
        file.write_all(bytes)?;
        self.written.insert(key.to_owned());
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<(), SidecarError> {
        let missing = self.layouts.len().saturating_sub(self.written.len());
        if missing != 0 {
            return Err(SidecarError::MissingWrites(missing));
        }
        if let Some(file) = self.file.take() {
            file.sync_all()?;
        }
        self.committed = true;
        Ok(())
    }
}

impl Drop for SafeTensorWriter {
    fn drop(&mut self) {
        self.file.take();
        if !self.committed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Read;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::Value;

    use super::{SafeTensorWriter, SidecarError, TensorDType, TensorSpec};

    fn temporary_path(name: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("riley-native-{name}-{nonce}.safetensors"))
    }

    #[test]
    fn writes_canonical_header_and_random_access_payloads() {
        let path = temporary_path("sidecar");
        let specs = [
            TensorSpec {
                key: "z/bf16".to_owned(),
                shape: vec![2, 2],
                dtype: TensorDType::Bf16,
            },
            TensorSpec {
                key: "a/f32".to_owned(),
                shape: vec![2],
                dtype: TensorDType::F32,
            },
        ];
        let mut writer = SafeTensorWriter::create(&path, &specs).expect("create");
        writer
            .write_tensor("a/f32", &[9, 10, 11, 12, 13, 14, 15, 16])
            .expect("out-of-order write");
        writer
            .write_tensor("z/bf16", &[1, 2, 3, 4, 5, 6, 7, 8])
            .expect("write");
        writer.finish().expect("finish");

        let mut file = fs::File::open(&path).expect("open");
        let mut length = [0_u8; 8];
        file.read_exact(&mut length).expect("length");
        let header_len = usize::try_from(u64::from_le_bytes(length)).expect("usize");
        assert_eq!(header_len % 8, 0);
        let mut header = vec![0_u8; header_len];
        file.read_exact(&mut header).expect("header");
        let parsed: Value = serde_json::from_slice(&header).expect("JSON with space padding");
        assert_eq!(parsed["a/f32"]["dtype"], "F32");
        assert_eq!(parsed["a/f32"]["data_offsets"], serde_json::json!([8, 16]));
        assert_eq!(parsed["z/bf16"]["data_offsets"], serde_json::json!([0, 8]));
        let mut data = Vec::new();
        file.read_to_end(&mut data).expect("payload");
        assert_eq!(
            data,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
        fs::remove_file(path).expect("remove");
    }

    #[test]
    fn failed_or_incomplete_writes_are_removed() {
        let path = temporary_path("rollback");
        let specs = [TensorSpec {
            key: "only".to_owned(),
            shape: vec![1],
            dtype: TensorDType::F32,
        }];
        let mut writer = SafeTensorWriter::create(&path, &specs).expect("create");
        assert!(matches!(
            writer.write_tensor("only", &[0]),
            Err(SidecarError::InvalidWriteLength { .. })
        ));
        drop(writer);
        assert!(!path.exists());
    }
}
