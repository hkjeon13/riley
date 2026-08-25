use std::collections::BTreeSet;

use serde::de::{DeserializeOwned, Error as _, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};

use crate::{ArtifactKind, ModelError, ModelResult};

pub(crate) fn from_slice<T: DeserializeOwned>(
    input: &[u8],
    artifact: ArtifactKind,
) -> ModelResult<T> {
    let mut deserializer = serde_json::Deserializer::from_slice(input);
    let value = UniqueValue::deserialize(&mut deserializer)
        .and_then(|value| {
            deserializer.end()?;
            Ok(value.0)
        })
        .map_err(|error| ModelError::InvalidJson {
            artifact,
            reason: error.to_string(),
        })?;

    serde_json::from_value(value).map_err(|error| ModelError::InvalidJson {
        artifact,
        reason: error.to_string(),
    })
}

struct UniqueValue(Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(UniqueValueVisitor)
    }
}

struct UniqueValueVisitor;

impl<'de> Visitor<'de> for UniqueValueVisitor {
    type Value = UniqueValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
        Number::from_f64(value)
            .map(Value::Number)
            .map(UniqueValue)
            .ok_or_else(|| E::custom("JSON number is not finite"))
    }

    fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
        self.visit_string(value.to_owned())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        UniqueValue::deserialize(deserializer)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<UniqueValue>()? {
            values.push(value.0);
        }
        Ok(UniqueValue(Value::Array(values)))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut object: A) -> Result<Self::Value, A::Error> {
        let mut keys = BTreeSet::new();
        let mut values = Map::new();
        while let Some(key) = object.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(A::Error::custom(format!("duplicate object key {key:?}")));
            }
            let value = object.next_value::<UniqueValue>()?;
            values.insert(key, value.0);
        }
        Ok(UniqueValue(Value::Object(values)))
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::from_slice;
    use crate::{ArtifactKind, ModelError};

    #[derive(Debug, Deserialize)]
    struct Wrapper {
        nested: serde_json::Value,
    }

    #[test]
    fn rejects_nested_duplicate_keys() {
        let error =
            from_slice::<Wrapper>(br#"{"nested":{"same":1,"same":2}}"#, ArtifactKind::Config)
                .expect_err("duplicate keys must fail");
        assert!(matches!(error, ModelError::InvalidJson { .. }));
        assert!(error.to_string().contains("duplicate object key \"same\""));
    }

    #[test]
    fn accepts_unique_nested_keys() {
        let parsed = from_slice::<Wrapper>(
            br#"{"nested":{"first":1,"second":2}}"#,
            ArtifactKind::Config,
        )
        .expect("unique keys must parse");
        assert_eq!(parsed.nested["second"], 2);
    }
}
