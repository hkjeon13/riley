//! JSON Schema to byte-grammar compilation for structured output.
//!
//! Every keyword that constrains a value is either enforced by the grammar or
//! rejected with a named error: a schema must never look enforced when it is
//! not. Output is compact JSON with at most one space after `:` and `,`, and
//! object properties in schema order, which keeps a model from filling its
//! token budget with whitespace.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use riley_runtime::grammar::{ByteClass, Grammar, GrammarBuilder, GrammarElement};
use serde_json::{Map, Value};

/// Keywords that only annotate a schema.
const ANNOTATIONS: [&str; 10] = [
    "$schema",
    "$id",
    "$comment",
    "$defs",
    "definitions",
    "title",
    "description",
    "default",
    "examples",
    "strict",
];
/// Upper bound on expanded string lengths and array sizes.
const MAX_REPETITION: u64 = 1_024;
/// Bound on grammar rules for one schema.
const MAX_RULES: usize = 200_000;

/// Why a schema cannot be enforced; the text goes to the client.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchemaError(pub String);

impl std::fmt::Display for SchemaError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for SchemaError {}

fn unsupported(what: impl std::fmt::Display) -> SchemaError {
    SchemaError(format!("unsupported JSON Schema: {what}"))
}

fn bytes(text: &str) -> Vec<GrammarElement> {
    text.bytes()
        .map(|byte| GrammarElement::Bytes(ByteClass::byte(byte)))
        .collect()
}

fn class(first: u8, last: u8) -> GrammarElement {
    GrammarElement::Bytes(ByteClass::range(first, last))
}

const fn rule(id: u32) -> GrammarElement {
    GrammarElement::Rule(id)
}

fn concat(parts: Vec<Vec<GrammarElement>>) -> Vec<GrammarElement> {
    parts.into_iter().flatten().collect()
}

/// Rules shared by every schema, built on first use.
#[derive(Default)]
struct Common {
    string_char: Option<u32>,
    multibyte_char: Option<u32>,
    string: Option<u32>,
    value: Option<u32>,
    separator: Option<u32>,
    integer: Option<u32>,
    natural: Option<u32>,
    number: Option<u32>,
    unsigned_number: Option<u32>,
}

/// Compiles schemas into one grammar.
#[derive(Default)]
pub struct SchemaCompiler {
    builder: GrammarBuilder,
    common: Common,
    root: Value,
    references: BTreeMap<String, u32>,
    active_inline: HashSet<String>,
}

impl SchemaCompiler {
    /// Starts a compiler whose `$ref`s resolve against `root`.
    #[must_use]
    pub fn new(root: Value) -> Self {
        Self {
            root,
            ..Self::default()
        }
    }

    /// Replaces the document `$ref`s resolve against, keeping built rules.
    pub fn set_root(&mut self, root: Value) {
        self.root = root;
        self.references.clear();
    }

    /// Reserves an empty rule.
    pub fn rule(&mut self) -> u32 {
        self.builder.rule()
    }

    /// Adds one alternative to a rule.
    pub fn alternative(&mut self, id: u32, elements: Vec<GrammarElement>) {
        self.builder.alternative(id, elements);
    }

    /// Finishes the grammar.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaError`] if the schema built an invalid grammar.
    pub fn finish(self, start: u32) -> Result<Arc<Grammar>, SchemaError> {
        self.builder
            .build(start)
            .map(Arc::new)
            .map_err(|error| SchemaError(error.to_string()))
    }

    fn check_size(&self) -> Result<(), SchemaError> {
        if self.builder.rule_count() > MAX_RULES {
            return Err(unsupported("the schema is too large"));
        }
        Ok(())
    }

    fn separator(&mut self) -> u32 {
        if let Some(id) = self.common.separator {
            return id;
        }
        let id = self.builder.rule();
        self.builder.alternative(id, Vec::new());
        self.builder.alternative(id, bytes(" "));
        self.common.separator = Some(id);
        id
    }

    fn string_char(&mut self) -> u32 {
        if let Some(id) = self.common.string_char {
            return id;
        }
        let id = self.builder.rule();
        let plain = ByteClass::range(0x20, 0x21)
            .union(ByteClass::range(0x23, 0x5b))
            .union(ByteClass::range(0x5d, 0x7f));
        self.builder.alternative(id, vec![GrammarElement::Bytes(plain)]);
        let simple_escape = GrammarElement::Bytes(
            ByteClass::byte(b'"')
                .union(ByteClass::byte(b'\\'))
                .union(ByteClass::byte(b'/'))
                .union(ByteClass::byte(b'b'))
                .union(ByteClass::byte(b'f'))
                .union(ByteClass::byte(b'n'))
                .union(ByteClass::byte(b'r'))
                .union(ByteClass::byte(b't')),
        );
        self.builder
            .alternative(id, concat(vec![bytes("\\"), vec![simple_escape]]));
        let hex = GrammarElement::Bytes(
            ByteClass::range(b'0', b'9')
                .union(ByteClass::range(b'a', b'f'))
                .union(ByteClass::range(b'A', b'F')),
        );
        self.builder
            .alternative(id, concat(vec![bytes("\\u"), vec![hex; 4]]));
        let multibyte = self.multibyte_char();
        self.builder.alternative(id, vec![rule(multibyte)]);
        self.common.string_char = Some(id);
        id
    }

    fn multibyte_char(&mut self) -> u32 {
        if let Some(id) = self.common.multibyte_char {
            return id;
        }
        let id = self.builder.rule();
        // Well-formed UTF-8 only, so a split token cannot produce bad text.
        let continuation = class(0x80, 0xbf);
        self.builder
            .alternative(id, vec![class(0xc2, 0xdf), continuation]);
        self.builder
            .alternative(id, vec![class(0xe0, 0xe0), class(0xa0, 0xbf), continuation]);
        self.builder
            .alternative(id, vec![class(0xe1, 0xec), continuation, continuation]);
        self.builder
            .alternative(id, vec![class(0xed, 0xed), class(0x80, 0x9f), continuation]);
        self.builder
            .alternative(id, vec![class(0xee, 0xef), continuation, continuation]);
        self.builder.alternative(
            id,
            vec![class(0xf0, 0xf0), class(0x90, 0xbf), continuation, continuation],
        );
        self.builder.alternative(
            id,
            vec![class(0xf1, 0xf3), continuation, continuation, continuation],
        );
        self.builder.alternative(
            id,
            vec![class(0xf4, 0xf4), class(0x80, 0x8f), continuation, continuation],
        );
        self.common.multibyte_char = Some(id);
        id
    }

    fn string(&mut self) -> u32 {
        if let Some(id) = self.common.string {
            return id;
        }
        let character = self.string_char();
        let tail = self.builder.rule();
        self.builder
            .alternative(tail, vec![rule(character), rule(tail)]);
        self.builder.alternative(tail, Vec::new());
        let id = self.builder.rule();
        self.builder
            .alternative(id, concat(vec![bytes("\""), vec![rule(tail)], bytes("\"")]));
        self.common.string = Some(id);
        id
    }

    fn bounded_string(&mut self, minimum: u64, maximum: Option<u64>) -> Result<u32, SchemaError> {
        if minimum == 0 && maximum.is_none() {
            return Ok(self.string());
        }
        let character = self.string_char();
        let body = self.repetition(character, None, minimum, maximum)?;
        let id = self.builder.rule();
        self.builder
            .alternative(id, concat(vec![bytes("\""), vec![rule(body)], bytes("\"")]));
        Ok(id)
    }

    /// `item (separator item)*` with count bounds, as one rule.
    fn repetition(
        &mut self,
        item: u32,
        separator: Option<&[GrammarElement]>,
        minimum: u64,
        maximum: Option<u64>,
    ) -> Result<u32, SchemaError> {
        if maximum.is_some_and(|maximum| maximum < minimum) {
            return Err(unsupported("a maximum is below its minimum"));
        }
        if minimum > MAX_REPETITION || maximum.is_some_and(|value| value > MAX_REPETITION) {
            return Err(unsupported(format!(
                "repetition bounds above {MAX_REPETITION}"
            )));
        }
        let lead: Vec<GrammarElement> = separator.map(<[_]>::to_vec).unwrap_or_default();
        // `after` matches what may follow the (count)th item.
        let mut after = self.builder.rule();
        self.builder.alternative(after, Vec::new());
        if maximum.is_none() {
            let unbounded = after;
            self.builder.alternative(
                unbounded,
                concat(vec![lead.clone(), vec![rule(item)], vec![rule(unbounded)]]),
            );
        }
        let last = maximum.unwrap_or_else(|| minimum.max(1));
        for count in (1..last).rev() {
            let next = self.builder.rule();
            if count >= minimum {
                self.builder.alternative(next, Vec::new());
            }
            self.builder.alternative(
                next,
                concat(vec![lead.clone(), vec![rule(item)], vec![rule(after)]]),
            );
            after = next;
        }
        let id = self.builder.rule();
        if minimum == 0 {
            self.builder.alternative(id, Vec::new());
        }
        if last > 0 {
            self.builder.alternative(id, vec![rule(item), rule(after)]);
        }
        Ok(id)
    }

    fn natural(&mut self) -> u32 {
        if let Some(id) = self.common.natural {
            return id;
        }
        let digits = self.builder.rule();
        self.builder
            .alternative(digits, vec![class(b'0', b'9'), rule(digits)]);
        self.builder.alternative(digits, Vec::new());
        let id = self.builder.rule();
        self.builder.alternative(id, bytes("0"));
        self.builder
            .alternative(id, vec![class(b'1', b'9'), rule(digits)]);
        self.common.natural = Some(id);
        id
    }

    fn integer(&mut self) -> u32 {
        if let Some(id) = self.common.integer {
            return id;
        }
        let natural = self.natural();
        let id = self.builder.rule();
        self.builder.alternative(id, vec![rule(natural)]);
        self.builder
            .alternative(id, concat(vec![bytes("-"), vec![rule(natural)]]));
        self.common.integer = Some(id);
        id
    }

    fn unsigned_number(&mut self) -> u32 {
        if let Some(id) = self.common.unsigned_number {
            return id;
        }
        let natural = self.natural();
        let digits = self.builder.rule();
        self.builder
            .alternative(digits, vec![class(b'0', b'9'), rule(digits)]);
        self.builder.alternative(digits, vec![class(b'0', b'9')]);
        let fraction = self.builder.rule();
        self.builder.alternative(fraction, Vec::new());
        self.builder
            .alternative(fraction, concat(vec![bytes("."), vec![rule(digits)]]));
        let exponent = self.builder.rule();
        self.builder.alternative(exponent, Vec::new());
        let sign = GrammarElement::Bytes(ByteClass::byte(b'+').union(ByteClass::byte(b'-')));
        let marker = GrammarElement::Bytes(ByteClass::byte(b'e').union(ByteClass::byte(b'E')));
        self.builder
            .alternative(exponent, vec![marker, rule(digits)]);
        self.builder
            .alternative(exponent, vec![marker, sign, rule(digits)]);
        let id = self.builder.rule();
        self.builder
            .alternative(id, vec![rule(natural), rule(fraction), rule(exponent)]);
        self.common.unsigned_number = Some(id);
        id
    }

    fn number(&mut self) -> u32 {
        if let Some(id) = self.common.number {
            return id;
        }
        let unsigned = self.unsigned_number();
        let id = self.builder.rule();
        self.builder.alternative(id, vec![rule(unsigned)]);
        self.builder
            .alternative(id, concat(vec![bytes("-"), vec![rule(unsigned)]]));
        self.common.number = Some(id);
        id
    }

    /// `lo..=hi` for digit strings of equal length.
    fn digit_range(&mut self, id: u32, low: &[u8], high: &[u8]) {
        let (Some((&low_first, low_rest)), Some((&high_first, high_rest))) =
            (low.split_first(), high.split_first())
        else {
            self.builder.alternative(id, Vec::new());
            return;
        };
        // Without this the two recursions below double with every digit.
        if low.iter().all(|&digit| digit == b'0') && high.iter().all(|&digit| digit == b'9') {
            self.builder.alternative(id, vec![class(b'0', b'9'); low.len()]);
            return;
        }
        if low_first == high_first {
            let next = self.builder.rule();
            self.digit_range(next, low_rest, high_rest);
            self.builder
                .alternative(id, vec![class(low_first, low_first), rule(next)]);
            return;
        }
        let nines = vec![b'9'; low_rest.len()];
        let zeros = vec![b'0'; low_rest.len()];
        let first = self.builder.rule();
        self.digit_range(first, low_rest, &nines);
        self.builder
            .alternative(id, vec![class(low_first, low_first), rule(first)]);
        if low_first + 1 < high_first {
            let any = vec![class(b'0', b'9'); low_rest.len()];
            self.builder.alternative(
                id,
                concat(vec![vec![class(low_first + 1, high_first - 1)], any]),
            );
        }
        let last = self.builder.rule();
        self.digit_range(last, &zeros, high_rest);
        self.builder
            .alternative(id, vec![class(high_first, high_first), rule(last)]);
    }

    /// Non-negative integers in `low..=high`.
    fn natural_range(&mut self, id: u32, low: u64, high: u64) {
        let low_text = low.to_string().into_bytes();
        let high_text = high.to_string().into_bytes();
        for length in low_text.len()..=high_text.len() {
            let from = if length == low_text.len() {
                low_text.clone()
            } else {
                let mut floor = vec![b'0'; length];
                floor[0] = b'1';
                floor
            };
            let to = if length == high_text.len() {
                high_text.clone()
            } else {
                vec![b'9'; length]
            };
            let branch = self.builder.rule();
            self.digit_range(branch, &from, &to);
            self.builder.alternative(id, vec![rule(branch)]);
        }
    }

    fn integer_range(&mut self, minimum: Option<i64>, maximum: Option<i64>) -> Result<u32, SchemaError> {
        match (minimum, maximum) {
            (None, None) => Ok(self.integer()),
            (Some(0), None) => Ok(self.natural()),
            (low, high) => {
                let low = low.unwrap_or(-999_999_999_999_999_999);
                let high = high.unwrap_or(999_999_999_999_999_999);
                if low > high {
                    return Err(unsupported("minimum above maximum"));
                }
                let id = self.builder.rule();
                if high >= 0 {
                    self.natural_range(id, low.max(0).unsigned_abs(), high.unsigned_abs());
                }
                if low < 0 {
                    let negative = self.builder.rule();
                    let nearest = if high < 0 { high.unsigned_abs() } else { 1 };
                    self.natural_range(negative, nearest, low.unsigned_abs());
                    self.builder
                        .alternative(id, concat(vec![bytes("-"), vec![rule(negative)]]));
                }
                Ok(id)
            }
        }
    }

    /// Any JSON value.
    fn value(&mut self) -> u32 {
        if let Some(id) = self.common.value {
            return id;
        }
        let id = self.builder.rule();
        self.common.value = Some(id);
        let string = self.string();
        let number = self.number();
        let separator = self.separator();
        self.builder.alternative(id, vec![rule(string)]);
        self.builder.alternative(id, vec![rule(number)]);
        for literal in ["true", "false", "null"] {
            self.builder.alternative(id, bytes(literal));
        }
        let lead = concat(vec![bytes(","), vec![rule(separator)]]);
        let member = self.builder.rule();
        self.builder.alternative(
            member,
            concat(vec![
                vec![rule(string)],
                bytes(":"),
                vec![rule(separator), rule(id)],
            ]),
        );
        let members = self
            .repetition(member, Some(&lead), 0, None)
            .unwrap_or(member);
        self.builder
            .alternative(id, concat(vec![bytes("{"), vec![rule(members)], bytes("}")]));
        let items = self.repetition(id, Some(&lead), 0, None).unwrap_or(id);
        self.builder
            .alternative(id, concat(vec![bytes("["), vec![rule(items)], bytes("]")]));
        id
    }

    /// The grammar of `{...}` with any members, for `json_object` mode.
    pub fn any_object(&mut self) -> Result<u32, SchemaError> {
        let value = self.value();
        self.object_map(value, 0, None)
    }

    fn object_map(
        &mut self,
        value: u32,
        minimum: u64,
        maximum: Option<u64>,
    ) -> Result<u32, SchemaError> {
        let string = self.string();
        let separator = self.separator();
        let lead = concat(vec![bytes(","), vec![rule(separator)]]);
        let member = self.builder.rule();
        self.builder.alternative(
            member,
            concat(vec![
                vec![rule(string)],
                bytes(":"),
                vec![rule(separator), rule(value)],
            ]),
        );
        let members = self.repetition(member, Some(&lead), minimum, maximum)?;
        let id = self.builder.rule();
        self.builder
            .alternative(id, concat(vec![bytes("{"), vec![rule(members)], bytes("}")]));
        Ok(id)
    }

    fn literal_value(&mut self, value: &Value) -> Result<u32, SchemaError> {
        let text = serde_json::to_string(value).map_err(|error| SchemaError(error.to_string()))?;
        let id = self.builder.rule();
        self.builder.alternative(id, bytes(&text));
        Ok(id)
    }

    fn reference(&mut self, target: &str) -> Result<u32, SchemaError> {
        if let Some(&id) = self.references.get(target) {
            return Ok(id);
        }
        let pointer = target
            .strip_prefix('#')
            .ok_or_else(|| unsupported(format!("non-local $ref {target}")))?;
        let schema = self
            .root
            .pointer(pointer)
            .cloned()
            .ok_or_else(|| SchemaError(format!("unresolved $ref {target}")))?;
        let id = self.builder.rule();
        self.references.insert(target.to_owned(), id);
        let body = self.schema(&schema)?;
        self.builder.alternative(id, vec![rule(body)]);
        Ok(id)
    }

    /// Compiles one schema node into a rule.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaError`] for a keyword the grammar cannot enforce.
    pub fn schema(&mut self, schema: &Value) -> Result<u32, SchemaError> {
        self.check_size()?;
        let object = match schema {
            Value::Bool(true) => return Ok(self.value()),
            Value::Bool(false) => return Err(unsupported("the false schema")),
            Value::Object(object) => object,
            _ => return Err(SchemaError("a schema must be an object".to_owned())),
        };
        if let Some(target) = object.get("$ref") {
            let target = target
                .as_str()
                .ok_or_else(|| SchemaError("$ref must be a string".to_owned()))?;
            if let Some(key) = object
                .keys()
                .find(|key| *key != "$ref" && !ANNOTATIONS.contains(&key.as_str()))
            {
                return Err(unsupported(format!("{key} beside $ref")));
            }
            return self.reference(target);
        }
        if let Some(value) = object.get("const") {
            return self.literal_value(value);
        }
        if let Some(values) = object.get("enum") {
            let values = values
                .as_array()
                .filter(|values| !values.is_empty())
                .ok_or_else(|| SchemaError("enum must be a non-empty array".to_owned()))?;
            let id = self.builder.rule();
            for value in values {
                let branch = self.literal_value(value)?;
                self.builder.alternative(id, vec![rule(branch)]);
            }
            return Ok(id);
        }
        for keyword in ["anyOf", "oneOf"] {
            if let Some(branches) = object.get(keyword) {
                let branches = branches
                    .as_array()
                    .filter(|branches| !branches.is_empty())
                    .ok_or_else(|| SchemaError(format!("{keyword} must be a non-empty array")))?;
                let id = self.builder.rule();
                for branch in branches {
                    let branch = self.schema(branch)?;
                    self.builder.alternative(id, vec![rule(branch)]);
                }
                return Ok(id);
            }
        }
        if let Some(branches) = object.get("allOf") {
            return match branches.as_array().map(Vec::as_slice) {
                Some([only]) => self.schema(only),
                _ => Err(unsupported("allOf with more than one schema")),
            };
        }
        match object.get("type") {
            None if object.contains_key("properties") => self.typed("object", object),
            None => {
                if let Some(key) = object
                    .keys()
                    .find(|key| !ANNOTATIONS.contains(&key.as_str()))
                {
                    return Err(unsupported(format!("{key} without type")));
                }
                Ok(self.value())
            }
            Some(Value::String(name)) => self.typed(name, object),
            Some(Value::Array(names)) => {
                let id = self.builder.rule();
                for name in names {
                    let name = name
                        .as_str()
                        .ok_or_else(|| SchemaError("type entries must be strings".to_owned()))?;
                    let branch = self.typed(name, object)?;
                    self.builder.alternative(id, vec![rule(branch)]);
                }
                Ok(id)
            }
            Some(_) => Err(SchemaError("type must be a string or an array".to_owned())),
        }
    }

    fn typed(&mut self, name: &str, object: &Map<String, Value>) -> Result<u32, SchemaError> {
        const TYPE_KEYWORDS: [(&str, &[&str]); 7] = [
            ("null", &[]),
            ("boolean", &[]),
            ("string", &["minLength", "maxLength", "format", "pattern"]),
            ("integer", &["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]),
            ("number", &["minimum"]),
            ("array", &["items", "minItems", "maxItems"]),
            (
                "object",
                &["properties", "required", "additionalProperties", "minProperties", "maxProperties"],
            ),
        ];
        let allowed = TYPE_KEYWORDS
            .iter()
            .find(|(type_name, _)| *type_name == name)
            .map(|(_, keywords)| *keywords)
            .ok_or_else(|| SchemaError(format!("unknown type {name}")))?;
        // In a multi-type schema a keyword of another type does not apply.
        let multiple = object.get("type").is_some_and(Value::is_array);
        for key in object.keys() {
            let known_elsewhere = TYPE_KEYWORDS
                .iter()
                .any(|(_, keywords)| keywords.contains(&key.as_str()));
            if key != "type"
                && !ANNOTATIONS.contains(&key.as_str())
                && !allowed.contains(&key.as_str())
                && !(multiple && known_elsewhere)
            {
                return Err(unsupported(format!("{key} on type {name}")));
            }
        }
        match name {
            "null" | "boolean" => {
                let id = self.builder.rule();
                if name == "null" {
                    self.builder.alternative(id, bytes("null"));
                } else {
                    self.builder.alternative(id, bytes("true"));
                    self.builder.alternative(id, bytes("false"));
                }
                Ok(id)
            }
            "string" => self.string_schema(object),
            "integer" => {
                let bound = |key: &str| -> Result<Option<i64>, SchemaError> {
                    object.get(key).map_or(Ok(None), |value| {
                        value.as_i64().map(Some).ok_or_else(|| {
                            unsupported(format!("non-integer {key} on an integer"))
                        })
                    })
                };
                let above = |key: &str, step: i64| -> Result<Option<i64>, SchemaError> {
                    Ok(bound(key)?.map(|value| value.saturating_add(step)))
                };
                let tightest = |a: Option<i64>, b: Option<i64>, pick: fn(i64, i64) -> i64| match (a, b) {
                    (Some(a), Some(b)) => Some(pick(a, b)),
                    (a, b) => a.or(b),
                };
                self.integer_range(
                    tightest(bound("minimum")?, above("exclusiveMinimum", 1)?, i64::max),
                    tightest(bound("maximum")?, above("exclusiveMaximum", -1)?, i64::min),
                )
            }
            "number" => match object.get("minimum").map(Value::as_f64) {
                None => Ok(self.number()),
                Some(Some(0.0)) => Ok(self.unsigned_number()),
                Some(_) => Err(unsupported("number minimum other than 0")),
            },
            "array" => self.array_schema(object),
            _ => self.object_schema(object),
        }
    }

    fn string_schema(&mut self, object: &Map<String, Value>) -> Result<u32, SchemaError> {
        if let Some(pattern) = object.get("pattern") {
            if ["format", "minLength", "maxLength"]
                .iter()
                .any(|key| object.contains_key(*key))
            {
                return Err(unsupported("pattern with format or length bounds"));
            }
            let pattern = pattern
                .as_str()
                .ok_or_else(|| SchemaError("pattern must be a string".to_owned()))?;
            return self.pattern(pattern);
        }
        if let Some(format) = object.get("format") {
            if object.contains_key("minLength") || object.contains_key("maxLength") {
                return Err(unsupported("format with length bounds"));
            }
            return self.string_format(format.as_str().unwrap_or_default());
        }
        let bound = |key: &str| object.get(key).and_then(Value::as_u64);
        self.bounded_string(bound("minLength").unwrap_or(0), bound("maxLength"))
    }

    /// A JSON Schema `pattern`: unanchored ends match any text, as in the spec.
    fn pattern(&mut self, pattern: &str) -> Result<u32, SchemaError> {
        let body = pattern.strip_prefix('^');
        let anchored_start = body.is_some();
        let body = body.unwrap_or(pattern);
        let stripped = body.strip_suffix('$').filter(|rest| !rest.ends_with('\\'));
        let anchored_end = stripped.is_some();
        let characters: Vec<char> = stripped.unwrap_or(body).chars().collect();
        let mut position = 0;
        let matched = self.pattern_alternation(&characters, &mut position)?;
        if position != characters.len() {
            return Err(unsupported(format!("pattern {pattern:?}: unbalanced group")));
        }
        let free = self.string_char();
        let any = self.repetition(free, None, 0, None)?;
        let mut elements = bytes("\"");
        if !anchored_start {
            elements.push(rule(any));
        }
        elements.push(rule(matched));
        if !anchored_end {
            elements.push(rule(any));
        }
        elements.extend(bytes("\""));
        let id = self.builder.rule();
        self.builder.alternative(id, elements);
        Ok(id)
    }

    fn pattern_alternation(
        &mut self,
        characters: &[char],
        position: &mut usize,
    ) -> Result<u32, SchemaError> {
        let id = self.builder.rule();
        loop {
            let sequence = self.pattern_sequence(characters, position)?;
            self.builder.alternative(id, sequence);
            if characters.get(*position) != Some(&'|') {
                return Ok(id);
            }
            *position += 1;
        }
    }

    fn pattern_sequence(
        &mut self,
        characters: &[char],
        position: &mut usize,
    ) -> Result<Vec<GrammarElement>, SchemaError> {
        let mut sequence = Vec::new();
        while let Some(&next) = characters.get(*position) {
            if matches!(next, '|' | ')') {
                break;
            }
            let atom = self.pattern_atom(characters, position)?;
            let (minimum, maximum) = match pattern_quantifier(characters, position)? {
                Some(bounds) => bounds,
                None => {
                    sequence.extend(atom);
                    continue;
                }
            };
            let item = self.builder.rule();
            self.builder.alternative(item, atom);
            sequence.push(rule(self.repetition(item, None, minimum, maximum)?));
        }
        Ok(sequence)
    }

    fn pattern_atom(
        &mut self,
        characters: &[char],
        position: &mut usize,
    ) -> Result<Vec<GrammarElement>, SchemaError> {
        let next = characters[*position];
        *position += 1;
        match next {
            '.' => Ok(vec![rule(self.string_char())]),
            '(' => {
                if characters.get(*position) == Some(&'?') {
                    if characters.get(*position + 1) != Some(&':') {
                        return Err(unsupported("pattern lookaround"));
                    }
                    *position += 2;
                }
                let group = self.pattern_alternation(characters, position)?;
                if characters.get(*position) != Some(&')') {
                    return Err(unsupported("pattern with an unclosed group"));
                }
                *position += 1;
                Ok(vec![rule(group)])
            }
            '[' => self.pattern_class(characters, position),
            '\\' => {
                let escape = *characters
                    .get(*position)
                    .ok_or_else(|| unsupported("pattern ending in a backslash"))?;
                *position += 1;
                match escape_class(escape) {
                    Some(set) => Ok(vec![GrammarElement::Bytes(set)]),
                    None if escape.is_ascii_punctuation() => pattern_literal(escape),
                    None => Err(unsupported(format!("pattern escape \\{escape}"))),
                }
            }
            '*' | '+' | '?' | '{' | '^' | '$' => {
                Err(unsupported(format!("pattern operator {next} in this position")))
            }
            literal => pattern_literal(literal),
        }
    }

    fn pattern_class(
        &mut self,
        characters: &[char],
        position: &mut usize,
    ) -> Result<Vec<GrammarElement>, SchemaError> {
        let negated = characters.get(*position) == Some(&'^');
        if negated {
            *position += 1;
        }
        let mut set = ByteClass::default();
        let mut first = true;
        loop {
            let next = *characters
                .get(*position)
                .ok_or_else(|| unsupported("pattern with an unclosed class"))?;
            *position += 1;
            if next == ']' && !first {
                break;
            }
            first = false;
            let low = if next == '\\' {
                let escape = *characters
                    .get(*position)
                    .ok_or_else(|| unsupported("pattern ending in a backslash"))?;
                *position += 1;
                if let Some(escaped) = escape_class(escape) {
                    set = set.union(escaped);
                    continue;
                }
                escape
            } else {
                next
            };
            let low = plain_ascii(low)?;
            let range = characters.get(*position) == Some(&'-')
                && characters.get(*position + 1).is_some_and(|end| *end != ']');
            let high = if range {
                let high = plain_ascii(characters[*position + 1])?;
                *position += 2;
                high
            } else {
                low
            };
            set = set.union(ByteClass::range(low, high));
        }
        if !negated {
            return Ok(vec![GrammarElement::Bytes(set)]);
        }
        let mut complement = ByteClass::default();
        for value in 0x20..=0x7e_u8 {
            if !set.contains(value) && value != b'"' && value != b'\\' {
                complement = complement.union(ByteClass::byte(value));
            }
        }
        let id = self.builder.rule();
        let multibyte = self.multibyte_char();
        self.builder.alternative(id, vec![GrammarElement::Bytes(complement)]);
        self.builder.alternative(id, vec![rule(multibyte)]);
        Ok(vec![rule(id)])
    }

    fn string_format(&mut self, format: &str) -> Result<u32, SchemaError> {
        let digit = || class(b'0', b'9');
        let date = vec![
            digit(), digit(), digit(), digit(),
            class(b'-', b'-'), class(b'0', b'1'), digit(),
            class(b'-', b'-'), class(b'0', b'3'), digit(),
        ];
        let time = vec![
            class(b'0', b'2'), digit(), class(b':', b':'),
            class(b'0', b'5'), digit(), class(b':', b':'),
            class(b'0', b'5'), digit(),
        ];
        let id = self.builder.rule();
        let body = match format {
            "date" => date,
            "time" => concat(vec![time, bytes("Z")]),
            "date-time" => concat(vec![date, bytes("T"), time, bytes("Z")]),
            "uuid" => {
                let hex = GrammarElement::Bytes(
                    ByteClass::range(b'0', b'9').union(ByteClass::range(b'a', b'f')),
                );
                let dash = class(b'-', b'-');
                concat(vec![
                    vec![hex; 8], vec![dash], vec![hex; 4], vec![dash], vec![hex; 4],
                    vec![dash], vec![hex; 4], vec![dash], vec![hex; 12],
                ])
            }
            other => return Err(unsupported(format!("string format {other}"))),
        };
        self.builder
            .alternative(id, concat(vec![bytes("\""), body, bytes("\"")]));
        Ok(id)
    }

    fn array_schema(&mut self, object: &Map<String, Value>) -> Result<u32, SchemaError> {
        let item = match object.get("items") {
            Some(schema) => self.schema(schema)?,
            None => self.value(),
        };
        let separator = self.separator();
        let lead = concat(vec![bytes(","), vec![rule(separator)]]);
        let bound = |key: &str| object.get(key).and_then(Value::as_u64);
        let items = self.repetition(
            item,
            Some(&lead),
            bound("minItems").unwrap_or(0),
            bound("maxItems"),
        )?;
        let id = self.builder.rule();
        self.builder
            .alternative(id, concat(vec![bytes("["), vec![rule(items)], bytes("]")]));
        Ok(id)
    }

    fn object_schema(&mut self, object: &Map<String, Value>) -> Result<u32, SchemaError> {
        let bound = |key: &str| object.get(key).and_then(Value::as_u64);
        let Some(properties) = object.get("properties").and_then(Value::as_object) else {
            if object.contains_key("required") {
                return Err(unsupported("required without properties"));
            }
            let value = match object.get("additionalProperties") {
                Some(schema @ Value::Object(_)) => self.schema(schema)?,
                Some(Value::Bool(false)) => {
                    let id = self.builder.rule();
                    self.builder.alternative(id, bytes("{}"));
                    return Ok(id);
                }
                _ => self.value(),
            };
            return self.object_map(value, bound("minProperties").unwrap_or(0), bound("maxProperties"));
        };
        if object.contains_key("minProperties") || object.contains_key("maxProperties") {
            return Err(unsupported("minProperties with properties"));
        }
        if matches!(object.get("additionalProperties"), Some(Value::Object(_))) {
            return Err(unsupported("additionalProperties schema with properties"));
        }
        let required: Vec<&str> = match object.get("required") {
            None => Vec::new(),
            Some(names) => names
                .as_array()
                .map(|names| names.iter().filter_map(Value::as_str).collect())
                .ok_or_else(|| SchemaError("required must be an array".to_owned()))?,
        };
        if let Some(name) = required.iter().find(|name| !properties.contains_key(**name)) {
            return Err(unsupported(format!("required {name} without a property")));
        }
        let separator = self.separator();
        let lead = concat(vec![bytes(","), vec![rule(separator)]]);
        // Built back to front: `rest[needs_comma]` matches the properties
        // after this one, with or without a comma before the next member.
        let mut rest = [self.builder.rule(), self.builder.rule()];
        self.builder.alternative(rest[0], Vec::new());
        self.builder.alternative(rest[1], Vec::new());
        for (name, schema) in properties.iter().rev() {
            let key = serde_json::to_string(name).map_err(|error| SchemaError(error.to_string()))?;
            let value = self.schema(schema)?;
            let member = concat(vec![
                bytes(&key),
                bytes(":"),
                vec![rule(separator), rule(value)],
            ]);
            let optional = !required.contains(&name.as_str());
            let without = self.builder.rule();
            self.builder
                .alternative(without, concat(vec![member.clone(), vec![rule(rest[1])]]));
            let with = self.builder.rule();
            self.builder.alternative(
                with,
                concat(vec![lead.clone(), member, vec![rule(rest[1])]]),
            );
            if optional {
                self.builder.alternative(without, vec![rule(rest[0])]);
                self.builder.alternative(with, vec![rule(rest[1])]);
            }
            rest = [without, with];
        }
        let id = self.builder.rule();
        self.builder
            .alternative(id, concat(vec![bytes("{"), vec![rule(rest[0])], bytes("}")]));
        Ok(id)
    }
}

/// ASCII that appears as itself inside a JSON string.
fn plain_ascii(character: char) -> Result<u8, SchemaError> {
    u8::try_from(character)
        .ok()
        .filter(|byte| (0x20..0x7f).contains(byte) && !matches!(byte, b'"' | b'\\'))
        .ok_or_else(|| unsupported(format!("pattern character {character:?} in a class")))
}

fn pattern_literal(character: char) -> Result<Vec<GrammarElement>, SchemaError> {
    if matches!(character, '"' | '\\') || character.is_control() {
        return Err(unsupported(format!(
            "pattern character {character:?}, which JSON escapes"
        )));
    }
    let mut buffer = [0; 4];
    Ok(bytes(character.encode_utf8(&mut buffer)))
}

fn escape_class(escape: char) -> Option<ByteClass> {
    let digits = ByteClass::range(b'0', b'9');
    match escape {
        'd' => Some(digits),
        'w' => Some(
            digits
                .union(ByteClass::range(b'a', b'z'))
                .union(ByteClass::range(b'A', b'Z'))
                .union(ByteClass::byte(b'_')),
        ),
        's' => Some(ByteClass::byte(b' ')),
        _ => None,
    }
}

fn pattern_quantifier(
    characters: &[char],
    position: &mut usize,
) -> Result<Option<(u64, Option<u64>)>, SchemaError> {
    let bounds = match characters.get(*position) {
        Some('*') => (0, None),
        Some('+') => (1, None),
        Some('?') => (0, Some(1)),
        Some('{') => {
            let close = characters[*position..]
                .iter()
                .position(|character| *character == '}')
                .ok_or_else(|| unsupported("pattern with an unclosed {"))?;
            let body: String = characters[*position + 1..*position + close].iter().collect();
            *position += close;
            let number = |text: &str| {
                text.parse::<u64>()
                    .map_err(|_| unsupported(format!("pattern bound {{{body}}}")))
            };
            match body.split_once(',') {
                None => {
                    let count = number(&body)?;
                    (count, Some(count))
                }
                Some((low, "")) => (number(low)?, None),
                Some((low, high)) => (number(low)?, Some(number(high)?)),
            }
        }
        _ => return Ok(None),
    };
    *position += 1;
    if matches!(characters.get(*position), Some('?' | '+')) {
        return Err(unsupported("lazy or possessive pattern quantifiers"));
    }
    Ok(Some(bounds))
}

/// Compiles a whole JSON Schema document.
///
/// # Errors
///
/// Returns [`SchemaError`] for a schema the grammar cannot enforce.
pub fn compile_schema(schema: &Value) -> Result<Arc<Grammar>, SchemaError> {
    let mut compiler = SchemaCompiler::new(schema.clone());
    let start = compiler.schema(schema)?;
    compiler.finish(start)
}

/// Compiles `response_format: {"type": "json_object"}`.
///
/// # Errors
///
/// Returns [`SchemaError`] only on an internal grammar failure.
pub fn compile_json_object() -> Result<Arc<Grammar>, SchemaError> {
    let mut compiler = SchemaCompiler::new(Value::Null);
    let start = compiler.any_object()?;
    compiler.finish(start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use riley_runtime::grammar::GrammarAutomaton;
    use serde_json::json;

    fn accepts(schema: &Value, text: &str) -> bool {
        let mut automaton = GrammarAutomaton::new(compile_schema(schema).unwrap());
        let mut cursor = automaton.start();
        automaton.advance(&mut cursor, text.as_bytes()) && automaton.is_accepting(&mut cursor)
    }

    fn rejected(schema: Value) -> String {
        compile_schema(&schema).unwrap_err().0
    }

    #[test]
    fn scalars_and_enums() {
        assert!(accepts(&json!({"type": "boolean"}), "true"));
        assert!(!accepts(&json!({"type": "boolean"}), "tru"));
        assert!(accepts(&json!({"type": "null"}), "null"));
        assert!(accepts(&json!({"enum": ["a", 1, null]}), "\"a\""));
        assert!(accepts(&json!({"enum": ["a", 1, null]}), "1"));
        assert!(!accepts(&json!({"enum": ["a", 1, null]}), "2"));
        assert!(accepts(&json!({"const": {"x": [1]}}), "{\"x\":[1]}"));
        assert!(accepts(&json!({"type": ["string", "null"]}), "null"));
        assert!(accepts(&json!({"type": "number"}), "-1.5e+3"));
        assert!(!accepts(&json!({"type": "number"}), "01"));
        assert!(!accepts(&json!({"type": "number"}), "1."));
        assert!(accepts(&json!({"type": "number", "minimum": 0}), "2.5"));
        assert!(!accepts(&json!({"type": "number", "minimum": 0}), "-2.5"));
    }

    #[test]
    fn integer_ranges_are_exact() {
        for (low, high) in [(0, 12), (5, 5), (7, 1234), (-15, 30), (-99, -7), (100, 999)] {
            let schema = json!({"type": "integer", "minimum": low, "maximum": high});
            for value in -1300_i64..1300 {
                assert_eq!(
                    accepts(&schema, &value.to_string()),
                    (low..=high).contains(&value),
                    "{value} in {low}..={high}"
                );
            }
        }
        let positive = json!({"type": "integer", "minimum": 1});
        assert!(accepts(&positive, "1"));
        assert!(accepts(&positive, "90071992547"));
        assert!(!accepts(&positive, "0"));
        assert!(!accepts(&positive, "-1"));
        // A one-sided bound stays small enough to share an object.
        let object = json!({
            "type": "object",
            "properties": {
                "pages": {"type": "integer", "minimum": 1},
                "summary": {"type": "string", "maxLength": 8}
            },
            "required": ["pages", "summary"]
        });
        assert!(accepts(&object, r#"{"pages":12,"summary":"x"}"#));
        assert!(!accepts(&object, r#"{"pages":0,"summary":"x"}"#));
    }

    #[test]
    fn strings_cover_escapes_utf8_and_lengths() {
        let schema = json!({"type": "string"});
        assert!(accepts(&schema, r#""a\n\"bé한😀""#));
        assert!(accepts(&schema, r#""é""#));
        assert!(!accepts(&schema, "\"a\nb\""));
        assert!(!accepts(&schema, r#""\x""#));
        let mut automaton = GrammarAutomaton::new(compile_schema(&schema).unwrap());
        let mut cursor = automaton.start();
        assert!(!automaton.advance(&mut cursor, b"\"\xff"));
        assert!(!automaton.advance(&mut cursor, b"\"\xed\xa0"));

        let bounded = json!({"type": "string", "minLength": 2, "maxLength": 3});
        assert!(!accepts(&bounded, "\"a\""));
        assert!(accepts(&bounded, "\"ab\""));
        assert!(accepts(&bounded, "\"한글자\""));
        assert!(!accepts(&bounded, "\"abcd\""));
        assert!(accepts(&json!({"type": "string", "format": "date"}), "\"2026-11-02\""));
        assert!(!accepts(&json!({"type": "string", "format": "date"}), "\"2026-1-2\""));
        assert!(accepts(
            &json!({"type": "string", "format": "uuid"}),
            "\"123e4567-e89b-42d3-a456-426614174000\""
        ));
    }

    #[test]
    fn objects_arrays_and_references() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "string"},
                "c": {"type": "array", "items": {"$ref": "#/$defs/n"}, "minItems": 1, "maxItems": 2}
            },
            "required": ["b"],
            "additionalProperties": false,
            "$defs": {"n": {"type": "boolean"}}
        });
        assert!(accepts(&schema, r#"{"b":"x"}"#));
        assert!(accepts(&schema, r#"{"a":1,"b":"x"}"#));
        assert!(accepts(&schema, r#"{"a": 1, "b": "x", "c": [true, false]}"#));
        assert!(accepts(&schema, r#"{"b":"x","c":[true]}"#));
        assert!(!accepts(&schema, r#"{"a":1}"#));
        assert!(!accepts(&schema, r#"{"b":"x","c":[]}"#));
        assert!(!accepts(&schema, r#"{"b":"x","c":[true,true,true]}"#));
        assert!(!accepts(&schema, r#"{"b":"x","d":1}"#));
        assert!(!accepts(&schema, r#"{,"b":"x"}"#));
    }

    #[test]
    fn all_optional_properties_never_leave_a_stray_comma() {
        let schema = json!({"properties": {"a": {"type": "null"}, "b": {"type": "null"}}});
        for text in ["{}", r#"{"a":null}"#, r#"{"b":null}"#, r#"{"a":null,"b":null}"#] {
            assert!(accepts(&schema, text), "{text}");
        }
        assert!(!accepts(&schema, r#"{,"b":null}"#));
        assert!(!accepts(&schema, r#"{"a":null,}"#));
    }

    #[test]
    fn recursive_references_and_any_json() {
        let tree = json!({
            "$ref": "#/$defs/node",
            "$defs": {"node": {"type": "object", "properties": {
                "next": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}
            }, "required": ["next"]}}
        });
        assert!(accepts(&tree, r#"{"next":{"next":{"next":null}}}"#));
        assert!(!accepts(&tree, r#"{"next":{"next":{}}}"#));

        let grammar = compile_json_object().unwrap();
        let mut automaton = GrammarAutomaton::new(grammar);
        let mut cursor = automaton.start();
        assert!(automaton.advance(
            &mut cursor,
            br#"{"a": [1, {"b": null}, "x"], "c": {"d": -1.5}}"#
        ));
        assert!(automaton.is_accepting(&mut cursor));
        let mut cursor = automaton.start();
        assert!(!automaton.advance(&mut cursor, b"[1]"));

        let map = json!({"type": "object", "additionalProperties": {"type": "integer"}});
        assert!(accepts(&map, r#"{"x":1,"y":2}"#));
        assert!(!accepts(&map, r#"{"x":"1"}"#));
    }

    #[test]
    fn patterns_match_like_a_regular_expression() {
        let anchored = |pattern: &str| json!({"type": "string", "pattern": pattern});
        let phone = anchored(r"^\d{3}-\d{4}$");
        assert!(accepts(&phone, "\"555-1234\""));
        assert!(!accepts(&phone, "\"555-123\""));
        assert!(!accepts(&phone, "\"x555-1234\""));
        let word = anchored(r"^(cat|dog)s?$");
        for (text, expected) in [("cat", true), ("dogs", true), ("cow", false), ("catss", false)] {
            assert_eq!(accepts(&word, &format!("\"{text}\"")), expected, "{text}");
        }
        let class = anchored(r"^[a-c_]+[^a-c]$");
        assert!(accepts(&class, "\"ab_x\""));
        assert!(accepts(&class, "\"a한\""));
        assert!(!accepts(&class, "\"aa\""));
        let open = anchored("b+");
        assert!(accepts(&open, "\"abbc\""));
        assert!(!accepts(&open, "\"ac\""));
        assert!(accepts(&anchored(r"^a\.b$"), "\"a.b\""));
        assert!(!accepts(&anchored(r"^a\.b$"), "\"axb\""));
    }

    #[test]
    fn exclusive_integer_bounds() {
        let schema = json!({"type": "integer", "exclusiveMinimum": 2, "exclusiveMaximum": 5, "minimum": 1});
        for (value, expected) in [(2, false), (3, true), (4, true), (5, false)] {
            assert_eq!(accepts(&schema, &value.to_string()), expected, "{value}");
        }
    }

    #[test]
    fn keywords_that_cannot_be_enforced_are_named() {
        assert!(rejected(json!({"type": "string", "pattern": "(?=a)b"})).contains("lookaround"));
        assert!(rejected(json!({"type": "string", "pattern": "a+?"})).contains("lazy"));
        assert!(rejected(json!({"type": "string", "pattern": "\\bword"})).contains("escape"));
        assert!(rejected(json!({"type": "string", "pattern": "a", "maxLength": 3})).contains("pattern"));
        assert!(rejected(json!({"type": "array", "uniqueItems": true})).contains("uniqueItems"));
        assert!(rejected(json!({"type": "number", "maximum": 3})).contains("maximum"));
        assert!(rejected(json!({"type": "number", "minimum": 2})).contains("minimum"));
        assert!(rejected(json!({"not": {}})).contains("not"));
        assert!(rejected(json!({"$ref": "http://x/y"})).contains("$ref"));
        assert!(rejected(json!({"$ref": "#/nope"})).contains("unresolved"));
        assert!(rejected(json!({"type": "string", "format": "email"})).contains("email"));
        assert!(rejected(json!({"type": "wat"})).contains("wat"));
    }
}
