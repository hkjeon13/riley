//! Small bounded HTTP/1.1 framing used by the initial serving surface.
//!
//! The server intentionally handles one request per connection and always
//! responds with `Connection: close`. It supports fixed-length JSON requests;
//! transfer coding, upgrades, trailers, and request pipelining are rejected.

use std::error;
use std::fmt;
use std::io::{self, Read, Write};

const READ_CHUNK_BYTES: usize = 4_096;
const HEADER_TERMINATOR: &[u8; 4] = b"\r\n\r\n";

/// Resource bounds applied before an HTTP request is published to the API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HttpLimits {
    /// Maximum bytes through the terminating empty header line.
    pub maximum_header_bytes: usize,
    /// Maximum number of non-request-line header fields.
    pub maximum_header_fields: usize,
    /// Maximum fixed-length request body.
    pub maximum_body_bytes: usize,
    /// Maximum origin-form request-target bytes.
    pub maximum_target_bytes: usize,
}

impl Default for HttpLimits {
    fn default() -> Self {
        Self {
            maximum_header_bytes: 16 * 1_024,
            maximum_header_fields: 64,
            maximum_body_bytes: 1_048_576,
            maximum_target_bytes: 2_048,
        }
    }
}

impl HttpLimits {
    /// Rejects zero bounds and arithmetic that cannot size the bounded buffer.
    ///
    /// # Errors
    ///
    /// Returns [`HttpReadError::InvalidLimits`] before reading from the socket.
    pub fn validate(self) -> Result<Self, HttpReadError> {
        if self.maximum_header_bytes < HEADER_TERMINATOR.len() {
            return Err(HttpReadError::InvalidLimits {
                field: "maximum_header_bytes",
            });
        }
        if self.maximum_header_fields == 0 {
            return Err(HttpReadError::InvalidLimits {
                field: "maximum_header_fields",
            });
        }
        if self.maximum_body_bytes == 0 {
            return Err(HttpReadError::InvalidLimits {
                field: "maximum_body_bytes",
            });
        }
        if self.maximum_target_bytes == 0 {
            return Err(HttpReadError::InvalidLimits {
                field: "maximum_target_bytes",
            });
        }
        self.maximum_header_bytes
            .checked_add(self.maximum_body_bytes)
            .ok_or(HttpReadError::InvalidLimits {
                field: "maximum_header_bytes + maximum_body_bytes",
            })?;
        Ok(self)
    }
}

/// HTTP methods accepted by the initial API surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HttpMethod {
    /// Read-only endpoint.
    Get,
    /// Fixed-length JSON request endpoint.
    Post,
}

/// One completely framed request, independent of connection lifetime.
#[derive(Debug, Eq, PartialEq)]
pub struct HttpRequest {
    method: HttpMethod,
    target: String,
    body: Vec<u8>,
}

impl HttpRequest {
    /// Parsed method.
    #[must_use]
    pub const fn method(&self) -> HttpMethod {
        self.method
    }

    /// ASCII origin-form request target, including an optional query.
    #[must_use]
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Fixed-length request body.
    #[must_use]
    pub fn body(&self) -> &[u8] {
        &self.body
    }
}

/// Checked framing, protocol, resource, or I/O failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum HttpReadError {
    /// Server limits cannot define a bounded parser.
    InvalidLimits { field: &'static str },
    /// Socket I/O failed or timed out.
    Io(io::Error),
    /// The peer closed before the declared request was complete.
    UnexpectedEof { stage: &'static str },
    /// Header bytes exceeded the configured aggregate bound.
    HeaderTooLarge { maximum_bytes: usize },
    /// Header field count exceeded the configured bound.
    TooManyHeaders { maximum_fields: usize },
    /// Body bytes exceeded the configured bound.
    BodyTooLarge {
        maximum_bytes: usize,
        declared_bytes: usize,
    },
    /// The request line or a header violates the strict HTTP/1.1 subset.
    Malformed { reason: &'static str },
    /// A method outside GET and POST was supplied.
    UnsupportedMethod,
    /// A POST omitted its required fixed body length.
    LengthRequired,
    /// Transfer coding or another unsupported framing mechanism was requested.
    UnsupportedFraming,
    /// POST content is not JSON.
    UnsupportedMediaType,
}

impl HttpReadError {
    /// Stable public status code; the detailed variant stays server-side.
    #[must_use]
    pub const fn status_code(&self) -> u16 {
        match self {
            Self::HeaderTooLarge { .. } | Self::TooManyHeaders { .. } => 431,
            Self::BodyTooLarge { .. } => 413,
            Self::UnsupportedMethod => 405,
            Self::LengthRequired => 411,
            Self::UnsupportedFraming => 501,
            Self::UnsupportedMediaType => 415,
            Self::InvalidLimits { .. } | Self::Io(_) => 500,
            Self::UnexpectedEof { .. } | Self::Malformed { .. } => 400,
        }
    }
}

impl fmt::Display for HttpReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits { field } => write!(formatter, "invalid HTTP limit {field}"),
            Self::Io(source) => write!(formatter, "HTTP socket I/O failed: {source}"),
            Self::UnexpectedEof { stage } => {
                write!(formatter, "peer closed during HTTP {stage}")
            }
            Self::HeaderTooLarge { maximum_bytes } => {
                write!(formatter, "HTTP headers exceed {maximum_bytes} bytes")
            }
            Self::TooManyHeaders { maximum_fields } => {
                write!(
                    formatter,
                    "HTTP request exceeds {maximum_fields} header fields"
                )
            }
            Self::BodyTooLarge {
                maximum_bytes,
                declared_bytes,
            } => write!(
                formatter,
                "HTTP body declares {declared_bytes} bytes, exceeding {maximum_bytes}"
            ),
            Self::Malformed { reason } => write!(formatter, "malformed HTTP request: {reason}"),
            Self::UnsupportedMethod => formatter.write_str("unsupported HTTP method"),
            Self::LengthRequired => formatter.write_str("POST requires Content-Length"),
            Self::UnsupportedFraming => formatter.write_str("HTTP transfer framing is unsupported"),
            Self::UnsupportedMediaType => formatter.write_str("POST requires application/json"),
        }
    }
}

impl error::Error for HttpReadError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Io(source) => Some(source),
            _ => None,
        }
    }
}

impl From<io::Error> for HttpReadError {
    fn from(source: io::Error) -> Self {
        Self::Io(source)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ParsedHeaders {
    content_length: Option<usize>,
    content_type_json: bool,
}

/// Reads exactly one bounded, non-pipelined HTTP/1.1 request.
///
/// # Errors
///
/// Returns a structured error for I/O failure, incomplete input, unsupported
/// framing, malformed syntax, or any configured resource bound.
pub fn read_request(
    reader: &mut impl Read,
    limits: HttpLimits,
) -> Result<HttpRequest, HttpReadError> {
    let limits = limits.validate()?;
    let total_capacity = limits
        .maximum_header_bytes
        .checked_add(limits.maximum_body_bytes)
        .ok_or(HttpReadError::InvalidLimits {
            field: "maximum_header_bytes + maximum_body_bytes",
        })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(total_capacity)
        .map_err(|_| HttpReadError::InvalidLimits {
            field: "HTTP request buffer allocation",
        })?;
    let mut chunk = [0_u8; READ_CHUNK_BYTES];

    let header_end = loop {
        if let Some(offset) = find_bytes(&bytes, HEADER_TERMINATOR) {
            let end = offset + HEADER_TERMINATOR.len();
            if end > limits.maximum_header_bytes {
                return Err(HttpReadError::HeaderTooLarge {
                    maximum_bytes: limits.maximum_header_bytes,
                });
            }
            break end;
        }
        if bytes.len() >= limits.maximum_header_bytes {
            return Err(HttpReadError::HeaderTooLarge {
                maximum_bytes: limits.maximum_header_bytes,
            });
        }
        let remaining_header = limits.maximum_header_bytes - bytes.len();
        let read_bound = remaining_header.min(chunk.len());
        let count = reader.read(&mut chunk[..read_bound])?;
        if count == 0 {
            return Err(HttpReadError::UnexpectedEof { stage: "headers" });
        }
        bytes.extend_from_slice(&chunk[..count]);
    };

    let (method, target, headers) = parse_head(&bytes[..header_end], limits)?;
    let content_length = match (method, headers.content_length) {
        (HttpMethod::Post, None) => return Err(HttpReadError::LengthRequired),
        (HttpMethod::Post, Some(length)) => length,
        (HttpMethod::Get, Some(0) | None) => 0,
        (HttpMethod::Get, Some(_)) => {
            return Err(HttpReadError::Malformed {
                reason: "GET bodies are not accepted",
            });
        }
    };
    if method == HttpMethod::Post && !headers.content_type_json {
        return Err(HttpReadError::UnsupportedMediaType);
    }
    if content_length > limits.maximum_body_bytes {
        return Err(HttpReadError::BodyTooLarge {
            maximum_bytes: limits.maximum_body_bytes,
            declared_bytes: content_length,
        });
    }
    let request_len =
        header_end
            .checked_add(content_length)
            .ok_or(HttpReadError::BodyTooLarge {
                maximum_bytes: limits.maximum_body_bytes,
                declared_bytes: content_length,
            })?;
    while bytes.len() < request_len {
        let remaining = request_len - bytes.len();
        let read_bound = remaining.min(chunk.len());
        let count = reader.read(&mut chunk[..read_bound])?;
        if count == 0 {
            return Err(HttpReadError::UnexpectedEof { stage: "body" });
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    if bytes.len() > request_len {
        return Err(HttpReadError::Malformed {
            reason: "request pipelining is not accepted",
        });
    }

    bytes.copy_within(header_end..request_len, 0);
    bytes.truncate(content_length);
    Ok(HttpRequest {
        method,
        target,
        body: bytes,
    })
}

#[allow(clippy::too_many_lines)]
fn parse_head(
    head: &[u8],
    limits: HttpLimits,
) -> Result<(HttpMethod, String, ParsedHeaders), HttpReadError> {
    let head = std::str::from_utf8(head).map_err(|_| HttpReadError::Malformed {
        reason: "headers must be UTF-8 ASCII",
    })?;
    if head
        .bytes()
        .any(|byte| (byte < b' ' && byte != b'\r' && byte != b'\n') || byte == 0x7f)
    {
        return Err(HttpReadError::Malformed {
            reason: "headers contain a control byte",
        });
    }
    let mut lines = head[..head.len() - HEADER_TERMINATOR.len()].split("\r\n");
    let request_line = lines.next().ok_or(HttpReadError::Malformed {
        reason: "request line is missing",
    })?;
    let mut request_parts = request_line.split(' ');
    let raw_method = request_parts.next().unwrap_or_default();
    let target = request_parts.next().unwrap_or_default();
    let version = request_parts.next().unwrap_or_default();
    if request_parts.next().is_some()
        || raw_method.is_empty()
        || target.is_empty()
        || version != "HTTP/1.1"
    {
        return Err(HttpReadError::Malformed {
            reason: "request line must be METHOD target HTTP/1.1",
        });
    }
    let method = match raw_method {
        "GET" => HttpMethod::Get,
        "POST" => HttpMethod::Post,
        _ => return Err(HttpReadError::UnsupportedMethod),
    };
    if !target.starts_with('/')
        || target.len() > limits.maximum_target_bytes
        || !target.bytes().all(|byte| byte.is_ascii_graphic())
    {
        return Err(HttpReadError::Malformed {
            reason: "request target must be bounded ASCII origin-form",
        });
    }

    let mut parsed = ParsedHeaders::default();
    let mut host_seen = false;
    let mut content_type_seen = false;
    for (index, line) in lines.enumerate() {
        if index >= limits.maximum_header_fields {
            return Err(HttpReadError::TooManyHeaders {
                maximum_fields: limits.maximum_header_fields,
            });
        }
        if line.is_empty() || line.starts_with([' ', '\t']) {
            return Err(HttpReadError::Malformed {
                reason: "empty or folded header field",
            });
        }
        let Some((name, raw_value)) = line.split_once(':') else {
            return Err(HttpReadError::Malformed {
                reason: "header field is missing a colon",
            });
        };
        if name.is_empty() || name.ends_with([' ', '\t']) || !name.bytes().all(is_header_name_byte)
        {
            return Err(HttpReadError::Malformed {
                reason: "invalid header field name",
            });
        }
        let value = raw_value.trim_matches([' ', '\t']);
        if !value.bytes().all(|byte| byte == b'\t' || byte >= b' ') {
            return Err(HttpReadError::Malformed {
                reason: "invalid header field value",
            });
        }
        if name.eq_ignore_ascii_case("host") {
            if host_seen || value.is_empty() {
                return Err(HttpReadError::Malformed {
                    reason: "Host must occur exactly once and be non-empty",
                });
            }
            host_seen = true;
        } else if name.eq_ignore_ascii_case("content-length") {
            if parsed.content_length.is_some() || value.is_empty() {
                return Err(HttpReadError::Malformed {
                    reason: "Content-Length must occur at most once",
                });
            }
            parsed.content_length = Some(parse_decimal_usize(value)?);
        } else if name.eq_ignore_ascii_case("content-type") {
            if content_type_seen {
                return Err(HttpReadError::Malformed {
                    reason: "Content-Type must occur at most once",
                });
            }
            content_type_seen = true;
            parsed.content_type_json = is_json_content_type(value);
        } else if name.eq_ignore_ascii_case("transfer-encoding")
            || name.eq_ignore_ascii_case("trailer")
            || name.eq_ignore_ascii_case("upgrade")
            || name.eq_ignore_ascii_case("expect")
        {
            return Err(HttpReadError::UnsupportedFraming);
        }
    }
    if !host_seen {
        return Err(HttpReadError::Malformed {
            reason: "HTTP/1.1 Host header is required",
        });
    }
    Ok((method, target.to_owned(), parsed))
}

fn parse_decimal_usize(value: &str) -> Result<usize, HttpReadError> {
    if !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(HttpReadError::Malformed {
            reason: "Content-Length must contain decimal digits",
        });
    }
    value
        .parse::<usize>()
        .map_err(|_| HttpReadError::Malformed {
            reason: "Content-Length does not fit the platform",
        })
}

fn is_json_content_type(value: &str) -> bool {
    let mut parts = value.split(';');
    if !parts
        .next()
        .is_some_and(|media| media.trim().eq_ignore_ascii_case("application/json"))
    {
        return false;
    }
    parts.all(|parameter| {
        let parameter = parameter.trim();
        !parameter.is_empty()
            && parameter.split_once('=').is_some_and(|(name, value)| {
                name.trim().eq_ignore_ascii_case("charset")
                    && value.trim().eq_ignore_ascii_case("utf-8")
            })
    })
}

const fn is_header_name_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|candidate| candidate == needle)
}

/// Writes a complete close-delimited response with an exact Content-Length.
///
/// # Errors
///
/// Propagates the first writer failure.
pub fn write_response(
    writer: &mut impl Write,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> io::Result<()> {
    write!(
        writer,
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\n\r\n",
        reason_phrase(status),
        body.len()
    )?;
    writer.write_all(body)
}

/// Writes the streaming response head. The connection close delimits the body.
///
/// # Errors
///
/// Propagates the first writer failure.
pub fn write_sse_head(writer: &mut impl Write) -> io::Result<()> {
    writer.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\n\r\n",
    )
}

const fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        411 => "Length Required",
        413 => "Content Too Large",
        415 => "Unsupported Media Type",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Error",
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{
        HttpLimits, HttpMethod, HttpReadError, read_request, write_response, write_sse_head,
    };

    fn post(body: &[u8]) -> Vec<u8> {
        let mut request = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .into_bytes();
        request.extend_from_slice(body);
        request
    }

    #[test]
    fn fixed_json_and_get_requests_are_framed_exactly() {
        let body = br#"{"model":"fixture","prompt":"hello"}"#;
        let request =
            read_request(&mut Cursor::new(post(body)), HttpLimits::default()).expect("valid POST");
        assert_eq!(request.method(), HttpMethod::Post);
        assert_eq!(request.target(), "/v1/completions");
        assert_eq!(request.body(), body);

        let get = b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n";
        let request =
            read_request(&mut Cursor::new(get), HttpLimits::default()).expect("valid GET");
        assert_eq!(request.method(), HttpMethod::Get);
        assert_eq!(request.target(), "/healthz");
        assert!(request.body().is_empty());
    }

    #[test]
    fn framing_ambiguity_and_missing_required_fields_fail_closed() {
        let duplicate = b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n";
        assert!(matches!(
            read_request(&mut Cursor::new(duplicate), HttpLimits::default()),
            Err(HttpReadError::Malformed { .. })
        ));

        let transfer = b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n";
        assert!(matches!(
            read_request(&mut Cursor::new(transfer), HttpLimits::default()),
            Err(HttpReadError::UnsupportedFraming)
        ));

        let no_host = b"GET /healthz HTTP/1.1\r\n\r\n";
        assert!(matches!(
            read_request(&mut Cursor::new(no_host), HttpLimits::default()),
            Err(HttpReadError::Malformed { .. })
        ));

        let no_length =
            b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n\r\n";
        assert!(matches!(
            read_request(&mut Cursor::new(no_length), HttpLimits::default()),
            Err(HttpReadError::LengthRequired)
        ));
    }

    #[test]
    fn header_body_target_and_field_bounds_are_enforced() {
        let body = br#"{"x":1}"#;
        let limits = HttpLimits {
            maximum_header_bytes: 128,
            maximum_header_fields: 4,
            maximum_body_bytes: body.len() - 1,
            maximum_target_bytes: 32,
        };
        assert!(matches!(
            read_request(&mut Cursor::new(post(body)), limits),
            Err(HttpReadError::BodyTooLarge { .. } | HttpReadError::HeaderTooLarge { .. })
        ));

        let target = b"GET /target-too-long HTTP/1.1\r\nHost: x\r\n\r\n";
        let limits = HttpLimits {
            maximum_target_bytes: 4,
            ..HttpLimits::default()
        };
        assert!(matches!(
            read_request(&mut Cursor::new(target), limits),
            Err(HttpReadError::Malformed { .. })
        ));

        let fields = b"GET / HTTP/1.1\r\nHost: x\r\nA: 1\r\nB: 2\r\n\r\n";
        let limits = HttpLimits {
            maximum_header_fields: 2,
            ..HttpLimits::default()
        };
        assert!(matches!(
            read_request(&mut Cursor::new(fields), limits),
            Err(HttpReadError::TooManyHeaders { .. })
        ));
    }

    #[test]
    fn response_headers_are_close_delimited_and_non_sniffable() {
        let mut response = Vec::new();
        write_response(&mut response, 429, "application/json", b"{}").expect("in-memory response");
        let response = String::from_utf8(response).expect("ASCII response");
        assert!(response.starts_with("HTTP/1.1 429 Too Many Requests\r\n"));
        assert!(response.contains("Content-Length: 2\r\n"));
        assert!(response.contains("Connection: close\r\n"));
        assert!(response.ends_with("\r\n\r\n{}"));

        let mut sse = Vec::new();
        write_sse_head(&mut sse).expect("in-memory SSE head");
        let sse = String::from_utf8(sse).expect("ASCII SSE head");
        assert!(sse.contains("Content-Type: text/event-stream; charset=utf-8\r\n"));
        assert!(!sse.contains("Content-Length"));
    }
}
