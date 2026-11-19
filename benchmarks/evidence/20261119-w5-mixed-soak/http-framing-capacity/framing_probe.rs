use std::env;
use std::fs;
use std::hint::black_box;
use std::io::{Cursor, Read};
use std::time::Instant;

mod http;

struct Fragmented<'a> {
    rest: &'a [u8],
    maximum_chunk: usize,
}

impl Read for Fragmented<'_> {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        let count = output.len().min(self.maximum_chunk);
        self.rest.read(&mut output[..count])
    }
}

fn packet(header: &[u8], payload: &[u8]) -> Vec<u8> {
    [header, payload].concat()
}

fn main() {
    use http::{HttpLimits, HttpMethod, HttpReadError, read_request};

    let fixture = fs::read(env::args().nth(1).expect("framing inputs")).expect("read inputs");
    let mut encoded = Cursor::new(fixture.as_slice());
    let mut cases = Vec::new();
    while encoded.position() < fixture.len() as u64 {
        let mut prefix = [0; 4];
        encoded
            .read_exact(&mut prefix)
            .expect("fixture body length");
        let mut body = vec![0; u32::from_le_bytes(prefix) as usize];
        encoded.read_exact(&mut body).expect("fixture request body");
        let head = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len(),
        );
        cases.push((packet(head.as_bytes(), &body), head.len()));
    }

    let limits = HttpLimits::default();
    let maximum = vec![b'x'; limits.maximum_body_bytes];
    let header = format!(
        "POST /body HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        maximum.len(),
    );
    let full = packet(header.as_bytes(), &maximum);
    let mut input = Fragmented {
        rest: &full,
        maximum_chunk: 127,
    };
    assert_eq!(
        read_request(&mut input, limits)
            .expect("maximum fragmented body")
            .body(),
        maximum
    );
    let input = &full[..full.len() - 1];
    assert!(matches!(
        read_request(&mut Cursor::new(input), limits),
        Err(HttpReadError::UnexpectedEof { stage: "body" })
    ));
    let oversized = format!(
        "POST /body HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        maximum.len() + 1,
    );
    assert!(matches!(
        read_request(&mut Cursor::new(oversized.as_bytes()), limits),
        Err(HttpReadError::BodyTooLarge { .. })
    ));
    let pipeline = b"GET /readyz HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\nGET /metrics HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
    assert!(matches!(
        read_request(&mut Cursor::new(pipeline), limits),
        Err(HttpReadError::Malformed {
            reason: "request pipelining is not accepted"
        })
    ));
    let mut capacities = Vec::with_capacity(cases.len());
    let mut times = Vec::with_capacity(cases.len());
    for (index, (wire, head_len)) in cases.iter().enumerate() {
        let mut reader = Fragmented {
            rest: wire,
            maximum_chunk: [131, 4096][index % 2],
        };
        let start = Instant::now();
        let request = read_request(black_box(&mut reader), limits).expect("replay request");
        times.push(start.elapsed().as_nanos());
        assert_eq!(request.method(), HttpMethod::Post);
        assert_eq!(request.target(), "/v1/chat/completions");
        assert_eq!(request.authorization(), None);
        assert_eq!(request.body(), &wire[*head_len..]);
        capacities.push(request.probe_body_capacity());
    }
    capacities.sort_unstable();
    times.sort_unstable();
    let count = cases.len();
    println!(
        "FRAMING requests={count} body_capacity_p50={} body_capacity_p99={} body_capacity_max={} time_p50_ns={} time_p95_ns={}",
        capacities[count / 2],
        capacities[count * 99 / 100],
        capacities[count - 1],
        times[count / 2],
        times[count * 95 / 100],
    );
}
