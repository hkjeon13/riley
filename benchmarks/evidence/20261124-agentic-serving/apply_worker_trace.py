"""Add connection wait diagnostics to the pre-existing per-request GPU trace build."""
from pathlib import Path

path = Path("crates/riley-server/src/service.rs")
source = path.read_text()
for anchor, replacement in [
    ("#[derive(Debug)]\nstruct ConnectionJob", "thread_local! { static HTTP_QUEUE_WAIT_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }\n\n#[derive(Debug)]\nstruct ConnectionJob"),
    ("metrics.record_worker_wait(wait_ns);", 'metrics.record_worker_wait(wait_ns); HTTP_QUEUE_WAIT_NS.set(wait_ns); eprintln!("HTTP_DEQUEUED {wait_ns}");'),
    ('eprintln!("AGENT_RENDER {} {begin} {end}", submitted.metadata().request_id);', 'eprintln!("AGENT_RENDER {} {begin} {end}", submitted.metadata().request_id); eprintln!("HTTP_REQUEST_WAIT {} {}", submitted.metadata().request_id, HTTP_QUEUE_WAIT_NS.get());'),
]:
    assert source.count(anchor) == 1, anchor
    source = source.replace(anchor, replacement)
path.write_text(source)
