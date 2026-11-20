"""Add a prompt-free KV admission trace to the held diagnostic build."""
from pathlib import Path
import sys


def edit(path, changes):
    text = path.read_text()
    for old, new in changes:
        if text.count(old) != 1:
            raise ValueError(f"{path.name}: expected one match for {old!r}")
        text = text.replace(old, new, 1)
    path.write_text(text)


root = Path(sys.argv[1])
edit(root / "src/crates/riley-scheduler/src/scheduler.rs", [
    ("    fn admit_waiting(&mut self, now_ns: u64) -> SchedulerResult<()> {",
     """    fn trace_w3_kv(&self, request: RequestId, event: &str, now_ns: u64, needed: usize) {
        static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *TRACE.get_or_init(|| std::env::var_os("RILEY_W3_KV_TRACE").is_some()) {
            eprintln!(
                "W3-KV {} {event} {now_ns} {needed} {} {} {} {} {}",
                request.get(),
                self.promised_kv_blocks,
                self.config.max_promised_kv_blocks,
                self.active_sequences,
                self.config.max_active_sequences,
                self.waiting.len(),
            );
        }
    }

    fn admit_waiting(&mut self, now_ns: u64) -> SchedulerResult<()> {"""),
    ("""            if !self.can_admit(promised) {
                return Ok(());
            }""",
     """            if !self.can_admit(promised) {
                self.trace_w3_kv(request_id, "blocked", now_ns, promised);
                return Ok(());
            }"""),
    ("""            record.ready_since_ns = now_ns;
            observe_metric(
                &mut self.metrics_degraded,
                self.metrics.record_admission(waited_ns),""",
     """            record.ready_since_ns = now_ns;
            self.trace_w3_kv(request_id, "admitted", now_ns, promised);
            observe_metric(
                &mut self.metrics_degraded,
                self.metrics.record_admission(waited_ns),"""),
    ("""        Ok(Submission { request_id, state })""",
     """        self.trace_w3_kv(
            request_id,
            if can_admit { "submit_active" } else { "submit_queued" },
            now_ns,
            promised_kv_blocks,
        );
        Ok(Submission { request_id, state })"""),
    ("""        let mut record = self.requests.swap_remove(index);
        self.completion_outbox.push_back(RequestCompletion {
            request_id,
            reason,""",
     """        let mut record = self.requests.swap_remove(index);
        if previous_state != RequestState::Waiting {
            self.trace_w3_kv(request_id, "released", now_ns, record.promised_kv_blocks);
        }
        self.completion_outbox.push_back(RequestCompletion {
            request_id,
            reason,"""),
])
edit(root / "src/crates/riley-server/src/engine.rs", [
    ("""                .map_err(|source| scheduler_admission_error(&source))?;
            let structured = match request.grammar.as_ref() {""",
     """                .map_err(|source| scheduler_admission_error(&source))?;
            super::w3_trace(
                format_args!("{}", metadata.request_id),
                "kv_scheduler_id",
                format_args!("{}", submission.request_id().get()),
            );
            let structured = match request.grammar.as_ref() {"""),
])
edit(root / "tools/run_profile_retry.py", [
    ('MODES = ("riley-first", "qkv-first", "qkv-second", "riley-second")',
     'MODES = ("riley-first", "riley-second")'),
    ('    total_kv = expected_blocks * block_bytes',
     '''    if mode == "riley-second":
        expected_blocks = 768
        blocks = ["--kv-blocks", str(expected_blocks)]
    total_kv = expected_blocks * block_bytes'''),
    ('    overrides = {"RILEY_W3_REQUEST_TRACE": "1"}',
     '    overrides = {"RILEY_W3_REQUEST_TRACE": "1", "RILEY_W3_KV_TRACE": "1"}'),
])
edit(root / "tools/profile_workload.py", [
    ("\ndef run(plan_path):",
     '''
def idle_snapshot(address):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        with urllib.request.urlopen(f"http://{address}/metrics", timeout=3) as response:
            data = json.load(response)
        if all(type(data[field]) is int and data[field] == 0
               for field in ("active_requests", "waiting_requests", "kv_allocated_blocks")):
            return data
        time.sleep(0.1)
    raise TimeoutError("server did not drain request state and physical KV blocks")


def run(plan_path):'''),
    ('            for name, cell in plan["cells"].items():',
     '''            save(out / "idle-before.json", idle_snapshot(plan["address"]))
            for name, cell in plan["cells"].items():'''),
    ('        finally:\n            if server.poll() is None:',
     '''            save(out / "drain.json", idle_snapshot(plan["address"]))
        finally:
            if server.poll() is None:'''),
])
