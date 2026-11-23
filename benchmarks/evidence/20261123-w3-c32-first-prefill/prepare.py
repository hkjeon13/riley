"""Expose the scheduler's existing timestamps at the first uncached prefill."""
from pathlib import Path
import sys


def insert(path, anchor, block):
    source = path.read_text()
    if source.count(anchor) != 1:
        raise ValueError(f"unexpected source anchor: {path} {anchor!r}")
    path.write_text(source.replace(anchor, block + anchor))


root = Path(sys.argv[1])
insert(
    root / "crates/riley-scheduler/src/scheduler.rs",
    "    #[must_use]\n    pub fn request_state(",
    """    /// Returns submission and activation times before the first uncached prefill commits.
    #[must_use]
    pub fn first_prefill_ready_times(&self, request_id: RequestId) -> Option<(u64, u64)> {
        let record = &self.requests[self.record_index(request_id)?];
        (record.prefill_cursor == 0 && record.sequence.is_some())
            .then_some((record.submitted_at_ns, record.ready_since_ns))
    }

""",
)
insert(
    root / "crates/riley-server/src/engine.rs",
    "                            crate::w3_trace::prefill(\n",
    """                            if let Some((submitted, ready)) = self.scheduler.as_ref()
                                .and_then(|scheduler| scheduler.first_prefill_ready_times(item.request_id()))
                            {
                                crate::w3_trace::activation(
                                    self.requests[index].engine_id.get(),
                                    submitted,
                                    ready,
                                    now_ns,
                                );
                            }
""",
)
trace = root / "crates/riley-server/src/w3_trace.rs"
insert(trace, "    Phase(u64, &'static str, u64),", "    Activation(u64, u64, u64, u64),\n")
insert(
    trace, "#[allow(clippy::too_many_arguments)]\npub(crate) fn prefill(",
    """pub(crate) fn activation(request: u64, submitted: u64, ready: u64, planned: u64) {
    push(Record::Activation(request, submitted, ready, planned));
}

""",
)
insert(
    trace, "                Record::Phase(id, phase, ns)",
    """                Record::Activation(id, submitted, ready, planned) => {
                    eprintln!("W3-ACTIVATION {id} {submitted} {ready} {planned}");
                }
""",
)
