"""Measure live scheduler occupancy without tracing request timings."""
from pathlib import Path
import sys


def insert(path, anchor, block):
    source = path.read_text()
    if source.count(anchor) != 1:
        raise ValueError(f"missing diagnostic anchor: {path} {anchor!r}")
    path.write_text(source.replace(anchor, block + anchor))


base = Path(sys.argv[1]) / "crates/riley-server/src"
trace = base / "w3_trace.rs"
insert(trace, "const LIMIT: usize", """use std::sync::atomic::{AtomicBool, Ordering};

static LIVE_PEAK: AtomicBool = AtomicBool::new(false);

/// Starts live occupancy measurement after startup warmup and before HTTP serving.
pub fn begin_live_peak() {
    LIVE_PEAK.store(true, Ordering::Release);
}

pub(crate) fn live_peak_enabled() -> bool {
    LIVE_PEAK.load(Ordering::Acquire)
}

""")
insert(base / "main.rs", "    let request_limits = RequestLimits {\n        max_output_tokens,",
       "    riley_server::w3_trace::begin_live_peak();\n")
engine = base / "engine.rs"
insert(engine, "        final_c02_metrics: Option<C02CaptureMetrics>,\n        clock: Instant,",
       "        live_active_peak: usize,\n        live_decode_peak: usize,\n        live_steps_above_32: usize,\n")
insert(engine, "                final_c02_metrics: None,\n                clock: Instant::now(),",
       "                live_active_peak: 0,\n                live_decode_peak: 0,\n                live_steps_above_32: 0,\n")
insert(engine, "            self.snapshot_cancel_deltas(&plan)?;", """
            let live_active = if crate::w3_trace::live_peak_enabled() {
                self.scheduler.as_ref().map_or(0, Scheduler::active_sequence_count)
            } else {
                0
            };
""")
insert(engine, "            record_committed_batch_shape(\n                &mut self.batch_shapes,", """
            if live_active != 0 {
                self.live_active_peak = self.live_active_peak.max(live_active);
                self.live_decode_peak = self.live_decode_peak.max(plan.decode_items().len());
                self.live_steps_above_32 += usize::from(live_active > 32);
            }
""")
insert(engine, "            self.close_resources()?;\n            Ok(events)", """
            eprintln!(
                "W3-LIVE-PEAK {} {} {}",
                self.live_active_peak, self.live_decode_peak, self.live_steps_above_32,
            );
""")
