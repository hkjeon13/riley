//! Host-side step timing for diagnosis (`RILEY_HOST_STEP_TRACE=1`).
//!
//! Each engine-loop segment adds the wall time it took to a process-wide
//! counter; the totals are printed to stderr every 128 steps. With the
//! variable unset every call is one predictable branch.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub const CANCEL_COMMANDS: usize = 0;
pub const PLAN: usize = 1;
pub const PREPARE: usize = 2;
pub const PACK: usize = 3;
pub const UPLOAD: usize = 4;
pub const SUBMIT: usize = 5;
pub const WAIT: usize = 6;
pub const DOWNLOAD: usize = 7;
pub const SAMPLE: usize = 8;
pub const COMMIT: usize = 9;
pub const EVENTS: usize = 10;
pub const PUBLISH: usize = 11;
const NAMES: [&str; 12] = [
    "cancel+commands",
    "plan",
    "prepare",
    "pack",
    "upload",
    "submit",
    "wait",
    "download",
    "sample",
    "commit",
    "events",
    "publish",
];
static TOTALS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
static STEPS: AtomicU64 = AtomicU64::new(0);
/// Steps that recorded a new step graph, and steps that ran eagerly because
/// no graph was recorded for their shape and the recording budget was spent.
static CAPTURED: AtomicU64 = AtomicU64::new(0);
static EAGER: AtomicU64 = AtomicU64::new(0);

pub fn note_step_kind(capturing: bool, replaying: bool) {
    if enabled() && !replaying {
        if capturing { &CAPTURED } else { &EAGER }.fetch_add(1, Ordering::Relaxed);
    }
}

#[must_use]
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RILEY_HOST_STEP_TRACE").is_some())
}

#[must_use]
pub fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}

/// Adds the time since `*mark` to `segment` and moves the mark to now.
pub fn lap(mark: &mut Option<Instant>, segment: usize) {
    if let Some(previous) = mark {
        let now = Instant::now();
        let elapsed = u64::try_from(now.duration_since(*previous).as_nanos()).unwrap_or(0);
        TOTALS[segment].fetch_add(elapsed, Ordering::Relaxed);
        *previous = now;
    }
}

/// Adds the time since the last call on this thread to `segment`.
///
/// For code that has no mark to thread through: the engine loop runs on one
/// thread, so consecutive calls partition its time.
pub fn lap_here(segment: usize) {
    use std::cell::Cell;
    thread_local! {
        static LAST: Cell<Option<Instant>> = const { Cell::new(None) };
    }
    if !enabled() {
        return;
    }
    let now = Instant::now();
    if let Some(previous) = LAST.replace(Some(now)) {
        let elapsed = u64::try_from(now.duration_since(previous).as_nanos()).unwrap_or(0);
        TOTALS[segment].fetch_add(elapsed, Ordering::Relaxed);
    }
}

pub fn step_done() {
    if !enabled() {
        return;
    }
    let steps = STEPS.fetch_add(1, Ordering::Relaxed) + 1;
    if steps % 128 == 0 {
        let mut line = String::from("HOSTSTEP us/step over 128 steps:");
        for (name, total) in NAMES.iter().zip(&TOTALS) {
            let total = total.swap(0, Ordering::Relaxed);
            line.push_str(&format!(" {name}={:.1}", total as f64 / 128_000.0));
        }
        line.push_str(&format!(
            " | captured={} eager={}",
            CAPTURED.swap(0, Ordering::Relaxed),
            EAGER.swap(0, Ordering::Relaxed)
        ));
        eprintln!("{line}");
    }
}
