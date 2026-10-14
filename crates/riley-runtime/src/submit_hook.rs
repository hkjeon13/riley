//! A per-thread callback run by the batch executor after an iteration's GPU
//! work has been submitted and before the executor waits for it.
//!
//! The engine thread uses it to do host work that does not feed the submitted
//! iteration (publishing the previous iteration's events) while the GPU is
//! busy, instead of between two iterations while it is idle. The callback is
//! a plain function: its state lives with whoever installs it.

use std::cell::Cell;

thread_local! {
    static AFTER_SUBMIT: Cell<Option<fn()>> = const { Cell::new(None) };
}

/// Installs or clears the calling thread's callback.
pub fn set(callback: Option<fn()>) {
    AFTER_SUBMIT.with(|slot| slot.set(callback));
}

/// Runs the calling thread's callback, if one is installed.
pub fn run() {
    if let Some(callback) = AFTER_SUBMIT.with(Cell::get) {
        callback();
    }
}
