//! POSIX shutdown-signal ownership for the production server binary.
//!
//! SIGINT and SIGTERM are blocked before any backend or service thread is
//! created. A dedicated synchronous waiter consumes exactly one pending
//! signal; no async signal handler touches Rust state.

use std::io;
use std::mem::MaybeUninit;

/// Graceful process-shutdown signal received by [`ShutdownSignals::wait`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShutdownSignal {
    /// Interactive interrupt (`SIGINT`).
    Interrupt,
    /// Service-manager termination (`SIGTERM`).
    Terminate,
}

/// Calling-thread signal mask and synchronous wait set.
///
/// Create this value on the process main thread before spawning backend or
/// service threads so they inherit the blocked mask.
pub struct ShutdownSignals {
    wait_set: libc::sigset_t,
    previous_mask: libc::sigset_t,
}

impl ShutdownSignals {
    /// Blocks SIGINT and SIGTERM on the calling thread and captures its prior
    /// mask for restoration.
    pub fn block() -> io::Result<Self> {
        let mut wait_set = MaybeUninit::<libc::sigset_t>::uninit();
        let mut previous_mask = MaybeUninit::<libc::sigset_t>::uninit();
        // SAFETY: both pointers identify writable sigset_t storage. The set is
        // initialized before it is passed to sigaddset or pthread_sigmask.
        unsafe {
            check_errno(libc::sigemptyset(wait_set.as_mut_ptr()))?;
            let mut wait_set = wait_set.assume_init();
            check_errno(libc::sigaddset(&mut wait_set, libc::SIGINT))?;
            check_errno(libc::sigaddset(&mut wait_set, libc::SIGTERM))?;
            check_pthread(libc::pthread_sigmask(
                libc::SIG_BLOCK,
                &wait_set,
                previous_mask.as_mut_ptr(),
            ))?;
            Ok(Self {
                wait_set,
                previous_mask: previous_mask.assume_init(),
            })
        }
    }

    /// Waits synchronously until SIGINT or SIGTERM is pending for the caller.
    pub fn wait(&self) -> io::Result<ShutdownSignal> {
        let mut signal = 0;
        // SAFETY: wait_set remains initialized and immutable for this call;
        // signal points to valid writable integer storage.
        unsafe {
            check_pthread(libc::sigwait(&self.wait_set, &mut signal))?;
        }
        match signal {
            libc::SIGINT => Ok(ShutdownSignal::Interrupt),
            libc::SIGTERM => Ok(ShutdownSignal::Terminate),
            _ => Err(io::Error::other("sigwait returned an unexpected signal")),
        }
    }
}

impl Drop for ShutdownSignals {
    fn drop(&mut self) {
        // SAFETY: previous_mask was initialized by pthread_sigmask and the
        // null old-set pointer explicitly requests no output.
        let _ = unsafe {
            libc::pthread_sigmask(libc::SIG_SETMASK, &self.previous_mask, std::ptr::null_mut())
        };
    }
}

fn check_errno(status: libc::c_int) -> io::Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn check_pthread(status: libc::c_int) -> io::Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(status))
    }
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Duration;

    use super::{ShutdownSignal, ShutdownSignals};

    fn receives_thread_directed(signal: libc::c_int, expected: ShutdownSignal) {
        let signals = ShutdownSignals::block().expect("block shutdown signals");
        // pthread_kill targets this blocked test thread specifically, avoiding
        // process-global delivery to the parallel Rust test harness.
        let target = unsafe { libc::pthread_self() };
        let sender = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            // SAFETY: target names the live parent test thread until join.
            assert_eq!(unsafe { libc::pthread_kill(target, signal) }, 0);
        });
        assert_eq!(signals.wait().expect("wait for shutdown signal"), expected);
        sender.join().expect("signal sender");
    }

    #[test]
    fn synchronously_receives_sigint_and_sigterm() {
        receives_thread_directed(libc::SIGINT, ShutdownSignal::Interrupt);
        receives_thread_directed(libc::SIGTERM, ShutdownSignal::Terminate);
    }
}
