//! Threads for the parallel functions of the extension.
//!
//! Two rules, each from a hang or a stall that happened:
//!
//! * **A pool of its own for every call, dropped at the end.** rayon's global
//!   pool outlives a `fork()` without its threads, and a child process that
//!   uses it waits for them for ever. That is Python's `multiprocessing` on
//!   Linux. A pool costs about 1 ms to build at 20 threads.
//! * **Python's lock is released while the work runs,** and the thread Python
//!   called watches for Ctrl-C: signals reach Python on its main thread only.
//!   Without this a long call blocks every other Python thread and cannot be
//!   interrupted.
//!
//! `count(0)` means the same in every function: RAYON_NUM_THREADS if set,
//! else the cores this process may use (a cluster job's allocation).

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::time::Duration;

/// The number of threads for a call: what was asked for; else
/// RAYON_NUM_THREADS; else the cores this process may use. Reads the
/// environment, not rayon's global pool, which asking would create.
pub fn count(asked: usize) -> usize {
    if asked > 0 {
        return asked;
    }
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|x| x.parse::<usize>().ok())
        .filter(|&x| x > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1))
}

/// A pool of `n` threads (`count` applied) for one call.
pub fn pool(n: usize) -> Result<rayon::ThreadPool, String> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(count(n))
        .build()
        .map_err(|e| format!("thread pool: {}", e))
}

fn panic_message(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "a panic without a message".into()
    }
}

/// Run `job` on a pool of `n_threads` threads (`count` applied), with
/// Python's lock released.
///
/// The calling thread looks for a signal every 100 ms. On Ctrl-C it sets the
/// flag `job` is given, waits for `job` to return, and raises
/// KeyboardInterrupt; `job` should check the flag between units of work and
/// return early. A panic in `job` comes back as RuntimeError naming `what`.
///
/// For Rust-only work: `job` must not call into Python, since the calling
/// thread holds the lock while it checks for signals.
pub fn run_released<T, F>(py: Python<'_>, what: &str, n_threads: usize, job: F) -> PyResult<T>
where
    T: Send,
    F: FnOnce(&AtomicBool) -> T + Send,
{
    let cancel = AtomicBool::new(false);
    let mut interrupted: Option<PyErr> = None;
    let outcome = py.detach(|| -> Result<T, String> {
        let pool = pool(n_threads)?;
        let (done, wait) = channel();
        std::thread::scope(|scope| {
            let cancel = &cancel;
            scope.spawn(move || {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pool.install(|| job(cancel))));
                let _ = done.send(r.map_err(panic_message));
            });
            loop {
                match wait.recv_timeout(Duration::from_millis(100)) {
                    Ok(r) => return r,
                    Err(RecvTimeoutError::Timeout) => {
                        if interrupted.is_none() {
                            if let Err(e) = Python::attach(|py| py.check_signals()) {
                                interrupted = Some(e);
                                cancel.store(true, Relaxed);
                            }
                        }
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        return Err("the worker thread ended without a result".into())
                    }
                }
            }
        })
    });
    if let Some(e) = interrupted {
        return Err(e);
    }
    outcome.map_err(|e| PyRuntimeError::new_err(format!("{}: {}", what, e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_follows_the_request_then_the_environment() {
        assert_eq!(count(3), 3);
        assert!(count(0) >= 1);
    }

    #[test]
    fn pool_has_the_asked_size() {
        let p = pool(3).unwrap();
        assert_eq!(p.install(rayon::current_num_threads), 3);
    }
}
