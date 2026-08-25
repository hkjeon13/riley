//! Bounded scheduler counters, gauges, and nearest-rank rolling percentiles.

use crate::config::SchedulerConfig;
use crate::error::{SchedulerError, SchedulerResult};

/// Gauges describing the scheduler's current bounded ownership state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SchedulerGauges {
    /// Requests retained in the bounded waiting queue.
    pub waiting_requests: usize,
    /// Prompt tokens retained by waiting requests.
    pub waiting_prompt_tokens: usize,
    /// Sequences currently owning a KV reservation.
    pub active_sequences: usize,
    /// Worst-case KV blocks promised to active requests.
    pub promised_kv_blocks: usize,
    /// KV blocks currently allocated by the pool.
    pub allocated_kv_blocks: usize,
    /// Physical blocks in the fixed KV pool.
    pub physical_kv_blocks: usize,
    /// Terminal request identities retained for bounded idempotency lookup.
    pub retained_terminal_requests: usize,
    /// Completion payloads awaiting explicit recovery after a failed operation.
    pub pending_completions: usize,
    /// Fixed scheduler-owned completion outbox capacity.
    pub completion_capacity: usize,
    /// Whether the scheduler still accepts new submissions.
    pub accepting: bool,
    /// Immutable plans currently outstanding; the initial scheduler supports 0 or 1.
    pub outstanding_iterations: usize,
}

/// Measurements attached to one successfully completed iteration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IterationMetricSample {
    /// Number of request work items in the plan.
    pub batch_size: usize,
    /// Total prefill input tokens in the plan.
    pub prefill_tokens: usize,
    /// Total decode input tokens in the plan.
    pub decode_tokens: usize,
    /// Host scheduler time spent constructing and accounting for the plan.
    pub scheduler_cpu_ns: u64,
    /// Runtime-reported GPU execution duration.
    pub gpu_execution_ns: u64,
    /// Runtime-reported idle gap preceding this iteration.
    pub gpu_idle_gap_ns: u64,
}

impl IterationMetricSample {
    /// Checked total input-token count.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::ArithmeticOverflow`] if the two stage counts
    /// cannot be represented by `usize`.
    pub fn total_tokens(self) -> SchedulerResult<usize> {
        self.prefill_tokens.checked_add(self.decode_tokens).ok_or(
            SchedulerError::ArithmeticOverflow {
                field: "iteration metric token count",
            },
        )
    }
}

/// Summary of one fixed rolling sample window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetricWindowSnapshot {
    sample_count: usize,
    capacity: usize,
    p95: Option<u64>,
}

impl MetricWindowSnapshot {
    /// Number of retained samples, capped by [`Self::capacity`].
    #[must_use]
    pub const fn sample_count(self) -> usize {
        self.sample_count
    }

    /// Immutable maximum retained sample count.
    #[must_use]
    pub const fn capacity(self) -> usize {
        self.capacity
    }

    /// Nearest-rank p95 of retained samples, or `None` before the first sample.
    #[must_use]
    pub const fn p95(self) -> Option<u64> {
        self.p95
    }
}

/// Read-only point-in-time scheduler metrics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulerMetricsSnapshot {
    /// Sticky signal that at least one non-authoritative metrics update failed.
    /// Scheduler ownership/state remains authoritative when this is true.
    pub metrics_degraded: bool,
    /// Accepted submission attempts.
    pub requests_submitted: u64,
    /// Requests moved from waiting to active ownership.
    pub requests_admitted: u64,
    /// Requests that finished normally.
    pub requests_finished: u64,
    /// Requests that entered the failed terminal state.
    pub requests_failed: u64,
    /// Requests rejected at a bounded capacity boundary.
    pub requests_rejected: u64,
    /// Cancellation events accepted from callers or emitted during shutdown.
    /// This event counter is intentionally not mutually exclusive with other
    /// terminal counters.
    pub requests_cancelled: u64,
    /// Queued requests failed by the admission deadline.
    pub admission_timeouts: u64,
    /// Successfully completed GPU iterations.
    pub iterations_completed: u64,
    /// Iterations explicitly aborted, poisoned, or contained before successful
    /// publication.
    pub iterations_aborted: u64,
    /// Cumulative prefill input tokens committed by completed iterations.
    pub prefill_tokens: u64,
    /// Cumulative decode input tokens committed by completed iterations.
    pub decode_tokens: u64,
    /// Latest ownership and queue gauges.
    pub gauges: SchedulerGauges,
    /// Rolling queue wait for successfully admitted requests in nanoseconds.
    /// Timed-out and cancelled waits are represented by their counters rather
    /// than this latency distribution.
    pub queue_wait_ns: MetricWindowSnapshot,
    /// Rolling scheduler CPU time in nanoseconds.
    pub scheduler_cpu_ns: MetricWindowSnapshot,
    /// Rolling runtime-reported GPU execution time in nanoseconds.
    pub gpu_execution_ns: MetricWindowSnapshot,
    /// Rolling runtime-reported GPU idle gap in nanoseconds.
    pub gpu_idle_gap_ns: MetricWindowSnapshot,
    /// Rolling iteration batch size.
    pub iteration_batch_size: MetricWindowSnapshot,
    /// Rolling iteration input-token count.
    pub batched_tokens: MetricWindowSnapshot,
}

/// Concise compatibility name for the public point-in-time snapshot.
pub type MetricsSnapshot = SchedulerMetricsSnapshot;

impl SchedulerMetricsSnapshot {
    /// KV allocation utilization in basis points, when capacity is non-zero.
    #[must_use]
    pub fn kv_utilization_basis_points(self) -> Option<u16> {
        if self.gauges.physical_kv_blocks == 0 {
            return None;
        }
        let allocated = u128::try_from(self.gauges.allocated_kv_blocks).ok()?;
        let capacity = u128::try_from(self.gauges.physical_kv_blocks).ok()?;
        let basis_points = allocated.saturating_mul(10_000) / capacity;
        u16::try_from(basis_points).ok()
    }

    /// Cumulative prefill share of committed input tokens, in basis points.
    /// This ratio is token-weighted rather than request- or iteration-weighted.
    #[must_use]
    pub fn prefill_token_ratio_basis_points(self) -> Option<u16> {
        let total = self.prefill_tokens.checked_add(self.decode_tokens)?;
        if total == 0 {
            return None;
        }
        let basis_points =
            u128::from(self.prefill_tokens).saturating_mul(10_000) / u128::from(total);
        u16::try_from(basis_points).ok()
    }
}

/// Mutable, fixed-capacity metric collector owned by a scheduler.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchedulerMetrics {
    requests_submitted: u64,
    requests_admitted: u64,
    requests_finished: u64,
    requests_failed: u64,
    requests_rejected: u64,
    requests_cancelled: u64,
    admission_timeouts: u64,
    iterations_completed: u64,
    iterations_aborted: u64,
    prefill_tokens: u64,
    decode_tokens: u64,
    gauges: SchedulerGauges,
    queue_wait_ns: RollingWindow,
    scheduler_cpu_ns: RollingWindow,
    gpu_execution_ns: RollingWindow,
    gpu_idle_gap_ns: RollingWindow,
    iteration_batch_size: RollingWindow,
    batched_tokens: RollingWindow,
}

impl SchedulerMetrics {
    /// Allocates all rolling windows to their immutable capacity up front.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error for zero capacity or a bounded
    /// host-allocation error before returning a collector.
    pub fn new(window_samples: usize) -> SchedulerResult<Self> {
        Ok(Self {
            requests_submitted: 0,
            requests_admitted: 0,
            requests_finished: 0,
            requests_failed: 0,
            requests_rejected: 0,
            requests_cancelled: 0,
            admission_timeouts: 0,
            iterations_completed: 0,
            iterations_aborted: 0,
            prefill_tokens: 0,
            decode_tokens: 0,
            gauges: SchedulerGauges::default(),
            queue_wait_ns: RollingWindow::new(window_samples, "queue wait window")?,
            scheduler_cpu_ns: RollingWindow::new(window_samples, "scheduler CPU window")?,
            gpu_execution_ns: RollingWindow::new(window_samples, "GPU execution window")?,
            gpu_idle_gap_ns: RollingWindow::new(window_samples, "GPU idle-gap window")?,
            iteration_batch_size: RollingWindow::new(window_samples, "batch-size window")?,
            batched_tokens: RollingWindow::new(window_samples, "batched-token window")?,
        })
    }

    /// Allocates a collector using the validated configuration's metric bound.
    ///
    /// # Errors
    ///
    /// Propagates bounded host-allocation failure from [`Self::new`].
    pub fn from_config(config: &SchedulerConfig) -> SchedulerResult<Self> {
        Self::new(config.metrics_window_samples)
    }

    /// Records a retained request submission.
    ///
    /// # Errors
    ///
    /// Returns a metric-overflow error without mutating the counter.
    pub fn record_submission(&mut self) -> SchedulerResult<()> {
        checked_increment(&mut self.requests_submitted, "requests_submitted")
    }

    /// Records active admission and its queue wait.
    ///
    /// # Errors
    ///
    /// Returns a metric-overflow error without recording either value.
    pub fn record_admission(&mut self, queue_wait_ns: u64) -> SchedulerResult<()> {
        let next = checked_next(self.requests_admitted, "requests_admitted")?;
        self.requests_admitted = next;
        self.queue_wait_ns.record(queue_wait_ns);
        Ok(())
    }

    /// Records a normal terminal completion.
    ///
    /// # Errors
    ///
    /// Returns a metric-overflow error without mutating the counter.
    pub fn record_finished(&mut self) -> SchedulerResult<()> {
        checked_increment(&mut self.requests_finished, "requests_finished")
    }

    /// Records a failed terminal request.
    ///
    /// # Errors
    ///
    /// Returns a metric-overflow error without mutating the counter.
    pub fn record_failed(&mut self) -> SchedulerResult<()> {
        checked_increment(&mut self.requests_failed, "requests_failed")
    }

    /// Records a capacity rejection.
    ///
    /// # Errors
    ///
    /// Returns a metric-overflow error without mutating the counter.
    pub fn record_rejection(&mut self) -> SchedulerResult<()> {
        checked_increment(&mut self.requests_rejected, "requests_rejected")
    }

    /// Records caller cancellation.
    ///
    /// # Errors
    ///
    /// Returns a metric-overflow error without mutating the counter.
    pub fn record_cancellation(&mut self) -> SchedulerResult<()> {
        checked_increment(&mut self.requests_cancelled, "requests_cancelled")
    }

    /// Records an admission deadline failure.
    ///
    /// # Errors
    ///
    /// Returns a metric-overflow error without mutating the counter.
    pub fn record_admission_timeout(&mut self) -> SchedulerResult<()> {
        checked_increment(&mut self.admission_timeouts, "admission_timeouts")
    }

    /// Records rollback of a planned iteration.
    ///
    /// # Errors
    ///
    /// Returns a metric-overflow error without mutating the counter.
    pub fn record_aborted_iteration(&mut self) -> SchedulerResult<()> {
        checked_increment(&mut self.iterations_aborted, "iterations_aborted")
    }

    /// Atomically validates and records all counters/windows for one iteration.
    ///
    /// # Errors
    ///
    /// Returns for an empty batch, arithmetic conversion/overflow, or a counter
    /// overflow. No counter or rolling window changes on error.
    pub fn record_iteration(&mut self, sample: IterationMetricSample) -> SchedulerResult<()> {
        if sample.batch_size == 0 {
            return Err(SchedulerError::InvalidMetricSample {
                field: "batch_size",
                reason: "completed iteration batch size must be greater than zero",
            });
        }
        let total_tokens = sample.total_tokens()?;
        if total_tokens == 0 {
            return Err(SchedulerError::InvalidMetricSample {
                field: "tokens",
                reason: "completed iteration must contain at least one token",
            });
        }
        let prefill = u64::try_from(sample.prefill_tokens).map_err(|_| {
            SchedulerError::ArithmeticOverflow {
                field: "prefill token metric conversion",
            }
        })?;
        let decode = u64::try_from(sample.decode_tokens).map_err(|_| {
            SchedulerError::ArithmeticOverflow {
                field: "decode token metric conversion",
            }
        })?;
        let batch_size =
            u64::try_from(sample.batch_size).map_err(|_| SchedulerError::ArithmeticOverflow {
                field: "batch-size metric conversion",
            })?;
        let total_tokens_u64 =
            u64::try_from(total_tokens).map_err(|_| SchedulerError::ArithmeticOverflow {
                field: "batched-token metric conversion",
            })?;

        let next_iterations = checked_next(self.iterations_completed, "iterations_completed")?;
        let next_prefill =
            self.prefill_tokens
                .checked_add(prefill)
                .ok_or(SchedulerError::MetricOverflow {
                    metric: "prefill_tokens",
                })?;
        let next_decode =
            self.decode_tokens
                .checked_add(decode)
                .ok_or(SchedulerError::MetricOverflow {
                    metric: "decode_tokens",
                })?;

        self.iterations_completed = next_iterations;
        self.prefill_tokens = next_prefill;
        self.decode_tokens = next_decode;
        self.scheduler_cpu_ns.record(sample.scheduler_cpu_ns);
        self.gpu_execution_ns.record(sample.gpu_execution_ns);
        self.gpu_idle_gap_ns.record(sample.gpu_idle_gap_ns);
        self.iteration_batch_size.record(batch_size);
        self.batched_tokens.record(total_tokens_u64);
        Ok(())
    }

    /// Replaces all ownership gauges after validating their relationships.
    ///
    /// # Errors
    ///
    /// Returns an invalid-sample error without changing gauges when allocated or
    /// promised blocks exceed physical capacity, or more than one plan is live.
    pub fn set_gauges(&mut self, gauges: SchedulerGauges) -> SchedulerResult<()> {
        if gauges.allocated_kv_blocks > gauges.physical_kv_blocks {
            return Err(SchedulerError::InvalidMetricSample {
                field: "allocated_kv_blocks",
                reason: "must not exceed physical_kv_blocks",
            });
        }
        if gauges.promised_kv_blocks > gauges.physical_kv_blocks {
            return Err(SchedulerError::InvalidMetricSample {
                field: "promised_kv_blocks",
                reason: "must not exceed physical_kv_blocks",
            });
        }
        if gauges.allocated_kv_blocks > gauges.promised_kv_blocks {
            return Err(SchedulerError::InvalidMetricSample {
                field: "allocated_kv_blocks",
                reason: "must not exceed promised_kv_blocks",
            });
        }
        if gauges.outstanding_iterations > 1 {
            return Err(SchedulerError::InvalidMetricSample {
                field: "outstanding_iterations",
                reason: "the V1 scheduler permits at most one immutable plan",
            });
        }
        if gauges.pending_completions > gauges.completion_capacity {
            return Err(SchedulerError::InvalidMetricSample {
                field: "pending_completions",
                reason: "must not exceed completion_capacity",
            });
        }
        self.gauges = gauges;
        Ok(())
    }

    /// Returns all counters, gauges, and nearest-rank p95 values.
    ///
    /// # Errors
    ///
    /// Returns bounded host-allocation failure if a temporary percentile scratch
    /// vector cannot be allocated. The collector remains unchanged.
    pub fn snapshot(&self) -> SchedulerResult<SchedulerMetricsSnapshot> {
        Ok(SchedulerMetricsSnapshot {
            metrics_degraded: false,
            requests_submitted: self.requests_submitted,
            requests_admitted: self.requests_admitted,
            requests_finished: self.requests_finished,
            requests_failed: self.requests_failed,
            requests_rejected: self.requests_rejected,
            requests_cancelled: self.requests_cancelled,
            admission_timeouts: self.admission_timeouts,
            iterations_completed: self.iterations_completed,
            iterations_aborted: self.iterations_aborted,
            prefill_tokens: self.prefill_tokens,
            decode_tokens: self.decode_tokens,
            gauges: self.gauges,
            queue_wait_ns: self.queue_wait_ns.snapshot()?,
            scheduler_cpu_ns: self.scheduler_cpu_ns.snapshot()?,
            gpu_execution_ns: self.gpu_execution_ns.snapshot()?,
            gpu_idle_gap_ns: self.gpu_idle_gap_ns.snapshot()?,
            iteration_batch_size: self.iteration_batch_size.snapshot()?,
            batched_tokens: self.batched_tokens.snapshot()?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RollingWindow {
    samples: Vec<u64>,
    capacity: usize,
    next: usize,
}

impl RollingWindow {
    fn new(capacity: usize, resource: &'static str) -> SchedulerResult<Self> {
        if capacity == 0 {
            return Err(SchedulerError::InvalidConfiguration {
                field: "metrics_window_samples",
                reason: "must be greater than zero",
            });
        }
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(capacity)
            .map_err(|_| SchedulerError::HostAllocation {
                resource,
                requested_elements: capacity,
            })?;
        Ok(Self {
            samples,
            capacity,
            next: 0,
        })
    }

    fn record(&mut self, value: u64) {
        if self.samples.len() < self.capacity {
            self.samples.push(value);
            return;
        }
        self.samples[self.next] = value;
        self.next = (self.next + 1) % self.capacity;
    }

    fn snapshot(&self) -> SchedulerResult<MetricWindowSnapshot> {
        let mut sorted = Vec::new();
        sorted.try_reserve_exact(self.samples.len()).map_err(|_| {
            SchedulerError::HostAllocation {
                resource: "metric percentile scratch",
                requested_elements: self.samples.len(),
            }
        })?;
        sorted.extend_from_slice(&self.samples);
        sorted.sort_unstable();
        let p95 = if sorted.is_empty() {
            None
        } else {
            let rank = sorted
                .len()
                .checked_mul(95)
                .ok_or(SchedulerError::ArithmeticOverflow {
                    field: "metric p95 nearest rank",
                })?;
            let index = rank.div_ceil(100).saturating_sub(1);
            sorted.get(index).copied()
        };
        Ok(MetricWindowSnapshot {
            sample_count: self.samples.len(),
            capacity: self.capacity,
            p95,
        })
    }
}

fn checked_next(value: u64, metric: &'static str) -> SchedulerResult<u64> {
    value
        .checked_add(1)
        .ok_or(SchedulerError::MetricOverflow { metric })
}

fn checked_increment(value: &mut u64, metric: &'static str) -> SchedulerResult<()> {
    let next = checked_next(*value, metric)?;
    *value = next;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{IterationMetricSample, SchedulerMetrics};

    #[test]
    fn rolling_p95_uses_nearest_rank_and_stays_bounded() {
        let mut metrics = SchedulerMetrics::new(4).unwrap();
        for wait in [10, 20, 30, 40, 100] {
            metrics.record_admission(wait).unwrap();
        }
        let snapshot = metrics.snapshot().unwrap();
        assert_eq!(snapshot.queue_wait_ns.sample_count(), 4);
        assert_eq!(snapshot.queue_wait_ns.p95(), Some(100));
    }

    #[test]
    fn iteration_metrics_are_recorded_together() {
        let mut metrics = SchedulerMetrics::new(2).unwrap();
        metrics
            .record_iteration(IterationMetricSample {
                batch_size: 3,
                prefill_tokens: 8,
                decode_tokens: 2,
                scheduler_cpu_ns: 11,
                gpu_execution_ns: 12,
                gpu_idle_gap_ns: 13,
            })
            .unwrap();
        let snapshot = metrics.snapshot().unwrap();
        assert_eq!(snapshot.prefill_tokens, 8);
        assert_eq!(snapshot.decode_tokens, 2);
        assert_eq!(snapshot.batched_tokens.p95(), Some(10));
    }
}
