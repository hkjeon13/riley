"""DIAGNOSTIC ONLY (not committed): per-phase host timers inside the CUDA backend step, printed every 4000 GPU steps."""
import pathlib
p = pathlib.Path('/data/riley-serving-261007/qwen-algo-diagsrc/crates/riley-server/src/engine.rs'); s = p.read_text()
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70]); s = s.replace(old, new)
rep('''            let now_ns = self.now_ns();
            let planning = self
                .scheduler_mut()?
                .plan_iteration(now_ns)''', '''            let phase_t0 = std::time::Instant::now();
            let now_ns = self.now_ns();
            let planning = self
                .scheduler_mut()?
                .plan_iteration(now_ns)''')
rep('''            self.snapshot_cancel_deltas(&plan)?;
            let selection = self.sampling_selection_for_plan(&plan)?;''', '''            let phase_t1 = std::time::Instant::now();
            self.snapshot_cancel_deltas(&plan)?;
            let selection = self.sampling_selection_for_plan(&plan)?;''')
rep('''            let expected_active_rows = plan.total_tokens();
            let (mut downloaded, timing, staged_shape) =''', '''            let expected_active_rows = plan.total_tokens();
            let phase_t2 = std::time::Instant::now();
            let (mut downloaded, timing, staged_shape) =''')
rep('''            let sampling = self.sample_iteration(&plan, &downloaded, selection);''', '''            let phase_t3 = std::time::Instant::now();
            let sampling = self.sample_iteration(&plan, &downloaded, selection);
            let phase_t4 = std::time::Instant::now();''')
rep('''            record_committed_batch_shape(
                &mut self.batch_shapes,
                staged_shape,
                updates.iteration_metric().is_some(),
            );''', '''            let phase_t5 = std::time::Instant::now();
            record_committed_batch_shape(
                &mut self.batch_shapes,
                staged_shape,
                updates.iteration_metric().is_some(),
            );''')
rep('''            if !updates.settlement_failures().is_empty() {
                eprintln!(
                    "riley scheduler contained {} settlement failure(s)",
                    updates.settlement_failures().len()
                );
            }''', '''            if !updates.settlement_failures().is_empty() {
                eprintln!(
                    "riley scheduler contained {} settlement failure(s)",
                    updates.settlement_failures().len()
                );
            }
            {
                use std::sync::atomic::{AtomicU64, Ordering as O};
                static ACC: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
                static LAST_END_NS: AtomicU64 = AtomicU64::new(0);
                static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let epoch = *EPOCH.get_or_init(std::time::Instant::now);
                let phase_t6 = std::time::Instant::now();
                let ns = |a: std::time::Instant, b: std::time::Instant| u64::try_from((b - a).as_nanos()).unwrap_or(0);
                let start_ns = ns(epoch, phase_t0);
                let previous_end = LAST_END_NS.swap(ns(epoch, phase_t6), O::Relaxed);
                let between = if previous_end != 0 && start_ns > previous_end && start_ns - previous_end < 5_000_000 { start_ns - previous_end } else { 0 };
                for (index, value) in [ns(phase_t0, phase_t1), ns(phase_t1, phase_t2), ns(phase_t2, phase_t3), ns(phase_t3, phase_t4), ns(phase_t4, phase_t5), ns(phase_t5, phase_t6), between].into_iter().enumerate() {
                    ACC[index].fetch_add(value, O::Relaxed);
                }
                let steps = ACC[7].fetch_add(1, O::Relaxed) + 1;
                if steps % 4000 == 0 {
                    let mean = |index: usize| ACC[index].swap(0, O::Relaxed) as f64 / 4000.0 / 1000.0;
                    eprintln!(
                        "RILEY_PHASE_US plan={:.1} pre={:.1} execute={:.1} sample={:.1} commit={:.1} events={:.1} between_steps={:.1}",
                        mean(0), mean(1), mean(2), mean(3), mean(4), mean(5), mean(6)
                    );
                }
            }''')
p.write_text(s)
print("patched")
