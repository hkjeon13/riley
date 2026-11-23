"""Apply the prefill timeline to the frozen benchmark source only."""

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent


def edit(path, replacements):
    source = path.read_text()
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError(f"unexpected diagnostic source in {path}: {before!r}")
        source = source.replace(before, after)
    path.write_text(source)


def instrument(root):
    edit(root / "lib.rs", [
        ('pub mod service;\n', 'pub mod service;\n#[cfg(all(feature = "server", feature = "cuda"))]\nmod rag_steps;\n'),
    ])
    edit(root / "engine.rs", [
        ('            self.requests.push(CudaRequest {',
         '''            crate::rag_steps::admit(request_id.get(), submission.request_id().get());
            self.requests.push(CudaRequest {'''),
        ('            let (mut downloaded, timing, staged_shape) =',
         '''            let trace_start = crate::rag_steps::clock();
            let (mut downloaded, timing, staged_shape) ='''),
        ('            let sampling = self.sample_iteration(&plan, &downloaded, selection);',
         '''            let trace_end = crate::rag_steps::clock();
            let sampling = self.sample_iteration(&plan, &downloaded, selection);'''),
        ('            self.record_committed_audit_tokens(updates.token_events())?;',
         '''            if let (Some(start), Some(end)) = (trace_start, trace_end) {
                crate::rag_steps::step(&plan, start, end, timing.gpu_execution_ns());
            }
            self.record_committed_audit_tokens(updates.token_events())?;'''),
    ])
    edit(root / "service.rs", [
        ('        if std::env::var_os("RILEY_W3_EXPORT_OBSERVATIONS").as_deref()',
         '''        #[cfg(feature = "cuda")]
        crate::rag_steps::export();
        if std::env::var_os("RILEY_W3_EXPORT_OBSERVATIONS").as_deref()'''),
        ('        self.observations.record(RequestObservation {',
         '''        #[cfg(feature = "cuda")]
        if let Some(first) = self.time_to_first_token {
            crate::rag_steps::visible(&self.metadata.request_id, self.admitted_at, first);
        }
        self.observations.record(RequestObservation {'''),
    ])
    with (root / "rag_steps.rs").open("x") as output:
        output.write((HERE / "rag_steps.rs").read_text())


if __name__ == "__main__":
    instrument(Path(sys.argv[1]) / "crates/riley-server/src")
