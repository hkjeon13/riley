import pathlib,re
p=pathlib.Path("crates/riley-server/src/engine.rs");s=p.read_text()
helper='''
pub(crate) mod diag {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;
    pub static ACC: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
    pub static N: AtomicU64 = AtomicU64::new(0);
    thread_local! { pub static LAST: std::cell::Cell<Option<Instant>> = const { std::cell::Cell::new(None) }; }
    pub fn mark(index: usize) {
        let now = Instant::now();
        LAST.with(|last| {
            if let Some(previous) = last.get() { ACC[index].fetch_add((now - previous).as_nanos() as u64, Ordering::Relaxed); }
            last.set(Some(now));
        });
    }
    pub fn tick() {
        if (N.fetch_add(1, Ordering::Relaxed) + 1) % 512 == 0 {
            let v: Vec<String> = ["loop", "plan", "authorize", "execute", "sample", "commit", "events", "publish"].iter().enumerate()
                .map(|(i, name)| format!("{name}={:.1}", ACC[i].swap(0, Ordering::Relaxed) as f64 / 512e3)).collect();
            eprintln!("DIAGSTEP us/iter {}", v.join(" "));
        }
    }
}
'''
s=s.replace("\nfn publish_events<B: EngineBackend>(",helper+"\nfn publish_events<B: EngineBackend>(",1)
def after(pattern,mark,count=1):
    global s
    m=list(re.finditer(pattern,s,flags=re.S));assert len(m)>=1,pattern
    m=m[0];s=s[:m.end()]+mark+s[m.end():]
# step start: time since last mark = worker loop (cancellations, commands)
after(r"fn step\(&mut self\) -> Result<Vec<BackendEvent>, BackendError> \{\n            if let Some\(event\) = self\.pending_events\.pop_front\(\) \{\n                return Ok\(vec!\[event\]\);\n            \}\n","            super::diag::mark(0);\n")
after(r"\.map_err\(\|source\| internal\(format!\(\"iteration planning failed: \{source\}\"\)\)\)\?;\n","            super::diag::mark(1);\n")
after(r"\.map_err\(\|e\|internal\(format!\(\"V3 authority failed: \{e\}\"\)\)\)\?;\n","                    super::diag::mark(2);\n")
after(r"\.map_err\(\|e\|internal\(format!\(\"V3 iteration failed: \{\}\",e\.error\(\)\)\)\)\?;\n","                    super::diag::mark(3);\n")
after(r"let sampling = self\.sample_iteration\(&plan, &downloaded, selection\);\n","            super::diag::mark(4);\n")
s=s.replace("            self.record_committed_audit_tokens(updates.token_events())?;\n","            super::diag::mark(5);\n            self.record_committed_audit_tokens(updates.token_events())?;\n",1)
i=s.index("            events.extend(self.process_completions(updates.completions())?);");j=s.index("            Ok(events)\n",i)
s=s[:j]+"            super::diag::mark(6);\n"+s[j:]
old="                Ok(events) => publish_events(&mut backend, &mut requests, events),\n                Err(error) => {\n                    eprintln!(\"riley engine backend failure"
assert old in s
s=s.replace(old,"                Ok(events) => {publish_events(&mut backend, &mut requests, events);diag::mark(7);diag::tick();}\n                Err(error) => {\n                    eprintln!(\"riley engine backend failure",1)
p.write_text(s)
