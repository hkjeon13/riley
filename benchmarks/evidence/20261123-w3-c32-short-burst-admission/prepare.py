"""Add a trace mode limited to admission stages and engine execution intervals."""
from hashlib import sha256
from pathlib import Path
import sys


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"missing trace anchor: {before}")
    return source.replace(before, after, 1)


root = Path(sys.argv[1]) / "crates/riley-server/src"
paths = {
    "w3_trace.rs": "49151531eeacb6bba9817fddcf2a906ca631cdfa38c8b029f0244f041a1ab905",
    "engine.rs": "dd1eeb80a8df52c78d361cc5dcfa0222309818e4c69ba81fd707bdba964fb4ad",
}
for name, expected in paths.items():
    if sha256((root / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"archived source drift: {name}")

trace_path = root / "w3_trace.rs"
source = trace_path.read_text()
source = replace_once(source, "    Phase(u64, &'static str, u64),",
                      """    Phase(u64, &'static str, u64),
    EngineSpan { start: u64, end: u64, rows: usize, decode: usize },""")
source = replace_once(source, "static ACTIVE: OnceLock<bool> = OnceLock::new();",
                      """#[derive(Clone, Copy)]
enum Mode { Off, Full, Admission }

static MODE: OnceLock<Mode> = OnceLock::new();

fn mode() -> Mode {
    *MODE.get_or_init(|| match std::env::var_os("RILEY_W3_STAGE_TRACE").as_deref() {
        Some(value) if value == "1" => Mode::Full,
        Some(value) if value == "admission" => Mode::Admission,
        _ => Mode::Off,
    })
}

pub(crate) fn admission_clock() -> Option<u64> {
    if matches!(mode(), Mode::Admission) { monotonic_clock() } else { None }
}

pub(crate) fn engine_span(start: u64, end: u64, rows: usize, decode: usize) {
    push(Record::EngineSpan { start, end, rows, decode });
}""")
source = replace_once(source, """#[allow(unsafe_code)] // clock_gettime writes into a valid, exclusively borrowed timespec.
pub(crate) fn clock() -> Option<u64> {
    if !*ACTIVE.get_or_init(|| std::env::var_os("RILEY_W3_STAGE_TRACE").is_some_and(|v| v == "1")) {
        return None;
    }""",
                      """pub(crate) fn clock() -> Option<u64> {
    if matches!(mode(), Mode::Full) { monotonic_clock() } else { None }
}

#[allow(unsafe_code)] // clock_gettime writes into a valid, exclusively borrowed timespec.
fn monotonic_clock() -> Option<u64> {""")
source = replace_once(source, """    if let Some(ns) = clock() {
        push(Record::Phase(id, name, ns));""",
                      """    if !matches!(mode(), Mode::Off) {
        if let Some(ns) = monotonic_clock() {
            push(Record::Phase(id, name, ns));
        }""")
source = replace_once(source, """                Record::Phase(id, phase, ns) => eprintln!("W3-REQUEST-PHASE {id} {phase} {ns}"),""",
                      """                Record::Phase(id, phase, ns) => eprintln!("W3-REQUEST-PHASE {id} {phase} {ns}"),
                Record::EngineSpan { start, end, rows, decode } =>
                    eprintln!("W3-ENGINE-SPAN {start} {end} {rows} {decode}"),""")
trace_path.write_text(source)

engine_path = root / "engine.rs"
source = engine_path.read_text()
source = replace_once(source, "            let (mut downloaded, timing, staged_shape) =",
                      """            let admission_trace_start = crate::w3_trace::admission_clock();
            let (mut downloaded, timing, staged_shape) =""")
source = replace_once(source, "            if let Some(start) = trace_start {",
                      """            if let Some(start) = admission_trace_start {
                if let Some(end) = crate::w3_trace::admission_clock() {
                    crate::w3_trace::engine_span(
                        start, end, plan.total_tokens(), plan.decode_items().len(),
                    );
                }
            }
            if let Some(start) = trace_start {""")
engine_path.write_text(source)
