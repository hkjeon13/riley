"""Add an isolated transport trace to the archived Coder-14B experiment."""

from hashlib import sha256
from pathlib import Path
import subprocess
import sys

SOURCE_HASHES = {
    "service.rs": "12b4bf4bb16c588b3b130bd1e9fd1b76f376ae8632f3997b29ce39b7e39198c9",
    "w3_trace.rs": "fbd1fba632d8eaf9e1555dc8c03dc2e4ab218b18d7ec4457f70d6ee5a3c89279",
}
RUNNER_HASHES = {
    "load.py": "85df2bb6f0fc8a653d92d9711a7d46709e0c029d6e18e67ec9cc0734d5b6c9f4",
    "replay.py": "ae66ca1446f777dd077f9de57fd619cac8278bd268715597c4c5e5da979d0b6e",
    "run_case.py": "0fc60a66cedadf703255e1e47314365f1c655dc2af614408065562b8450dc28e",
}


def check_inputs(root, expected):
    for name, digest in expected.items():
        if sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"archived input changed: {name}")


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"expected one archived anchor: {old!r}")
    return source.replace(old, new, 1)


def enable_transport_clock(source):
    replacements = (
        ("enum Mode { Off, Full, Admission }", "enum Mode { Off, Full, Admission, Transport }"),
        ('        Some(value) if value == "admission" => Mode::Admission,',
         '        Some(value) if value == "admission" => Mode::Admission,\n'
         '        Some(value) if value == "transport" => Mode::Transport,'),
        ('pub(crate) fn parsed_http() {\n    if let Some(ns) = clock() {',
         'pub(crate) fn transport_clock() -> Option<u64> {\n'
         '    if matches!(mode(), Mode::Transport) { monotonic_clock() } else { None }\n'
         '}\n\npub(crate) fn parsed_http() {\n    if let Some(ns) = clock().or_else(transport_clock) {'),
        ('    if !matches!(mode(), Mode::Off) {',
         '    if matches!(mode(), Mode::Full | Mode::Admission) {'),
    )
    for old, new in replacements:
        source = replace_once(source, old, new)
    return source


def transport_marks(source):
    source = replace_once(source, "    let http_parsed = crate::w3_trace::take_http();",
                         "    let http_parsed = crate::w3_trace::take_http();\n"
                         "    let transport_submit = crate::w3_trace::transport_clock();")
    source = replace_once(source, "    let queue_wait = admission_started.elapsed();",
                         "    let transport_admitted = crate::w3_trace::transport_clock();\n"
                         "    let queue_wait = admission_started.elapsed();")
    source = replace_once(
        source,
        '    crate::w3_trace::http_phase(&submitted.metadata().request_id, "http_parsed", http_parsed);',
        '''    let request_id = &submitted.metadata().request_id;
    crate::w3_trace::http_phase(request_id, "http_parsed", http_parsed);
    crate::w3_trace::http_phase(request_id, "transport_submit", transport_submit);
    crate::w3_trace::http_phase(request_id, "transport_admitted", transport_admitted);''',
    )
    source = replace_once(source, '''                "visible",
                crate::w3_trace::clock(),
''', '''                "visible",
                crate::w3_trace::clock().or_else(crate::w3_trace::transport_clock),
''')
    source = replace_once(source, "    if write_sse_head(stream).is_err() {",
                         '''    crate::w3_trace::http_phase(
        &tracker.metadata.request_id,
        "head_write_started",
        crate::w3_trace::transport_clock(),
    );
    if write_sse_head(stream).is_err() {''')
    start = source.index("fn stream_completion(")
    stop = source.index("\nfn ", start + 1)
    body = source[start:stop]
    body = replace_once(body, "    let mut encoder = SseStreamEncoder::new_with_options(",
                       '''    crate::w3_trace::http_phase(
        &tracker.metadata.request_id,
        "head_write_returned",
        crate::w3_trace::transport_clock(),
    );
    let mut encoder = SseStreamEncoder::new_with_options(''')
    visible = '''                if matches!(&event, GenerationEvent::TokenDelta { .. })
                    || matches!(&event, GenerationEvent::CommittedToken { text, .. } if !text.is_empty())
                {
                    tracker.token_delta();
                }'''
    body = replace_once(body, visible, '''                let visible = matches!(&event, GenerationEvent::TokenDelta { .. })
                    || matches!(&event, GenerationEvent::CommittedToken { text, .. } if !text.is_empty());
                let first_text = visible && tracker.time_to_first_token.is_none();
                if visible {
                    tracker.token_delta();
                }''')
    body = replace_once(body, '''                if terminal {
                    let Ok(done)''', '''                if first_text {
                    crate::w3_trace::http_phase(
                        &tracker.metadata.request_id,
                        "text_write_returned",
                        crate::w3_trace::transport_clock(),
                    );
                }
                if terminal {
                    let Ok(done)''')
    return source[:start] + body + source[stop:]


def prepare(source, runner, load_patch):
    crate = source / "crates/riley-server/src"
    check_inputs(crate, SOURCE_HASHES)
    check_inputs(runner, RUNNER_HASHES)
    trace, service = (crate / name for name in ("w3_trace.rs", "service.rs"))
    changed_trace = enable_transport_clock(trace.read_text())
    changed_service = transport_marks(service.read_text())

    replayer = runner / "replay.py"
    changed_replayer = replace_once(
        replayer.read_text(), "capture_token_ids=True, connection=connection)",
        "capture_token_ids=True, capture_transport_timing=True, connection=connection)",
    )
    entry = runner / "run_case.py"
    changed_entry = replace_once(entry.read_text(), '("control", "admission")', '("control", "transport")')
    changed_entry = replace_once(changed_entry, '("admission" if trace == "admission" else "0")',
                                 '("transport" if trace == "transport" else "0")')
    changed_entry = replace_once(changed_entry, '"stage_trace": trace == "admission"',
                                 '"stage_trace": trace == "transport"')
    # The benchmark-only change is applied to the archived copy, which also records response IDs.
    with load_patch.open("rb") as patch:
        subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "load.py"],
                       cwd=runner, stdin=patch, check=True)
    for path, text in ((trace, changed_trace), (service, changed_service),
                       (replayer, changed_replayer), (entry, changed_entry)):
        path.write_text(text)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: prepare.py SOURCE_DIRECTORY RUNNER_DIRECTORY LOAD_PATCH")
    prepare(*map(Path, sys.argv[1:]))
