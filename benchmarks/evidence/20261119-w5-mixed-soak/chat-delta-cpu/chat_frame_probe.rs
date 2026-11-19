#![allow(unsafe_code)]

use sha2::{Digest, Sha256};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

use crate::chat::{ChatChunkShaper, ToolStyle};
use crate::domain::{
    FinishReason, GenerationEvent, RequestMetadata, ServiceErrorClass, TokenLogprob,
    TokenProbability, TokenUsage,
};
use crate::openai::{LogprobShape, SseStreamEncoder};

#[global_allocator]
static ALLOCATOR: MeasuringAllocator = MeasuringAllocator;

static MEASURE: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

struct MeasuringAllocator;

impl MeasuringAllocator {
    fn allocated(size: usize) {
        if MEASURE.load(Relaxed) {
            ALLOCATIONS.fetch_add(1, Relaxed);
            BYTES.fetch_add(size, Relaxed);
        }
    }
}

// SAFETY: all methods delegate to System, preserving its pointers and layouts.
unsafe impl GlobalAlloc for MeasuringAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::allocated(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::allocated(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::allocated(new_size);
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[derive(Clone, Copy)]
struct Options {
    token_ids: bool,
    logprobs: bool,
    usage: bool,
    style: Option<ToolStyle>,
}

impl Options {
    fn encoder(self) -> SseStreamEncoder {
        SseStreamEncoder::new_with_options(
            RequestMetadata::new("cmpl-probe-12345", "fixture", u64::MAX).unwrap(),
            self.token_ids || self.logprobs,
            self.usage,
            1024,
            8192,
        )
        .with_response_options(self.token_ids, self.logprobs.then_some(LogprobShape::Chat))
    }
}

fn frame(
    encoder: &mut SseStreamEncoder,
    shaper: &mut ChatChunkShaper,
    event: &GenerationEvent,
    direct: bool,
) -> String {
    if direct
        && matches!(
            event,
            GenerationEvent::TokenDelta { .. } | GenerationEvent::CommittedToken { .. }
        )
    {
        encoder
            .encode_event_with_delta(event, |chunk| shaper.encode_delta(chunk))
            .unwrap()
    } else {
        shaper.shape(&encoder.encode_event(event).unwrap())
    }
}

fn token_events(parts: &[&str]) -> Vec<GenerationEvent> {
    parts
        .iter()
        .map(|piece| GenerationEvent::TokenDelta {
            text: (*piece).to_owned(),
        })
        .collect()
}

fn committed_events(parts: &[&str], chosen: f32) -> Vec<GenerationEvent> {
    parts
        .iter()
        .enumerate()
        .map(|(index, piece)| GenerationEvent::CommittedToken {
            text: (*piece).to_owned(),
            token_id: u32::MAX - index as u32,
            generated_index: index,
            prompt_token_ids: (index == 0).then(|| (0..4096).collect()),
            logprob: Some(Box::new(TokenLogprob {
                chosen: TokenProbability {
                    bytes: vec![0xed, 0x95, 0x9c],
                    logprob: chosen,
                },
                top: [
                    (-0.5, vec![0xff]),
                    (-12.75, b"x".to_vec()),
                    (-32.125, b" ".to_vec()),
                ]
                .into_iter()
                .map(|(logprob, bytes)| TokenProbability { bytes, logprob })
                .collect(),
            })),
        })
        .collect()
}

fn compare_wire(options: Options, events: &[GenerationEvent]) -> (usize, String) {
    let mut legacy = options.encoder();
    let mut direct = options.encoder();
    let mut legacy_shaper = ChatChunkShaper::new(options.style);
    let mut direct_shaper = ChatChunkShaper::new(options.style);
    let mut differences = 0;
    let mut wire = Sha256::new();
    for event in events {
        let old = frame(&mut legacy, &mut legacy_shaper, event, false);
        let new = frame(&mut direct, &mut direct_shaper, event, true);
        wire.update((old.len() as u64).to_le_bytes());
        wire.update(old.as_bytes());
        if old != new {
            if differences == 0 {
                println!("first unequal sampled frame old={old:?} new={new:?}");
            }
            differences += 1;
        }
    }
    assert_eq!(legacy.phase(), direct.phase());
    let done = legacy.encode_done().unwrap();
    assert_eq!(done, direct.encode_done().unwrap());
    wire.update(done.as_bytes());
    let digest = wire
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    (differences, digest)
}

fn measured(options: Options, events: &[GenerationEvent], direct: bool) -> (usize, usize) {
    ALLOCATIONS.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    MEASURE.store(true, Relaxed);
    for _ in 0..64 {
        let mut encoder = options.encoder();
        let mut shaper = ChatChunkShaper::new(options.style);
        for event in events {
            black_box(frame(&mut encoder, &mut shaper, black_box(event), direct));
        }
        black_box(encoder.encode_done().unwrap());
    }
    MEASURE.store(false, Relaxed);
    (ALLOCATIONS.load(Relaxed), BYTES.load(Relaxed))
}

fn terminal(events: &mut Vec<GenerationEvent>, reason: FinishReason) {
    events.push(GenerationEvent::Finished {
        reason,
        usage: TokenUsage::new(4096, events.len() as u64)
            .unwrap()
            .with_cached_prompt_tokens(32),
    });
}

#[test]
fn compare_chat_frame_wire_and_allocations() {
    let no_extras = Options {
        token_ids: false,
        logprobs: false,
        usage: true,
        style: None,
    };
    let ids = Options {
        token_ids: true,
        ..no_extras
    };
    let logprobs = Options {
        logprobs: true,
        ..no_extras
    };
    let mut ordinary = token_events(&["한", "a", "\"", "\n", "data: [DONE]\n\n", "", "😀"]);
    let mut tool = token_events(&[
        "<tool",
        "_call>",
        "{\"name\":\"weather\",",
        "\"arguments\":{\"city\":\"서울\"}}",
        "</tool_call>",
    ]);
    let mut llama_tool = token_events(&[
        "<|python_tag|>",
        "{\"name\":\"weather\",\"parameters\":",
        "{\"city\":\"서울\"}}",
    ]);
    let mut incomplete_tool = token_events(&["<tool_call>", "{\"name"]);
    let mut committed = committed_events(&["한", "", "x", "\u{0000}"], -0.123_456_78);
    let mut failed = token_events(&["before", "failure"]);
    failed.push(GenerationEvent::Failed {
        class: ServiceErrorClass::Internal,
    });
    for events in [&mut ordinary, &mut tool, &mut llama_tool, &mut committed] {
        terminal(events, FinishReason::Stop);
    }
    terminal(&mut incomplete_tool, FinishReason::Length);

    let hermes = Options {
        style: Some(ToolStyle::Hermes),
        ..no_extras
    };
    let llama = Options {
        style: Some(ToolStyle::LlamaJson),
        ..no_extras
    };
    let cases = [
        ("visible", no_extras, ordinary),
        ("hermes", hermes, tool),
        ("llama", llama, llama_tool),
        ("incomplete_tool", hermes, incomplete_tool),
        ("committed_ids", ids, committed.clone()),
        ("logprobs", logprobs, committed),
        ("failed", no_extras, failed),
    ];
    for (name, options, events) in cases {
        let (different_frames, digest) = compare_wire(options, &events);
        let old = measured(options, &events, false);
        let new = measured(options, &events, true);
        println!(
            "FRAME_PROBE {name} events={} repetitions=64 different={} old_calls={} new_calls={} old_bytes={} new_bytes={} sha256={digest}",
            events.len(),
            different_frames,
            old.0,
            new.0,
            old.1,
            new.1
        );
        assert_eq!(different_frames, 0, "{name}");
    }

    let metadata = RequestMetadata::new("cmpl-float", "fixture", 90).unwrap();
    let mut differences = 0;
    for raw in 0..4096 {
        let selected = -(raw as f32 / 133.7 + 0.000_010_013);
        let event = committed_events(&["x"], selected).remove(0);
        let encoder = SseStreamEncoder::new_with_options(metadata.clone(), true, true, 8, 8192)
            .with_response_options(false, Some(LogprobShape::Chat));
        let mut candidate = encoder.clone();
        let mut baseline = encoder;
        let old = frame(
            &mut baseline,
            &mut ChatChunkShaper::default(),
            &event,
            false,
        );
        let new = frame(
            &mut candidate,
            &mut ChatChunkShaper::default(),
            &event,
            true,
        );
        if old != new {
            differences += 1;
            if differences == 1 {
                println!(
                    "FLOAT_EXAMPLE f32_bits=0x{:08x} old={old:?} new={new:?}",
                    selected.to_bits()
                );
            }
        }
    }
    println!("FLOAT_SWEEP samples=4096 different={differences}");
    assert_eq!(differences, 0);
}
