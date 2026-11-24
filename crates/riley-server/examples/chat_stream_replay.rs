//! Compare chat SSE delivery on recorded text deltas, with no model or network timing.
use riley_server::chat::{ChatChunkShaper, ToolStyle};
use riley_server::domain::{FinishReason, GenerationEvent, RequestMetadata, TokenUsage};
use riley_server::openai::SseStreamEncoder;
use serde::Deserialize;
use serde_json::json;
use std::hint::black_box;
use std::time::Instant;

#[derive(Deserialize)]
struct Case {
    style: Option<String>,
    pieces: Vec<String>,
}

fn render(
    events: &[GenerationEvent],
    style: Option<ToolStyle>,
    direct: bool,
    mut emit: impl FnMut(String),
) {
    let metadata = RequestMetadata::new("cmpl-replay", "coder-7b", 1).unwrap();
    let mut encoder = SseStreamEncoder::new_with_options(metadata, false, true, 512, 8192);
    let mut chat = ChatChunkShaper::new(style);
    for event in events {
        let frame = if direct {
            chat.encode_event(&mut encoder, event).unwrap()
        } else {
            chat.shape(&encoder.encode_event(event).unwrap())
        };
        emit(frame);
    }
    assert_eq!(encoder.encode_done().unwrap(), "data: [DONE]\n\n");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cases: Vec<Case> = serde_json::from_reader(std::io::stdin().lock())?;
    let repeats: usize = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "200".into())
        .parse()?;
    assert!(repeats > 0 && !cases.is_empty());
    let mut results = Vec::new();
    for case in cases {
        let style = match case.style.as_deref() {
            None => None,
            Some("hermes") => Some(ToolStyle::Hermes),
            Some("llama") => Some(ToolStyle::LlamaJson),
            _ => return Err("unknown tool style".into()),
        };
        let deltas = case.pieces.len();
        assert!(deltas > 0);
        let mut events: Vec<_> = case
            .pieces
            .into_iter()
            .map(|text| GenerationEvent::TokenDelta { text })
            .collect();
        events.push(GenerationEvent::Finished {
            reason: FinishReason::Stop,
            usage: TokenUsage::new(32, deltas.try_into()?)?,
        });
        let mut reference = Vec::new();
        render(&events, style, false, |frame| reference.push(frame));
        let mut candidate = Vec::new();
        render(&events, style, true, |frame| candidate.push(frame));
        assert_eq!(reference, candidate);
        let bytes = reference.iter().map(String::len).sum::<usize>();
        let mut rounds = Vec::new();
        for round in 0..10 {
            let mut times = [0; 2];
            for direct in [round % 2 != 0, round % 2 == 0] {
                let start = Instant::now();
                for _ in 0..repeats {
                    render(&events, style, direct, |frame| {
                        black_box(frame);
                    });
                }
                times[usize::from(direct)] = start.elapsed().as_nanos();
            }
            rounds.push(times);
        }
        results.push(json!({"style":case.style, "deltas":deltas, "bytes":bytes, "repeats":repeats,"round_ns":rounds}));
    }
    println!("{}", serde_json::to_string(&results)?);
    Ok(())
}
