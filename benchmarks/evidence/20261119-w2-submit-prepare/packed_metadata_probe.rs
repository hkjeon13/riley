use std::hint::black_box;
use std::time::Instant;

use riley_cuda::PackedBatchHostV1;

struct Fixture {
    offsets: Vec<u32>,
    ids: Vec<u32>,
    valid: Vec<u16>,
    slots: Vec<u32>,
    positions: Vec<u32>,
}

impl Fixture {
    fn new(sequences: u32, context: u32, rows_per_sequence: u32, reverse_rows: bool) -> Self {
        let pages_per_sequence = context.div_ceil(16);
        let total_pages = sequences * pages_per_sequence;
        let offsets = (0..=sequences)
            .map(|slot| slot * pages_per_sequence)
            .collect();
        // A descending physical-page order keeps the two host builds on the same block validation path.
        let ids = (0..total_pages).rev().collect();
        let mut valid = vec![16; total_pages as usize];
        let mut slots = Vec::new();
        let mut positions = Vec::new();
        for slot in 0..sequences {
            let tail = (slot + 1) * pages_per_sequence - 1;
            valid[tail as usize] = (1 + (context - 1) % 16) as u16;
            for position in context - rows_per_sequence..context {
                slots.push(slot);
                positions.push(position);
            }
        }
        if reverse_rows {
            slots.reverse();
            positions.reverse();
        }
        Self {
            offsets,
            ids,
            valid,
            slots,
            positions,
        }
    }

    fn construct(&self) -> PackedBatchHostV1<'_> {
        let f = black_box(self);
        PackedBatchHostV1::new(
            &f.offsets,
            &f.ids,
            &f.valid,
            &f.slots,
            &f.positions,
            f.ids.len() as u64,
        )
        .expect("fixture must pass the actual public Rust constructor")
    }
}

fn main() {
    let fixtures = [
        ("decode32-129", Fixture::new(32, 129, 1, false)),
        ("decode32-1025", Fixture::new(32, 1025, 1, false)),
        ("decode32-4097", Fixture::new(32, 4097, 1, false)),
        ("prefill1024-ordered", Fixture::new(4, 1024, 256, false)),
        ("prefill1024-reversed", Fixture::new(4, 1024, 256, true)),
    ];
    for (name, fixture) in fixtures {
        let iterations = 10000;
        for _ in 0..1000 {
            black_box(fixture.construct());
        }
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(fixture.construct());
        }
        println!(
            "{name} {:.3}",
            start.elapsed().as_secs_f64() * 1e9 / f64::from(iterations)
        );
    }
}
