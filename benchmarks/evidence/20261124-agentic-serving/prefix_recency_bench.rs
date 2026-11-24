#[allow(dead_code)]
#[path = "../src/prefix_cache.rs"]
mod prefix_cache;
use prefix_cache::{PrefixCache, ROOT};
use riley_runtime::paged_kv::{KV_BLOCK_SIZE, KvBlockPool, KvLayout};
use riley_scheduler::error;
use serde_json::json;
use std::time::Instant;

fn sample(tokens: usize, chunk: usize, paths: usize) -> (u128, u128) {
    let n = paths * tokens / KV_BLOCK_SIZE;
    let mut pool = KvBlockPool::new(KvLayout::checked(1, n, 1, 64).unwrap()).unwrap();
    let mut sequence = pool.create_sequence(n * KV_BLOCK_SIZE).unwrap();
    let reserve = sequence.reserve_to(&mut pool, n * KV_BLOCK_SIZE).unwrap();
    sequence.commit(&mut pool, reserve).unwrap();
    let blocks: Vec<_> = (0..n)
        .map(|_| sequence.share_next_block(&mut pool).unwrap())
        .collect();
    let mut cache = PrefixCache::new(n).unwrap();
    let mut tips = vec![ROOT; paths];
    let mut touch_ns = 0;
    let start = Instant::now();
    for first in (0..tokens / KV_BLOCK_SIZE).step_by(chunk / KV_BLOCK_SIZE) {
        for (path, tip) in tips.iter_mut().enumerate() {
            for offset in 0..chunk / KV_BLOCK_SIZE {
                let i = path * tokens / KV_BLOCK_SIZE + first + offset;
                *tip = cache
                    .insert(*tip, &[i as u32; KV_BLOCK_SIZE], blocks[i])
                    .unwrap();
            }
            let touch_start = Instant::now();
            cache.touch_path(*tip).unwrap();
            touch_ns += touch_start.elapsed().as_nanos();
        }
    }
    std::hint::black_box(cache.len());
    (start.elapsed().as_nanos(), touch_ns)
}

fn main() {
    let mut results = Vec::new();
    for tokens in [8192, 32768, 131072] {
        for chunk in [32, 256] {
            for paths in [1, 2] {
                sample(tokens, chunk, paths);
                let ns: Vec<_> = (0..5).map(|_| sample(tokens, chunk, paths)).collect();
                results.push(json!({"tokens_per_path":tokens,"chunk":chunk,"paths":paths,"publish_touch_ns":ns}));
            }
        }
    }
    println!("{}", serde_json::Value::Array(results));
}
