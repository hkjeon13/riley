
#[cfg(test)]
mod staged_event_reuse_probe {
    use super::*;
    use std::time::Instant;

    #[test]
    fn stage_and_deliver_on_active_channels() {
        let stats = Arc::new(EngineStats::default());
        let mut requests = BTreeMap::new();
        let mut receivers = Vec::new();
        for id in 1..=32 {
            let (events, receiver) = std::sync::mpsc::sync_channel(1);
            stats.active_requests.fetch_add(1, Ordering::Relaxed);
            requests.insert(
                EngineRequestId(id),
                RequestControl {
                    cancellation: Arc::new(AtomicU8::new(REQUEST_LIVE)),
                    events,
                    stats: Arc::clone(&stats),
                },
            );
            receivers.push(receiver);
        }

        let mut sample = Vec::new();
        for step in 0..11000 {
            let events = requests
                .keys()
                .map(|id| BackendEvent::new(*id, GenerationEvent::TokenDelta {
                    text: String::from("byte fixture"),
                }))
                .collect();
            let start = Instant::now();
            stage_events(&mut requests, events);
            deliver_staged_events();
            let elapsed = start.elapsed().as_nanos();
            if step >= 1000 {
                sample.push(elapsed);
            }
            for receiver in &receivers {
                assert_eq!(
                    receiver.try_recv(),
                    Ok(GenerationEvent::TokenDelta { text: String::from("byte fixture") }),
                );
            }
        }
        sample.sort_unstable();
        let capacity = OUTBOX.with(|outbox| outbox.borrow().staged.capacity());
        println!(
            "OUTBOX_PROBE samples={} p50_ns={} p95_ns={} retained_capacity={}",
            sample.len(), sample[sample.len() / 2], sample[sample.len() * 95 / 100], capacity,
        );
    }
}
