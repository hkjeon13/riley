    #[test]
    fn short_only_prefill_budget_preserves_long_and_decode_bounds() {
        fn planned_tokens(limit: Option<usize>, decoding: usize, lengths: &[usize]) -> usize {
            let mut config = test_scheduler_with_max_active_sequences(2).config().clone();
            config.max_active_sequences = 32;
            config.max_waiting_requests = 32;
            config.max_waiting_prompt_tokens = 4096;
            config.max_sequence_tokens = 8192;
            config.iteration_token_budget = 1024;
            config.max_prefill_chunk_tokens = 1024;
            config.max_promised_kv_blocks = 512;
            let mut scheduler = Scheduler::new(config, KvLayout::checked(1, 512, 1, 64).unwrap()).unwrap();
            scheduler.short_prefill_burst_decode_limit = limit;
            scheduler.set_prefill_burst_policy(Some(crate::PrefillBurstPolicy {
                steady_budget: 256,
                max_decode_rows: 32,
                min_prompt_tokens: 512,
            })).unwrap();

            if decoding > 0 {
                for _ in 0..decoding {
                    scheduler.submit(RequestDescriptor::new(vec![1], 2), 0).unwrap();
                }
                let plan = scheduler.plan_iteration(1).unwrap().into_parts().0.unwrap();
                assert_eq!(plan.total_tokens(), decoding);
                let output = (0..decoding).map(|index| {
                    IterationOutput::new(OutputSlot::new(u32::try_from(index).unwrap()), 2, false)
                }).collect();
                let result = IterationResult::new(plan.iteration_id(), output, 1, 0).unwrap();
                scheduler.complete_iteration(&result, 2).unwrap();
            }
            for &length in lengths {
                scheduler.submit(RequestDescriptor::new(vec![1; length], 2), 2).unwrap();
            }
            let plan = scheduler.plan_iteration(3).unwrap().into_parts().0.unwrap();
            assert_eq!(plan.decode_items().len(), decoding);
            plan.total_tokens()
        }

        let short = [72; 8];
        for (limit, decoding, lengths, expected) in [
            (None, 0, short.as_slice(), 256),
            (Some(8), 0, short.as_slice(), 512),
            (Some(8), 9, short.as_slice(), 256),
            (Some(16), 9, short.as_slice(), 512),
            (Some(16), 17, short.as_slice(), 256),
            (Some(16), 0, &[63; 8], 256),
            (Some(16), 0, &[72, 600], 672),
        ] {
            assert_eq!(planned_tokens(limit, decoding, lengths), expected);
        }
    }
