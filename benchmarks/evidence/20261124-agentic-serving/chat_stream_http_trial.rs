// Included only in the isolated test copy of service.rs.
#[test]
fn recorded_final_chat_streams_over_tcp() {
    use std::io::{BufRead, BufReader};
    let cases: Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("RILEY_STREAM_CASES").unwrap()).unwrap(),
    )
    .unwrap();
    let tokens: Vec<String> = cases[0]["pieces"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap().to_owned())
        .collect();
    let wave_sizes = [1_usize, 8, 16];
    let waves = 10;
    let total = wave_sizes.iter().sum::<usize>() * waves;
    let backend = TestBackend::new((0..total).map(|_| Script::Complete(tokens.clone())));
    let template = ChatTemplate::compile(&riley_model::ChatTemplateSource {
        template: "{% for m in messages %}{{ m.content }}{% endfor %}".to_owned(),
        bos_token: None,
        eos_token: None,
    })
    .unwrap();
    let mut config = test_config();
    config.worker_threads = 16;
    let server =
        start_server_with_chat(config, backend, None, false, Some(Arc::new(template))).unwrap();
    let addr = server.local_address();
    let body = post_body_to("/v1/chat/completions", br#"{"model":"fixture-model","messages":[{"role":"user","content":"replay"}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":64}"#);
    let mut rows = Vec::new();
    for size in wave_sizes {
        for _ in 0..waves {
            let barrier = std::sync::Barrier::new(size);
            let rows_one = thread::scope(|scope| {
                let mut handles = Vec::new();
                for _ in 0..size {
                    handles.push(scope.spawn(|| {
                        barrier.wait();
                        let started = Instant::now();
                        let mut sock = TcpStream::connect(addr).unwrap();
                        sock.write_all(&body).unwrap();
                        let mut reader = BufReader::new(sock);
                        let mut payloads = Vec::new();
                        let mut line = String::new();
                        loop {
                            line.clear();
                            assert!(reader.read_line(&mut line).unwrap() > 0);
                            let Some(data) = line.strip_prefix("data: ") else {continue};
                            let at = started.elapsed().as_nanos();
                            if data.trim_end() == "[DONE]" {break}
                            payloads.push((at,data.trim_end().to_owned()));
                        }
                        assert_eq!(payloads.len(),tokens.len()+2);
                        for ((_,payload),token) in payloads.iter().zip(&tokens) {
                            let v:Value = serde_json::from_str(payload).unwrap();
                            assert_eq!(v["choices"][0]["delta"]["content"].as_str(), Some(token.as_str()));
                        }
                        let finish:Value = serde_json::from_str(&payloads[tokens.len()].1).unwrap();
                        let usage:Value = serde_json::from_str(&payloads[tokens.len()+1].1).unwrap();
                        assert_eq!(finish["choices"][0]["finish_reason"], "stop");
                        assert_eq!(usage["usage"]["completion_tokens"],tokens.len());
                        json!({"clients":size,"first_ns":payloads[0].0,"last_ns":payloads[tokens.len()-1].0})
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect::<Vec<_>>()
            });
            rows.extend(rows_one);
        }
    }
    server.shutdown_with_metrics().unwrap();
    std::fs::write(
        std::env::var_os("RILEY_STREAM_OUT").unwrap(),
        serde_json::to_vec(&rows).unwrap(),
    )
    .unwrap();
}
