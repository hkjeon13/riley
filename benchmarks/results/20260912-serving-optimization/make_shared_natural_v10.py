from pathlib import Path
b=Path('/tmp/riley-shared-integration-v10');s=(b/'crates/riley-scheduler/tests/shared_model_diagnostic_gpu.rs').read_text();a=s.index('#[test]');z=s.index('    let path =',a)
s=s[:a]+'''#[test]
#[ignore = "requires CUDA, checkpoint and predeclared natural corpus"]
fn shared_natural_teacher_forced() -> Result {
    let path=std::env::var_os("RILEY_NATURAL_CORPUS").ok_or("corpus missing")?;
    let data: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(path)?)?;
    assert_eq!(data.len(),64);
    let base=std::env::var_os("RILEY_SHARED_DUMP").ok_or("dump missing")?;
    std::fs::create_dir(base)?;
    for (case,chunk) in data.chunks_exact(8).enumerate() {
        let corpus: Vec<Vec<u32>>=chunk.iter().map(|x|x["tokens"].as_array().unwrap().iter().map(|t|t.as_u64().unwrap() as u32).collect()).collect();
        assert!(corpus.iter().all(|x|x.len()==160));run_mode(&corpus,case)?;
    }
    Ok(())
}
fn run_mode(corpus: &[Vec<u32>], case: usize) -> Result {
    let capacity=8;let cancel_and_replace=false;let alternate=false;
'''+s[z:]
s=s.replace('let prompt = (0..128).map(|i| (i * 311 + r * 977 + 13) % 49152).collect();','let prompt = corpus[r as usize][..128].to_vec();').replace('RequestDescriptor::new(prompt, 32 - r as usize * 3)','RequestDescriptor::new(prompt, 32)')
s=s.replace('ok_or("dump path missing")?);','ok_or("dump path missing")?).join(format!("batch{case}"));')
s=s.replace('            samples[slot] = SampledIterationToken::new(oracle.greedy_token()?, false);','            samples[slot] = SampledIterationToken::new(corpus[work.request_id().get() as usize - 1][work.target_logical_length()], false);')
s=s.replace('(0..capacity).map(|r| 32 - r * 3).sum::<usize>()','capacity * 32')
p=b/'crates/riley-scheduler/tests/shared_natural_model_gpu.rs';p.write_text(s)
s=Path('/tmp/run_shared_diagnostic_v10.py').read_text().replace("env['RILEY_SHARED_DUMP']=str(r/'shared-model-diagnostic-v10-r1')", "env['RILEY_SHARED_DUMP']=str(r/'shared-natural-v10/dumps')\nenv['RILEY_NATURAL_CORPUS']=str(r/'shared-natural-v10/corpus.json')").replace('shared_model_diagnostic_gpu','shared_natural_model_gpu').replace('model-diagnostic','natural-model').replace("'shared-v10-r2-'","'shared-v10-'");Path('/tmp/run_shared_natural_v10.py').write_text(s)
