from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs';s=p.read_text().replace('logits.iter().all(|&v|v==0)','all_zero(logits)').replace('bytes[e.rows.len()*RESULT_BYTES..].iter().all(|&v|v==0)','all_zero(&bytes[e.rows.len()*RESULT_BYTES..])');pos=s.index('pub fn validate_result<');s=s[:pos]+'''// A reduction permits vectorized scanning while checking every inactive byte.
fn all_zero(bytes:&[u8])->bool {bytes.iter().fold(0u8,|bits,&value|bits|value)==0}
'''+s[pos:];p.write_text(s)
p=r/'crates/riley-runtime/src/sampling.rs';s=p.read_text();needle='        self.probabilities.fill(0.0);';pos=s.index(needle,s.index('pub fn process_bf16_native'));s=s[:pos]+'''        // Preserve the full public distribution, without materializing candidates
        // or applying an identity repetition penalty for deterministic sampling.
        if params.temperature == 0.0 && params.repetition_penalty == 1.0 {
            let mask = match constraints {
                TokenConstraints::AllowAll => None,
                TokenConstraints::AllowedMask(mask) => Some(mask),
            };
            let mut winner = None;
            let mut maximum = f32::NEG_INFINITY;
            for (token, bytes) in logits_bf16_native.chunks_exact(BF16_BYTES).enumerate() {
                if mask.is_none_or(|mask| mask[token]) {
                    let value = decode_finite_bf16(bytes);
                    if value > maximum { maximum = value; winner = Some(token); }
                }
            }
            let winner = winner.ok_or(SamplingError::AllMasked)?;
            self.probabilities.fill(0.0);
            self.log_probabilities.fill(f64::NEG_INFINITY);
            self.processed_logits.fill(f64::NEG_INFINITY);
            self.probabilities[winner] = 1.0;
            self.log_probabilities[winner] = 0.0;
            self.processed_logits[winner] = f64::from(maximum);
            return Ok(self.distribution(true, 1, Some(winner as u32)));
        }

'''+s[pos:];p.write_text(s)
