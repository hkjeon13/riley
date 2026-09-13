from pathlib import Path
import hashlib,json
root=Path('/tmp/riley-opt-260912');d=root/'prefill-routing-v1'
a=(d/'all.bf16').read_bytes();m=(d/'mixed.bf16').read_bytes()
assert a==(root/'flashinfer-prefill-sync-v1/output-v2.bf16').read_bytes()
assert m==(d/'mixed-memcheck.bf16').read_bytes()
offset=0;equal=0;untouched=0
for rows in [[1,17,33],[16,32],[127,128,129],[1]*31+[993]]:
    for n in rows:
        count=n*576*2
        if n==1:
            # BF16(-99.0) = 0xc2c6, native little endian.
            assert m[offset:offset+count]==b'\xc6\xc2'*(count//2)
            untouched+=count//2
        else:
            assert m[offset:offset+count]==a[offset:offset+count]
            equal+=count//2
        offset+=count
assert offset==len(a)
assert '0 hazards displayed (0 errors, 0 warnings)' in (d/'racecheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in (d/'memcheck.log').read_text()
files=['kernels/optional/flashinfer_prefill.cu','kernels/optional/flashinfer_api.h','benchmarks/analysis/flashinfer_prefill_probe.cu']
result={'prefill_values_bitwise_equal_full_adapter':equal,'decode_values_untouched_in_matched_cases':untouched,'full_adapter_unchanged_values':len(a)//2,'all_sha256':hashlib.sha256(a).hexdigest(),'mixed_sha256':hashlib.sha256(m).hexdigest(),'memcheck_racecheck_outputs_bitwise_equal':True,'source_sha256':{f:hashlib.sha256((root/'hardware-validation-source'/f).read_bytes()).hexdigest() for f in files}}
(d/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
