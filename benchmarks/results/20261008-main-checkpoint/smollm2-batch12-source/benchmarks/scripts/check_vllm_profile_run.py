"""Validate one pinned vLLM-reference candidate run; never assert an E0 speedup."""
from __future__ import annotations
import argparse,hashlib,json,struct
from pathlib import Path
from jsonschema import Draft202012Validator,FormatChecker

def token_hash(tokens):
    if not tokens or any(type(t) is not int or not 0<=t<2**32 for t in tokens):
        raise ValueError('reference token IDs must be nonempty u32 integers')
    return hashlib.sha256(b''.join(struct.pack('<I',t) for t in tokens)).hexdigest()

def validate(run,binding):
    schema=json.loads((Path(__file__).resolve().parents[1]/'schemas/vllm-profile-run.schema.json').read_text())
    Draft202012Validator(schema,format_checker=FormatChecker()).validate(run)
    for field in ['source','environment','workload']:
        if run[field]!=binding[field]:raise ValueError(f'{field} differs from pinned preparation')
    if run['status']!='success' or run['failure_count']!=0:raise ValueError('run failed')
    w=run['workload']
    if (w['concurrency'],w['prompt_tokens'],w['output_tokens'])!=(1,128,32):raise ValueError('unsupported workload')
    if w['sampling_id']!='greedy' or w['seed'] is not None:raise ValueError('sampling contract changed')
    if len(binding['input_token_ids'])!=128 or len(binding['generated_token_ids'])!=32:raise ValueError('reference shape changed')
    expected_prompt=token_hash(binding['input_token_ids']);expected_output=token_hash(binding['generated_token_ids'])
    requests=run['requests']
    if len(requests)!=w['measured_iterations']:raise ValueError('missing or extra requests')
    if [row['input_index'] for row in requests]!=list(range(len(requests))):raise ValueError('request indices changed')
    for row in requests:
        if (row['prompt_token_count'],row['requested_output_token_count'],row['generated_token_count'])!=(128,32,32):raise ValueError('token counts changed')
        if row['prompt_u32le_sha256']!=expected_prompt or row['generated_u32le_sha256']!=expected_output:raise ValueError('cross-engine token mismatch')
        if row['e2e_ms']<row['ttft_ms']:raise ValueError('request time ordering invalid')
    trace=run['trace']
    if trace['dropped_records'] or trace['retained_records']>trace['capacity']:raise ValueError('incomplete trace')
    if run['aggregate']['cuda']['stream_span_ns']!={'validity':'unmeasured','value':None}:raise ValueError('full graph CUDA timing is not qualified')
    return {'schema_version':'riley.vllm-profile-validation.v1','valid':True,'requests':len(requests),'tokens_exact':True,'e0_claim':False,'speedup_claim':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--binding',required=True,type=Path);args=p.parse_args()
    print(json.dumps(validate(json.loads(args.run.read_text()),json.loads(args.binding.read_text())),indent=2))
if __name__=='__main__':main()
