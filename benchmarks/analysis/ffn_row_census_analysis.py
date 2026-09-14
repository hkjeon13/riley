"""Validate bounded per-prefill row counts against ordinary batch accounting."""
import argparse,json,pathlib
from mixed_batch_cost import parse as parse_cost

def parse(path):
    cost=parse_cost(path);counts={};overflow=None;packed={};packed_overflow=None
    for line in path.read_text().splitlines():
        if line.startswith('RILEY_PACKED_FFN_ROWS '):
            fields=dict(x.split('=') for x in line.split()[1:]);row=int(fields['rows']);count=int(fields['count'])
            assert 1<=row<=1024 and count>0 and row not in packed
            packed[row]=count
        elif line.startswith('RILEY_PACKED_FFN_ROWS_OVERFLOW '):
            assert packed_overflow is None;packed_overflow=int(line.split('count=')[1])
        elif line.startswith('RILEY_PREFILL_ROWS '):
            fields=dict(x.split('=') for x in line.split()[1:]);row=int(fields['rows']);count=int(fields['count'])
            assert 1<=row<=1024 and count>0 and row not in counts
            counts[row]=count
        elif line.startswith('RILEY_PREFILL_ROWS_OVERFLOW '):
            assert overflow is None;overflow=int(line.split('count=')[1])
    assert overflow==0 and cost['overflow_steps']==0
    expected=sum(r['prefill_requests']*r['count'] for r in cost['rows'])
    assert sum(counts.values())==expected and expected>0
    assert packed_overflow==0
    expected_packed={};actual_packed={}
    for r in cost['rows']:
        if r['prefill_requests']>0:expected_packed[r['rows_upper']]=expected_packed.get(r['rows_upper'],0)+r['count']
    for row,n in packed.items():
        upper=((row+15)//16)*16;actual_packed[upper]=actual_packed.get(upper,0)+n
    assert expected_packed==actual_packed and sum(packed.values())>0
    packed_steps=sum(packed.values());packed_large=sum(n for row,n in packed.items() if row>=192)
    large=sum(n for row,n in counts.items() if row>=192)
    tokens=sum(row*n for row,n in counts.items())
    return {'packed_ffn_steps':packed_steps,'packed_ffn_rows':packed,'m32_eligible_steps':packed_large,'m32_eligible_step_percent':100*packed_large/packed_steps,'packed_scope':'total packed rows including decode rows in successful prefill/mixed iterations; warmup included; adaptive selector uses these rows','scope':'successful ordinary per-request prefill input rows, including warmup; not kernel timing or offered-load statistics','prefill_items':expected,'input_rows':tokens,'at_least_192_items':large,'at_least_192_item_percent':large*100/expected,'at_least_192_input_row_percent':100*sum(row*n for row,n in counts.items() if row>=192)/tokens,'row_counts':counts,'ordinary_batch_cost':cost}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('log',type=pathlib.Path);ap.add_argument('out',type=pathlib.Path);args=ap.parse_args()
    args.out.write_text(json.dumps(parse(args.log),indent=2)+'\n')
