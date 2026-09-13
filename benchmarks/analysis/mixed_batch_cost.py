"""Bounded successful-step host execution costs; never pure GPU time or a fitted policy."""
import argparse,json,pathlib

def parse(path):
    rows=[];overflow=None;expected=None;seen=set()
    for line in path.read_text().splitlines():
        if line.startswith('RILEY_HOST_PHASE '):
            fields=dict(x.split('=',1)for x in line.split()[1:])
            if fields['kind'] in ('decode','prefill_or_mixed'):expected=(expected or 0)+int(fields['steps'])
        if line.startswith('RILEY_MIXED_COST_OVERFLOW '):
            assert overflow is None
            overflow=int(line.split('count=')[1])
        elif line.startswith('RILEY_MIXED_COST '):
            row={k:int(v)for k,v in (x.split('=',1)for x in line.split()[1:])}
            key=tuple(row[k]for k in ['rows_upper','decode_rows','prefill_requests','context_upper'])
            assert key not in seen;seen.add(key)
            assert row['count']>0 and 0<=row['min_ns']<=row['max_ns']
            assert row['min_ns']*row['count']<=row['execute_wall_ns']<=row['max_ns']*row['count']
            row['mean_execute_wall_ms']=row['execute_wall_ns']/row['count']/1e6;rows.append(row)
    assert overflow is not None and overflow>=0 and expected is not None
    assert sum(r['count']for r in rows)+overflow==expected
    return {'scope':'successful ordinary iterations including prefill/mixed; execute host wall includes prepare, transfers, GPU wait, download/validation; paired decode excluded; no percentile or GPU-time claim','overflow_steps':overflow,'ordinary_steps':expected,'rows':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('log',type=pathlib.Path);p.add_argument('out',type=pathlib.Path);a=p.parse_args()
    a.out.write_text(json.dumps(parse(a.log),indent=2)+'\n')
