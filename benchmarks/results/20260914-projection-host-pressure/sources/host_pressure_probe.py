"""Read-only Linux host sampling; never stops processes or qualifies serving results."""
import argparse,json,pathlib,time


def snapshot():
    root=pathlib.Path('/proc'); result={'monotonic_ns':time.monotonic_ns(),'pressure':{},'processes':{}}
    for resource in ('cpu','io','memory'):
        result['pressure'][resource]={line.split()[0]:int(dict(x.split('=') for x in line.split()[1:])['total']) for line in (root/'pressure'/resource).read_text().splitlines()}
    result['meminfo']={line.split(':')[0]:int(line.split()[1]) for line in (root/'meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapFree:','SwapTotal:','Dirty:','Writeback:'))}
    result['vmstat']={k:int(v) for k,v in (line.split() for line in (root/'vmstat').read_text().splitlines()) if k in ('pswpin','pswpout','pgmajfault')}
    for p in root.iterdir():
        if not p.name.isdigit():continue
        try:
            stat=(p/'stat').read_text(); tail=stat.rsplit(') ',1)[1].split()
            io={k:int(v) for k,v in (line.split(':') for line in (p/'io').read_text().splitlines())}
            result['processes'][p.name]={'comm':stat.split('(',1)[1].rsplit(')',1)[0],'start_ticks':int(tail[19]),'cpu_ticks':int(tail[11])+int(tail[12]),'read_bytes':io['read_bytes'],'write_bytes':io['write_bytes']}
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return result


def delta(a,b):
    elapsed_us=(b['monotonic_ns']-a['monotonic_ns'])/1000
    assert elapsed_us>0
    processes=[]
    for pid,q in b['processes'].items():
        p=a['processes'].get(pid)
        if p is None or p['start_ticks']!=q['start_ticks']:continue
        changes={k:q[k]-p[k] for k in ('cpu_ticks','read_bytes','write_bytes')}
        if any(changes.values()):processes.append({'pid':int(pid),'comm':q['comm'],**changes})
    return {'elapsed_us':elapsed_us,'pressure_pct':{resource:{kind:100*(value-a['pressure'][resource][kind])/elapsed_us for kind,value in levels.items()} for resource,levels in b['pressure'].items()},'vmstat_delta':{k:v-a['vmstat'][k] for k,v in b['vmstat'].items()},'meminfo_after_kib':b['meminfo'],'top_io':sorted(processes,key=lambda p:p['read_bytes']+p['write_bytes'],reverse=True)[:8],'top_cpu':sorted(processes,key=lambda p:p['cpu_ticks'],reverse=True)[:8]}


def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=pathlib.Path);p.add_argument('--samples',type=int,default=6);p.add_argument('--interval',type=float,default=5);args=p.parse_args()
    assert 2<=args.samples<=60 and 1<=args.interval<=10
    assert not args.out.exists()
    before=snapshot();report={'scope':'host-wide PSI and readable surviving process counters; not serving causality or per-cgroup attribution','samples':[]}
    for _ in range(args.samples):
        time.sleep(args.interval);after=snapshot();report['samples'].append(delta(before,after));before=after
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'samples':len(report['samples']),'io_full_pct':[round(r['pressure_pct']['io']['full'],2) for r in report['samples']],'cpu_some_pct':[round(r['pressure_pct']['cpu']['some'],2) for r in report['samples']]}))

if __name__=='__main__':main()
