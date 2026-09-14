"""Predeclared host-wide start condition, not proof of an isolated benchmark."""
import json,pathlib,time
POLICY={'version':'psi-start-v1','interval_seconds':2,'consecutive_samples':3,'max_cpu_some_pct':5,'max_io_full_pct':5,'max_memory_full_pct':0.5}

def snapshot():
    return {'monotonic_ns':time.monotonic_ns(),'totals':{resource:{line.split()[0]:int(dict(x.split('=') for x in line.split()[1:])['total']) for line in pathlib.Path('/proc/pressure',resource).read_text().splitlines()} for resource in ('cpu','io','memory')}}

def rates(before,after):
    us=(after['monotonic_ns']-before['monotonic_ns'])/1000
    assert us>0
    result={}
    for key,resource,kind in [('cpu_some_pct','cpu','some'),('io_full_pct','io','full'),('memory_full_pct','memory','full')]:
        change=after['totals'][resource][kind]-before['totals'][resource][kind];assert change>=0
        result[key]=100*change/us
    return result

def acceptable(measured):
    return all(measured[k]<=POLICY['max_'+k] for k in ('cpu_some_pct','io_full_pct','memory_full_pct'))

def wait_quiet(path,timeout):
    assert 10<=timeout<=600
    path=pathlib.Path(path);assert not path.exists()
    report={'policy':POLICY,'timeout_seconds':timeout,'samples':[],'passed':False}
    deadline=time.monotonic()+timeout;before=snapshot();streak=0
    try:
        while time.monotonic()+POLICY['interval_seconds']<=deadline:
            time.sleep(POLICY['interval_seconds']);after=snapshot();measured=rates(before,after)
            streak=streak+1 if acceptable(measured) else 0
            report['samples'].append({'before_ns':before['monotonic_ns'],'after_ns':after['monotonic_ns'],**measured,'quiet_streak':streak});before=after
            if streak==POLICY['consecutive_samples']:report['passed']=True;return report
        raise RuntimeError('host quiet start condition timed out; preserve the whole attempt, do not retry a slow measured lane selectively')
    finally:
        path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--timeout',type=int,default=60);a=p.parse_args()
    print(json.dumps(wait_quiet(a.out,a.timeout)))
