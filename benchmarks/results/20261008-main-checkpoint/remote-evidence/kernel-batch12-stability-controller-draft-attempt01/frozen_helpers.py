import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from serving_token_client_v2 import TokenHttpClient, TokenReference
from mixed_phase_v34 import mixed_phase
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def smi(query, kind='gpu'):
    return subprocess.check_output(['nvidia-smi', '--query-'+kind+'='+query,
                                   '--format=csv,noheader,nounits'], text=True).strip()
def host():
    before=time.perf_counter_ns()
    epoch=time.time_ns();monotonic=time.monotonic_ns();raw=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    after=time.perf_counter_ns()
    clocks={'perf_counter_before_ns':before,'perf_counter_after_ns':after,'epoch_ns':epoch,'monotonic_ns':monotonic,'monotonic_raw_ns':raw,'perf_counter_implementation':time.get_clock_info('perf_counter').implementation}
    return {'time_ns': epoch, 'clocks':clocks, 'pressure': {k: Path('/proc/pressure',k).read_text()
            for k in ['cpu','io','memory']}, 'loadavg': Path('/proc/loadavg').read_text(),
            'gpu': smi('uuid,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu')}
def process_group_memory(group):
    rows=subprocess.check_output(['ps','-e','-o','pid=,pgid=,rss='],text=True)
    observed={}
    for row in rows.splitlines():
        pid,pgid,rss=map(int,row.split())
        if pgid==group:observed[str(pid)]=rss
    return {'rss_KiB_by_pid':observed,'rss_sum_KiB':sum(observed.values()),
            'scope':'instantaneous process-group RSS sum; shared pages may count more than once; sampled each second'}
def quiet_host_gate(prefix):
    policy = plan['host_start_gate']; records=[]; streak=0
    deadline = time.monotonic()+policy['timeout_seconds']
    while time.monotonic()<deadline:
        h=host(); records.append(h)
        def avg(resource, kind):
            row=next(x for x in h['pressure'][resource].splitlines() if x.startswith(kind+' '))
            return float(row.split()[1].split('=')[1])
        quiet = (avg('cpu','some')<=policy['cpu_some_max'] and
                 avg('io','full')<=policy['io_full_max'] and
                 avg('memory','full')<=policy['memory_full_max'])
        streak=streak+1 if quiet else 0
        if streak>=policy['consecutive']:
            write(prefix+'-host-gate.json', {'passed':True,'policy':policy,'samples':records});return
        time.sleep(policy['interval_seconds'])
    write(prefix+'-host-gate.json', {'passed':False,'policy':policy,'samples':records})
    raise RuntimeError('whole attempt aborted: host start gate timeout '+prefix)
def ready(p, port):
    deadline=time.monotonic()+plan['startup_http_ready_timeout_seconds']
    while p.poll() is None and time.monotonic()<deadline:
        c=http.client.HTTPConnection('127.0.0.1',port,timeout=1)
        try:
            c.request('GET','/v1/models'); response=c.getresponse();body=response.read()
            if response.status==200 and any(x.get('id')=='g04-smol' for x in json.loads(body).get('data',[])):return
        except (OSError,http.client.HTTPException):pass
        finally:c.close()
        time.sleep(.1)
    raise RuntimeError(f'server early exit {p.returncode}' if p.poll() is not None else f"HTTP readiness timeout after {plan['startup_http_ready_timeout_seconds']} seconds")
def stop(p):
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=40)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
root=Path("/data/riley-serving-261007")
code=root/"kernel-batch12-controller-attempt01"
plan=json.loads((code/"batch-plan.json").read_text())
out=None
def write(name,value):
 (out/name).write_text(json.dumps(value,indent=2)+"\n")
