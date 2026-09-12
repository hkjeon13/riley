from pathlib import Path
import sys,json
r=Path('/tmp/riley-opt-260912');sys.path.insert(0,str(r));import remote_session_round45 as s
v=s.read_private(s.ROOT/'verified.json');rows=[{'pid':x['new_pid'],'start':x['start'],'port':x['port']} for x in v['processes']]
assert all(not s.original_running(x) and not s.listening(x['port']) for x in rows)
watchdogs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:a=(p/'cmdline').read_bytes().split(b'\0')
 except OSError:continue
 if len(a)>2 and a[1]==str(r/'remote_session_round45.py').encode() and a[2]==b'watchdog':watchdogs.append(int(p.name))
assert not watchdogs
x={'user_requested_keep_stopped':True,'processes':rows,'running':False,'ports_closed':True,'restoration_scheduled':False,'watchdogs':watchdogs,'note':'Round45 restored before instruction took effect. Exact restored successors then received SIGTERM. Initial immediate port check raced socket teardown; fresh process/port verification passed.'}
(r/'blender-keep-stopped-public.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
