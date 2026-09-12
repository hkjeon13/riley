from pathlib import Path
import sys,os,signal,time,json,fcntl
r=Path('/tmp/riley-opt-260912');sys.path.insert(0,str(r));import remote_session_round45 as s
with (s.ROOT/'restore.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 rows=s.read_private(s.SNAPSHOT);v=s.read_private(s.ROOT/'verified.json');runtime=s.bound_runtime();current=[];fds=[]
 try:
  for rec in v['processes']:
   row=next(x for x in rows if x['pid']==rec['original_pid']);now=s.tagged_process(row,rec['tag'],runtime)
   assert now['pid']==rec['new_pid'] and now['start']==rec['start'] and s.verify_private_maps(now['pid'],runtime)['ready']
   fd=os.pidfd_open(now['pid']);assert s.process_start(now['pid'])==now['start'];fds.append(fd);current.append({'pid':now['pid'],'start':now['start'],'port':rec['port']})
  for fd in fds:signal.pidfd_send_signal(fd,signal.SIGTERM)
  deadline=time.monotonic()+30
  while any(s.original_running(row) for row in current):
   assert time.monotonic()<deadline,'Blender did not terminate';time.sleep(.2)
  assert all(not s.listening(row['port']) for row in current)
 finally:
  for fd in fds:os.close(fd)
 receipt={'user_requested_keep_stopped':True,'processes':current,'running':False,'ports_closed':True,'restoration_scheduled':False,'note':'Round45 restoration completed before instruction was applied; exact restored successors were then terminated. Existing watchdog sees completed verified journal and cannot relaunch.'}
 (r/'blender-keep-stopped-public.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
