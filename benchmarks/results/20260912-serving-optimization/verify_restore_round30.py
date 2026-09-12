import sys,json
from pathlib import Path
r=Path('/tmp/riley-opt-260912');sys.path.insert(0,str(r));import remote_session_round30 as s
rows=s.read_private(s.SNAPSHOT);v=s.read_private(s.ROOT/'verified.json');runtime=s.bound_runtime();public=[]
for record in v['processes']:
 row=next(x for x in rows if x['pid']==record['original_pid'])
 now=s.tagged_process(row,record['tag'],runtime)
 assert now['pid']==record['new_pid'] and now['start']==record['start'] and s.listening(record['port'])
 assert s.verify_private_maps(now['pid'],runtime)['ready']
 public.append({'pid':now['pid'],'start':now['start'],'port':record['port'],'live_verified':True})
(r/'round30-restoration-public.json').write_text(json.dumps(public,indent=2)+'\n');print(json.dumps(public))
