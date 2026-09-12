import ast,hashlib,json,pathlib
r=pathlib.Path('/tmp/riley-opt-260912'); old=r/'remote_session_round32.py';out=r/'remote_session_round33.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old)=='628ef742e555a231e1e04dd25a1617185ec941d8a35936a93a92d16fd1da6831'
s=old.read_text();lines=s.splitlines(keepends=True);edits=[]
values={'ROOT':"ROOT = CAMPAIGN / 'blender-round33'\n",'PREVIOUS':"PREVIOUS = CAMPAIGN / 'blender-round32'\n",'PREVIOUS_PINS':'PREVIOUS_PINS = '+repr({n:sha(r/'blender-round32'/n) for n in ('session.json','runtime.json','verified.json')})+'\n','PREDECESSOR_HELPER_SHA':'PREDECESSOR_HELPER_SHA = '+repr(sha(old))+'\n'}
for node in ast.parse(s).body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in values:edits.append((node.lineno-1,node.end_lineno,values[node.targets[0].id]))
 if isinstance(node,ast.FunctionDef) and node.name in ('qualified_successors','predecessor_module'):
  t=''.join(lines[node.lineno-1:node.end_lineno]);t=t.replace('round32','round33').replace('round31','round32').replace('round33','round33');edits.append((node.lineno-1,node.end_lineno,t))
assert len(edits)==6
for a,b,t in sorted(edits,reverse=True):lines[a:b]=[t]
s=''.join(lines).replace("'riley.round32-private-runtime.v1'","'riley.round33-private-runtime.v1'",1)
s=s.replace('three verified round31 successors','three verified round32 successors').replace('State is private to round32','State is private to round33')
with out.open('x') as f:f.write(s)
print(json.dumps({'helper_sha256':sha(out),'predecessor_sha256':sha(old)}))
