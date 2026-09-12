import ast,hashlib,json,pathlib
r=pathlib.Path('/tmp/riley-opt-260912'); old=r/'remote_session_round36.py';out=r/'remote_session_round37.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old)=='1c795763acba75fc7de761591d12db5c138e295177736c39017ad6ca3c210557'
s=old.read_text();lines=s.splitlines(keepends=True);edits=[]
values={'ROOT':"ROOT = CAMPAIGN / 'blender-round37'\n",'PREVIOUS':"PREVIOUS = CAMPAIGN / 'blender-round36'\n",'PREVIOUS_PINS':'PREVIOUS_PINS = '+repr({n:sha(r/'blender-round36'/n) for n in ('session.json','runtime.json','verified.json')})+'\n','PREDECESSOR_HELPER_SHA':'PREDECESSOR_HELPER_SHA = '+repr(sha(old))+'\n'}
for node in ast.parse(s).body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in values:edits.append((node.lineno-1,node.end_lineno,values[node.targets[0].id]))
 if isinstance(node,ast.FunctionDef) and node.name in ('qualified_successors','predecessor_module'):
  t=''.join(lines[node.lineno-1:node.end_lineno]);t=t.replace('round36','round37').replace('round35','round36').replace('round37','round37');edits.append((node.lineno-1,node.end_lineno,t))
assert len(edits)==6
for a,b,t in sorted(edits,reverse=True):lines[a:b]=[t]
s=''.join(lines).replace("'riley.round36-private-runtime.v1'","'riley.round37-private-runtime.v1'",1)
s=s.replace('three verified round35 successors','three verified round36 successors').replace('State is private to round36','State is private to round37')
with out.open('x') as f:f.write(s)
print(json.dumps({'helper_sha256':sha(out),'predecessor_sha256':sha(old)}))
