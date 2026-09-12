import ast,hashlib,json,pathlib
r=pathlib.Path('/tmp/riley-opt-260912'); old=r/'remote_session_round42.py';out=r/'remote_session_round43.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old)=='2ef0f891b0fa155836dd79eb1957905e53521f9eafa8a913f2409a9d773d047d'
s=old.read_text();lines=s.splitlines(keepends=True);edits=[]
values={'ROOT':"ROOT = CAMPAIGN / 'blender-round43'\n",'PREVIOUS':"PREVIOUS = CAMPAIGN / 'blender-round42'\n",'PREVIOUS_PINS':'PREVIOUS_PINS = '+repr({n:sha(r/'blender-round42'/n) for n in ('session.json','runtime.json','verified.json')})+'\n','PREDECESSOR_HELPER_SHA':'PREDECESSOR_HELPER_SHA = '+repr(sha(old))+'\n'}
for node in ast.parse(s).body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in values:edits.append((node.lineno-1,node.end_lineno,values[node.targets[0].id]))
 if isinstance(node,ast.FunctionDef) and node.name in ('qualified_successors','predecessor_module'):
  t=''.join(lines[node.lineno-1:node.end_lineno]);t=t.replace('round42','round43').replace('round41','round42').replace('round43','round43');edits.append((node.lineno-1,node.end_lineno,t))
assert len(edits)==6
for a,b,t in sorted(edits,reverse=True):lines[a:b]=[t]
s=''.join(lines).replace("'riley.round42-private-runtime.v1'","'riley.round43-private-runtime.v1'",1)
s=s.replace('three verified round41 successors','three verified round42 successors').replace('State is private to round42','State is private to round43')
with out.open('x') as f:f.write(s)
print(json.dumps({'helper_sha256':sha(out),'predecessor_sha256':sha(old)}))
