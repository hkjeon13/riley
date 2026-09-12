import ast,hashlib,json,pathlib
r=pathlib.Path('/tmp/riley-opt-260912'); old=r/'remote_session_round43.py';out=r/'remote_session_round44.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old)=='56b66e191fed43f1d0dc7396897ab121229fab080aec9218eec44ba79b512078'
s=old.read_text();lines=s.splitlines(keepends=True);edits=[]
values={'ROOT':"ROOT = CAMPAIGN / 'blender-round44'\n",'PREVIOUS':"PREVIOUS = CAMPAIGN / 'blender-round43'\n",'PREVIOUS_PINS':'PREVIOUS_PINS = '+repr({n:sha(r/'blender-round43'/n) for n in ('session.json','runtime.json','verified.json')})+'\n','PREDECESSOR_HELPER_SHA':'PREDECESSOR_HELPER_SHA = '+repr(sha(old))+'\n'}
for node in ast.parse(s).body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in values:edits.append((node.lineno-1,node.end_lineno,values[node.targets[0].id]))
 if isinstance(node,ast.FunctionDef) and node.name in ('qualified_successors','predecessor_module'):
  t=''.join(lines[node.lineno-1:node.end_lineno]);t=t.replace('round43','round44').replace('round42','round43').replace('round44','round44');edits.append((node.lineno-1,node.end_lineno,t))
assert len(edits)==6
for a,b,t in sorted(edits,reverse=True):lines[a:b]=[t]
s=''.join(lines).replace("'riley.round43-private-runtime.v1'","'riley.round44-private-runtime.v1'",1)
s=s.replace('three verified round42 successors','three verified round43 successors').replace('State is private to round43','State is private to round44')
with out.open('x') as f:f.write(s)
print(json.dumps({'helper_sha256':sha(out),'predecessor_sha256':sha(old)}))
