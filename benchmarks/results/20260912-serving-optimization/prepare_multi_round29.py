import ast,hashlib,json,pathlib
r=pathlib.Path('/tmp/riley-opt-260912'); old=r/'remote_session_round28.py';out=r/'remote_session_round29.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old)=='b1483a14987826350b9f6e40f9b49dc23908d66df228eeb2b1e0568a25c0e506'
s=old.read_text();lines=s.splitlines(keepends=True);edits=[]
values={'ROOT':"ROOT = CAMPAIGN / 'blender-round29'\n",'PREVIOUS':"PREVIOUS = CAMPAIGN / 'blender-round28'\n",'PREVIOUS_PINS':'PREVIOUS_PINS = '+repr({n:sha(r/'blender-round28'/n) for n in ('session.json','runtime.json','verified.json')})+'\n','PREDECESSOR_HELPER_SHA':'PREDECESSOR_HELPER_SHA = '+repr(sha(old))+'\n'}
for node in ast.parse(s).body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in values:edits.append((node.lineno-1,node.end_lineno,values[node.targets[0].id]))
 if isinstance(node,ast.FunctionDef) and node.name in ('qualified_successors','predecessor_module'):
  t=''.join(lines[node.lineno-1:node.end_lineno]);t=t.replace('round28','round29').replace('round27','round28').replace('round29','round29');edits.append((node.lineno-1,node.end_lineno,t))
assert len(edits)==6
for a,b,t in sorted(edits,reverse=True):lines[a:b]=[t]
s=''.join(lines).replace("'riley.round28-private-runtime.v1'","'riley.round29-private-runtime.v1'",1)
s=s.replace('three verified round27 successors','three verified round28 successors').replace('State is private to round28','State is private to round29')
with out.open('x') as f:f.write(s)
print(json.dumps({'helper_sha256':sha(out),'predecessor_sha256':sha(old)}))
