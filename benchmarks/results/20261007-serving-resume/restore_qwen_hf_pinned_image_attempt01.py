import json,subprocess,time
from pathlib import Path
R=Path('/data/riley-serving-261007');O=R/'qwen-hf-pinned-image-recovery-attempt01';O.mkdir()
image='vllm/vllm-openai@sha256:7ef5a35d1ef8ce2cf9d671dd91eec6e367c5849262e0362b4d3d4a26be0d87d2'
failure=None
try:
 before=subprocess.run(['docker','image','inspect',image,'--format','{{.Id}}'],capture_output=True,text=True)
 (O/'preparation.json').write_text(json.dumps({'image':image,'prior_inspect_exit':before.returncode,'prior_image_id':before.stdout.strip(),'scope':'restore exact HF stage producer image only; original serving baseline version unchanged; no GPU execution'},indent=2)+'\n')
 with (O/'pull.log').open('x') as f:p=subprocess.run(['docker','pull',image],stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,'pinned Docker image pull failed; no substitute accepted'
 inspect=subprocess.check_output(['docker','image','inspect',image],text=True)
 (O/'image-inspect.json').write_text(inspect)
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'completion.json').write_text(json.dumps({'failure':failure,'image':image,'time_ns':time.time_ns(),'GPU_execution':'미실행','goal_achieved':False},indent=2)+'\n')
